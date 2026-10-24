"""Label-free serving parity and synchronized batch-one core latency checks."""
from datetime import datetime, timezone
import json
import resource
import sys
import time

import numpy as np

from serving.runtime import ROOT


def benchmark(engine, records, meta):
    """Compare serving with offline scoring on training histories, not test labels."""
    import torch
    from diagnostics.tail_admission import verify_unchanged
    from scripts.evaluate_hybrid_validation import combine
    from src.data.dataset import SASRecEvalDataset
    from src.evaluation import rank_candidates
    config = meta['config']
    device = engine.scorer.device
    output = ROOT / config['output_dir'] / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    report = dict(meta, status='RUNNING', batch_size=1, concurrency=1,
                  timing_scope='in-process request validation, inference, transfer, CF, ranking; excludes HTTP and startup',
                  warmup_per_mode=config['warmup'], requests_per_mode=config['requests'])
    path = output / 'results.json'
    try:
        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        # Deterministically span history lengths without reading held-out targets.
        ordered = sorted(range(len(records)), key=lambda i: (len(records[i]['train']), records[i]['user_id']))
        positions = np.linspace(0, len(ordered) - 1, config['requests'], dtype=int)
        indices = [ordered[int(i)] for i in positions]
        requests = [{'history': records[i]['train'], 'k': 20} for i in indices]
        if config['max_k'] < 20:
            raise ValueError('Benchmark requires max_k >= 20')
        dataset = SASRecEvalDataset(ROOT / meta['training_config']['data_dir'] / 'sequences.jsonl',
                                    engine.scorer.model.sequence_length, 'val')
        max_error = 0.0
        # Match the original offline batch size as well as its input adapter.
        with torch.inference_mode():
            for start in range(0, len(indices), meta['training_config']['eval_batch_size']):
                chunk = indices[start:start + meta['training_config']['eval_batch_size']]
                examples = [dataset[i] for i in chunk]
                ids = torch.stack([x['input_ids'] for x in examples]).to(device)
                mask = torch.stack([x['padding_mask'] for x in examples]).to(device)
                model = engine.scorer.model
                rows = model.score_candidates(model(ids, mask)[:, -1], engine.scorer.candidates).cpu().numpy()
                for index, row in zip(chunk, rows):
                    history = records[index]['train']
                    cf = engine.cf.score(history)
                    cf[0] = 0
                    cf[history] = 0
                    expected = combine(np.concatenate(([0.], row)), cf, .25)
                    actual, _ = engine.scores(history)
                    max_error = max(max_error, float(np.max(np.abs(expected - actual))))
                    if not np.allclose(expected, actual, rtol=0, atol=1e-6):
                        raise ValueError('Offline/serving score parity failed')
                    served = engine.recommend({'history': history, 'k': 20})
                    if [x['item_id'] for x in served['items']] != rank_candidates(expected, history, 20):
                        raise ValueError('Offline/serving Top-20 parity failed')
        report['parity'] = {'users': len(indices), 'max_absolute_score_error': max_error,
                            'score_atol': 1e-6, 'top20_exact': True, 'histories': 'training only; no quality metrics'}
        lengths = [len(r['history']) for r in requests]
        report['workload'] = {'min_history': min(lengths), 'max_history': max(lengths),
                              'median_history': float(np.median(lengths)), 'k': 20, 'selection': 'evenly spaced in history-length order'}
        measurements = {}
        for name, hybrid in [('raw_hstu', False), ('hybrid', True)]:
            for i in range(config['warmup']):
                engine.recommend(requests[i % len(requests)], hybrid=hybrid)
            sync()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for request in requests:
                sync()
                begin = time.perf_counter()
                response = engine.recommend(request, hybrid=hybrid)
                sync()
                timings.append((time.perf_counter() - begin) * 1000)
                if not response['items']:
                    raise ValueError('Unexpected empty ranking in benchmark workload')
            measurements[name] = {'p50_ms': float(np.percentile(timings, 50)),
                                  'p95_ms': float(np.percentile(timings, 95)),
                                  'p99_ms': float(np.percentile(timings, 99)),
                                  'mean_ms': float(np.mean(timings)), 'samples_ms': timings,
                                  'serial_requests_per_second': 1000 / float(np.mean(timings))}
            if device.type == 'cuda':
                measurements[name]['memory_bytes'] = {'peak_allocated': torch.cuda.max_memory_allocated(device),
                                                       'peak_reserved': torch.cuda.max_memory_reserved(device)}
        report['measurements'] = measurements
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_peak_rss_bytes'] = peak_rss if sys.platform == 'darwin' else peak_rss * 1024
        verify_unchanged(meta['inputs_sha256'])
        report['status'] = 'PASS'
        summary = {k: {a: b for a, b in v.items() if a != 'samples_ms'} for k, v in measurements.items()}
        print(json.dumps({'parity': report['parity'], 'measurements': summary}), flush=True)
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f"Saved: {path}\nServing benchmark: {report['status']}", flush=True)
