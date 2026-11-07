"""Profile frozen serving or disposable training windows on the actual server."""
import argparse
import copy
from contextlib import nullcontext
import json
import random
import resource
import sys
import time
from pathlib import Path

from benchmarks.common import digest, distribution, output_dir, provenance, write
from benchmarks.instrument import Instrument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serving', 'training'])
    parser.add_argument('--config', default='configs/hybrid-serving-server.json')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--trace-steps', type=int, default=10)
    parser.add_argument('--output', default='runs/performance')
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.trace_steps) < 1:
        parser.error('steps, warmup and trace-steps must be positive')
    out = output_dir(args.output)
    report = {'status': 'RUNNING', 'arguments': vars(args), 'provenance': provenance(),
              'scope': 'Plain timings are benchmarks; synchronized stage timings and profiler traces are diagnostic only.'}
    try:
        import numpy as np
        import torch
        from serving.runtime import load_runtime, ROOT
        from src import microlens_training as training
        from src.data.dataset import SASRecTrainDataset, SASRecEvalDataset
        from diagnostics.tail_admission import verify_unchanged
        engine, records, meta = load_runtime(args.config)
        report['runtime_manifest'] = meta
        device = engine.scorer.device
        config = meta['training_config']
        seed = config['seed']
        ordered = sorted(records, key=lambda r: (len(r['train']), r['user_id']))
        requests = [{'history': ordered[int(i)]['train'], 'k': min(20, meta['config']['max_k'])}
                    for i in np.linspace(0, len(ordered) - 1, args.steps, dtype=int)]
        write(out / 'requests.json', requests)
        report['requests_sha256'] = digest(out / 'requests.json')
        reference_responses = {}
        report['memory_scope'] = 'Process allocator peaks; admitted model remains resident during disposable training. Excludes CPU CF and external GPU allocations.'
        report['workload'] = {'history_lengths': distribution([len(r['history']) for r in requests]),
                              'selection': 'evenly spaced training histories ordered by length',
                              'catalog_items': engine.num_items, 'seed': seed}
        report['passes'] = {}
        for mode in ['plain', 'stages', 'trace']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            instrument = Instrument(torch, device, mode)
            model = copy.deepcopy(engine.scorer.model) if args.action == 'training' else engine.scorer.model
            count = args.trace_steps if mode == 'trace' else args.steps
            if args.action == 'training':
                report['training_scope'] = 'Disposable copy of admitted best weights; fresh AdamW, bounded steps, no checkpoint writes or model selection. Not full epoch time.'
                dataset = SASRecTrainDataset(ROOT / config['data_dir'] / 'sequences.jsonl', model.sequence_length)
                loader = training.make_loader(dataset, list(range(len(dataset))), config['batch_size'],
                                              config['num_workers'], seed, device, shuffle=True)
                iterator = iter(loader)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                               weight_decay=config['weight_decay'], betas=(.9, .98))
                def step(i):
                    nonlocal iterator
                    with instrument.region('data_wait'):
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            iterator = iter(loader)
                            batch = next(iterator)
                    # Count on CPU without adding a new device synchronization.
                    positions = int(batch['padding_mask'].sum())
                    batch = training.move_batch(batch, device)
                    training.optimization_step(model, optimizer, batch, config)
                    return positions
            else:
                def step(i):
                    response = engine.recommend(requests[i % len(requests)])
                    key = i % len(requests)
                    if mode == 'plain':
                        reference_responses[key] = response
                    elif key in reference_responses and response != reference_responses[key]:
                        raise ValueError('Instrumentation changed serving response')
                    return 1
            # Warmup is uninstrumented and identical across diagnostic passes.
            instrument.mode = 'plain'
            for i in range(args.warmup):
                step(i)
            instrument.mode = mode
            instrument.sync()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            if mode != 'plain':
                instrument.wrap(model, 'forward', 'user_encoder')
                instrument.wrap(model, 'score_candidates', 'candidate_scoring')
                if args.action == 'training':
                    instrument.wrap(training, 'move_batch', 'host_to_device')
                    instrument.wrap(training, 'full_catalog_loss', 'loss_and_target_preparation')
                    instrument.wrap(torch.autograd, 'backward', 'backward')
                    instrument.wrap(optimizer, 'step', 'optimizer')
                    instrument.wrap(training, 'optimization_step', 'checks_and_step_overhead')
                else:
                    from serving import core
                    instrument.wrap(type(engine.scorer), '__call__', 'input_and_transfer')
                    instrument.wrap(engine.cf, 'score', 'itemcf')
                    instrument.wrap(core, 'rank_candidates', 'filter_and_topk')
                    instrument.wrap(core, 'validate_request', 'request_validation')
                    instrument.wrap(core, 'combine', 'fusion')
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = (torch.profiler.profile(activities=activities, record_shapes=True,
                        profile_memory=True) if mode == 'trace' else nullcontext())
            times, units = [], 0
            try:
                begin = time.perf_counter()
                with profiler as prof:
                    for i in range(count):
                        # Plain training measures a continuous window; serving is synchronized latency.
                        if args.action == 'serving':
                            instrument.sync()
                        start = time.perf_counter()
                        with instrument.region('request_or_step_overhead'):
                            units += step(args.warmup + i)
                        if args.action == 'serving':
                            instrument.sync()
                        times.append((time.perf_counter() - start) * 1000)
                instrument.sync()
                elapsed = time.perf_counter() - begin
                if mode == 'trace':
                    prof.export_chrome_trace(str(out / 'trace.json'))
                    (out / 'operators.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=60))
                report['passes'][mode] = {'seconds': elapsed, 'steps': count, 'units': units,
                    'unit': 'supervised_positions' if args.action == 'training' else 'requests',
                    'units_per_second': units / elapsed, 'wall_step_ms': distribution(times),
                    'samples_ms': times, 'exclusive_stage_ms': {k: distribution(v) for k, v in instrument.samples.items()},
                    'exclusive_stage_total_ms': {k: sum(v) for k, v in instrument.samples.items()}}
                if device.type == 'cuda':
                    report['passes'][mode]['memory_bytes'] = {'peak_allocated': torch.cuda.max_memory_allocated(device),
                                                            'peak_reserved': torch.cuda.max_memory_reserved(device)}
            finally:
                instrument.close()
            if args.action == 'training' and mode == 'plain':
                evaluation = SASRecEvalDataset(ROOT / config['data_dir'] / 'sequences.jsonl', model.sequence_length, 'val')
                indices = list(range(min(len(evaluation), args.steps * config['eval_batch_size'])))
                instrument.sync()
                start = time.perf_counter()
                training.evaluate_hstu(model, evaluation, indices, config, device, records)
                instrument.sync()
                report['bounded_validation'] = {'seconds': time.perf_counter() - start, 'users': len(indices),
                    'scope': 'Timing only on disposable model; metrics discarded; no test labels evaluated'}
            del model
        from benchmarks.common import stage_chart
        stage_chart(out / 'stages.svg', report['passes']['stages']['exclusive_stage_total_ms'])
        verify_unchanged(meta['inputs_sha256'])
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_lifetime_peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(out / 'results.json', report)
        print(f"Performance benchmark: {report['status']}\nSaved: {out}")


if __name__ == '__main__':
    main()
