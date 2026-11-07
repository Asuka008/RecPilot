"""Run or verify bounded serving against the unchanged admitted recommendation core."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import signal
import threading
import time

from benchmarks.common import digest, output_dir, provenance, write
from scheduled_serving.config import load_config
from scheduled_serving.http import BoundedHTTPServer
from scheduled_serving.scheduler import Metrics, Scheduler


def exchange(port, method, path, payload=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        body = None if payload is None else json.dumps(payload)
        connection.request(method, path, body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        return response.status, json.loads(response.read()), dict(response.getheaders())
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve', 'check'])
    parser.add_argument('--config', type=Path, default=Path('configs/scheduled-serving-server.json'))
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1000:
        parser.error('samples must be between 1 and 1000')
    config = load_config(args.config)
    out = output_dir(config['output_dir'])
    report = {'status': 'RUNNING', 'action': args.action, 'config': config, 'provenance': provenance(),
              'scope': 'Single-worker scheduling only; unchanged recommendation formula, no batching or new quality selection.'}
    server = scheduler = thread = None
    try:
        import torch
        from serving.runtime import ROOT, load_runtime
        from serving.core import validate_request
        from diagnostics.tail_admission import verify_unchanged
        engine, records, meta = load_runtime(ROOT / config['runtime_config'])
        report['runtime_manifest'] = meta
        sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'scheduled_serving').glob('*.py'))}
        identity = {'model_id': engine.identity, 'config': config, 'sources': sources}
        service_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        report['service_identity'] = {**identity, 'service_id': service_id}
        report['config_sha256'] = digest(args.config)
        print('Artifact admission: PASS', flush=True)
        print('Actual runtime:', json.dumps(meta['runtime']), flush=True)
        def initialize():
            if engine.scorer.device.type == 'cuda':
                torch.cuda.set_device(engine.scorer.device)
        def validate(request):
            validate_request(request, engine.num_items, engine.max_history, engine.max_k)
        requests, expected = [], []
        if args.action == 'check':
            ordered = sorted(records, key=lambda r: (len(r['train']), r['user_id']))
            for i in range(args.samples):
                index = i * (len(ordered) - 1) // max(1, args.samples - 1)
                request = {'history': ordered[index]['train'], 'k': min(20, engine.max_k)}
                requests.append(request)
            requests.append({'history': [], 'k': min(20, engine.max_k)})
            expected = [engine.recommend(request) for request in requests]
        scheduler = Scheduler(engine.recommend, config['queue_capacity'], Metrics(), initialize)
        # Warm the actual execution thread before declaring HTTP readiness.
        for i in range(meta['config']['warmup']):
            request = {'history': records[i % len(records)]['train'], 'k': min(20, engine.max_k)}
            scheduler.wait(scheduler.submit(request, time.monotonic() + 30))
        # Reset observations after warmup, without replacing the metrics object.
        with scheduler.metrics.lock:
            scheduler.metrics.counters.clear()
            scheduler.metrics.histograms.clear()
        manifest = {'service_id': service_id, 'runtime': meta['runtime'], 'scheduler_config': config}
        port = 0 if args.action == 'check' else config['port']
        server = BoundedHTTPServer((config['host'], port), scheduler, validate,
                                   engine.identity, engine.num_items, config, manifest)
        write(out / 'results.json', report)
        if args.action == 'check':
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
            thread.start()
            port = server.server_port
            status, health, _ = exchange(port, 'GET', '/health')
            if status != 200 or health['model_id'] != engine.identity:
                raise ValueError('Health identity mismatch')
            for request, reference in zip(requests, expected):
                status, actual, headers = exchange(port, 'POST', '/recommend', request)
                if status != 200 or actual != reference:
                    raise ValueError('Scheduled HTTP response differs from direct reference')
                if 'Server-Timing' not in headers or 'X-Request-ID' not in headers:
                    raise ValueError('Missing timing or request identity')
            status, _, _ = exchange(port, 'POST', '/recommend', {'history': [0], 'k': 1})
            if status != 400:
                raise ValueError('Invalid history was not rejected')
            status, snapshot, _ = exchange(port, 'GET', '/metrics')
            if status != 200 or snapshot['model_id'] != engine.identity:
                raise ValueError('Metrics identity mismatch')
            report['parity'] = {'requests': len(requests), 'scores_and_topk_exact': True,
                                'scope': 'Training histories plus empty-history fallback; no quality evaluation'}
            report['status'] = 'PASS'
        else:
            def terminate(signum, frame):
                raise KeyboardInterrupt
            previous = signal.signal(signal.SIGTERM, terminate)
            print(f'Ready: http://{config["host"]}:{port} (bounded single-worker; batch=1)', flush=True)
            print(f'Run report: {out / "results.json"}', flush=True)
            try:
                server.serve_forever(poll_interval=.1)
            except KeyboardInterrupt:
                report['status'] = 'STOPPED'
            finally:
                signal.signal(signal.SIGTERM, previous)
        verify_unchanged(meta['inputs_sha256'])
        if sources != {p: digest(ROOT / p) for p in sources} or digest(args.config) != report['config_sha256']:
            raise ValueError('Scheduling sources or configuration changed during execution')
    except KeyboardInterrupt:
        report['status'] = 'INTERRUPTED'
        raise
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if server:
            if thread:
                server.shutdown()
                thread.join()
            report['shutdown'] = server.stop()
            report['metrics'] = server.snapshot()
            if args.action == 'check' and not all(report['shutdown'].values()):
                report.update(status='FAIL', error='Check did not drain cleanly')
        elif scheduler:
            report['shutdown'] = {'worker_drained': scheduler.close(config['shutdown_grace_seconds'])}
        write(out / 'results.json', report)
        print(f'Scheduled serving: {report["status"]}\nSaved: {out}', flush=True)
    if report['status'] == 'FAIL':
        raise RuntimeError(report['error'])


if __name__ == '__main__':
    main()
