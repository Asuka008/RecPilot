"""Run the existing load generator with per-point scheduler metric snapshots."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

from benchmarks.common import digest, distribution, output_dir, provenance, write
from benchmarks.http_load import exchange, run_load, validate_response


def read_metrics(host, port, identity, service_id, instance_id):
    status, value = exchange(host, port, 'GET', '/metrics', None, 5)
    if status != 200 or value['model_id'] != identity or value['service_id'] != service_id or value['instance_id'] != instance_id:
        raise ValueError('Metrics identity mismatch')
    return value


def metric_delta(before, after):
    """Subtract cumulative counters/buckets, never subtract lifetime maxima."""
    if before['service_id'] != after['service_id'] or before.get('instance_id') != after.get('instance_id'):
        raise ValueError('Service identity changed')
    counters = {k: after['counters'].get(k, 0) - before['counters'].get(k, 0)
                for k in set(before['counters']) | set(after['counters'])}
    if any(v < 0 for v in counters.values()):
        raise ValueError('Counters reset during measurement')
    histograms = {}
    for key, end in after['histograms'].items():
        start = before['histograms'].get(key, {'count': 0, 'sum_ms': 0,
                    'bucket_counts': [0] * len(end['bucket_counts']), 'bucket_upper_ms': end['bucket_upper_ms']})
        if start['bucket_upper_ms'] != end['bucket_upper_ms']:
            raise ValueError('Histogram boundaries changed')
        count = end['count'] - start['count']
        buckets = [a - b for a, b in zip(end['bucket_counts'], start['bucket_counts'])]
        total = end['sum_ms'] - start['sum_ms']
        if count < 0 or total < -1e-6 or any(v < 0 for v in buckets) or sum(buckets) != count:
            raise ValueError('Histogram reset or inconsistency')
        histograms[key] = {'count': count, 'sum_ms': total, 'mean_ms': total / count if count else None,
                           'bucket_counts': buckets, 'bucket_upper_ms': end['bucket_upper_ms']}
    return {'counters': counters, 'histograms': histograms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--output', default='runs/scheduled-load')
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost'} or url.path not in {'', '/'} or url.query or url.fragment or url.username:
        parser.error('Only plain loopback HTTP origins are supported')
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('duration must be positive and finite')
    out = output_dir(args.output)
    base = {'arguments': {**vars(args), 'requests': str(args.requests)}, 'provenance': provenance(),
            'collector_sha256': digest(__file__), 'requests_sha256': digest(args.requests),
            'scope': 'Same benchmark client and connection policy as baseline; scheduler telemetry added. Single-worker, no batching.'}
    points = [('closed', c, 100.) for c in [1, 4, 16, 32]] + [('open', 32, r) for r in [50., 100., 200., 400.]]
    reports = []
    status = 'RUNNING'
    failure = None
    try:
        requests = json.loads(args.requests.read_text())
        if not isinstance(requests, list) or not requests:
            raise ValueError('Expected nonempty request array')
        host, port = url.hostname, url.port or 80
        status, health = exchange(host, port, 'GET', '/health', None, 5)
        if status != 200 or health['status'] != 'ready':
            raise ValueError('Scheduled service not ready')
        base['health'] = health
        identity, service_id = health['model_id'], health['service_id']
        instance_id = health['instance_id']
        for mode, concurrency, rate in points:
            label = f'{mode}-{concurrency if mode == "closed" else int(rate)}'
            report = {**base, 'status': 'RUNNING', 'setting': label}
            reports.append(report)
            try:
                # Warmup is outside both the load and the telemetry delta.
                for i in range(20):
                    item = requests[i % len(requests)]
                    status, value = exchange(host, port, 'POST', '/recommend', item, 5)
                    if status != 200:
                        raise ValueError('Warmup failed')
                    validate_response(value, item, identity, health['catalog_items'])
                before = read_metrics(host, port, identity, service_id, instance_id)
                if before['scheduler']['active_inference'] or before['scheduler']['queue_depth']:
                    raise ValueError('Service has unrelated queued or active inference')
                report['server_before'] = before
                measurement = run_load(host, port, requests, identity, health['catalog_items'],
                                       mode, args.duration, concurrency, rate, 5)
                report['measurement'] = measurement
                deadline = time.monotonic() + 5
                while True:
                    after = read_metrics(host, port, identity, service_id, instance_id)
                    report['server_after'] = after
                    if not after['scheduler']['active_inference'] and not after['scheduler']['queue_depth'] and after['connections'] <= 1:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Service failed to drain after load')
                    time.sleep(.05)
                report['server_before'], report['server_after'] = before, after
                report['server_delta'] = metric_delta(before, after)
                rows = measurement['samples']
                report['all_attempt_http_ms'] = distribution([r['http_ms'] for r in rows])
                report['error_counts'] = dict(Counter(r['error'] for r in rows if r['error']))
                report['status'] = 'COMPLETE'
                print(json.dumps({'setting': label, 'success_qps': measurement['success_qps_in_window'],
                     'errors': report['error_counts'], 'client_dropped': measurement['client_dropped'],
                     'successful_p99_ms': measurement['successful_http_ms'].get('p99'),
                     'all_attempt_p99_ms': report['all_attempt_http_ms'].get('p99')}), flush=True)
            except Exception as error:
                report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                write(out / (label + '.json'), report)
        status = 'COMPLETE'
    except BaseException as error:
        status = 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAIL'
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(out / 'summary.json', {**base, 'status': status, 'error': failure,
              'settings': [r['setting'] for r in reports]})
        print(f'Saved load reports: {out}', flush=True)


if __name__ == '__main__':
    main()
