"""Reconstruct the scheduled-arrival timeline from saved open-200 request samples."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from benchmarks.common import digest, distribution, output_dir, write


QUEUE_SUMMARY = '2a24734d2e2beea49ca9c857cb2d7a4b2f11b008ef42069bd88c19b855470b7c'
TRANSPORT_SUMMARY = '93f5220783b2f9650f65b2b3452f40998c527eb53e474a5b6aa83cbe0d337b20'


def summarize(rows):
    success = [r for r in rows if r['error'] is None]
    slow = [r for r in success if r['http_ms'] > 20]
    return {
        'attempted': len(rows), 'successful': len(success),
        'errors': dict(Counter(r['error'] for r in rows if r['error'] is not None)),
        'successful_http_p99_ms': distribution([r['http_ms'] for r in success]).get('p99'),
        'dispatch_lag_p99_ms': distribution([r['dispatch_lag_ms'] for r in rows]).get('p99'),
        'successful_over_20ms': len(slow),
        'dispatch_lag_at_least_5ms': sum(r['dispatch_lag_ms'] >= 5 for r in rows),
        'slow_and_dispatched_late': sum(r['dispatch_lag_ms'] >= 5 for r in slow),
    }


def analyze_point(point):
    m = point['measurement']
    if (point['status'], point['setting'], m['mode'], m['offered_rate'], m['duration_seconds'],
            m['concurrency_limit'], m['offered']) != ('COMPLETE', 'open-200', 'open', 200, 60, 32, 12000):
        raise ValueError('Expected a completed 60-second open-200 point')
    rows = m['samples']
    seconds = [[] for _ in range(60)]
    seen = set()
    for r in rows:
        i = r['index']
        if type(i) is not int or not 0 <= i < 12000 or i in seen:
            raise ValueError('Duplicate or invalid scheduled arrival index')
        seen.add(i)
        times = [r[k] for k in ('dispatch_lag_ms', 'http_ms', 'scheduled_to_completion_ms')]
        if (any(not math.isfinite(x) or x < 0 for x in times)
                or not math.isclose(times[0] + times[1], times[2], abs_tol=1e-5)
                or (r['error'] is None and r['status'] != 200)):
            raise ValueError('Inconsistent raw sample')
        seconds[i // 200].append(r)
    totals = summarize(rows)
    if (totals['attempted'] != m['attempted'] or totals['successful'] != m['successful']
            or totals['errors'] != point['error_counts'] or len(rows) + m['client_dropped'] != 12000
            or totals['successful_http_p99_ms'] != m['successful_http_ms'].get('p99')):
        raise ValueError('Raw sample counts or successful p99 differ from the load report')
    bins = [dict(start_second=i, dropped=200 - len(group), **summarize(group))
            for i, group in enumerate(seconds)]
    counts = [b['successful_over_20ms'] for b in bins]
    best = max(range(56), key=lambda i: sum(counts[i:i + 5]))
    totals.update(client_dropped=m['client_dropped'], seconds_with_slow_success=sum(c > 0 for c in counts),
                  busiest_five_seconds_start=best if any(counts) else None,
                  slow_in_busiest_five_seconds=sum(counts[best:best + 5]), seconds=bins)
    return totals


def find_summary(root, family, expected):
    for path in sorted((root / family).glob('*/summary.json')):
        if digest(path) == expected:
            return path
    return None


def analyze_run(path):
    summary = json.loads(path.read_bytes())
    matched = summary['matched']
    results = []
    for trial in summary['trials']:
        if trial['transport'] != 'handler-pool':
            continue
        directory = path.parent / trial['directory']
        files = list(directory.glob('load/*/open-200.json'))
        if len(files) != 1:
            raise ValueError(f'Expected one raw open-200.json under {directory}')
        raw = files[0].read_bytes()
        point = json.loads(raw)
        health = point['health']
        capacity = trial.get('queue_capacity', matched['config'].get('queue_capacity'))
        if (point['requests_sha256'] != matched['requests_sha256']
                or health['service_id'] != trial['service_id'] or health['instance_id'] != trial['instance_id']
                or health['model_id'] != matched['model_id'] or health['scheduler_config']['queue_capacity'] != capacity):
            raise ValueError(f'Load point belongs to a different trial: {files[0]}')
        metrics = analyze_point(point)
        saved = next(p for p in trial['points'] if p['setting'] == 'open-200')
        if (any(metrics[key] != saved[key] for key in ('successful_http_p99_ms', 'client_dropped', 'attempted', 'successful'))
                or metrics['errors'] != saved['error_counts']):
            raise ValueError(f'Point differs from submitted summary: {files[0]}')
        results.append(dict(round=trial['round'], queue_capacity=capacity, instance_id=trial['instance_id'],
                            file=str(files[0]), sha256=hashlib.sha256(raw).hexdigest(), **metrics))
    return {'summary_path': str(path), 'summary_sha256': digest(path), 'trials': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    args = parser.parse_args()
    queue = find_summary(args.runs, 'queue-comparison', QUEUE_SUMMARY)
    if queue is None:
        parser.error(f'Cannot find the submitted queue comparison under {args.runs}/queue-comparison')
    report = {'queue_comparison': analyze_run(queue)}
    earlier = find_summary(args.runs, 'transport-comparison', TRANSPORT_SUMMARY)
    report['earlier_handler_pool'] = analyze_run(earlier) if earlier else None
    dest = output_dir(args.runs / 'tail-analysis') / 'tail-analysis.json'
    write(dest, report)
    print(f'Saved: {dest}')


if __name__ == '__main__':
    main()
