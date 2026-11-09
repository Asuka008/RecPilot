"""Bounded loopback HTTP load: closed concurrency or scheduled open arrivals."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from benchmarks.common import digest, distribution, output_dir, provenance, write


def validate_response(value, request, identity, catalog):
    """Reject successful HTTP responses with broken recommendation contracts."""
    if value['model_id'] != identity:
        raise ValueError('Model identity changed')
    ids = [x['item_id'] for x in value['items']]
    if len(ids) > request['k'] or len(set(ids)) != len(ids):
        raise ValueError('Invalid recommendation count or duplicates')
    if any(type(i) is not int or not 1 <= i <= catalog or i in request['history'] for i in ids):
        raise ValueError('Invalid or seen candidate')
    if len(ids) != min(request['k'], catalog - len(set(request['history']))):
        raise ValueError('Unexpected candidate count')
    scores = [x['score'] for x in value['items']]
    if any(not math.isfinite(s) for s in scores) or scores != sorted(scores, reverse=True):
        raise ValueError('Invalid score ordering')


def exchange(host, port, method, path, payload, timeout, include_retry_after=False):
    # Each request uses a new connection, matching the current HTTP/1.0 server.
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        body = None if payload is None else json.dumps(payload)
        connection.request(method, path, body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        result = response.status, json.loads(raw)
        if include_retry_after:
            return (*result, response.getheader('Retry-After'))
        return result
    finally:
        connection.close()


def run_load(host, port, requests, identity, catalog, mode, duration, concurrency, rate, timeout,
             pacing=None):
    if pacing is not None and (mode != 'closed' or pacing not in ('immediate', 'retry-after')):
        raise ValueError('503 pacing requires a closed-loop client and a known policy')
    samples, lock = [], threading.Lock()
    start = time.perf_counter()
    end = start + duration
    slots = threading.BoundedSemaphore(concurrency)
    def send(index, scheduled):
        actual = time.perf_counter()
        status, error = None, None
        delay_seconds = None
        try:
            request = requests[index % len(requests)]
            if pacing is None:
                status, value = exchange(host, port, 'POST', '/recommend', request, timeout)
            else:
                status, value, header = exchange(host, port, 'POST', '/recommend', request,
                                                 timeout, include_retry_after=True)
                if status == 503 and header and header.isascii() and header.isdecimal():
                    delay_seconds = int(header)
            if status != 200:
                error = f'http_{status}'
            else:
                validate_response(value, request, identity, catalog)
        except Exception as exc:
            error = type(exc).__name__
        done = time.perf_counter()
        sample = {'index': index, 'status': status, 'error': error,
                'dispatch_lag_ms': (actual - scheduled) * 1000,
                'http_ms': (done - actual) * 1000,
                'scheduled_to_completion_ms': (done - scheduled) * 1000,
                'completed_in_window': done <= end}
        if pacing is not None:
            sample['retry_after_seconds'] = delay_seconds
        with lock:
            samples.append(sample)
        return sample
    def closed(worker):
        index = worker
        while time.perf_counter() < end:
            sample = send(index, time.perf_counter())
            if pacing == 'retry-after' and sample['status'] == 503:
                remaining = max(0., end - time.perf_counter())
                pause = min(remaining, sample['retry_after_seconds'] or 0)
                if pause > 0:
                    time.sleep(pause)
                sample['pacing_sleep_requested_seconds'] = pause
            index += concurrency
    def bounded_send(index, scheduled):
        try:
            send(index, scheduled)
        finally:
            slots.release()
    dropped = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        if mode == 'closed':
            futures = [pool.submit(closed, i) for i in range(concurrency)]
        else:
            futures = []
            for index in range(math.ceil(duration * rate)):
                scheduled = start + index / rate
                delay = scheduled - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                if time.perf_counter() >= end or not slots.acquire(blocking=False):
                    dropped += 1
                    continue
                futures.append(pool.submit(bounded_send, index, scheduled))
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - start
    successful = [s for s in samples if s['error'] is None]
    return {'mode': mode, 'duration_seconds': duration, 'elapsed_including_drain_seconds': elapsed,
            'concurrency_limit': concurrency, 'offered_rate': rate if mode == 'open' else None,
            'attempted': len(samples), 'successful': len(successful), 'errors': len(samples) - len(successful),
            'client_dropped': dropped, 'offered': len(samples) + dropped,
            'success_fraction_of_offered': len(successful) / max(1, len(samples) + dropped),
            'success_qps_in_window': sum(s['completed_in_window'] for s in successful) / duration,
            'success_qps_including_drain': len(successful) / elapsed,
            'successful_http_ms': distribution([s['http_ms'] for s in successful]),
            'successful_scheduled_to_completion_ms': distribution([s['scheduled_to_completion_ms'] for s in successful]),
            'all_attempt_dispatch_lag_ms': distribution([s['dispatch_lag_ms'] for s in samples]),
            'samples': sorted(samples, key=lambda s: s['index'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--mode', choices=['closed', 'open'], default='closed')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--rate', type=float, default=100)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', default='runs/http-load')
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != 'http' or url.hostname not in {'localhost', '127.0.0.1'} or url.path not in {'', '/'} or url.query or url.fragment or url.username:
        parser.error('Only plain loopback HTTP origins are supported')
    if any(not math.isfinite(v) or v <= 0 for v in [args.duration, args.rate, args.timeout]) or args.concurrency < 1 or args.warmup < 0:
        parser.error('duration, rate, timeout and concurrency must be positive; warmup nonnegative')
    out = output_dir(args.output)
    report = {'status': 'RUNNING', 'arguments': {**vars(args), 'requests': str(args.requests)},
              'provenance': provenance(), 'requests_sha256': digest(args.requests),
              'scope': 'Client end-to-end HTTP, new TCP connection per request. Includes network/server queueing; cannot isolate server queue or GPU time. Client overload is counted, never silently queued.'}
    try:
        requests = json.loads(args.requests.read_text())
        if not isinstance(requests, list) or not requests:
            raise ValueError('Expected nonempty request array')
        host, port = url.hostname, url.port or 80
        status, health = exchange(host, port, 'GET', '/health', None, args.timeout)
        if status != 200 or health['status'] != 'ready':
            raise ValueError('Server not ready')
        report['health'] = health
        for request in requests:
            if set(request) != {'history', 'k'} or type(request['k']) is not int or not 1 <= request['k'] <= 100:
                raise ValueError('Invalid workload schema')
            if not isinstance(request['history'], list) or len(request['history']) > 4096 or any(type(i) is not int or not 1 <= i <= health['catalog_items'] for i in request['history']):
                raise ValueError('Invalid workload history')
        for i in range(args.warmup):
            request = requests[i % len(requests)]
            status, value = exchange(host, port, 'POST', '/recommend', request, args.timeout)
            if status != 200:
                raise ValueError('Warmup failed')
            validate_response(value, request, health['model_id'], health['catalog_items'])
        report['measurement'] = run_load(host, port, requests, health['model_id'], health['catalog_items'],
                                        args.mode, args.duration, args.concurrency, args.rate, args.timeout)
        # Errors under overload are measurements, not hidden by a PASS assertion.
        report['status'] = 'COMPLETE'
    except Exception as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(out / 'results.json', report)
        print(f"HTTP load: {report['status']}\nSaved: {out}")


if __name__ == '__main__':
    main()
