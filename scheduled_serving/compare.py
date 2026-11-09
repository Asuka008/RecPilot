"""Compare handler reuse and deadline monitoring on the frozen HTTP workload."""
import argparse
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmarks.common import digest, output_dir, provenance, write
from scheduled_serving.config import load_config
from scheduled_serving.http import TRANSPORTS
from scheduled_serving.load import SETTINGS, PACING_SETTINGS
from scheduled_serving.run import run_service


CONFIG = Path('configs/scheduled-serving-server.json')
CORPUS = Path('runs/performance/20261107T210935152729Z/requests.json')
CORPUS_SHA256 = '7977026a9873e0ee897fdb8b7dab642104e280f585592dd633fa5460204a103b'
MODEL_ID = '17aaae44451f3d9fe5de7e4ee9a6e2e3a3ff981bf55374c4d836227f76d9abdb'
TARGET_P99_MS = 20
POINT_SECONDS = 60


def collect(directory, requests, settings, port, health):
    if health['model_id'] != MODEL_ID:
        raise ValueError('Model differs from the frozen reference')
    subprocess.run([sys.executable, '-m', 'scheduled_serving.load',
                    '--url', f'http://127.0.0.1:{port}', '--requests', str(requests),
                    '--settings', *settings, '--duration', str(POINT_SECONDS), '--output', str(directory / 'load')],
                   check=True)


def trial(directory, requests, settings, transport):
    run_service('check', CONFIG, load_config(CONFIG), 100, transport, directory,
                on_verified=partial(collect, directory, requests, settings))


def summarize_point(point):
    measurement = point['measurement']
    delta = point['server_delta']
    before, after = point['server_before'], point['server_after']
    p99 = measurement['successful_http_ms'].get('p99')
    row = {key: measurement[key] for key in ('offered', 'attempted', 'successful', 'client_dropped',
           'success_qps_in_window', 'success_fraction_of_offered')}
    row.update(setting=point['setting'], error_counts=point['error_counts'], successful_http_p99_ms=p99,
               all_attempt_http_p99_ms=point['all_attempt_http_ms'].get('p99'),
               successful_scheduled_p99_ms=measurement['successful_scheduled_to_completion_ms'].get('p99'),
               dispatch_lag_p99_ms=measurement['all_attempt_dispatch_lag_ms'].get('p99'),
               server_cpu_seconds=after['process_cpu_seconds'] - before['process_cpu_seconds'],
               counters=delta['counters'],
               queue_mean_ms=delta['histograms'].get('queue_started_ms', {}).get('mean_ms'),
               core_mean_ms=delta['histograms'].get('core_ms', {}).get('mean_ms'))
    if point['setting'] == 'open-200':
        row['met_open_200_target'] = (row['offered'] == row['successful'] == 12000
                                      and p99 is not None and p99 <= TARGET_P99_MS)
    if point['setting'] in PACING_SETTINGS:
        row['client_pacing'] = point['client_pacing']
        if row['client_pacing']['policy'] != PACING_SETTINGS[point['setting']]:
            raise ValueError('Client pacing policy differs from requested setting')
    return row


def read_trial(directory, settings, transport):
    server = json.loads((directory / 'results.json').read_text())
    (load_summary,) = (directory / 'load').glob('*/summary.json')
    load = json.loads(load_summary.read_text())
    instance = server['metrics']['instance_id']
    service = server['service_identity']
    health = load['health']
    if (server['status'] != 'PASS' or not all(server['shutdown'].values())
            or server['parity']['requests'] != 101 or not server['parity']['scores_and_topk_exact']
            or server['metrics']['connections'] != 0 or load['status'] != 'COMPLETE'
            or load['settings'] != settings or load['requests_sha256'] != CORPUS_SHA256
            or health['instance_id'] != instance or health['service_id'] != service['service_id']
            or health['transport'] != transport or health['model_id'] != MODEL_ID):
        raise ValueError(f'Incomplete or mismatched trial in {directory}')
    rows = []
    for setting in settings:
        point = json.loads((load_summary.parent / (setting + '.json')).read_text())
        m = point['measurement']
        mode, concurrency, rate = SETTINGS[setting]
        if (point['status'] != 'COMPLETE' or point['health'] != health or point['setting'] != setting
                or point['requests_sha256'] != CORPUS_SHA256 or m['duration_seconds'] != POINT_SECONDS
                or (m['mode'], m['concurrency_limit'], m['offered_rate'])
                != (mode, concurrency, rate if mode == 'open' else None)):
            raise ValueError(f'Incomplete or mismatched load point {setting}')
        rows.append(summarize_point(point))
    runtime = health['runtime']
    # Available GPU memory fluctuates. Compare allocation size and record free memory per trial.
    matched = {'runtime': {k: v for k, v in runtime.items() if k != 'gpu_free_memory_bytes'},
               'model_id': health['model_id'], 'requests_sha256': load['requests_sha256'],
               'config': service['config'], 'config_sha256': server['config_sha256'],
               'sources': service['sources'], 'git_commit': load['provenance']['git_commit'],
               'benchmark_sources': load['provenance']['benchmark_sources'],
               'input_hashes': server['runtime_manifest']['inputs_sha256']}
    return matched, {'directory': directory.name, 'transport': transport, 'instance_id': instance,
                     'service_id': service['service_id'], 'runtime': runtime, 'parity': server['parity'],
                     'shutdown': server['shutdown'], 'thread_starts_total': {
                         k: server['metrics']['counters'].get(k, 0)
                         for k in ('handler_threads_started', 'deadline_threads_started')}, 'points': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, default=CORPUS)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument('--pacing', action='store_true',
                            help='Compare immediate and Retry-After clients using the same handler pool')
    experiment.add_argument('--settings', nargs='+', choices=('open-200', 'open-400', 'closed-32'),
                        default=['open-200'])
    args = parser.parse_args()
    if len(set(args.settings)) != len(args.settings):
        parser.error('settings must be distinct')
    if not args.requests.is_file() or digest(args.requests) != CORPUS_SHA256:
        parser.error(f'Provide the original request corpus with --requests (SHA-256 {CORPUS_SHA256})')
    if args.pacing:
        settings = list(PACING_SETTINGS)
        trials = [(label, 'handler-pool', [label]) for label in settings]
        out = output_dir('runs/pacing-comparison')
    else:
        settings = args.settings
        trials = [(transport, transport, settings) for transport in TRANSPORTS]
        out = output_dir('runs/transport-comparison')
    summary = out / 'summary.json'
    report = {'status': 'RUNNING', 'settings': settings, 'duration_seconds': POINT_SECONDS,
              'rounds': 2, 'provenance': provenance(), 'trials': []}
    if not args.pacing:
        report['open_200_target'] = {'successful_of_offered': 12000, 'successful_p99_ms_max': TARGET_P99_MS}
    ctx = multiprocessing.get_context('spawn')
    try:
        for round_no, order in enumerate((trials, trials[::-1]), 1):
            for label, transport, trial_settings in order:
                directory = out / f'{round_no}-{label}'
                directory.mkdir()
                print(f'Round {round_no}/2, {label}', flush=True)
                child = ctx.Process(target=trial, args=(directory, args.requests, trial_settings, transport))
                child.start()
                try:
                    child.join()
                finally:
                    if child.is_alive():
                        # Let the trial stop its client and write the shutdown report.
                        try:
                            os.kill(child.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                        child.join(15)
                    if child.is_alive():
                        child.terminate()
                        child.join()
                if child.exitcode:
                    raise RuntimeError(f'Trial failed in {directory} (exit {child.exitcode})')
                matched, row = read_trial(directory, trial_settings, transport)
                if matched != report.setdefault('matched', matched):
                    raise ValueError(f'Runtime, inputs or code changed in {directory}')
                report['trials'].append(dict(round=round_no, **row))
                write(summary, report)
        if 'open-200' in settings:
            report['met_open_200_target_both_rounds'] = {
                name: all(p['met_open_200_target'] for t in report['trials'] if t['transport'] == name
                          for p in t['points'] if p['setting'] == 'open-200') for name in TRANSPORTS}
            print('200/s target in both rounds:', report['met_open_200_target_both_rounds'], flush=True)
        if args.pacing:
            report['paired_rounds'] = []
            for round_no in (1, 2):
                points = {t['points'][0]['client_pacing']['policy']: t['points'][0]
                          for t in report['trials'] if t['round'] == round_no}
                a, b = points['immediate'], points['retry-after']
                report['paired_rounds'].append({
                    'round': round_no,
                    'both_clients_received_503': all(p['error_counts'].get('http_503', 0) > 0
                                                     for p in (a, b)),
                    'retry_after_minus_immediate_success_qps': b['success_qps_in_window'] - a['success_qps_in_window'],
                    'retry_after_minus_immediate_late_results': b['counters'].get('late_results_discarded', 0)
                                                              - a['counters'].get('late_results_discarded', 0)})
        report['status'] = 'COMPLETE'
    except BaseException as error:
        report.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAIL',
                      error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write(summary, report)
        name = 'Client pacing' if args.pacing else 'Transport'
        print(f'{name} comparison: {report["status"]}\nSummary: {summary}', flush=True)


if __name__ == '__main__':
    main()
