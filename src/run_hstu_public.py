"""Prepare only ML-1M and run one epoch of the pinned official HSTU recipe."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from urllib.request import urlretrieve
from zipfile import ZipFile

# Workers use an absolute script path so their working directory can be isolated.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.check_hstu_reference import command, verify_upstream
from src.hstu_public_config import (
    DATA_FILES, audit_sequences, bindings, sha256, validate_config, verify_data,
)

URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
EXPECTED = {'users': 6040, 'items': 3706, 'interactions': 1000209, 'max_item_id': 3952}


def prepare_data(directory, spec, seed):
    """Reuse verified data or invoke only the official ML-1M preprocessor."""
    if (directory / 'manifest.json').exists():
        return verify_data(directory, spec['revision'], seed)
    archive = directory / 'tmp/movielens1m.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        print(f'Downloading ML-1M only: {URL}', flush=True)
        urlretrieve(URL, temporary)
        temporary.replace(archive)
    with ZipFile(archive) as source:
        for member in source.namelist():
            path = Path(member)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'ml-1m':
                raise ValueError(f'Unexpected ML-1M archive member: {member}')
        if source.testzip() is not None:
            raise ValueError('Corrupt ML-1M archive')
    import numpy as np
    from generative_recommenders.research.data.preprocessor import get_common_preprocessors
    random.seed(seed)
    np.random.seed(seed)
    os.chdir(directory)
    get_common_preprocessors()['ml-1m'].preprocess_rating()
    counts = audit_sequences(directory / 'tmp/ml-1m/sasrec_format.csv')
    if counts != EXPECTED:
        raise ValueError(f'Unexpected ML-1M counts: {counts}')
    manifest = {'dataset': 'ml-1m', 'download_url': URL, 'seed': seed,
                'upstream_revision': spec['revision'], 'counts': counts,
                'files_sha256': {name: sha256(directory / name) for name in DATA_FILES}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def train(config, upstream, run_dir, report):
    """Call the unmodified official trainer with one NCCL rank and full epoch eval."""
    import numpy as np
    import torch
    import fbgemm_gpu  # noqa: F401
    import gin
    from generative_recommenders.research.trainer.train import train_fn
    if not torch.cuda.is_available():
        raise RuntimeError('This gate requires the server CUDA environment')
    torch.cuda.set_device(0)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    report['runtime'].update({
        'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'precision': 'float32', 'tf32': True, 'world_size': 1,
        'deterministic_algorithms': False,
    })
    print(f"GPU: {report['runtime']['gpu']}; "
          f"memory={report['runtime']['gpu_total_memory_bytes'] / 2**30:.2f} GiB; "
          f"batch={config['batch_size']}; FP32 with TF32", flush=True)
    spec = report['upstream']
    gin.parse_config_file(str(upstream / spec['recipe']))
    overrides = bindings(config)
    gin.parse_config(overrides)
    report['overrides'] = overrides
    (run_dir / 'overrides.gin').write_text('\n'.join(overrides) + '\n')
    (run_dir / 'effective-config.gin').write_text(gin.config_str())
    os.chdir(run_dir)
    try:
        train_fn(rank=0, world_size=1, master_port=config['master_port'])
        torch.cuda.synchronize()
    finally:
        report['runtime'].update({
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        })
        print(f"Peak GPU allocated/reserved: "
              f"{report['runtime']['peak_allocated_bytes'] / 2**30:.3f}/"
              f"{report['runtime']['peak_reserved_bytes'] / 2**30:.3f} GiB", flush=True)
        (run_dir / 'operative-config.gin').write_text(gin.operative_config_str())
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    scalars = {}
    for log_dir in (run_dir / 'exps').rglob('*'):
        if log_dir.is_dir() and list(log_dir.glob('events.out.tfevents.*')):
            events = EventAccumulator(str(log_dir), size_guidance={'scalars': 0}).Reload()
            for tag in events.Tags()['scalars']:
                values = [{'step': event.step, 'value': event.value}
                          for event in events.Scalars(tag)]
                if any(not math.isfinite(event['value']) for event in values):
                    raise ValueError(f'Nonfinite TensorBoard scalar: {tag}')
                scalars[tag] = values
    report['scalars'] = scalars
    for tag in ('losses/ar_loss', 'eval_epoch/ndcg@10', 'eval_epoch/hr@10'):
        if not scalars.get(tag):
            raise RuntimeError(f'Missing completed training/evaluation evidence: {tag}')
    checkpoints = sorted(str(path.relative_to(run_dir)) for path in (run_dir / 'ckpts').rglob('*')
                         if path.is_file())
    if not checkpoints:
        raise RuntimeError('Official trainer did not save its final checkpoint')
    report['checkpoints'] = checkpoints
    report['epoch_metrics'] = {key: values[-1]['value'] for key, values in scalars.items()
                               if key.startswith('eval_epoch/')}
    print(f"Epoch metrics: {report['epoch_metrics']}", flush=True)


def worker(args, config):
    """Persist failure evidence as well as successful execution provenance."""
    run_dir, upstream, directory = args.run_dir, args.upstream_dir, args.data_dir
    spec = json.loads((ROOT / 'configs/hstu-reference-upstream.json').read_text())
    report = {'status': 'RUNNING', 'action': args.action, 'config': config, 'upstream': spec,
              'scope': 'Official ML-1M data preparation or one-epoch FP32 pipeline gate; not a reproduction score',
              'runtime': {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable}}
    start = time.monotonic()
    try:
        if sys.version_info[:2] != (3, 11) or platform.system() != 'Linux':
            raise RuntimeError('Activate the Linux Python 3.11 HSTU reference environment')
        verify_upstream(upstream, spec)
        packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                    if d.metadata['Name']}
        report['runtime']['packages'] = packages
        for name, expected in spec['expected_packages'].items():
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise RuntimeError(f'{name}: expected {expected}, found {actual}')
        report['git'] = {'commit': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                         'dirty': bool(command(['git', '-C', str(ROOT), 'status', '--porcelain']))}
        sys.path.insert(0, str(upstream))
        if args.action == 'prepare':
            report['data'] = prepare_data(directory, spec, config['seed'])
            print(f"ML-1M verified: {report['data']['counts']}", flush=True)
        else:
            report['data'] = verify_data(directory, spec['revision'], config['seed'])
            report['data_manifest_sha256'] = sha256(directory / 'manifest.json')
            (run_dir / 'tmp').symlink_to(directory / 'tmp', target_is_directory=True)
            train(config, upstream, run_dir, report)
            verify_data(directory, spec['revision'], config['seed'])
        report['status'] = 'PASS'
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = {'type': type(error).__name__, 'message': str(error),
                           'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (run_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"Saved: {run_dir / 'results.json'}", flush=True)
    print(f"HSTU ML-1M {args.action}: {report['status']}", flush=True)
    return 0 if report['status'] == 'PASS' else 1


def main():
    """Launch a logged child so upstream relative paths never touch its checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/hstu-ml1m-smoke-server.json')
    parser.add_argument('--upstream-dir', type=Path,
                        default=Path.home() / 'upstream/generative-recommenders-reference')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'runs/hstu-ml1m-data')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    for key in ('config', 'upstream_dir', 'data_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.worker:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        return worker(args, config)
    if args.data_dir == args.upstream_dir or args.upstream_dir in args.data_dir.parents:
        raise ValueError('Keep generated data outside the upstream checkout')
    run_dir = (Path(config['output_dir']).expanduser().resolve() /
               datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if run_dir == args.data_dir or args.data_dir in run_dir.parents or run_dir in args.data_dir.parents:
        raise ValueError('Data and experiment directories must be separate')
    if args.upstream_dir == run_dir or args.upstream_dir in run_dir.parents:
        raise ValueError('Keep experiment outputs outside the upstream checkout')
    run_dir.mkdir(parents=True, exist_ok=False)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    invocation = [sys.executable, '-u', str(Path(__file__).resolve()), args.action,
                  '--config', str(args.config), '--upstream-dir', str(args.upstream_dir),
                  '--data-dir', str(args.data_dir), '--worker', '--run-dir', str(run_dir)]
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with (run_dir / 'console.log').open('w') as log:
        with subprocess.Popen(invocation, cwd=run_dir, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return process.wait()


if __name__ == '__main__':
    raise SystemExit(main())
