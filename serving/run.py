"""Run a local recommendation endpoint or label-free serving benchmark."""
import argparse
import json
from pathlib import Path

from serving.runtime import ROOT, load_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['serve', 'benchmark'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/hybrid-serving-server.json')
    args = parser.parse_args()
    engine, records, meta = load_runtime(args.config)
    print('Artifact admission: PASS', flush=True)
    print('Actual runtime:', json.dumps(meta['runtime']), flush=True)
    print('Startup seconds:', meta['startup_seconds'], flush=True)
    if args.action == 'benchmark':
        from serving.benchmark import benchmark
        benchmark(engine, records, meta)
    else:
        from serving.http import serve
        serve(engine, meta['config'])


if __name__ == '__main__':
    main()
