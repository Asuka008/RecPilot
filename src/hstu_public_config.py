"""Dependency-free configuration and audit helpers for the official ML-1M gate."""

import csv
import hashlib
import json
from pathlib import Path


DATA_FILES = (
    'tmp/movielens1m.zip', 'tmp/ml-1m/ratings.dat', 'tmp/ml-1m/users.dat',
    'tmp/ml-1m/movies.dat', 'tmp/ml-1m/sasrec_format.csv',
    'tmp/processed/ml-1m/movies.csv', 'tmp/processed/ml-1m/ratings.csv',
    'tmp/processed/ml-1m/users.csv',
)


def sha256(path):
    """Hash artifacts without loading the whole file into memory."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_config(config):
    """Keep the first public-data gate small and single-device."""
    expected = {'seed', 'batch_size', 'eval_batch_size', 'num_workers',
                'prefetch_factor', 'master_port', 'output_dir'}
    if set(config) != expected:
        raise ValueError(f'Expected configuration keys: {sorted(expected)}')
    for key in expected - {'output_dir'}:
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'{key} must be an integer in its supported range')
    if config['master_port'] > 65535 or config['batch_size'] > 128:
        raise ValueError('Use a valid port and a conservative batch size <= 128')
    if not isinstance(config['output_dir'], str) or not config['output_dir']:
        raise ValueError('output_dir must be a nonempty path')


def bindings(config):
    """Override runtime settings only; retain the pinned architecture and objective."""
    validate_config(config)
    return [
        'train_fn.num_epochs = 1',
        f"train_fn.local_batch_size = {config['batch_size']}",
        f"train_fn.eval_batch_size = {config['eval_batch_size']}",
        f"train_fn.random_seed = {config['seed']}",
        'train_fn.main_module_bf16 = False',
        'train_fn.full_eval_every_n = 1',
        f"create_data_loader.num_workers = {config['num_workers']}",
        f"create_data_loader.prefetch_factor = {config['prefetch_factor']}",
    ]


def audit_sequences(path):
    """Validate aligned chronological sequences using only the standard library."""
    users, items, interactions = set(), set(), 0
    with Path(path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            user = int(row['user_id'])
            ids = [int(x) for x in row['sequence_item_ids'].split(',')]
            times = [int(x) for x in row['sequence_timestamps'].split(',')]
            ratings = [int(x) for x in row['sequence_ratings'].split(',')]
            if user in users or len(ids) < 3 or not len(ids) == len(times) == len(ratings):
                raise ValueError('Duplicate user or unaligned/short sequence')
            if any(x <= 0 for x in ids) or times != sorted(times):
                raise ValueError('Nonpositive item ID or nonchronological timestamps')
            users.add(user)
            items.update(ids)
            interactions += len(ids)
    return {'users': len(users), 'items': len(items), 'interactions': interactions,
            'max_item_id': max(items, default=0)}


def verify_data(directory, revision, seed):
    """Reject stale preprocessing or modified artifacts before training."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['upstream_revision'] != revision or manifest['seed'] != seed:
        raise ValueError('Dataset revision/seed mismatch; use a separate data directory')
    if set(manifest['files_sha256']) != set(DATA_FILES):
        raise ValueError('Dataset manifest is incomplete')
    for name, expected in manifest['files_sha256'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    return manifest
