"""Dependency-free configuration and audit helpers for the official ML-1M gate."""

import csv
import hashlib
import json
import math
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
    """Validate single-device smoke and fixed-recipe training configurations."""
    expected = {'seed', 'batch_size', 'eval_batch_size', 'num_workers',
                'prefetch_factor', 'master_port', 'output_dir'}
    if not expected <= set(config) or set(config) - expected - {'num_epochs'}:
        raise ValueError(f'Expected configuration keys: {sorted(expected)}')
    for key in expected - {'output_dir'}:
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'{key} must be an integer in its supported range')
    if config['master_port'] > 65535 or config['batch_size'] > 128:
        raise ValueError('Use a valid port and a conservative batch size <= 128')
    if not isinstance(config['output_dir'], str) or not config['output_dir']:
        raise ValueError('output_dir must be a nonempty path')
    epochs = config.get('num_epochs', 1)
    if type(epochs) is not int or epochs not in (1, 101):
        raise ValueError('num_epochs must be 1 for smoke or 101 for the official recipe')


def validate_action(config, action):
    """Prevent a smoke command from starting formal training accidentally."""
    validate_config(config)
    epochs = config.get('num_epochs', 1)
    if action == 'smoke' and epochs != 1:
        raise ValueError('smoke requires one epoch; use train with the formal configuration')
    if action == 'train' and (epochs, config['batch_size'], config['eval_batch_size']) != (101, 128, 128):
        raise ValueError('train requires the official 101 epochs and train/eval batch 128')


def bindings(config):
    """Override runtime settings only; retain the pinned architecture and objective."""
    validate_config(config)
    return [
        f"train_fn.num_epochs = {config.get('num_epochs', 1)}",
        f"train_fn.local_batch_size = {config['batch_size']}",
        f"train_fn.eval_batch_size = {config['eval_batch_size']}",
        f"train_fn.random_seed = {config['seed']}",
        'train_fn.main_module_bf16 = False',
        'train_fn.full_eval_every_n = 1',
        f"create_data_loader.num_workers = {config['num_workers']}",
        f"create_data_loader.prefetch_factor = {config['prefetch_factor']}",
    ]


REFERENCE_TAGS = ('hr@10', 'ndcg@10', 'hr@50', 'ndcg@50', 'hr@200', 'ndcg@200')


def epoch_history(scalars, num_epochs):
    """Require exactly one complete, finite evaluation at each expected epoch."""
    history = [{'epoch': epoch} for epoch in range(num_epochs)]
    expected_steps = set(range(num_epochs))
    for tag in REFERENCE_TAGS:
        events = scalars.get(f'eval_epoch/{tag}', [])
        if len(events) != num_epochs or {event['step'] for event in events} != expected_steps:
            raise ValueError(f'Missing or duplicate epoch evaluation: {tag}')
        for event in events:
            value = event['value']
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid ranking metric: {tag}')
            history[event['step']][tag] = value
    return history


def compare_reference(final_metrics, reference):
    """Compare the preselected final epoch; never select an epoch by test scores."""
    tolerance = reference['absolute_tolerance']
    if not isinstance(tolerance, (int, float)) or not 0 < tolerance < 1:
        raise ValueError('Invalid reference comparison tolerance')
    if set(reference['metrics']) != set(REFERENCE_TAGS):
        raise ValueError('Expected all six official reference metrics')
    comparisons = {}
    for tag in REFERENCE_TAGS:
        actual, target = final_metrics[tag], reference['metrics'][tag]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (actual, target)):
            raise ValueError(f'Invalid reference or final metric: {tag}')
        gap = actual - target
        comparisons[tag] = {'actual': actual, 'reference': target, 'signed_gap': gap,
                            'within_tolerance': abs(gap) <= tolerance}
    return {'status': ('WITHIN_TOLERANCE' if all(row['within_tolerance'] for row in comparisons.values())
                       else 'REVIEW_REQUIRED'),
            'selection': 'final epoch 100, fixed before execution',
            'absolute_tolerance': tolerance, 'metrics': comparisons,
            'interpretation': 'Engineering comparison band, not a statistical confidence interval or proof of reproduction'}


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
