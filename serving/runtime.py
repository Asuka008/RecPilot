"""Load authenticated artifacts once and expose portable FP32 inference."""
import json
import os
from pathlib import Path
import platform
import time

from scripts.hybrid_test_admission import admit_test
from scripts.compare_title_validation import digest
from scripts.evaluate_selected_title import file_digest
from serving.core import Recommender

ROOT = Path(__file__).resolve().parents[1]


class TorchScorer:
    """Use the same left padding and scoring head as offline evaluation."""

    def __init__(self, model, device):
        import torch
        self.model, self.device = model, device
        self.candidates = torch.arange(1, model.num_items + 1, device=device)

    def __call__(self, history):
        import torch
        from src.data.dataset import _left_pad_history
        ids, mask = _left_pad_history(history, self.model.sequence_length)
        with torch.inference_mode():
            ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            mask = torch.tensor([mask], dtype=torch.bool, device=self.device)
            return self.model.score_candidates(self.model(ids, mask)[:, -1], self.candidates)[0].cpu().numpy()


def load_runtime(path):
    """Keep deployment settings separate from the immutable training recipe."""
    start = time.perf_counter()
    config = json.loads(Path(path).read_text())
    required = {'recipe', 'device', 'host', 'port', 'max_history', 'max_k', 'warmup', 'requests', 'output_dir'}
    if set(config) != required or config['host'] != '127.0.0.1':
        raise ValueError('Expected documented config and loopback host')
    for key, limit in [('port', 65535), ('max_history', 4096), ('max_k', 100), ('warmup', 1000), ('requests', 10000)]:
        if type(config[key]) is not int or not 1 <= config[key] <= limit:
            raise ValueError(f'Invalid {key}')
    recipe_path = ROOT / config['recipe']
    recipe = json.loads(recipe_path.read_text())
    reports, baseline, checkpoints, tracked, expected = admit_test(ROOT, recipe)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from diagnostics.tail_admission import verify_payload, verify_unchanged
    from src.run_baselines import load_data, git_state
    from src.baselines import ItemCF, Popularity
    from src.models.id_hstu import IDHSTU
    training = reports['id']['config']
    records, n, hashes = load_data(ROOT / training['data_dir'])
    users = digest([r['user_id'] for r in records])
    identity = {'data_sha256': hashes, 'train_users_sha256': users, 'eval_users_sha256': users}
    if identity != expected or digest(identity) != reports['id']['provenance']['data_id']:
        raise ValueError('Data identity differs')
    torch.manual_seed(training['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    payload = torch.load(checkpoints['id'], map_location='cpu', weights_only=True)
    verify_payload(payload, reports['id'])
    model = IDHSTU(num_items=n, **training['model'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.to(device).eval()
    del payload
    cf = ItemCF(records, n, baseline['config']['itemcf_neighbors'])
    for file in list((ROOT / 'serving').glob('*.py')) + [Path(path).resolve(), recipe_path, ROOT / 'scripts/hybrid_test_admission.py']:
        tracked[str(file)] = file_digest(file)
    verify_unchanged(tracked)
    model_id = digest({'checkpoint': file_digest(checkpoints['id']), 'data': identity,
                       'recipe': recipe, 'serving_sources': {str(p.relative_to(ROOT)): file_digest(p) for p in (ROOT / 'serving').glob('*.py')}})
    engine = Recommender(TorchScorer(model, device), cf, Popularity(records, n), n, model_id,
                         config['max_history'], config['max_k'])
    runtime = {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
               'cuda_runtime': torch.version.cuda, 'device': str(device), 'precision': 'FP32',
               'tf32': False, 'torch_threads': torch.get_num_threads(), 'catalog_items': n,
               'platform': platform.platform(), 'itemcf_neighbors': baseline['config']['itemcf_neighbors'],
               'itemcf_edges': sum(len(v) for v in cf.neighbors.values())}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        runtime.update(gpu=torch.cuda.get_device_name(device), gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                       gpu_free_memory_bytes=torch.cuda.mem_get_info(device)[0])
    meta = {'model_id': model_id, 'runtime': runtime, 'config': config, 'training_config': training,
            'inputs_sha256': tracked, 'git': git_state(), 'startup_seconds': time.perf_counter() - start}
    return engine, records, meta
