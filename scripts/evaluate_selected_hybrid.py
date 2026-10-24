"""Evaluate one frozen validation-selected hybrid on test."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.evaluate_hybrid_validation import combine
from scripts.hybrid_test_admission import admit_test

def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/hybrid-selected-test-server.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    recipe=json.loads(args.config.read_text())
    admitted=admit_test(ROOT,recipe)
    print('Frozen hybrid test admission: PASS',flush=True)
    if args.check_only: return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from diagnostics.tail_admission import verify_payload, verify_unchanged, source_hashes
    from scripts.compare_title_validation import digest
    from scripts.evaluate_selected_title import file_digest
    from scripts.audit_recommendation_mechanisms import write_json
    from src.run_baselines import load_data, git_state
    from src.baselines import ItemCF
    from src.models.id_hstu import IDHSTU
    from src.data.dataset import SASRecEvalDataset
    from src.microlens_training import make_loader
    from src.evaluation import RankingAccumulator
    directory = ROOT/'runs/hybrid-selected-test'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True)
    result = {'status':'RUNNING', 'split':'test', 'alphas':[recipe['alpha']],
              'formula':'(1-alpha)*(HSTU_cosine+1)/2 + alpha*ItemCF/max(ItemCF); zero CF falls back to raw HSTU',
              'selection_rule':'already frozen on validation; no test selection',
              'scope':'frozen recipe test evaluation; reused benchmark, no selection on test'}
    start = time.perf_counter()
    write_json(directory/'results.json',result)
    try:
        reports, baseline, checkpoints, tracked, expected_identity=admitted
        tracked[str(Path(__file__).resolve())]=file_digest(Path(__file__))
        tracked[str(args.config.resolve())]=file_digest(args.config)
        tracked.update({str(ROOT/k):v for k,v in source_hashes(ROOT, ('diagnostics',)).items()})
        for helper in ('hybrid_test_admission.py','compare_title_validation.py','evaluate_selected_title.py','diagnose_hstu_scores.py','audit_recommendation_mechanisms.py'):
            path=ROOT/'scripts'/helper
            tracked[str(path)]=file_digest(path)
        config=reports['id']['config']
        records,n,hashes=load_data(ROOT/config['data_dir'])
        users=[r['user_id'] for r in records]
        identity={'data_sha256':hashes,'train_users_sha256':digest(users),'eval_users_sha256':digest(users)}
        if identity!=expected_identity or digest(identity)!=reports['id']['provenance']['data_id']:
            raise ValueError('Actual data/user identity differs')
        torch.manual_seed(config['seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        device=torch.device(config['device'])
        runtime={'torch':str(torch.__version__),'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'device':str(device),'precision':'FP32; TF32 off'}
        if device.type=='cuda':
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            runtime.update(gpu=torch.cuda.get_device_name(device),gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
        print('Actual runtime:',json.dumps(runtime),flush=True)
        model=IDHSTU(num_items=n,**config['model']).to(device)
        payload=torch.load(checkpoints['id'],map_location='cpu',weights_only=True)
        verify_payload(payload,reports['id'])
        model.load_state_dict(payload['model_state_dict'],strict=True)
        model.eval()
        del payload
        cf=ItemCF(records,n,baseline['config']['itemcf_neighbors'])
        frequency=Counter(i for r in records for i in r['train'])
        accumulators={str(a):RankingAccumulator([10,20],frequency) for a in result['alphas']}
        by_user={r['user_id']:r for r in records}
        dataset=SASRecEvalDataset(ROOT/config['data_dir']/'sequences.jsonl',config['model']['sequence_length'],'test')
        loader=make_loader(dataset,list(range(len(records))),config['eval_batch_size'],config['num_workers'],config['seed'],device)
        with torch.no_grad():
            for index,batch in enumerate(loader,1):
                ids=batch['input_ids'].to(device);mask=batch['padding_mask'].to(device)
                values=model.score_candidates(model(ids,mask)[:,-1],torch.arange(1,n+1,device=device)).cpu().numpy()
                for user,row in zip(batch['user_id'].tolist(),values):
                    r=by_user[user];history=r['train']+[r['val']];a=np.concatenate(([0.],row));b=cf.score(history)
                    # Scale only eligible CF scores so filtered history cannot set its range.
                    b[0]=0;b[history]=0
                    for alpha in result['alphas']:
                        accumulators[str(alpha)].add(combine(a,b,alpha),history,r['test'])
                if index%100==0 or index==len(loader):
                    print(f'Frozen hybrid test batches {index}/{len(loader)}',flush=True)
        variants={key:acc.result('test') for key,acc in accumulators.items()}
        result['evaluation']=variants[str(recipe['alpha'])]
        result['frozen_recipe']=recipe
        result['provenance']={'git':git_state(),'inputs_sha256':tracked,'data_identity':identity,'seed':config['seed'],'training_config':config}
        verify_unchanged(tracked)
        result['runtime']=runtime
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            result['memory_bytes']={'peak_allocated':torch.cuda.max_memory_allocated(device),'peak_reserved':torch.cuda.max_memory_reserved(device)}
        result['status']='PASS'
        for alpha,evaluation in variants.items():
            print(json.dumps({'alpha':alpha,'groups':evaluation['groups_by_target_train_frequency']}),flush=True)
        print('Frozen alpha:',recipe['alpha'],flush=True)
    except Exception as error:
        result.update(status='FAIL',error=traceback.format_exc())
        print(result['error'],flush=True)
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        write_json(directory/'results.json',result)
        print(f"Saved: {directory/'results.json'}\nFrozen hybrid test: {result['status']}",flush=True)
    if result['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    run()
