"""Bounded validation-only score fusion of frozen HSTU and training-only ItemCF."""
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


def combine(neural, collaborative, alpha):
    """Map cosine to [0,1], scale positive CF scores by their user maximum."""
    import numpy as np
    a, b = np.asarray(neural, dtype=np.float64), np.asarray(collaborative, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected aligned finite scores')
    if not 0 <= alpha <= 1 or (b < 0).any():
        raise ValueError('Invalid weight or negative CF score')
    if alpha == 0 or b.max() == 0:
        return a.copy()
    return (1-alpha) * (a+1)/2 + alpha*b/b.max()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/recommendation-mechanism-audit-server.json')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from diagnostics.tail_admission import admit, verify_payload, verify_unchanged, source_hashes
    from diagnostics.tail_statistics import verify_diagnostic_anchor
    from scripts.compare_title_validation import best_validation, digest
    from scripts.evaluate_selected_title import file_digest
    from scripts.audit_recommendation_mechanisms import write_json
    from src.run_baselines import load_data, git_state
    from src.baselines import ItemCF
    from src.models.id_hstu import IDHSTU
    from src.data.dataset import SASRecEvalDataset
    from src.microlens_training import make_loader
    from src.evaluation import RankingAccumulator
    directory = ROOT/'runs/hybrid-validation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True)
    result = {'status':'RUNNING', 'split':'val', 'alphas':[0, .1, .25, .5],
              'formula':'(1-alpha)*(HSTU_cosine+1)/2 + alpha*ItemCF/max(ItemCF); zero CF falls back to raw HSTU',
              'selection_rule':'max overall validation NDCG@10 among variants with no overall regression versus raw ID and at least one 1-5 hit; smaller alpha breaks ties',
              'scope':'single-seed diagnostic; no test evaluation or new training'}
    start = time.perf_counter()
    write_json(directory/'results.json',result)
    try:
        recipe=json.loads(args.config.read_text())
        reports, baseline, checkpoints, tracked=admit(ROOT,recipe)
        tracked[str(Path(__file__).resolve())]=file_digest(Path(__file__))
        tracked[str(args.config.resolve())]=file_digest(args.config)
        tracked.update({str(ROOT/k):v for k,v in source_hashes(ROOT, ('diagnostics',)).items()})
        for helper in ('compare_title_validation.py','evaluate_selected_title.py','diagnose_hstu_scores.py','audit_recommendation_mechanisms.py'):
            path=ROOT/'scripts'/helper
            tracked[str(path)]=file_digest(path)
        config=reports['id']['config']
        records,n,hashes=load_data(ROOT/config['data_dir'])
        users=[r['user_id'] for r in records]
        identity={'data_sha256':hashes,'train_users_sha256':digest(users),'eval_users_sha256':digest(users)}
        if digest(identity)!=reports['id']['provenance']['data_id']:
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
        cf_acc=RankingAccumulator([10,20],frequency)
        by_user={r['user_id']:r for r in records}
        dataset=SASRecEvalDataset(ROOT/config['data_dir']/'sequences.jsonl',config['model']['sequence_length'],'val')
        loader=make_loader(dataset,list(range(len(records))),config['eval_batch_size'],config['num_workers'],config['seed'],device)
        with torch.no_grad():
            for index,batch in enumerate(loader,1):
                ids=batch['input_ids'].to(device);mask=batch['padding_mask'].to(device)
                values=model.score_candidates(model(ids,mask)[:,-1],torch.arange(1,n+1,device=device)).cpu().numpy()
                for user,row in zip(batch['user_id'].tolist(),values):
                    r=by_user[user];a=np.concatenate(([0.],row));b=cf.score(r['train'])
                    # Scale only eligible CF scores so filtered history cannot set its range.
                    b[0]=0;b[r['train']]=0
                    cf_acc.add(b,r['train'],r['val'])
                    for alpha in result['alphas']:
                        accumulators[str(alpha)].add(combine(a,b,alpha),r['train'],r['val'])
                if index%100==0 or index==len(loader):
                    print(f'Hybrid validation batches {index}/{len(loader)}',flush=True)
        variants={key:acc.result('val') for key,acc in accumulators.items()}
        result['anchor_coverage']={
            'id':verify_diagnostic_anchor('id',variants['0'],best_validation(reports['id']),1e-10),
            'itemcf':verify_diagnostic_anchor('itemcf',cf_acc.result('val'),baseline['models']['itemcf'],1e-10)}
        result['variants']=variants
        base=variants['0']['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']
        eligible=[a for a in result['alphas'] if variants[str(a)]['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']>=base and variants[str(a)]['groups_by_target_train_frequency']['1-5']['metrics']['Recall@20']>0]
        winner=max(eligible,key=lambda a:variants[str(a)]['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']) if eligible else None
        result['selection']={'alpha':winner,'status':'candidate_for_review' if winner is not None else 'no_qualifying_candidate','not_independent_confirmation':True}
        result['corrected_hstu_validation_reference']=.040344999048154036
        result['provenance']={'git':git_state(),'inputs_sha256':tracked,'data_identity':identity,'seed':config['seed'],'training_config':config}
        verify_unchanged(tracked)
        result['runtime']=runtime
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            result['memory_bytes']={'peak_allocated':torch.cuda.max_memory_allocated(device),'peak_reserved':torch.cuda.max_memory_reserved(device)}
        result['status']='PASS'
        for alpha,evaluation in variants.items():
            print(json.dumps({'alpha':alpha,'groups':evaluation['groups_by_target_train_frequency']}),flush=True)
        print('Selection:',result['selection'],flush=True)
    except Exception as error:
        result.update(status='FAIL',error=traceback.format_exc())
        print(result['error'],flush=True)
    finally:
        result['elapsed_seconds']=time.perf_counter()-start
        write_json(directory/'results.json',result)
        print(f"Saved: {directory/'results.json'}\nHybrid validation: {result['status']}",flush=True)
    if result['status']!='PASS':
        raise SystemExit(1)


if __name__=='__main__':
    run()
