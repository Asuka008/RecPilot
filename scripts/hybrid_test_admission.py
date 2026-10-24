"""Read-only admission for the validation-selected hybrid test recipe."""
import json
import math
from diagnostics.tail_admission import admit, verify_unchanged
from scripts.compare_title_validation import best_validation
from scripts.evaluate_selected_title import file_digest


def selected_alpha(report):
    """Recompute the declared finite-grid winner without reading test metrics."""
    if report['status'] != 'PASS' or report['split'] != 'val' or report['alphas'] != [0,.1,.25,.5]:
        raise ValueError('Expected the completed preregistered validation grid')
    variants=report['variants']
    if set(variants) != {'0','0.1','0.25','0.5'}:
        raise ValueError('Validation variants differ')
    for variant in variants.values():
        if variant['split']!='val': raise ValueError('Only validation may select alpha')
        for axis in ('groups_by_target_train_frequency','groups_by_history_length'):
            if variant[axis].keys() != variants['0'][axis].keys():
                raise ValueError('Group keys differ')
            for name, group in variant[axis].items():
                if group['users'] != variants['0'][axis][name]['users']:
                    raise ValueError('User groups differ')
                if any(not math.isfinite(v) for v in group['metrics'].values()):
                    raise ValueError('Nonfinite validation metric')
    base=variants['0']['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']
    eligible=[a for a in report['alphas'] if variants[str(a)]['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']>=base and variants[str(a)]['groups_by_target_train_frequency']['1-5']['metrics']['Recall@20']>0]
    winner=max(eligible,key=lambda a:variants[str(a)]['groups_by_target_train_frequency']['all']['metrics']['NDCG@10']) if eligible else None
    if report['selection']['alpha']!=winner or report['selection']['status']!='candidate_for_review' or winner is None:
        raise ValueError('Validation selection differs')
    return winner


def admit_test(root, recipe):
    """Bind current source/data/checkpoints to the successful validation run."""
    if set(recipe)!={'validation_report','audit_config','alpha'} or recipe['alpha']!=.25:
        raise ValueError('Only the frozen alpha=.25 recipe is admitted')
    path=root/recipe['validation_report']
    report=json.loads(path.read_text())
    if selected_alpha(report)!=recipe['alpha']:
        raise ValueError('Frozen alpha differs from validation winner')
    tracked=dict(report['provenance']['inputs_sha256'])
    verify_unchanged(tracked)
    audit_path=(root/recipe['audit_config']).resolve()
    if str(audit_path) not in tracked:
        raise ValueError('Audit config not authenticated by validation')
    reports,baseline,checkpoints,current=admit(root,json.loads(audit_path.read_text()))
    for item,sha in current.items():
        if tracked.get(item)!=sha:
            raise ValueError('Current artifact was not authenticated by validation')
    if report['provenance']['training_config']!=reports['id']['config']:
        raise ValueError('Training configuration differs')
    from diagnostics.tail_statistics import verify_diagnostic_anchor
    verify_diagnostic_anchor('id',report['variants']['0'],best_validation(reports['id']),1e-10)
    tracked[str(path.resolve())]=file_digest(path)
    return reports,baseline,checkpoints,tracked,report['provenance']['data_identity']
