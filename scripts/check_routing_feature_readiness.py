"""Read-only functional feature check; does not change any fitting/clinical Gate."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_validation_role_snapshot import PUB
from app.locdi.original_source_inventory import read


def check():
    report=read(PUB/'routing_root_summary.json');train=next(r for r in report['domain_stats'] if r['view']=='U2_Native_Train8')
    blocks=[]
    if train['empty_token_patients']==train['patients']:blocks.append('ALL_NATIVE_TRAIN_OBSERVATION_TOKEN_SETS_EMPTY')
    if all(r['native_U2_feature_positive_jaccard_groups']==0 for r in report['routing_support']):blocks.append('ZERO_POSITIVE_JACCARD_SUPPORT_ON_INITIAL_AND_LATER_EVALUATION_INPUTS')
    return {'status':'BLOCKED' if blocks else 'FEATURE_CHECK_ONLY_NOT_FIT_OR_ADMISSION_APPROVAL',
        'blocker_codes':blocks,'complete_train_groups':8,'train_rows':40,'positive_outcomes':report['actual_train_positive_rows'],
        'empty_feature_groups':train['empty_token_patients'],'existing_fit_contract_changed':False,
        'recommendation':'DO_NOT_FIT_OR_REFIT_CURRENT_EMPTY_FEATURE_ROWS','LLM_generation':0,'fit_calls':0}


if __name__=='__main__':
    out=check();print(json.dumps(out))
    if out['status']=='BLOCKED':raise SystemExit(2)
