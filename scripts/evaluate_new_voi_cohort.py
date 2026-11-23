"""C4-34 fresh single-M5 batch CLI. Default dry-run, no generation or fitting."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_new_voi_cohort import PUB,PR,RAW,OLD,OLDPUB,verify_plan
from app.locdi.llm_setup import digest
from app.locdi.new_cohort_execution import approval,verify_receipt,admit,execute_panel,summarize
from app.locdi.voi_new_cohort_evaluation import POLICIES


def frozen_model():
    m=read(OLDPUB/'final_m5_artifact_manifest.json')
    if sha(OLD/'final_m5_model.json')!=m['artifact_sha256'] or sha(OLD/'final_action_selection.json')!=m['selection_sha256']:
        raise ValueError('FINAL_MODEL_OR_ACTION_SELECTION_CHANGED')
    model=read(OLD/'final_m5_model.json')
    if len(model['diagnoses'])!=15 or len(model['actions'])!=8:
        raise ValueError('M5_DIAGNOSIS15_ACTION8')
    original=read(OLDPUB/'new_cohort_evaluation_protocol.json')
    from app.locdi.dynamic_horizon import policy_spec
    if model['provenance']['smoothing_alpha']!=1 or original['dynamic_spec']!=policy_spec():
        raise ValueError('ALPHA_OR_DYNAMIC_CHANGED')
    return model,m


def receipt_bundle(plan):
    r=read(PR/'actual_generation_receipt.json')
    verify_receipt(r,plan,check)
    ar=read(PR/'generation_approval.json')
    approval(ar,'C4_34_GENERATION_ONCE',plan['fingerprint'])
    if sha(PR/'generation_approval.json')!=r['approval_sha256']:
        raise ValueError('GENERATION_APPROVAL_HASH')
    started=read(PR/'generation_started.json')
    if started['plan_fingerprint']!=plan['fingerprint'] or started['approval_sha256']!=r['approval_sha256']:
        raise ValueError('STARTED_CHECKPOINT_BINDING')
    final=read(PR/'generation_container_final.json')
    if final['returncode'] or sha(PR/'generation_container_final.json')!=r['container_final_inspection_sha256']:
        raise ValueError('CONTAINER_INSPECTION_RECEIPT')
    container=json.loads(final['stdout'])[0]
    if container['Image']!=plan['image_id'] or container['State']['ExitCode']!=0 or container['State'].get('OOMKilled'):
        raise ValueError('GENERATOR_CONTAINER_NOT_SUCCESSFUL')
    actual={str(f):sha(f) for f in RAW.rglob('*') if f.is_file()}
    if actual!=r['raw_files']:
        raise ValueError('EXTRA_OR_MISSING_RAW_SOURCE')
    if not list((RAW/'json').glob('*.json')):
        raise ValueError('NATIVE_JSON_EXPORT_MISSING')
    metas=[read(f) for f in (RAW/'metadata').glob('*.json')]
    if len(metas)!=1 or metas[0].get('seed')!=2026111601 or metas[0].get('clinicianSeed')!=2026111601:
        raise ValueError('GENERATED_METADATA_SEED')
    if metas[0].get('referenceTime')!='20261116' or metas[0].get('endTime')!='20261116':
        raise ValueError('GENERATED_REFERENCE_DATE')
    return r,metas[0]


def inspect_source():
    plan=verify_plan();model,am=frozen_model();receipt,metadata=receipt_bundle(plan)
    visits,ledger,summary=admit(RAW,model,read(OLD/'exclusion_inventory.json'))
    if metadata['patientCount']!=summary['source']['generated_patients']:
        raise ValueError('METADATA_CSV_PATIENT_COUNT')
    if check(receipt['raw_files'])['changed_or_missing']:raise ValueError('SOURCE_CHANGED_DURING_INSPECTION')
    write(PR/'admitted_visits.json',visits);write(PR/'source_eligibility_ledger.json',ledger)
    admission={'status':summary['status'],'receipt_sha256':sha(PR/'actual_generation_receipt.json'),
        'plan_fingerprint':plan['fingerprint'],'model_sha256':am['artifact_sha256'],
        'selection_sha256':am['selection_sha256'],'visits_sha256':sha(PR/'admitted_visits.json'),
        'ledger_sha256':sha(PR/'source_eligibility_ledger.json'),'summary':summary,
        'policies':POLICIES,'budgets':list(range(6)),'random_repeats':32,'bootstrap_repetitions':2000,
        'bootstrap_seed':42,'execution_budget':{'calls':20000,'states':50000000,'wall_seconds':3600},
        'code_hashes':plan['code_hashes']}
    admission['fingerprint']=digest(admission);write(PUB/'source_inspection_manifest.json',admission)
    write(PUB/'actual_cohort_source_eligibility.json',summary)
    print(json.dumps({'status':summary['status'],'complete_patients':summary['complete_patients'],
        'complete_visits':summary['complete_visits'],'admission_fingerprint':admission['fingerprint'],'evaluation':0}))


def evaluate(execute,approval_file):
    plan=verify_plan();model,am=frozen_model()
    if not execute:
        source=PUB/'source_inspection_manifest.json'
        print(json.dumps({'status':'DRY_RUN','generation':0,'evaluation':0,
            'source_manifest_present':source.exists(),'policies':POLICIES,'budgets':list(range(6)),
            'requires':'--execute --approval-file with exact admission fingerprint; separate generation receipt'}));return
    receipt_bundle(plan)
    a=read(PUB/'source_inspection_manifest.json')
    if a['fingerprint']!=digest({k:v for k,v in a.items() if k!='fingerprint'}) or a['status']!='TECHNICAL_PANEL_AVAILABLE':
        raise ValueError('SOURCE_ADMISSION_UNAVAILABLE')
    if a['plan_fingerprint']!=plan['fingerprint'] or a['model_sha256']!=am['artifact_sha256'] or a['selection_sha256']!=am['selection_sha256']:
        raise ValueError('SOURCE_MODEL_PLAN_BINDING')
    if a['code_hashes']!=plan['code_hashes'] or check(a['code_hashes'])['changed_or_missing']:
        raise ValueError('SOURCE_ADMISSION_CODE_HASH')
    if sha(PR/'actual_generation_receipt.json')!=a['receipt_sha256'] or sha(PR/'admitted_visits.json')!=a['visits_sha256'] or sha(PR/'source_eligibility_ledger.json')!=a['ledger_sha256']:
        raise ValueError('SOURCE_INSPECTION_CHANGED')
    approval_record=read(Path(approval_file)) if approval_file else None
    approval(approval_record,'C4_34_FROZEN_POLICY_EVALUATION',a['fingerprint'])
    # Recompute identity and every value from actual raw, not merely trust admission.
    actual,ledger,summary=admit(RAW,model,read(OLD/'exclusion_inventory.json'))
    if actual!=read(PR/'admitted_visits.json') or ledger!=read(PR/'source_eligibility_ledger.json'):
        raise ValueError('SOURCE_ADAPTER_REPLAY_DIFFERENCE')
    if (PR/'evaluation_started.json').exists():raise ValueError('NO_EVALUATION_RESUME')
    write(PR/'evaluation_approval.json',approval_record)
    write(PR/'evaluation_started.json',{'at':datetime.now(timezone.utc).isoformat(),'admission_fingerprint':a['fingerprint'],
        'approval_sha256':sha(PR/'evaluation_approval.json')})
    try:
        with (PR/'episode_trace.jsonl').open('x',encoding='utf-8') as f:
            def sink(r):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            with patch.object(socket.socket,'connect',side_effect=RuntimeError('NETWORK_DISABLED')):
                records,membership,resources=execute_panel(model,actual,sink)
        write(PR/'score_rows.json',records);write(PR/'identity_membership.json',membership)
        results=summarize(records,membership)
        write(PUB/'actual_new_cohort_policy_results.json',{**results,'planner_resources':resources,
            'status':'FRESH_SYNTHETIC_EVALUATION_COMPLETE','performance_level':'synthetic source agreement, not clinical validation'})
        write(PUB/'actual_patient_paired_comparison.json',results['comparison'])
        index={str(f.relative_to(ROOT)):sha(f) for f in PR.rglob('*') if f.is_file()}
        write(PR/'evaluation_private_index.json',{'files':index,'fingerprint':digest(index)})
        pins={str(f.relative_to(ROOT)):sha(f) for f in [PUB/'actual_new_cohort_policy_results.json',PUB/'actual_patient_paired_comparison.json',PR/'evaluation_private_index.json']}
        write(PUB/'evaluation_completion_manifest.json',{'files':pins,'fingerprint':digest(pins),
            'admission_fingerprint':a['fingerprint'],'status':'FRESH_SYNTHETIC_EVALUATION_COMPLETE'})
        print(json.dumps({'status':'FRESH_SYNTHETIC_EVALUATION_COMPLETE','patients':summary['complete_patients'],'decision':results['decision']}))
    except Exception as e:
        write(PUB/'actual_evaluation_failure.json',{'status':'FAILED_PARTIAL_TRACE_PRESERVED','error_type':type(e).__name__,
            'no_failed_patients_removed':True,'no_scores_synthesized':True})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['inspect-source','evaluate'],nargs='?',default='evaluate')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approval-file');args=parser.parse_args()
    if args.stage=='inspect-source':inspect_source()
    else:evaluate(args.execute,args.approval_file)
