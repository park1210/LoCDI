"""C4-35 two explicit approvals. Default is dry-run; no generation/fitting."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.review_new_cohort_independence import PUB,PR,P33,U33,verify
from app.locdi.llm_setup import digest
from app.locdi.voi_fresh_cohort_admission import approve_admission,run_approved


def admit(approval_file,execute=False):
    verify()
    final=PUB/'proposed_admission_contract_final.json'
    contract=read(final if final.exists() else PUB/'proposed_admission_contract.json')
    if not execute:
        print(json.dumps({'status':'DRY_RUN_ADMISSION_NOT_APPROVED','contract_fingerprint':contract['fingerprint'],
            'proposed_patients':contract['proposed_patients'],'proposed_visits':contract['proposed_visits'],'actual_admission':0}));return
    record=read(Path(approval_file)) if approval_file else None
    proposal=read(PR/'proposed_panel_review.json')
    from scripts.review_new_cohort_independence import P34
    if check(read(P34/'actual_generation_receipt.json')['raw_files'])['changed_or_missing']:
        raise ValueError('RAW_INTEGRITY_BEFORE_ADMISSION')
    if check(read(PR/'comparison_source_hashes.json'))['changed_or_missing']:
        raise ValueError('COMPARISON_SOURCE_CHANGED')
    if contract.get('review_evidence_hashes') and check(contract['review_evidence_hashes'])['changed_or_missing']:
        raise ValueError('REVIEW_EVIDENCE_CHANGED')
    if record and record.get('incomplete_legacy_source_review_accepted') is not True:
        raise ValueError('LEGACY_MISSING_SOURCE_SCOPE_REVIEW_REQUIRED')
    manifest=approve_admission(proposal,contract,record)
    write(PR/'admission_approval.json',record);write(PR/'approved_admission.json',manifest)
    write(PUB/'actual_admission_manifest.json',{k:v for k,v in manifest.items() if k!='panel'})
    spec=read(PUB/'evaluation_protocol.json');spec.pop('fingerprint')
    spec['admission_panel_fingerprint']=manifest['panel_fingerprint'];spec['status']='AWAITING_SEPARATE_EVALUATION_APPROVAL'
    spec['model_sha256']=sha(P33/'final_m5_model.json');spec['admission_manifest_sha256']=sha(PR/'approved_admission.json')
    spec['evaluation_slots']=manifest['visits']*9*6;spec['registered_replay_paths']=manifest['visits']*40*6
    spec['fingerprint']=digest(spec);write(PUB/'approved_panel_evaluation_protocol.json',spec)
    print(json.dumps({'status':manifest['status'],'patients':manifest['patients'],'visits':manifest['visits'],
        'evaluation_protocol_fingerprint':spec['fingerprint'],'policy_evaluation':0}))


def evaluate(approval_file,execute=False):
    verify()
    if not execute:
        print(json.dumps({'status':'DRY_RUN','generation':0,'admission_applied':False,'policy_evaluation':0,
            'requires':'Approval A for revised admission, then Approval B for exact admitted panel evaluation'}));return
    manifest=read(PR/'approved_admission.json');spec=read(PUB/'approved_panel_evaluation_protocol.json')
    if spec['fingerprint']!=digest({k:v for k,v in spec.items() if k!='fingerprint'}):raise ValueError('EVALUATION_PROTOCOL_HASH')
    if sha(PR/'approved_admission.json')!=spec['admission_manifest_sha256'] or sha(P33/'final_m5_model.json')!=spec['model_sha256']:
        raise ValueError('ADMISSION_OR_MODEL_CHANGED')
    record=read(Path(approval_file)) if approval_file else None
    from app.locdi.new_cohort_execution import approval
    approval(record,'C4_35_FROZEN_POLICY_EVALUATION',spec['fingerprint'])
    write(PR/'evaluation_approval.json',record);write(PR/'evaluation_started.json',{'at':datetime.now(timezone.utc).isoformat(),'protocol':spec['fingerprint']})
    try:
        with (PR/'fresh_episode_trace.jsonl').open('x',encoding='utf-8') as f:
            def sink(row):f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            records,membership,resources,results=run_approved(manifest,read(P33/'final_m5_model.json'),spec,record,sink)
        write(PR/'fresh_score_rows.json',records);write(PR/'fresh_membership.json',membership)
        write(PUB/'fresh_cohort_native_policy_metrics.json',results)
        write(PUB/'fresh_cohort_resource_usage.json',resources)
        print(json.dumps({'status':'EVALUATION_COMPLETED','registered_slots':len(records)}))
    except Exception as e:
        write(PUB/'fresh_execution_partial.json',{'status':'EXECUTION_PARTIAL','error_type':type(e).__name__,
            'partial_trace_preserved':True,'no_failed_slots_removed':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['admit','evaluate'],nargs='?',default='evaluate')
    p.add_argument('--execute',action='store_true');p.add_argument('--approval-file');a=p.parse_args()
    admit(a.approval_file,a.execute) if a.stage=='admit' else evaluate(a.approval_file,a.execute)
