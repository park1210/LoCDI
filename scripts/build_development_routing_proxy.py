"""User-only Source-label proxy derivation/fitting. Default inspect, no fitting.

Separate from clinical Gate/reviewed_fit_001. No model calls in this script.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha
from scripts.run_common_menu_b0_b4 import load_plan
from scripts.collect_common_routing_training import training_paths
from app.locdi.development_routing_proxy import MARKERS,derive_rows,admit_development_proxy
from app.locdi.llm_setup import digest,validate_setup
from app.core.routing_utility import fit_utility
from app.core.agent_registry import AgentRegistry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-id',default='dev_source_001')
    p.add_argument('--prepare-approval',action='store_true')
    p.add_argument('--prepare-unfitted',action='store_true',help='Non-generating abstention-only contract; no fitting or collection claim')
    p.add_argument('--build',action='store_true');p.add_argument('--allow-fit',action='store_true')
    p.add_argument('--authorization',type=Path);a=p.parse_args()
    if a.prepare_approval and (a.build or a.allow_fit or a.authorization):p.error('APPROVAL_PREPARATION_ONLY')
    if a.prepare_unfitted and (a.prepare_approval or a.build or a.allow_fit or a.authorization):p.error('UNFITTED_PREPARATION_ONLY')
    if not a.build and (a.allow_fit or a.authorization):p.error('FITTING_FLAGS_REQUIRE_BUILD')
    pub,pr,ep=load_plan(a.plan_id)
    if ep.get('experiment_kind')!='DEVELOPMENT_EXPERIMENT':raise ValueError('PROXY_IS_DEVELOPMENT_ONLY')
    tp,tpr=training_paths(a.plan_id);cp=tpr/'plan.json';coll=tpr/'collection_001/collection.json'
    plan=read(cp);config=read(pr/'llm_config.json');registry=validate_setup(config)['prepared']['registry']
    patients=[b['patient_id'] for b in read(pr/'bindings.json').values()]
    if a.prepare_unfitted:
        if coll.exists():raise ValueError('ACTUAL_COLLECTION_EXISTS_DO_NOT_IGNORE_IT')
        derived=derive_rows(plan,{'records':[]},tpr/'collection_001/requests',registry,config,patients)
        evidence={'schema_version':'locdi.routing.unfitted_source_proxy.v1',**MARKERS,
            'collection_plan':str(cp.resolve()),'collection_file':None,'collection_status':'NOT_COLLECTED',
            'request_root':str((tpr/'collection_001/requests').resolve()),
            'origin_files':{str(cp.resolve()):sha(cp)},'routing_artifact':None,
            'fit_rows_fingerprint':digest(derived['rows']),'complete_train_groups':0,
            'utility_generated':False,'purpose':'EXPLICIT_UNFITTED_CORE_ABSTENTION_ONLY'}
        admit_development_proxy(evidence,registry,patients,config)
        write(pr/'development_unfitted_utility_001/development_routing_evidence.json',evidence)
        write(pub/'development_unfitted_status.json',{**MARKERS,'collection_status':'NOT_COLLECTED',
            'fit_calls':0,'utility_generated':False,'routing_behavior':'UNCHANGED_CORE_ABSTENTION'})
        print(json.dumps({'status':'UNFITTED_ABSTENTION_CONTRACT_ONLY','generation':0,'fit_calls':0,'utility_generated':False}));return
    collection=read(coll)
    derived=derive_rows(plan,collection,tpr/'collection_001/requests',registry,config,patients)
    approval_fp=digest({'training_plan':plan['fingerprint'],'collection':sha(coll),'rows':digest(derived['rows'])})
    status={**MARKERS,'complete_train_proxy_groups':derived['complete_train_groups'],
        'group_status':[{'ordinal':i,'split':g['split'],'complete_proxy_group':g['complete_proxy_group'],
            'statuses':[r['status'] for r in g['outcomes']]} for i,g in enumerate(derived['groups'],1)],
        'fit_calls':0,'fewer_than_two_groups_implies_insufficient_support':derived['complete_train_groups']<2}
    if a.prepare_approval:
        write(tpr/'proxy_fit_approval_template.json',{'scope':'DEVELOPMENT_SOURCE_LABEL_PROXY_BUILD_ONCE',
            'plan_fingerprint':ep['fingerprint'],'proxy_input_fingerprint':approval_fp,'max_fitting':1,
            'operator_approved':False,'accept_source_label_proxy':False,'operator_ref':None,'user_statement':None,
            'recorded_at':None,'expires_at':None,**MARKERS})
    if not a.build:
        print(json.dumps(status,ensure_ascii=True));return
    if not(a.allow_fit and a.authorization):p.error('EXPLICIT_USER_BUILD_ALLOW_FIT_APPROVAL_REQUIRED')
    approval=read(a.authorization)
    if (approval.get('scope')!='DEVELOPMENT_SOURCE_LABEL_PROXY_BUILD_ONCE'
        or approval.get('plan_fingerprint')!=ep['fingerprint'] or approval.get('proxy_input_fingerprint')!=approval_fp
        or approval.get('operator_approved') is not True or approval.get('accept_source_label_proxy') is not True
        or approval.get('max_fitting')!=1 or not approval.get('operator_ref') or not approval.get('user_statement')
        or any(approval.get(k)!=v for k,v in MARKERS.items())):raise ValueError('ACTUAL_PROXY_FIT_APPROVAL_REQUIRED')
    issued=datetime.fromisoformat(approval['recorded_at']);expires=datetime.fromisoformat(approval['expires_at']);now=datetime.now(timezone.utc)
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires or not 0<(expires-issued).total_seconds()<=86400:
        raise ValueError('PROXY_APPROVAL_EXPIRED')
    folder=pr/'development_proxy_utility_001';folder.mkdir(exist_ok=False)
    write(folder/'operator_approval.json',approval)
    artifact=None
    if derived['rows']:
        artifact=fit_utility(derived['rows'],AgentRegistry(registry),min_samples=2)
        write(folder/'routing_utility_source_label_proxy.json',{'artifact':artifact,**MARKERS})
        status['fit_calls']=1
    origin={str(cp.resolve()):sha(cp),str(coll.resolve()):sha(coll),**derived['checkpoint_files']}
    evidence={'schema_version':'locdi.routing.source_label_proxy.v1',**MARKERS,
        'collection_plan':str(cp.resolve()),'collection_file':str(coll.resolve()),
        'request_root':str((tpr/'collection_001/requests').resolve()),'origin_files':origin,
        'routing_artifact':artifact,'fit_rows_fingerprint':digest(derived['rows']),
        'complete_train_groups':derived['complete_train_groups'],
        'evaluation_patients_excluded_fingerprint':digest(sorted(patients)),
        'purpose':'SOURCE_DIAGNOSIS_EXACT_AGREEMENT_PROXY_NOT_CLINICAL_CORRECTNESS'}
    # Re-derive provenance and original Core constructor; no scores are invented.
    admission=admit_development_proxy(evidence,registry,patients,config)
    write(folder/'proxy_assessment_private.json',derived)
    write(folder/'development_routing_evidence.json',evidence)
    write(pub/'development_proxy_utility_status.json',{**status,'utility_status':admission['utility_status'],
        'clinical_gate':'UNCHANGED_NOT_ADMITTED','fitting_method':'UNCHANGED_CORE_JACCARD_BETA_MIN_SAMPLES_2'})
    print(json.dumps({**status,'utility_status':admission['utility_status']},ensure_ascii=True))


if __name__=='__main__':main()
