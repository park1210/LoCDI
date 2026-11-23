"""User-only one-shot existing Core fit AFTER independent and legacy Gates.

Default checks only. No LLM calls, new labels, implicit fitting or gate bypass.
"""
from datetime import datetime,timezone
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha
from scripts.run_common_menu_b0_b4 import load_plan
from app.locdi.llm_setup import digest,validate_setup
from validation.routing_correctness_gate import to_policy_fit_rows
from app.core.routing_utility import fit_utility
from app.core.agent_registry import AgentRegistry
from app.locdi.common_menu_backend import routing_admission
import json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan-id',default='common_all_001')
    p.add_argument('--reviewed-input',type=Path,required=True);p.add_argument('--authorization',type=Path)
    p.add_argument('--save-approval-template',type=Path)
    p.add_argument('--fit',action='store_true');p.add_argument('--allow-fit',action='store_true');a=p.parse_args()
    pub,pr,plan=load_plan(a.plan_id);review=read(a.reviewed_input);config=read(pr/'llm_config.json')
    training=pr/'routing_training';blank=read(training/'review_input_blank.json')
    for key in ('registry','split_manifest','decisions','outcomes'):
        if review['training_data'][key]!=blank['training_data'][key]:raise ValueError('COLLECTED_DATA_MUST_NOT_BE_REWRITTEN_BY_REVIEW')
    for key in ('decision_contexts','holdout_patient_ids','dataset_fingerprint'):
        if review['validator_options'][key]!=blank['validator_options'][key]:raise ValueError('ORIGINAL_REVIEW_SOURCE_CONTEXT_CHANGED')
    setup=validate_setup(config)
    if setup['static_readiness']!='READY':raise ValueError('CURRENT_MODEL_PROMPT_REGISTRY_NOT_READY')
    options=review['validator_options']
    if options.get('allow_test_only') is True or options.get('registry_manifest')!=setup['prepared']['registry']:
        raise ValueError('REAL_CURRENT_REGISTRY_INDEPENDENT_REVIEW_REQUIRED')
    if (review.get('model_identifier')!=config['model_identifier']
        or review.get('prompt_fingerprints')!=setup['prepared']['prompt_fingerprints']):raise ValueError('REVIEW_BINDING_MISMATCH')
    converted=to_policy_fit_rows(review['training_data'],**options)
    if a.save_approval_template:
        if a.fit or a.allow_fit:raise ValueError('TEMPLATE_CREATION_MUST_NOT_FIT')
        if converted['report']['utility_fit_ready'] is not True:raise ValueError('CANNOT_PREPARE_FIT_APPROVAL_WHILE_GATE_BLOCKED')
        write(a.save_approval_template,{'scope':'REVIEWED_CURRENT_3B_ROUTING_FIT_ONCE','plan_fingerprint':plan['fingerprint'],
            'reviewed_input_fingerprint':digest(review),'max_fitting':1,'operator_approved':False,
            'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None})
    if not a.fit:
        print(json.dumps({'fit_called':False,'gate':converted['report']},ensure_ascii=True));return
    if not(a.allow_fit and a.authorization):p.error('NEW_EXPLICIT_FIT_APPROVAL_REQUIRED')
    if converted['rows'] is None or converted['report']['utility_fit_ready'] is not True:raise ValueError('EXISTING_GATES_BLOCKED')
    approval=read(a.authorization)
    if (approval.get('scope')!='REVIEWED_CURRENT_3B_ROUTING_FIT_ONCE' or approval.get('plan_fingerprint')!=plan['fingerprint']
        or approval.get('reviewed_input_fingerprint')!=digest(review) or approval.get('operator_approved') is not True
        or approval.get('max_fitting')!=1 or not approval.get('operator_ref') or not approval.get('user_statement')):
        raise ValueError('PINNED_ONE_FIT_APPROVAL_REQUIRED')
    issued=datetime.fromisoformat(approval['recorded_at']);expires=datetime.fromisoformat(approval['expires_at']);now=datetime.now(timezone.utc)
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires or not 0<(expires-issued).total_seconds()<=86400:
        raise ValueError('FIT_APPROVAL_TIME_INVALID')
    folder=pr/'reviewed_fit_001';folder.mkdir(exist_ok=False)
    write(folder/'approval.json',approval)
    # Actual fitting call: user-only, once, unchanged accuracy-only formula.
    artifact=fit_utility(converted['rows'],AgentRegistry(setup['prepared']['registry']),min_samples=2)
    write(folder/'routing_utility.json',artifact)
    evidence={**review,'routing_artifact':artifact,
        'origin_files':{str(path.resolve()):sha(path) for path in (
            training/'plan.json',training/'collection_001/collection.json',training/'review_input_blank.json',a.reviewed_input)}}
    bindings=read(pr/'bindings.json')
    gate=routing_admission(evidence,setup['prepared']['registry'],[b['patient_id'] for b in bindings.values()],config)
    write(folder/'routing_evidence.json',evidence)
    write(pub/'routing_fit_status.json',{'fit_calls':1,'independent_contract_ready':gate['correctness']['utility_fit_ready'],
        'clinical_research_admission':False,'research_gate_authentication':'NOT_INSTALLED',
        'artifact_fingerprint':digest(artifact)})


if __name__=='__main__':main()
