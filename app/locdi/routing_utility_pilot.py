"""Bounded, explicit-opt-in one-Decision Specialty collection. No fitting/judging.

Reuses existing preparation and real LocalOllamaAdapter. Raw responses and
BEFORE/AFTER checkpoints are private; no retries, repairs, resume or overwrite.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_collector import validate_preparation, atomic_json
from app.locdi.ollama_adapter import LocalOllamaAdapter
from app.llm.prompt_loader import specialty_system_prompt


def plan_pilot(bundle, max_requests=5):
    if type(max_requests)is not int or not 1<=max_requests<=5:raise ValueError('PILOT_GLOBAL_LIMIT_1_TO_5')
    agents=validate_preparation(bundle)
    candidate=next(c for c in bundle['candidates'] if c['decision']['split']=='train')
    cfg=bundle['llm_config']
    if any(cfg.get(k)!=v for k,v in {'max_output_tokens':1024,'timeout_seconds':120,
        'request_retries':0,'temperature':0,'seed':42}.items()):raise ValueError('APPROVED_GENERATION_SCOPE_MISMATCH')
    plan={'version':'locdi.routing_utility_pilot.c4_11.v1','purpose':'SOURCE_PROXY_DEVELOPMENT_COLLECTION_ONLY',
        'selection':'FIRST_TRAIN_DECISION_IN_EXISTING_PREPARATION_ORDER','case_ref':candidate['case_ref'],
        'split':'train','planned_agents':[a['agent_id'] for a in agents],
        'preparation_fingerprint':bundle['fingerprint'],'registry_fingerprint':digest(bundle['registry']),
        'shared_input_fingerprint':candidate['shared_input_fingerprint'],
        'model':cfg['model'],'model_identifier':cfg['model_identifier'],'model_version':cfg['model_version'],
        'generation':{k:cfg[k] for k in ('temperature','seed','max_output_tokens','timeout_seconds')},
        'max_requests':max_requests,'retry':0,'repair':0,'server_change':False,
        'assumptions':bundle['assumptions'],'clinical_metrics':'NOT_EVALUATED','fit_permission':False}
    return plan,candidate,agents


def locations(namespace,run_id):
    for value in (namespace,run_id):
        if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',value):raise ValueError('INVALID_NAMESPACE_OR_RUN_ID')
    pub=ROOT/'outputs/research_evaluation'/namespace/'live_pilot'/run_id
    private=ROOT/'outputs/private/research_evaluation'/namespace/'live_pilot'/run_id
    for path in (pub,private):
        if not path.resolve().is_relative_to((ROOT/'outputs').resolve()):raise ValueError('OUTPUT_PATH_ESCAPE')
        if path.exists():raise ValueError('RUN_ID_OR_OUTPUT_ALREADY_USED')
    return pub,private


def collect_pilot(bundle, *, namespace='c4_11_v1',run_id='c4_11_collect_001',execute=False,
                  allow_llm=False,accept_proxy=False,max_requests=5):
    if any(type(x)is not bool for x in (execute,allow_llm,accept_proxy)):raise ValueError('EXPLICIT_BOOLEAN_APPROVAL_REQUIRED')
    if execute and not(allow_llm and accept_proxy):raise ValueError('EXPLICIT_GENERATION_AND_PROXY_ACCEPTANCE_REQUIRED')
    public,private=locations(namespace,run_id)
    plan,candidate,agents=plan_pilot(bundle,max_requests)
    if not execute:return {'status':'DRY_RUN','plan':plan,'generation_requests':0,'client_constructed':False}
    private.mkdir(parents=True);public.mkdir(parents=True)
    approval={'approval_type':'OPERATOR_EXPLICIT_CHAT_APPROVAL','source':'USER_REPORTED_OPERATOR_STATEMENT',
        'evidence':'Proxy를 수용하고 제한된 5회 수집 허용','recorded_at':datetime.now(timezone.utc).isoformat(),
        'scope_plan_fingerprint':digest(plan),'clinical_reference_approved':False,'utility_fit_approved':False}
    atomic_json(private/'approval.json',approval);atomic_json(private/'plan.json',plan)
    atomic_json(public/'plan.json',plan)
    cfg=deepcopy(bundle['llm_config']);cfg['max_requests']=max_requests
    client=None;records=[];attempts=0;stop=None
    try:
        client=LocalOllamaAdapter(cfg,authorized=True)
        for i,agent in enumerate(agents,1):
            if attempts>=max_requests:stop='GLOBAL_REQUEST_LIMIT';break
            if digest(candidate['shared_payload'])!=candidate['shared_input_fingerprint']:raise ValueError('SNAPSHOT_CHANGED')
            payload={'department':agent['department'],**deepcopy(candidate['shared_payload'])}
            prompt=specialty_system_prompt(agent['department'])
            # Actual project prompt fingerprint is SHA256 UTF-8, not JSON digest.
            import hashlib
            if hashlib.sha256(prompt.encode('utf8')).hexdigest()!=candidate['decision']['generation']['prompt_fingerprints'][agent['agent_id']]:
                raise ValueError('PROMPT_FINGERPRINT_CHANGED')
            attempts+=1;ref='Request%03d'%attempts
            before={'run_id':run_id,'request_ref':ref,'case_ref':candidate['case_ref'],'agent_id':agent['agent_id'],
                'model':agent['model'],'version':agent['version'],'split':'train',
                'shared_input_fingerprint':candidate['shared_input_fingerprint'],'full_payload_fingerprint':digest(payload),
                'prompt_fingerprint':hashlib.sha256(prompt.encode('utf8')).hexdigest(),
                'generation':deepcopy(plan['generation']),'generation_count_before':client.request_count,
                'request_started_at':datetime.now(timezone.utc).isoformat(),
                'patient_id':candidate['decision']['patient_id'],'decision_id':candidate['decision']['decision_id'],
                'payload':payload,'system_prompt':prompt}
            atomic_json(private/(ref+'_before.json'),before)
            after={k:v for k,v in before.items() if k not in ('payload','system_prompt')}
            after.update(response_status='REQUEST_FAILED',response=None,raw_response=None,correct=None,
                         correctness_status='NOT_REVIEWED',error_type=None)
            try:
                raw=client.chat_json(prompt,json.dumps(payload,ensure_ascii=False),model=agent['model'],schema=bundle['response_schema'])
                response=json.loads(raw)
                after.update(response=response,response_fingerprint=digest(response),
                    response_status='UNRESOLVED' if response['certainty']=='unresolved' or response['diagnosis']=='Unresolved' else 'SUPPORTED')
            except Exception as exc:
                after['error_type']=type(exc).__name__;stop='REQUEST_OR_RESPONSE_ERROR'
            after.update(raw_response=client.last_response_content,
                generation_count_after=client.request_count,identity_checks_after=client.identity_check_count,
                observed_model_identifier=client.last_observed_model_identifier,
                recorded_at=datetime.now(timezone.utc).isoformat())
            if after['raw_response'] is not None:after['raw_response_sha256']=hashlib.sha256(after['raw_response'].encode('utf8')).hexdigest()
            atomic_json(private/(ref+'_after.json'),after);records.append(after)
            if stop:break
        counts=client.public_metadata()
        summary={'run_id':run_id,'plan_fingerprint':digest(plan),'case_ref':candidate['case_ref'],
            'status':'PILOT_COLLECTION_COMPLETE_NOT_CLINICALLY_REVIEWED' if len(records)==5 and stop is None else 'PILOT_PARTIAL_OR_FAILED',
            'attempts':attempts,'generation_requests':counts['request_count'],'identity_checks':counts['identity_check_count'],
            'supported':sum(r['response_status']=='SUPPORTED' for r in records),'unresolved':sum(r['response_status']=='UNRESOLVED' for r in records),
            'errors':sum(r['error_type'] is not None for r in records),'stop_reason':stop,
            'stored_responses':len(records),  # Legacy name: persisted outcome/attempt records, including failures.
            'stored_attempt_records':len(records),
            'received_model_responses':sum(r['raw_response'] is not None for r in records),
            'schema_valid_responses':sum(r['response'] is not None for r in records),
            'missing_agents':len(agents)-len(records),
            'clinical_metrics':'NOT_EVALUATED','correctness_nonnull':0,'utility_fit_count':0,
            'training_readiness':'BLOCKED','merged_with_baseline':False,'retry':0,'repair':0,
            'budget_includes_all_generations':True,'server_context':'NOT_CHECKED_BY_THIS_COLLECTOR'}
        atomic_json(public/'summary.json',summary);atomic_json(private/'summary.json',summary)
        return summary
    except Exception as exc:
        atomic_json(private/'failed.json',{'status':'FAILED','attempts':attempts,
            'generation_requests':client.request_count if client is not None else 0,'persisted_after_checkpoints':len(records),'error_type':type(exc).__name__})
        raise
    finally:
        if client is not None:client.close()
