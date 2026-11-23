"""Read-only old-run audit and separate fresh collection. Default never generates.

No resume, response mixing, or edits to the old collector/approval/receipts.
Only explicit user execute+allow-llm with a fresh approval can call a model.
"""
import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_common_menu_b0_b4 import load_plan,M5
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collector import load_preparation,validate_preparation
from app.locdi.routing_collection_preparation import project_input,response_schema
from app.locdi.m5_common_source import verify_csv
from app.locdi.frozen_m5_runtime_adapter import bind_value
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.locdi.development_routing_proxy import MARKERS
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.routing_utility import observed_tokens
from app.llm.prompt_loader import specialty_system_prompt
from validation.routing_training_data import _leaves

NS='B34_COLLECTION_RECOVERY_v1'
PLAN_ID='source_collection_recovery_001'
COLLECTION_ID='collection_recovery_001'
OLD=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/source_ready_002/routing_training'
PUBLIC=ROOT/'outputs/research_evaluation'/NS
PRIVATE=ROOT/'outputs/private/research_evaluation'/NS/PLAN_ID
SCOPE='CURRENT_3B_ROUTING_RECOVERY_COLLECTION_15_ONLY'


def audit():
    run=OLD/'collection_001';old=read(OLD/'plan.json');data=read(run/'collection.json')
    before=sorted((run/'requests').glob('Request*_before.json'));after=sorted((run/'requests').glob('Request*_after.json'))
    rows=[]
    for bp in before:
        ap=bp.with_name(bp.name.replace('_before.json','_after.json'));b=read(bp);a=read(ap) if ap.exists() else {}
        if a and a['request_fingerprint']!=b['request_fingerprint']:raise ValueError('OLD_CHECKPOINT_LINK_MISMATCH')
        rows.append({'request_number':int(bp.name[7:11]),'status':a.get('status','AFTER_MISSING'),
            'HTTP_status':a.get('HTTP_status'),'response_received':a.get('response_received') is True,
            'error_type':a.get('error_type'),'wall_seconds':a.get('wall_seconds'),
            'receipt_state':'VALID_RESPONSE_RECORDED' if a.get('status')=='VALID' else 'ATTEMPT_WITHOUT_CONFIRMED_RESPONSE'})
    report={'status':'PARTIAL_COLLECTION_INTERRUPTED_WITH_UNCONFIRMED_LAST_REQUEST',**MARKERS,
        'client_generation_attempts_recorded':data['generation'],'before_checkpoints':len(before),'after_checkpoints':len(after),
        'HTTP200_count':sum(r['HTTP_status']==200 for r in rows),'schema_valid_responses':sum(r['status']=='VALID' for r in rows),
        'saved_response_records':len(data['records']),'supported_records':sum(r.get('response',{}).get('certainty')=='supported' for r in data['records']),
        'unresolved_records':sum(r.get('response',{}).get('certainty')=='unresolved' for r in data['records']),
        'unconfirmed_last_attempts':sum(r['receipt_state']!='VALID_RESPONSE_RECORDED' for r in rows),
        'not_attempted_slots':15-data['generation'],'original_stop_reason':data['stop_reason'],
        'termination_cause':'NOT_RECORDED; BaseException interruption is consistent with the existing control flow, not proven',
        'public_status_missing':not (ROOT/'outputs/research_evaluation/B34_CONNECTION_v1/source_ready_002/routing_training/collection_status.json').exists(),
        'recorded_timeout_errors':sum('timeout' in (r['error_type'] or '').lower() for r in rows),
        'unknown_request_server_outcome':'NOT_CHECKED','new_generation_by_codex':0,'requests':rows}
    write(PUBLIC/'old_collection_audit.json',report)
    print(json.dumps(report,ensure_ascii=True))


def fresh_validation(model,excluded):
    admitted=read(ROOT/'outputs/private/research_evaluation/c4_35_v1/approved_admission.json')['panel']
    visit=next((v for v in sorted(admitted,key=lambda v:(v['patient_id'],v['visit_id']))
                if v['patient_id'] not in excluded|set(model['provenance']['training_patients'])),None)
    if visit is None:raise ValueError('NO_UNUSED_DISJOINT_VALIDATION_SOURCE')
    conditions=Path(next(iter(visit['values'].values()))['source_file']).parent/'conditions.csv'
    references=[]
    with conditions.open(encoding='utf-8-sig',newline='') as f:
        for n,row in enumerate(csv.DictReader(f),2):
            if row['PATIENT']==visit['patient_id'] and row['ENCOUNTER']==visit['visit_id']:
                references.append({'source_file':str(conditions),'source_row':n,'source_record':row})
    verify_csv([{'source_metadata':{'source_file':v['source_file'],'source_row':v['source_row'],
        'source_record':v['source_record']}} for v in visit['values'].values()])
    findings=[];history=[]
    for key in model['provenance']['action_key_mapping']:
        bound=bind_value(model,visit,key);raw=visit['values'][key]['source_record']
        text=raw['DESCRIPTION']+': '+json.dumps({'value':raw['VALUE'],'units':raw['UNITS']},ensure_ascii=False,sort_keys=True)
        if bound['type']=='ask_patient':history.append({'question':bound['target'],'answer':text,'answer_status':'known'})
        else:findings.append(text)
    projection=ROOT/'outputs/private/research_evaluation/c4_37_v1/new_observation_memory_projection.jsonl';past=None
    with projection.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if (row['identity']['patient_id'],row['identity']['visit_id'])==(visit['patient_id'],visit['visit_id']):past=row['past_visits'];break
    if past is None:raise ValueError('NEW_VALIDATION_PAST_SOURCE_NOT_FOUND')
    identity={'patient_id':visit['patient_id'],'visit_id':visit['visit_id'],'started_at':visit['decision_time']}
    memory=PatientMemoryStore([r for v in past for r in v['records']]).retrieve(identity,' '.join(findings),'selective',5)
    obs={'chief_complaint':'','demographics':{},'elapsed_minutes':45,'current_symptoms':[],
        'history':history,'objective_findings':findings,'longitudinal_memory':memory}
    fp=digest({'values':visit['values'],'observation':obs})
    return {'case_ref':'FreshValidationRecovery001','visit_started_at':visit['decision_time'],'source_fingerprint':fp,
        'visibility':{k:{'source_ref':'fresh-source:'+fp,'available_minute':45,'basis':'RECORDED_TIME_PROXY',
            'actual_release_status':'NOT_RECORDED','review_status':'NOT_REVIEWED'} for k in _leaves(obs)},
        'decision':{'patient_id':visit['patient_id'],'visit_id':visit['visit_id'],
            'decision_id':'common-validation-recovery:'+visit['visit_id']+':minute45','minute':45,'split':'validation',
            'source_ref':'fresh-source:'+fp,'observation':obs,'allowed_actions':['diagnose']},
        'reference_candidates':{'Source_candidates_only':visit['reference_candidates'],'clinical_review':'NOT_REVIEWED',
            'source_condition_rows':references,'diagnosis_confirmed_at':visit['diagnosis_confirmed_at'],
            'clinical_timeline':visit['clinical_timeline']}}


def prepare():
    if PRIVATE.exists():raise ValueError('FRESH_RECOVERY_PLAN_ONLY')
    _,pr,evalplan=load_plan(PLAN_ID);old=read(OLD/'plan.json');bundle=load_preparation('phase24b_candidates_v2');validate_preparation(bundle)
    config=read(pr/'llm_config.json');config['max_requests']=15;ready=validate_setup(config)
    if ready['static_readiness']!='READY' or ready['prepared']['registry']!=old['registry']:raise ValueError('CURRENT_BINDING_CHANGED')
    registry=ready['prepared']['registry'];model=read(M5);evaluation={b['patient_id'] for b in read(pr/'bindings.json').values()}
    used={c['decision']['patient_id'] for c in old['candidates']};used_decisions={c['decision']['decision_id'] for c in old['candidates']}
    selected=[]
    for c in bundle['candidates']:
        d=c['decision']
        if d['split']=='train' and d['patient_id'] not in used|evaluation|{x['decision']['patient_id'] for x in selected} and d['decision_id'] not in used_decisions:
            selected.append(deepcopy(c))
        if len(selected)==2:break
    if len(selected)!=2:raise ValueError('TWO_UNUSED_EXISTING_TRAIN_PATIENTS_NOT_FOUND')
    refpath=ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/reference_candidates.json';refs={c['case_ref']:c for c in read(refpath)}
    for c in selected:c['reference_candidates']=deepcopy(refs[c['case_ref']])
    selected.append(fresh_validation(model,used|evaluation|{c['decision']['patient_id'] for c in selected}))
    agents=[a for a in registry['agents'] if a['role']=='specialty']
    generation={'model_identifier':config['model_identifier'],'temperature':0,'seed':42,'max_output_tokens':1024,
        'prompt_fingerprints':{a['agent_id']:ready['prepared']['prompt_fingerprints'][a['agent_id']] for a in agents}}
    for c in selected:
        c['decision']['generation']=deepcopy(generation);c['decision']['registry_fingerprint']=digest(registry)
        c['decision']['tokens']=observed_tokens(c['decision']['observation'])
        c['shared_payload']=project_input(c['decision']['observation'],list(model['diagnoses']))
        c['shared_input_fingerprint']=digest(c['shared_payload'])
    old_payloads={c['shared_input_fingerprint'] for c in old['candidates']}
    if old_payloads&{c['shared_input_fingerprint'] for c in selected} or len({c['shared_input_fingerprint'] for c in selected})!=3:
        raise ValueError('DUPLICATE_SHARED_MODEL_INPUT')
    plan=deepcopy(old);plan.update(version='locdi.routing.fresh_recovery.v1',plan_id=PLAN_ID,collection_id=COLLECTION_ID,
        evaluation_plan_fingerprint=evalplan['fingerprint'],registry=registry,llm_config=config,candidates=selected,
        selection='first two unused existing Train patients in original order; first disjoint unused admitted Validation patient',
        prior_collection_path=str(OLD/'collection_001'),prior_attempts=12,prior_uncertain_attempts=1,
        resume=False,previous_responses_merged=False,prior_patients_excluded=sorted(used))
    prep=ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json'
    plan['source_files'].update({str(prep):sha(prep),str(refpath):sha(refpath)})
    from app.locdi.routing_collection_preparation import BANK,SOURCE,DATABASE
    plan['source_files'].update({str(p):sha(p) for p in (BANK,SOURCE,DATABASE)})
    plan['code_files'].update({str(Path(__file__).resolve()):sha(Path(__file__))})
    plan['fingerprint']=digest({k:v for k,v in plan.items() if k!='fingerprint'})
    write(PRIVATE/'plan.json',plan)
    write(PRIVATE/'approval_template.json',{'scope':SCOPE,'plan_fingerprint':plan['fingerprint'],
        'plan_id':PLAN_ID,'collection_id':COLLECTION_ID,'max_generation':15,'operator_approved':False,
        'accept_source_proxy_for_collection_only':False,'no_other_ollama_work':False,
        'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None,**MARKERS})
    write(PUBLIC/'new_collection_preparation.json',{'plan_id':PLAN_ID,'collection_id':COLLECTION_ID,
        'plan_fingerprint':plan['fingerprint'],'train_patients':2,'validation_patients':1,'agents':5,
        'previous_patient_overlap':0,'previous_decision_overlap':0,'previous_exact_input_overlap':0,
        'evaluation_patient_overlap':0,'validation_M5_train_overlap':0,'previous_responses_merged':False,
        'max_generation':15,'generation_executed':0,'prior_client_attempts':12,'total_attempt_cap_across_runs':27,
        'source_task_review':'NOT_REVIEWED',**MARKERS})
    print(json.dumps({'status':'NEW_INDEPENDENT_COLLECTION_PREPARED','plan_fingerprint':plan['fingerprint'],'generation':0}))


def load_checked():
    plan=read(PRIVATE/'plan.json')
    if plan['fingerprint']!=digest({k:v for k,v in plan.items() if k!='fingerprint'}):raise ValueError('RECOVERY_PLAN_CHANGED')
    load_plan(PLAN_ID)
    if check({**plan['source_files'],**plan['code_files']})['status']!='PASS':raise ValueError('RECOVERY_SOURCE_OR_CODE_CHANGED')
    if (PRIVATE/COLLECTION_ID).exists():raise ValueError('RECOVERY_COLLECTION_USED_NO_RESUME')
    return plan


def execute(authorization):
    plan=load_checked();a=read(authorization)
    required={'scope':SCOPE,'plan_fingerprint':plan['fingerprint'],'plan_id':PLAN_ID,'collection_id':COLLECTION_ID,
        'max_generation':15,'operator_approved':True,'accept_source_proxy_for_collection_only':True,'no_other_ollama_work':True,**MARKERS}
    if any(a.get(k)!=v for k,v in required.items()) or not a.get('operator_ref') or not a.get('user_statement'):
        raise ValueError('NEW_OPERATOR_APPROVAL_FOR_THIS_RECOVERY_ONLY')
    start_at=datetime.fromisoformat(a['recorded_at']);expires=datetime.fromisoformat(a['expires_at']);now=datetime.now(timezone.utc)
    if start_at.tzinfo is None or expires.tzinfo is None or not start_at<=now<expires or not 0<(expires-start_at).total_seconds()<=86400:raise ValueError('RECOVERY_APPROVAL_TIME_INVALID')
    folder=PRIVATE/COLLECTION_ID;folder.mkdir(exist_ok=False);write(folder/'approval.json',a)
    client=None;token=None;records=[];stop=None;interrupted=None;began=time.monotonic();budget=GenerationBudget(15,15)
    def guard():
        if datetime.now(timezone.utc)>=expires:raise ValueError('RECOVERY_APPROVAL_EXPIRED')
        if time.monotonic()-began>1800 or _process_private_bytes()>4*1024**3 or sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>1024**3:
            raise ValueError('RECOVERY_RESOURCE_BUDGET')
        if check(plan['source_files'])['status']!='PASS':raise ValueError('RECOVERY_SOURCE_CHANGED')
    budget.resource_guard=guard
    try:
        client=BudgetedNativeERClient(plan['llm_config'],authorized=True,contract_fingerprint=plan['fingerprint'],checkpoint_root=folder/'requests')
        token=_BUDGET.set(budget)
        for c in plan['candidates']:
            for agent in [a for a in plan['registry']['agents'] if a['role']=='specialty']:
                d=c['decision'];ref=c['case_ref']+'_'+agent['agent_id']
                record={'response_ref':ref,'decision_id':d['decision_id'],'patient_id':d['patient_id'],'source_ref':d['source_ref'],
                    'purpose':'routing_training','agent_id':agent['agent_id'],'model':agent['model'],'version':agent['version'],
                    'configuration_fingerprint':digest(d),'observation_fingerprint':digest(d['observation']),
                    'generation':d['generation'],'tokens':d['tokens'],'execution_status':'FAILED','response_contract_status':'NOT_RECORDED',
                    'correct':None,'review_status':'NOT_REVIEWED'}
                try:
                    raw=client.chat_json(specialty_system_prompt(agent['department']),json.dumps({'department':agent['department'],**deepcopy(c['shared_payload'])},ensure_ascii=False),model=agent['model'],schema=response_schema(plan['catalog']))
                    answer=json.loads(raw);record.update(response=answer,response_fingerprint=digest(answer),execution_status='SUCCEEDED',
                        response_contract_status='VALID',observed_model_identifier=plan['llm_config']['model_identifier'],
                        assessment_status='UNRESOLVED' if answer['certainty']=='unresolved' or answer['diagnosis']=='Unresolved' else 'RESPONSE_RECORDED')
                except BaseException as exc:
                    record['error_type']=type(exc).__name__;stop='INTERRUPTED' if not isinstance(exc,Exception) else 'FIRST_TECHNICAL_FAILURE'
                    if not isinstance(exc,Exception):interrupted=exc
                write(folder/(ref+'.json'),record);records.append(record)
                if stop:break
            if stop:break
    except BaseException as exc:
        stop=stop or ('INTERRUPTED' if not isinstance(exc,Exception) else 'SETUP_OR_STORAGE_FAILURE')
        interrupted=exc
    finally:
        if token is not None:_BUDGET.reset(token)
        if client is not None:client.close()
        generation=client.request_count if client is not None else 0
        write(folder/'collection.json',{'records':records,'generation':generation,'generation_budget_reserved':budget.used,
            'stop_reason':stop,'wall_seconds':time.monotonic()-began,'fit_ready':False,'clinical_reviews_created':0,**MARKERS})
        after=[read(p) for p in (folder/'requests').glob('Request*_after.json')]
        write(PUBLIC/'new_collection_status.json',{'plan_id':PLAN_ID,'collection_id':COLLECTION_ID,'generation':generation,
            'records_written':len(records),'responses_received':sum(r.get('response_received') is True for r in after),
            'schema_valid_responses':sum(r.get('status')=='VALID' for r in after),'not_attempted_generation_slots':15-generation,
            'stop_reason':stop,'utility_fit_calls':0,**MARKERS})
    if interrupted is not None:raise interrupted
    if stop:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit','prepare','check','execute'],nargs='?',default='check')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');p.add_argument('--authorization',type=Path);a=p.parse_args()
    if a.mode!='execute' and (a.execute or a.allow_llm or a.authorization):p.error('EXECUTION_FLAGS_ONLY_FOR_EXECUTE')
    if a.mode=='audit':audit()
    elif a.mode=='prepare':prepare()
    elif a.mode=='check':print(json.dumps({'status':'PREPARED_NEW_APPROVAL_REQUIRED','fingerprint':load_checked()['fingerprint'],'generation':0}))
    else:
        if not(a.execute and a.allow_llm and a.authorization):p.error('EXPLICIT_USER_EXECUTE_ALLOW_LLM_AND_NEW_APPROVAL_REQUIRED')
        execute(a.authorization)


if __name__=='__main__':main()
