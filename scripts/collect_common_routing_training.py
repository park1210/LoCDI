# -*- coding: utf-8 -*-
"""User-only new 3B collection: two Train patients and one disjoint validation.

No labels/fitting/reviews are generated. Default check never creates a client.
User must prepare the common evaluation plan first, then approve this plan.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_common_menu_b0_b4 import paths,load_plan,M5
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collector import load_preparation,validate_preparation
from app.locdi.routing_collection_preparation import project_input,response_schema
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.routing_utility import observed_tokens
from app.locdi.frozen_m5_runtime_adapter import bind_value
from app.locdi.m5_common_source import verify_csv
from app.llm.prompt_loader import specialty_system_prompt
from validation.routing_training_data import _leaves
from validation.routing_correctness_gate import empty_review_results
from app.locdi.development_routing_proxy import MARKERS


def training_paths(name):
    pub,pr=paths(name);return pub/'routing_training',pr/'routing_training'


def prepare(name):
    _,evalpr,evalplan=load_plan(name);pub,pr=training_paths(name)
    if pub.exists() or pr.exists():raise ValueError('NEW_TRAIN_PREPARATION_ONLY')
    bundle=load_preparation('phase24b_candidates_v2');validate_preparation(bundle)
    config=read(evalpr/'llm_config.json');config['max_requests']=15;ready=validate_setup(config)
    if ready['static_readiness']!='READY':raise ValueError('CURRENT_3B_SETUP_REQUIRED')
    registry=ready['prepared']['registry'];model=read(M5)
    hold={b['patient_id'] for b in read(evalpr/'bindings.json').values()}
    selected=[deepcopy(c) for c in bundle['candidates'] if c['decision']['split']=='train'][:2]
    references=read(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/reference_candidates.json')
    reference_map={r['case_ref']:r for r in references}
    for candidate in selected:candidate['reference_candidates']=deepcopy(reference_map[candidate['case_ref']])
    if len(selected)!=2:raise ValueError('TWO_TRAIN_SOURCE_CANDIDATES_REQUIRED')
    train={c['decision']['patient_id'] for c in selected}
    if len(train)!=2 or train&hold:raise ValueError('TRAIN_EVALUATION_PATIENT_OVERLAP')
    admission_path=ROOT/'outputs/private/research_evaluation/c4_35_v1/approved_admission.json'
    admitted=read(admission_path)['panel']
    validation=next((v for v in sorted(admitted,key=lambda v:(v['patient_id'],v['visit_id']))
        if v['patient_id'] not in train|hold|set(model['provenance']['training_patients'])),None)
    if validation is None:raise ValueError('DISJOINT_ROUTING_VALIDATION_SOURCE_MISSING')
    conditions_file=Path(next(iter(validation['values'].values()))['source_file']).parent/'conditions.csv'
    condition_refs=[]
    with conditions_file.open(encoding='utf-8-sig',newline='') as f:
        import csv
        for n,row in enumerate(csv.DictReader(f),2):
            if row['PATIENT']==validation['patient_id'] and row['ENCOUNTER']==validation['visit_id']:
                condition_refs.append({'source_file':str(conditions_file),'source_row':n,'source_record':row})
    fields=[];history=[]
    verify_csv([{'source_metadata':{'source_file':v['source_file'],'source_row':v['source_row'],
        'source_record':v['source_record']}} for v in validation['values'].values()])
    for key in model['provenance']['action_key_mapping']:
        bound=bind_value(model,validation,key);value=validation['values'][key];raw=value['source_record']
        text=raw['DESCRIPTION']+': '+json.dumps({'value':raw['VALUE'],'units':raw['UNITS']},ensure_ascii=False,sort_keys=True)
        if bound['type']=='ask_patient':history.append({'question':bound['target'],'answer':text,'answer_status':'known'})
        else:fields.append(text)
    obs={'chief_complaint':'','demographics':{},'elapsed_minutes':45,'current_symptoms':[],
         'history':history,'objective_findings':fields,'longitudinal_memory':{'mode':'selective','records':[]}}
    # Past-source Memory is required to be supplied/verified before collection;
    # do not fabricate records if no exact patient projection can be found.
    projection=ROOT/'outputs/private/research_evaluation/c4_37_v1/new_observation_memory_projection.jsonl'
    past=None
    with projection.open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if (row['identity']['patient_id'],row['identity']['visit_id'])==(validation['patient_id'],validation['visit_id']):
                past=row['past_visits'];break
    if past is None:raise ValueError('VALIDATION_PAST_SOURCE_PROJECTION_MISSING')
    records=[r for v in past for r in v['records']]
    identity={'patient_id':validation['patient_id'],'visit_id':validation['visit_id'],'started_at':validation['decision_time']}
    obs['longitudinal_memory']=PatientMemoryStore(records).retrieve(identity,' '.join(fields),'selective',5)
    source_fp=digest({'values':validation['values'],'observation':obs})
    selected.append({'case_ref':'FreshValidation001','visit_started_at':validation['decision_time'],
        'source_fingerprint':source_fp,'visibility':{k:{'source_ref':'fresh-source:'+source_fp,'available_minute':45,
            'basis':'RECORDED_TIME_PROXY','actual_release_status':'NOT_RECORDED','review_status':'NOT_REVIEWED'} for k in _leaves(obs)},
        'decision':{'patient_id':validation['patient_id'],'visit_id':validation['visit_id'],
            'decision_id':'common-validation:'+validation['visit_id']+':minute45','minute':45,'split':'validation',
            'source_ref':'fresh-source:'+source_fp,'observation':obs,'allowed_actions':['diagnose']},
        'reference_candidates':{'Source_candidates_only':validation['reference_candidates'],'clinical_review':'NOT_REVIEWED',
            'source_condition_rows':condition_refs,'diagnosis_confirmed_at':validation['diagnosis_confirmed_at'],
            'clinical_timeline':validation['clinical_timeline']}})
    agents=[a for a in registry['agents'] if a['role']=='specialty']
    generation={'model_identifier':config['model_identifier'],'temperature':0,'seed':42,'max_output_tokens':1024,
        'prompt_fingerprints':{a['agent_id']:ready['prepared']['prompt_fingerprints'][a['agent_id']] for a in agents}}
    for c in selected:
        d=c['decision'];d['registry_fingerprint']=digest(registry);d['generation']=deepcopy(generation)
        d['tokens']=observed_tokens(d['observation'])
        c['shared_payload']=project_input(d['observation'],list(model['diagnoses']))
        c['shared_input_fingerprint']=digest(c['shared_payload'])
    if len({c['shared_input_fingerprint'] for c in selected})!=len(selected):
        raise ValueError('DUPLICATE_MODEL_VISIBLE_TRAIN_VALIDATION_INPUT')
    plan={'version':'locdi.common_menu.current3b.training.v1','evaluation_plan_fingerprint':evalplan['fingerprint'],
        'registry':registry,'llm_config':config,'catalog':list(model['diagnoses']),'candidates':selected,
        'generation_total':15,'wall_seconds':1800,'process_bytes':4*1024**3,'output_bytes':1024**3,
        'selection':'first2 existing train candidates; first disjoint admitted fresh patient, no response/outcome selection',
        'review_status':'NOT_REVIEWED','clinical_reference_created':False,'fit_called':False,
        'source_files':{str(admission_path):sha(admission_path),str(projection):sha(projection),str(conditions_file):sha(conditions_file)},
        'code_files':{str(Path(__file__)):sha(Path(__file__))}}
    if evalplan.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT':plan.update(MARKERS)
    plan['fingerprint']=digest(plan);write(pr/'plan.json',plan)
    write(pr/'source_review_packet_precollection.json',{'review_status':'NOT_REVIEWED','responses_included':False,
        'items':[{k:deepcopy(c[k]) for k in ('case_ref','decision','visit_started_at','source_fingerprint','visibility','reference_candidates')}
                 for c in selected]})
    write(pr/'review_results_blank.json',empty_review_results())
    write(pr/'approval_template.json',{'scope':'CURRENT_3B_ROUTING_COLLECTION_15_ONLY','plan_fingerprint':plan['fingerprint'],
        'max_generation':15,'operator_approved':False,'accept_source_proxy_for_collection_only':False,
        'no_other_ollama_work':False,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None})
    write(pub/'readiness.json',{'status':'SOURCE_CANDIDATES_PREPARED_NOT_CLINICALLY_REVIEWED',
        'train_patients':2,'validation_patients':1,'specialty_agents':5,'max_generation':15,
        'routing_validation_M5_train_overlap':0,'evaluation_patient_overlap':0,'fit_ready':False,
        'generation':0,'source_task_suitability_review_required':True,
        'clinical_review_blocks_development':False,'clinical_review_blocks_clinical_evaluation':True,
        **(MARKERS if evalplan.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT' else {})})
    print(json.dumps({'plan':plan['fingerprint'],'generation':0,'review':'NOT_REVIEWED'}))


def review_packet(name):
    _,evalpr,ep=load_plan(name);pub,pr=training_paths(name);plan=read(pr/'plan.json')
    collection=read(pr/'collection_001/collection.json')
    from app.locdi.dataset_loader import load_cases
    holdout=sorted({c['patient_id'] for c in load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')})
    manifest={'schema_version':'locdi.routing_split.v1','dataset_fingerprint':sha(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'),
        'holdout_fingerprint':digest(holdout),'patients':{'train':[],'validation':[],'test':holdout},'decisions':[]}
    sources=[];contexts=[]
    for c in plan['candidates']:
        d=c['decision'];manifest['patients'][d['split']].append(d['patient_id'])
        manifest['decisions'].append({k:d[k] for k in ('patient_id','decision_id','split')})
        source_fp=c['source_fingerprint']
        sources.append({'source_ref':d['source_ref'],'purpose':'routing_training',
            **{k:deepcopy(d[k]) for k in ('patient_id','visit_id','decision_id','minute','observation')},
            'source_fingerprint':source_fp,'visit_started_at':c['visit_started_at'],
            'visibility':{k:v['available_minute'] for k,v in c['visibility'].items()},'review':{'status':'NOT_REVIEWED'}})
        contexts.append({'case_ref':c['case_ref'],**{k:d[k] for k in ('patient_id','visit_id','decision_id')},
            'encounter_started_at':c['visit_started_at'],'source_fingerprint':source_fp,'visibility':deepcopy(c['visibility']),
            'reference_candidates':deepcopy(c['reference_candidates']),'diagnosis_catalog':plan['catalog'],
            'original_response_fingerprints':{r['response_ref']:r.get('response_fingerprint') for r in collection['records'] if r['decision_id']==d['decision_id']}})
    data={'schema_version':'locdi.routing_collection.v1','purpose':'routing_training','registry':plan['registry'],
        'split_manifest':manifest,'decisions':[c['decision'] for c in plan['candidates']],
        'outcomes':collection['records'],'judgments':[]}
    options={'review_results':empty_review_results(),'decision_contexts':contexts,'registry_manifest':plan['registry'],
        'source_catalog':sources,'label_catalog':[],'holdout_patient_ids':holdout,'dataset_fingerprint':manifest['dataset_fingerprint']}
    write(pr/'review_input_blank.json',{'training_data':data,'validator_options':options,
        'model_identifier':plan['llm_config']['model_identifier'],
        'prompt_fingerprints':validate_setup(plan['llm_config'])['prepared']['prompt_fingerprints'],
        'review_status':'NOT_REVIEWED','routing_artifact':None})
    blind=[{k:deepcopy(v) for k,v in c.items() if k!='original_response_fingerprints'} for c in contexts]
    write(pr/'reference_source_packet.json',{'sources':sources,'decision_contexts':blind,
        'agent_responses_included':False,'independent_review_status':'NOT_REVIEWED'})
    write(pr/'response_review_packet.json',{'records':collection['records'],
        'correctness':None,'review_status':'NOT_REVIEWED','reference_review_must_be_completed_first':True})
    write(pub/'review_packet_status.json',{'responses':len(collection['records']),'reviewed_correctness':0,
        'clinical_verdicts_created':0,'training_ready':False})


def execute(name,approval_file):
    pub,pr=training_paths(name);plan=read(pr/'plan.json');a=read(approval_file)
    if plan['fingerprint']!=digest({k:v for k,v in plan.items() if k!='fingerprint'}):raise ValueError('PLAN_CHANGED')
    load_plan(name)
    if check({**plan['source_files'],**plan['code_files']})['status']!='PASS':raise ValueError('SOURCE_CODE_CHANGED')
    if (a.get('scope')!='CURRENT_3B_ROUTING_COLLECTION_15_ONLY' or a.get('plan_fingerprint')!=plan['fingerprint']
        or a.get('max_generation')!=15 or a.get('operator_approved') is not True
        or a.get('accept_source_proxy_for_collection_only') is not True or a.get('no_other_ollama_work') is not True
        or not all(isinstance(a.get(k),str) and a[k].strip() for k in ('user_statement','operator_ref'))):
        raise ValueError('NEW_ACTUAL_COLLECTION_APPROVAL_REQUIRED')
    issued=datetime.fromisoformat(a['recorded_at']);expires=datetime.fromisoformat(a['expires_at']);now=datetime.now(timezone.utc)
    if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires or not 0<(expires-issued).total_seconds()<=86400:
        raise ValueError('APPROVAL_TIME_INVALID')
    run=pr/'collection_001';run.mkdir(exist_ok=False);write(run/'approval.json',a)
    config=deepcopy(plan['llm_config'])
    client=BudgetedNativeERClient(config,authorized=True,contract_fingerprint=plan['fingerprint'],checkpoint_root=run/'requests')
    started=time.monotonic();records=[];stop=None
    source_stats={k:(Path(k).stat().st_size,Path(k).stat().st_mtime_ns) for k in plan['source_files']}
    def guard():
        if datetime.now(timezone.utc)>=expires:raise ValueError('COLLECTION_APPROVAL_EXPIRED')
        if time.monotonic()-started>1800:raise ValueError('COLLECTION_WALL_BUDGET')
        if _process_private_bytes()>plan['process_bytes']:raise ValueError('COLLECTION_MEMORY_BUDGET')
        if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>plan['output_bytes']:raise ValueError('COLLECTION_OUTPUT_BUDGET')
        if any((Path(k).stat().st_size,Path(k).stat().st_mtime_ns)!=v for k,v in source_stats.items()):raise ValueError('COLLECTION_SOURCE_CHANGED')
    budget=GenerationBudget(15,15,resource_guard=guard);token=_BUDGET.set(budget)
    try:
        for candidate in plan['candidates']:
            for agent in [a for a in plan['registry']['agents'] if a['role']=='specialty']:
                d=candidate['decision'];ref=candidate['case_ref']+'_'+agent['agent_id']
                record={'response_ref':ref,'decision_id':d['decision_id'],'patient_id':d['patient_id'],
                    'source_ref':d['source_ref'],'purpose':'routing_training','agent_id':agent['agent_id'],
                    'model':agent['model'],'version':agent['version'],'configuration_fingerprint':digest(d),
                    'observation_fingerprint':digest(d['observation']),'generation':d['generation'],'tokens':d['tokens'],
                    'execution_status':'FAILED','response_contract_status':'NOT_RECORDED','correct':None,'review_status':'NOT_REVIEWED'}
                try:
                    payload={'department':agent['department'],**deepcopy(candidate['shared_payload'])}
                    raw=client.chat_json(specialty_system_prompt(agent['department']),json.dumps(payload,ensure_ascii=False),
                        model=agent['model'],schema=response_schema(plan['catalog']))
                    answer=json.loads(raw);record.update(response=answer,response_fingerprint=digest(answer),
                        execution_status='SUCCEEDED',response_contract_status='VALID',observed_model_identifier=config['model_identifier'],
                        assessment_status='UNRESOLVED' if answer['certainty']=='unresolved' or answer['diagnosis']=='Unresolved' else 'RESPONSE_RECORDED')
                except Exception as exc:record['error_type']=type(exc).__name__;stop='FIRST_TECHNICAL_FAILURE'
                write(run/(ref+'.json'),record);records.append(record)
                if stop:break
            if stop:break
    finally:
        _BUDGET.reset(token);client.close()
        write(run/'collection.json',{'records':records,'generation':client.request_count,'stop_reason':stop,
            'wall_seconds':time.monotonic()-started,'fit_ready':False,'clinical_reviews_created':0,
            **(MARKERS if plan.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT' else {})})
    after=[read(p) for p in (run/'requests').glob('Request*_after.json')]
    write(pub/'collection_status.json',{'generation':client.request_count,'registered_tasks':15,
        'records_written':len(records),'responses_received':sum(x.get('response_received') is True for x in after),
        'schema_valid_responses':sum(x.get('status')=='VALID' for x in after),
        'not_attempted_tasks':15-len(records),'stop_reason':stop,
        'correctness':'NOT_REVIEWED','utility_fitting_calls':0,
        **(MARKERS if plan.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT' else {})})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','check','execute','review-packet'],nargs='?',default='check')
    p.add_argument('--plan-id',default='common_all_001');p.add_argument('--authorization',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');args=p.parse_args()
    if args.mode!='execute' and (args.execute or args.allow_llm or args.authorization):p.error('EXECUTION_FLAGS_ARE_EXECUTE_ONLY')
    if args.mode=='prepare':prepare(args.plan_id)
    elif args.mode=='review-packet':review_packet(args.plan_id)
    elif args.mode=='execute':
        if not(args.execute and args.allow_llm and args.authorization):p.error('NEW_USER_EXECUTE_ALLOW_LLM_APPROVAL_REQUIRED')
        execute(args.plan_id,args.authorization)
    else:
        pub,pr=training_paths(args.plan_id);plan=read(pr/'plan.json')
        print(json.dumps({'status':'PREPARED_OPERATOR_PROXY_APPROVAL_REQUIRED' if plan.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT' else 'PREPARED_NEW_APPROVAL_AND_SOURCE_REVIEW_REQUIRED',
            'plan':plan['fingerprint'],'generation':0,'fit_ready':False,
            **(MARKERS if plan.get('experiment_kind')=='DEVELOPMENT_EXPERIMENT' else {})}))
