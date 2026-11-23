"""Source-only, Native-shaped static snapshot collection, NOT Native Episodes.

Explicit recorded-time predisclosure is a separate development assumption.
No fitting, symptom authoring, Source edits, clinical release or grants.
"""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from collections import Counter
import sqlite3,argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.routing_data_reuse import BANK,SOURCE,DATABASE,read as source_read
from app.locdi.routing_collection_preparation import _snapshot,_fixed_memory
from app.locdi.m5_common_source import verify_csv
from app.locdi.source_paths import resolve_source_file
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.synthetic_pilot_collection import response_schema
from app.llm.prompt_loader import specialty_system_prompt
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient,input_guard
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.routing_collector import atomic_json
from scripts.run_original50_d0_d2 import M5,write_equal

NS='U1_SUPPORT_VOI_EXPANSION_v1';NAME='u2_source_snapshot_collect_001';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
CFG=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/original50_d0d2_full_003_source/llm_config.json'
CODE=['scripts/prepare_u2_source_collection.py','scripts/Invoke-LoCDIU2SourceCollection.ps1',
      'app/locdi/routing_collection_preparation.py','app/locdi/routing_data_reuse.py','app/locdi/source_paths.py',
      'app/locdi/synthetic_pilot_collection.py','app/llm/prompt_loader.py']

def reference_check(bank,source,catalog):
    patient=bank['training_patient_id'];visit=source['index_encounter']['encounter_id'];start=source['index_encounter']['start_time']
    refs=[r for r in bank['source_provenance']['condition_rows'] if r['source_record'].get('PATIENT')==patient and r['source_record'].get('ENCOUNTER')==visit]
    labels={r['source_record'].get('DESCRIPTION') for r in refs}
    if not refs or len(labels)!=1 or None in labels:return None,'AMBIGUOUS_OR_MISSING_CURRENT_REFERENCE'
    if any(not r['source_record'].get('START') or r['source_record']['START'][:10]>start[:10] for r in refs):return None,'FUTURE_REFERENCE'
    label=next(iter(labels))
    if label not in catalog:return None,'OUTSIDE_CATALOG'
    return label,'SAME_DAY_OR_EARLIER_SOURCE_DATE_PROXY_INTRADAY_UNREVIEWED'

def prepare():
    pub,pr=PUB/NAME,PR/NAME
    if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY_NO_OVERWRITE')
    bank=source_read(BANK);sources={r['case_id']:r for r in source_read(SOURCE)};model=read(M5);catalog=model['diagnoses'];holdout={c['patient_id'] for c in original_cases()}
    oldprep=read(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json');old_reserved={pid for ids in oldprep['split_manifest']['patients'].values() for pid in ids}
    available=[];reasons=Counter();all_source_refs=[]
    for b in bank:
        s=sources[b['training_case_id']];label,status=reference_check(b,s,catalog)
        if b['training_patient_id'] in holdout:reasons['ORIGINAL50_OVERLAP']+=1;continue
        if b['training_patient_id'] in old_reserved:reasons['PREVIOUS_ROUTING_TRAIN_OR_VALIDATION_RESERVED']+=1;continue
        if label is None:reasons[status]+=1;continue
        available.append((b,s,label,status))
    # No evaluation label, department or previous Agent performance participates.
    available.sort(key=lambda x:digest(['u2-source-independent-v1',20261122,x[0]['training_patient_id'],x[0]['training_case_id']]))
    selected=[];seen=set()
    for row in available:
        if row[0]['training_patient_id'] in seen:continue
        seen.add(row[0]['training_patient_id']);selected.append(row)
        if len(selected)==10:break
    if len(selected)!=10:raise ValueError('INSUFFICIENT_DISJOINT_REFERENCE_CANDIDATES')
    cfg=deepcopy(read(CFG));cfg['max_requests']=1;ready=validate_setup(cfg)
    if ready['static_readiness']!='READY':raise ValueError('STATIC_CONFIG_NOT_READY')
    registry=ready['prepared']['registry'];agents=[a for a in registry['agents'] if a['role']=='specialty'];assert len(agents)==5
    menu={}
    for key,aid in model['provenance']['action_key_mapping'].items():
        kind,system,code,unit,resource=json.loads(key);menu[code,unit]=(kind,next(a for a in model['actions'] if a['id']==aid)['target'])
    rows=[];references=[];ledger=[];bounds=[]
    connection=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        for ordinal,(b,s,label,status) in enumerate(selected,1):
            context=json.loads(connection.execute('SELECT payload FROM contexts WHERE cid=?',(b['training_case_id'],)).fetchone()[0])
            if context['patient_id']!=b['training_patient_id'] or context['index_encounter']!=s['index_encounter']:raise ValueError('CONTEXT_SOURCE_BINDING_MISMATCH')
            narrow=deepcopy(b);allowed=set()
            for e in s['source_events']:
                if (e['target'].get('code'),e['evidence'].get('units')) in menu:allowed.add(e['event_id'])
            for e in s['symptom_sources']:
                raw=e['source_metadata']['_source']['source_record']
                if (raw.get('CODE'),raw.get('UNITS')) in menu:allowed.add(e['source_id'])
            narrow['source_provenance']['observation_event_ids']=[i for i in b['source_provenance']['observation_event_ids'] if i in allowed]
            provisional,_,_=_snapshot(narrow,s,context,{'mode':'selective','records':[]})
            query=' '.join(provisional['objective_findings']+[r['answer'] for r in provisional['history']]);memory,rejected=_fixed_memory(connection,context,query)
            observation,links,future=_snapshot(narrow,s,context,memory)
            events={}
            selected_ids=set(narrow['source_provenance']['observation_event_ids'])
            for e in s['source_events']:
                if e['event_id'] in selected_ids:
                    key=e['target'].get('code'),e['evidence'].get('units')
                    if menu[key][0]!='Test':continue
                    text='{}: {} {}'.format(e['target']['description'],e['evidence']['value'],e['evidence'].get('units') or '').strip()
                    if text in observation['objective_findings']:events[text]=e
            rendered=[];measurements=[]
            for text in observation['objective_findings']:
                e=events[text];kind,target=menu[e['target']['code'],e['evidence']['units']]
                evidence=json.dumps({'value':e['evidence']['value'],'units':e['evidence']['units'],'recorded_at':e['timestamp']},sort_keys=True,ensure_ascii=False,separators=(',',':'))
                rendered.append(target+': '+evidence);measurements.append({'evidence_index':len(rendered)-1,'kind':'Test','description':e['target']['description'],'code':e['target']['code'],'value':e['evidence']['value'],'unit':e['evidence']['units']})
                all_source_refs.append({'source_metadata':e['source_metadata']})
            observation['objective_findings']=rendered
            for item in s['symptom_sources']:
                if item['source_id'] in selected_ids:all_source_refs.append({'source_metadata':item['source_metadata']['_source']})
            for ref in b['source_provenance']['condition_rows']:all_source_refs.append({'source_metadata':ref})
            shared={'observation':observation,'released_measurements':measurements,'diagnosis_catalog':list(catalog),
                'instruction':'Return one diagnostic proposal from the fixed catalog or Unresolved. This is an explicitly predisclosed static Source snapshot, not an executed Native test or Hospital clinical finalization. Only supplied patient facts may be used.'}
            rows.append({'ordinal':ordinal,'patient_id':b['training_patient_id'],'visit_id':s['index_encounter']['encounter_id'],'decision_id':'u2-source:'+b['training_case_id']+':minute1',
                'source_context_id':b['training_case_id'],'split':'train','shared':shared,'shared_fingerprint':digest(shared),'source_fingerprint':digest({'bank':b,'source':s}),
                'reference_fingerprint':digest(b['source_provenance']['condition_rows'])})
            references.append({'ordinal':ordinal,'patient_id':b['training_patient_id'],'visit_id':s['index_encounter']['encounter_id'],'label':label,'timing_status':status,'source_condition_rows':b['source_provenance']['condition_rows'],'clinical_review':'NOT_REVIEWED'})
            ledger.append({'ordinal':ordinal,'available_minute':1,'release_policy':'EXPLICIT_STATIC_RECORDED_TIME_PREDISCLOSURE_PROXY','actual_native_test_actions_executed':False,'actual_native_Ask_executed':False,
                'clinical_release_approved':False,'original_links':links,'transform':'M5 code/unit target + original recorded_at/value/unit; no new clinical values',
                'excluded_future_observations':future,'memory_rejections':rejected,'chief_complaint_present':bool(observation['chief_complaint']),'current_symptom_count':len(observation['current_symptoms'])})
            for agent in agents:bounds.append(input_guard(specialty_system_prompt(agent['department']),json.dumps({'department':agent['department'],**shared},sort_keys=True),response_schema(catalog))['conservative_input_token_bound'])
    finally:connection.close()
    verify_csv(all_source_refs)
    pub.mkdir(parents=True);pr.mkdir(parents=True)
    atomic_json(pr/'candidates.json',rows);atomic_json(pr/'reference_private.json',references);atomic_json(pr/'visibility_ledger.json',ledger)
    train_ids={r['patient_id'] for r in rows};assert len(train_ids)==10 and not train_ids&holdout and not train_ids&old_reserved
    reservations=[{'patient_id':b['training_patient_id'],'context_id':b['training_case_id'],'split':'train'} for b in bank if b['training_patient_id'] in train_ids]
    atomic_json(pr/'split_reservations.json',reservations)
    files=[BANK,SOURCE,DATABASE,M5,CFG,ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json']+[ROOT/f for f in CODE]
    files+=[resolve_source_file(e['source_metadata']['source_file']) for e in all_source_refs];files+=[pr/f for f in ('candidates.json','reference_private.json','visibility_ledger.json','split_reservations.json')]
    # Current clients, Core equations and actual Prompt/Registry are all pinned.
    from scripts.run_u1_voi_expansion import load as load_a
    _,_,a,_=load_a();pins={**a['pins'],**{str(f.resolve()):sha_file(f) for f in set(files)}}
    p={'plan_id':NAME,'run_id':'run_'+NAME,'scope':'U2_SOURCE_STATIC_FIVE_AGENT_COLLECTION_ONLY','snapshot_kind':'NATIVE_SHAPED_STATIC_SOURCE_NOT_NATIVE_EPISODE',
       'registered_patients':10,'agents':agents,'registry':registry,'config':cfg,'catalog':catalog,'prompt_fingerprints':ready['prepared']['prompt_fingerprints'],
       'generation_total':50,'calls_per_patient_agent':1,'retry':0,'repair':0,'warmup':0,'wall_seconds':1800,'process_bytes':4*1024**3,'output_bytes':1024**3,
       'min_samples_for_future_fitting':2,'fitting_in_this_plan':False,'U2_artifact_created':False,'actual_collected_complete_groups':0,
       'Source_reference_policy':'single current Encounter Source label, not future date, same-day intraday unreviewed','training_validation':'TRAIN10_ONLY_VALIDATION_NOT_AVAILABLE',
       'Source_gap':'No authored chief complaint or symptoms added; D2 semantic and Native sequential-task shift remain','pins':pins,**MARKERS}
    p['fingerprint']=digest(p);atomic_json(pub/'plan.json',p)
    approval={**MARKERS,'scope':p['scope'],'plan_fingerprint':p['fingerprint'],'max_generation':50,'operator_approved':False,'no_other_ollama_work':False,
        'accept_static_predisclosure_proxy':False,'accept_source_reference_date_proxy':False,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None,'clinical_release_approved':False}
    atomic_json(pr/'approval_template.json',approval)
    report={'patients':10,'reserved_visits':len(reservations),'generation_cap':50,'bank_patients':len({b['training_patient_id'] for b in bank}),'bank_visits':len(bank),
        'excluded_reasons':dict(reasons),'original50_overlap':0,'old_routing_reservation_overlap':0,'all_Source_CSV_rows_verified':True,
        'explicit_chief_complaints':sum(x['chief_complaint_present'] for x in ledger),'actual_Native_acquisition_executed':False,
        'max_input_bound':max(bounds),'future_U2_admission':'BLOCKED_UNTIL_ACTUAL_COLLECTION_FIT_AND_TRANSFER_VALIDATION','clinical_accuracy':'NOT_EVALUATED'}
    atomic_json(pub/'source_readiness.json',report);return {'status':'SOURCE_PROXY_PLAN_PREPARED_NOT_COLLECTED','plan_id':NAME,'fingerprint':p['fingerprint'],'generation_cap':50,**report}

def load():
    pub,pr=PUB/NAME,PR/NAME;p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('PIN_CHANGED')
    return pub,pr,p

def check():
    pub,pr,p=load();blocks=[]
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_NEW_SOURCE_SNAPSHOT_APPROVAL_REQUIRED','blocker_codes':blocks,'plan_id':NAME,'fingerprint':p['fingerprint'],'generation_cap':50,'actual_LLM_generation':0,'Native_episode_contract':'NOT_CLAIMED'}

def permission(a,p):
    if any(a.get(k)!=v for k,v in MARKERS.items()) or a.get('clinical_release_approved') is not False:raise ValueError('DEVELOPMENT_ONLY')
    if a.get('scope')!=p['scope'] or a.get('plan_fingerprint')!=p['fingerprint'] or a.get('max_generation')!=50:raise ValueError('EXACT_NEW_SOURCE_APPROVAL_REQUIRED')
    if any(a.get(k) is not True for k in ('operator_approved','no_other_ollama_work','accept_static_predisclosure_proxy','accept_source_reference_date_proxy')):raise ValueError('ACTUAL_STATIC_PROXY_OPERATOR_APPROVAL_REQUIRED')
    if not all(isinstance(a.get(k),str) and a[k].strip() for k in ('operator_ref','user_statement')):raise ValueError('ACTUAL_OPERATOR_STATEMENT')
    issued=datetime.fromisoformat(a['recorded_at']);expiry=datetime.fromisoformat(a['expires_at']);now=datetime.now(timezone.utc)
    if issued.tzinfo is None or expiry.tzinfo is None or not issued<=now<expiry or not 0<(expiry-issued).total_seconds()<=86400:raise ValueError('ACTUAL_APPROVAL_TIME')

def collect(approval):
    if check()['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p=load();a=read(approval);permission(a,p);run=pr/p['run_id'];run.mkdir(exist_ok=False);atomic_json(run/'approval.json',a);atomic_json(run/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'plan_fingerprint':p['fingerprint'],'TEST_ONLY':False})
    started=time.monotonic();rows=[];stop=None;stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if time.monotonic()-started>=1800 or _process_private_bytes()>=p['process_bytes'] or sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('GLOBAL_RESOURCE_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stats.items()):raise ValueError('SOURCE_OR_CODE_CHANGED')
    budget=GenerationBudget(50,1,resource_guard=guard)
    with (run/'responses.jsonl').open('x',encoding='utf-8') as stream:
        for case in read(pr/'candidates.json'):
            for agent in p['agents']:
                r={'ordinal':case['ordinal'],'agent_id':agent['agent_id'],'department':agent['department'],'shared_fingerprint':case['shared_fingerprint'],'status':'NOT_ATTEMPTED','generation_attempts':0,'response':None}
                if stop is None:
                    before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                    try:
                        guard();folder=run/('Patient%03d'%case['ordinal'])/agent['department'];client=BudgetedNativeERClient(p['config'],authorized=True,contract_fingerprint=p['fingerprint'],checkpoint_root=folder)
                        text=client.chat_json(specialty_system_prompt(agent['department']),json.dumps({'department':agent['department'],**case['shared']},sort_keys=True),p['config']['model'],schema=response_schema(p['catalog']))
                        response=json.loads(text);r.update(response=response,response_fingerprint=digest(response),status='RESPONSE_RECORDED');guard()
                    except Exception as e:r.update(status='REQUEST_FAILED',error_type=type(e).__name__,private_error=str(e));stop='FIRST_ERROR_NO_RETRY'
                    finally:r['generation_attempts']=budget.used-before;_BUDGET.reset(token)
                rows.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
    c={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,'plan_fingerprint':p['fingerprint'],'wall_seconds':time.monotonic()-started,'TEST_ONLY':False,**MARKERS};atomic_json(run/'collection.json',c);atomic_json(pub/'execution_status.json',{k:v for k,v in c.items() if k!='rows'});return c['status']

def analyze():
    from jsonschema import Draft202012Validator
    pub,pr,p=load();run=pr/p['run_id'];c=read(run/'collection.json');rows=[];groups=[];refs={r['ordinal']:r for r in read(pr/'reference_private.json')};cases={r['ordinal']:r for r in read(pr/'candidates.json')}
    if len(c['rows'])!=50 or c['plan_fingerprint']!=p['fingerprint'] or c['generation']!=sum(r['generation_attempts'] for r in c['rows']):raise ValueError('COLLECTION_COUNTER_MISMATCH')
    agents={a['agent_id']:a for a in p['agents']};expected={(i,aid) for i in cases for aid in agents}
    seen=set()
    for r in c['rows']:
        key=r['ordinal'],r['agent_id']
        if key in seen or key not in expected or r['department']!=agents[r['agent_id']]['department']:raise ValueError('DUPLICATE_OR_UNREGISTERED_RESPONSE')
        seen.add(key);correct=None
        if r['status']=='RESPONSE_RECORDED':
            folder=run/('Patient%03d'%r['ordinal'])/r['department'];b=read(folder/'Request0001_before.json');a=read(folder/'Request0001_after.json');request=b['request'];response=r['response'];case=cases[r['ordinal']]
            if r['shared_fingerprint']!=case['shared_fingerprint'] or digest(request)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or a['status']!='VALID' or a['HTTP_status']!=200 or digest(response)!=r['response_fingerprint'] or response!=json.loads(a['response_envelope']['message']['content']):raise ValueError('RAW_RESPONSE_LINEAGE_MISMATCH')
            if request['messages']!=[{'role':'system','content':specialty_system_prompt(r['department'])},{'role':'user','content':json.dumps({'department':r['department'],**case['shared']},sort_keys=True)}] or request['format']!=response_schema(p['catalog']):raise ValueError('ACTUAL_SHARED_INPUT_CHANGED')
            from app.locdi.llm_setup import normalize_model_identifier
            if b['contract']!=p['fingerprint'] or request['model']!=p['config']['model'] or request['options']!={'num_ctx':32768,'temperature':0,'seed':42,'num_predict':1024}:raise ValueError('ACTUAL_MODEL_OPTIONS_CHANGED')
            if any(normalize_model_identifier(identity['digest'])!=p['config']['model_identifier'] for identity in (b['identity_before'],a['identity_after'])) or a['identity_after']['context_length']!=32768:raise ValueError('ACTUAL_MODEL_CONTEXT_MISMATCH')
            Draft202012Validator(request['format']).validate(response)
            if response['certainty']=='supported':correct=response['diagnosis']==refs[r['ordinal']]['label']
        rows.append({**r,'correctness_proxy':correct})
    if seen!=expected:raise ValueError('MISSING_REGISTERED_RESPONSE_SLOT')
    for i in range(1,11):
        group=[r for r in rows if r['ordinal']==i];groups.append({'ordinal':i,'complete_collection':len(group)==5 and all(r['status']=='RESPONSE_RECORDED' for r in group),'complete_proxy_group':len(group)==5 and all(type(r['correctness_proxy']) is bool for r in group)})
    report={'complete_collection_groups':sum(g['complete_collection'] for g in groups),'complete_proxy_groups':sum(g['complete_proxy_group'] for g in groups),'by_agent':{a['department']:dict(Counter(str(r['correctness_proxy']) for r in rows if r['agent_id']==a['agent_id'])) for a in p['agents']},'U2_artifact_created':False,'fitting':0,'new_generation':0,'Native_episode_success_estimate':'NOT_VALIDATED',**MARKERS}
    write_equal(run/'analysis_private.json',{'rows':rows,'groups':groups});write_equal(pub/'saved_results.json',report);return report

def check_future_u2():return {'status':'BLOCKED','blocker_codes':['U2_NOT_COLLECTED_OR_FITTED','U2_FIT_LINEAGE_AND_NATIVE_TRANSFER_ADMISSION_NOT_VALIDATED'],'reserved_plan_id':'u2_d2_b34_future_001','executable_plan_created':False,'generation_cap':0}

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','collect','analyze','check-u2-evaluation']);q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');a=q.parse_args()
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check()
    elif a.mode=='analyze':out=analyze()
    elif a.mode=='check-u2-evaluation':out=check_future_u2()
    else:
        if not a.execute or not a.approval:q.error('NEW_USER_COLLECTION_APPROVAL_REQUIRED')
        out={'status':collect(a.approval)}
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
if __name__=='__main__':main()
