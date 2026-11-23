"""Original50 D2 execution: B012 ready separately; B34 fail-closed admission.

Default modes never contact Ollama. Only execute + explicit flags + fresh
operator approval can open an inference transport. No fitting implementation.
"""
from pathlib import Path
from contextlib import ExitStack
from collections import Counter
from datetime import datetime,timezone
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_original50_final import PUB,PR
from app.locdi.original_source_inventory import read,sha_file,original_cases,DATA
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collector import atomic_json
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes,features_for
from app.locdi.augmentation_experiment_backend import make_paired_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from scripts.run_original50_d0_d2 import AUG,M5

STAGES={'Smoke5':5,'Verify10':10,'Full50':50}
CONDITIONS=['B0','B1','B2']
BLOCKERS=['NO_ADMITTED_ROUTING_UTILITY','U2_NATIVE_TRAIN_FEATURES_EMPTY',
          'U2_VALIDATION_OVERLAPS_M5_TRAIN','NATIVE_TRANSFER_ADMISSION_NOT_INSTALLED']

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:raise FileExistsError('EXISTING_RESULT_DIFFERS_NO_OVERWRITE')
    else:atomic_json(path,value)

def identity(stage,scope='B012'):
    if stage not in STAGES or scope not in ('B012','B034'):raise ValueError('STAGE_SCOPE')
    return 'o50_d2_'+stage.lower()+'_'+scope.lower()+'_002'

def load(stage,scope='B012'):
    name=identity(stage,scope);pub=PUB/name;pr=PR/name;p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('PIN_CHANGED: '+str(f))
    return pub,pr,p

def feature_state():
    from scripts.run_u2_native_collection import load as native_load
    _,upr,_=native_load('Full')
    # Inspect actual frozen source observations, not prior report flags.
    contexts=read(upr/'patients.json');from app.core.routing_utility import observed_tokens
    rows=[]
    for row in contexts:
        if row['split']!='train':continue
        # Native source stores the same initial public state in its Case.
        state=row['case']['agent_visible']['current_initial_state']
        obs={'chief_complaint':state.get('doctor_visible_info',{}).get('chief_complaint',''),
             'current_symptoms':state.get('patient_visible_info',{}).get('associated_symptoms',[])}
        rows.append(len(observed_tokens(obs)))
    f=ROOT/'outputs/private/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/u2_native_development_fit_001/U2_native_development_utility.json'
    return {'U2_artifact_exists':f.exists(),'U2_artifact_sha256':sha_file(f) if f.exists() else None,
            'initial_train_token_counts':rows,'U2_transfer_admitted':False,'U3_admitted':False,
            'automatic_utility_selection':False,'blocker_codes':BLOCKERS}

def prepare():
    if (PUB/'execution_preparation_002.json').exists():raise FileExistsError('PREPARATION_EXISTS_USE_CHECK')
    data=read(PUB/'data_readiness.json')
    if data['Source_runtime_ready']!=50:raise ValueError('FRESH_SOURCE50_REQUIRED')
    report=feature_state();write_new(PUB/'routing_admission_current.json',report)
    # Original stage ordering was fixed by patient hash without correctness.
    oldtasks=read(PR/'original50_d2_full50_001/tasks.json')
    order=list(dict.fromkeys(t['original_ordinal'] for t in oldtasks))
    histories=PR/'TEST_ONLY_full50/history_selection_plans.json'
    if len(read(histories))!=50:raise ValueError('HISTORY_PLANS50_REQUIRED')
    ready=validate_setup(read(PR/'config.json'))
    if ready['static_readiness']!='READY':raise ValueError('STATIC_MODEL_REGISTRY_PROMPT')
    # Pin every used native implementation, original prompts, sources and audits.
    files=set()
    for folder in ('app/core','app/locdi','validation','prompts'):
        files.update(f for f in (ROOT/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.json','.txt','.md'))
    files.update(Path(f) for f in read(PR/'protection_before.json'))
    files.update((PR/'bindings50.json',PR/'case_audit_private.json',PR/'config.json',histories,
                  PUB/'data_readiness.json',PUB/'routing_admission_current.json',AUG/'AUGMENTED_CASES.jsonl',DATA,M5))
    files.update(ROOT/f for f in ('scripts/run_original50_final.py','scripts/Invoke-LoCDIOriginal50Final.ps1',
                                  'scripts/prepare_original50_final.py','scripts/analyze_original50_d0_d2.py',
                                  'scripts/probe_original50_full_offline.py','scripts/probe_original50_final_runner.py'))
    pins={str(f.resolve()):sha_file(f) for f in files}
    old={r['ordinal'] for r in data['rows'] if r['old_development10']}
    common={'source':sha_file(DATA),'D2':sha_file(AUG/'AUGMENTED_CASES.jsonl'),
            'config':digest(read(PR/'config.json')),'bindings':digest(read(PR/'bindings50.json')),
            'history':digest(read(histories)),'utility':None,'conditions':CONDITIONS,'pins':pins}
    for stage,n in STAGES.items():
        for scope,conds in (('B012',CONDITIONS),('B034',['B0','B1','B2','B3','B4'])):
            name=identity(stage,scope);pub=PUB/name;pr=PR/name
            if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY')
            tasks=[{'episode_ordinal':j+1,'original_ordinal':o,'case_ref':'Original%03d'%o,'condition':c,
                    'old_development10':o in old}
                   for j,(o,c) in enumerate((o,c) for o in order[:n] for c in conds)]
            write_new(pr/'tasks.json',tasks)
            p={'version':'original50.final.d2.v1','plan_id':name,'run_id':'run_'+name,'stage':stage,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY',
               'execution_scope':scope,'conditions':conds,'registered_patients':n,'registered_episodes':len(tasks),
               'dataset_view':'D2','D0_validated_but_not_executed':True,'paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY',
               'dataset_views':['D0','D2'],'backend_view_capability_not_execution_scope':True,
               'generation_per_episode':12,'generation_total':len(tasks)*12,'B4_terminal_in_episode_cap':True,
               'context':32768,'output_tokens':1024,'timeout_seconds':120,'temperature':0,'seed':42,
               'model':read(PR/'config.json')['model'],'model_identifier':read(PR/'config.json')['model_identifier'],
               'wall_seconds':1800 if n==5 else 3600 if n==10 else 14400,'process_bytes':4*1024**3,'output_bytes':4*1024**3,
               'retry':0,'repair':0,'warmup':0,'no_resume':True,'selection':'FROZEN_PATIENT_HASH_SEED42_NO_CORRECTNESS',
               'comparison_fingerprint':digest(common),'utility_sha256':None,'utility_selection':'NOT_ADMITTED',
               'B1':'CONTEXT_BUDGETED_HISTORY_NOT_FULL_HISTORY','B2':'ORIGINAL_SELECTIVE_TOP5',
               'promotion_policy':'TECHNICAL_ONLY_SMOKE_HEADROOM1024_AND_MAX11_CALLS_ELSE_VERIFY10',
               'blocker_codes':BLOCKERS if scope=='B034' else [],'clinically_reviewed':False,
               'pins':{**pins,str((pr/'tasks.json').resolve()):sha_file(pr/'tasks.json')},**MARKERS}
            p['fingerprint']=digest(p);write_new(pub/'plan.json',p)
            a={**MARKERS,**{k:p[k] for k in ('scope','paired_scope','dataset_views','conditions')},
               'plan_fingerprint':p['fingerprint'],'max_generation':p['generation_total'],'execution_scope':scope,'dataset_view':'D2',
               'operator_approved':False,'no_other_ollama_work':False,'clinical_release_approved':False,
               'accept_source_label_proxy':False,'accept_recorded_time_proxy':False,'accept_common_m5_menu':False,
               'accept_label_conditioned_development':False,'accept_augmentation_initial_time_policy':False,
               'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
            write_new(pr/'approval_template.json',a)
    out={'status':'B012_STAGED_B34_BLOCKED','actual_generation':0,'actual_fitting':0,'Source_ready_patients':50,
         'Fake250':'REQUIRED_AT_PREFLIGHT_TEST_ONLY_NOT_ADMISSION','utility_admitted':False,'B34_blockers':BLOCKERS}
    write_new(PUB/'execution_preparation_002.json',out);return out

def permission(a,p):
    from scripts.run_original50_d0_d2 import permission as paired_permission
    paired_permission(a,p)
    if a.get('execution_scope')!=p['execution_scope'] or a.get('dataset_view')!='D2':raise ValueError('EXACT_FINAL_SCOPE_APPROVAL_REQUIRED')

def technical_proof(stage):
    pub,pr,p=load(stage);run=pr/p['run_id'];c=read(run/'collection.json');tasks=read(pr/'tasks.json')
    if c.get('TEST_ONLY') is not False or c.get('status')!='COMPLETED' or c['plan_fingerprint']!=p['fingerprint']:raise ValueError('REAL_COMPLETED_PRECEDING_STAGE_REQUIRED')
    lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if lines!=c['rows'] or len(lines)!=len(tasks) or sum(r['request_count'] for r in lines)!=c['generation']:raise ValueError('PROMOTION_COLLECTION_COUNTS')
    headroom=32768;maxcalls=0
    for row,task in zip(lines,tasks):
        if any(row.get(k)!=v for k,v in task.items()) or row['status']!='COMPLETED' or not 0<row['request_count']<=12:raise ValueError('PROMOTION_TASK_NOT_COMPLETE')
        folder=run/'requests'/('Episode%04d'%row['episode_ordinal'])
        bb=sorted(folder.glob('Request*_before.json'));aa=sorted(folder.glob('Request*_after.json'))
        if len(bb)!=len(aa) or len(bb)!=row['request_count']:raise ValueError('PROMOTION_CHECKPOINT_COUNT')
        for bf,af in zip(bb,aa):
            b=read(bf);a=read(af);request=b['request']
            if digest(request)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or b['contract']!=p['fingerprint']:raise ValueError('PROMOTION_REQUEST_LINK')
            if a.get('status')!='VALID' or a.get('HTTP_status')!=200 or a['identity_after']['context_length']!=32768:raise ValueError('PROMOTION_REAL_RESPONSE_IDENTITY')
            from app.locdi.llm_setup import normalize_model_identifier
            if normalize_model_identifier(a['identity_after']['digest'])!=p['model_identifier']:raise ValueError('PROMOTION_DIGEST')
            headroom=min(headroom,32768-1024-512-b['guard']['conservative_input_token_bound'])
        maxcalls=max(maxcalls,row['request_count'])
        audit=read(folder/'dataset_view_audit.json')
        if audit['view']!='D2':raise ValueError('PROMOTION_D2_VIEW')
        for file in ('actual_core_decisions.json','raw_runner_result.json','episode_and_policy.json'):
            if not (folder/file).exists():raise ValueError('PROMOTION_MISSING_EXPORT')
    return {'stage':stage,'comparison_fingerprint':p['comparison_fingerprint'],'minimum_input_headroom':headroom,
            'max_calls_per_episode':maxcalls,'Verify10_skip_allowed':headroom>=1024 and maxcalls<=11,
            'collection_sha256':sha_file(run/'collection.json'),'accuracy_used_in_decision':False}

def check(stage,scope='B012'):
    _,pr,p=load(stage,scope);blocks=list(p['blocker_codes']);promotion=None
    for file,count in (('TEST_ONLY_full50.json',250),('TEST_ONLY_runner150.json',150)):
        path=PUB/file
        if not path.exists() or read(path).get('status')!='PASS' or read(path).get('registered_episodes')!=count:
            blocks.append('FRESH_'+str(count)+'_FAKE_CORE_REQUIRED')
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    if scope=='B012' and stage!='Smoke5':
        try:
            promotion=technical_proof('Smoke5')
            if stage=='Full50' and not promotion['Verify10_skip_allowed']:promotion=technical_proof('Verify10')
            if promotion['comparison_fingerprint']!=p['comparison_fingerprint']:raise ValueError('STAGE_CONTRACT_CHANGED')
        except (ValueError,FileNotFoundError,KeyError):blocks.append('PRECEDING_REAL_TECHNICAL_STAGE_REQUIRED')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_USER_APPROVAL_REQUIRED','stage':stage,'plan_id':p['plan_id'],
            'registered_patients':p['registered_patients'],'registered_episodes':p['registered_episodes'],
            'generation_cap':p['generation_total'],'wall_seconds':p['wall_seconds'],'blocker_codes':blocks,'promotion':promotion,
            'new_generation':0,'model_server_context':'NOT_CHECKED_NO_INFERENCE','clinical_accuracy':'NOT_EVALUATED'}

def execute(stage,approval,*,test_only=False):
    if check(stage)['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p=load(stage);a=read(approval);permission(a,p);run=pr/p['run_id']
    if test_only and 'TEST_ONLY_RUNNER' not in pr.parts:raise ValueError('TEST_ONLY_SEPARATE_NAMESPACE_REQUIRED')
    run.mkdir(exist_ok=False)
    atomic_json(run/'approval.json',a);atomic_json(run/'started.json',{'plan_fingerprint':p['fingerprint'],'TEST_ONLY':test_only,'at':datetime.now(timezone.utc).isoformat()})
    cases=original_cases();tasks=read(pr/'tasks.json');ordinals=set(t['original_ordinal'] for t in tasks)
    wrappers={'Original%03d'%i:cases[i-1] for i in ordinals}
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    begin=time.monotonic();stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if time.monotonic()-begin>=p['wall_seconds'] or _process_private_bytes()>=p['process_bytes']:raise ValueError('GLOBAL_WALL_OR_MEMORY_BUDGET')
        if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('GLOBAL_OUTPUT_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stats.items()):raise ValueError('PIN_CHANGED_DURING_RUN')
    budget=GenerationBudget(p['generation_total'],12,resource_guard=guard);stop=None;rows=[];backend=None
    try:
        backend=make_paired_backend(wrappers,dataset_view='D2',augmented_rows={r:aug[int(r[-3:])] for r in wrappers},catalog=evidence_catalog(),
            config=read(PR/'config.json'),bindings=read(PR/'bindings50.json'),model=read(M5),authorization=a,
            plan_fingerprint=p['fingerprint'],checkpoint_root=run/'requests',routing_evidence=None,
            history_policy=ContextBudgetedHistory(read(PR/'TEST_ONLY_full50/history_selection_plans.json')),
            experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
    except Exception as e:
        stop='BACKEND_SETUP_FAILED';atomic_json(run/'setup_failure.json',{'error_type':type(e).__name__,'private_error':str(e)})
    with ExitStack() as safety, (run/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        if test_only:
            from unittest.mock import patch
            safety.enter_context(patch('socket.socket.connect',side_effect=AssertionError('TEST_ONLY_NETWORK_DENIED')))
            safety.enter_context(patch('socket.create_connection',side_effect=AssertionError('TEST_ONLY_NETWORK_DENIED')))
        for task in tasks:
            r={**task,'status':'NOT_ATTEMPTED','request_count':0,'result':None};before=budget.used
            if stop is None:
                budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard();result=backend({'case_ref':task['case_ref']},features_for(task['condition']));r.update(result=result,status=result['status'].upper())
                    if result['request_count']!=budget.used-before:raise ValueError('REQUEST_COUNT_MISMATCH')
                    guard()
                    if r['status']!='COMPLETED':stop='FIRST_NONCOMPLETED_EPISODE'
                except Exception as e:r.update(status='FAILED',error_type=type(e).__name__,private_error=str(e));stop='FIRST_TECHNICAL_FAILURE'
                finally:_BUDGET.reset(token)
            r.update(request_count=budget.used-before,stop_reason=stop);rows.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
    out={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,
         'wall_seconds':time.monotonic()-begin,'plan_fingerprint':p['fingerprint'],'stop_reason':stop,'TEST_ONLY':test_only,**MARKERS}
    atomic_json(run/'collection.json',out);atomic_json(pub/'execution_status.json',{k:v for k,v in out.items() if k!='rows'})
    load(stage);return {k:v for k,v in out.items() if k!='rows'}

def analyze(stage):
    from scripts.analyze_original50_d0_d2 import facts
    from app.locdi.full50_evaluation_metrics import aggregate_by_exposure
    pub,pr,p=load(stage);run=pr/p['run_id'];c=read(run/'collection.json');tasks=read(pr/'tasks.json')
    lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if c['rows']!=lines or len(lines)!=len(tasks) or c['plan_fingerprint']!=p['fingerprint'] or sum(r['request_count'] for r in lines)!=c['generation']:raise ValueError('TRACE_COUNTS_OR_PLAN')
    sources={r['summary']['ordinal']:r for r in read(PR/'case_audit_private.json')};private=[];metrics=[]
    originals=original_cases()
    from app.locdi.original_source_inventory import eligible
    for r,t in zip(lines,tasks):
        if any(r.get(k)!=v for k,v in t.items()):raise ValueError('TASK_MISMATCH')
        folder=run/'requests'/('Episode%04d'%r['episode_ordinal']);f=facts(folder,r['result'],sources[r['original_ordinal']]['reference'])
        blocked_before_generation=sum(read(x).get('status')=='BLOCKED_BEFORE_GENERATION' and read(x).get('generation_attempted') is False
                                      for x in folder.glob('Request*_after.json'))
        if f['request_count_checkpoint']!=r['request_count']+blocked_before_generation:raise ValueError('CHECKPOINT_COUNT')
        f['blocked_before_generation_count']=blocked_before_generation
        if r['status']!='COMPLETED':
            for key in ('source_exact','normalized','department_match','algorithmic_source_exact'):f[key]=None
        private.append({'task':r,'facts':f})
        before=[read(x) for x in sorted(folder.glob('Request*_before.json'))]
        bodies=[json.loads(x['request']['messages'][1]['content']) for x in before]
        roles=Counter(b.get('role','UNKNOWN')+':'+b.get('department','') for b in bodies)
        identity=originals[r['original_ordinal']-1]['agent_visible']['longitudinal']
        cutoff_bad=sum(not eligible(record,identity) for body in bodies
                       for record in body['observation'].get('longitudinal_memory',{}).get('records',[]))
        metrics.append({'ordinal':r['original_ordinal'],'condition':r['condition'],'status':r['status'],'plan_fingerprint':p['fingerprint'],
            'source_exact':f['source_exact'],'source_department':f['department_match'],'normalized':f['normalized'],
            'algorithmic_source_exact':f['algorithmic_source_exact'] if r['condition']=='B4' else None,
            'certainty':f['prediction']['certainty'],'generation':r['request_count'],'usage':f['usage'],'role_counts':dict(roles),
            'action_counts':(r['result'] or {}).get('action_counts',{}),
            'wall_seconds':(r['result'] or {}).get('computation',{}).get('wall_seconds',0),
            'routing_calls':len(f['routing_trace']),'routing_selected':sum(x.get('decision_status')=='selected' for x in f['routing_trace']),
            'routing_abstained':sum(x.get('decision_status')=='abstained' for x in f['routing_trace']),
            'memory_selected_count':len(f['last_memory']),'memory_cutoff_violations':cutoff_bad,
            'current_evidence_count':len(f['released_evidence']),'STOP':len(f['voi_trace'])})
    ordinals=list(dict.fromkeys(t['original_ordinal'] for t in tasks));old={t['original_ordinal'] for t in tasks if t['old_development10']}
    aggregate=aggregate_by_exposure(metrics,ordinals,old,plan_fingerprint=p['fingerprint'])
    # B34 were not registered in this partial contract: never call them executed/failed.
    for stratum in aggregate.values():
        if 'conditions' not in stratum:continue
        for condition in ('B3','B4'):stratum['conditions'][condition]={'status':'BLOCKED_NOT_REGISTERED_IN_B012_CONTRACT','blocker_codes':BLOCKERS}
        stratum['pairs']=[]
    transitions=[];lookup={(r['ordinal'],r['condition']):r for r in metrics}
    for o in ordinals:
        for left,right in (('B0','B1'),('B0','B2'),('B1','B2')):
            x,y=lookup[o,left],lookup[o,right]
            tag='INCOMPLETE_PAIR' if x['status']!=y['status'] or x['status']!='COMPLETED' else ('CORRECT_RETAINED' if y['source_exact'] is True else 'CORRECT_LOST') if x['source_exact'] is True else ('BECAME_CORRECT' if y['source_exact'] is True else 'NO_CORRECT_BOTH')
            transitions.append({'ordinal':o,'comparison':left+'_'+right,'transition':tag})
    write_new(run/'analysis_private.json',{'plan_fingerprint':p['fingerprint'],'rows':private,'metrics':metrics,'transitions':transitions})
    normalized={cond:{'registered':len(ordinals),'normalized_success':sum(r['normalized'] is True for r in metrics if r['condition']==cond)} for cond in CONDITIONS}
    report={'status':'SAVED_TRACE_ANALYZED','stage':stage,'plan_fingerprint':p['fingerprint'],'registered_patients':len(ordinals),
            'registered_episodes':len(tasks),'generation':c['generation'],'TEST_ONLY':c['TEST_ONLY'],'strata':aggregate,'normalized':normalized,
            'paired_transitions':transitions,'existing_results_merged':False,'D2_label_conditioned':True,
            'Source_Exact_primary':'SUCCESS_COUNT/REGISTERED_PATIENTS','whole50_independent_holdout':False,**MARKERS}
    write_new(pub/'saved_results.json',report);return {'status':report['status'],'new_generation':0,'generation':c['generation']}

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze','promotion']);q.add_argument('--stage',choices=STAGES,default='Smoke5');q.add_argument('--all-conditions',action='store_true');q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');q.add_argument('--allow-llm',action='store_true');a=q.parse_args()
    if a.mode!='execute' and (a.execute or a.allow_llm or a.approval):q.error('EXECUTE_FLAGS_ONLY')
    if a.mode!='check' and a.all_conditions:q.error('B34_ADMISSION_BLOCKED_NO_EXECUTION')
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check(a.stage,'B034' if a.all_conditions else 'B012')
    elif a.mode=='analyze':out=analyze(a.stage)
    elif a.mode=='promotion':out=technical_proof(a.stage)
    else:
        if not(a.execute and a.allow_llm and a.approval):q.error('EXACT_USER_APPROVAL_EXECUTE_ALLOW_LLM_REQUIRED')
        out=execute(a.stage,a.approval)
    print(json.dumps(out,ensure_ascii=True))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)

if __name__=='__main__':main()
