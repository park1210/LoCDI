"""Pinned matched D0/D2 B0..B4 development execution. Default is no generation."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,re,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_common_menu_b0_b4 import prepare as base_prepare,load_plan as base_load,preflight as base_preflight,M5,DATA,OLD
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.original50_augmentation import validate_augmentation,initial_preview,SCOPE
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.augmentation_experiment_backend import make_paired_backend,POLICY
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes,features_for,_no_reference
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.routing_collector import atomic_json
from app.locdi.synthea_diagnosis_audit import matches

NS='ORIGINAL50_D0_D2_EXPERIMENT_v1';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
AUG=ROOT/'outputs/private/research_evaluation/ORIGINAL50_LITERATURE_AUGMENTATION_v1'
UTILITY=ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/development_routing_evidence.json'
CONDITIONS=['B0','B1','B2','B3','B4'];VIEWS=['D0','D2']
CODE=['scripts/run_original50_d0_d2.py','scripts/Invoke-LoCDIOriginal50Augmentation.ps1',
    'scripts/analyze_original50_d0_d2.py','app/locdi/augmentation_experiment_backend.py',
    'app/locdi/original50_augmentation.py','app/locdi/literature_augmentation_catalog.py',
    'app/locdi/evidence_subset_wire.py']
SMOKE='original50_d0d2_smoke_005';FULL='original50_d0d2_full_003'

def paths(name):
    if not re.fullmatch('[A-Za-z0-9_]{1,50}',name):raise ValueError('NEW_SAFE_PLAN_ID')
    return PUB/name,PR/name
def write_equal(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError('EXISTING_ARTIFACT_DIFFERS_NO_OVERWRITE')
    else:atomic_json(path,value)
def tasks(panel,views=VIEWS):
    rows=[];view_counts=Counter()
    for i,patient in enumerate(panel,1):
        order=views if i%2 else list(reversed(views))
        for condition in CONDITIONS:
            for view in order:
                view_counts[view]+=1
                rows.append({'episode_ordinal':len(rows)+1,'view_episode_ordinal':view_counts[view],
                    'patient_ordinal':i,'case_ref':patient['case_ref'],'condition':condition,'dataset_view':view})
    return rows

def prepare(name,*,stage='Smoke',smoke_id=SMOKE):
    pub,pr=paths(name)
    if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY_NO_OVERWRITE')
    if stage not in ('Smoke','Full'):raise ValueError('STAGE')
    proof=smoke_proof(smoke_id) if stage=='Full' else None
    patients=1 if stage=='Smoke' else 10;base_id=name+'_source'
    base_prepare(base_id,CONDITIONS,UTILITY,'DEVELOPMENT_EXPERIMENT',patients,True,True,None,False,False,True)
    _,bp,base=base_load(base_id)
    readiness=base_preflight(base_id)
    if readiness['status']=='BLOCKED':raise ValueError('UNCHANGED_SOURCE_UTILITY_PREFLIGHT_BLOCKED: '+','.join(readiness['blocker_codes']))
    cases=original_cases();index=read(bp/'case_index.json');panel=read(bp/'panel.json');catalog=evidence_catalog()
    original_fixed=read(OLD/'case_index.json');fixed_panel=read(OLD/'panel.json')
    if panel!=fixed_panel[:patients] or index!={k:v for k,v in original_fixed.items() if k in {p['case_ref'] for p in panel}}:raise ValueError('FIXED_PATIENT_PANEL_CHANGED')
    all_aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    if sha_file(AUG/'AUGMENTED_CASES.jsonl')!=read(AUG.parent.parent.parent/'research_evaluation/ORIGINAL50_LITERATURE_AUGMENTATION_v1/completion_summary.json')['data_sha256']:raise ValueError('AUGMENTATION_DATASET_CHANGED')
    pr.mkdir(parents=True);pub.mkdir(parents=True);data_audit=[]
    for ref,n in index.items():
        c=cases[n-1];r=all_aug[n];validate_augmentation(c,r,catalog,verify_source_files=True)
        if any(item['available_minute']!=0 for item in [r['literature_informed_synthetic']['chief_complaint'],*r['literature_informed_synthetic']['symptoms']]):raise ValueError('UNSUPPORTED_NONINITIAL_SYNTHETIC_RELEASE')
        if r['literature_informed_synthetic']['synthetic_test_results']:raise ValueError('NEW_TEST_RESULT_CONTRACT_NOT_ALLOWED')
        d0=initial_preview(c,r,catalog);d2=initial_preview(c,r,catalog,view='D2',accept_development_proxy=True)
        data_audit.append({'case_ref':ref,'original_ordinal':n,'original_case_fp':digest(c),'D0_initial_fp':digest(d0),'D2_initial_fp':digest(d2),
            'row_fp':r['row_fingerprint'],'source_reference_timing':r['source_reference']['timing'],'clinical_reviewed':False,
            'source_evidence_changed':False,'synthetic_release_minute':0,'synthetic_objective_evidence':False})
    atomic_json(pr/'dataset_audit.json',data_audit);atomic_json(pr/'tasks.json',tasks(panel))
    files=[ROOT/n for n in CODE]+[AUG/'AUGMENTED_CASES.jsonl',pr/'dataset_audit.json',pr/'tasks.json',UTILITY]
    # Exact Utility lineage also remains immutable during this experiment.
    files += [Path(n) for n in read(UTILITY)['origin_files']]
    if proof:files += [Path(n) for n in proof['pins']]
    plan={'version':'original50.matched_d0_d2.v1','plan_id':name,'run_id':'run_'+name,'stage':stage,'base_source_plan':base_id,'base_fingerprint':base['fingerprint'],
        'scope':base['scope'],'paired_scope':'ORIGINAL50_D0_D2_DEVELOPMENT_ONLY','dataset_views':VIEWS,'conditions':CONDITIONS,
        'registered_patients':patients,'registered_episodes':patients*10,'generation_per_episode':12,'generation_total':patients*120,
        'model':read(bp/'llm_config.json')['model'],'model_identifier':base['model_identifier'],'context':32768,'output_tokens':1024,'timeout_seconds':120,
        'temperature':0,'seed':42,'retry':0,'repair':0,'warmup':0,'no_resume':True,'source_policy':base['source_policy'],'augmentation_policy':POLICY,
        'dataset_versions':{'D0':'Original50 exact '+sha_file(DATA),'D2':'original50_augmentation.v1 '+sha_file(AUG/'AUGMENTED_CASES.jsonl')},
        'synthetic_release':'INITIAL_MINUTE_0_SUBJECTIVE_ONLY','M5_menu':'EXISTING_8_SOURCE_ACTIONS_NO_ADDITIONS','frozen_m5_sha256':sha_file(M5),
        'catalog':'SAME_EXISTING_15_IN_BOTH_VIEWS','prompt_change':False,'output_instruction_change':'evidence_subset.v3 only; same in D0 and D2',
        'evidence_wire':'evidence_citation_subset_code.v3 + memory_citation_subset_code.v1',
        'history_policy':'B1_CONTEXT_BUDGETED_HISTORY.v1_FIXED_EXISTING_SELECTION_NOT_FULL_HISTORY',
        'wall_seconds':1800 if stage=='Smoke' else 10800,'process_bytes':4*1024**3,'output_bytes':1024**3 if stage=='Smoke' else 2*1024**3,
        'stop_policy':'FIRST_NONCOMPLETED_OR_TECHNICAL_SOURCE_IDENTITY_CONTEXT_OR_GLOBAL_ERROR_STOP_ENTIRE_BATCH',
        'smoke_proof':proof,'input_code_pins':{str(p.resolve()):sha_file(p) for p in set(files)},**MARKERS,
        'D2_scope':SCOPE,'independent_generalization_claim':False}
    common={k:plan[k] for k in ('model_identifier','context','output_tokens','timeout_seconds','temperature','seed','retry','repair',
        'source_policy','augmentation_policy','dataset_versions','M5_menu','frozen_m5_sha256','catalog','evidence_wire','history_policy')}
    common['prompt_contents']={b['prompt']:sha_file(ROOT/b['prompt']) for b in read(bp/'llm_config.json')['agent_bindings']}
    common['routing_evidence_sha256']=sha_file(UTILITY)
    plan['frozen_comparison_contract_fingerprint']=digest(common)
    if proof and proof['frozen_comparison_contract_fingerprint']!=plan['frozen_comparison_contract_fingerprint']:raise ValueError('SMOKE_FULL_COMPARISON_CONTRACT_CHANGED')
    plan['fingerprint']=digest(plan);atomic_json(pub/'plan.json',plan)
    a={**MARKERS,'scope':plan['scope'],'paired_scope':plan['paired_scope'],'plan_fingerprint':plan['fingerprint'],
        'conditions':CONDITIONS,'dataset_views':VIEWS,'max_generation':plan['generation_total'],'clinical_release_approved':False,
        'accept_source_label_proxy':False,'accept_recorded_time_proxy':False,'accept_common_m5_menu':False,
        'accept_label_conditioned_development':False,'accept_augmentation_initial_time_policy':False,
        'operator_approved':False,'no_other_ollama_work':False,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
    atomic_json(pr/'approval_template.json',a)
    return {'status':'PREPARED_NOT_EXECUTED','plan_id':name,'stage':stage,'fingerprint':plan['fingerprint'],'registered_episodes':plan['registered_episodes'],
        'generation_cap':plan['generation_total'],'actual_LLM_generation':0}

def load(name):
    pub,pr=paths(name);p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['input_code_pins'].items():
        if sha_file(f)!=h:raise ValueError('CODE_DATA_OR_LINEAGE_PIN_CHANGED')
    _,bp,base=base_load(p['base_source_plan'])
    if base['fingerprint']!=p['base_fingerprint']:raise ValueError('BASE_PLAN_CHANGED')
    return pub,pr,p,bp

def permission(a,p):
    from scripts.run_common_menu_b0_b4 import permission as base_permission
    base_permission(a,p)
    if (a.get('paired_scope')!=p['paired_scope'] or a.get('dataset_views')!=VIEWS or
        a.get('accept_label_conditioned_development') is not True or a.get('accept_augmentation_initial_time_policy') is not True):raise ValueError('NEW_D0_D2_OPERATOR_APPROVAL_REQUIRED')

def check(name):
    pub,pr,p,bp=load(name);base=base_preflight(p['base_source_plan']);blockers=list(base['blocker_codes'])
    if (pr/p['run_id']).exists():blockers.append('RUN_USED_NO_RESUME')
    if p['stage']=='Full':smoke_proof(p['smoke_proof']['plan_id'])
    return {'status':'BLOCKED' if blockers else 'STATIC_CONTRACT_READY_USER_APPROVAL_REQUIRED','blocker_codes':blockers,'plan_fingerprint':p['fingerprint'],
        'registered_episodes':p['registered_episodes'],'generation_cap':p['generation_total'],'actual_LLM_generation':0,
        'actual_server_identity_context':'NOT_CHECKED_NO_MODEL_REQUEST','clinical_accuracy':'NOT_EVALUATED'}

def audit_requests(folder,row,p,expected_state):
    before=sorted(folder.glob('Request*_before.json'));after=sorted(folder.glob('Request*_after.json'))
    if len(before)!=row['request_count'] or len(after)!=row['request_count'] or not before:raise ValueError('SMOKE_REQUEST_CHECKPOINT_COUNT')
    pins={}
    for bfile,afile in zip(before,after):
        b=read(bfile);a=read(afile);req=b['request'];payload=json.loads(req['messages'][1]['content']);_no_reference(payload)
        if digest(req)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or a.get('status')!='VALID' or a.get('HTTP_status')!=200:raise ValueError('SMOKE_INVALID_REQUEST_OR_RESPONSE')
        ident=a['identity_after']
        if normalize_model_identifier(ident.get('digest'))!=p['model_identifier'] or ident.get('context_length')!=32768:raise ValueError('SMOKE_MODEL_OR_CONTEXT')
        if b.get('contract')!=p['fingerprint'] or req['model']!=p['model'] or req['options']!={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42} or req.get('stream') is not False:raise ValueError('SMOKE_REQUEST_MODEL_PROMPT_OPTIONS')
        if payload['observation']['chief_complaint']!=expected_state['doctor_visible_info'].get('chief_complaint',''):raise ValueError('SMOKE_DATASET_INITIAL_NOT_DELIVERED')
        if not set(expected_state['patient_visible_info'].get('associated_symptoms',[]))<=set(payload['observation']['current_symptoms']):raise ValueError('SMOKE_SYMPTOM_NOT_DELIVERED')
        for path in (bfile,afile):pins[str(path.resolve())]=sha_file(path)
    return pins

def smoke_proof(name):
    pub,pr,p,bp=load(name)
    if p['stage']!='Smoke':raise ValueError('REAL_SMOKE_PLAN_REQUIRED')
    run=pr/p['run_id'];file=run/'collection.json'
    if not file.exists():raise ValueError('SMOKE_NOT_EXECUTED_FULL_PLAN_NOT_CREATED')
    c=read(file);expected=read(pr/'tasks.json')
    if c.get('test_only') is not False or c['plan_fingerprint']!=p['fingerprint'] or c['status']!='COMPLETED' or len(c['rows'])!=10:raise ValueError('REAL_COMPLETE_SMOKE_REQUIRED')
    if sum(r['request_count'] for r in c['rows'])!=c['generation'] or not 0<c['generation']<=120:raise ValueError('SMOKE_BUDGET_MISMATCH')
    pins={str(file.resolve()):sha_file(file),str((pub/'plan.json').resolve()):sha_file(pub/'plan.json')}
    for r,t in zip(c['rows'],expected):
        if any(r.get(k)!=v for k,v in t.items()) or r['status']!='COMPLETED':raise ValueError('SMOKE_TASK_OR_EPISODE_NOT_COMPLETE')
        folder=run/'requests'/r['dataset_view']/('Episode%04d'%r['view_episode_ordinal']);audit=read(folder/'dataset_view_audit.json')
        if audit['view']!=r['dataset_view']:raise ValueError('SMOKE_VIEW_AUDIT')
        pins.update(audit_requests(folder,r,p,audit['actual_initial_state']))
        for n in ('dataset_view_audit.json','raw_runner_result.json','actual_core_decisions.json','episode_and_policy.json'):
            f=folder/n
            if not f.exists():raise ValueError('SMOKE_EXPORT_MISSING')
            pins[str(f.resolve())]=sha_file(f)
    return {'plan_id':name,'plan_fingerprint':p['fingerprint'],'registered_episodes':10,'status':'REAL_TECHNICAL_SMOKE_COMPLETED',
        'pins':pins,'frozen_comparison_contract_fingerprint':p['frozen_comparison_contract_fingerprint'],
        'diagnosis_accuracy_not_a_promotion_requirement':True}

def execute(name,approval):
    status=check(name)
    if status['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p,bp=load(name);a=read(approval);permission(a,p);run=pr/p['run_id'];run.mkdir(exist_ok=False)
    atomic_json(run/'started.json',{'plan_fingerprint':p['fingerprint'],'at':datetime.now(timezone.utc).isoformat(),'test_only':False})
    atomic_json(run/'approval.json',a)
    data=original_cases();idx=read(bp/'case_index.json');cases={ref:data[n-1] for ref,n in idx.items()};bindings=read(bp/'bindings.json')
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())};rows_by_ref={ref:aug[n] for ref,n in idx.items()}
    begin=time.monotonic();source_stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in {**p['input_code_pins'],**read(ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/p['base_source_plan']/'plan.json')['input_hashes']}}
    def guard():
        permission(a,p)
        if time.monotonic()-begin>=p['wall_seconds'] or _process_private_bytes()>=p['process_bytes']:raise ValueError('GLOBAL_WALL_OR_MEMORY_BUDGET')
        if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('GLOBAL_OUTPUT_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in source_stats.items()):raise ValueError('PINNED_SOURCE_OR_CODE_CHANGED_DURING_RUN')
    budget=GenerationBudget(p['generation_total'],12,resource_guard=guard);backend={};rows=[];stop=None
    try:
        for view in VIEWS:backend[view]=make_paired_backend(cases,dataset_view=view,augmented_rows=rows_by_ref,catalog=evidence_catalog(),
            config=read(bp/'llm_config.json'),bindings=bindings,model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],
            checkpoint_root=run/'requests'/view,routing_evidence=read(UTILITY),history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),
            experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
    except Exception as e:stop='BACKEND_SETUP_FAILED';atomic_json(run/'setup_failure.json',{'error_type':type(e).__name__,'private_error':str(e)})
    with (run/'episodes.jsonl').open('x',encoding='utf-8') as f:
        for task in read(pr/'tasks.json'):
            r={**task,'status':'NOT_ATTEMPTED','request_count':0,'result':None,'error_type':None}
            if stop is None:
                before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard();result=backend[r['dataset_view']]({'case_ref':r['case_ref']},features_for(r['condition']))
                    r.update(result=result,status=result['status'].upper())
                    if result['request_count']!=budget.used-before:raise ValueError('REQUEST_COUNTER_MISMATCH')
                    guard()
                    if result['status']!='completed':stop='FIRST_NONCOMPLETED_EPISODE'
                except Exception as e:r.update(status='FAILED',error_type=type(e).__name__,private_error=str(e));stop='FIRST_TECHNICAL_FAILURE'
                finally:r['request_count']=budget.used-before;_BUDGET.reset(token)
            r['stop_reason']=stop;rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
    report={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','plan_fingerprint':p['fingerprint'],'test_only':False,
        'rows':rows,'generation':budget.used,'wall_seconds':time.monotonic()-begin,'stop_reason':stop,**MARKERS,'D2_scope':SCOPE}
    atomic_json(run/'collection.json',report);atomic_json(pub/'execution_status.json',{k:v for k,v in report.items() if k!='rows'})
    load(name)  # Post-run integrity check, preserving partial outputs on failure.
    return report['status']

def analyze(name):
    from scripts.analyze_original50_d0_d2 import analyze as run
    return run(name)

def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('mode',choices=['prepare-smoke','prepare-full','check','execute','analyze','monitor']);q.add_argument('--plan-id',required=True)
    q.add_argument('--smoke-id',default=SMOKE);q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');q.add_argument('--allow-llm',action='store_true');a=q.parse_args()
    if a.mode!='execute' and (a.approval or a.execute or a.allow_llm):q.error('EXECUTION_FLAGS_EXECUTE_ONLY')
    if a.mode=='prepare-smoke':out=prepare(a.plan_id)
    elif a.mode=='prepare-full':out=prepare(a.plan_id,stage='Full',smoke_id=a.smoke_id)
    elif a.mode=='check':out=check(a.plan_id)
    elif a.mode=='execute':
        if not(a.execute and a.allow_llm and a.approval):q.error('USER_EXECUTE_ALLOW_LLM_NEW_APPROVAL_REQUIRED')
        out={'status':execute(a.plan_id,a.approval)}
        out.update(read(paths(a.plan_id)[0]/'execution_status.json'))
    elif a.mode=='analyze':out=analyze(a.plan_id)
    else:
        pub,pr=paths(a.plan_id);p=read(pub/'plan.json');f=pr/p['run_id']/'episodes.jsonl';rr=[]
        if f.exists():
            for line in f.read_text(encoding='utf-8').splitlines():
                try:rr.append(json.loads(line))
                except ValueError:pass
        out={'recorded_episodes':len(rr),'states':dict(Counter(r['status'] for r in rr)),'generation_saved_rows':sum(r['request_count'] for r in rr)}
    print(json.dumps(out,ensure_ascii=True))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
if __name__=='__main__':main()
