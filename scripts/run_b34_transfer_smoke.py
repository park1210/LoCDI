"""Opt-in U0/U1 and matched U1 VoI smokes; user-only generation."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,json,re,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.authored_utility_transfer import verify,EXPECTED_SHA,SCHEMA,make_transfer_backend
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.original_source_inventory import read,original_cases,sha_file
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collector import atomic_json
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes,features_for
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from scripts.run_original50_d0_d2 import load as source_load,AUG,M5,UTILITY as U0,permission as source_permission
from scripts.analyze_b34_utility_voi import U1

NS='B34_UTILITY_VOI_SMOKE_v1';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
IDS={'Routing':'u1_routing_smoke_002','VoI':'u1_voi_smoke_002'}
CODE=['scripts/run_b34_transfer_smoke.py','scripts/analyze_b34_utility_voi.py','scripts/Invoke-LoCDIB34Transfer.ps1','app/locdi/authored_utility_transfer.py',
      'app/locdi/augmentation_experiment_backend.py','app/locdi/evidence_subset_wire.py']

def paths(name):
    if not re.fullmatch('[A-Za-z0-9_]{1,60}',name):raise ValueError('SAFE_NEW_PLAN_ID_REQUIRED')
    return PUB/name,PR/name

def prepare(stage):
    name=IDS[stage];pub,pr=paths(name)
    if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY_NO_OVERWRITE')
    _,source_pr,parent,bp=source_load('original50_d0d2_smoke_005');cfg=read(bp/'llm_config.json');reg=validate_setup(cfg)['prepared']['registry'];idx=read(bp/'case_index.json');ref=read(bp/'panel.json')[0]['case_ref'];original=original_cases()[idx[ref]-1]
    v=verify(U1,reg,[original['patient_id']],cfg)
    pr.mkdir(parents=True);pub.mkdir(parents=True)
    split={'authored_plan_fingerprint':v['source_plan']['fingerprint'],'interpretation':'ACTUAL_AUTHOR_TASK_TRAIN_SPLITS_ONLY_NOT_SYNTHEA_ENCOUNTERS',
        'candidates':[{'decision':{'patient_id':t['patient_id'],'split':t['split']}} for t in v['source_plan']['tasks']]}
    atomic_json(pr/'authored_split_index.json',split)
    envelope={'schema_version':SCHEMA,'utility_path':str(U1.resolve()),'utility_sha256':EXPECTED_SHA,
        'collection_plan':str((pr/'authored_split_index.json').resolve()),'purpose':'AUTHORED_PROXY_DOMAIN_TRANSFER_NATIVE_DEVELOPMENT_ONLY',**MARKERS}
    atomic_json(pr/'U1_transfer_evidence.json',envelope)
    tasks=[{'branch':'R0','condition':'B3','branch_episode':1},{'branch':'R1','condition':'B3','branch_episode':1}] if stage=='Routing' else [{'branch':'R1','condition':'B3','branch_episode':1},{'branch':'R1','condition':'B4','branch_episode':2}]
    atomic_json(pr/'tasks.json',tasks)
    source_pub=ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/parent['base_source_plan']/'plan.json';base=read(source_pub)
    pins={**parent['input_code_pins'],**base['input_hashes'],**base.get('code_hashes',{}),**v['origin_files']}
    files=[ROOT/f for f in CODE]+[U0,source_pub,pr/'authored_split_index.json',pr/'U1_transfer_evidence.json',pr/'tasks.json']
    pins.update({str(f.resolve()):sha_file(f) for f in files})
    plan={'plan_id':name,'stage':stage,'run_id':'run_'+name,'source_plan':'original50_d0d2_smoke_005','source_fingerprint':parent['fingerprint'],
        'scope':parent['scope'],'paired_scope':parent['paired_scope'],'experiment_scope':'AUTHORED_U1_DOMAIN_TRANSFER_B3B4_ONLY',
        'conditions':['B3'] if stage=='Routing' else ['B3','B4'],'dataset_views':['D0','D2'],'actual_dataset_view':'D2',
        'patient_count':1,'selection_rule':'EXISTING_FIXED_PANEL_FIRST_PATIENT_NO_GOLD_BASED_SELECTION','case_ref':ref,
        'registered_episodes':2,'generation_total':24,'generation_per_episode':12,'wall_seconds':900,'process_bytes':4*1024**3,'output_bytes':1024**3,
        'U0_sha256':sha_file(U0),'U1_sha256':EXPECTED_SHA,'Model':parent['model_identifier'],'context':32768,'output_tokens':1024,
        'temperature':0,'seed':42,'timeout_seconds':120,'retry':0,'repair':0,'warmup':0,'no_resume':True,'frozen_M5_sha256':sha_file(M5),
        'source_policy':parent['source_policy'],'augmentation_policy':parent['augmentation_policy'],'evidence_wire':parent['evidence_wire'],
        'counterfactual_routing_support':'LOW_DOMAIN_SUPPORT_NOT_CONFIRMED_ACCURACY_IMPROVEMENT','U1_stage':'initial_full_direct_diagnosis_to_native_action_domain_transfer',
        'pins':pins,**MARKERS}
    plan['fingerprint']=digest(plan);atomic_json(pub/'plan.json',plan)
    approval={**MARKERS,'scope':plan['scope'],'paired_scope':plan['paired_scope'],'experiment_scope':plan['experiment_scope'],
        'conditions':plan['conditions'],'dataset_views':['D0','D2'],'max_generation':24,'plan_fingerprint':plan['fingerprint'],
        'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None,'clinical_release_approved':False}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer'):approval[k]=False
    atomic_json(pr/'approval_template.json',approval)
    return {'status':'PREPARED_NOT_EXECUTED','plan_id':name,'fingerprint':plan['fingerprint'],'generation_cap':24}

def load(name):
    pub,pr=paths(name);p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT_MISMATCH')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('SOURCE_CODE_ARTIFACT_PIN_CHANGED')
    _,_,parent,bp=source_load(p['source_plan'])
    if parent['fingerprint']!=p['source_fingerprint']:raise ValueError('SOURCE_PLAN_CHANGED')
    return pub,pr,p,bp

def check(name):
    pub,pr,p,bp=load(name);blocked=[]
    if (pr/p['run_id']).exists():blocked.append('RUN_USED_NO_RESUME')
    if p['stage']=='VoI':
        _,rp,r,_=load(IDS['Routing']);file=rp/r['run_id']/'collection.json'
        if not file.exists() or read(file)['status']!='COMPLETED':blocked.append('ACTUAL_ROUTING_SMOKE_NOT_COMPLETED')
    return {'status':'BLOCKED' if blocked else 'STATIC_READY_USER_APPROVAL_REQUIRED','blocker_codes':blocked,'plan_id':name,'fingerprint':p['fingerprint'],'generation_cap':24,'actual_LLM_generation':0}

def permission(a,p):
    source_permission(a,p)
    if a.get('experiment_scope')!=p['experiment_scope'] or a.get('accept_authored_utility_transfer') is not True:raise ValueError('NEW_AUTHORED_TRANSFER_OPERATOR_APPROVAL_REQUIRED')

def execute(name,approval):
    if check(name)['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p,bp=load(name);a=read(approval);permission(a,p);run=pr/p['run_id'];run.mkdir(exist_ok=False);atomic_json(run/'approval.json',a)
    atomic_json(run/'started.json',{'fingerprint':p['fingerprint'],'at':datetime.now(timezone.utc).isoformat(),'TEST_ONLY':False})
    idx=read(bp/'case_index.json');ref=p['case_ref'];case=original_cases()[idx[ref]-1]
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    begun=time.monotonic();stop=None;rows=[];clients={};stat={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if time.monotonic()-begun>=900 or _process_private_bytes()>=p['process_bytes'] or sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('GLOBAL_RESOURCE_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stat.items()):raise ValueError('PIN_CHANGED_DURING_RUN')
    budget=GenerationBudget(24,12,resource_guard=guard)
    for branch in ('R0','R1'):
        clients[branch]=make_transfer_backend({ref:case},dataset_view='D2',augmented_rows={ref:aug[idx[ref]]},catalog=evidence_catalog(),
            config=read(bp/'llm_config.json'),bindings=read(bp/'bindings.json'),model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],
            checkpoint_root=run/'requests'/branch,routing_evidence=read(U0) if branch=='R0' else read(pr/'U1_transfer_evidence.json'),
            history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
    with (run/'episodes.jsonl').open('x',encoding='utf-8') as out:
        for t in read(pr/'tasks.json'):
            row={**t,'status':'NOT_ATTEMPTED','request_count':0,'result':None}
            if stop is None:
                before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard();r=clients[t['branch']]({'case_ref':ref},features_for(t['condition']));row.update(result=r,status=r['status'].upper());guard()
                    if r['status']!='completed':stop='FIRST_NONCOMPLETED'
                except Exception as e:row.update(status='FAILED',private_error=str(e),error_type=type(e).__name__);stop='FIRST_TECHNICAL_ERROR'
                finally:row['request_count']=budget.used-before;_BUDGET.reset(token)
            row['stop_reason']=stop;rows.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
    c={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,'wall_seconds':time.monotonic()-begun,'plan_fingerprint':p['fingerprint'],'TEST_ONLY':False,**MARKERS}
    atomic_json(run/'collection.json',c);atomic_json(pub/'execution_status.json',{k:v for k,v in c.items() if k!='rows'});return c['status']

def analyze(name):
    from scripts.analyze_original50_d0_d2 import facts
    from scripts.run_original50_d0_d2 import write_equal
    pub,pr,p,bp=load(name);run=pr/p['run_id'];c=read(run/'collection.json');ref=read(bp/'panel.json')[0]['reference'];out=[];private=[]
    for r in c['rows']:
        folder=run/'requests'/r['branch']/('Episode%04d'%r['branch_episode']);f=facts(folder,r['result'],ref)
        out.append({**{k:r[k] for k in ('branch','condition','status','request_count')},'Source_Exact':f['source_exact'],
            'Source_department':f['department_match'],'certainty':f['prediction']['certainty'],'routing_distribution':dict(Counter(t['selected_specialty'] or 'ABSTAIN' for t in f['routing_trace']))})
        private.append({'row':r,'facts':f})
    write_equal(run/'analysis_private.json',private);report={'rows':out,'generation':c['generation'],'new_generation':0,**MARKERS};write_equal(pub/'saved_results.json',report);return report

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze']);q.add_argument('--stage',choices=list(IDS),default='Routing');q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');a=q.parse_args();name=IDS[a.stage]
    if a.mode=='prepare':out=prepare(a.stage)
    elif a.mode=='check':out=check(name)
    elif a.mode=='analyze':out=analyze(name)
    else:
        if not a.execute or not a.approval:q.error('USER_EXECUTE_AND_NEW_APPROVAL_REQUIRED')
        out={'status':execute(name,a.approval)}
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
if __name__=='__main__':main()
