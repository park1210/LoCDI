"""Frozen U1 D2 B3/B4 full10 comparison; no LLM without new user approval."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.authored_utility_transfer import verify,EXPECTED_SHA,make_transfer_backend
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collector import atomic_json
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes,features_for
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from scripts.run_original50_d0_d2 import load as source_load,AUG,M5,permission as source_permission,write_equal
from scripts.run_b34_transfer_smoke import load as smoke_load,IDS as SMOKES
from scripts.analyze_b34_utility_voi import U1

NS='U1_SUPPORT_VOI_EXPANSION_v1';NAME='u1_d2_b34_10p_001';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
CODE=['scripts/run_u1_voi_expansion.py','scripts/Invoke-LoCDIU1VoIExpansion.ps1','app/locdi/authored_utility_transfer.py']

def task_list(panel):
    out=[]
    for i,row in enumerate(panel,1):
        for condition in (['B3','B4'] if i%2 else ['B4','B3']):
            out.append({'episode_ordinal':len(out)+1,'panel_ordinal':i,'case_ref':row['case_ref'],'condition':condition})
    return out

def smoke_evidence():
    pub,pr,p,bp=smoke_load(SMOKES['VoI']);run=pr/p['run_id'];c=read(run/'collection.json')
    if c['status']!='COMPLETED' or c.get('TEST_ONLY') is not False or len(c['rows'])!=2 or c['plan_fingerprint']!=p['fingerprint']:raise ValueError('ACTUAL_COMPLETED_U1_VOI_SMOKE_REQUIRED')
    if any(r['status']!='COMPLETED' for r in c['rows']) or not 0<c['generation']<=24:raise ValueError('SMOKE_COMPLETION_OR_BUDGET')
    return {str(f.resolve()):sha_file(f) for f in run.rglob('*') if f.is_file()}

def prepare():
    pub,pr=PUB/NAME,PR/NAME
    if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY_NO_OVERWRITE')
    _,source_pr,parent,bp=source_load('original50_d0d2_full_003');cfg=read(bp/'llm_config.json');reg=validate_setup(cfg)['prepared']['registry'];index=read(bp/'case_index.json');raw=original_cases();patients=[raw[i-1]['patient_id'] for i in index.values()]
    if set(patients)&set(read(M5)['provenance']['training_patients']):raise ValueError('M5_TRAIN_EVALUATION_OVERLAP')
    v=verify(U1,reg,patients,cfg);smoke=smoke_evidence();pub.mkdir(parents=True);pr.mkdir(parents=True)
    # Exact existing authored split index/evidence, retained as its own Source domain.
    _,spr,_,_=smoke_load(SMOKES['VoI']);envelope=read(spr/'U1_transfer_evidence.json');atomic_json(pr/'U1_transfer_evidence.json',envelope)
    tasks=task_list(read(bp/'panel.json'));atomic_json(pr/'tasks.json',tasks)
    base_file=ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/parent['base_source_plan']/'plan.json';base=read(base_file)
    pins={**parent['input_code_pins'],**base['input_hashes'],**base['code_hashes'],**v['origin_files'],**smoke}
    extra=[ROOT/f for f in CODE]+[pr/'tasks.json',pr/'U1_transfer_evidence.json',Path(envelope['collection_plan']),base_file]
    pins.update({str(f.resolve()):sha_file(f) for f in extra})
    p={'plan_id':NAME,'run_id':'run_'+NAME,'source_plan':'original50_d0d2_full_003','source_fingerprint':parent['fingerprint'],
       'scope':parent['scope'],'paired_scope':parent['paired_scope'],'experiment_scope':'U1_D2_MATCHED_B3_B4_FULL10_ONLY',
       'conditions':['B3','B4'],'dataset_views':['D0','D2'],'actual_dataset_view':'D2','registered_patients':10,'registered_episodes':20,
       'generation_total':240,'generation_per_episode':12,'wall_seconds':5400,'process_bytes':4*1024**3,'output_bytes':2*1024**3,
       'Model':parent['model_identifier'],'model':parent['model'],'context':32768,'output_tokens':1024,'timeout_seconds':120,'temperature':0,'seed':42,
       'retry':0,'repair':0,'warmup':0,'no_resume':True,'U1_sha256':EXPECTED_SHA,'frozen_M5_sha256':sha_file(M5),
       'source_policy':parent['source_policy'],'augmentation_policy':parent['augmentation_policy'],'evidence_wire':parent['evidence_wire'],
       'patient_selection':'EXACT_EXISTING_FIXED10_NO_REPLACEMENT_OR_SOURCE_LABEL_SELECTION','order':'patient order; odd B3-B4/even B4-B3',
       'memory':'EXISTING_TOP5','knowledge':'NONE_BOTH_CONDITIONS','clinical_gate':'UNCHANGED','pins':pins,**MARKERS}
    p['fingerprint']=digest(p);atomic_json(pub/'plan.json',p)
    a={**MARKERS,'scope':p['scope'],'paired_scope':p['paired_scope'],'experiment_scope':p['experiment_scope'],'conditions':p['conditions'],'dataset_views':['D0','D2'],
       'plan_fingerprint':p['fingerprint'],'max_generation':240,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None,'clinical_release_approved':False}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy','accept_authored_utility_transfer','accept_routing_abstention_er_fallback'):a[k]=False
    atomic_json(pr/'approval_template.json',a)
    return {'status':'PREPARED_NOT_EXECUTED','plan_id':NAME,'fingerprint':p['fingerprint'],'episodes':20,'generation_cap':240}

def load():
    pub,pr=PUB/NAME,PR/NAME;p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('PIN_CHANGED')
    _,_,parent,bp=source_load(p['source_plan'])
    if parent['fingerprint']!=p['source_fingerprint']:raise ValueError('SOURCE_PLAN_CHANGED')
    return pub,pr,p,bp

def check():
    pub,pr,p,bp=load();blocks=[]
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_NEW_USER_APPROVAL_REQUIRED','blocker_codes':blocks,'plan_id':NAME,'fingerprint':p['fingerprint'],'generation_cap':240,'actual_LLM_generation':0}

def permission(a,p):
    source_permission(a,p)
    if a.get('experiment_scope')!=p['experiment_scope'] or a.get('accept_authored_utility_transfer') is not True or a.get('accept_routing_abstention_er_fallback') is not True:raise ValueError('NEW_FULL10_TRANSFER_FALLBACK_APPROVAL_REQUIRED')

def execute(approval):
    if check()['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p,bp=load();a=read(approval);permission(a,p);run=pr/p['run_id'];run.mkdir(exist_ok=False);atomic_json(run/'approval.json',a);atomic_json(run/'started.json',{'plan_fingerprint':p['fingerprint'],'at':datetime.now(timezone.utc).isoformat(),'TEST_ONLY':False})
    raw=original_cases();index=read(bp/'case_index.json');cases={ref:raw[i-1] for ref,i in index.items()};aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())};augmented={ref:aug[i] for ref,i in index.items()}
    begun=time.monotonic();stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if time.monotonic()-begun>=p['wall_seconds'] or _process_private_bytes()>=p['process_bytes'] or sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('GLOBAL_RESOURCE_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stats.items()):raise ValueError('PIN_CHANGED_DURING_RUN')
    budget=GenerationBudget(240,12,resource_guard=guard);stop=None;rows=[]
    try:
        backend=make_transfer_backend(cases,dataset_view='D2',augmented_rows=augmented,catalog=evidence_catalog(),config=read(bp/'llm_config.json'),bindings=read(bp/'bindings.json'),model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],checkpoint_root=run/'requests',routing_evidence=read(pr/'U1_transfer_evidence.json'),history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
    except Exception as e:stop='BACKEND_SETUP_FAILED';atomic_json(run/'setup_failure.json',{'private_error':str(e),'error_type':type(e).__name__})
    with (run/'episodes.jsonl').open('x',encoding='utf-8') as f:
        for t in read(pr/'tasks.json'):
            row={**t,'status':'NOT_ATTEMPTED','request_count':0,'result':None}
            if stop is None:
                before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard();r=backend({'case_ref':t['case_ref']},features_for(t['condition']));row.update(status=r['status'].upper(),result=r);guard()
                    if r['request_count']!=budget.used-before:raise ValueError('REQUEST_COUNTER_MISMATCH')
                    if r['status']!='completed':stop='FIRST_NONCOMPLETED'
                except Exception as e:row.update(status='FAILED',private_error=str(e),error_type=type(e).__name__);stop='FIRST_TECHNICAL_ERROR'
                finally:row['request_count']=budget.used-before;_BUDGET.reset(token)
            row['stop_reason']=stop;rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    c={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,'wall_seconds':time.monotonic()-begun,'plan_fingerprint':p['fingerprint'],'TEST_ONLY':False,**MARKERS}
    atomic_json(run/'collection.json',c);atomic_json(pub/'execution_status.json',{k:v for k,v in c.items() if k!='rows'});return c['status']

def analyze():
    from scripts.analyze_original50_d0_d2 import facts
    pub,pr,p,bp=load();run=pr/p['run_id'];c=read(run/'collection.json');tasks=read(pr/'tasks.json');lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if lines!=c['rows'] or len(lines)!=20 or c['plan_fingerprint']!=p['fingerprint'] or sum(r['request_count'] for r in lines)!=c['generation']:raise ValueError('COLLECTION_MISMATCH')
    details=[];summaries={};pairs=[];lookup={}
    for row,t in zip(lines,tasks):
        if any(row[k]!=v for k,v in t.items()):raise ValueError('TASK_MISMATCH')
        folder=run/'requests'/('Episode%04d'%row['episode_ordinal']);f=facts(folder,row['result'],read(bp/'panel.json')[row['panel_ordinal']-1]['reference'])
        if row['status']!='COMPLETED':
            for k in ('source_exact','normalized','department_match','algorithmic_source_exact'):f[k]=None
        payloads=[json.loads(read(b)['request']['messages'][1]['content']) for b in sorted(folder.glob('Request*_before.json'))]
        last=payloads[-1] if payloads else {};last_file=sorted(folder.glob('Request*_core_response.json'));response=read(last_file[-1])['decoded_response'] if last_file else {};indices=response.get('evidence_indices',[])
        runtime=read(folder/'runtime_snapshot_private.json') if (folder/'runtime_snapshot_private.json').exists() else {}
        f.update(final_provided_evidence=last.get('evidence',[]),final_cited_evidence=[last['evidence'][i] for i in indices],received_tests=runtime.get('test_results',[]),actual_role_counts=dict(Counter(b['role'] for b in payloads)))
        item={**row,'facts':f};details.append(item);lookup[row['panel_ordinal'],row['condition']]=item
    for condition in ('B3','B4'):
        rows=[r for r in details if r['condition']==condition];usage=Counter();actions=Counter();roles=Counter();route=Counter()
        for r in rows:
            usage.update(r['facts']['usage']);actions.update((r['result'] or {}).get('action_counts',{}));roles.update(r['facts']['actual_role_counts']);route.update(t['selected_specialty'] or 'ABSTAIN' for t in r['facts']['routing_trace'])
        summaries[condition]={'registered':10,'states':dict(Counter(r['status'] for r in rows)),'Source_Exact':sum(r['facts']['source_exact'] is True for r in rows),'Source_Exact_registered_rate':sum(r['facts']['source_exact'] is True for r in rows)/10,
            'M5_algorithmic_Source_Exact':sum(r['facts']['algorithmic_source_exact'] is True for r in rows) if condition=='B4' else None,'supported':sum(r['facts']['prediction']['certainty']=='supported' and r['status']=='COMPLETED' for r in rows),'unresolved':sum(r['facts']['prediction']['certainty']=='unresolved' and r['status']=='COMPLETED' for r in rows),
            'generation':sum(r['request_count'] for r in rows),'usage':dict(usage),'action_counts':dict(actions),'role_counts':dict(roles),'routing_decisions':dict(route),'initial_routing':dict(Counter((r['facts']['routing_trace'][0]['selected_specialty'] or 'ABSTAIN') if r['facts']['routing_trace'] else 'NOT_RUN' for r in rows)),
            'Planner':sum((r['result'] or {}).get('computation',{}).get('planner_calls',0) for r in rows),'STOP':sum((r['result'] or {}).get('voi_STOP_count',0) for r in rows),'episode_wall_seconds':sum((r['result'] or {}).get('computation',{}).get('wall_seconds',0) for r in rows)}
    for i in range(1,11):
        a,b=lookup[i,'B3'],lookup[i,'B4'];x,y=a['facts'],b['facts'];off={r['test_name'] for r in x['received_tests']};extra=[r for r in y['received_tests'] if r['test_name'] not in off]
        pairs.append({'panel_ordinal':i,'B3_status':a['status'],'B4_status':b['status'],'B3_Source_Exact':x['source_exact'],'B4_Source_Exact':y['source_exact'],'INCORRECT_TO_CORRECT':x['source_exact'] is False and y['source_exact'] is True,'CORRECT_TO_INCORRECT':x['source_exact'] is True and y['source_exact'] is False,'unscored_present':x['source_exact'] is None or y['source_exact'] is None,
            'extra_tests_received':len(extra),'extra_tests_final_provided':sum(r['summary'] in y['final_provided_evidence'] for r in extra),'extra_tests_final_cited':sum(r['summary'] in y['final_cited_evidence'] for r in extra)})
    report={'plan_id':NAME,'conditions':summaries,'pairs':pairs,'generation':c['generation'],'wall_seconds':c['wall_seconds'],'new_generation':0,**MARKERS}
    write_equal(run/'analysis_private.json',details);write_equal(pub/'saved_results.json',report);return report

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze']);q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');a=q.parse_args()
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check()
    elif a.mode=='analyze':out=analyze()
    else:
        if not a.execute or not a.approval:q.error('NEW_USER_EXECUTION_APPROVAL_REQUIRED')
        out={'status':execute(a.approval)}
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
if __name__=='__main__':main()
