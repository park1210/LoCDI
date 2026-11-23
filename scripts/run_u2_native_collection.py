"""User-executed Native five-Agent collection; no fitting or clinical admission."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sqlite3,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_terminal_only_replay import PUB,PR
from scripts.prepare_u2_source_collection import load as load_static
from scripts.audit_b34_diagnosis_recovery import native_candidate
from scripts.run_original50_d0_d2 import M5
from app.locdi.routing_data_reuse import BANK,SOURCE,DATABASE,read as source_read
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.case_adapter import adapt_case
from app.locdi.m5_common_source import verify_csv
from app.locdi.native_agent_collection import collect_one_backend
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.locdi.development_routing_proxy import MARKERS

NAMES={'Smoke':'u2_native_5agent_smoke_001','Full':'u2_native_5agent_full_001'}


def prepare():
    if any((PUB/n).exists() or (PR/n).exists() for n in NAMES.values()):raise FileExistsError('NEW_NATIVE_PLAN_ONLY')
    _,spr,sp=load_static();candidates=read(spr/'candidates.json');refs={r['ordinal']:r for r in read(spr/'reference_private.json')}
    bank={r['training_case_id']:r for r in source_read(BANK)};source={r['case_id']:r for r in source_read(SOURCE)}
    con=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);rows=[]
    cfg=read(ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/original50_d0d2_full_003_source/llm_config.json')
    ready=validate_setup(cfg);registry=ready['prepared']['registry'];model=read(M5)
    try:
        for c in candidates:
            cid=c['source_context_id'];ctx=json.loads(con.execute('SELECT payload FROM contexts WHERE cid=?',(cid,)).fetchone()[0])
            case,binding,rejected=native_candidate(bank[cid],source[cid],ctx,con,model)
            adapted=adapt_case(case,registry_manifest=registry)
            rows.append({'ordinal':c['ordinal'],'split':'train' if c['ordinal']<=8 else 'validation',
                'case':case,'binding':binding,'reference':refs[c['ordinal']],
                'validation':{r.validator:r.status for r in adapted.validation_results},'rejected_memory':rejected})
    finally:con.close()
    verify_csv([r['source_event'] for c in rows for r in c['binding']['rows']]+[
        {'source_metadata':r} for c in rows for r in c['reference']['source_condition_rows']])
    holdout={r['patient_id'] for r in original_cases()};train={r['case']['patient_id'] for r in rows if r['split']=='train'};val={r['case']['patient_id'] for r in rows if r['split']=='validation'}
    if (train|val)&holdout or train&val or len(train)!=8 or len(val)!=2:raise ValueError('PATIENT_SPLIT')
    for stage,name in NAMES.items():
        pub,pr=PUB/name,PR/name;pub.mkdir(parents=True);pr.mkdir(parents=True)
        selected=rows[:1] if stage=='Smoke' else rows
        atomic_json(pr/'patients.json',selected);atomic_json(pr/'llm_config.json',cfg)
        agents=[a for a in registry['agents'] if a['role']=='specialty']
        pins={**sp['pins']}
        for f in [Path(__file__),ROOT/'app/locdi/native_agent_collection.py',ROOT/'scripts/audit_b34_diagnosis_recovery.py',
                  ROOT/'scripts/Invoke-LoCDIU2NativeCollection.ps1',pr/'patients.json',pr/'llm_config.json']:
            pins[str(f.resolve())]=sha_file(f)
        p={'plan_id':name,'run_id':'run_'+name,'stage':stage,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY',
           'experiment_scope':'U2_FIXED_FIVE_SPECIALTY_NATIVE_COLLECTION_ONLY','conditions':['B2'],
           'features_note':'selective memory engine features only; this is NOT a B2 evaluation',
           'agents':agents,'registry':registry,'generation_total':len(selected)*5*12,'episode_generation':12,
           'registered_patients':len(selected),'registered_episodes':len(selected)*5,
           'wall_seconds':1800 if stage=='Smoke' else 5400,'process_bytes':4*1024**3,'output_bytes':2*1024**3,
           'retry':0,'repair':0,'warmup':0,'no_resume':True,'Source_time':'UNCHANGED_COMMON_RECORDED_TIME_PROXY_5MIN_ACTION_GATED',
           'reference_policy':'single current Encounter Source diagnosis; not future date; intraday UNREVIEWED',
           'train_patients':8,'validation_patients':2,'train_validation_overlap':0,'Original50_overlap':0,
           'validation_M5_training_overlap':len(val&set(model['provenance']['training_patients'])),
           'U2_native_artifact_created':False,'B4_admission':'BLOCKED_ROUTING_VALIDATION_VOI_TRAIN_OVERLAP',
           'pins':pins,**MARKERS};p['fingerprint']=digest(p);atomic_json(pub/'plan.json',p)
        a={**MARKERS,'scope':p['scope'],'conditions':['B2'],'plan_fingerprint':p['fingerprint'],
           'experiment_scope':p['experiment_scope'],'max_generation':p['generation_total'],'clinical_release_approved':False,
           'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
        for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_native_training_intervention','accept_source_reference_date_proxy'):a[k]=False
        atomic_json(pr/'approval_template.json',a)
    return {'status':'NATIVE_SOURCE_PLANS_PREPARED_NOT_COLLECTED','Native_validated':len(rows),'train':8,'validation':2,
            'Original50_overlap':0,'validation_M5_train_overlap':len(val&set(model['provenance']['training_patients'])),
            'Smoke_max_generation':60,'Full_max_generation':600,'actual_generation':0}


def load(stage):
    name=NAMES[stage];p=read(PUB/name/'plan.json')
    if digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint']:raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('PIN_CHANGED')
    return PUB/name,PR/name,p


def permission(a,p):
    from scripts.run_common_menu_b0_b4 import permission as common_permission
    common_permission(a,p)
    if (a.get('experiment_scope')!=p['experiment_scope']
        or a.get('accept_native_training_intervention') is not True
        or a.get('accept_source_reference_date_proxy') is not True):raise ValueError('NEW_NATIVE_COLLECTION_APPROVAL_REQUIRED')


def check(stage):
    pub,pr,p=load(stage);blocks=[]
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    proof=PUB/'U2_NATIVE_TEST_ONLY_validation.json'
    if not proof.exists() or read(proof).get('status')!='PASS':blocks.append('FIVE_AGENT_OFFLINE_VALIDATION_REQUIRED')
    if stage=='Full':
        smoke=PUB/NAMES['Smoke']/'saved_results.json'
        if not smoke.exists() or read(smoke).get('complete_collection_groups')!=1:blocks.append('ACTUAL_NATIVE_SMOKE_NOT_COMPLETED_AND_ANALYZED')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_NEW_USER_APPROVAL_REQUIRED','blocker_codes':blocks,
            'plan_id':p['plan_id'],'generation_cap':p['generation_total'],'actual_generation':0,
            'B4_admission':p['B4_admission']}


def execute(stage,approval):
    if check(stage)['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p=load(stage);a=read(approval);permission(a,p);run=pr/p['run_id'];run.mkdir(exist_ok=False)
    atomic_json(run/'approval.json',a);atomic_json(run/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'TEST_ONLY':False})
    start=time.monotonic();stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if time.monotonic()-start>=p['wall_seconds'] or _process_private_bytes()>=p['process_bytes'] or sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('RESOURCE_BUDGET')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stats.items()):raise ValueError('PIN_CHANGED_DURING_RUN')
    budget=GenerationBudget(p['generation_total'],12,resource_guard=guard);rows=[];stop=None
    with (run/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for patient in read(pr/'patients.json'):
            for agent in p['agents']:
                row={'ordinal':patient['ordinal'],'split':patient['split'],'agent_id':agent['agent_id'],
                     'department':agent['department'],'status':'NOT_ATTEMPTED','generation':0}
                if stop is None:
                    budget.episode_used=0;before=budget.used;token=_BUDGET.set(budget)
                    try:
                        guard();folder=run/('Patient%02d'%patient['ordinal'])/agent['department']
                        r=collect_one_backend(patient['case'],department=agent['department'],config=read(pr/'llm_config.json'),
                            bindings={'TRAIN':patient['binding']},model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],
                            checkpoint_root=folder,experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                        row.update(status=r['status'].upper(),result=r)
                        if r['status']!='completed':stop='FIRST_NONCOMPLETED'
                        guard()
                    except Exception as e:row.update(status='FAILED',error_type=type(e).__name__,private_error=str(e));stop='FIRST_ERROR_NO_RETRY'
                    finally:row['generation']=budget.used-before;_BUDGET.reset(token)
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
    c={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,
       'plan_fingerprint':p['fingerprint'],'wall_seconds':time.monotonic()-start,'stop_reason':stop,**MARKERS}
    atomic_json(run/'collection.json',c);return {'status':c['status'],'generation':budget.used}


def analyze(stage):
    from collections import Counter
    pub,pr,p=load(stage);run=pr/p['run_id'];c=read(run/'collection.json');patients={r['ordinal']:r for r in read(pr/'patients.json')}
    if c['plan_fingerprint']!=p['fingerprint'] or sum(r['generation'] for r in c['rows'])!=c['generation']:raise ValueError('COLLECTION_LINEAGE')
    rows=[];initial={}
    for r in c['rows']:
        correct=None
        if r['status']=='COMPLETED':
            folder=run/('Patient%02d'%r['ordinal'])/r['department']/'Episode0001'
            before=sorted(folder.glob('Request*_before.json'));after=sorted(folder.glob('Request*_after.json'))
            if len(before)!=r['generation'] or len(after)!=len(before):raise ValueError('CHECKPOINT_COUNT')
            for bfile,afile in zip(before,after):
                b,a=read(bfile),read(afile);payload=json.loads(b['request']['messages'][1]['content'])
                if digest(b['request'])!=b['request_fingerprint'] or a['status']!='VALID' or a['request_fingerprint']!=b['request_fingerprint'] or a['HTTP_status']!=200:raise ValueError('REQUEST_LINEAGE')
                if payload['role']!='specialty' or payload['department']!=r['department']:raise ValueError('ACTUAL_SPECIALTY_ROLE_MISMATCH')
            initial.setdefault(r['ordinal'],set()).add(digest(json.loads(read(before[0])['request']['messages'][1]['content'])['observation']))
            prediction=r['result']['prediction']
            if prediction['certainty']=='supported':correct=prediction['diagnosis']==patients[r['ordinal']]['reference']['label']
        rows.append({**r,'correctness_proxy':correct})
    if any(len(v)!=1 for v in initial.values()):raise ValueError('FIVE_AGENT_INITIAL_OBSERVATION_MISMATCH')
    groups=[]
    for i in patients:
        rr=[r for r in rows if r['ordinal']==i]
        if len(rr)!=5 or {r['agent_id'] for r in rr}!={a['agent_id'] for a in p['agents']}:raise ValueError('FIVE_AGENT_SLOT_CONTRACT')
        groups.append({'ordinal':i,'split':patients[i]['split'],'complete_collection':all(r['status']=='COMPLETED' for r in rr),
                       'complete_scorable':all(type(r['correctness_proxy']) is bool for r in rr),
                       'positives':sum(r['correctness_proxy'] is True for r in rr)})
    result={'status':c['status'],'complete_collection_groups':sum(r['complete_collection'] for r in groups),
            'complete_scorable_train_groups':sum(r['complete_scorable'] and r['split']=='train' for r in groups),
            'all_wrong_scorable_groups':sum(r['complete_scorable'] and r['positives']==0 for r in groups),
            'multiple_positive_groups':sum(r['positives']>1 for r in groups),'generation':c['generation'],
            'by_agent':{a['department']:dict(Counter(str(r['correctness_proxy']) for r in rows if r['agent_id']==a['agent_id'])) for a in p['agents']},
            'fitting':0,'U2_artifact_created':False,'B4_admission':p['B4_admission'],**MARKERS}
    for path,value in [(run/'analysis_private.json',{'rows':rows,'groups':groups}),(pub/'saved_results.json',result)]:
        if path.exists():
            if read(path)!=value:raise FileExistsError('ANALYSIS_DIFFERS_NO_OVERWRITE')
        else:atomic_json(path,value)
    return result


def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze']);q.add_argument('--stage',choices=NAMES,default='Smoke');q.add_argument('--execute',action='store_true');q.add_argument('--approval',type=Path);a=q.parse_args()
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check(a.stage)
    elif a.mode=='analyze':out=analyze(a.stage)
    else:
        if not a.execute or not a.approval:q.error('NEW_OPERATOR_APPROVAL_REQUIRED')
        out=execute(a.stage,a.approval)
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
if __name__=='__main__':main()
