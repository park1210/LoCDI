"""User-only six-role diagnostic comparison, two saved Validation Snapshots.

No U2/U3 fitting or B3/B4 evaluation. Source/model/clinical Gates unchanged.
"""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
from copy import deepcopy
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_u2_native_collection import load as native_load
from scripts.run_original50_d0_d2 import M5
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.native_agent_collection import collect_one_backend
from app.locdi.validation_role_snapshot import client_type,diagnostic_task_budget,ValidationPrefixTransport,VERSION
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.locdi.development_routing_proxy import MARKERS

NS='ROUTING_ROOT_CAUSE_20261123_v1';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
DEFAULT='role_validation_snapshot_001'


def paths(name):
    if not name or not name.replace('_','').isalnum():raise ValueError('SAFE_NEW_PLAN_ID')
    return PUB/name,PR/name


def prepare(name):
    pub,pr=paths(name)
    if pub.exists() or pr.exists():raise FileExistsError('NEW_PLAN_ONLY_NO_OVERWRITE')
    _,npr,np=native_load('Full');run=npr/np['run_id'];c=read(run/'collection.json')
    if c['status']!='COMPLETED':raise ValueError('COMPLETED_SOURCE_COLLECTION_REQUIRED')
    patients=read(npr/'patients.json');val=[r for r in patients if r['split']=='validation']
    train={r['case']['patient_id'] for r in patients if r['split']=='train'};evaluation={r['patient_id'] for r in original_cases()}
    if len(val)!=2 or any(r['case']['patient_id'] in train|evaluation for r in val):raise ValueError('INDEPENDENT_TRAIN_EVALUATION_PATIENTS')
    agents=[a for a in np['registry']['agents'] if a['role']=='specialty'];acquisition=agents[0]['department']
    if acquisition!='cardiology':raise ValueError('PRESPECIFIED_FIRST_REGISTRY_BINDING_CHANGED')
    roles=[{'role':'er','department':'','role_ref':'ER'}]+[{'role':'specialty','department':a['department'],'role_ref':a['department']} for a in agents]
    tasks=[];pins=dict(np['pins'])
    for patient in val:
        source=run/('Patient%02d'%patient['ordinal'])/acquisition/'Episode0001'
        before=sorted(source.glob('Request*_before.json'));after=sorted(source.glob('Request*_after.json'))
        if not before or len(before)!=len(after):raise ValueError('COMPLETE_ACQUISITION_SOURCE')
        p=json.loads(read(before[-1])['request']['messages'][1]['content'])
        frozen=digest({k:p.get(k) for k in ('observation','evidence','supported_diagnosis_eligible_indices')})
        for role in roles:tasks.append({'task_ordinal':len(tasks)+1,'validation_ordinal':patient['ordinal'],**role,
            'source_episode':str(source.resolve()),'final_request_number':len(before),'frozen_patient_fingerprint':frozen})
        pins.update({str(f.resolve()):sha_file(f) for f in source.rglob('*') if f.is_file()})
    pub.mkdir(parents=True);pr.mkdir(parents=True)
    atomic_json(pr/'patients.json',val);atomic_json(pr/'tasks.json',tasks);atomic_json(pr/'config.json',read(npr/'llm_config.json'))
    for f in [Path(__file__),ROOT/'app/locdi/validation_role_snapshot.py',ROOT/'scripts/Invoke-LoCDIValidationRoles.ps1',pr/'patients.json',pr/'tasks.json',pr/'config.json']:
        pins[str(f.resolve())]=sha_file(f)
    p={'plan_id':name,'run_id':'run_'+name,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','conditions':['B2'],
       'scope_note':'B2 selective-memory engine used only as acquisition replay host; NOT B2/B3/B4 evaluation',
       'experiment_scope':'VALIDATION_SIX_ROLE_DIAGNOSTIC_ONLY_VOI_OFF','contract':VERSION,'registry':np['registry'],
       'registered_patients':2,'registered_terminal_requests':12,'generation_total':12,'generation_per_task':1,
       'cached_prefix_responses_total':sum(t['final_request_number']-1 for t in tasks),
       'patient_selection':'EXISTING_TWO_VALIDATION_PATIENTS_NO_OUTCOME_SELECTION',
       'snapshot_selection':'LAST_REQUEST_OF_FIRST_REGISTERED_AGENT_CARDIOLOGY_NOT_BEST_RESPONSE',
       'validation_M5_overlap':np['validation_M5_training_overlap'],'B4_admission':'UNCHANGED_BLOCKED',
       'comparison':'SAME_SNAPSHOT_TERMINAL_JUDGMENTS_NOT_NATIVE_SEQUENTIAL_CAPABILITY',
       'retry':0,'repair':0,'warmup':0,'no_resume':True,'wall_seconds':1800,'process_bytes':4*1024**3,'output_bytes':1024**3,
       'pins':pins,**MARKERS};p['fingerprint']=digest(p);atomic_json(pub/'plan.json',p)
    a={**MARKERS,'scope':p['scope'],'conditions':['B2'],'experiment_scope':p['experiment_scope'],
       'plan_fingerprint':p['fingerprint'],'max_generation':12,'clinical_release_approved':False,
       'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu',
              'accept_frozen_snapshot_role_comparison','accept_validation_m5_overlap_for_voi_off_only','accept_no_further_acquisition_task_budget'):a[k]=False
    atomic_json(pr/'approval_template.json',a)
    return {'status':'PREPARED_NOT_EXECUTED','plan_id':name,'generation_cap':12,'fingerprint':p['fingerprint'],'actual_generation':0}


def load(name):
    pub,pr=paths(name);p=read(pub/'plan.json')
    if digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint']:raise ValueError('PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('PIN_CHANGED')
    return pub,pr,p


def permission(a,p):
    from scripts.run_common_menu_b0_b4 import permission as common_permission
    common_permission(a,p)
    if (a.get('experiment_scope')!=p['experiment_scope'] or any(a.get(k) is not True for k in
        ('accept_frozen_snapshot_role_comparison','accept_validation_m5_overlap_for_voi_off_only','accept_no_further_acquisition_task_budget'))):raise ValueError('NEW_EXACT_ROLE_DIAGNOSTIC_APPROVAL')


def one(task,patient,*,p,config,authorization,folder,factory=None):
    import app.locdi.current_ablation_llm_backend as native
    source=Path(task['source_episode']);before=[read(f) for f in sorted(source.glob('Request*_before.json'))];after=[read(f) for f in sorted(source.glob('Request*_after.json'))]
    original_factory=native._native_transport;original_reserve=native.reserve_generation;controllers=[]
    def transport(prepared):
        c=ValidationPrefixTransport(before,after,prepared=prepared,factory=factory or original_factory,role=task['role'],department=task['department']);controllers.append(c);return c
    def reserve():
        if len(controllers)!=1:raise ValueError('ONE_TRANSPORT_PER_TASK')
        return controllers[0].reserve(original_reserve)
    cls=client_type(final_request_number=task['final_request_number'],role=task['role'],department=task['department'],registry=p['registry'])
    try:
        with diagnostic_task_budget(task['final_request_number']),patch('app.locdi.common_menu_backend.CommonMenuNativeClient',new=cls),\
                patch.object(native,'_native_transport',new=transport),patch.object(native,'reserve_generation',new=reserve):
            result=collect_one_backend(patient['case'],department='cardiology',config=config,bindings={'TRAIN':patient['binding']},
                model=read(M5),authorization=authorization,plan_fingerprint=p['fingerprint'],checkpoint_root=folder,
                experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
    finally:
        if controllers:
            c=controllers[0];atomic_json(folder/'role_replay_provenance.json',{'contract':VERSION,
                'cached_prefix_responses':c.index,'new_generation_attempts':c.attempts,'new_terminal_posts':c.posts,
                'actual_terminal_role':task['role'],'actual_terminal_department':task['department'],
                'replay_host_role':'FIXED_CARDIOLOGY_NATIVE_SOURCE','host_audit_is_not_new_ER_OR_SPECIALTY_NATIVE_EXECUTION':True,
                'cached_identity_not_current_observation':True,'diagnostic_only_budget_not_original_native_budget_exhaustion':True})
    c=controllers[0]
    if result['status']!='completed' or c.posts!=1 or c.index!=len(before)-1:raise ValueError('ROLE_DIAGNOSTIC_NOT_COMPLETED')
    decoded=read(folder/'Episode0001'/('Request%04d_core_response.json'%task['final_request_number']))['decoded_response']
    if decoded['action']!='diagnose':raise ValueError('FINAL_DIAGNOSTIC_ONLY')
    correct=(decoded['target']==patient['reference']['label']) if decoded['certainty']=='supported' else None
    b=read(folder/'Episode0001'/('Request%04d_before.json'%task['final_request_number']));body=json.loads(b['request']['messages'][1]['content'])
    if digest({k:body.get(k) for k in ('observation','evidence','supported_diagnosis_eligible_indices')})!=task['frozen_patient_fingerprint']:raise ValueError('FROZEN_PATIENT_FIELDS_CHANGED')
    return {'task_ordinal':task['task_ordinal'],'validation_ordinal':task['validation_ordinal'],'role_ref':task['role_ref'],
            'status':'COMPLETED','new_generation':1,'certainty':decoded['certainty'],'prediction':decoded,
            'source_exact':correct,'core_revalidated':True,'frozen_patient_fingerprint':task['frozen_patient_fingerprint']}


def check(name):
    pub,pr,p=load(name);blocks=[]
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    proof=pub/'TEST_ONLY_validation.json'
    if not proof.exists() or read(proof).get('status')!='PASS':blocks.append('FAKE_CORE_ROLE_VALIDATION_REQUIRED')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_NEW_USER_APPROVAL_REQUIRED','blocker_codes':blocks,
            'plan_id':name,'generation_cap':12,'actual_generation':0,'B4_admission':'UNCHANGED_BLOCKED'}


def execute(name,approval):
    if check(name)['status']=='BLOCKED':raise ValueError('PREFLIGHT_BLOCKED')
    pub,pr,p=load(name);a=read(approval);permission(a,p);run=pr/p['run_id'];run.mkdir(exist_ok=False)
    atomic_json(run/'approval.json',a);atomic_json(run/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'plan_fingerprint':p['fingerprint'],'TEST_ONLY':False})
    patients={r['ordinal']:r for r in read(pr/'patients.json')};start=time.monotonic();stats={f:(Path(f).stat().st_size,Path(f).stat().st_mtime_ns) for f in p['pins']}
    def guard():
        permission(a,p)
        if time.monotonic()-start>=p['wall_seconds'] or _process_private_bytes()>=p['process_bytes'] or sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>=p['output_bytes']:raise ValueError('RESOURCE_LIMIT')
        if any((Path(f).stat().st_size,Path(f).stat().st_mtime_ns)!=s for f,s in stats.items()):raise ValueError('PIN_CHANGED_DURING_RUN')
    budget=GenerationBudget(12,1,resource_guard=guard);rows=[];stop=None
    with (run/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for t in read(pr/'tasks.json'):
            r={'task_ordinal':t['task_ordinal'],'validation_ordinal':t['validation_ordinal'],'role_ref':t['role_ref'],'status':'NOT_ATTEMPTED','new_generation':0}
            if stop is None:
                before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard();r=one(t,patients[t['validation_ordinal']],p=p,config=read(pr/'config.json'),authorization=a,folder=run/('Task%02d'%t['task_ordinal']));guard()
                    if budget.used-before!=1:raise ValueError('NEW_GENERATION_COUNT')
                except Exception as e:r.update(status='FAILED',error_type=type(e).__name__,private_error=str(e));stop='FIRST_ERROR_NO_RETRY'
                finally:r['new_generation']=budget.used-before;_BUDGET.reset(token)
            rows.append(r);stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
    out={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'generation':budget.used,'plan_fingerprint':p['fingerprint'],'stop_reason':stop,**MARKERS}
    atomic_json(run/'collection.json',out);return {'status':out['status'],'generation':budget.used}


def analyze(name):
    from collections import Counter
    pub,pr,p=load(name);c=read(pr/p['run_id']/'collection.json')
    if c['plan_fingerprint']!=p['fingerprint'] or sum(r['new_generation'] for r in c['rows'])!=c['generation']:raise ValueError('COLLECTION_COUNT')
    role_names=['ER']+[a['department'] for a in p['registry']['agents'] if a['role']=='specialty']
    report={'comparison':p['comparison'],'registered_patients':2,'generation':c['generation'],'new_generation_in_analysis':0,
        'by_role':{role:{'registered':2,'states':dict(Counter(r['status'] for r in c['rows'] if r['role_ref']==role)),
            'Source_Exact':sum(r.get('source_exact') is True for r in c['rows'] if r['role_ref']==role),
            'unresolved':sum(r.get('certainty')=='unresolved' for r in c['rows'] if r['role_ref']==role)} for role in role_names},**MARKERS}
    target=pub/'saved_results.json'
    if target.exists():
        if read(target)!=report:raise ValueError('EXISTING_ANALYSIS_DIFFERS')
    else:atomic_json(target,report)
    return report


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze']);q.add_argument('--plan-id',default=DEFAULT);q.add_argument('--execute',action='store_true');q.add_argument('--approval',type=Path);a=q.parse_args()
    if a.mode=='prepare':out=prepare(a.plan_id)
    elif a.mode=='check':out=check(a.plan_id)
    elif a.mode=='analyze':out=analyze(a.plan_id)
    else:
        if not a.execute or not a.approval:q.error('NEW_USER_EXECUTION_APPROVAL_REQUIRED')
        out=execute(a.plan_id,a.approval)
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)
