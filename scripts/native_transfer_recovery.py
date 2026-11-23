"""Short-path Native transfer recovery. Never resumes an old run or refits.

Original pinned modules, patient inputs, labels and fitted candidate are reused
read-only. New paths, Plan fingerprint and operator approval are mandatory.
"""
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
from unittest.mock import patch
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import native_routing_transfer as t
from scripts import run_native_menu_full50 as full
from app.locdi.storage_path_budget import validation_layout,evaluation_layout
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest

OLDPUB=t.PUB;OLDPR=t.PR;OLDVAL=t.VALIDATION;OLDFIT=OLDPR/t.FIT
PUB=ROOT/'outputs/research_evaluation/nt3';PR=ROOT/'outputs/private/nt3'
VAL='v003';EVIDENCE=PR/'admit/evidence.json'
STAGES=full.STAGES

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:raise FileExistsError('EXISTING_RECOVERY_ARTIFACT_DIFFERS')
    else:atomic_json(path,value)

def verify_candidate():
    f=OLDFIT/'candidate_utility.json'
    if not f.exists():raise ValueError('PREVIOUSLY_APPROVED_FIT_CANDIDATE_REQUIRED_NO_AUTOFIT')
    candidate=read(f);rows,_,_,plan,_=t.derive_training()
    expected=[{**{k:r[k] for k in ('patient_id','decision_id','agent_id','correct')},
        'tokens':t.tokens(r['observation'],r['observation']['_routing_public_menu'])} for r in rows]
    if (candidate['feature_version']!=t.VERSION or candidate['fit_rows_fingerprint']!=digest(rows)
        or candidate['artifact']['registry']!=plan['registry'] or candidate['artifact']['min_samples']!=2
        or sorted(map(digest,expected))!=sorted(map(digest,candidate['artifact']['samples']))):raise ValueError('EXISTING_CANDIDATE_LINEAGE_CHANGED')
    for p,h in candidate['source_pins'].items():
        if sha_file(p)!=h:raise ValueError('ORIGINAL_CANDIDATE_PIN_CHANGED')
    return {'status':'EXISTING_FIT_VERIFIED_REUSED_NO_REFIT','sha256':sha_file(f),'samples':len(expected),'fit_calls':0}

@contextmanager
def scope():
    # Absolute FIT keeps the actual previously fitted file in place.
    with patch.object(t,'PUB',PUB),patch.object(t,'PR',PR),patch.object(t,'VALIDATION',VAL),\
            patch.object(t,'FIT',OLDFIT),patch.object(t,'EVIDENCE',EVIDENCE),\
            patch.object(full,'PUB',PUB),patch.object(full,'PR',PR),\
            patch.object(full,'name',new=lambda s:{'Smoke5':'s3','Verify10':'q3','Full50':'f3'}[s]):yield

def prepare():
    if (PUB/'recovery_plan.json').exists():raise FileExistsError('RECOVERY_PREPARED_USE_CHECK_NO_OVERWRITE')
    fit=verify_candidate();oldpub,oldpr,old=t.load_validation();oldrun=oldpr/old['run_id'];collection=read(oldrun/'collection.json')
    if collection['generation']!=7 or collection['status']!='PARTIAL_OR_FAILED':raise ValueError('EXPECTED_FAILED_RUN_CHANGED_INSPECT_FIRST')
    before={str(f.resolve()):sha_file(f) for folder in (oldpr,OLDFIT) for f in folder.rglob('*') if f.is_file()}
    before.update(read(OLDPR/'preservation_before.json'))
    write_new(PR/'preservation_before.json',before)
    # Share immutable input content, not old results/approvals/responses.
    for filename in ('train_fit_rows_private.json','collection_index.json','source_code_pins_002.json'):
        write_new(PR/filename,read(OLDPR/filename))
    patients=read(oldpr/'patients.json')
    for patient in patients:
        patient['provenance']['routing_outcomes_not_yet_observed']=False
        patient['provenance']['prior_partial_native_validation_exposure']=True
    write_new(PR/VAL/'patients.json',patients);write_new(PR/VAL/'llm_config.json',read(oldpr/'llm_config.json'))
    p=deepcopy(old);p.pop('fingerprint');p.update(plan_id=VAL,run_id='r003',
        storage_policy='SHORT_PRIVATE_CHECKPOINT_NAMESPACE.v1',previous_failed_plan=old['plan_id'],
        previous_generation_preserved=7,previous_results_merged=False,no_resume=True,
        repeated_validation_after_storage_fix=True,fit_reused_sha256=fit['sha256'])
    files=[Path(__file__),ROOT/'scripts/Invoke-LoCDINativeRecovery.ps1',ROOT/'app/locdi/storage_path_budget.py',
        PR/VAL/'patients.json',PR/VAL/'llm_config.json',PR/'collection_index.json',OLDFIT/'candidate_utility.json']
    p['pins']={**p['pins'],**{str(f.resolve()):sha_file(f) for f in files}}
    p['fingerprint']=digest(p);write_new(PUB/VAL/'plan.json',p)
    a=read(oldpr/'approval_template.json');a.update(plan_fingerprint=p['fingerprint'],
        accept_storage_recovery_and_repeat_validation=False,previous_generation_not_reused=7)
    write_new(PR/VAL/'approval_template.json',a)
    # Reuse the actual completed TEST_ONLY backend proof only when its backend
    # and patient clinical content are unchanged; recovery storage is tested anew.
    write_new(PUB/'TEST_ONLY_validation_and_full250.json',read(OLDPUB/'TEST_ONLY_validation_and_full250.json'))
    for stage,n in STAGES.items():evaluation_layout(PR/{'Smoke5':'s3','Verify10':'q3','Full50':'f3'}[stage]/('run_'+{'Smoke5':'s3','Verify10':'q3','Full50':'f3'}[stage]),n*5)
    layout=validation_layout(PR/VAL/p['run_id'],p['agents'])
    report={'status':'NEW_RECOVERY_PLAN_PREPARED_NOT_EXECUTED','plan_id':VAL,'run_id':p['run_id'],
        'plan_fingerprint':p['fingerprint'],'registered_episodes':10,'generation_cap':120,
        'max_reserved_path_chars':max(x['reserved_max_file_chars'] for x in layout),
        'existing_fit':fit,'previous_generation_preserved':7,'previous_completed':3,'previous_failed':1,'previous_not_attempted':6,
        'new_generation':0,'new_fit_calls':0,'prior_results_merged':False}
    write_new(PUB/'recovery_plan.json',report);return report

def storage_check():
    with scope():_,pr,p=t.load_validation()
    result=validation_layout(pr/p['run_id'],p['agents'])
    return {'status':'PATH_LAYOUT_PASS','max_reserved_chars':max(r['reserved_max_file_chars'] for r in result),'limit':239}

def check(stage='Validation'):
    storage=storage_check()
    proof=PUB/'TEST_ONLY_storage_recovery.json'
    if not proof.exists() or read(proof).get('status')!='PASS':return {'status':'BLOCKED','blocker_codes':['NEW_STORAGE_REGRESSION_REQUIRED'],'generation':0}
    with scope():
        if stage=='Validation':result=t.validation_check()
        else:result=full.check(stage)
    return {**result,'storage':storage,'existing_fit_reused':True}

def execute(stage,approval):
    state=check(stage)
    if state['status']=='BLOCKED':raise ValueError('RECOVERY_PREFLIGHT_BLOCKED')
    a=read(approval)
    if a.get('accept_storage_recovery_and_repeat_validation') is not True:raise ValueError('NEW_RECOVERY_APPROVAL_REQUIRED')
    with scope():
        if stage=='Validation':
            with t.validation_collector_scope() as collector:return collector.execute('Validation',approval)
        return full.execute(stage,approval)

def prepare_evaluation():
    with scope():
        admission=t.admission_check()
        if admission['status']=='BLOCKED' or not EVIDENCE.exists():raise ValueError('REAL_VALIDATION_AND_ADMISSION_REQUIRED')
        names={s:full.name(s) for s in STAGES}
    if any((PUB/n).exists() or (PR/n).exists() for n in names.values()):raise FileExistsError('NEW_SHORT_EVALUATION_PLANS_ONLY')
    evidence=read(EVIDENCE);_,bp,base=full.legacy.load('Full50')
    old_tasks=read(bp/'tasks.json');order=list(dict.fromkeys(r['original_ordinal'] for r in old_tasks));old={r['original_ordinal'] for r in old_tasks if r['old_development10']}
    extras=[Path(__file__),ROOT/'scripts/Invoke-LoCDINativeRecovery.ps1',ROOT/'app/locdi/storage_path_budget.py',PUB/'recovery_plan.json']
    pins={**base['pins'],**evidence['origin_files'],str(EVIDENCE.resolve()):sha_file(EVIDENCE),**{str(f.resolve()):sha_file(f) for f in extras}}
    common=digest({'parent':base['comparison_fingerprint'],'utility':sha_file(EVIDENCE),'feature':t.VERSION,'storage':'SHORT_PRIVATE_CHECKPOINT_NAMESPACE.v1','pins':pins})
    for stage,n in STAGES.items():
        name=names[stage];pr=PR/name;pub=PUB/name;p=deepcopy(base);p.pop('fingerprint')
        p.update(plan_id=name,run_id='run_'+name,stage=stage,conditions=full.CONDS,execution_scope='B0_B4_NATIVE_PUBLIC_MENU_TRANSFER',
            registered_patients=n,registered_episodes=n*5,generation_total=n*60,comparison_fingerprint=common,
            utility_sha256=sha_file(EVIDENCE),utility_selection='ADMITTED_NATIVE_SOURCE_PROXY_WITH_REPORTED_VALIDATION_LIMITS',
            blocker_codes=[],wall_seconds=1800 if n==5 else 3600 if n==10 else 14400,feature_version=t.VERSION,
            storage_policy='SHORT_PRIVATE_CHECKPOINT_NAMESPACE.v1',previous_partial_validation_generation=7,
            source_parent_B012_plan_fingerprint=base['fingerprint'])
        evaluation_layout(pr/p['run_id'],n*5)
        tasks=[{'episode_ordinal':i+1,'original_ordinal':ordinal,'case_ref':'Original%03d'%ordinal,
                'condition':condition,'old_development10':ordinal in old}
               for i,(ordinal,condition) in enumerate((o,c) for o in order[:n] for c in full.CONDS)]
        write_new(pr/'tasks.json',tasks);p['pins']={**pins,str((pr/'tasks.json').resolve()):sha_file(pr/'tasks.json')}
        p['fingerprint']=digest(p);write_new(pub/'plan.json',p)
        a=read(bp/'approval_template.json');a.update(conditions=full.CONDS,plan_fingerprint=p['fingerprint'],execution_scope=p['execution_scope'],
            max_generation=n*60,accept_native_public_menu_feature=False,accept_prior_VoI_exposure_and_bounded_independence=False,
            accept_storage_recovery_and_repeat_validation=False,admitted_utility_sha256=p['utility_sha256'])
        write_new(pr/'approval_template.json',a)
    return {'status':'SHORT_PATH_B0_B4_PLANS_PREPARED_NOT_EXECUTED','Full50_episodes':250,'generation_cap':3000,'new_generation':0}

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','check','execute','analyze','admission-check','admit','prepare-evaluation'])
    q.add_argument('--stage',choices=['Validation',*STAGES],default='Validation');q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');a=q.parse_args()
    if a.mode!='execute' and (a.execute or a.approval):q.error('EXECUTION_FLAGS_EXECUTE_ONLY')
    if a.mode=='prepare':out=prepare()
    elif a.mode=='check':out=check(a.stage)
    elif a.mode=='execute':
        if not a.execute or not a.approval:q.error('USER_NEW_APPROVAL_EXECUTE_REQUIRED')
        out=execute(a.stage,a.approval)
    elif a.mode=='prepare-evaluation':out=prepare_evaluation()
    else:
        with scope():
            if a.mode=='analyze':out=t.validation_analyze() if a.stage=='Validation' else full.analyze(a.stage)
            else:out=t.admission_check(write=a.mode=='admit')
    print(json.dumps(out))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)

if __name__=='__main__':main()
