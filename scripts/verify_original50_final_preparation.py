"""Post-preparation verification only; no inference/fitting/approval generation."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_original50_final import PUB,PR,load,check,STAGES
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.routing_collector import atomic_json

def main():
    target=PUB/'final_verification.json'
    if target.exists():raise FileExistsError('VERIFICATION_EXISTS_NO_OVERWRITE')
    fake=read(PUB/'TEST_ONLY_full50.json');flow=read(PUB/'TEST_ONLY_runner150.json');reg=read(PUB/'regression_final.json')
    if fake['status']!='PASS' or flow['status']!='PASS' or reg['status']!='PASS':raise ValueError('INCOMPLETE_OR_FAILED_TEST_PROOF')
    before=read(PR/'final_protection_baseline.json');changed=[f for f,h in before.items() if not Path(f).exists() or sha_file(f)!=h]
    stages={stage:{'B012':check(stage),'B0_B4':check(stage,'B034')} for stage in STAGES}
    ids={c['patient_id'] for c in original_cases()}
    from scripts.run_u2_native_collection import load as native_load
    _,npr,_=native_load('Full');ids.update(r['case']['patient_id'] for r in read(npr/'patients.json'))
    from scripts.run_original50_d0_d2 import M5
    ids.update(read(M5)['provenance']['training_patients'])
    docs=ROOT/'docs/B0_B4_??/27_Original50_????_????'
    leaks=[]
    for folder in (docs,PUB):
        for f in folder.rglob('*'):
            if f.is_file() and f.suffix in ('.md','.json','.csv'):
                t=f.read_text(encoding='utf-8-sig')
                if any(x in t for x in ids):leaks.append(str(f.relative_to(ROOT)))
    user_runs=[]
    for stage in STAGES:
        _,pr,p=load(stage)
        if (pr/p['run_id']).exists():user_runs.append(p['plan_id'])
    ok=not changed and not leaks and not user_runs and stages['Smoke5']['B012']['status']=='STATIC_READY_USER_APPROVAL_REQUIRED'
    report={'status':'B012_SMOKE_READY_B34_BLOCKED' if ok else 'BLOCKED','at_utc':datetime.now(timezone.utc).isoformat(),
            'Source_ready_patients':50,'related_unittest_PASS':reg['tests'],'Fake_all_tests_Core_PASS':fake['registered_episodes'],
            'Fake_B012_runner_and_analysis_PASS':flow['registered_episodes'],'Fake_profiles_not_real_results':True,
            'protected_files':len(before),'protected_changed':len(changed),'public_identifier_leaks':len(leaks),
            'new_actual_LLM_generation':0,'new_actual_data_fitting':0,'new_actual_approval_created':False,'actual_live_runs_created':user_runs,
            'whole_repository_tests':'NOT_RUN','full_B0_B4_execution_ready':False,'stages':stages,
            'clinical_accuracy':'NOT_EVALUATED','source_label_proxy':True,'independent_clinical_review':False,'research_evaluation_ready':False}
    atomic_json(PR/'final_protection_details.json',{'changed':changed,'public_leaks':leaks})
    files=[f for f in (ROOT/'scripts').glob('*original50_final*') if f.is_file()]+[ROOT/'scripts/Invoke-LoCDIOriginal50Final.ps1',ROOT/'tests/locdi/test_original50_final_execution.py']
    atomic_json(PUB/'implementation_manifest.json',{str(f.relative_to(ROOT)):sha_file(f) for f in files})
    atomic_json(target,report)
    print(json.dumps({k:v for k,v in report.items() if k!='stages'}))
    if not ok:raise SystemExit(2)

if __name__=='__main__':main()
