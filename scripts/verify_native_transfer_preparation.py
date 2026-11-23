"""Final non-generating preparation/retention audit; writes only new reports."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import native_routing_transfer as t
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.routing_collector import atomic_json
from scripts.run_original50_d0_d2 import M5

def main():
    target=t.PUB/'final_preparation_verification.json'
    if target.exists():raise FileExistsError('FINAL_VERIFICATION_EXISTS_NO_OVERWRITE')
    fake=read(t.PUB/'TEST_ONLY_validation_and_full250.json');reg=read(t.PUB/'related_regression_final.json')
    pins=read(t.PR/'preservation_before.json');changed=[p for p,h in pins.items() if not Path(p).exists() or sha_file(p)!=h]
    for p,h in read(t.PR/'source_code_pins.json').items():
        if sha_file(p)!=h:raise ValueError('NEW_SOURCE_CODE_PIN_CHANGED')
    pub,pr,p=t.load_validation();validation_status=t.validation_check();admission=t.admission_check()
    docs=next(next((ROOT/'docs').glob('B0_B4_*')).glob('28_*'))
    files=list(docs.glob('*.md'))
    ids={c['patient_id'] for c in original_cases()}|set(read(M5)['provenance']['training_patients'])
    ids.update(r['case']['patient_id'] for r in read(pr/'patients.json'))
    leaks=[]
    for root in (t.PUB,docs):
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.md','.csv'):
                text=f.read_text(encoding='utf-8-sig')
                if any(pid in text for pid in ids):leaks.append(str(f.relative_to(ROOT)))
    ok=fake['status']==reg['status']=='PASS' and not changed and not leaks and len(files)==10
    value={'status':'PREPARATION_PASS_REAL_FIT_AND_VALIDATION_PENDING' if ok else 'BLOCKED',
        'protected_files':len(pins),'protected_changed':len(changed),'public_identifier_leaks':len(leaks),
        'related_tests_PASS':reg['tests'],'new_Source_native_Fake_PASS':fake['validation_native_episodes_PASS'],
        'Original50_B0_B4_Fake_PASS':fake['Original50_conditions_PASS'],'documentation_files':len(files),
        'Source_validation_static':validation_status,'actual_admission':admission,'actual_new_LLM_generation':0,
        'actual_new_Source_fitting':0,'candidate_utility_created':(t.PR/t.FIT/'candidate_utility.json').exists(),
        'transfer_evidence_created':t.EVIDENCE.exists(),'actual_validation_run_created':(pr/p['run_id']).exists(),
        'whole_repository_tests':'NOT_RUN','clinical_accuracy':'NOT_EVALUATED'}
    atomic_json(t.PR/'preservation_details.json',{'changed':changed,'leaks':leaks})
    atomic_json(t.PUB/'implementation_and_documentation_manifest.json',{str(f.relative_to(ROOT)):sha_file(f) for f in [
        ROOT/'app/locdi/native_menu_routing.py',ROOT/'app/locdi/c435_native_validation_source.py',
        ROOT/'scripts/native_routing_transfer.py',ROOT/'scripts/run_native_menu_full50.py',ROOT/'scripts/Invoke-LoCDINativeTransfer.ps1',
        ROOT/'scripts/probe_native_menu_transfer.py',ROOT/'tests/locdi/test_native_menu_transfer.py',*files]})
    atomic_json(target,value);print(json.dumps(value))
    if not ok:raise SystemExit(2)

if __name__=='__main__':main()
