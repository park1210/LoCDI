"""Dynamic admission inspection only. Never runs a model or creates an approval."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_original50_full_readiness import PUB,snapshot_status
from app.locdi.original_source_inventory import read


def status():
    source=read(PUB/'data_readiness.json');split=read(PUB/'validation_independence.json');saved=snapshot_status()
    blockers=[]
    if saved['U2_Full'].get('status')!='COMPLETED':blockers.append('U2_NATIVE_FULL_COLLECTION_NOT_COMPLETED')
    if not saved['U2_artifact']['exists']:blockers.append('U2_NATIVE_FIT_ARTIFACT_NOT_AVAILABLE')
    blockers.append('NATIVE_U2_TRANSFER_ADMISSION_NOT_INSTALLED')
    if split['U2_validation_M5_overlap']:blockers.append('ROUTING_VALIDATION_VOI_TRAIN_OVERLAP')
    if not split['new_validation_patients']:blockers.append('NO_INDEPENDENT_BANK_VALIDATION_CANDIDATE')
    if saved['terminal'].get('status')!='COMPLETED':blockers.append('TERMINAL_REPLAY_NOT_COMPLETED')
    fake=PUB/'TEST_ONLY_full50.json'
    if not fake.exists() or read(fake).get('status')!='PASS':blockers.append('FULL50_FAKE_VERIFICATION_NOT_COMPLETED')
    return {'status':'BLOCKED','Source_ready_patients':source['Source_runtime_ready'],'registered_patients':50,
        'U2_Full_status':saved['U2_Full'].get('status'),'U2_Full_saved_generations':saved['U2_Full'].get('generation'),
        'U2_artifact_exists':saved['U2_artifact']['exists'],'Terminal_replay_status':saved['terminal']['status'],
        'Terminal_replay_Source_Exact':saved['terminal'].get('T1_Source_Exact'),
        'blocker_codes':blockers,'full50_execute_command_available':False,
        'new_LLM_generation':0,'new_utility_fitting':0,'clinical_accuracy':'NOT_EVALUATED'}


if __name__=='__main__':
    print(json.dumps(status()));raise SystemExit(2)
