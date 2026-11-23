"""D7 candidates: immutable hashes and allowed scope, no new datasets/labels."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_suitability import sha
from app.locdi.external_dialogue import MANIFEST,PRIVATE as HB_PRIVATE
from app.locdi.source_replay import PUBLIC as D3_PUBLIC,PRIVATE as D3_PRIVATE,atomic_rows


def build():
    environment=ROOT/'outputs/private/healthbench_environment/D6_v1'
    review=ROOT/'outputs/private/reference_review/D4_reference_v1'
    hb=json.loads(MANIFEST.read_text(encoding='utf-8'))
    groups=[('A','Original LoCDI Development Dataset','longitudinal_single_file_v34.1',
        [ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'],'SYNTHEA_DERIVATIVE','UNKNOWN','DEVELOPMENT_ONLY','EXPOSED_50'),
        ('B','D3 Source Replay Dataset','simulation_source_replay_v1',
        [p for base in (D3_PUBLIC,D3_PRIVATE) for p in base.iterdir() if p.is_file()],
        'PINNED_SYNTHEA_SOURCE_POINTERS','UNKNOWN_NOT_APPROVED_FOR_REDISTRIBUTION','EXPLICIT_DEVELOPMENT_PROXY_ONLY','EXPOSED_29_REPLAY_PATIENTS'),
        ('C','D4 HealthBench Raw',hb['revision'],[MANIFEST]+[ROOT/e['path'] for e in hb['files'].values()],
        'OFFICIAL_HEALTHBENCH_REVISION','MIT_DATASET_CARD_AND_EVALUATOR_CODE','PRIVATE_INPUTS_PENDING_EXTERNAL_EVALUATION_APPROVAL','D4_SCHEMA_INSPECTION_AND_D5_PILOT_3'),
        ('D','D5 External Dialogue Adapter','locdi.D5.external_adapter.v1',
        [ROOT/p for p in ('app/locdi/external_dialogue.py','app/locdi/external_benchmark_contract.py','scripts/run_healthbench_pilot.py','datasets/external/D5_healthbench_adapter_contract.json')]+[HB_PRIVATE/'selection_manifest.json'],
        'LOCAL_D5_IMPLEMENTATION','LOCAL_CODE_RIGHTS_NOT_INFERRED_FROM_DATA_LICENSE','DEVELOPMENT_DIALOGUE_ADAPTER','PILOT_3_FIXED_EXCLUSION'),
        ('E','D6 Grader Environment Specification','D6_v1_CP311_WIN_AMD64',
        [p for p in environment.rglob('*') if p.is_file() and not any(x in p.parts for x in ('runtime','__pycache__'))]+[ROOT/p for p in ('scripts/prepare_healthbench_environment.py','scripts/smoke_healthbench_official.py','app/locdi/healthbench_grader.py')],
        'PINNED_EVALUATOR_AND_AUDITED_HASH_LOCK','PSF_MIT_PUBLIC_DOMAIN_BSD_APACHE_MPL_AND_BUNDLED_NOTICES','OFFLINE_IMPORT_AND_TEST_ONLY_PROTOCOL','NO_REAL_GRADING'),
        ('F','Internal Reference Review Packets','D4_reference_v1',list(review.glob('*')),
        'EXISTING_INTERNAL_SOURCES','UNKNOWN_PRIVATE_ONLY','BLINDED_REVIEW_INPUT_NOT_GOLD','REVIEWERS_UNASSIGNED')]
    private=[];public=[]
    for key,name,version,paths,source,license_status,allowed,exposure in groups:
        entries=[]
        for path in sorted(set(paths)):
            if not path.is_file():raise FileNotFoundError('FREEZE_CANDIDATE_FILE_NOT_FOUND')
            entries.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,
                'private':path.is_relative_to((ROOT/'outputs/private'))})
        meta={'bundle':key,'name':name,'version':version,'source':source,'license_access_status':license_status,
            'validation_status':'HASHED_REVALIDATED_STRUCTURAL_OR_OFFLINE_CONTRACT_SCOPE',
            'clinical_review_status':'NOT_REVIEWED' if key in 'ABF' else 'NOT_PATIENT_DIAGNOSIS_GOLD',
            'allowed_use':allowed,'blocked_use':['UNAPPROVED_CLINICAL_RESEARCH','UNAPPROVED_TRAINING','UNAPPROVED_REDISTRIBUTION'],
            'development_exposure':exposure,'freeze_eligibility':'ELIGIBLE_LOCAL_SCOPED_CANDIDATE_NOT_CLINICAL_RELEASE',
            'file_count':len(entries),'files_fingerprint':digest(entries),'clinical_research_ready':False}
        private.append({**meta,'files':entries});public.append(meta)
    destination=ROOT/'outputs/private/dataset_release/D6_release_candidate_v1';destination.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':'locdi.D6.private_release_candidate.v1','bundles':private,'datasets_merged':False,'new_patient_data_generated':False}
    manifest['manifest_fingerprint']=digest(manifest)
    atomic_rows(destination,'manifest.json',manifest,jsonl=False)
    summary={'schema_version':'locdi.D6.release_candidate.v1','private_manifest_path':(destination/'manifest.json').relative_to(ROOT).as_posix(),
        'private_manifest_sha256':sha(destination/'manifest.json'),'manifest_fingerprint':manifest['manifest_fingerprint'],
        'bundles':public,'actual_freeze_executed':False,'clinical_research_ready':False,'new_dataset_created':False}
    atomic_rows(ROOT/'outputs/reports','D6_release_candidate_v1.json',summary,jsonl=False)
    print(json.dumps(summary,ensure_ascii=True))


if __name__=='__main__':build()
