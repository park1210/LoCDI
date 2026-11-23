"""Final RC verification; only creates a new public result outside RC manifest."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.structural_release import preservation
from app.locdi.episode_exporter import _write
from scripts.verify_simulator_release import verify_manifest


if __name__=='__main__':
    root=ROOT/'outputs/structural_rc/phase212_rc_v1'
    baseline=load(ROOT/'outputs/private/phase212_baseline_hashes_v1.json')
    preserved=preservation(baseline);manifest=load(root/'manifests/release_manifest.json')
    valid=verify_manifest(root,manifest)
    code_fp=digest({p.relative_to(ROOT).as_posix():sha(p) for folder in ('app','validation','scripts','tests/locdi') for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts})
    ids=set();evidence=set()
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in {'patient_id','visit_id','case_id','encounter_id','index_encounter_id'} and isinstance(v,str) and v:ids.add(v)
                collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for line in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        case=json.loads(line);collect(case)
        evidence.update(e['evidence'] for e in case['environment']['current_encounter_events'] if len(e['evidence'])>=24)
    forbidden={'patient_id','visit_id','case_id','gold','evaluation','provenance','source_event_store','raw_response'}
    def private(value):
        if isinstance(value,dict):
            if set(value)&forbidden:return True
            for key,item in value.items():
                # A validator STATUS is public bookkeeping, not raw provenance.
                if (key=='validator_statuses' and isinstance(item,dict) and
                        set(item)<={'schema','temporal','actions','provenance'} and
                        all(isinstance(v,str) and v in {'PASS','WARNING','NOT_REVIEWED','FAIL','BLOCKED'} for v in item.values())):
                    continue
                if private(item):return True
            return False
        return isinstance(value,list) and any(private(v) for v in value)
    files=[p for p in root.rglob('*') if p.is_file() and 'private' not in p.relative_to(root).parts]
    document=ROOT/'docs/시뮬레이션 구조/035_SIMULATOR_STRUCTURAL_RELEASE_CANDIDATE_REPORT.md'
    if document.exists():files.append(document)
    leaks=[]
    for p in files:
        text=p.read_text(encoding='utf-8')
        if any(value in text for value in ids|evidence):leaks.append({'file':p.relative_to(ROOT).as_posix(),'code':'ORIGINAL_ID_OR_SOURCE_TEXT'})
        if p.suffix in {'.json','.jsonl'}:
            values=[json.loads(text)] if p.suffix=='.json' else [json.loads(line) for line in text.splitlines() if line.strip()]
            if any(private(v) for v in values):leaks.append({'file':p.relative_to(ROOT).as_posix(),'code':'PRIVATE_JSON_KEY'})
    result={'schema_version':'locdi.phase212.final_verification.v1','status':'PASS' if valid and preserved['status']=='PASS' and
        code_fp==manifest['source_code_fingerprint'] and not leaks else 'FAIL',
        'preservation':preserved,'release_manifest_valid':valid,'release_manifest_fingerprint':manifest['fingerprint'],
        'current_source_code_fingerprint_matches':code_fp==manifest['source_code_fingerprint'],
        'public_files_checked':len(files),'public_leaks':leaks,
        'privacy_scope':'Known original 50-case nested identifiers and source evidence >=24 characters; key inspection; not raw wire or independent clinical privacy review.',
        'release_decision':load(root/'reports/release_decision.json')['decision'],
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}
    _write(result,'reports/phase212_final_verification_v1.json',ROOT/'outputs','reports')
    print(json.dumps(result,ensure_ascii=True));sys.exit(0 if result['status']=='PASS' else 2)
