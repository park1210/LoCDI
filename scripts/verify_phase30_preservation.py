"""Verify original hashes and new public boundaries without rewriting history."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.structural_release import preservation
from app.locdi.research_protocol_audit import anchored_sources
from app.locdi.episode_exporter import _write
from scripts.verify_simulator_release import verify_manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification-id',default='phase30_final_v1')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.verification_id):parser.error('invalid verification ID')
    destination=ROOT/'outputs/research_protocol'/args.verification_id
    if destination.exists():parser.error('verification ID already used')
    baseline=load(ROOT/'outputs/private/phase30_baseline_hashes_v1.json')
    kept=preservation(baseline)
    rc=ROOT/'outputs/structural_rc/phase212_rc_v1'
    manifest=load(rc/'manifests/release_manifest.json')
    original=anchored_sources(ROOT,load(ROOT/'outputs/private/phase30_rc_source_inventory_v1.json'),manifest['source_code_fingerprint'])
    identifiers=set();evidence=set()
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in {'patient_id','visit_id','case_id','encounter_id','index_encounter_id'} and isinstance(v,str) and v:identifiers.add(v)
                collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for line in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        c=json.loads(line);collect(c)
        evidence.update(e['evidence'] for e in c['environment']['current_encounter_events'] if len(e['evidence'])>=24)
    for p in (ROOT/'outputs/private/routing_collection/phase24b_collect_001').glob('Candidate*.json'):
        collect(load(p))
    forbidden={'patient_id','visit_id','case_id','gold','evaluation','provenance','source_event_store','raw_response'}
    def private(value):
        if isinstance(value,dict):return bool(set(value)&forbidden) or any(private(v) for v in value.values())
        return isinstance(value,list) and any(private(v) for v in value)
    base=ROOT/'outputs/research_protocol'
    files=[p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
    files.append(ROOT/'docs/시뮬레이션 구조/036_RESEARCH_PROTOCOL_AND_EVALUATION_READINESS_REPORT.md')
    leaks=[]
    for p in files:
        raw=p.read_text(encoding='utf8')
        if any(s in raw for s in identifiers|evidence):leaks.append({'file':p.relative_to(ROOT).as_posix(),'code':'ORIGINAL_IDENTIFIER_OR_SOURCE_TEXT'})
        if p.suffix in ('.json','.jsonl'):
            values=[json.loads(raw)] if p.suffix=='.json' else [json.loads(s) for s in raw.splitlines() if s.strip()]
            if any(private(v) for v in values):leaks.append({'file':p.relative_to(ROOT).as_posix(),'code':'PRIVATE_JSON_KEY'})
    valid=verify_manifest(rc,manifest)
    result={'schema_version':'locdi.phase30.preservation.v1','status':'PASS' if kept['status']=='PASS' and valid and original['status']=='PASS' and not leaks else 'FAIL',
        'preservation':kept,'original_rc_sources':original,'rc_manifest_valid':valid,
        'public_files_checked':len(files),'public_leaks':leaks,
        'scope':'Known original LoCDI nested identifiers, collection identifiers, Source evidence >=24 chars and private JSON keys; not raw wire or independent clinical privacy review',
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}
    _write(result,'reports/final_preservation_v1.json',destination,'reports')
    print(json.dumps(result,ensure_ascii=True));sys.exit(0 if result['status']=='PASS' else 2)
