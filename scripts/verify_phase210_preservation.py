"""Read-only preservation/privacy checks; writes only a new verification result."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.dataset_freeze import sha, load
from app.locdi.llm_setup import ROOT
from app.locdi.episode_exporter import _write
from app.locdi.handoff_audit import audit_saved_success, RUN_ID


def verify():
    baseline = load(ROOT/'outputs/private/phase210_baseline_hashes_v1.json')
    changed = [p for p, expected in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p) != expected]
    run = ROOT/'outputs/development_handoff'/RUN_ID
    run_hashes = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob('*') if p.is_file()}
    cases = [json.loads(line) for line in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    forbidden_ids = set()
    evidence = set()
    def collect_identifiers(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {'patient_id', 'case_id', 'visit_id', 'encounter_id', 'index_encounter_id'} and isinstance(item, str) and item:
                    forbidden_ids.add(item)
                collect_identifiers(item)
        elif isinstance(value, list):
            for item in value:
                collect_identifiers(item)
    for case in cases:
        collect_identifiers(case)
        for event in case['environment']['current_encounter_events']:
            value = event.get('evidence')
            if isinstance(value, str) and len(value) >= 24:
                evidence.add(value)
    files = list((ROOT/'outputs/phase210_test_only').rglob('*.json')) + list((ROOT/'outputs/phase210_test_only').rglob('*.jsonl'))
    files += [ROOT/'outputs/reports/phase210_saved_success_v2.json', ROOT/'docs/시뮬레이션 구조/033_HANDOFF_ROUTING_VOI_INTEGRATION_REPORT.md']
    leaks = []
    private_keys = {'gold', 'evaluation', 'provenance', 'source_event_store', 'patient_id', 'visit_id', 'case_id'}
    def keys(value):
        if isinstance(value, dict):
            return set(value).intersection(private_keys).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    for p in files:
        text = p.read_text(encoding='utf-8')
        if any(value in text for value in forbidden_ids) or any(value in text for value in evidence):
            leaks.append({'file':p.relative_to(ROOT).as_posix(), 'error':'SOURCE_TEXT_OR_IDENTIFIER'})
        if p.suffix in {'.json', '.jsonl'}:
            values = [json.loads(text)] if p.suffix == '.json' else [json.loads(line) for line in text.splitlines() if line.strip()]
            if any(keys(v) for v in values):
                leaks.append({'file':p.relative_to(ROOT).as_posix(), 'error':'PRIVATE_FIELD'})
    regression = load(ROOT/'outputs/reports/phase210_regression_v1.json')
    freeze = load(ROOT/'outputs/reports/phase210_dataset_freeze_verification_v1.json')
    saved = audit_saved_success()
    result = {'schema_version':'locdi.phase210.final_verification.v1',
        'status':'PASS' if not changed and not leaks and regression['success'] and freeze['status']=='PASS' and saved['status']=='PASS' else 'FAIL',
        'existing_files_checked':len(baseline), 'existing_files_changed':changed,
        'core_files_checked':sum(p.startswith('app/core/') for p in baseline),
        'saved_pilot_file_count':len(run_hashes), 'saved_pilot_hashes':run_hashes,
        'public_files_checked':len(files), 'identifier_source_text_or_private_field_leaks':leaks,
        'privacy_scope':'All 50 nested patient/case/visit/encounter identifiers, current event evidence strings >=24 characters, and forbidden JSON keys. Not raw wire validation.',
        'saved_run_audit':saved['status'], 'regression':regression,
        'dataset_freeze':freeze, 'actual_generation_calls':0, 'utility_fitting':0, 'clinical_verdicts_generated':0,
        'research_evaluation_ready':False}
    _write(result, 'reports/phase210_final_verification_v1.json', ROOT/'outputs', 'reports')
    return result


if __name__ == '__main__':
    result = verify()
    print(json.dumps({k:v for k,v in result.items() if k != 'saved_pilot_hashes'}, ensure_ascii=True))
    sys.exit(0 if result['status']=='PASS' else 2)
