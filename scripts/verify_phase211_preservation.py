"""Read-only baseline, source, public output and current Plan verification."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.episode_exporter import _write


if __name__=='__main__':
    baseline=load(ROOT/'outputs/private/phase211_baseline_hashes_v1.json')
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    saved_runs={}
    for parent in ('outputs/development_handoff','outputs/development_pilot'):
        for directory in (ROOT/parent).iterdir():
            if not directory.is_dir():continue
            expected={p for p in baseline if p.startswith(directory.relative_to(ROOT).as_posix()+'/')}
            actual={p.relative_to(ROOT).as_posix() for p in directory.rglob('*') if p.is_file()}
            saved_runs[directory.name]={'file_count':len(actual),'inventory_unchanged':actual==expected,
                'hashes_unchanged':not bool(expected.intersection(changed))}
    cases=[json.loads(line) for line in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    identifiers=set();evidence=set()
    def collect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in {'patient_id','case_id','visit_id','encounter_id','index_encounter_id'} and isinstance(v,str) and v:identifiers.add(v)
                collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    for case in cases:
        collect(case)
        evidence.update(e['evidence'] for e in case['environment']['current_encounter_events'] if len(e['evidence'])>=24)
    forbidden={'patient_id','visit_id','case_id','gold','evaluation','provenance','source_event_store'}
    def private_keys(value):
        if isinstance(value,dict):return bool(set(value)&forbidden) or any(private_keys(v) for v in value.values())
        return isinstance(value,list) and any(private_keys(v) for v in value)
    files=[]
    for folder in ('episodes','metrics','reports'):
        files.extend(p for p in (ROOT/'outputs/phase211'/folder).rglob('*') if p.is_file())
    files.append(ROOT/'docs/시뮬레이션 구조/034_ARTIFACT_READINESS_AND_DECISION_ALIGNMENT_REPORT.md')
    leaks=[]
    for p in files:
        raw=p.read_text(encoding='utf-8')
        if any(v in raw for v in identifiers|evidence):leaks.append({'file':p.relative_to(ROOT).as_posix(),'code':'ORIGINAL_ID_OR_SOURCE_TEXT'})
        if p.suffix in ('.json','.jsonl'):
            values=[json.loads(raw)] if p.suffix=='.json' else [json.loads(line) for line in raw.splitlines() if line.strip()]
            if any(private_keys(v) for v in values):leaks.append({'file':p.relative_to(ROOT).as_posix(),'code':'PRIVATE_KEY'})
    snapshot=load(ROOT/'outputs/phase211/reports/phase211_current_snapshot_v2.json')
    scenarios=load(ROOT/'outputs/phase211/reports/phase211_audit_v1.json')
    plans=snapshot['live_plans']
    codes_current=all(sha(ROOT/p)==h for plan in plans for p,h in plan['plan']['code_sha256'].items())
    plans_valid=all(digest(plan['plan'])==plan['plan_fingerprint'] and plan['blocker_codes'] for plan in plans)
    expected={'A':'completed','B':'completed','C':'completed','D':'completed','E':'completed',
        'F_TIMEOUT':'failed','G_BUDGET':'failed','H_INVALID_EVIDENCE':'failed'}
    scenarios_valid=len(scenarios['records'])==40 and all(r['status']==expected[r['scenario']] and r['source_boundary_pass'] and r['export_status']=='EXPORTED' for r in scenarios['records'])
    regression=load(ROOT/'outputs/reports/phase211_regression_v1.json');freeze=load(ROOT/'outputs/reports/phase211_dataset_freeze_v1.json')
    result={'schema_version':'locdi.phase211.final_verification.v1',
        'status':'PASS' if not changed and not leaks and codes_current and plans_valid and scenarios_valid and
            regression['success'] and freeze['status']=='PASS' and all(r['inventory_unchanged'] and r['hashes_unchanged'] for r in saved_runs.values()) else 'FAIL',
        'existing_files_checked':len(baseline),'existing_files_changed':changed,
        'core_files_checked':sum(p.startswith('app/core/') for p in baseline),'saved_runs':saved_runs,
        'public_files_checked':len(files),'public_leaks':leaks,
        'privacy_scope':'Known nested original 50-case identifiers, current source evidence strings >=24 characters, forbidden JSON keys; not raw wire or independent clinical privacy review.',
        'plans_current_code_hashes_match':codes_current,'plan_fingerprints_valid_and_blocked':plans_valid,
        'scenario_count':len(scenarios['records']),'all_scenarios_match_expected':scenarios_valid,
        'regression':regression,'dataset_freeze':freeze,
        'artifact_snapshot_sha256':sha(ROOT/'outputs/phase211/reports/phase211_current_snapshot_v2.json'),
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0,'research_evaluation_ready':False}
    _write(result,'reports/phase211_final_verification_v1.json',ROOT/'outputs','reports')
    print(json.dumps(result,ensure_ascii=True));sys.exit(0 if result['status']=='PASS' else 2)
