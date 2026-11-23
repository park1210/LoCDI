"""Independent model-free final publication verification, exclusive seal."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_final_3b_evaluation import PUB, PR, MODEL3, DIGEST3
from scripts.audit_research_gaps import read, rows, sha, write, check
from scripts.finalize_final_3b_results import recheck_collections, DOC
from app.locdi.final_diagnosis_evaluation import diagnosis_match
from app.locdi.current_ablation_runner import reaggregate_trace
from app.locdi.dataset_loader import load_cases


def main(write_seal=True):
    live = PR/'live/final_exp_001'; snapshots = read(PR/'diagnostic_snapshot_panel.json')
    pilot = read(PR/'diagnosis_model_pilot_results.json'); pilot_checks = []
    for n, row in enumerate(pilot, 1):
        original = snapshots[n-1]
        before = read(live/'diagnostic_snapshots'/('Task%04d'%n)/'Request0001_before.json')
        after = read(live/'diagnostic_snapshots'/('Task%04d'%n)/'Request0001_after.json')
        request = before['request']; source = original['request']; value = json.loads(after['response_envelope']['message']['content'])
        prediction = {'certainty': value.get('certainty'), 'diagnosis': value.get('target') if value.get('action') == 'diagnose' else None,
                      'department': value.get('department'), 'action': value.get('action')}
        expected_match = diagnosis_match(prediction['diagnosis'], original['reference']['accepted_diagnoses'], certainty=prediction['certainty'])
        pilot_checks.append(prediction == row.get('prediction') and expected_match == row.get('match')
            and request['messages'] == source['messages'] and request['format'] == source['format'])
    memory = reaggregate_trace(PR/'final_exp_memory_001/episodes.jsonl', read(PR/'memory_execution_contract.json'), read(PR/'memory_panel.json'))
    memory_match = memory == read(PUB/'final_exp_memory_001/metrics.json')
    old = read(ROOT/'outputs/research_evaluation/c4_39_v1/final_research_results.json')['rows']
    new = read(PUB/'final_research_results.json')['rows']
    registered_states = {c: Counter(r['status'] for r in rows(PR/'final_exp_memory_001/episodes.jsonl') if r['condition'] == c) for c in ['B0','B2']}
    requests = []
    for path in live.rglob('Request*_after.json'):
        a = read(path); b = read(path.with_name(path.name.replace('_after','_before')))
        stage = path.relative_to(live).parts[0]
        expected = read(PR/'llm_config_1b5.json')['model_identifier'] if stage == 'external_1b5' else DIGEST3
        requests.append({'attempted': a.get('generation_attempted',True), 'json_valid': a['status']=='VALID',
                         'HTTP_200': a.get('HTTP_status')==200,
                         'identity': ('sha256:'+b['identity_before']['digest'].removeprefix('sha256:') == expected
                                      and 'sha256:'+a['identity_after']['digest'].removeprefix('sha256:') == expected),
                         'context': a['identity_after']['context_length']==32768,
                         'model': a['response_envelope']['model']})
    # Actual IDs/clinical original texts remain private. Scan exact source IDs
    # rather than assuming hash-like strings in a public result are patient IDs.
    cases = load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    ids = set()
    for case in cases:
        ids.update([case['patient_id'],case['case_id'],case['agent_visible']['index_encounter_id']])
    public_files = list(PUB.rglob('*.json'))+list(PUB.rglob('*.csv'))+list(DOC.glob('*.md'))
    public_files += [ROOT/'docs/연구 평가/FINAL_EXP_3B_MEMORY_ROUTING_EVALUATION_REPORT.md']
    leaked_files = []
    for path in public_files:
        text = path.read_text(encoding='utf-8-sig')
        if any(identifier in text for identifier in ids if len(identifier)>6):
            leaked_files.append(str(path.relative_to(ROOT)))
    protected = check(read(PR/'protected_input_hashes.json'))
    protocol = read(PUB/'final_exp_protocol_final.json'); pin_check = check(protocol['pins'])
    collections = recheck_collections()
    checks = {'snapshot20_exact_reaggregation': len(pilot_checks)==20 and all(pilot_checks),
        'memory_exact_reaggregation_including_failed_and_not_attempted': memory_match,
        'collection55_exact_reaggregation': collections['status']=='EXACT_MATCH' and collections['verified_received_outcomes']==55,
        'old42_rows_preserved': new[:42]==old,
        'source_ids_absent_from_public_outputs': not leaked_files,
        'protected302_unchanged': protected['status']=='PASS', 'protocol335_unchanged': pin_check['status']=='PASS',
        'all135_request_identity_and_context_verified': len(requests)==135 and all(r['identity'] and r['context'] for r in requests),
        'B2_not_scored_as_zero': all(r['value'] is None for r in new[42:] if r['group']=='FINAL_EXP_Memory_3B' and r['condition']=='B2' and r['metric']!='completed'),
        'no_utility_created': not (PR/'source_qa_development_utility.json').exists(),
        'thesis_addendum7_present': len(list(DOC.glob('FINAL_EXP*.md')))==7}
    out = {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'registered_memory_states':{k:dict(v) for k,v in registered_states.items()},
        'actual_generation':sum(r['attempted'] for r in requests),
        'HTTP_200':sum(r['HTTP_200'] for r in requests), 'JSON_schema_valid':sum(r['json_valid'] for r in requests),
        'actual_model_calls':dict(Counter(r['model'] for r in requests)),
        'Identity_checks_recorded':2*len(requests),
        'Core_semantic_failure':1, 'source_identifier_leaked_files':leaked_files,
        'final_status':'FINAL_EXP_PARTIAL_EVALUATED',
        'tests':{'latest_related':read(PUB/'test_results_final.json'), 'reporting':read(PUB/'reporting_test_results.json'),
                 'nonduplicated_latest_total':189, 'whole_repository_tests':False},
        'files':{str(p.relative_to(ROOT)):sha(p) for p in public_files},
        'posthoc_generation':0,'rerun_existing_models':False}
    if write_seal:
        write(PUB/'verification_final.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='files'}))
    if out['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='reaggregate saved traces without writing or model calls')
    args=parser.parse_args()
    main(write_seal=not args.check_only)
