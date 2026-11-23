"""C4-38 transport-free preparation / explicitly approved B0-B2 partial run.

The original50 are retained. Input-size eligible ordinals are selected before
new outputs, never by correctness. Free-text diagnosis is the unchanged ER
contract, not a manufactured candidate catalog or injected Source reference.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, write, sha, check
from app.locdi.llm_setup import load_setup, validate_setup, digest
from app.locdi.dataset_loader import load_cases
from app.locdi.current_ablation_runner import make_contract, run_paired_ablation, reaggregate_trace
from app.locdi.current_ablation_llm_backend import static_preflight, safe_input_bound, make_er_backend, InputContextBudgetError
from app.locdi.new_cohort_execution import approval

PUB = ROOT / 'outputs/research_evaluation/c4_38_v1'
PR = ROOT / 'outputs/private/research_evaluation/c4_38_v1'
DATA = ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl'
RUN = 'c4_38_er_b0_b2_001'
SCOPE = 'C4_38_CURRENT_ABLATION_DEVELOPMENT'


def prepare():
    config = load_setup()
    static = static_preflight(config)
    if static['status'] != 'STATIC_CONTRACT_READY':
        raise ValueError('STATIC_SETUP_BLOCKED')
    prepared = validate_setup(config)['prepared']
    captures = read(PR / 'er_initial_requests_private.json')
    checks = []
    for row in captures:
        r = row['request']
        try:
            bound = safe_input_bound(r['system'], r['user'], r['schema'])
            checks.append({'ordinal': row['ordinal'], 'condition': row['condition'], 'pass': True, 'guard': bound})
        except InputContextBudgetError:
            checks.append({'ordinal': row['ordinal'], 'condition': row['condition'], 'pass': False,
                           'reason': 'FULL_INPUT_CONSERVATIVE_CONTEXT_LIMIT'})
    eligible = sorted({r['ordinal'] for r in checks if all(
        x['pass'] for x in checks if x['ordinal'] == r['ordinal'])})
    if not eligible or any(sum(x['ordinal'] == n for x in checks) != 3 for n in eligible):
        raise ValueError('NO_COMMON_THREE_CONDITION_PANEL')
    cases = load_cases(DATA)
    panel = []
    case_index = {}
    for n in eligible:
        c = cases[n - 1]
        ref = 'CurrentCase%03d' % n
        patient = 'Patient' + digest({'dataset': sha(DATA), 'patient': c['patient_id']})[:16]
        panel.append({'case_ref': ref, 'patient_ref': patient,
            'execution_case': {'case_ref': ref},
            'reference': {'accepted_diagnoses': c['evaluation']['diagnosis_labels'],
                          'accepted_departments': c['evaluation']['accepted_departments_source_policy']}})
        case_index[ref] = n
    common = {'dataset_sha256': sha(DATA), 'patient_split_sha256': digest([r['patient_ref'] for r in panel]),
        'model_bindings': [{'agent_id': b['agent_id'], 'model': config['model'], 'version': config['model_version'],
            'digest': config['model_identifier'], 'prompt_sha256': sha(ROOT / b['prompt'])} for b in config['agent_bindings']],
        'registry_sha256': sha(ROOT / config['registry_path']), 'prompt_sha256': digest(prepared['prompt_fingerprints']),
        'decoding': {'temperature': 0, 'seed': 42, 'max_output_tokens': 1024, 'context': 32768,
                     'timeout_seconds': 120, 'transport': 'OLLAMA_NATIVE_CHAT'},
        'seed': 42, 'memory_source': 'fixed',
        'memory_sha256': digest([cases[n-1]['agent_visible']['past_visits'] for n in eligible]),
        'action_catalog_sha256': sha(ROOT / 'app/locdi/action_registry.py'),
        'diagnosis_catalog': {'mode': 'existing_free_text',
            'validator_sha256': sha(ROOT / 'app/core/hospital_llm_policy.py'), 'source_label_injected': False},
        'source_visibility': {'recorded_time_proxy_accepted': True, 'clinical_release_review': 'NOT_REVIEWED',
                              'future_source': 'existing projection/Action contract only'},
        'time_contract': {'mode': 'existing Hospital Scheduler', 'minute_limit': 120,
                          'timing_defaults': 'existing development simulation, not measured clinical turnaround'},
        'reference_policy': {'name': 'SOURCE_AGREEMENT_ONLY', 'clinical_reference': False,
                             'independent_review': 'NOT_REVIEWED', 'memory_annotation': 'NOT_EVALUATED'},
        'minute_limit': 120, 'repeats': 1,
        'request_budget': {'total': len(panel)*3*12, 'per_episode': 12, 'retry': 0, 'repair': 0},
        'resource_budget': {'wall_seconds': 1800, 'output_bytes': 1024**3, 'memory_bytes': 4*1024**3}}
    contract = make_contract(common, panel)
    input_paths = [DATA, ROOT / 'configs/development_llm.json', ROOT / config['registry_path'],
                   PR / 'er_initial_requests_private.json', PUB / 'full_history_context_preflight.json']
    input_paths += [ROOT / b['prompt'] for b in config['agent_bindings']]
    input_paths += [ROOT / config['patient_prompt']]
    code = ['scripts/run_current_er_ablation.py', 'app/locdi/current_ablation_runner.py',
            'app/locdi/current_ablation_llm_backend.py', 'app/locdi/run_simulation.py',
            'app/locdi/episode_exporter.py', 'app/locdi/case_adapter.py',
            'app/core/longitudinal_runner.py', 'app/core/hospital_llm_policy.py',
            'app/core/longitudinal_memory.py', 'app/locdi/action_registry.py']
    protocol = {'version': 'c4_38.current_er_partial.v1', 'run_id': RUN, 'scope': SCOPE,
        'contract_fingerprint': contract['fingerprint'], 'cohort_ordinals': eligible,
        'registered_original50': 50, 'selected_partial_panel': len(panel), 'planned_episodes': len(panel)*3,
        'selection': 'input-size only, no current response selection; no FullHistory truncation',
        'full50_official': 'BLOCKED', 'initial_payload_checks': checks, 'static_setup': static,
        'generation_budget': common['request_budget'], 'resource_budget': common['resource_budget'],
        'conditions': ['B0', 'B1', 'B2'], 'research_evaluation_ready': False,
        'context_contract': 'num_ctx32768 per request; /api/ps after response before any later request',
        'no_restart_download_warmup': True, 'no_resume': True, 'stop_on_first_technical_failure': True,
        'input_hashes': {str(p.resolve()): sha(p) for p in input_paths},
        'code_hashes': {n: sha(ROOT/n) for n in code}}
    write(PR / 'er_panel.json', panel)
    write(PR / 'er_case_index.json', case_index)
    write(PUB / 'er_execution_contract.json', contract)
    write(PUB / 'er_execution_protocol.json', protocol)
    write(PUB / 'er_dry_run.json', run_paired_ablation(contract, panel))
    print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'contract': contract['fingerprint'],
                      'ordinals': eligible, 'maximum_generation': common['request_budget']['total']}))


def execute(path):
    protocol = read(PUB / 'er_execution_protocol.json')
    contract = read(PUB / 'er_execution_contract.json')
    if contract['fingerprint'] != protocol['contract_fingerprint']:
        raise ValueError('CONTRACT_CHANGED')
    for group in ['input_hashes', 'code_hashes']:
        if check(protocol[group])['changed_or_missing']:
            raise ValueError('PINNED_INPUT_OR_CODE_CHANGED')
    record = read(Path(path))
    approval(record, SCOPE, contract['fingerprint'])
    panel = read(PR / 'er_panel.json')
    cases = load_cases(DATA)
    index = read(PR / 'er_case_index.json')
    backend = make_er_backend({ref: cases[n-1] for ref, n in index.items()}, config=load_setup(),
        contract_fingerprint=contract['fingerprint'], checkpoint_root=PR / RUN / 'requests', authorized=True)
    result = run_paired_ablation(contract, panel, run_one=backend, execute=True, approval_record=record,
        public_root=PUB / RUN, private_root=PR / RUN)
    print(json.dumps({'status': result['status'], 'generation_count': result['generation_count'],
                      'trace_reaggregation': result['trace_reaggregation']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'execute', 'reaggregate'])
    p.add_argument('--approval-file')
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'execute':
        if not args.approval_file:
            p.error('--approval-file required')
        execute(args.approval_file)
    else:
        value = reaggregate_trace(PR/RUN/'episodes.jsonl', read(PUB/'er_execution_contract.json'), read(PR/'er_panel.json'))
        print(json.dumps(value, ensure_ascii=True))
