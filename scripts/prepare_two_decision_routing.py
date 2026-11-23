"""C4-13 offline protocol freeze. No model client, approval or clinical labels."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import load_preparation, PRIVATE_ROOT
from app.locdi.routing_two_decision_collection import build_plan, signed, verify_fingerprint, verify_pins, SELECTED, evaluation_protocol
from app.locdi.routing_data_reuse import read
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from validation.routing_correctness_gate import validate_evaluation_gate
from scripts.enrich_full_cohort_sources import write


def prepare(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace): raise ValueError('INVALID_NAMESPACE')
    public = ROOT / 'outputs/research_evaluation' / namespace
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    if public.exists() or private.exists(): raise ValueError('NO_OVERWRITE')
    prior = ROOT / 'outputs/private/research_evaluation/c4_12_v1'
    mp = ROOT / 'outputs/research_evaluation/c4_12_v1/output_manifest.json'
    manifest = load(mp); verify_fingerprint(manifest)
    ip = prior / 'private_index.json'
    if sha(ip) != manifest['private_index_sha256']: raise ValueError('PARENT_PRIVATE_INDEX')
    index = load(ip); verify_fingerprint(index)
    pins = load(prior / 'protected_hashes.json'); pins.update(manifest['files']); pins.update(index['files'])
    for p in (mp, ip): pins[p.relative_to(ROOT).as_posix()] = sha(p)
    sources = load(prior / 'source_hashes.json')
    verify_pins(pins); verify_pins(sources)
    print('C4-12 protected/source hashes verified', flush=True)
    bundle = load_preparation('phase24b_candidates_v2')
    parent = load(ROOT / 'outputs/research_evaluation/c4_12_v1/next_collection_plan.json')
    plan = build_plan(bundle, parent)
    facts = {f['case_ref']: f for f in read(prior / 'source_decision_eligibility.jsonl')}
    source_checks = load(ROOT / 'outputs/research_evaluation/c4_12_v1/source_and_response_verification.json')
    summaries = []; detailed = []
    for ref in SELECTED:
        f = facts[ref]
        if (not f['patient_encounter_link'] or not f['native_encounter_start_matches'] or f['native_selected_after_decision']
                or f['definitely_future_selected_reference'] or not next(x for x in source_checks['csv_native_join_checks'] if x['case_ref'] == ref)['csv_native_multiset_equal']):
            raise ValueError('SELECTED_SOURCE_NOT_VERIFIED')
        summaries.append({'case_ref': ref, 'source_status': 'SOURCE_VERIFIED', 'reference_status': 'REFERENCE_REVIEW_REQUIRED',
            'proxy_scope_status': 'NOT_ACCEPTED', 'not_evaluable': None,
            'patient_encounter_link_verified': True, 'recorded_before_decision': True,
            'current_encounter_conditions': f['current_encounter_condition_count'], 'actual_release_recorded': f['actual_release_recorded'],
            'proxy_entries': f['proxy_entries'], 'clinical_correctness': None})
        detailed.append(deepcopy(f))
    selected = [c for c in bundle['candidates'] if c['case_ref'] in SELECTED]
    version = signed({'schema_version': 'locdi.routing_dataset.c4_13.v1', 'parent_preparation': 'phase24b_candidates_v2',
        'parent_plan_fingerprint': parent['fingerprint'], 'registered_decisions': 15,
        'all_registered': [{'case_ref': c['case_ref'], 'split': c['decision']['split'],
            'collection_scope': 'NEW_WHOLE_GROUP' if c['case_ref'] in SELECTED else 'VALIDATION_PRESERVED_NOT_RECOLLECTED' if c['decision']['split'] == 'validation' else 'TRAIN_NOT_SELECTED',
            'reason': 'FROZEN_C4_12_SOURCE_ONLY_FIRST_TWO_TRAIN_RULE' if c['case_ref'] in SELECTED else
                'ORIGINAL_VALIDATION_PATIENT_SPLIT_PRESERVED' if c['decision']['split'] == 'validation' else
                'FUTURE_SELECTED_CONDITION_NOT_CURRENT_REFERENCE' if c['case_ref'] == 'Candidate010' else 'OUTSIDE_FIRST_TWO_ELIGIBLE_SOURCE_ORDINALS',
            'source_fingerprint': c['source_fingerprint'], 'input_fingerprint': c['shared_input_fingerprint']} for c in bundle['candidates']],
        'new_collection_train': SELECTED, 'unselected_train_count': 8, 'validation_count': 5,
        'registry_fingerprint': plan['registry_fingerprint'], 'catalog_fingerprint': plan['catalog_fingerprint'],
        'development_exposed': True, 'independent_holdout': False,
        'baseline_use': 'FROZEN_75_RESPONSE_BASELINE_ONLY_NO_TRAIN_RESPONSE_MIXING',
        'new_use': 'TEN_NEW_ATTEMPTS_AND_WHOLE_GROUP_READINESS_ONLY_NO_AUTOMATIC_FIT',
        'evaluable_group': 'Five distinct registered Agents, same Decision snapshot, supported schema-valid actual responses plus independent current Reference/correctness and unchanged Gate',
        'original_registration_deleted': 0})
    bindings = {}
    for folder in ('prompts',):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file(): bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    for name in ['configs/agent_registry.specialty_development.json', 'configs/development_specialty_llm.json',
        'app/locdi/routing_two_decision_collection.py', 'scripts/collect_two_decision_routing.py',
        'app/locdi/ollama_adapter.py', 'app/llm/ollama_client.py', 'app/locdi/routing_collector.py',
        'app/locdi/routing_collection_preparation.py', 'validation/routing_correctness_gate.py',
        'configs/evaluation/routing_policy_v1.json', 'schemas/routing_review_results_v1.schema.json']:
        bindings[name] = sha(ROOT / name)
    for p in (PRIVATE_ROOT / 'phase24b_candidates_v2').glob('*.json'):
        bindings[p.relative_to(ROOT).as_posix()] = sha(p)
    data, options = actual_gate_inputs(); options['min_samples'] = 2
    gate = validate_evaluation_gate(data, **options)
    public.mkdir(parents=True); private.mkdir(parents=True)
    write(private / 'protected_hashes.json', pins); write(private / 'source_hashes.json', sources)
    write(private / 'selected_source_reference_facts.json', detailed)
    write(private / 'selected_snapshots.json', selected)
    write(private / 'existing_gate_result.json', gate)
    write(private / 'bindings.json', {'source_pins': sources, 'binding_pins': bindings})
    write(private / 'plan.json', plan)
    write(private / 'protocol_manifest.json', signed({'schema_version': 'locdi.c4_13.protocol.v1',
        'files': {n: sha(private / n) for n in ('plan.json', 'bindings.json')}}))
    write(public / 'reference_feasibility.json', {'selected': summaries, 'clinical_reference_ready': False,
        'automatic_facts': ['Patient/Encounter linkage', 'Native recorded onset and explicit System/Code', 'Source/response integrity'],
        'human_decisions_pending': ['Current diagnosis task/scope', 'Multiple conditions and past/current distinction', 'Actual visibility or clinical proxy scope', 'Independent response correctness']})
    write(public / 'dataset_version_manifest.json', version)
    write(public / 'reference_protocol.json', evaluation_protocol())
    write(public / 'collection_plan.json', plan)
    write(public / 'timeout_mitigation_options.json', {'baseline': {'timeout_seconds': 120, 'max_output_tokens': 1024,
        'recorded_context': 4096, 'recorded_input_tokens': 1231, 'recorded_load_seconds': 50.6,
        'gpu_discovery_watchdog_timeout': True, 'http_500_and_cancel': True},
        'selected_option': 'UNCHANGED_BASELINE_NO_WARMUP_NO_SETTINGS_CHANGE',
        'unproven_causes': ['Input overflow', 'OOM', 'JSON schema error'],
        'future_options_not_executed': [{'option': 'Warm-up', 'extra_generations': 'REQUIRES_SEPARATE_BUDGET_AND_APPROVAL', 'effect': 'NOT_VERIFIED'},
            {'option': 'Timeout change', 'baseline_seconds': 120, 'proposed_seconds': None, 'effect': 'NOT_VERIFIED_REQUIRES_NEW_PINNED_PLAN'},
            {'option': 'Context/server change', 'proposed_value': None, 'effect': 'NOT_AUTHORIZED_NOT_EXECUTED'}],
        'new_observation_contract': {'request_wall_seconds': 'MEASURED_MONOTONIC', 'model_loading_seconds': 'NULL_UNLESS_EXPOSED_NO_INFERENCE_FROM_WALL',
            'http_status': 'REAL_HTTP_RESPONSE_HOOK_OR_NULL', 'raw_received_json_schema_supported': 'SEPARATE_COUNTS'}})
    write(public / 'utility_fit_requirements.json', {'existing_baseline_gate': {k: gate[k] for k in ('training_readiness', 'blocker_codes', 'complete_train_groups', 'utility_fit_ready')},
        'new_data_gate': 'NOT_RUN_NO_NEW_RESPONSES_OR_REVIEW_PACKAGE', 'min_samples': 2,
        'requirements': ['Two distinct complete supported Train Decisions each five Agents', 'Current independent Reference and response review',
            'Unchanged Registry/Model/Prompt/source lineage', 'No patient split overlap', 'Positive Jaccard support for every candidate Agent',
            'Existing Correctness Gate AND legacy Validator', 'No baseline/new response replacement', 'Full original15 registration and prospective scope retained'],
        'source_proxy': 'SEPARATE_EXPLICIT_REVIEWED_PROXY_PATH_NOT_APPROVED_NOT_INDEPENDENT_GOLD',
        'followup_function': 'app.locdi.routing_two_decision_collection.assess_run', 'fitting_called': False})
    write(public / 'verification.json', {'status': 'C4_13_REFERENCE_BLOCKED', 'collector_contract': 'PENDING_TESTS',
        'source_verified_decisions': 2, 'registered_decisions': 15, 'generation_requests': 0, 'fitting_calls': 0,
        'clinical_judgments_created': 0, 'protected_files': len(pins), 'source_files': len(sources),
        'plan_fingerprint': plan['fingerprint'], 'research_evaluation_ready': False})
    return {'status': 'C4_13_REFERENCE_BLOCKED', 'plan_fingerprint': plan['fingerprint'], 'generation_requests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--namespace', default='c4_13_v1')
    print(json.dumps(prepare(parser.parse_args().namespace), ensure_ascii=True))
