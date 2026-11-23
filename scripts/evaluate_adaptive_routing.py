"""C4-08 read-only audit or hash-pinned independent routing evaluation. Offline only."""
import argparse
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load, sha
from app.locdi.dataset_loader import load_cases
from app.locdi.routing_collector import load_preparation
from app.locdi.routing_collection_preparation import BANK, SOURCE, CATALOG, TAXONOMY, CONFIG
from app.locdi.routing_training_setup import REGISTRY, HOLDOUT, HOSPITAL
from app.locdi.routing_data_reuse import read
from app.locdi.routing_policy_analysis import independent_split_check
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from app.locdi.clinical_gap_analysis import routing_groups
from validation.routing_correctness_gate import validate_evaluation_gate
from app.locdi.routing_evaluation import (VERSION, REFERENCE_POLICY, paired_route, observable,
                                         evaluate, split_check)
from scripts.enrich_full_cohort_sources import write
from scripts.audit_memory_review_progress import check_hashes


def protected_assets():
    previous = ROOT / 'outputs/private/research_evaluation/c4_07_v1'
    pins = load(previous / 'protected_hashes.json')
    mpath = ROOT / 'outputs/research_evaluation/c4_07_v1/reports/output_manifest.json'
    m = load(mpath)
    if digest({k: v for k, v in m.items() if k != 'fingerprint'}) != m['fingerprint']:
        raise ValueError('PREVIOUS_MANIFEST_FINGERPRINT')
    ipath = previous / 'private_index.json'
    if sha(ipath) != m['private_index_sha256']: raise ValueError('PRIVATE_INDEX_HASH')
    pins.update(m['files']); pins.update(load(ipath)['files'])
    pins[mpath.relative_to(ROOT).as_posix()] = sha(mpath)
    pins[ipath.relative_to(ROOT).as_posix()] = sha(ipath)
    # Human review working files may legitimately advance between phases.
    # Capture their current state, never alter or re-pin a completed artifact.
    for folder in ('c4_05_reviews', 'c4_06_reviews'):
        for p in (ROOT / 'outputs/private/research_evaluation' / folder).glob('**/*'):
            if p.is_file(): pins[p.relative_to(ROOT).as_posix()] = sha(p)
    check_hashes(pins)
    return pins


def case_contract(bundle):
    candidates = sorted(a['department'] for a in bundle['registry']['agents'] if a['role'] == 'specialty')
    cases = []
    for c in bundle['candidates']:
        d = c['decision']
        obs = observable(d['observation'])
        if any(not 0 <= row['available_minute'] <= d['minute'] for row in c['visibility'].values()):
            raise ValueError('UNRELEASED_SOURCE_LEDGER')
        if digest(obs.get('longitudinal_memory', {})) != c['memory_fingerprint']:
            raise ValueError('MEMORY_FINGERPRINT_MISMATCH')
        cases.append({'case_ref': c['case_ref'], 'patient_id': d['patient_id'], 'visit_id': d['visit_id'],
                      'split': d['split'], 'decision_time': (timestamp(c['visit_started_at']) + timedelta(minutes=d['minute'])).isoformat(),
                      'observation': obs, 'candidate_departments': candidates,
                      'source_fingerprint': c['source_fingerprint']})
    split_check(cases)
    return cases


def reference_template(cases):
    return {'policy_version': REFERENCE_POLICY, 'cases_fingerprint': digest(cases),
            'reviews': [{'case_ref': c['case_ref'], 'source_fingerprint': c['source_fingerprint'],
                         'observation_fingerprint': digest(c['observation']),
                         'candidate_fingerprint': digest(sorted(c['candidate_departments'])),
                         'status': 'NOT_REVIEWED', 'accepted_specialties': None,
                         'reviewer': None, 'reference_available_at': None, 'timing_status': 'REFERENCE_TIMING_NOT_REVIEWED',
                         'observation_reviewed': None, 'memory_scope_reviewed': None,
                         'visibility_proxy_reviewed': None, 'reference_evidence_fingerprint': None,
                         'clinical_evidence_rationale': None, 'not_evaluable_reason': None} for c in cases]}


def audit(namespace):
    pub, private = locations(namespace)
    pins = protected_assets()
    bundle = load_preparation('phase24b_candidates_v2')
    paths = dict(bank=BANK, sources=SOURCE, catalog=CATALOG, taxonomy=TAXONOMY,
                 config=CONFIG, registry=REGISTRY, holdout=HOLDOUT)
    for key, path in paths.items():
        if sha(path) != bundle['source_pins'][key]: raise ValueError('PREPARATION_SOURCE_HASH_MISMATCH')
    if load(REGISTRY) != bundle['registry']: raise ValueError('REGISTRY_SNAPSHOT_MISMATCH')
    cases = case_contract(bundle)
    bank = read(BANK); original = load_cases(HOLDOUT)
    voi_path = BANK.parent / 'voi_model.json'; voi = load(voi_path)
    splits = bundle['split_manifest']['patients']
    sets = {k: set(splits[k]) for k in ('train', 'validation', 'test')}
    actual_holdout = {c['patient_id'] for c in original}
    if sets['test'] != actual_holdout: raise ValueError('HOLDOUT_MEMBERSHIP_MISMATCH')
    if any(sets[a] & sets[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('PATIENT_SPLIT_OVERLAP')
    voi_train = set(voi['provenance']['training_patients'])
    bank_patients = {r['training_patient_id'] for r in bank}
    if voi_train != bank_patients: raise ValueError('VOI_TRAIN_BANK_MEMBERSHIP_CHANGED')
    overlap = independent_split_check({'routing_train': sets['train'], 'routing_validation': sets['validation'], 'voi_train': voi_train})
    overlap.update(train_patients=len(sets['train']), validation_patients=len(sets['validation']), holdout_patients=len(actual_holdout),
                   routing_train_holdout_overlap=len(sets['train'] & actual_holdout),
                   routing_validation_holdout_overlap=len(sets['validation'] & actual_holdout),
                   holdout_voi_train_overlap=len(actual_holdout & voi_train), routing_train_voi_train_overlap=len(sets['train'] & voi_train))
    registry = AgentRegistry(bundle['registry'])
    traces = paired_route(cases, registry, static_department='neurology')
    if traces != paired_route(cases, registry, static_department='neurology'): raise ValueError('NONDETERMINISTIC_TRACE')
    for a, b in zip(traces[::2], traces[1::2]):
        if any(a[k] != b[k] for k in ('observation_fingerprint', 'candidate_fingerprint', 'memory_fingerprint', 'source_fingerprint')):
            raise ValueError('PAIRED_INPUT_MISMATCH')
    data, options = actual_gate_inputs()
    gate = validate_evaluation_gate(data, **options)
    groups = routing_groups(data)
    gate_public = {k: gate[k] for k in ('source_reference_ready', 'clinical_review_ready', 'response_policy_ready',
                   'complete_train_groups', 'validation_group_count', 'utility_fit_ready', 'training_readiness', 'blocker_codes')}
    gate_public['supplied_review_reference_count'] = len(options['review_results']['reference_reviews'])
    legacy = []
    for p in sorted((HOSPITAL / 'app/data').glob('*/routing_utility*.json')):
        a = load(p)
        legacy.append({'artifact_ref': 'ArchivedUtility%03d' % (len(legacy) + 1),
                       'registry_matches': a.get('registry') == bundle['registry'],
                       'samples': len(a.get('samples', [])), 'admitted': False})
    refs = load(ROOT / 'outputs/private/routing_collection/phase24b_candidates_v2/reference_candidates.json')
    summary = {'status': 'C4_08_EVALUATION_INFRASTRUCTURE_CONTRACT_TESTED_CLINICAL_BLOCKED',
               'schema_version': VERSION, 'bank_rows': len(bank), 'bank_patients': len(bank_patients),
               'routing_prepared_decisions': len(cases), 'specialty_candidates': 5,
               'holdout_encounters': len(original), 'holdout_patients': len(actual_holdout),
               'explicit_chief_complaint_prepared': sum(bool(c['observation'].get('chief_complaint')) for c in cases),
               'explicit_chief_complaint_holdout': sum(bool(c['agent_visible']['current_initial_state']['doctor_visible_info'].get('chief_complaint')) for c in original),
               'unreviewed_source_reference_candidates': sum(r['clinical_review_status'] != 'REVIEWED' for r in refs),
               'split_audit': overlap, 'collection': {k: groups[k] for k in ('response_count', 'supported', 'unresolved',
                   'correctness_nonnull', 'collection_complete_groups', 'supported_only_complete_groups')},
               'existing_correctness_gate': gate_public, 'archived_utility_inventory': legacy,
               'actual_structural_routing_decisions': len(traces), 'repeat_trace_equality': True,
               'repeat_structural_routing_decisions': len(traces),
               'core_adaptive_route_calls_including_repeat': len(cases) * 2,
               'static_selected': 15, 'adaptive_missing_utility': sum(r['decision']['reason_code'] == 'MISSING_UTILITY' for r in traces),
               'reference_based_metrics': 'NOT_RUN', 'exploratory_development_label_accuracy': 'NOT_RUN',
               'adaptive_without_abstention': 'NOT_IMPLEMENTED_NOT_A_DISTINCT_BASELINE',
               'condition_B_C_relationship': 'SAME_NATIVE_ADAPTIVE_WITH_ABSTENTION',
               'top_k': 'NOT_SUPPORTED_NO_RANKED_SET_CONTRACT',
               'development_exposure': {'original50': 'PREVIOUS_RUNTIME_AND_DEVELOPMENT_EXPOSED_NOT_BLIND',
                                        'selected15': 'USED_FOR_RESPONSE_AND_POLICY_DEVELOPMENT',
                                        'remaining_bank': 'SOURCE_LABEL_AND_VOI_TRAINING_EXPOSED_NOT_BLIND'},
               'blocker_codes': ['INDEPENDENT_SPECIALTY_SET_REFERENCE_MISSING', 'CURRENT_REGISTRY_VERIFIED_UTILITY_MISSING',
                                 'EXISTING_CORRECTNESS_GATE_BLOCKED', 'ACTUAL_SOURCE_VISIBILITY_UNREVIEWED',
                                 'COMBINED_VALIDATION_OVERLAPS_VOI_TRAIN'],
               'protected_files': len(pins), 'protected_changed_or_missing': 0,
               'llm_generations': 0, 'utility_fitting': 0, 'research_evaluation_ready': False}
    source_pins = {str(p): sha(p) for p in list(paths.values()) + [voi_path]}
    check_hashes(pins); check_hashes(source_pins)
    write(private / 'cases.json', cases)
    write(private / 'specialty_reference_blank.json', reference_template(cases))
    write(private / 'source_structural_traces.json', traces)
    write(private / 'existing_gate_result.json', gate)
    write(private / 'protected_hashes.json', pins)
    write(private / 'source_hashes.json', source_pins)
    write(private / 'source_manifest.json', {'cases_fingerprint': digest(cases), 'files': source_pins})
    write(private / 'registry_snapshot.json', bundle['registry'])
    write(pub / 'reports/readiness.json', summary)
    return summary


def locations(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace): raise ValueError('INVALID_NAMESPACE')
    pub = ROOT / 'outputs/research_evaluation' / namespace
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    if pub.exists() or private.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    return pub, private


def pinned(path, expected):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT / 'outputs/private'): raise ValueError('PRIVATE_INPUT_REQUIRED')
    if not expected or sha(p) != expected: raise ValueError('INPUT_HASH_MISMATCH')
    return load(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit'); a.add_argument('--namespace', default='c4_08_v1')
    e = sub.add_parser('evaluate')
    for name in ('cases', 'registry', 'reference', 'utility', 'utility-proof', 'sources'):
        e.add_argument('--' + name, required=True); e.add_argument('--' + name + '-sha256', required=True)
    e.add_argument('--static-department', required=True); e.add_argument('--output-namespace', required=True)
    e.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    def guard(event, values):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'): raise RuntimeError('C4_08_OFFLINE_ONLY')
    sys.addaudithook(guard)
    if args.command == 'audit': result = audit(args.namespace)
    else:
        pub, private = locations(args.output_namespace)
        inputs = {k: pinned(getattr(args, k.replace('-', '_')), getattr(args, k.replace('-', '_') + '_sha256'))
                  for k in ('cases', 'registry', 'reference', 'utility', 'utility-proof', 'sources')}
        if inputs['sources']['cases_fingerprint'] != digest(inputs['cases']): raise ValueError('SOURCE_CASE_BINDING')
        if not inputs['sources'].get('files'): raise ValueError('NONEMPTY_SOURCE_HASHES_REQUIRED')
        check_hashes(inputs['sources']['files'])
        summary, traces = evaluate(inputs['cases'], AgentRegistry(inputs['registry'], inputs['utility']), inputs['reference'],
            static_department=args.static_department, utility_proof=inputs['utility-proof'], evaluation_split=args.split)
        check_hashes(inputs['sources']['files'])
        write(private / 'paired_traces.json', traces); write(pub / 'reports/metrics.json', summary)
        result = {'status': 'INDEPENDENT_REFERENCE_PAIRED_EVALUATION_RECORDED', 'research_evaluation_ready': False}
    print(json.dumps(result, ensure_ascii=True))
