"""C3-03 offline analysis; exclusive output namespaces, no execution/fitting option."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.routing_policy_analysis import build, test_only_cases
from app.locdi.dataset_freeze import load, sha, verify as verify_freeze
from app.locdi.llm_setup import digest
from app.locdi.source_replay import atomic_rows
from app.locdi.research_protocol_audit import inspect_rc
from app.locdi.structural_release import preservation
from scripts.audit_c3_01 import privacy_check


def findings(outputs):
    r, s = outputs['existing_response_groups'], outputs['split_independence_analysis']
    probes = {r['case']: r for r in outputs['policy_test_only_cases']['cases']}
    unresolved = ('one_unresolved', 'multiple_unresolved', 'all_unresolved')
    checks = {
        'actual_75': (r['decision_count'], r['response_count'], r['supported'], r['unresolved'],
            r['named_diagnosis_unresolved'], r['correctness_nonnull']) == (15, 75, 22, 53, 52, 0),
        'four_group_counts': (r['collected_complete_groups'], r['supported_only_complete_groups'],
            r['independently_reviewed_complete_groups'], r['gate_accepted_fit_groups']) == (15, 0, 0, 0),
        'collection_integrity': not r['collection_integrity']['integrity_issue_codes'],
        'current_gate_blocked': not r['gate']['utility_fit_ready'] and r['gate']['positive_similarity_support']['status'] == 'NOT_CHECKED',
        'frozen_splits': s['membership_recheck'] == 'PASS' and s['routing_validation_voi_train_overlap'] == 5 and
            s['routing_train_validation_overlap'] == 0 and not s['split_reassigned'],
        'supported_test_contract': probes['all_five_supported_independent_test_labels']['development_contract_ready'],
        'unresolved_blocked': all('UNRESOLVED_EXCLUDED_FROM_DIAGNOSIS_FITTING' in probes[n]['blocker_codes'] for n in unresolved),
        'partial_core_rejected': probes['candidate_support_imbalance']['partial_artifact_rejected'],
        'router_abstention': all(probes[n]['decision_status'] == 'abstained' for n in
            ('all_similarity_zero', 'tied_best', 'only_some_candidates_meet_min_samples')),
        'memory_and_timing_blocked': all(probes[n]['blocker_codes'] for n in
            ('same_current_observation_different_memory', 'reference_timing_not_reviewed')),
        'approval_absent': outputs['policy_approval_register']['APPROVED_POLICY'] is None,
        'c3_04_design_only': not outputs['c3_04_handoff']['clinical_admission_allowed']}
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--namespace', default='c3_03_v1')
    parser.add_argument('--regression', action='store_true')
    args = parser.parse_args(argv)
    if args.regression and not args.save:
        parser.error('--regression requires --save in a new namespace')
    def network_guard(event, arguments):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'):
            raise RuntimeError('C3_03_OFFLINE_NO_NETWORK')
    sys.addaudithook(network_guard)
    try:
        root = private = None
        if args.save:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}', args.namespace):
                raise ValueError('INVALID_NAMESPACE')
            root = ROOT/'outputs/research_evaluation'/args.namespace
            private = ROOT/'outputs/private/research_evaluation'/args.namespace
            if (not root.resolve().is_relative_to((ROOT/'outputs/research_evaluation').resolve()) or
                not private.resolve().is_relative_to((ROOT/'outputs/private').resolve())):
                raise ValueError('OUTPUT_REDIRECTED')
            if root.exists() or private.exists():
                raise FileExistsError('NAMESPACE_ALREADY_USED')
            root.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
        outputs, links, evidence = build()
        sys.path.insert(0, str(ROOT/'tests/locdi'))
        from test_routing_correctness_gate import test_only_reviewed_data
        from unittest.mock import patch
        with patch('app.core.routing_utility.fit_utility', side_effect=AssertionError('NO_FITTING')) as fit:
            outputs['policy_test_only_cases'] = test_only_cases(test_only_reviewed_data)
        fit.assert_not_called()
        check = findings(outputs)
        if root:
            for name, value in outputs.items():
                atomic_rows(root, name+'.json', value, jsonl=False)
            atomic_rows(private, 'source_links.json', links, jsonl=False)
        print(json.dumps({'analysis_contract': check, 'stage': 'ANALYSIS_SAVED' if root else 'READ_ONLY',
                          'next': 'REGRESSION' if args.regression else 'NOT_REQUESTED'}), flush=True)
        regression = {'status': 'NOT_CHECKED'}
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            folder = root/'regression'; folder.mkdir(exist_ok=False)
            regression = run_regression(folder)
            atomic_rows(folder, 'result.json', regression, jsonl=False)
        d7, rc = verify_freeze(full=False), inspect_rc()
        kept = preservation(load(ROOT/'outputs/private/c3_03_baseline_hashes_v1.json'))
        document = next((ROOT/'docs').rglob('C3_03_ROUTING_POLICY_AND_INDEPENDENT_VALIDATION_REPORT.md'), None)
        privacy = privacy_check(root, document) if root else {'status': 'NOT_CHECKED_NOT_SAVED'}
        passed = all(r['status'] == 'PASS' for r in (check, d7, rc, kept))
        if root: passed = passed and privacy['status'] == 'PASS'
        if args.regression: passed = passed and regression['status'] == 'PASS'
        ready = passed and root is not None and args.regression
        summary = {'schema_version': 'locdi.c3_03.readiness.v1',
            'C3_03_ROUTING_POLICY_ANALYSIS': 'PASS_DEVELOPMENT_ONLY' if ready else 'INCOMPLETE' if passed else 'FAIL',
            'ROUTING_POLICY_APPROVED': False, 'ROUTING_ARTIFACT_READY': False,
            'CLINICAL_REFERENCE_READY': False, 'OFFICIAL_B0_B4_READY': False,
            'findings': check, 'regression': regression, 'D7': d7, 'RC': rc, 'preservation': kept, 'privacy': privacy,
            'input_evidence': evidence, 'private_index_sha256': sha(private/'source_links.json') if private else None,
            'actual_generation_calls': 0, 'healthbench_grading': 0, 'utility_fitting': 0,
            'voi_probability_estimation': 0, 'clinical_verdicts_generated': 0,
            'test_only_scope': 'Existing regression may exercise in-memory fixture fitting; no real artifact fitting or fitting in C3-03 probes',
            'source_scan_scope': 'Frozen small canonical files, bank229, real75, prior inventories; no large raw/SQLite rescan'}
        if root:
            atomic_rows(root, 'readiness_summary.json', summary, jsonl=False)
        print(json.dumps({k: summary[k] for k in ('C3_03_ROUTING_POLICY_ANALYSIS', 'regression', 'privacy', 'preservation')}, ensure_ascii=True), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        print(json.dumps({'C3_03_ROUTING_POLICY_ANALYSIS': 'FAIL', 'error_type': type(exc).__name__,
                          'error_code': 'ANALYSIS_OR_NEW_NAMESPACE_UNAVAILABLE', 'actual_generation_calls': 0}), flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
