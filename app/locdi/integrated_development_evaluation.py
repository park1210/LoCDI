"""C4-36 passive TEST_ONLY probe over the existing Hospital runtime.

No performance evaluation with fitted artifacts, no new memory likelihoods.
The caller must supply existing explicit TEST_ONLY fixtures.
"""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

from app.core.longitudinal_memory import PatientMemoryStore
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.information_value import InformationPolicy
from app.core.voi_planning import plan_voi
from app.core.diagnostic_belief import posterior
from app.locdi.integrated_runtime import run_integrated
from app.locdi.llm_setup import digest
from app.locdi.integrated_framework_readiness import reject_hidden

TECHNICAL_CONDITIONS = [
    {'id': 'T0', 'memory': False, 'voi': False, 'routing': 'er_only'},
    {'id': 'T1', 'memory': True, 'voi': False, 'routing': 'er_only'},
    {'id': 'T2', 'memory': False, 'voi': True, 'routing': 'er_only'},
    {'id': 'T3', 'memory': True, 'voi': True, 'routing': 'er_only'},
    {'id': 'T4', 'memory': True, 'voi': True, 'routing': 'rule'},
    {'id': 'T5', 'memory': True, 'voi': True, 'routing': 'adaptive_TEST_ONLY'},
]


def technical_protocol(case, source_sha, code_hashes):
    protocol = {'version': 'locdi.integrated_framework.technical.c4_36.v1',
        'purpose': 'TEST_ONLY_CONTRACT_EXECUTION', 'source_case_fingerprint': digest(case),
        'source_sha256': source_sha, 'cohort': 'Original50 ordinal1 ONLY',
        'conditions': deepcopy(TECHNICAL_CONDITIONS), 'source_and_reference_equal_across_conditions': True,
        'reference_used_in_policy': False, 'clinical_scoring': 'NOT_EVALUATED',
        'M5_injected': False, 'existing_test_fixture': 'tests/locdi/runtime_integration_support.py:options_for',
        'new_probabilities_created': False, 'code_hashes': code_hashes,
        'budget': {'runtime_invocations': 6, 'minute_limit_per_condition': 90, 'LLM_generation': 0,
                   'network': 'DISABLED', 'retries': 0},
        'dynamic_hospital_probe': 'BLOCKED_NO_EXISTING_RUNTIME_HOOK',
        'actual_source_performance': 'BLOCKED_NO_COMPATIBLE_MEMORY_LIKELIHOOD_AND_NEW_APPROVAL'}
    protocol['fingerprint'] = digest(protocol)
    return protocol


def run_technical(case, protocol, options_factory, *, output_root):
    if protocol.get('fingerprint') != digest({k: v for k, v in protocol.items() if k != 'fingerprint'}):
        raise ValueError('TECHNICAL_PROTOCOL_HASH')
    if protocol['purpose'] != 'TEST_ONLY_CONTRACT_EXECUTION' or protocol['conditions'] != TECHNICAL_CONDITIONS:
        raise ValueError('TEST_ONLY_FIXED_CONDITIONS_REQUIRED')
    if digest(case) != protocol['source_case_fingerprint']:
        raise ValueError('SOURCE_CASE_CHANGED')
    public, private = [], []
    for condition in protocol['conditions']:
        opts = options_factory(case, routing=condition['routing'] == 'adaptive_TEST_ONLY', voi=condition['voi'])
        if any(not a['version'].startswith('TEST_ONLY') for a in opts.registry_manifest['agents']):
            raise ValueError('TEST_ONLY_BINDINGS_REQUIRED')
        features = deepcopy(opts.runner_options)
        features.update(memory_mode='selective' if condition['memory'] else 'none',
                        top_k=5 if condition['memory'] else None)
        if condition['routing'] == 'rule':
            features.update(routing_mode='rule', primary_router='rule_similarity')
        opts = replace(opts, runner_options=features)
        observed = []
        original = InformationPolicy.propose
        def observe(policy, runtime, snapshot, **kwargs):
            reject_hidden(snapshot)
            observed.append(deepcopy(snapshot))
            return original(policy, runtime, snapshot, **kwargs)
        with patch('app.locdi.run_simulation.run_with_recovery', wraps=run_with_recovery) as runner, \
             patch('app.core.longitudinal_runner.HospitalParallelScheduler', wraps=HospitalParallelScheduler) as scheduler, \
             patch.object(PatientMemoryStore, 'retrieve', autospec=True, side_effect=PatientMemoryStore.retrieve) as retrieval, \
             patch('app.core.information_value.plan_voi', wraps=plan_voi) as planner, \
             patch('app.core.information_outcomes.posterior', wraps=posterior) as bayes, \
             patch.object(InformationPolicy, 'propose', new=observe):
            result = run_integrated(case, options=opts, test_only=True,
                output_root=output_root, export_id='C436_TEST_ONLY_' + condition['id'])
        episode, audit = result['episode'], result['audit']
        consistent = bool(episode and audit and episode['workflow'] == audit['workflow']
                          and episode['result'] == audit['terminal'])
        public.append({'condition': condition, 'status': result['status'], 'blocker_codes': result['blocker_codes'],
            'calls': {'runner': runner.call_count, 'scheduler': scheduler.call_count,
                      'memory_retrieve': retrieval.call_count, 'planner': planner.call_count,
                      'posterior': bayes.call_count, 'voi_propose': len(observed)},
            'memory_records_delivered_to_voi': [len(s.get('longitudinal_memory', {}).get('records', [])) for s in observed],
            'memory_selected_count': len(result['raw']['execution']['visits'][0]['memory_context']['records'])
                if result.get('raw') and result['raw'].get('execution') and 'visits' in result['raw']['execution'] else None,
            'voi_matched_counts': [v['matched_outcome_count'] for v in audit['voi']] if audit else [],
            'routing_statuses': [v['status'] for v in audit['routing']] if audit else [],
            'runtime_integration_ready': result['readiness']['runtime_integration_ready'],
            'episode_audit_consistent': consistent, 'terminal': deepcopy(episode['result']) if episode else None,
            'clinical_metrics': 'NOT_EVALUATED', 'performance_metrics': None,
            'belief_memory_causality': 'NOT_ESTIMATED_TEST_FIXTURE_HAS_NO_MEMORY_LIKELIHOOD'})
        private.append({'condition': condition['id'], 'voi_input_snapshots': observed,
                        'raw_result': result['raw']})
    return {'purpose': protocol['purpose'], 'conditions': public,
            'all_technical_completed': all(r['status'] == 'completed' and r['runtime_integration_ready']
                                            and r['episode_audit_consistent'] for r in public),
            'clinical_metrics': 'NOT_EVALUATED', 'actual_performance_evaluations': 0,
            'new_LLM_generation': 0, 'fitting': 0}, private


def block_source_evaluation():
    """No unimplemented research execution is represented as an executable experiment."""
    return {'status': 'BLOCKED', 'executed': False, 'blocker_codes': [
        'NO_VALIDATED_MEMORY_TO_M5_LIKELIHOOD', 'NO_CLEARED_HOSPITAL_SOURCE_ENCODING',
        'MISSING_ROUTING_UTILITY', 'NEW_INTEGRATED_EVALUATION_APPROVAL_NOT_PROVIDED'],
        'C4_35_APPROVAL_REUSABLE': False}
