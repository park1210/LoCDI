"""TEST_ONLY records in memory. No clinical answers, generation or fitting."""
from copy import deepcopy
import inspect
import json
import unittest
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import fit_utility, observed_tokens
from app.locdi.llm_setup import digest
from app.locdi.routing_training_setup import training_preflight
from scripts.prepare_routing_training_data import main
from validation.routing_training_data import (validate_training_data, validate_split_manifest,
                                             prepare_collection, to_fit_rows, _leaves)
from test_routing_voi_readiness import test_only_registry


def test_only_data():
    registry = test_only_registry()
    review = {'status': 'REVIEWED', 'reviewer_ref': 'TEST_ONLY_INDEPENDENT_REVIEWER',
              'procedure_version': 'TEST_ONLY_V1', 'reviewed_at': '2026-10-29T00:00:00+00:00'}
    observation = {'chief_complaint': 'TEST_ONLY_SIGNAL', 'elapsed_minutes': 3,
                   'history': [], 'objective_findings': [], 'longitudinal_memory': {'records': []}}
    agents = [a for a in registry['agents'] if a['role'] == 'specialty']
    decision = {'patient_id': 'TEST_ONLY_TRAIN', 'visit_id': 'TEST_ONLY_VISIT', 'decision_id': 'TEST_ONLY_DECISION',
                'minute': 3, 'split': 'train', 'source_ref': 'TEST_ONLY_SOURCE', 'observation': observation,
                'registry_fingerprint': digest(registry), 'tokens': observed_tokens(observation),
                'allowed_actions': ['ask_patient', 'diagnose'], 'generation': {
                    'model_identifier': 'sha256:' + '1' * 64, 'temperature': 0, 'seed': 42, 'max_output_tokens': 1024,
                    'prompt_fingerprints': {a['agent_id']: '2' * 64 for a in agents}}}
    source = {'source_ref': 'TEST_ONLY_SOURCE', 'purpose': 'TEST_ONLY',
              **{k: deepcopy(decision[k]) for k in ('patient_id', 'visit_id', 'decision_id', 'minute', 'observation')},
              'visibility': {p: 0 for p in _leaves(observation)}, 'source_fingerprint': '3' * 64, 'review': deepcopy(review)}
    split = {'schema_version': 'locdi.routing_split.v1', 'dataset_fingerprint': '4' * 64,
             'holdout_fingerprint': digest(['TEST_ONLY_HOLDOUT']),
             'patients': {'train': ['TEST_ONLY_TRAIN'], 'validation': ['TEST_ONLY_VALIDATION'], 'test': ['TEST_ONLY_HOLDOUT']},
             'decisions': [{k: decision[k] for k in ('patient_id', 'decision_id', 'split')}]}
    label = {'label_ref': 'TEST_ONLY_LABEL', 'source_ref': 'TEST_ONLY_REFERENCE_SOURCE', 'purpose': 'TEST_ONLY',
             'source_fingerprint': '5' * 64, 'patient_id': decision['patient_id'], 'decision_id': decision['decision_id'],
             'label_fingerprint': '6' * 64, 'clinical_review_status': 'REVIEWED', 'review': deepcopy(review),
             'independent_reference': True, 'evaluation_method': 'independent_reference_match'}
    data = {'schema_version': 'locdi.routing_collection.v1', 'purpose': 'TEST_ONLY', 'registry': deepcopy(registry),
            'split_manifest': split, 'decisions': [decision], 'outcomes': [], 'judgments': []}
    fingerprint = prepare_collection(decision, registry, source)['configuration_fingerprint']
    for index, agent in enumerate(agents):
        response_ref = 'TEST_ONLY_RESPONSE_%d' % index
        response_fp = digest(response_ref)
        data['outcomes'].append({'response_ref': response_ref, 'patient_id': decision['patient_id'],
            'decision_id': decision['decision_id'], 'agent_id': agent['agent_id'], 'model': agent['model'], 'version': agent['version'],
            'purpose': 'TEST_ONLY', 'source_ref': decision['source_ref'], 'configuration_fingerprint': fingerprint,
            'observation_fingerprint': digest(observation), 'tokens': deepcopy(decision['tokens']),
            'generation': deepcopy(decision['generation']), 'execution_status': 'SUCCEEDED', 'response_contract_status': 'VALID',
            'response_fingerprint': response_fp, 'recorded_at': '2026-10-28T00:00:00+00:00',
            'observed_model_identifier': decision['generation']['model_identifier']})
        data['judgments'].append({'response_ref': response_ref, 'response_fingerprint': response_fp,
            'patient_id': decision['patient_id'], 'decision_id': decision['decision_id'], 'agent_id': agent['agent_id'],
            'purpose': 'TEST_ONLY', 'status': 'REVIEWED', 'correct': index == 0, 'label_ref': label['label_ref'],
            'label_fingerprint': label['label_fingerprint'], 'evaluation_method': label['evaluation_method'], 'review': deepcopy(review)})
    options = dict(registry_manifest=registry, source_catalog=[source], label_catalog=[label],
                   holdout_patient_ids=['TEST_ONLY_HOLDOUT'], dataset_fingerprint='4' * 64, allow_test_only=True)
    return data, options


class RoutingTrainingContracts(unittest.TestCase):
    def setUp(self):
        self.data, self.options = test_only_data()

    def validate(self):
        return validate_training_data(self.data, **self.options)

    def blocked(self, code=None):
        result = to_fit_rows(self.data, **self.options)
        self.assertEqual(result['report']['training_readiness'], 'BLOCKED')
        self.assertIsNone(result['rows'])
        if code:
            self.assertIn(code, result['report']['blocker_codes'])

    def test_complete_group_private_rows_match_actual_fit_signature_without_call(self):
        before = deepcopy((self.data, self.options))
        with patch('app.core.routing_utility.fit_utility', side_effect=AssertionError('Fitting forbidden')) as fit:
            result = to_fit_rows(self.data, **self.options)
        fit.assert_not_called()
        self.assertEqual(result['report']['training_readiness'], 'TEST_ONLY_CONTRACT_VALID')
        self.assertEqual(result['report']['complete_group_count'], 1)
        self.assertFalse(result['report']['research_readiness'])
        self.assertEqual(len(result['rows']), 2)
        inspect.signature(fit_utility).bind(result['rows'], AgentRegistry(self.options['registry_manifest']))
        self.assertEqual(before, (self.data, self.options))
        self.assertEqual({type(r['correct']) for r in result['rows']}, {bool})

    def test_plan_has_all_specialties_same_input_no_fabricated_responses(self):
        plan = prepare_collection(self.data['decisions'][0], self.options['registry_manifest'], self.options['source_catalog'][0])
        self.assertEqual(plan['status'], 'PLANNED_NOT_EXECUTED')
        self.assertEqual(len(plan['tasks']), 2)
        self.assertEqual(plan['tasks'][0]['shared_input'], plan['tasks'][1]['shared_input'])
        self.assertNotIn('correct', json.dumps(plan))
        self.assertNotIn('label_ref', json.dumps(plan))

    def test_missing_specialty_rejects_entire_group(self):
        self.data['outcomes'].pop()
        self.blocked('SPECIALTY_GROUP_INCOMPLETE_OR_DUPLICATE')

    def test_different_observation_tokens_and_decoding_block(self):
        for key, value in [('observation_fingerprint', '0' * 64), ('tokens', ['TEST_ONLY_OTHER']), ('generation', {})]:
            with self.subTest(key=key):
                self.data, self.options = test_only_data()
                self.data['outcomes'][0][key] = value
                self.blocked('SHARED_INPUT_OR_SOURCE_BINDING_MISMATCH')

    def test_duplicate_agent_and_response_rejected(self):
        self.data['outcomes'][1]['agent_id'] = self.data['outcomes'][0]['agent_id']
        self.blocked('SPECIALTY_GROUP_INCOMPLETE_OR_DUPLICATE')
        self.data, self.options = test_only_data()
        self.data['outcomes'].append(deepcopy(self.data['outcomes'][0]))
        self.blocked('COLLECTION_OR_CATALOG_SCHEMA_INVALID')

    def test_patient_holdout_and_visit_cross_split_overlap(self):
        self.data['split_manifest']['patients']['train'].append('TEST_ONLY_HOLDOUT')
        self.blocked('PATIENT_SPLIT_OVERLAP_OR_DUPLICATE')
        self.data, self.options = test_only_data()
        self.data['split_manifest']['patients']['validation'].append('TEST_ONLY_TRAIN')
        self.blocked('PATIENT_SPLIT_OVERLAP_OR_DUPLICATE')

    def test_correctness_missing_unreviewed_nonboolean_never_imputed_false(self):
        for change in ('missing', 'unreviewed', 'nonboolean'):
            self.data, self.options = test_only_data()
            judgment = self.data['judgments'][0]
            if change == 'missing':
                judgment.pop('correct')
            elif change == 'unreviewed':
                judgment['status'] = 'NOT_REVIEWED'
            else:
                judgment['correct'] = 0
            self.blocked('CORRECTNESS_NOT_REVIEWED')

    def test_gold_nested_and_extra_top_level_input_block(self):
        self.data['decisions'][0]['observation']['history'] = [{'gold': 'TEST_ONLY_PRIVATE'}]
        self.blocked('OBSERVATION_VISIBILITY_OR_COLLECTION_CONTRACT_INVALID')
        self.data, self.options = test_only_data()
        self.data['decisions'][0]['gold'] = 'TEST_ONLY_PRIVATE'
        self.blocked('OBSERVATION_VISIBILITY_OR_COLLECTION_CONTRACT_INVALID')

    def test_future_and_incomplete_visibility_block(self):
        self.options['source_catalog'][0]['visibility']['/chief_complaint'] = 4
        self.blocked('OBSERVATION_VISIBILITY_OR_COLLECTION_CONTRACT_INVALID')
        self.data, self.options = test_only_data()
        self.options['source_catalog'][0]['visibility'].pop('/chief_complaint')
        self.blocked('OBSERVATION_VISIBILITY_OR_COLLECTION_CONTRACT_INVALID')

    def test_registry_model_version_digest_mismatch(self):
        self.data['registry']['agents'][1]['version'] = 'TEST_ONLY_WRONG'
        self.blocked('REGISTRY_SNAPSHOT_OR_DATA_SCHEMA_MISMATCH')
        for key in ('model', 'version', 'observed_model_identifier'):
            self.data, self.options = test_only_data()
            self.data['outcomes'][0][key] = 'TEST_ONLY_WRONG'
            self.blocked()

    def test_source_label_response_link_mismatch(self):
        self.data['outcomes'][0]['source_ref'] = 'TEST_ONLY_OTHER'
        self.blocked('SHARED_INPUT_OR_SOURCE_BINDING_MISMATCH')
        self.data, self.options = test_only_data()
        self.data['judgments'][0]['response_fingerprint'] = '0' * 64
        self.blocked('INDEPENDENT_LABEL_OR_RESPONSE_LINK_INVALID')

    def test_malformed_split_duplicate_decision_and_missing_holdout(self):
        for change in ('schema', 'duplicate', 'holdout'):
            self.data, self.options = test_only_data()
            split = self.data['split_manifest']
            if change == 'schema':
                split['schema_version'] = 'WRONG'
            elif change == 'duplicate':
                split['decisions'].append(deepcopy(split['decisions'][0]))
            else:
                split['patients']['test'] = []
            self.blocked()

    def test_independent_review_required_not_department_or_self_confidence(self):
        for method in ('accepted_department_match', 'response_returned', 'scheduler_completed', 'self_confidence'):
            self.data, self.options = test_only_data()
            self.options['label_catalog'][0]['evaluation_method'] = method
            self.data['judgments'][0]['evaluation_method'] = method
            self.blocked('INDEPENDENT_LABEL_OR_RESPONSE_LINK_INVALID')

    def test_test_only_rejected_by_default_and_cli_never_fits(self):
        self.options.pop('allow_test_only')
        self.blocked('TEST_ONLY_OR_PURPOSE_NOT_AUTHORIZED')
        with patch('scripts.prepare_routing_training_data.training_preflight', return_value={'training_readiness': 'BLOCKED'}), patch('builtins.print'):
            self.assertEqual(main([]), 2)
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                main(['--execute'])

    def test_default_real_sources_block_without_private_values(self):
        with patch('app.locdi.ollama_adapter._backend', side_effect=AssertionError('Generation forbidden')):
            report = training_preflight()
        self.assertEqual(report['training_readiness'], 'BLOCKED')
        self.assertEqual(report['complete_group_count'], 0)
        self.assertEqual(report['train_patient_count'], 0)
        self.assertEqual(report['evaluation_patient_count'], 50)
        self.assertEqual(report['utility_fit_count'], 0)
        self.assertEqual(report['generation_requests'], 0)
        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(v) for v in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(v) for v in value))
            return set()
        for forbidden in ('patient_id', 'diagnosis_labels', 'source_event_store'):
            self.assertNotIn(forbidden, keys(report))
        self.assertNotIn('TEST_ONLY_SIGNAL', json.dumps(report))

    def test_failed_response_and_missing_labels_block(self):
        self.data['outcomes'][0]['execution_status'] = 'FAILED'
        self.blocked('RESPONSE_NOT_EVALUABLE')
        self.data, self.options = test_only_data()
        self.options['label_catalog'] = []
        self.blocked('INDEPENDENT_LABEL_OR_RESPONSE_LINK_INVALID')

    def test_private_validation_and_test_groups_are_never_returned_as_train(self):
        # Reuse abstract in-memory fixture, reserving a separate holdout decision.
        extra, extra_options = test_only_data()
        substitutions = {'TEST_ONLY_TRAIN': 'TEST_ONLY_HOLDOUT', 'TEST_ONLY_DECISION': 'TEST_ONLY_TEST_DECISION',
                         'TEST_ONLY_SOURCE': 'TEST_ONLY_TEST_SOURCE', 'TEST_ONLY_LABEL': 'TEST_ONLY_TEST_LABEL',
                         'TEST_ONLY_RESPONSE_': 'TEST_ONLY_TEST_RESPONSE_'}
        def rename(value):
            text = json.dumps(value)
            for before, after in substitutions.items():
                text = text.replace(before, after)
            return json.loads(text)
        decision = rename(extra['decisions'][0]); decision['split'] = 'test'
        source = rename(extra_options['source_catalog'][0])
        label = rename(extra_options['label_catalog'][0])
        plan = prepare_collection(decision, self.options['registry_manifest'], source)
        outcomes = rename(extra['outcomes']); judgments = rename(extra['judgments'])
        for outcome, judgment in zip(outcomes, judgments):
            outcome['configuration_fingerprint'] = plan['configuration_fingerprint']
            outcome['response_fingerprint'] = digest(outcome['response_ref'])
            judgment['response_fingerprint'] = outcome['response_fingerprint']
        self.data['decisions'].append(decision)
        self.data['outcomes'].extend(outcomes); self.data['judgments'].extend(judgments)
        self.data['split_manifest']['decisions'].append({k: decision[k] for k in ('patient_id', 'decision_id', 'split')})
        self.options['source_catalog'].append(source); self.options['label_catalog'].append(label)
        result = to_fit_rows(self.data, **self.options)
        self.assertEqual(result['report']['complete_group_count'], 2)
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual({r['patient_id'] for r in result['rows']}, {'TEST_ONLY_TRAIN'})
        self.assertEqual({r['split'] for r in result['rows']}, {'train'})

    def test_memory_cross_patient_future_visit_and_cutoff_rejected(self):
        decision = deepcopy(self.data['decisions'][0])
        source = deepcopy(self.options['source_catalog'][0])
        source['visit_started_at'] = '2026-10-28T00:00:00+00:00'
        record = {'patient_id': decision['patient_id'], 'visit_id': 'TEST_ONLY_PAST',
                  'visit_started_at': '2026-10-27T00:00:00+00:00', 'available_at': '2026-10-27T00:00:00+00:00',
                  'text': 'TEST_ONLY_MEMORY'}
        for field, value in [('patient_id', 'TEST_ONLY_OTHER_PATIENT'), ('visit_id', decision['visit_id']),
                             ('available_at', '2026-10-29T00:00:00+00:00')]:
            bad = deepcopy(record); bad[field] = value
            decision['observation']['longitudinal_memory']['records'] = [bad]
            decision['tokens'] = observed_tokens(decision['observation'])
            source['observation'] = deepcopy(decision['observation'])
            source['visibility'] = {p: 0 for p in _leaves(source['observation'])}
            with self.assertRaises(ValueError):
                prepare_collection(decision, self.options['registry_manifest'], source)


if __name__ == '__main__':
    unittest.main()
