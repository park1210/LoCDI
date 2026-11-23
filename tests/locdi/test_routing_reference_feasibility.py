"""TEST_ONLY deterministic facts, never clinical review or live generation."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from scripts.audit_routing_reference_feasibility import (time_fact, code_fact, group_fact,
    select_prospective, classify_attempt, inventory_get, ROOT)
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility
from test_routing_voi_readiness import test_only_registry, test_only_routing
from app.locdi import live_handoff as live


class ReferenceFeasibilityContracts(unittest.TestCase):
    def test_future_day(self):
        self.assertEqual(time_fact('2020-01-05', '2020-01-01T12:01:00Z'), 'DEFINITELY_AFTER_DECISION')

    def test_candidate010_four_day_shape(self):
        self.assertEqual(time_fact('2020-05-05', '2020-05-01T00:01:00Z'), 'DEFINITELY_AFTER_DECISION')

    def test_same_day_not_automatically_available(self):
        self.assertEqual(time_fact('2020-01-01', '2020-01-01T12:01:00Z'), 'SAME_DAY_DATE_ONLY_AMBIGUOUS')

    def test_prior_day_still_not_release_evidence(self):
        self.assertEqual(time_fact('2019-12-31', '2020-01-01T12:01:00Z'), 'RECORDED_BEFORE_DECISION_RELEASE_UNKNOWN')

    def test_explicit_time_future(self):
        self.assertEqual(time_fact('2020-01-01T12:02:00Z', '2020-01-01T12:01:00Z'), 'DEFINITELY_AFTER_DECISION')

    def test_explicit_time_by_cutoff(self):
        self.assertEqual(time_fact('2020-01-01T12:01:00Z', '2020-01-01T12:01:00Z'), 'RECORDED_BY_DECISION_RELEASE_UNKNOWN')

    def test_missing_is_not_false(self):
        self.assertEqual(time_fact(None, '2020-01-01T12:01:00Z'), 'MISSING')
        self.assertIsNone(code_fact([], [])['correct'])

    def test_timezone_required(self):
        with self.assertRaises(ValueError): time_fact('2020-01-01', '2020-01-01T12:01:00')

    def test_code_match_not_correctness(self):
        result = code_fact([['TEST_ONLY_SYSTEM', 'A']], [['TEST_ONLY_SYSTEM', 'A']])
        self.assertEqual(result['state'], 'SAME_EXPLICIT_CODE'); self.assertIsNone(result['correct'])

    def test_code_difference_not_error(self):
        result = code_fact([['TEST_ONLY_SYSTEM', 'B']], [['TEST_ONLY_SYSTEM', 'A']])
        self.assertEqual(result['state'], 'DIFFERENT_EXPLICIT_CODE'); self.assertIsNone(result['correct'])

    def test_multiple_patient_conditions_not_unique_gold(self):
        result = code_fact([['TEST_ONLY_SYSTEM', 'B']], [['TEST_ONLY_SYSTEM', 'A'], ['TEST_ONLY_SYSTEM', 'B']])
        self.assertEqual(result['state'], 'SAME_EXPLICIT_CODE'); self.assertIsNone(result['correct'])

    def test_ambiguous_code_no_synonym_approval(self):
        result = code_fact([['TEST_ONLY_SYSTEM', 'A'], ['TEST_ONLY_SYSTEM', 'B']], [])
        self.assertIn('REVIEW_REQUIRED', result['state']); self.assertIsNone(result['correct'])

    def outcomes(self):
        return [{'agent_id': 'TEST_ONLY_%d' % i, 'response_status': 'SUPPORTED' if i < 2 else 'UNRESOLVED'} for i in range(5)]

    def test_unresolved_preserved(self):
        original = self.outcomes(); snapshot = deepcopy(original)
        g = group_fact('TEST_ONLY_CASE', 'train', original, [r['agent_id'] for r in original])
        self.assertEqual(g['unresolved'], 3); self.assertEqual(original, snapshot)
        self.assertTrue(g['collected_complete']); self.assertFalse(g['complete_training_group_eligible'])

    def test_duplicate_agent_not_complete(self):
        original = self.outcomes(); agents = [r['agent_id'] for r in original]
        original[-1]['agent_id'] = original[0]['agent_id']
        self.assertFalse(group_fact('TEST_ONLY', 'train', original, agents)['collected_complete'])

    def test_missing_agent_not_complete(self):
        original = self.outcomes(); agents = [r['agent_id'] for r in original]
        self.assertFalse(group_fact('TEST_ONLY', 'train', original[:-1], agents)['collected_complete'])

    def test_five_supported_still_requires_review(self):
        original = self.outcomes()
        for r in original: r['response_status'] = 'SUPPORTED'
        self.assertFalse(group_fact('TEST_ONLY', 'train', original, [r['agent_id'] for r in original])['complete_training_group_eligible'])

    def candidates(self):
        return [{'case_ref': 'TEST_ONLY_%d' % i, 'decision': {'split': 'validation' if i == 0 else 'train', 'patient_id': 'TEST_ONLY_P%d' % i}} for i in range(4)]

    def facts(self):
        return [{'case_ref': c['case_ref'], 'patient_encounter_link': True, 'definitely_future_selected_reference': i == 1} for i, c in enumerate(self.candidates())]

    def test_prospective_excludes_validation_and_future(self):
        selected = select_prospective(self.candidates(), self.facts())
        self.assertEqual([c['case_ref'] for c in selected], ['TEST_ONLY_2', 'TEST_ONLY_3'])

    def test_prospective_distinct_patients(self):
        cs = self.candidates(); cs[3]['decision']['patient_id'] = cs[2]['decision']['patient_id']
        with self.assertRaises(ValueError): select_prospective(cs, self.facts())

    def test_prospective_selection_does_not_use_response_results(self):
        cs = self.candidates(); before = select_prospective(cs, self.facts())
        for c in cs: c['responses'] = ['TEST_ONLY_WRONG_OR_RIGHT']
        self.assertEqual([c['case_ref'] for c in before], [c['case_ref'] for c in select_prospective(cs, self.facts())])

    def test_native_abstention_missing_utility(self):
        r = AgentRegistry(test_only_registry()).route({'chief_complaint': 'TEST_ONLY'}, ['neurology', 'cardiology'], mode='adaptive')
        self.assertEqual(r['decision_status'], 'abstained'); self.assertIsNone(r['selected_specialty'])

    def test_registry_mismatch_rejected(self):
        registry = test_only_registry(); artifact = test_only_routing(registry)
        registry['agents'][0]['version'] = 'TEST_ONLY_CHANGED'
        with self.assertRaises(ValueError): RoutingUtility(artifact, AgentRegistry(registry))

    def test_timeout_no_response_classification(self):
        self.assertEqual(classify_attempt({'error_type': 'LLMTimeoutError', 'raw_response': None}), 'CLIENT_TIMEOUT_NO_RESPONSE_CONTENT')

    def test_schema_error_after_response_separate(self):
        self.assertEqual(classify_attempt({'error_type': 'LLMResponseError', 'raw_response': 'TEST_ONLY'}), 'RESPONSE_RECEIVED_SEE_SCHEMA_STATUS')

    def test_inventory_no_generation_endpoint(self):
        client = Mock()
        for forbidden in ('/api/generate', '/api/chat', '/v1/chat/completions', 'https://other.example/api/tags'):
            with self.assertRaises(ValueError): inventory_get(client, forbidden)
        client.get.assert_not_called(); client.post.assert_not_called()

    def test_inventory_get_only(self):
        client = Mock(); client.get.return_value.json.return_value = {'TEST_ONLY': True}
        self.assertEqual(inventory_get(client, '/api/ps'), {'TEST_ONLY': True})
        client.get.assert_called_once_with('http://127.0.0.1:11434/api/ps'); client.post.assert_not_called()

    def test_protected_hash_mismatch_detected(self):
        from scripts.audit_memory_review_progress import check_hashes
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'TEST_ONLY'; path.write_bytes(b'TEST_ONLY')
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(check_hashes({str(path): h}), 1)
            with self.assertRaises(ValueError): check_hashes({str(path): '0' * 64})
            self.assertEqual(path.read_bytes(), b'TEST_ONLY')

    def test_safe_unique_run_id_preflight_no_generation_no_write(self):
        run_id = uuid.uuid4().hex
        destination = live.HANDOFF_ROOT / run_id
        self.assertFalse(destination.exists())
        config = live.load_config(ROOT / 'configs/development_handoff_empty_evidence_v3.json')
        with patch('app.locdi.ollama_adapter._backend') as backend, patch('socket.create_connection', side_effect=AssertionError('NO_NETWORK')):
            result = live.preflight(run_id=run_id, config=config)
        backend.assert_not_called(); self.assertEqual(result['blocker_codes'], [])
        self.assertFalse(destination.exists())
        self.assertEqual(result['plan']['specialty_semantic_repair']['max_attempts'], 1)


if __name__ == '__main__': unittest.main()
