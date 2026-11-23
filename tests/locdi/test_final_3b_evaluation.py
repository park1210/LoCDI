"""TEST_ONLY nonclinical outcomes; no transport or fitting of real responses."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import tempfile

from app.locdi.final_diagnosis_evaluation import diagnosis_match
from app.locdi.final_3b_routing_evaluation import complete_groups, fit_source_utility, evaluate_policies
from app.core.routing_utility import RoutingUtility, observed_tokens
from app.core.agent_registry import AgentRegistry
from app.locdi.llm_setup import digest
from scripts.run_final_3b_evaluation import duplicate_clusters, verify_approval, SCOPE, Execution, SharedIntegrityError
from app.locdi.llm_setup import inspect_local_model


def fixture():
    departments = ['cardiology', 'neurology', 'nephrology', 'endocrinology', 'pulmonology']
    agents = [{'agent_id': 'TEST_ONLY_' + d, 'department': d, 'role': 'specialty',
               'model': 'TEST_ONLY:fake', 'version': 'TEST_ONLY', 'keywords': [d]} for d in departments]
    bindings = [{**a, 'model_identifier': 'sha256:' + 'a'*64} for a in agents]
    qs = [{'question_id': 'TEST_ONLY_Q%d' % i, 'case_group': 'TEST_ONLY_P%d' % i,
           'input': {'question': 'TEST_ONLY cardiology observation repeated %d' % i, 'options': []},
           'role': 'train'} for i in range(2)]
    rows = [{'question_id': q['question_id'], 'agent_id': a['agent_id'],
             'state': 'CORRECT_ANSWER' if j == 0 else 'INCORRECT_ANSWER', 'answer_correct': j == 0,
             'response_sha256': 'b'*64, 'reference_fingerprint': 'c'*64, 'model_identifier': a['model_identifier']}
            for q in qs for j, a in enumerate(bindings)]
    er = {'agent_id': 'TEST_ONLY_ER', 'department': '', 'role': 'er', 'model': 'TEST_ONLY:fake',
          'version': 'TEST_ONLY', 'keywords': []}
    return qs, bindings, rows, {'schema_version': 1, 'agents': [er] + agents}


class DiagnosisMatchTests(unittest.TestCase):
    def test_original_exact_preserved(self):
        self.assertTrue(diagnosis_match('TEST_ONLY condition', ['TEST_ONLY condition'])['original_exact'])

    def test_normalization_separate(self):
        r = diagnosis_match('TEST_ONLY CONDITION.', ['test_only condition'])
        self.assertFalse(r['original_exact']); self.assertTrue(r['normalized_exact'])

    def test_similar_disease_not_equivalent(self):
        r = diagnosis_match('TEST_ONLY acute A', ['TEST_ONLY chronic A'])
        self.assertIsNone(r['concept_match']); self.assertEqual(r['classification'], 'REVIEW_REQUIRED')

    def test_unresolved_not_false(self):
        self.assertIsNone(diagnosis_match(None, ['TEST_ONLY'], certainty='unresolved')['concept_match'])

    def test_unverified_synonym_rejected(self):
        with self.assertRaises(ValueError):
            diagnosis_match('A', ['B'], equivalences=[{'terms': ['A', 'B']}])

    def test_explicit_evidence_only(self):
        rule = {'terms': ['TEST_ONLY A', 'TEST_ONLY B'], 'evidence_path': 'TEST_ONLY',
                'evidence_sha256': 'a'*64, 'relation': 'SAME_STANDARD_CODE', 'review_status': 'VERIFIED'}
        self.assertTrue(diagnosis_match('TEST_ONLY A', ['TEST_ONLY B'], equivalences=[rule])['concept_match'])


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.q, self.b, self.r, self.reg = fixture()

    def test_five_complete_groups(self):
        self.assertEqual(sum(g['complete_scorable'] for g in complete_groups(self.q, self.b, self.r)), 2)

    def test_unresolved_group_cannot_be_completed_by_deletion(self):
        self.r[0].update(state='UNRESOLVED', answer_correct=None)
        groups = complete_groups(self.q, self.b, self.r)
        self.assertTrue(groups[0]['complete_collection']); self.assertFalse(groups[0]['complete_scorable'])
        self.assertFalse(complete_groups(self.q, self.b, self.r[1:])[0]['complete_collection'])

    def test_unresolved_false_rejected(self):
        self.r[0].update(state='UNRESOLVED', answer_correct=False)
        with self.assertRaises(ValueError): complete_groups(self.q, self.b, self.r)

    def test_unresolved_true_rejected(self):
        self.r[0].update(state='UNRESOLVED', answer_correct=True)
        with self.assertRaises(ValueError): complete_groups(self.q, self.b, self.r)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): complete_groups(self.q, self.b, self.r + [self.r[0]])

    def test_non_train_fit_rejected(self):
        self.q[0]['role'] = 'validation'
        with self.assertRaises(ValueError): fit_source_utility(self.q, self.b, self.r, self.reg, provenance_verified=True)

    def test_real_provenance_required(self):
        with self.assertRaises(ValueError): fit_source_utility(self.q, self.b, self.r, self.reg)

    def test_missing_response_pin_rejected(self):
        self.r[0]['response_sha256'] = None
        with self.assertRaises(ValueError): fit_source_utility(self.q, self.b, self.r, self.reg, provenance_verified=True)

    def test_model_binding_rejected(self):
        self.r[0]['model_identifier'] = 'sha256:' + 'd'*64
        with self.assertRaises(ValueError): fit_source_utility(self.q, self.b, self.r, self.reg, provenance_verified=True)

    def test_min_samples_existing_two(self):
        with self.assertRaises(ValueError): fit_source_utility(self.q[:1], self.b, self.r[:5], self.reg, provenance_verified=True)

    def test_no_agent_differentiation(self):
        for r in self.r: r.update(state='INCORRECT_ANSWER', answer_correct=False)
        with self.assertRaises(ValueError): fit_source_utility(self.q, self.b, self.r, self.reg, provenance_verified=True)

    def test_original_formula(self):
        artifact = fit_source_utility(self.q, self.b, self.r, self.reg, provenance_verified=True)
        utility = RoutingUtility(artifact, AgentRegistry(self.reg))
        observation = {'objective_findings': ['TEST_ONLY cardiology observation repeated 0']}
        from app.core.routing_utility import jaccard_similarity
        weights = [jaccard_similarity(observed_tokens(observation), r['tokens']) for r in artifact['samples'] if r['agent_id'] == self.b[0]['agent_id']]
        self.assertEqual(utility.score(observation, self.reg['agents'][1])['score'], (1 + sum(weights)) / (2 + sum(weights)))
        self.assertEqual(artifact['min_samples'], 2)

    def test_missing_utility_not_adaptive_success(self):
        result = evaluate_policies(self.q, self.b, self.r, self.reg)
        self.assertEqual(result['summary']['Adaptive']['decision_states'], {'unavailable': 2})
        self.assertEqual(result['summary']['Adaptive']['correct_supported_yield'], 0)
        self.assertIsNone(result['summary']['Adaptive']['conditional_source_agreement'])

    def test_existing_tie_abstention(self):
        # Two equally successful agents: Core must abstain, no custom tie rule.
        for r in self.r:
            if r['agent_id'] == self.b[1]['agent_id']: r.update(state='CORRECT_ANSWER', answer_correct=True)
        artifact = fit_source_utility(self.q, self.b, self.r, self.reg, provenance_verified=True)
        result = evaluate_policies(self.q, self.b, self.r, self.reg, artifact)
        self.assertEqual(result['summary']['Adaptive']['decision_states'], {'abstained': 2})


class ApprovalAndSplitTests(unittest.TestCase):
    def test_no_approval_no_transport(self):
        with patch('scripts.run_final_3b_evaluation.BudgetedNativeERClient') as client:
            with self.assertRaises(ValueError): Execution({'fingerprint': 'a'*64}, {})
            client.assert_not_called()

    def test_exact_new_scope_required(self):
        n = datetime.now(timezone.utc)
        record = {'approved': True, 'scope': SCOPE, 'plan_fingerprint': 'a'*64,
                  'evidence_type': 'USER_EXPLICIT', 'user_statement': 'TEST_ONLY',
                  'recorded_at': n.isoformat(), 'expires_at': (n+timedelta(hours=1)).isoformat()}
        verify_approval(record, {'fingerprint': 'a'*64})
        record['scope'] = 'OLD_C4_APPROVAL'
        with self.assertRaises(ValueError): verify_approval(record, {'fingerprint': 'a'*64})

    def test_expired_rejected(self):
        n = datetime.now(timezone.utc)-timedelta(days=2)
        r = {'approved': True, 'scope': SCOPE, 'plan_fingerprint': 'a'*64, 'evidence_type': 'USER_EXPLICIT',
             'user_statement': 'TEST_ONLY', 'recorded_at': n.isoformat(), 'expires_at': (n+timedelta(hours=1)).isoformat()}
        with self.assertRaises(ValueError): verify_approval(r, {'fingerprint': 'a'*64})

    def test_variants_and_same_actor_cluster(self):
        q = [{'case_group': 'A', 'input': {'question': '{"Patient_Actor":"TEST_ONLY same actor", "x":1}'}},
             {'case_group': 'B', 'input': {'question': '{"Patient_Actor":"TEST_ONLY same actor", "x":2}'}}]
        clusters = duplicate_clusters(q)
        self.assertEqual(clusters[0], clusters[1])

    def test_input_mutation_blocks_next_request(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/'TEST_ONLY.json'; p.write_text('{}', encoding='utf-8')
            state = Execution.__new__(Execution)
            stat = p.stat(); state.input_stats = {str(p): (stat.st_size, stat.st_mtime_ns)}
            state.record = {}; state.protocol = {}
            p.write_text('{"changed":true}', encoding='utf-8')
            with patch('scripts.run_final_3b_evaluation.verify_approval'):
                with self.assertRaises(SharedIntegrityError): state.guard()


class ManifestWrapperTests(unittest.TestCase):
    def make(self, root, *, tamper_hash=False, tamper_content=False):
        root = Path(root); (root/'blobs').mkdir()
        blob = b'TEST_ONLY_NON_MODEL'; h = hashlib.sha256(blob).hexdigest()
        (root/'blobs'/('sha256-'+h)).write_bytes(blob)
        entry = {'mediaType': 'TEST_ONLY_MODEL', 'digest': 'sha256:'+h, 'size': len(blob)}
        original = {'schemaVersion': 2, 'mediaType': 'application/vnd.docker.distribution.manifest.v2+json',
                    'config': entry, 'layers': [entry]}
        raw = json.dumps(original).encode(); upstream_h = hashlib.sha256(raw).hexdigest()
        (root/'blobs'/('sha256-'+upstream_h)).write_bytes(b'x'*len(raw) if tamper_hash else raw)
        wrapped = deepcopy(original)
        if tamper_content: wrapped['layers'] = []
        wrapped['layers'].append({'mediaType': original['mediaType'], 'digest': 'sha256:'+upstream_h, 'size': len(raw)})
        p = root/'manifests/registry.ollama.ai/library/TEST_ONLY/fake'; p.parent.mkdir(parents=True)
        p.write_text(json.dumps(wrapped), encoding='utf-8')
        return 'sha256:'+upstream_h

    def test_hash_and_content_proved(self):
        with tempfile.TemporaryDirectory() as root:
            expected = self.make(root)
            self.assertEqual(inspect_local_model('TEST_ONLY:fake', root)['model_identifier'], expected)

    def test_inner_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, tamper_hash=True)
            with self.assertRaises(ValueError): inspect_local_model('TEST_ONLY:fake', root)

    def test_inner_content_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.make(root, tamper_content=True)
            with self.assertRaises(ValueError): inspect_local_model('TEST_ONLY:fake', root)


if __name__ == '__main__': unittest.main()
