"""TEST_ONLY collection analysis; no clinical judgments or transport calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_routing_collector import test_only_bundle
from app.locdi.llm_setup import digest, ROOT
from app.locdi.specialty_collection_analysis import analyze_records, save_review, _safe_json
from scripts.analyze_specialty_collection import main


def fixture():
    bundle, _ = test_only_bundle()
    for c in bundle['candidates']:
        c['memory_fingerprint'] = digest(c['decision']['observation']['longitudinal_memory'])
        c['source_fingerprint'] = '4' * 64
        c['visit_started_at'] = '2026-01-01T00:00:00+00:00'
    bundle.pop('fingerprint'); bundle['fingerprint'] = digest(bundle)
    # Reproduce disk loader order, which differs from project_input insertion order.
    bundle = json.loads(json.dumps(bundle, sort_keys=True))
    agents = [a for a in bundle['registry']['agents'] if a['role'] == 'specialty']
    records = []
    for c in bundle['candidates']:
        d = c['decision']
        for i, a in enumerate(agents):
            ref = c['case_ref'] + '_Agent%03d' % (i + 1)
            payload = {'department': a['department'], **deepcopy(c['shared_payload'])}
            answer = {'action': 'diagnose', 'diagnosis': 'Unresolved' if i == 0 else 'TEST_ONLY_A',
                      'certainty': 'unresolved' if i == 0 else 'supported', 'rationale': 'TEST_ONLY_PRIVATE_RESPONSE'}
            row = {k: deepcopy(d[k]) for k in ('patient_id', 'decision_id', 'source_ref', 'split', 'generation', 'tokens')}
            row.update({k: a[k] for k in ('agent_id', 'model', 'version', 'department', 'role')})
            row.update(case_ref=c['case_ref'], agent_ref='Agent%03d' % (i + 1), response_ref=ref,
                purpose='routing_training', configuration_fingerprint=digest(d), observation_fingerprint=digest(d['observation']),
                shared_input_fingerprint=digest(c['shared_payload']), full_payload_fingerprint=digest(payload),
                serialized_payload_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest(),
                system_fingerprint='2' * 64, correct=None, response=answer, response_fingerprint=digest(answer),
                observed_model_identifier=bundle['llm_config']['model_identifier'], execution_status='SUCCEEDED',
                response_contract_status='VALID', correctness_status='NOT_EVALUABLE' if i == 0 else 'NOT_REVIEWED')
            records.append((ref + '.json', row))
    options = {'header': {'bundle_fingerprint': bundle['fingerprint'], 'accepted_assumptions': bundle['assumptions'], 'global_request_limit': 75},
        'completion': {'status': 'FINISHED', 'attempts': 75, 'generation_requests': 75, 'stop_reason': None},
        'manifest': {'complete': True, 'preparation_fingerprint': digest(bundle), 'reference_fingerprint': digest([])},
        'current_registry': deepcopy(bundle['registry']), 'current_prompts': {a['department']: '2' * 64 for a in agents},
        'source_comparison': 'PASS'}
    return bundle, records, options


class CollectionAnalysisContracts(unittest.TestCase):
    def setUp(self):
        self.bundle, self.records, self.options = fixture()

    def analyze(self):
        return analyze_records(self.bundle, [], self.records, **self.options)

    def test_complete_disk_order_hashes_do_not_imply_correctness(self):
        s, p = self.analyze()
        self.assertEqual(s['state'], 'COLLECTION_COMPLETE_NOT_CLINICALLY_EVALUATED')
        self.assertEqual(s['classification_counts'], {'UNRESOLVED': 15, 'RESPONSE_RECORDED': 60})
        self.assertEqual(s['splits']['train']['collected_complete_groups'], 10)
        self.assertEqual(s['eligible_reviewed_train_groups'], 0)
        self.assertEqual(s['training_readiness'], 'BLOCKED')
        self.assertTrue(all(a['verdict']['correct'] is None for c in p['items'] for a in c['agent_reviews']))

    def test_public_has_no_private_identity_response_or_snapshot(self):
        s, p = self.analyze()
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(s))
        self.assertNotIn('chief_complaint', json.dumps(s))
        self.assertIn('TEST_ONLY_PRIVATE', json.dumps(p))

    def test_partial_records_do_not_invent_request_counts(self):
        self.records = self.records[:3]; self.options['completion'] = None
        s, _ = self.analyze()
        self.assertIsNone(s['recorded_generation_requests'])
        self.assertEqual(len(s['missing_tasks']), 72)
        self.assertEqual(s['splits']['train']['collected_complete_groups'], 0)

    def test_failed_run_preserves_actual_counts(self):
        self.records = self.records[:2]; self.options['completion'] = None
        self.options['failure'] = {'attempts': 2, 'generation_requests': 1}
        s, _ = self.analyze()
        self.assertEqual(s['state'], 'FAILED_WITH_PRESERVED_PARTIAL_RESULTS')
        self.assertEqual(s['recorded_generation_requests'], 1)

    def test_duplicate_and_orphan_not_complete(self):
        self.records.append(deepcopy(self.records[0]))
        row = deepcopy(self.records[1][1]); row['case_ref'] = 'TEST_ONLY_OTHER'
        self.records.append(('unknown.json', row))
        s, _ = self.analyze()
        self.assertEqual(s['orphan_record_count'], 1)
        self.assertIn('NO_DUPLICATE_TASK_OUTCOMES', s['integrity_issue_codes'])
        self.assertNotEqual(s['state'], 'COLLECTION_COMPLETE_NOT_CLINICALLY_EVALUATED')

    def test_all_binding_and_hash_fields_block_tampering(self):
        for field in ('model', 'version', 'system_fingerprint', 'shared_input_fingerprint', 'full_payload_fingerprint', 'serialized_payload_sha256'):
            with self.subTest(field=field):
                records = deepcopy(self.records); records[0][1][field] = 'TEST_ONLY_TAMPER'
                s, _ = analyze_records(self.bundle, [], records, **self.options)
                self.assertIn('FIELD_MISMATCH_' + field.upper(), s['integrity_issue_codes'])

    def test_response_content_hash_and_model_digest_checked(self):
        self.records[0][1]['response']['rationale'] = 'TEST_ONLY_TAMPER'
        self.records[1][1]['observed_model_identifier'] = 'TEST_ONLY_OTHER'
        s, _ = self.analyze()
        self.assertIn('RESPONSE_CONTENT_HASH_MISMATCH', s['integrity_issue_codes'])
        self.assertIn('MODEL_DIGEST_MISMATCH', s['integrity_issue_codes'])

    def test_schema_invalid_and_request_failure_distinguished(self):
        self.records[0][1]['response'] = {}
        self.records[1][1].update(execution_status='FAILED', error_type='LLMTimeoutError')
        s, _ = self.analyze()
        self.assertEqual(s['schema_invalid_count'], 1)
        self.assertEqual(s['classification_counts']['REQUEST_FAILED'], 1)
        self.assertEqual(s['timeout_count'], 1)

    def test_future_gold_memory_or_catalog_tamper_blocks(self):
        for kind in ('future', 'gold', 'memory', 'catalog'):
            with self.subTest(kind=kind):
                b = deepcopy(self.bundle); c = b['candidates'][0]
                if kind == 'future':
                    next(iter(c['visibility'].values()))['available_minute'] = 2
                elif kind == 'gold':
                    c['decision']['observation']['gold'] = 'TEST_ONLY_REFERENCE'
                elif kind == 'memory':
                    c['memory_fingerprint'] = '0' * 64
                else:
                    b['catalog'].append('TEST_ONLY_OTHER')
                s, _ = analyze_records(b, [], self.records, **self.options)
                self.assertIn('Candidate001_SNAPSHOT_LEDGER_AND_INPUT_SCOPE', s['integrity_issue_codes'])

    def test_registry_assumptions_manifest_and_audit_checked(self):
        self.options['current_registry'] = {}
        self.options['header']['accepted_assumptions'] = []
        self.options['manifest']['complete'] = False
        self.options['audit'] = {'generation_requests': 74}
        s, _ = self.analyze()
        for key in ('CURRENT_REGISTRY_SNAPSHOT', 'RECORDED_DEVELOPMENT_ASSUMPTIONS', 'PRIVATE_MANIFEST', 'RUN_AND_AUDIT_COUNTERS'):
            self.assertIn(key, s['integrity_issue_codes'])

    def test_automatic_correctness_and_request_limit_contradiction_block(self):
        self.records[0][1]['correct'] = False
        self.options['completion']['generation_requests'] = 76
        s, _ = self.analyze()
        self.assertIn('UNREVIEWED_CORRECTNESS_PRESENT', s['integrity_issue_codes'])
        self.assertIn('GLOBAL_RECORDED_REQUEST_LIMIT', s['integrity_issue_codes'])

    def test_review_save_no_clobber_and_escape(self):
        s, p = self.analyze()
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            root = Path(directory); public = root / 'public'; public.mkdir()
            with patch('app.locdi.specialty_collection_analysis.ROOT', root), patch('app.locdi.specialty_collection_analysis.PUBLIC_ROOT', public):
                dest = save_review(s, p, 'TEST_ONLY_REVIEW')
                before = (dest / 'review_packet.json').read_bytes()
                self.assertEqual(_safe_json(dest / 'review_packet.json'), p)
                with self.assertRaises(FileExistsError):
                    save_review(s, p, 'TEST_ONLY_REVIEW')
                with self.assertRaises(ValueError):
                    save_review(s, p, '../escape')
                self.assertEqual((dest / 'review_packet.json').read_bytes(), before)

    def test_cli_read_only_default_and_error_sanitization(self):
        with patch('scripts.analyze_specialty_collection.analyze_run', return_value=({}, {})), patch('scripts.analyze_specialty_collection.save_review') as writer, patch('builtins.print'):
            self.assertEqual(main([]), 0)
            writer.assert_not_called()
        with patch('scripts.analyze_specialty_collection.analyze_run', side_effect=ValueError('TEST_ONLY_PRIVATE_ERROR')), patch('builtins.print') as out:
            self.assertEqual(main([]), 2)
            self.assertNotIn('TEST_ONLY_PRIVATE', str(out.call_args))

    def test_strict_json_duplicate_keys_and_nan_rejected(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as directory:
            path = Path(directory) / 'test.json'
            for value in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(value, encoding='utf-8')
                with self.assertRaises(ValueError):
                    _safe_json(path)


if __name__ == '__main__':
    unittest.main()
