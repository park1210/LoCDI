"""TEST_ONLY nonclinical byte/option fixtures. No real answers or network."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_freeze import sha
from app.locdi.reference_source_resolution import source_inventory, resolve, require_identity, final_state
from app.locdi.reference_raw_validation import (parse_jsonl, inspect_zip, adapt_verified,
    cross_split_check, safe_public, validate_manifest, inventory_local_candidate)
from app.locdi.external_reference_task import index_contract, model_projection, require_use, qa_metrics
from app.locdi.reference_acquisition import acquire, validate_plan, fingerprint, PinnedApproval


IDENTITY = {'source_id': 'medqa_awinml_questions', 'revision': 'a'*40}


def rows():
    return [{'id': 'TEST_ONLY_PRIVATE_REF', 'question': 'TEST_ONLY select symbol beta?',
        'options': {'A': 'SYMBOL_ALPHA', 'B': 'SYMBOL_BETA', 'C': 'SYMBOL_GAMMA', 'D': 'SYMBOL_DELTA'},
        'answer_idx': 'B', 'answer': 'SYMBOL_BETA', 'explanation': 'TEST_ONLY_PRIVATE_EXPLANATION'}]


def encoded(r=None):
    return ('\n'.join(json.dumps(v) for v in (rows() if r is None else r))+'\n').encode('utf8')


def parse(data=None, **kw):
    data = encoded() if data is None else data
    options = dict(name='validation.jsonl', expected_sha256=hashlib.sha256(data).hexdigest(),
                   expected_bytes=len(data))
    options.update(kw)
    return parse_jsonl(data, **options)


def adapted(r=None, contract=None, **kw):
    r = rows() if r is None else r
    c = contract or index_contract(r, 'MedQA', IDENTITY['revision'], 'validation', 'medqa_letter')
    args = dict(contract_fingerprint=digest(c), identity=IDENTITY,
                expected_identity=IDENTITY, expected_option_count=4)
    args.update(kw)
    return adapt_verified(r, c, **args)


def zipped(items, compression=zipfile.ZIP_DEFLATED):
    out = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(out, 'w', compression=compression) as z:
            for name, data in items:
                z.writestr(name, data)
    return out.getvalue()


class RawValidationTests(unittest.TestCase):
    def code(self, expected, func, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, '^'+re.escape(expected)+'$'):
            func(*args, **kwargs)

    def test_jsonl_roundtrip_preserves_original(self):
        data = encoded(); before = hashlib.sha256(data).hexdigest()
        self.assertEqual(parse(data), rows())
        self.assertEqual(hashlib.sha256(data).hexdigest(), before)

    def test_legacy_osce_not_accepted_as_medqa(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'_medqa.jsonl'; p.write_bytes(b'{"OSCE_Examination":{"TEST_ONLY":"SYMBOL"}}\n')
            before = sha(p)
            with patch('app.locdi.reference_raw_validation.adapt_exam') as adapter:
                public, private = inventory_local_candidate(p)
            adapter.assert_not_called()
            self.assertEqual(public['direct_medqa_minimum_shape_count'], 0)
            self.assertEqual(public['osce_envelope_count'], 1)
            self.assertFalse(public['MEDQA_RAW_ACQUIRED']); self.assertFalse(public['license_evidence_accepted'])
            self.assertNotIn(str(p), json.dumps(public)); self.assertEqual(sha(p), before)
            self.assertEqual(private['explicit_path'], str(p.resolve()))

    def test_even_medqa_shape_does_not_authenticate_source(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'medqa.jsonl'; p.write_bytes(encoded())
            public, _ = inventory_local_candidate(p)
            self.assertEqual(public['direct_medqa_minimum_shape_count'], 1)
            self.assertFalse(public['MEDQA_RAW_ACQUIRED']); self.assertFalse(public['adapter_executed'])

    def test_no_automatic_candidate_search(self):
        with tempfile.TemporaryDirectory() as temp:
            self.code('EXPLICIT_LOCAL_FILE_NOT_FOUND', inventory_local_candidate, Path(temp)/'missing.jsonl')

    def test_cli_dry_run_has_no_acquisition_or_implicit_raw(self):
        from scripts.prepare_c3_07 import main
        with patch('sys.addaudithook'), patch('builtins.print') as out, \
                patch('scripts.prepare_c3_07.validate_manifest') as validation, \
                patch('scripts.prepare_c3_07.inventory_local_candidate') as inventory, \
                patch('app.locdi.reference_acquisition.https_transport') as transport:
            self.assertEqual(main([]), 0)
            validation.assert_not_called(); inventory.assert_not_called(); transport.assert_not_called()
            result = json.loads(out.call_args.args[0])
            self.assertEqual(result['mode'], 'DRY_RUN'); self.assertFalse(result['MEDQA_RAW_ACQUIRED'])

    def test_cli_existing_namespace_is_never_overwritten(self):
        from scripts.prepare_c3_07 import main
        path = ROOT/'outputs/research_evaluation/c3_06_v2/completion_manifest.json'; before = sha(path)
        with patch('sys.addaudithook'), patch('builtins.print') as out:
            self.assertEqual(main(['--save', '--namespace', 'c3_06_v2']), 2)
        self.assertEqual(sha(path), before)
        self.assertEqual(json.loads(out.call_args.args[0])['blocker_codes'], ['RAW_OR_PREPARATION_CONTRACT_REJECTED'])

    def test_hash_tamper(self):
        self.code('RAW_HASH_MISMATCH', parse, expected_sha256='0'*64)

    def test_bytes_mismatch(self):
        self.code('RAW_BYTES_MISMATCH', parse, expected_bytes=1)

    def test_size_cap(self):
        self.code('RAW_SIZE_LIMIT', parse, max_bytes=8)

    def test_infinite_size(self):
        self.code('FINITE_BYTE_BUDGET_REQUIRED', parse, max_bytes=float('inf'))

    def test_row_limit(self):
        r = rows()*2
        self.code('RAW_ROW_LIMIT', parse, encoded(r), max_rows=1)

    def test_json_duplicate_key(self):
        self.code('DUPLICATE_JSON_KEY', parse, b'{"question":"X","question":"Y"}\n')

    def test_invalid_utf8(self):
        self.code('INVALID_UTF8', parse, b'{"question":"\xff"}\n')

    def test_invalid_json(self):
        self.code('INVALID_JSON_RECORD', parse, b'{no}\n')

    def test_nonfinite_exponent(self):
        self.code('NONFINITE_JSON_NUMBER', parse, b'{"number":1e999}\n')

    def test_extension_mismatch(self):
        self.code('EXTENSION_CONTENT_MISMATCH', parse, name='file.zip')

    def test_magic_zip_mismatch(self):
        self.code('EXTENSION_CONTENT_MISMATCH', parse, b'PK\x03\x04TEST_ONLY')

    def test_magic_parquet_mismatch(self):
        self.code('EXTENSION_CONTENT_MISMATCH', parse, b'PAR1TEST_ONLY')

    def test_html_mismatch(self):
        self.code('EXTENSION_CONTENT_MISMATCH', parse, b'<html>TEST_ONLY</html>')

    def test_blank_record(self):
        self.code('BLANK_JSONL_RECORD', parse, b'\n')

    def test_revision_mismatch(self):
        self.code('SOURCE_REVISION_MISMATCH', require_identity, IDENTITY, dict(IDENTITY, revision='b'*40))

    def test_original_mirror_confusion(self):
        self.code('SOURCE_DISTRIBUTION_MISMATCH', require_identity, IDENTITY,
                  dict(IDENTITY, source_id='medqa_official_drive'))

    def test_index_wrong_revision(self):
        c = index_contract(rows(), 'MedQA', 'b'*40, 'validation', 'medqa_letter')
        self.code('INDEX_SOURCE_REVISION_MISMATCH', adapted, contract=c)

    def test_answer_index_range(self):
        r = rows(); r[0]['answer_idx'] = 'Z'
        self.code('ANSWER_KEY_MISSING_OR_OUT_OF_RANGE', adapted, r)

    def test_answer_index_missing(self):
        r = rows(); del r[0]['answer_idx']
        self.code('ANSWER_KEY_MISSING_OR_OUT_OF_RANGE', adapted, r)

    def test_changed_option_order_under_pin(self):
        r = rows(); c = index_contract(r, 'MedQA', IDENTITY['revision'], 'validation', 'medqa_letter')
        r[0]['options']['A'] = 'NEW_SYMBOL'
        self.code('SOURCE_OR_INDEX_CONTRACT_FINGERPRINT', adapted, r, contract=c)

    def test_noncanonical_options(self):
        r = rows(); r[0]['options'] = dict(reversed(list(r[0]['options'].items())))
        self.code('OPTION_ORDER_OR_COUNT', adapted, r)

    def test_variant_mismatch(self):
        self.code('SOURCE_OPTION_VARIANT_MISMATCH', adapted, expected_option_count=5)

    def test_five_option_preserved(self):
        r = rows(); r[0]['options']['E'] = 'SYMBOL_EPSILON'; r[0]['answer_idx'] = 'E'
        tasks, keys = adapted(r, expected_option_count=5)
        self.assertEqual(tasks[0]['answer_format']['labels'], list('ABCDE'))
        self.assertEqual(keys['AnswerKey000001']['answer_label'], 'E')

    def test_explicit_gold_in_question(self):
        r = rows(); r[0]['question'] += ' gold=B'
        self.code('EXPLICIT_ANSWER_OR_METADATA_IN_QUESTION', adapted, r)

    def test_private_explanation_in_question(self):
        r = rows(); r[0]['question'] += r[0]['explanation']
        self.code('PRIVATE_REFERENCE_IN_QUESTION', adapted, r)

    def test_duplicate_id(self):
        r = rows(); other = deepcopy(r[0]); other['question'] = 'TEST_ONLY other symbol?'; r.append(other)
        self.code('DUPLICATE_OR_INVALID_ID', adapted, r)

    def test_duplicate_question(self):
        r = rows(); other = deepcopy(r[0]); other['id'] = 'TEST_ONLY_SECOND'; r.append(other)
        self.code('DUPLICATE_QUESTION', adapted, r)

    def test_cross_split_question(self):
        r = rows(); other = deepcopy(r); other[0]['id'] = 'TEST_ONLY_SECOND'
        self.code('CROSS_SPLIT_DUPLICATE_QUESTION', cross_split_check, {'train': r, 'test': other})

    def test_cross_split_id(self):
        r = rows(); other = deepcopy(r); other[0]['question'] = 'TEST_ONLY other symbol?'
        self.code('CROSS_SPLIT_DUPLICATE_ID', cross_split_check, {'train': r, 'test': other})

    def test_original_dev_split_preserved(self):
        c = index_contract(rows(), 'MedQA', IDENTITY['revision'], 'dev', 'medqa_letter')
        t, _ = adapted(contract=c); self.assertEqual(t[0]['original_split'], 'dev')

    def test_projection_no_key_id_explanation(self):
        t, keys = adapted(); p = model_projection(t[0]); content = json.loads(p[0]['content'])
        self.assertEqual(set(content), {'question', 'options'})
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(p)); self.assertTrue(keys)
        self.assertEqual(t[0]['evaluation_status'], 'NOT_EVALUATED')

    def test_zip_path_traversal(self):
        for name in ('../bad.jsonl', '/bad.jsonl', 'C:/bad.jsonl', 'folder\\bad.jsonl'):
            with self.subTest(path_kind=name):
                data = zipped([(name.replace('\\', '/'), b'x')])
                if '\\' in name:
                    data = data.replace(name.replace('\\', '/').encode(), name.encode())
                self.code('ARCHIVE_PATH_TRAVERSAL', inspect_zip, data)

    def test_zip_duplicate_path(self):
        self.code('ARCHIVE_DUPLICATE_PATH', inspect_zip, zipped([('a.jsonl', b'x'), ('A.jsonl', b'y')]))

    def test_zip_bomb_ratio(self):
        self.code('ARCHIVE_EXPANSION_LIMIT', inspect_zip, zipped([('a.jsonl', b'x'*200000)]))

    def test_zip_expansion_bytes(self):
        self.code('ARCHIVE_EXPANSION_LIMIT', inspect_zip, zipped([('a.jsonl', b'abcdef')]), max_expanded_bytes=2)

    def test_zip_member_limit(self):
        self.code('ARCHIVE_MEMBER_LIMIT', inspect_zip, zipped([('a', b'x'), ('b', b'y')]), max_members=1)

    def test_zip_not_promoted_to_raw(self):
        r = inspect_zip(zipped([('qa/dev.jsonl', encoded())]))
        self.assertFalse(r['raw_adapter_ready'])
        self.assertEqual(r['blocker_codes'], ['ARCHIVE_SOURCE_TO_JSONL_LINEAGE_REQUIRED'])

    def test_unbounded_network_plan(self):
        from app.locdi.public_reference_preparation import acquisition_plans
        plan = acquisition_plans()[0]; plan['network_request_budget'] = None; plan['plan_fingerprint'] = fingerprint(plan)
        self.assertIn('NETWORK_REQUEST_BUDGET_REQUIRED_OR_OUT_OF_RANGE', validate_plan(plan)['blocker_codes'])

    def test_unapproved_download_no_transport(self):
        from app.locdi.public_reference_preparation import acquisition_plans
        with patch('app.locdi.reference_acquisition.https_transport') as transport:
            self.code('ACQUISITION_PLAN_BLOCKED', acquire, acquisition_plans()[0], execute=True, allow_download=True)
            transport.assert_not_called()

    def test_private_key_public_blocked(self):
        self.code('PRIVATE_FIELD_IN_PUBLIC_OUTPUT', safe_public,
            {'status': 'BLOCKED', 'external_evaluation_ready': False, 'answer_keys': {'B': 'PRIVATE'}})

    def test_private_text_in_public_blocker(self):
        self.code('PUBLIC_BLOCKER_INVALID', safe_public, {'status': 'BLOCKED',
            'external_evaluation_ready': False, 'blocker_codes': ['TEST_ONLY private text']})

    def test_ready_without_raw(self):
        self.code('RAW_REQUIRED_FOR_VALIDATION_PASS', final_state, True, False)

    def test_external_execution_false_even_raw_pass(self):
        self.assertFalse(final_state(True, True)['EXTERNAL_EVALUATION_READY'])

    def test_no_clinical_promotion(self):
        t, _ = adapted(); self.code('EXTERNAL_REFERENCE_PROMOTION_FORBIDDEN', require_use,
                                   t[0], purpose='clinical_gold')

    def test_no_routing_promotion(self):
        t, _ = adapted(); self.code('EXTERNAL_REFERENCE_PROMOTION_FORBIDDEN', require_use,
                                   t[0], purpose='routing_fit')

    def test_no_voi_promotion(self):
        t, _ = adapted(); self.code('EXTERNAL_REFERENCE_PROMOTION_FORBIDDEN', require_use,
                                   t[0], purpose='voi_likelihood')

    def test_missing_keys_not_scored(self):
        t, _ = adapted(); r = qa_metrics(t, ['B'])
        self.assertEqual(r['overall_answer_accuracy']['status'], 'NOT_EVALUATED')

    def test_documentary_claims_cannot_approve(self):
        for note in source_inventory():
            note.update(actual_hash_verified=True, accepted_license_proof=True, acquisition_approval=True)
            r = resolve(note); self.assertFalse(r['ACQUISITION_ALLOWED']); self.assertFalse(r['SOURCE_HASH_VERIFIED'])
            self.assertTrue(r['ACQUISITION_BLOCKED'])

    def test_explicit_manifest_pin_required(self):
        self.code('EVIDENCE_PRIVATE_PATH', validate_manifest, ROOT/'README.md', '0'*64)

    def test_full_fake_receipt_adapter_gate_roundtrip(self):
        # TEST_ONLY evidence names explicitly say TEST_ONLY; no real approver invented.
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_c307_', dir=ROOT/'outputs/private') as temp:
            base = Path(temp); data = encoded(); license_file = base/'license.txt'
            license_file.write_text('TEST_ONLY_SYMBOLIC_PERMISSION_NOT_REAL_RIGHTS', encoding='utf8')
            plan = {'schema_version': 'locdi.reference_acquisition_plan.v1', 'dataset_id': 'MedQA',
                'canonical_url': 'https://huggingface.co/datasets/awinml/medqa', 'pinned_revision': IDENTITY['revision'],
                'expected_files': [{'name': 'validation.jsonl', 'url': 'https://huggingface.co/datasets/awinml/medqa/resolve/'+IDENTITY['revision']+'/questions/validation.jsonl',
                                    'sha256': hashlib.sha256(data).hexdigest(), 'expected_bytes': len(data)}],
                'accepted_license_use_policy_evidence': {'reference': license_file.relative_to(ROOT).as_posix(), 'sha256': sha(license_file)},
                'maximum_bytes': 65536, 'destination_private_namespace': 'TEST_ONLY', 'expected_split': 'validation',
                'approval_reference': 'TEST_ONLY_NOT_ACTUAL_AUTHORITY', 'network_request_budget': 1,
                'timeout_seconds': 1, 'download_attempt_limit': 1}
            plan['plan_fingerprint'] = fingerprint(plan)
            approval = {'schema_version': 'locdi.acquisition_approval.v1', 'plan_fingerprint': plan['plan_fingerprint'],
                'approval_reference': plan['approval_reference'], 'license_evidence': plan['accepted_license_use_policy_evidence'],
                'scope': 'PRIVATE_RAW_ACQUISITION_ONLY', 'approved_by': 'TEST_ONLY_SYMBOL',
                'issued_at': '2000-01-01T00:00:00+00:00', 'expires_at': '2100-01-01T00:00:00+00:00'}
            approval_path = base/'approval.json'; approval_path.write_text(json.dumps(approval), encoding='utf8')
            class Response(io.BytesIO):
                headers = {'Content-Type': 'application/jsonl', 'Content-Length': str(len(data))}
            with patch('app.locdi.reference_acquisition.PRIVATE_ROOT', base/'fake_received'):
                acquire(plan, execute=True, allow_download=True,
                    authority=PinnedApproval(approval_path, sha(approval_path)), transport=lambda url, timeout: Response(data))
            receipt = base/'fake_received/MedQA'/IDENTITY['revision']/'receipt.json'
            c = index_contract(rows(), 'MedQA', IDENTITY['revision'], 'validation', 'medqa_letter')
            index = base/'index.json'; index.write_text(json.dumps(c), encoding='utf8')
            manifest = {'source_identity': IDENTITY, 'expected_source_identity': IDENTITY, 'splits': [{'plan': plan,
                'raw_evidence': {'receipt_path': str(receipt), 'receipt_sha256': sha(receipt),
                                 'approval_path': str(approval_path), 'approval_sha256': sha(approval_path)},
                'index_evidence': {'path': str(index), 'sha256': sha(index), 'fingerprint': digest(c)}, 'option_count': 4}]}
            path = base/'input.json'; path.write_text(json.dumps(manifest), encoding='utf8')
            before = sha(receipt.parent/'validation.jsonl')
            public, private = validate_manifest(path, sha(path))
            self.assertEqual(public['row_count'], 1); self.assertEqual(public['split_counts'], {'validation': 1})
            self.assertFalse(public['external_evaluation_ready'])
            self.assertEqual(sha(receipt.parent/'validation.jsonl'), before)
            self.assertIn('answer_keys_by_original_split', private)
            self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(public))
            # Exact source lineage is bound; a mirror label cannot mask a foreign URL.
            manifest['source_identity'] = manifest['expected_source_identity'] = dict(IDENTITY, source_id='medqa_official_drive')
            path.write_text(json.dumps(manifest), encoding='utf8')
            self.code('SOURCE_DISTRIBUTION_URL_MISMATCH', validate_manifest, path, sha(path))


if __name__ == '__main__':
    unittest.main()
