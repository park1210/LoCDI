"""TEST_ONLY offline text coverage and blank review contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_specialty_collection_analysis import fixture
from test_routing_training_data import test_only_data
from app.locdi.llm_setup import ROOT, digest
from app.locdi.unresolved_analysis import analyze, condition_link, name_mapping, response_state, save_analysis
from validation.routing_training_data import validate_training_data
from scripts.analyze_unresolved import main


def test_only_inputs():
    bundle, pairs, _ = fixture()
    refs = []
    for c in bundle['candidates']:
        d = c['decision']
        refs.append({'case_ref': c['case_ref'], 'declared_label': 'TEST_ONLY_A',
                     'source_condition_rows': [{'source_record': {'SYSTEM': 'TEST_ONLY', 'CODE': 'TEST_ONLY_CODE',
                        'DESCRIPTION': 'TEST_ONLY_A', 'PATIENT': d['patient_id'], 'ENCOUNTER': d['visit_id'], 'START': '2026-01-01'}}]})
    return bundle, refs, [r for _, r in pairs], {'TEST_ONLY_A': 'cardiology'}, {'cardiology': 'TEST_ONLY_PROMPT'}


class UnresolvedAnalysisContracts(unittest.TestCase):
    def test_exact_text_does_not_approve_code_equivalence_or_gold(self):
        r = name_mapping('TEST_ONLY_A', ['TEST_ONLY_A'])
        self.assertEqual(r['text_status'], 'EXACT_TEXT_PRESENT')
        self.assertIsNone(r['clinical_equivalence'])
        self.assertFalse(r['clinical_gold_approved'])
        self.assertTrue(r['code_mapping'].startswith('REVIEW_REQUIRED'))

    def test_case_whitespace_unicode_is_only_mapping_candidate(self):
        r = name_mapping('  test_only_a  ', ['TEST_ONLY_A'])
        self.assertEqual(r['normalized_candidate_ordinals'], [1])
        self.assertEqual(r['text_status'], 'REVIEW_REQUIRED')
        self.assertFalse(r['exact_catalog_ordinals'])

    def test_ambiguous_synonym_stage_and_hierarchy_never_auto_approved(self):
        for label in ['TEST_ONLY_ALIAS', 'TEST_ONLY_A stage 2', 'TEST_ONLY']:
            r = name_mapping(label, ['TEST_ONLY_A', 'TEST_ONLY_A stage 3'])
            self.assertEqual(r['text_status'], 'REVIEW_REQUIRED')
            self.assertIsNone(r['clinical_equivalence'])

    def test_named_diagnosis_unresolved_is_not_recorded_or_false(self):
        row = {'execution_status': 'SUCCEEDED', 'response_contract_status': 'VALID',
               'response': {'certainty': 'unresolved', 'diagnosis': 'TEST_ONLY_A'}}
        self.assertEqual(response_state(row), ('UNRESOLVED', 'NOT_EVALUABLE'))
        row['response'] = {'certainty': 'supported', 'diagnosis': 'Unresolved'}
        self.assertEqual(response_state(row), ('UNRESOLVED', 'NOT_EVALUABLE'))
        row['execution_status'] = 'FAILED'
        self.assertEqual(response_state(row)[0], 'REQUEST_FAILED')

    def test_all_75_analyzed_without_modifying_source(self):
        args = test_only_inputs(); before = digest(args)
        p, private, template = analyze(*args)
        self.assertEqual(digest(args), before)
        self.assertEqual(p['response_count'], 75)
        self.assertEqual(len(private['responses']), 75)
        self.assertEqual(len(template['items']), 75)
        self.assertEqual(p['named_diagnosis_unresolved_count'], 0)  # Fixture uses explicit Unresolved.
        self.assertEqual(p['literal_coverage']['present_decisions'], 15)
        self.assertEqual(p['training_readiness'], 'BLOCKED')

    def test_public_excludes_private_text_and_identifiers(self):
        p, private, template = analyze(*test_only_inputs())
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(p))
        self.assertNotIn('TEST_ONLY_CODE', json.dumps(p))
        self.assertTrue(all('rationale' not in row for row in template['items']))
        self.assertIn('TEST_ONLY_CODE', json.dumps(private))

    def test_reviewer_and_every_actual_verdict_stays_blank(self):
        _, private, template = analyze(*test_only_inputs())
        for row in template['items']:
            self.assertEqual(row['status'], 'NOT_REVIEWED')
            for field in ('reviewer_ref', 'reviewed_at', 'procedure_version', 'correct', 'abstention_appropriate', 'routing_success', 'reference_valid'):
                self.assertIsNone(row[field])
        self.assertTrue(all(r['correct'] is None for r in private['responses']))

    def test_future_condition_requires_reference_review_not_leakage_verdict(self):
        b, refs, *_ = test_only_inputs()
        refs[0]['source_condition_rows'][0]['source_record']['START'] = '2026-01-05'
        link = condition_link(b['candidates'][0], refs[0])[0]
        self.assertTrue(link['patient_link'] and link['encounter_link'])
        self.assertEqual(link['start_day_offset'], 4)
        self.assertEqual(link['timing_status'], 'REVIEW_REQUIRED_START_AFTER_ENCOUNTER')
        self.assertIsNone(link['clinical_reference_valid'])

    def test_literal_gap_is_not_automatically_medical_gap(self):
        args = list(test_only_inputs()); args[1][0]['declared_label'] = 'TEST_ONLY_ALIAS'
        public, private, _ = analyze(*args)
        self.assertEqual(public['literal_coverage']['absent_decisions'], 1)
        self.assertEqual(private['cases'][0]['declared_label_mapping']['text_status'], 'REVIEW_REQUIRED')
        self.assertEqual(public['causal_identification'], 'NOT_ESTABLISHED')

    def test_auto_correctness_and_partial_baseline_rejected(self):
        args = list(test_only_inputs()); args[2][0]['correct'] = False
        with self.assertRaises(ValueError):
            analyze(*args)
        args = list(test_only_inputs()); args[2].pop()
        with self.assertRaises(ValueError):
            analyze(*args)

    def test_training_validator_does_not_infer_abstention_policy(self):
        # Explicit pre-existing TEST_ONLY judgments, not judgments of real data.
        data, options = test_only_data()
        for row in data['outcomes']:
            row['response'] = {'diagnosis': 'Unresolved', 'certainty': 'unresolved'}
            row['correctness_status'] = 'NOT_EVALUABLE'
        result = validate_training_data(data, **options)
        self.assertEqual(result['training_readiness'], 'TEST_ONLY_CONTRACT_VALID')
        for j in data['judgments']:
            j['correct'] = None; j['status'] = 'NOT_REVIEWED'
        self.assertEqual(validate_training_data(data, **options)['training_readiness'], 'BLOCKED')

    def test_new_output_no_overwrite_and_path_escape(self):
        p, private, template = analyze(*test_only_inputs())
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            root = Path(directory); public = root/'public'; public.mkdir()
            with patch('app.locdi.unresolved_analysis.ROOT', root), patch('app.locdi.unresolved_analysis.PUBLIC_ROOT', public):
                dest = save_analysis(p, private, template, 'TEST_ONLY')
                before = (dest/'independent_review_template.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    save_analysis(p, private, template, 'TEST_ONLY')
                with self.assertRaises(ValueError):
                    save_analysis(p, private, template, '../escape')
                self.assertEqual((dest/'independent_review_template.json').read_bytes(), before)

    def test_cli_default_does_not_write_or_print_error_evidence(self):
        with patch('scripts.analyze_unresolved.load_analysis', return_value=({}, {}, {})), patch('scripts.analyze_unresolved.save_analysis') as save, patch('builtins.print'):
            self.assertEqual(main([]), 0)
            save.assert_not_called()
        with patch('scripts.analyze_unresolved.load_analysis', side_effect=ValueError('TEST_ONLY_PRIVATE_ERROR')), patch('builtins.print') as out:
            self.assertEqual(main([]), 2)
            self.assertNotIn('TEST_ONLY_PRIVATE', str(out.call_args))


if __name__ == '__main__':
    unittest.main()
