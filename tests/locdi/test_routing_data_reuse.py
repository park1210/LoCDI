"""TEST_ONLY source joins and duplication counts; no clinical judgments."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from app.locdi.routing_data_reuse import overlap, probe_comparison, bank_analysis, audit_existing_data
from scripts.audit_existing_routing_data import main


def test_only_bank():
    row = {'training_patient_id': 'TEST_ONLY_PRIVATE_PATIENT', 'training_case_id': 'TEST_ONLY_PRIVATE_CONTEXT',
           'training_label': 'TEST_ONLY_PRIVATE_LABEL', 'profile': {'age': 30, 'sex': 'TEST_ONLY', 'numeric': {'["TEST_ONLY_CODE","TEST_ONLY_UNIT"]': 2.0}},
           'source_provenance': {'context_id': 'TEST_ONLY_PRIVATE_CONTEXT', 'condition_rows': [],
                                 'observation_event_ids': ['TEST_ONLY_PRIVATE_EVENT']}}
    condition = {'PATIENT': row['training_patient_id'], 'ENCOUNTER': 'TEST_ONLY_PRIVATE_VISIT', 'DESCRIPTION': row['training_label']}
    ref = {'source_file': 'TEST_ONLY.csv', 'source_row': 2, 'source_record': condition}
    row['source_provenance']['condition_rows'] = [ref]
    context = {'case_id': row['training_case_id'], 'patient_id': row['training_patient_id'],
               'decision_time': '2026-10-28T00:00:00+00:00', 'index_encounter': {'encounter_id': 'TEST_ONLY_PRIVATE_VISIT'},
               'demographics': {'age_at_index': 30, 'sex': 'TEST_ONLY'}, 'past_encounters': []}
    source = {'patient_id': row['training_patient_id'], 'index_encounter': deepcopy(context['index_encounter']),
              'selected_actual_target_diagnosis': row['training_label'], 'symptom_sources': [],
              'source_events': [{'event_id': 'TEST_ONLY_PRIVATE_EVENT', 'patient_id': row['training_patient_id'],
                  'encounter_id': 'TEST_ONLY_PRIVATE_VISIT', 'timestamp': '2026-10-28T00:00:00+00:00',
                  'timestamp_lower_epoch_ms': 1793145600000, 'timestamp_upper_epoch_ms': 1793145600999,
                  'target': {'code': 'TEST_ONLY_CODE'}, 'evidence': {'units': 'TEST_ONLY_UNIT', 'value': 2.0}}]}
    options = {'holdout': ['TEST_ONLY_HOLDOUT'], 'sources': {row['training_case_id']: source},
               'contexts': {row['training_case_id']: context}, 'csv_records': {('TEST_ONLY.csv', 2): condition},
               'taxonomy': {row['training_label']: 'cardiology'}}
    return [row], options


class ReuseAuditContracts(unittest.TestCase):
    def test_overlap_unique_not_row_sum(self):
        self.assertEqual(overlap(['TEST_ONLY_A', 'TEST_ONLY_A', 'TEST_ONLY_B'], ['TEST_ONLY_A', 'TEST_ONLY_C']),
                         {'left_unique': 2, 'right_unique': 2, 'intersection': 1, 'union': 3})

    def test_versions_same_patient_different_decision_not_new_patient(self):
        a = {'patient_id': 'TEST_ONLY_PRIVATE', 'decision_id': 'TEST_ONLY_D1', 'observation': {'chief_complaint': 'TEST_ONLY_TEXT'}}
        b = deepcopy(a); b['decision_id'] = 'TEST_ONLY_D2'
        report = probe_comparison({'TEST_ONLY_V1': [a], 'TEST_ONLY_V2': [b]}, [])
        self.assertEqual(report['union_patients'], 1)
        self.assertEqual(report['union_decisions'], 2)
        self.assertEqual(report['pairs'][0]['observations']['intersection'], 1)
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(report))

    def test_exact_version_duplicates_cannot_add_candidates(self):
        p = {'patient_id': 'TEST_ONLY_P', 'decision_id': 'TEST_ONLY_D', 'observation': {}}
        report = probe_comparison({'TEST_ONLY_V1': [p], 'TEST_ONLY_V2': [p]}, [])
        self.assertEqual(report['union_decisions'], 1)
        self.assertEqual(report['pairs'][0]['full_rows']['intersection'], 1)

    def test_probe_holdout_overlap_reported(self):
        p = {'patient_id': 'TEST_ONLY_P', 'decision_id': 'TEST_ONLY_D', 'observation': {}}
        report = probe_comparison({'TEST_ONLY_V1': [p]}, ['TEST_ONLY_P'])
        self.assertEqual(report['versions']['TEST_ONLY_V1']['holdout_overlap'], 1)
        self.assertEqual(report['versions']['TEST_ONLY_V1']['approved_training_groups'], 0)

    def test_source_match_is_review_candidate_not_training_approval(self):
        rows, options = test_only_bank()
        report = bank_analysis(rows, **options)
        self.assertEqual(report['source_checks']['patient_context_visit_link_matches'], 1)
        self.assertEqual(report['source_checks']['numeric_profile_matches_source'], 1)
        self.assertEqual(report['source_linked_review_candidate_count'], 1)
        self.assertEqual(report['approved_training_group_count'], 0)
        self.assertEqual(report['validated_decision_count'], 0)
        self.assertEqual(report['status'], 'REQUIRES_REVIEW')

    def test_future_upper_bound_preserved_as_unverified_not_released(self):
        rows, options = test_only_bank()
        report = bank_analysis(rows, **options)
        self.assertEqual(report['source_checks']['upper_bound_after_decision'], 1)
        self.assertEqual(report['source_checks']['rows_with_later_recorded_upper_bound'], 1)
        self.assertEqual(report['source_checks']['nominal_timestamp_after_decision'], 0)
        self.assertEqual(report['release_visibility'], 'NOT_REVIEWED')

    def test_csv_mismatch_and_missing_context_reduce_candidates(self):
        rows, options = test_only_bank()
        event = options['sources'][rows[0]['training_case_id']]['source_events'][0]
        raw = {'PATIENT': rows[0]['training_patient_id'], 'ENCOUNTER': 'TEST_ONLY_PRIVATE_VISIT'}
        event['source_metadata'] = {'source_file': 'TEST_ONLY_OBSERVATION.csv', 'source_row': 2, 'source_record': raw}
        options['csv_records']['TEST_ONLY_OBSERVATION.csv', 2] = raw
        self.assertEqual(bank_analysis(rows, **options)['source_checks']['test_observation_csv_row_matches'], 1)
        options['csv_records'] = {}
        self.assertEqual(bank_analysis(rows, **options)['source_linked_review_candidate_count'], 0)
        self.assertEqual(bank_analysis(rows, **options)['source_checks']['test_observation_csv_row_matches'], 0)
        options['contexts'] = {}
        self.assertEqual(bank_analysis(rows, **options)['source_checks']['missing_source_or_context_rows'], 1)

    def test_patient_repetition_counted_and_holdout_overlap_excluded_from_approval(self):
        rows, options = test_only_bank()
        options['holdout'] = [rows[0]['training_patient_id']]
        report = bank_analysis(rows + rows, **options)
        self.assertEqual(report['unique_patients'], 1)
        self.assertEqual(report['patient_with_multiple_contexts_count'], 1)
        self.assertEqual(report['holdout_overlap_count'], 1)
        self.assertEqual(report['approved_training_group_count'], 0)

    def test_unmapped_labels_and_profile_mismatch_remain_unapproved(self):
        rows, options = test_only_bank()
        options['taxonomy'] = {}
        rows[0]['profile']['numeric'] = {}
        report = bank_analysis(rows, **options)
        self.assertEqual(report['specialty_source_label_counts'], {'UNMAPPED': 1})
        self.assertEqual(report['source_checks']['numeric_profile_matches_source'], 0)
        self.assertEqual(report['approved_training_group_count'], 0)

    def test_public_summary_contains_no_private_values_or_label_text(self):
        rows, options = test_only_bank()
        report = bank_analysis(rows, **options)
        encoded = json.dumps(report)
        for private in ('TEST_ONLY_PRIVATE_PATIENT', 'TEST_ONLY_PRIVATE_VISIT', 'TEST_ONLY_PRIVATE_CONTEXT',
                        'TEST_ONLY_PRIVATE_LABEL', 'TEST_ONLY_CODE', 'TEST_ONLY_UNIT'):
            self.assertNotIn(private, encoded)

    def test_missing_sources_report_missing_without_fabrication(self):
        with patch('pathlib.Path.is_file', return_value=False):
            report = audit_existing_data()
        self.assertEqual(report['classification'], 'MISSING')
        self.assertEqual(report['approved_training_group_count'], 0)

    def test_cli_sanitizes_source_errors(self):
        with patch('scripts.audit_existing_routing_data.audit_existing_data', side_effect=ValueError('TEST_ONLY_PRIVATE')), patch('builtins.print') as printer:
            self.assertEqual(main(), 2)
        self.assertNotIn('TEST_ONLY_PRIVATE', str(printer.call_args))


if __name__ == '__main__':
    unittest.main()
