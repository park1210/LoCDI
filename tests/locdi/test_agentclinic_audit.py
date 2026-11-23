"""TEST_ONLY symbols: read-only inventory, never clinical labels or simulation."""
import json
import unittest
from scripts.audit_agentclinic_dataset import audit, compare, parse, state, lexical_hit


def fixture(label='TEST_ONLY_LABEL_Z'):
    return {'OSCE_Examination': {
        'Objective_for_Doctor': 'TEST_ONLY_TASK',
        'Patient_Actor': {'Demographics': 'TEST_ONLY_ACTOR', 'History': 'TEST_ONLY_HISTORY',
                          'Symptoms': {'Primary_Symptom': 'TEST_ONLY_SYMPTOM', 'Secondary_Symptoms': []}},
        'Physical_Examination_Findings': {'TEST_ONLY_EXAM': 'TEST_ONLY_VALUE'},
        'Test_Results': {'TEST_ONLY_TEST': 'TEST_ONLY_VALUE'},
        'Correct_Diagnosis': label}}


def data(rows):
    return ('\n'.join(json.dumps(r) for r in rows) + '\n').encode('utf8')


class AgentClinicAuditContracts(unittest.TestCase):
    def test_line_endings_are_not_byte_identity(self):
        original = data([fixture(), fixture('TEST_ONLY_OTHER')])
        local = original[:-1].replace(b'\n', b'\r\n')
        result = compare(local, original)
        self.assertFalse(result['byte_identical'])
        self.assertNotEqual(result['local_sha256'], result['official_sha256'])
        self.assertTrue(result['ordered_parsed_records_equal'])
        self.assertTrue(result['line_endings_and_one_terminal_newline_only'])

    def test_reordered_or_changed_records_do_not_match(self):
        a, b = fixture(), fixture('TEST_ONLY_OTHER')
        self.assertFalse(compare(data([a, b]), data([b, a]))['ordered_parsed_records_equal'])
        b['Patient_Actor'] = {}  # Outside envelope, deliberately different structure.
        self.assertFalse(compare(data([a]), data([b]))['ordered_parsed_records_equal'])

    def test_invalid_json_utf8_and_duplicate_keys_are_not_certified(self):
        for raw in (b'{"x":1,"x":2}\n', b'not-json\n', b'\xff', b'[]\n', b'{"x":NaN}\n'):
            self.assertTrue(parse(raw)[1])
            self.assertFalse(compare(raw, raw)['ordered_parsed_records_equal'])

    def test_placeholder_is_distinct_from_negative_history(self):
        self.assertEqual(state('Information not provided.'), 'EXPLICIT_UNPROVIDED')
        self.assertEqual(state('No significant past medical history.'), 'PRESENT_NOT_CLINICALLY_REVIEWED')
        self.assertEqual(state(None), 'EMPTY_OR_NULL')

    def test_duplicate_case_and_narrative_groups(self):
        result, _ = audit(data([fixture(), fixture()]))
        self.assertEqual(result['duplicate_exact_record_groups'], [[1, 2]])
        self.assertEqual(result['duplicate_normalized_patient_narrative_groups'], [[1, 2]])

    def test_empty_tests_are_missing_content_not_invalid_schema(self):
        row = fixture()
        row['OSCE_Examination']['Test_Results'] = {}
        result, _ = audit(data([row]))
        self.assertEqual(result['schema_error_ordinals'], [])
        self.assertEqual(result['cases_with_empty_or_unprovided_core_values'], [1])
        self.assertEqual(result['test_top_level_branch_count'], 0)

    def test_private_reference_not_in_public_aggregate(self):
        result, private = audit(data([fixture()]))
        self.assertNotIn('TEST_ONLY_LABEL_Z', json.dumps(result))
        self.assertIn('TEST_ONLY_LABEL_Z', json.dumps(private))
        self.assertFalse(result['clinical_reference_approved'])
        self.assertFalse(result['case_flags'][0]['source_time_or_release_policy_verified'])

    def test_literal_test_clue_detected_without_word_substring_false_positive(self):
        row = fixture()
        row['OSCE_Examination']['Test_Results']['TEST_ONLY_TEST'] = 'Matches TEST_ONLY_LABEL_Z.'
        result, _ = audit(data([row]))
        self.assertEqual(result['literal_reference_case_counts']['Test_Results'], 1)
        self.assertFalse(lexical_hit('ABC', 'XABCD'))

    def test_parenthesis_variant_is_exploratory_not_literal_approval(self):
        row = fixture('TEST_ONLY_LONG_LABEL (XYZ)')
        row['OSCE_Examination']['Test_Results']['TEST_ONLY_TEST'] = 'XYZ'
        result, private = audit(data([row]))
        self.assertEqual(result['literal_reference_case_counts']['Test_Results'], 0)
        self.assertEqual(result['exploratory_reference_case_counts_REVIEW_REQUIRED']['Test_Results'], 1)
        self.assertEqual(private['cases'][0]['exploratory_matches'][0]['status'], 'REVIEW_REQUIRED')


if __name__ == '__main__':
    unittest.main()
