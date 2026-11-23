"""TEST_ONLY deterministic checks. No patient judgments or clinical labels."""
from copy import deepcopy
import unittest

from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import released_projection
from scripts.classify_pre_review_evidence import (numeric, compare_fact, memory_checks, query_checks,
    classifications, safe_source, compare_working, run)
import test_source_enrichment as source_fixtures


def fixture():
    raw = {'PATIENT': 'TEST_ONLY_P', 'ENCOUNTER': 'TEST_ONLY_V0', 'DATE': '2020-01-01T00:00:00Z',
           'CODE': 'TEST_ONLY_C', 'DESCRIPTION': 'TEST_ONLY_TERM', 'VALUE': '0', 'UNITS': 'TEST_ONLY_U'}
    text = ' | '.join([raw['DATE'], 'observations', raw['DESCRIPTION'], raw['VALUE'], raw['UNITS']])
    record = {'memory_id': 'TEST_ONLY_M', 'patient_id': 'TEST_ONLY_P', 'visit_id': 'TEST_ONLY_V0',
              'visit_started_at': '2019-12-31T00:00:00Z', 'observed_at': '2020-01-01T00:00:00.999000+00:00',
              'available_at': '2020-01-01T00:00:00.999000+00:00', 'source_id': 'TEST_ONLY_M',
              'kind': 'objective_finding', 'text': text, 'supersedes': []}
    origin = {'_source': {'source_record': raw}, 'type': 'observations', 'normalized_text': text,
              'patient_id': 'TEST_ONLY_P', 'encounter_id': 'TEST_ONLY_V0', 'record_id': 'TEST_ONLY_M',
              'code': 'TEST_ONLY_C', 'description': 'TEST_ONLY_TERM', 'value': '0', 'units': 'TEST_ONLY_U',
              'observed_at_upper_bound': record['observed_at'], 'available_at': record['available_at'],
              'timestamp': raw['DATE']}
    identity = {'patient_id': 'TEST_ONLY_P', 'visit_id': 'TEST_ONLY_V1', 'started_at': '2020-01-02T00:00:00Z'}
    return record, identity, deepcopy(record), {'record_fingerprint': digest(record)}, origin, raw, True, {record['memory_id']: record}


def states(args):
    return {c['rule_id']: c['result'] for c in memory_checks(*args)}


class PreReviewEvidenceTests(unittest.TestCase):
    def test_all_structural_checks_pass_without_clinical_promotion(self):
        checks = memory_checks(*fixture())
        self.assertIn('STRUCTURAL_PASS', classifications(checks))
        self.assertIn('REQUIRES_CLINICAL_REVIEW', classifications(checks))
        self.assertEqual(states(fixture())['M11_EXPLICIT_SUPERSEDES'], 'UNKNOWN')

    def test_patient_mismatch(self):
        f = list(fixture()); f[0]['patient_id'] = 'TEST_ONLY_OTHER'
        self.assertEqual(states(f)['M01_PATIENT'], 'FAIL')

    def test_current_visit_rejected(self):
        f = list(fixture()); f[0]['visit_id'] = f[1]['visit_id']
        self.assertEqual(states(f)['M02_PREVIOUS_VISIT'], 'FAIL')

    def test_future_cutoff_rejected_and_values_quarantined(self):
        f = list(fixture()); f[0]['available_at'] = '2021-01-01T00:00:00Z'
        checks = memory_checks(*f)
        self.assertEqual(states(f)['M04_FIXED_MEMORY_CUTOFF'], 'FAIL')
        self.assertIn('STRUCTURAL_FAIL', classifications(checks))
        self.assertIsNone(safe_source(f[4], 'STRUCTURAL_PASS' in classifications(checks)))

    def test_inverted_timestamp(self):
        f = list(fixture()); f[0]['observed_at'] = '2019-01-01T00:00:00Z'
        self.assertEqual(states(f)['M03_CHRONOLOGY'], 'FAIL')

    def test_original_fingerprint_mismatch(self):
        f = list(fixture()); f[3]['record_fingerprint'] = 'TEST_ONLY_BAD'
        self.assertEqual(states(f)['M06_ORIGINAL_FINGERPRINT'], 'FAIL')

    def test_original_record_mismatch(self):
        f = list(fixture()); f[2]['text'] = 'TEST_ONLY_CHANGED'
        self.assertEqual(states(f)['M05_ORIGINAL_RECORD'], 'FAIL')

    def test_source_missing_distinct_from_false(self):
        f = list(fixture()); f[4] = None; f[5] = None
        self.assertEqual(states(f)['M07_SOURCE_RECORD'], 'MISSING')
        self.assertNotIn('STRUCTURAL_PASS', classifications(memory_checks(*f)))

    def test_source_fingerprint_mismatch(self):
        f = list(fixture()); f[6] = False
        self.assertEqual(states(f)['M08_SOURCE_FINGERPRINT'], 'FAIL')

    def test_raw_value_tampering(self):
        f = list(fixture()); f[5] = dict(f[5], VALUE='999')
        self.assertEqual(states(f)['M08_SOURCE_FINGERPRINT'], 'FAIL')
        self.assertEqual(states(f)['M09_SOURCE_TO_MEMORY'], 'FAIL')

    def test_source_timestamp_bound_mismatch(self):
        f = list(fixture()); f[4]['timestamp'] = '2020-01-01T00:00:01Z'
        self.assertEqual(states(f)['M10_SOURCE_TIME_BOUND'], 'FAIL')

    def test_explicit_supersedes_does_not_imply_stale(self):
        f = list(fixture()); f[0]['supersedes'] = ['TEST_ONLY_OLD']
        f[7]['TEST_ONLY_OLD'] = dict(f[0], memory_id='TEST_ONLY_OLD', available_at='2019-12-31T00:00:00Z')
        self.assertEqual(states(f)['M11_EXPLICIT_SUPERSEDES'], 'PASS')
        self.assertNotIn('stale', repr(memory_checks(*f)))

    def test_future_supersedes_rejected(self):
        f = list(fixture()); f[0]['supersedes'] = ['TEST_ONLY_FUTURE']
        f[7]['TEST_ONLY_FUTURE'] = dict(f[0], available_at='2022-01-01T00:00:00Z')
        self.assertEqual(states(f)['M11_EXPLICIT_SUPERSEDES'], 'FAIL')

    def test_zero_is_value_not_missing(self):
        self.assertEqual(states(fixture())['M13_EXPLICIT_SOURCE_VALUE'], 'PASS')
        self.assertEqual(numeric(0), 0)
        for v in (None, False, True, '', 'nan', 'Infinity', '1 mg'):
            self.assertIsNone(numeric(v))

    def test_missing_observation_value_and_nonvalue_resource_differ(self):
        f = list(fixture()); f[4]['value'] = None
        self.assertEqual(states(f)['M13_EXPLICIT_SOURCE_VALUE'], 'MISSING')
        f[4]['type'] = 'conditions'
        self.assertEqual(states(f)['M13_EXPLICIT_SOURCE_VALUE'], 'NOT_APPLICABLE')

    def test_comparison_requires_namespace_units_and_type(self):
        a = {'code': 'TEST_ONLY_C', 'description': 'TEST_ONLY_TERM', 'units': 'TEST_ONLY_U',
             'value': '2', 'system': 'TEST_ONLY_SYSTEM', 'type': 'observations'}
        b = dict(a, value='5', source_type='observations')
        result = compare_fact(a, b)
        self.assertEqual(result['comparability'], 'PASS')
        self.assertEqual(result['released_minus_historical'], '3')
        for key in ('relevant', 'stale', 'conflict'): self.assertNotIn(key, result)
        b['system'] = None
        self.assertEqual(compare_fact(a, b)['comparability'], 'UNKNOWN')
        self.assertIsNone(compare_fact(a, b)['released_minus_historical'])
        b['units'] = None
        self.assertEqual(compare_fact(a, b)['comparability'], 'NOT_APPLICABLE')

    def test_term_match_is_not_code_equivalence(self):
        a = {'code': 'TEST_ONLY_A', 'description': 'TEST_ONLY_TERM', 'value': 1}
        b = {'code': 'TEST_ONLY_B', 'description': 'TEST_ONLY_TERM', 'value': 2}
        result = compare_fact(a, b)
        self.assertTrue(result['exact_term_match'])
        self.assertFalse(result['exact_code_text_match'])
        self.assertIsNone(result['released_minus_historical'])

    def test_final_stop_and_private_keys_never_projected(self):
        origin = dict(fixture()[4], STOP='2099-01-01', gold='TEST_ONLY_SECRET', stopReason='TEST_ONLY_SECRET')
        safe = safe_source(origin, True)
        self.assertFalse(set(safe) & {'STOP', 'gold', 'stopReason', '_source'})
        self.assertIsNone(safe_source(origin, False))

    def query(self):
        identity, obs, events, results = source_fixtures.SourceEnrichmentTests().release()
        q = released_projection(obs, events, results, identity)
        return q, {'agent_visible': {'longitudinal': identity}, 'environment': {'current_encounter_events': events}}

    def test_query_missing_context_not_automatically_unsuitable(self):
        q, c = self.query()
        result = {r['rule_id']: r['result'] for r in query_checks(q, c)}
        self.assertEqual(result['Q01_CHIEF_COMPLAINT'], 'MISSING')
        self.assertEqual(result['Q08_ACTUAL_CLINICAL_AVAILABLE_AT'], 'MISSING')
        self.assertEqual(result['Q06_SOURCE_VALUE_AND_QUERY_REPRODUCTION'], 'PASS')
        self.assertNotIn('query_suitable', repr(query_checks(q, c)))

    def test_future_unreleased_fact_fails(self):
        q, c = self.query(); q['source_links'][0]['observed_minute'] = q['decision_minute'] + 1
        result = {r['rule_id']: r['result'] for r in query_checks(q, c)}
        self.assertEqual(result['Q04_REPLAY_RELEASE_TIME'], 'FAIL')
        self.assertEqual(result['Q06_SOURCE_VALUE_AND_QUERY_REPRODUCTION'], 'FAIL')

    def test_gold_key_fails(self):
        q, c = self.query(); q['projection']['gold'] = 'TEST_ONLY_SECRET'
        self.assertIn('FAIL', [r['result'] for r in query_checks(q, c) if r['rule_id'] == 'Q07_HIDDEN_FIELD_WHITELIST'])

    def test_event_fingerprint_mismatch(self):
        q, c = self.query(); q['source_links'][0]['source_fingerprint'] = 'TEST_ONLY_CHANGED'
        result = {r['rule_id']: r['result'] for r in query_checks(q, c)}
        self.assertEqual(result['Q05_EVENT_FINGERPRINT'], 'FAIL')

    def test_future_recorded_source_and_missing_release(self):
        q, c = self.query(); q['source_facts'][0]['recorded_at'] = '2099-01-01T00:00:00Z'
        result = {r['rule_id']: r['result'] for r in query_checks(q, c)}
        self.assertEqual(result['Q10_SOURCE_FACT_NOT_FUTURE'], 'FAIL')
        q['source_links'] = []
        result = {r['rule_id']: r['result'] for r in query_checks(q, c)}
        self.assertEqual(result['Q03_ASK_RELEASE'], 'MISSING')
        self.assertEqual(result['Q03_TEST_RELEASE'], 'MISSING')

    def test_working_comparison_never_writes_clinical_fields(self):
        blank = {'schema_version': 'TEST_ONLY', 'manifest_fingerprint': 'TEST_ONLY',
                 'snapshots': [{'snapshot_id': 'TEST_ONLY', 'records': [{'memory_id': 'TEST_ONLY'}]}]}
        working = deepcopy(blank)
        working['snapshots'][0]['query_suitable'] = False
        before = deepcopy(working)
        result = compare_working(blank, working)
        self.assertEqual(result['status'], 'STRUCTURE_MATCH')
        self.assertFalse(result['equals_blank'])
        self.assertEqual(working, before)
        self.assertEqual(compare_working(blank, None)['status'], 'NOT_FOUND')

    def test_namespace_escape_blocked(self):
        with self.assertRaisesRegex(ValueError, 'INVALID_NAMESPACE'): run('../TEST_ONLY')

    def test_inputs_remain_unchanged(self):
        f = fixture(); before = deepcopy(f)
        memory_checks(*f)
        self.assertEqual(f, before)
