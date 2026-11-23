"""TEST_ONLY nonclinical Source/time/schema audit fixtures, no model client."""
from copy import deepcopy
import unittest
from app.locdi.synthea_diagnosis_audit import (matches,normalize,diagnosis_branch,schema_selectable,
    proxy_minute,reference_time,source_observation_matches,typed_objective_candidate,disclosure_status)


class AuditTests(unittest.TestCase):
    def test_exact_preserved_separate_normalization(self):
        r=matches('TEST_ONLY label.', ['test_only LABEL'])
        self.assertFalse(r['source_exact']);self.assertTrue(r['normalized_match']);self.assertIsNone(r['verified_concept_match'])

    def test_no_synonym_invention(self):
        r=matches('TEST_ONLY acute A',['TEST_ONLY chronic A']);self.assertFalse(r['normalized_match']);self.assertEqual(r['concept_status'],'NOT_EVALUATED')

    def test_null_not_false(self):self.assertIsNone(matches(None,['TEST_ONLY'])['source_exact'])

    def test_zero_value_is_real_numeric_fact(self):
        e={'event_type':'observations','evidence':{'category':'laboratory','value':'0','units':'TEST_ONLY'}}
        self.assertTrue(typed_objective_candidate(e))

    def test_missing_not_numeric(self):
        self.assertFalse(typed_objective_candidate({'event_type':'observations','evidence':{'category':'laboratory','value':None}}))

    def test_private_condition_never_objective(self):
        self.assertFalse(typed_objective_candidate({'event_type':'conditions','evidence':{'category':'laboratory','value':'1'}}))

    def test_proxy_upper_bound_ceil(self):self.assertEqual(proxy_minute({'timestamp_upper_epoch_ms':1001},1000),1)

    def test_future_cutoff(self):self.assertEqual(disclosure_status({'timestamp_upper_epoch_ms':120001},1,0),'AFTER_FIXED_CUTOFF')

    def test_proxy_not_release_approval(self):
        self.assertEqual(disclosure_status({'timestamp_upper_epoch_ms':1},1,0),'PROXY_TIME_ELIGIBLE_RELEASE_NOT_REVIEWED')

    def test_missing_timestamp_unknown(self):self.assertEqual(disclosure_status({},1,0),'TIMESTAMP_UNKNOWN')

    def test_same_day_not_temporal_approval(self):self.assertEqual(reference_time('2020-01-01','2020-01-01T00:00:01Z')['status'],'SAME_DAY_INTRADAY_NOT_VERIFIED')

    def test_later_reference_day_detected(self):self.assertEqual(reference_time('2020-01-05','2020-01-01T12:00:00Z')['day_difference'],4)

    def test_units_mismatch_not_hidden(self):
        e={'target':{'code':'TEST_ONLY'},'timestamp':'2020','evidence':{'value':'1','units':'A'}}
        r={'code':'TEST_ONLY','recorded_at':'2020','recorded_evidence':{'value':'1','units':'B'}}
        self.assertFalse(source_observation_matches(r,e));r['recorded_evidence']['units']='A';self.assertTrue(source_observation_matches(r,e))

    def test_offline_schema_probe_not_a_response_rewrite(self):
        s={'type':'object','properties':{'action':{'enum':['diagnose']},'target':{'type':'string'}},'required':['action','target']}
        r={'action':'diagnose','target':'TEST_ONLY_PREDICTION'};original=deepcopy(r)
        self.assertTrue(schema_selectable(s,r,'TEST_ONLY_REFERENCE'));self.assertEqual(r,original)

    def test_real_enum_coverage_not_assumed(self):
        s={'anyOf':[{'type':'object','properties':{'action':{'enum':['diagnose']},'target':{'enum':['TEST_ONLY_A']}}}]}
        self.assertEqual(len(diagnosis_branch(s)),1)
        self.assertFalse(schema_selectable(s,{'action':'diagnose','target':'TEST_ONLY_A'},'TEST_ONLY_B'))


if __name__=='__main__':unittest.main()
