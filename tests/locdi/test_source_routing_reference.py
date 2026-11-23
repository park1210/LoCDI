"""TEST_ONLY source/reference contracts, never clinical verdicts or generation."""
from copy import deepcopy
import unittest
from test_abstention_routing import fixture
from app.locdi.llm_setup import digest
from app.locdi.source_routing_reference import (narrow_candidate,time_state,summarize,benchmark,blind_packet,assert_blind,second_stage,rule_spec)
from app.locdi.abstention_routing import compare,validated_reviews
from app.core.longitudinal_memory import timestamp


def source_fixture():
    reg,cases=fixture();case=deepcopy(cases[0]);case.update(dataset='TEST_ONLY',visit_id='TEST_ONLY_VISIT')
    value={'start':int(timestamp('2025-12-31T00:00:00+00:00').timestamp()*1000),'stop':0,'codes':[{'system':'TEST_ONLY_SYSTEM','code':'TEST_ONLY_A','display':'TEST_ONLY'}]}
    fact={'fact_ref':'TEST_ONLY_FACT','patient_reference':case['patient_id'],'visit_reference':case['visit_id'],
        'source_value':value,'source_value_fingerprint':digest(value),'source_verified':True,
        'source_resource_type':'Condition',
        'recorded_at':'2025-12-31T00:00:00+00:00','available_at':'2025-12-31T01:00:00+00:00',
        'availability_evidence':'EXPLICIT_SOURCE_AVAILABLE_AT'}
    mapping={'version':'TEST_ONLY','mappings':[{'mapping_id':'TEST_ONLY_A','system':'TEST_ONLY_SYSTEM','code':'TEST_ONLY_A',
        'specialties':['cardiology'],'code_identity_verified':True,'specialty_scope_verified':True}]}
    return reg,case,fact,mapping


class SourceReferenceContracts(unittest.TestCase):
    def setUp(self):self.reg,self.case,self.fact,self.mapping=source_fixture()
    def build(self,facts=None):return narrow_candidate(self.case,[self.fact] if facts is None else facts,self.mapping)
    def mutate_code(self,**values):
        self.fact['source_value']['codes'][0].update(values);self.fact['source_value_fingerprint']=digest(self.fact['source_value'])
    def packet_rows(self):return [{'recorded_onset':'2025-12-31T00:00:00+00:00','encounter_started_at':'2025-12-30T00:00:00+00:00','source_codes':[{'code':'TEST_ONLY_A'}],'actual_available_at':None}]
    def test_exact_source_admission_silver_only(self):
        r=self.build();self.assertEqual(r['state'],'POSITIVE');self.assertEqual(r['accepted_specialties'],['cardiology']);self.assertFalse(r['clinical_reference']);self.assertIsNone(r['clinical_correctness']);self.assertIsNone(r['acceptable_abstention'])
    def test_patient_mismatch(self):
        self.fact['patient_reference']='TEST_ONLY_OTHER';r=self.build();self.assertEqual(r['state'],'ABSTAIN')
    def test_encounter_mismatch(self):
        self.fact['visit_reference']='TEST_ONLY_OTHER';self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_unverified_source_blocked(self):
        self.fact['source_verified']=False;self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_source_hash_mismatch(self):
        self.fact['source_value']['codes'][0]['display']='TEST_ONLY_CHANGE'
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):self.build()
    def test_recorded_time_matches_native_value(self):
        self.fact['recorded_at']='2025-12-30T00:00:00+00:00'
        self.assertIn('SOURCE_RECORDED_ONSET_MISMATCH',self.build()['blocker_codes'])
    def test_non_condition_not_reference(self):
        self.fact['source_resource_type']='Observation'
        self.assertIn('NOT_CONDITION_RESOURCE',self.build()['blocker_codes'])
    def test_system_not_inferred_from_code(self):
        self.mutate_code(system='WRONG_SYSTEM');self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_display_similarity_not_mapping(self):
        self.mutate_code(code='TEST_ONLY_UNMAPPED',display='TEST_ONLY_A cardiology');self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_missing_mapping_evidence(self):
        for k in ['code_identity_verified','specialty_scope_verified']:
            m=deepcopy(self.mapping);m['mappings'][0][k]=False
            self.assertEqual(narrow_candidate(self.case,[self.fact],m)['state'],'ABSTAIN')
    def test_future_onset(self):
        self.fact['recorded_at']='2026-01-02T00:00:00+00:00';self.assertEqual(time_state(self.fact,self.case['decision_time']),'FUTURE_ONSET');self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_onset_is_not_release(self):
        self.fact['available_at']=None;self.assertEqual(self.build()['state'],'ABSTAIN');self.assertIn('ONSET_BY_DECISION_RELEASE_UNKNOWN',self.build()['blocker_codes'])
    def test_future_release(self):
        self.fact['available_at']='2026-01-02T00:00:00+00:00';self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_reverse_release(self):
        self.fact['available_at']='2025-01-01T00:00:00+00:00';self.assertIn('REVERSED_AVAILABLE_TIME',self.build()['blocker_codes'])
    def test_claimed_proxy_not_explicit_availability(self):
        self.fact['availability_evidence']='RECORDED_PROXY';self.assertIn('UNVERIFIED_AVAILABLE_TIME',self.build()['blocker_codes'])
    def test_bad_timestamp(self):
        self.fact['recorded_at']='bad';self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_same_exact_boundary_allowed(self):
        self.fact['available_at']=self.case['decision_time'];self.assertEqual(self.build()['state'],'POSITIVE')
    def test_multiple_condition_union(self):
        self.mapping['mappings'].append({**self.mapping['mappings'][0],'mapping_id':'TEST_ONLY_B','code':'TEST_ONLY_B','specialties':['neurology']})
        f=deepcopy(self.fact);f['fact_ref']='TEST_ONLY_B';f['source_value']['codes'][0]['code']='TEST_ONLY_B';f['source_value_fingerprint']=digest(f['source_value'])
        self.assertEqual(self.build([self.fact,f])['accepted_specialties'],['cardiology','neurology'])
    def test_unknown_condition_blocks_complete_set(self):
        f=deepcopy(self.fact);f['source_value']['codes'][0]['code']='UNKNOWN';f['source_value_fingerprint']=digest(f['source_value'])
        r=self.build([self.fact,f]);self.assertEqual(r['state'],'ABSTAIN');self.assertIsNone(r['accepted_specialties'])
    def test_ended_condition_blocked(self):
        self.fact['source_value']['stop']=2;self.fact['source_value_fingerprint']=digest(self.fact['source_value']);self.assertEqual(self.build()['state'],'ABSTAIN')
    def test_duplicate_mapping_rejected(self):
        self.mapping['mappings']*=2
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):self.build()
    def test_missing_condition_not_negative(self):self.assertEqual(self.build([])['state'],'ABSTAIN')
    def test_explicit_scope_negative_not_absence(self):
        self.mapping['mappings'][0]['specialties']=[]
        self.assertEqual(self.build()['state'],'NEGATIVE');self.assertIsNone(self.build()['acceptable_abstention'])
    def test_coverage_preserves_unknown(self):
        r=self.build();self.fact['available_at']=None;q=self.build();q['case_ref']='TEST_ONLY_2'
        c=summarize([r,q]);self.assertEqual(c['registered_cases'],2);self.assertEqual(c['reference_coverage']['value'],.5)
    def test_empty_reference_no_scores(self):
        r=benchmark([self.case['case_ref']],[],[]);self.assertEqual(r['status'],'NOT_RUN_EMPTY_ADMITTED_REFERENCE');self.assertIsNone(r['source_grounded_agreement'])
    def test_narrow_metric_multiple_acceptance(self):
        r=self.build();r['accepted_specialties']=['cardiology','neurology']
        pred={'case_ref':self.case['case_ref'],'decision':{'decision_status':'selected','selected_specialty':'neurology','candidate_departments':self.case['candidate_departments']}}
        m=benchmark([self.case['case_ref']],[pred],[r]);self.assertEqual(m['source_grounded_agreement']['value'],1);self.assertEqual(m['multiple_specialty_acceptance'],1);self.assertEqual(m['clinical_metrics'],'NOT_EVALUATED')
    def test_narrow_abstain_zero_selected_null(self):
        r=self.build();pred={'case_ref':self.case['case_ref'],'decision':{'decision_status':'abstained','selected_specialty':None}}
        m=benchmark([self.case['case_ref']],[pred],[r]);self.assertIsNone(m['selective_accuracy_on_narrow_labels']['value']);self.assertEqual(m['abstention_rate']['value'],1)
    def test_clinical_reference_rejected_from_silver(self):
        r=self.build();r['clinical_reference']=True
        with self.assertRaises(ValueError):benchmark([self.case['case_ref']],[],[r])
    def test_unknown_not_admitted(self):
        r=self.build();r['source_time_eligible']=False
        with self.assertRaises(ValueError):benchmark([self.case['case_ref']],[],[r])
    def test_silver_not_independent_reference(self):
        with self.assertRaises(ValueError):validated_reviews([self.case],{'version':'SILVER','target':self.build()['target'],'reviews':[self.build()]},test_only=True)
    def test_blind_hides_extra_predictions(self):
        self.case['selected_specialty']='cardiology';self.case['actual_response']={'diagnosis':'TEST_ONLY'}
        p=blind_packet(self.case,self.packet_rows(),[]);assert_blind(p);self.assertNotIn('selected_specialty',p);self.assertNotIn('actual_response',p)
    def test_blind_prediction_key_forbidden(self):
        with self.assertRaises(ValueError):assert_blind({'nested':{'rationale':'TEST_ONLY'}})
    def test_blind_future_condition(self):
        rows=self.packet_rows();rows[0]['recorded_onset']='2027-01-01T00:00:00+00:00'
        with self.assertRaises(ValueError):blind_packet(self.case,rows,[])
    def test_blind_future_encounter(self):
        rows=self.packet_rows();rows[0]['encounter_started_at']='2027-01-01T00:00:00+00:00'
        with self.assertRaises(ValueError):blind_packet(self.case,[],rows)
    def test_blind_gold_in_observable_blocked(self):
        self.case['observation']['gold_department']='cardiology'
        with self.assertRaises(ValueError):blind_packet(self.case,[],[])
    def test_stage2_response_lineage(self):
        val={'diagnosis':'Unresolved','certainty':'unresolved','rationale':'TEST_ONLY'}
        row={'case_ref':self.case['case_ref'],'agent_id':'TEST_ONLY_AGENT','actual_response':val,'response_fingerprint':digest(val)}
        r=second_stage([self.case],[row]);self.assertFalse(r['stage1_review_completed']);self.assertIsNone(r['rows'][0]['agent_abstention_appropriateness'])
        row['response_fingerprint']='0'*64
        with self.assertRaises(ValueError):second_stage([self.case],[row])
    def test_duplicate_second_stage(self):
        val={'diagnosis':'TEST_ONLY'};row={'case_ref':self.case['case_ref'],'agent_id':'TEST_ONLY_AGENT','actual_response':val,'response_fingerprint':digest(val)}
        with self.assertRaises(ValueError):second_stage([self.case],[row,row])
    def test_lf_not_keyword_router(self):
        a=self.build();self.case['observation']['chief_complaint']='weakness diabetes'
        self.assertEqual(a,self.build())
    def test_existing_policies_repeat_unchanged(self):
        reg,cs=fixture();self.assertEqual(compare(cs,reg),compare(cs,reg));self.assertIsNone(reg.utility)


if __name__=='__main__':unittest.main()
