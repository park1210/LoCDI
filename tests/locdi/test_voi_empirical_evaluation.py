"""TEST_ONLY replay boundaries. No new clinical outcomes/probability artifacts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from app.locdi.dataset_freeze import sha
from app.locdi.llm_setup import digest
from app.locdi.routing_two_decision_collection import verify_pins
from app.locdi.voi_empirical_evaluation import released_observation,replay,reject_labels,decision_key,distribution_stats
from app.core.information_outcomes import InformationOutcomes
from test_routing_voi_readiness import test_only_voi
from test_voi_stop_consistency import interface


def source_fixture():
    case={'case_id':'TEST_ONLY_C','index_encounter_id':'TEST_ONLY_CURRENT',
        'longitudinal':{'patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_CURRENT','started_at':'2026-01-02T00:00:00+00:00'},
        'past_visits':[],'current_encounter_events':[{'action':'Test','target':'TEST_ONLY_T','available_minute':1,'evidence':'TEST_ONLY_A'}]}
    events=[{'case_id':'TEST_ONLY_C','event_type':'test_ordered','minute':2,'payload':{'requested_test':'TEST_ONLY_T'}},
        {'case_id':'TEST_ONLY_C','event_type':'result_received','minute':5,'payload':{'requested_test':'TEST_ONLY_T','test_result':{'summary':'TEST_ONLY_T: TEST_ONLY_A'}}}]
    return case,events,{'mode':'none','records':[]}


class SourceReplayBoundaries(unittest.TestCase):
    def setUp(self):self.c,self.events,self.memory=source_fixture()
    def obs(self,minute=6,index=2):return released_observation(self.c,self.events,minute,self.memory,before_event_index=index)
    def test_no_test_result_before_receipt(self):self.assertEqual(self.obs(4)['objective_findings'],[])
    def test_source_result_after_receipt(self):self.assertEqual(self.obs()['objective_findings'],['TEST_ONLY_T: TEST_ONLY_A'])
    def test_same_minute_event_order_not_clock_only(self):
        self.assertEqual(self.obs(5,1)['objective_findings'],[])
        self.assertEqual(self.obs(5,2)['objective_findings'],['TEST_ONLY_T: TEST_ONLY_A'])
    def test_boundary_required_and_valid(self):
        for index in [None,-1,3,True]:
            with self.subTest(index=index),self.assertRaises(ValueError):self.obs(index=index)
    def test_result_without_order_rejected(self):
        self.events.pop(0)
        with self.assertRaisesRegex(ValueError,'WITHOUT_RECORDED_ORDER'):self.obs(index=1)
    def test_event_source_value_mismatch_rejected(self):
        self.events[1]['payload']['test_result']['summary']='TEST_ONLY_CHANGED'
        with self.assertRaisesRegex(ValueError,'SOURCE_MISMATCH'):self.obs()
    def test_source_availability_future_rejected(self):
        self.c['current_encounter_events'][0]['available_minute']=7
        with self.assertRaisesRegex(ValueError,'RELEASE_BINDING'):self.obs()
    def test_other_case_result_not_admitted(self):
        self.events[1]['case_id']='TEST_ONLY_OTHER';self.assertEqual(self.obs()['objective_findings'],[])
    def test_duplicate_source_binding_rejected(self):
        self.c['current_encounter_events']*=2
        with self.assertRaisesRegex(ValueError,'AMBIGUOUS'):self.obs()
    def memory_record(self):
        r={'memory_id':'TEST_ONLY_M','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_PAST','visit_started_at':'2026-01-01T00:00:00+00:00','available_at':'2026-01-01T00:01:00+00:00','text':'TEST_ONLY_HISTORY'}
        self.c['past_visits']=[{'records':[deepcopy(r)]}];self.memory={'mode':'selective','records':[r]};return r
    def test_past_memory_accepted(self):self.memory_record();self.assertEqual(len(self.obs()['longitudinal_memory']['records']),1)
    def test_future_memory_rejected(self):
        r=self.memory_record();r['available_at']='2026-01-03T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'CUTOFF'):self.obs()
    def test_other_patient_memory_rejected(self):
        r=self.memory_record();r['patient_id']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):self.obs()
    def test_memory_source_text_mismatch(self):
        r=self.memory_record();r['text']='TEST_ONLY_MODIFIED'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):self.obs()
    def test_projection_does_not_mutate_source(self):
        before=deepcopy([self.c,self.events,self.memory]);self.obs();self.assertEqual(before,[self.c,self.events,self.memory])


class SelectorReplayContracts(unittest.TestCase):
    def setUp(self):
        self.a=test_only_voi();self.obs={'objective_findings':[],'history':[],'longitudinal_memory':{'mode':'selective','records':[]}}
        self.cs=interface(self.a['actions']);self.proposal={'action':'diagnose','target':'Unresolved','rationale':'TEST_ONLY existing proposal'}
    def replay(self):return replay(self.a,self.obs,self.cs,self.proposal)
    def test_existing_off_branch_preserved(self):self.assertEqual(self.replay()['off']['selected_action'],self.proposal)
    def test_no_off_proposal_no_invented_baseline(self):
        self.proposal=None
        with self.assertRaisesRegex(ValueError,'OFF_PROPOSAL'):self.replay()
    def test_horizons_existing_fixed_values(self):
        r=self.replay();self.assertEqual(set(r['horizons']),{'1','2','3'})
        self.assertTrue(all(x['plan']['horizon']==int(h) for h,x in r['horizons'].items()))
    def test_reproducible_and_inputs_immutable(self):
        before=deepcopy([self.a,self.obs,self.cs,self.proposal]);a=self.replay();self.assertEqual(a,self.replay());self.assertEqual(before,[self.a,self.obs,self.cs,self.proposal])
    def test_no_memory_model_no_direct_belief_change(self):
        self.obs['longitudinal_memory']['records']=[{'text':'TEST_ONLY_HISTORY','status_as_of_visit':'active'}]
        r=self.replay();self.assertEqual(r['memory_action_count'],0);self.assertFalse(r['memory_direct_belief_changed'])
    def test_no_reference_no_clinical_score(self):
        r=self.replay();self.assertNotIn('accuracy',r);self.assertNotIn('correct',r)
    def test_nested_gold_rejected(self):
        for key in ['gold','evaluation','source_conditions','reference_diagnosis']:
            with self.subTest(key=key),self.assertRaises(ValueError):reject_labels({'history':[{'nested':{key:'TEST_ONLY'}}]})
    def test_encoded_label_rejected(self):
        with self.assertRaises(ValueError):reject_labels({'history':['{"gold_department":"TEST_ONLY"}']})
    def test_invalid_probability_rejected(self):
        self.a['prior'][next(iter(self.a['prior']))]=float('nan')
        with self.assertRaises(ValueError):self.replay()
    def test_invalid_utility_rejected(self):
        self.a['terminal_utility'][next(iter(self.a['terminal_utility']))]={}
        with self.assertRaises(ValueError):self.replay()
    def test_invalid_horizon_rejected(self):
        self.a['horizon']=4
        with self.assertRaises(ValueError):self.replay()
    def test_candidate_not_modeled_rejected(self):
        self.cs[0]['outcome_model']='TEST_ONLY_UNKNOWN'
        with self.assertRaises(KeyError):self.replay()
    def test_artifact_fingerprint_not_local_horizon_mutation(self):
        expected=InformationOutcomes(self.a).fingerprint;self.assertEqual(self.replay()['artifact_fingerprint'],expected)
    def test_stop_binding_does_not_grade_diagnosis(self):
        self.assertEqual(decision_key({'action':'diagnose','target':'TEST_ONLY_A'}),decision_key({'action':'diagnose','target':'Unresolved'}))
    def test_empty_distribution_not_zero(self):self.assertIsNone(distribution_stats([])['mean'])
    def test_nonfinite_distribution_rejected(self):
        with self.assertRaises(ValueError):distribution_stats([float('inf')])
    def test_source_hash_change_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY';p.write_text('TEST_ONLY',encoding='utf-8');pin={str(p):sha(p)}
            p.write_text('TEST_ONLY_CHANGED',encoding='utf-8')
            with self.assertRaises(ValueError):verify_pins(pin)


if __name__=='__main__':unittest.main()
