"""TEST_ONLY temporal oracle, evaluator boundaries and missing-label semantics."""
import unittest
from copy import deepcopy
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.memory_evaluation import evaluate_memory
from scripts.audit_locdi_core_evaluation import eligible_reference,initial_query

class CoreEvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        self.identity={'patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_NOW','started_at':'2020-01-02T00:00:00+00:00'}
        self.record={'memory_id':'TEST_ONLY_M','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_PAST',
            'visit_started_at':'2020-01-01T00:00:00+00:00','observed_at':'2020-01-01T01:00:00+00:00',
            'available_at':'2020-01-01T02:00:00+00:00','source_id':'TEST_ONLY_SOURCE','kind':'history',
            'text':'TEST_ONLY signal','supersedes':[]}
    def test_same_patient_prior_visit_availability_and_inclusive_cutoff(self):
        variants=[self.record,dict(self.record,memory_id='TEST_ONLY_EQUAL',available_at=self.identity['started_at']),
            dict(self.record,memory_id='TEST_ONLY_LATE',available_at='2020-01-03T00:00:00+00:00'),
            dict(self.record,memory_id='TEST_ONLY_OTHER',patient_id='TEST_ONLY_Q'),
            dict(self.record,memory_id='TEST_ONLY_CURRENT',visit_id='TEST_ONLY_NOW')]
        expected={'TEST_ONLY_M','TEST_ONLY_EQUAL'}
        self.assertEqual(eligible_reference(variants,self.identity),expected)
        actual=PatientMemoryStore(variants).retrieve(self.identity,'signal','full_history')
        self.assertEqual({r['memory_id'] for r in actual['records']},expected)
    def test_none_full_and_selective_identical_pool_do_not_mutate(self):
        records=[dict(self.record,memory_id='TEST_ONLY_%d'%i) for i in range(7)]
        store=PatientMemoryStore(records);before=deepcopy(store.snapshot())
        self.assertEqual(len(store.retrieve(self.identity,'signal','none')['records']),0)
        self.assertEqual(len(store.retrieve(self.identity,'signal','full_history')['records']),7)
        self.assertEqual(len(store.retrieve(self.identity,'signal','selective',5)['records']),5)
        self.assertEqual(before,store.snapshot())
    def test_missing_relevance_is_unknown_not_zero(self):
        context=PatientMemoryStore([self.record]).retrieve(self.identity,'signal','selective')
        result=evaluate_memory(context,[],{},['TEST_ONLY_M'])
        self.assertIsNone(result['precision']['value']);self.assertIsNone(result['recall']['value'])
        self.assertEqual(result['recall']['status'],'missing_annotation')
    def test_empty_labeled_denominator_is_not_applicable(self):
        context=PatientMemoryStore([]).retrieve(self.identity,'signal','none')
        result=evaluate_memory(context,[],{'relevant_memory_ids':[]},[])
        self.assertIsNone(result['recall']['value']);self.assertEqual(result['recall']['status'],'not_applicable')
    def test_private_labels_and_hidden_results_do_not_change_query(self):
        engine={'case_id':'TEST_ONLY','doctor_visible_info':{'chief_complaint':'TEST_ONLY_QUERY'},
            'patient_visible_info':{},'evaluation':{'diagnosis_labels':['TEST_ONLY_SECRET']},
            'current_encounter_events':[{'evidence':'TEST_ONLY_FUTURE'}]}
        q=initial_query(engine);engine['evaluation']={'diagnosis_labels':['OTHER_SECRET']}
        engine['current_encounter_events']=[{'evidence':'OTHER_FUTURE'}]
        self.assertEqual(q,initial_query(engine));self.assertNotIn('SECRET',q);self.assertNotIn('FUTURE',q)
    def test_declared_doctor_fields_can_yield_empty_query(self):
        self.assertEqual(initial_query({'case_id':'TEST_ONLY','doctor_visible_info':{'initial_vitals':{}},'patient_visible_info':{}}),'')
