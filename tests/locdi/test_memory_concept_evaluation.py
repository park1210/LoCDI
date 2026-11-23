"""TEST_ONLY exact identity fixtures; no clinical labels or probabilities."""
from copy import deepcopy
import unittest
import tempfile
from pathlib import Path
from app.locdi.memory_concept_evaluation import (concept_key, evaluate_query, aggregate,
    public_query, ratio, validate_reference)
from app.locdi.llm_setup import digest


def alignment(code='TEST_CODE_A', unit='TEST_UNIT', system='TEST_ONLY_URI', kind='NativeSyntheaObservation'):
    return {'status':'VERIFIED_SOURCE_CODING','system_uri':system,'code':code,'unit':unit,
        'native':{'resource_type':kind,'unit':unit,'coding':{'code':code},
                  'pointer':'/record/encounters/0/observations/0'}}


def ref(mid, code='TEST_CODE_A', label='POSITIVE', **kw):
    return {'memory_id':mid,'label':label,'source_type':'observations',
        'source_alignment':alignment(code,**kw),'reason':'TEST_ONLY',
        'clinical_relevance':None,'stale':None,'conflict':None}


def run(rs, ids, queries=None):
    return evaluate_query(rs,queries or [alignment()],
        {'none':[],'full_history':[r['memory_id'] for r in rs],'selective':ids})


class ConceptContractTests(unittest.TestCase):
    def test_exact_key(self):
        self.assertEqual(concept_key(alignment())[0],('TEST_ONLY_URI','TEST_CODE_A','TEST_UNIT','NativeSyntheaObservation'))

    def test_systems_distinct(self):
        self.assertNotEqual(concept_key(alignment())[0],concept_key(alignment(system='TEST_OTHER_URI'))[0])

    def test_codes_distinct(self):
        self.assertNotEqual(concept_key(alignment())[0],concept_key(alignment(code='TEST_B'))[0])

    def test_unit_distinct_no_conversion(self):
        self.assertNotEqual(concept_key(alignment())[0],concept_key(alignment(unit='test_unit'))[0])

    def test_observation_type_must_be_verified(self):
        self.assertIsNone(concept_key(alignment(kind='UNKNOWN'))[0])

    def test_no_whitespace_or_case_folding(self):
        self.assertNotEqual(concept_key(alignment())[0],concept_key(alignment(code=' test_code_a '))[0])

    def test_missing_fields_unknown(self):
        for field in ['system_uri','code','unit']:
            a=alignment();a[field]='';self.assertIsNone(concept_key(a)[0])

    def test_native_unit_contradiction(self):
        a=alignment();a['native']['unit']='OTHER'
        with self.assertRaises(ValueError):concept_key(a)

    def test_non_observation_not_a_measurement_concept(self):
        self.assertEqual(concept_key({},'conditions'),(None,'NOT_APPLICABLE_NON_OBSERVATION'))

    def test_unknown_never_promoted_even_with_good_alignment(self):
        x=run([ref('TEST_1',label='UNKNOWN')],['TEST_1'])
        self.assertEqual(x['unknown_count'],1);self.assertIsNone(x['modes']['selective']['concept_hit'])
        self.assertEqual(x['modes']['selective']['retrieved_verified_concept_count'],0)

    def test_single_concept_hit_is_binary_not_record_recall(self):
        x=run([ref('TEST_%d'%i) for i in range(10)],['TEST_0'])['modes']['selective']
        self.assertTrue(x['concept_hit']);self.assertEqual(x['concept_coverage']['value'],1)
        self.assertEqual(x['record_recall']['value'],.1)

    def test_multiple_concepts_partial_hit(self):
        rs=[ref('TEST_A'),ref('TEST_B',code='TEST_B')]
        x=run(rs,['TEST_A'],[alignment(),alignment('TEST_B')])['modes']['selective']
        self.assertTrue(x['concept_hit']);self.assertEqual(x['concept_coverage']['value'],.5)

    def test_duplicates_different_ids(self):
        x=run([ref('TEST_%d'%i) for i in range(5)],['TEST_%d'%i for i in range(5)])['modes']['selective']
        self.assertEqual(x['duplicate_concept_count'],4);self.assertEqual(x['duplicate_concept_ratio']['value'],.8)
        self.assertEqual(x['retrieved_verified_concept_count'],1)

    def test_empty_positive_gold_no_denominator_one(self):
        x=run([ref('TEST_N',code='TEST_N',label='NEGATIVE')],['TEST_N'])
        self.assertEqual(x['gold_unique_concept_count'],0);self.assertIsNone(x['modes']['selective']['concept_hit'])
        self.assertEqual(x['modes']['selective']['concept_coverage']['denominator'],0)
        self.assertIsNone(x['modes']['selective']['concept_coverage']['value'])

    def test_unknown_blocks_complete_coverage_even_known_hit(self):
        x=run([ref('TEST_A'),ref('TEST_U',label='UNKNOWN')],['TEST_A'])
        self.assertTrue(x['modes']['selective']['known_positive_concept_hit'])
        self.assertIsNone(x['modes']['selective']['concept_coverage']['value'])

    def test_ceiling(self):
        x=run([ref('TEST_%d'%i) for i in range(10)],['TEST_0','TEST_1'])['modes']['selective']
        self.assertEqual(x['record_recall_ceiling_at5']['value'],.5)
        self.assertEqual(x['ceiling_normalized_record_recall']['value'],.4)

    def test_small_positive_capacity(self):
        x=run([ref('TEST_0')],['TEST_0'])['modes']['selective']
        self.assertEqual(x['record_recall_ceiling_at5']['value'],1)

    def test_macro_micro_differ_and_patients_not_deduplicated(self):
        a=run([ref('TEST_A%d'%i) for i in range(10)],['TEST_A0'])
        b=run([ref('TEST_B')],['TEST_B'])
        x=aggregate([a,b],'selective')
        self.assertAlmostEqual(x['record_recall_macro']['value'],.55)
        self.assertAlmostEqual(x['record_recall_micro']['value'],2/11)
        self.assertEqual(x['concept_coverage_micro']['denominator'],2)

    def test_off_control_and_full_history_different_budget(self):
        x=run([ref('TEST_%d'%i) for i in range(10)],['TEST_0'])
        self.assertEqual(x['modes']['full_history']['returned_records'],10)
        self.assertEqual(x['modes']['none']['concept_hit'],False)
        self.assertEqual(x['modes']['none']['retrieval_failure_reason'],'RETRIEVAL_DISABLED_CONTROL')
        self.assertIsNone(x['modes']['none']['record_precision']['value'])

    def test_immutable_inputs(self):
        rs=[ref('TEST_0')];before=deepcopy(rs);x=run(rs,['TEST_0']);public_query(x)
        self.assertEqual(rs,before)

    def test_clinical_label_rejected(self):
        x=ref('TEST_0');x['clinical_relevance']=True
        with self.assertRaises(ValueError):run([x],['TEST_0'])

    def test_missing_query_mapping_blocks_metric(self):
        a=alignment();a.pop('unit');x=run([ref('TEST_0')],['TEST_0'],[alignment(),a])
        self.assertFalse(x['concept_recall_eligible'])

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):run([ref('TEST_0')],['TEST_0','TEST_0'])

    def test_cross_patient_or_outside_pool_id_rejected(self):
        with self.assertRaises(ValueError):run([ref('TEST_0')],['TEST_OTHER_PATIENT'])

    def test_budget_rejected(self):
        with self.assertRaises(ValueError):run([ref('TEST_%d'%i) for i in range(6)],['TEST_%d'%i for i in range(6)])

    def test_no_complete_queries_null_not_zero(self):
        x=aggregate([run([ref('TEST_U',label='UNKNOWN')],[])],'selective')
        self.assertIsNone(x['concept_hit_rate']['value']);self.assertIsNone(x['record_recall_micro']['value'])

    def test_private_sets_removed(self):
        x=public_query(run([ref('TEST_0')],['TEST_0']))
        self.assertNotIn('private_sets',x);self.assertNotIn('private_sets',x['modes']['selective'])

    def test_negative_mapped_concept_is_unique_but_not_hit(self):
        x=run([ref('TEST_0'),ref('TEST_N','TEST_N','NEGATIVE')],['TEST_N'])['modes']['selective']
        self.assertEqual(x['retrieved_verified_concept_count'],1);self.assertFalse(x['concept_hit'])

    def test_positive_outside_query_rejected(self):
        with self.assertRaises(ValueError):run([ref('TEST_0','TEST_OTHER')],['TEST_0'])

    def test_zero_denominator(self):
        self.assertIsNone(ratio(0,0)['value'])


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.original={'memory_id':'TEST_M','patient_id':'TEST_P','visit_id':'TEST_PREVIOUS',
            'visit_started_at':'2020-01-01T00:00:00Z','observed_at':'2020-01-01T01:00:00Z',
            'available_at':'2020-01-01T02:00:00Z'}
        self.identity={'patient_id':'TEST_P','visit_id':'TEST_CURRENT','started_at':'2020-02-01T00:00:00Z'}
        rules=['M01_PATIENT','M02_PREVIOUS_VISIT','M03_CHRONOLOGY','M04_FIXED_MEMORY_CUTOFF',
               'M05_ORIGINAL_RECORD','M06_ORIGINAL_FINGERPRINT','M07_SOURCE_RECORD',
               'M08_SOURCE_FINGERPRINT','M09_SOURCE_TO_MEMORY','M10_SOURCE_TIME_BOUND']
        self.row=ref('TEST_M');self.row.update(original_record_fingerprint=digest(self.original),
            checks=[{'rule_id':r,'result':'PASS'} for r in rules]);self.pin()

    def pin(self):self.row['evidence_fingerprint']=digest({k:v for k,v in self.row.items() if k!='evidence_fingerprint'})

    def test_valid(self):self.assertTrue(validate_reference(self.row,self.original,self.identity))

    def test_reference_tamper(self):
        self.row['label']='NEGATIVE'
        with self.assertRaisesRegex(ValueError,'REFERENCE_FINGERPRINT'):validate_reference(self.row,self.original,self.identity)

    def test_original_tamper(self):
        self.original['patient_id']='OTHER'
        with self.assertRaisesRegex(ValueError,'ORIGINAL_MEMORY'):validate_reference(self.row,self.original,self.identity)

    def test_patient_boundary(self):
        self.identity['patient_id']='OTHER'
        with self.assertRaisesRegex(ValueError,'PATIENT_VISIT'):validate_reference(self.row,self.original,self.identity)

    def test_future_cutoff(self):
        self.identity['started_at']='2019-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'PATIENT_VISIT'):validate_reference(self.row,self.original,self.identity)

    def test_source_failure(self):
        self.row['checks'][0]['result']='UNKNOWN';self.pin()
        with self.assertRaisesRegex(ValueError,'SOURCE_CHECK'):validate_reference(self.row,self.original,self.identity)

    def test_file_hash_tamper_detected_without_source_modification(self):
        from scripts.audit_research_gaps import sha,check
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'TEST_ONLY.txt';p.write_text('TEST_ONLY',encoding='utf-8')
            h=sha(p);self.assertEqual(check({str(p):h})['status'],'PASS')
            self.assertEqual(check({str(p):'0'*64})['status'],'FAIL')
            self.assertEqual(p.read_text(encoding='utf-8'),'TEST_ONLY')

    def test_native_pointer_and_encounter_binding(self):
        from scripts.evaluate_memory_concepts import native_check
        obs={'value':1,'start':2,'unit':'TEST_UNIT','codes':[{'code':'TEST_A'}],'type':'TEST_A'}
        document={'record':{'encounters':[{'uuid':'TEST_VISIT','start':0,'observations':[obs]}]}}
        a={'native':{'pointer':'/record/encounters/0/observations/0','native_value':1,
            'start':2,'unit':'TEST_UNIT','all_codings':[{'code':'TEST_A'}],
            'native_type':'TEST_A','encounter_start':0}}
        self.assertTrue(native_check(a,document,'TEST_VISIT'))
        with self.assertRaisesRegex(ValueError,'ENCOUNTER'):native_check(a,document,'OTHER_VISIT')
        document['record']['encounters'][0]['observations'][0]['value']=2
        with self.assertRaisesRegex(ValueError,'POINTER_VALUE'):native_check(a,document,'TEST_VISIT')


if __name__=='__main__':unittest.main()
