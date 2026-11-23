"""TEST_ONLY contracts. No network, model requests, clinical labels or fitting."""
from copy import deepcopy
from types import SimpleNamespace
import json,socket,unittest
from unittest.mock import patch
from app.core.longitudinal_memory import PatientMemoryStore
from app.locdi.llm_setup import digest
from app.locdi.diagnostic_memory_rag import (DiagnosticMemoryRetriever,canonical_record,public_projection,
    knowledge_projection,historical_series,terms,INSTRUMENT)
from app.locdi.memory_rag_backend import make_memory_rag_backend
from app.locdi.medical_knowledge import MedicalKnowledgeIndex
from app.locdi.medical_knowledge_catalog import corpus

def fixture():
    identity={'patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_CURRENT','started_at':'2002-01-01T00:00:00Z'}
    records=[];sources={}
    for n,text in enumerate(['pain score','hemoglobin A1c glucose','blood pressure systolic','hemoglobin A1c glucose'],1):
        r={'memory_id':'TEST_ONLY_%d'%n,'patient_id':identity['patient_id'],'visit_id':'TEST_ONLY_PAST',
            'visit_started_at':'2000-01-01T00:00:00Z','observed_at':'2001-01-0%dT00:00:00Z'%n,
            'available_at':'2001-01-0%dT00:00:00Z'%n,'source_id':'TEST_ONLY_SRC%d'%n,
            'kind':'objective_finding','text':text,'supersedes':[]}
        records.append(r);sources[r['memory_id']]={'status':'VERIFIED_EXACT_CSV_SOURCE','record_fingerprint':digest(r),
            'source_file':'TEST_ONLY.csv','source_row':n+1,'source_type':'observations','code':'4548-4','unit':'%','value':str(n)}
    store=PatientMemoryStore(records)
    return identity,records,sources,store

class MemoryRAGContracts(unittest.TestCase):
    def setUp(self):
        p=patch.object(socket.socket,'connect',side_effect=AssertionError('NO_NETWORK_TEST_ONLY'));p.start();self.addCleanup(p.stop)
        self.identity,self.records,self.sources,self.store=fixture()
        self.pool=self.store.retrieve(self.identity,'pain','full_history')['records']
        self.legacy=self.store.retrieve(self.identity,'pain','selective',5)
    def retrieve(self,query='hemoglobin A1c',pool=None,**kw):
        return DiagnosticMemoryRetriever(self.sources).retrieve(self.legacy,self.pool if pool is None else pool,self.identity,query,**kw)
    def test_default_factory_delegates_without_any_extra_flags_or_patches(self):
        with patch('app.locdi.memory_rag_backend.make_common_backend',return_value='TEST_ONLY') as m:
            cases={'TEST_ONLY':{}};self.assertEqual(make_memory_rag_backend(cases,config='TEST_ONLY'),'TEST_ONLY')
            m.assert_called_once_with(cases,config='TEST_ONLY')
    def test_legacy_store_and_records_unchanged_after_opt_in(self):
        snapshot=self.store.snapshot();before=deepcopy(self.legacy);self.retrieve()
        self.assertEqual(self.store.snapshot(),snapshot);self.assertEqual(self.legacy,before)
        self.assertEqual(self.store.retrieve(self.identity,'pain','selective',5),before)
    def test_full_pool_finds_records_outside_legacy_top5(self):
        r,t=self.retrieve();self.assertNotIn('TEST_ONLY_2',[x['memory_id'] for x in self.legacy['records']])
        self.assertIn('TEST_ONLY_2',[x['memory_id'] for x in r['records']]);self.assertEqual(t['candidate_scope'],'ALL_ELIGIBLE_FIXED_PAST_RECORDS')
    def test_reproducible_tie_uses_more_recent_then_id(self):
        a=self.retrieve();self.assertEqual(a,self.retrieve());self.assertEqual(a[0]['records'][0]['memory_id'],'TEST_ONLY_4')
    def test_original_values_text_and_timestamps_preserved(self):
        r,_=self.retrieve()
        for x in r['records']:self.assertEqual(canonical_record(x),next(z for z in self.records if z['memory_id']==x['memory_id']))
    def test_zero_query_and_unmatched_query_do_not_guess(self):
        for q in ('','zzzzqqqq'):
            r,t=self.retrieve(q);self.assertEqual(r['records'],[]);self.assertEqual(t['status'],'MEMORY_UNAVAILABLE')
    def test_max_five_and_complete_record_character_budget(self):
        with self.assertRaises(ValueError):self.retrieve(top_k=6)
        r,_=self.retrieve(max_chars=2);self.assertEqual(r['records'],[]);self.assertEqual(r['records_json_chars'],2)
    def test_wrong_patient_rejected(self):
        p=deepcopy(self.pool);p[0]['patient_id']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):self.retrieve(pool=p)
    def test_current_visit_rejected(self):
        p=deepcopy(self.pool);p[0]['visit_id']=self.identity['visit_id']
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):self.retrieve(pool=p)
    def test_future_available_at_rejected(self):
        p=deepcopy(self.pool);p[0]['available_at']='2003-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'BOUNDARY'):self.retrieve(pool=p)
    def test_missing_source_rejected(self):
        self.sources.pop('TEST_ONLY_1')
        with self.assertRaisesRegex(ValueError,'UNVERIFIED'):self.retrieve()
    def test_tampered_text_rejected(self):
        p=deepcopy(self.pool);p[0]['text']='TEST_ONLY_FORGED'
        with self.assertRaisesRegex(ValueError,'UNVERIFIED'):self.retrieve(pool=p)
    def test_superseded_not_selected(self):
        p=deepcopy(self.pool)
        for r in p:r['status_as_of_visit']='superseded'
        self.assertEqual(self.retrieve(pool=p)[0]['records'],[])
    def test_exact_source_duplicate_removed_but_serial_measurements_retained(self):
        self.sources['TEST_ONLY_4']['source_row']=self.sources['TEST_ONLY_2']['source_row']
        r,t=self.retrieve();self.assertEqual(t['duplicate_source_records_removed'],1)
        self.assertEqual(len(r['records']),1)
    def test_series_preserves_unit_and_chronology_without_trend_claim(self):
        s=historical_series(self.records,self.sources);self.assertEqual(len(s),1)
        self.assertEqual([r['value'] for r in s[0]['records']],['1','2','3','4']);self.assertNotIn('abnormal',str(s))
    def test_numerical_values_alone_are_not_retrieval_tokens(self):
        self.assertEqual(terms('8.2 130 50'),[]);self.assertIn('4548-4',terms('4548-4'))

class QueryAndKnowledgeContracts(unittest.TestCase):
    def setUp(self):
        self.binding={'rows':[{'kind':'Test','target':'TEST_ONLY_A','description':'Hemoglobin A1c','code':'4548-4','unit':'%',
            'evidence':json.dumps({'value':'8.2','units':'%'}),'available_minute':5}]}
        row=self.binding['rows'][0];self.text=row['target']+': '+row['evidence']
        self.obs={'elapsed_minutes':5,'chief_complaint':'','current_symptoms':[INSTRUMENT],
            'objective_findings':[self.text],'history':[],'working_assessment':'TEST_ONLY_PRIVATE_PREDICTION',
            'longitudinal_memory':{'records':[{'text':'TEST_ONLY_HISTORICAL_DIAGNOSIS'}]}}
        self.runtime=SimpleNamespace(current_time_min=5,arrival_time_min=0,completed_tests=['TEST_ONLY_A'],objective_findings=[self.text])
    def test_test_receipt_does_not_require_ask_history(self):
        q,t=public_projection(self.obs,self.binding,self.runtime);self.assertIn('Hemoglobin A1c',t['query'])
        self.assertEqual(q['current_symptoms'],[]);self.assertEqual(t['visibility_status'],'RUNTIME_RECEIPT_VERIFIED')
    def test_unreceived_test_rejected(self):
        self.runtime.completed_tests=[]
        with self.assertRaisesRegex(ValueError,'NOT_RECEIVED'):public_projection(self.obs,self.binding,self.runtime)
    def test_future_measurement_rejected(self):
        self.obs['elapsed_minutes']=0
        with self.assertRaisesRegex(ValueError,'NOT_YET'):public_projection(self.obs,self.binding)
    def test_gold_fields_rejected(self):
        self.obs['gold']={'diagnosis':'TEST_ONLY'}
        with self.assertRaisesRegex(ValueError,'PRIVATE'):public_projection(self.obs,self.binding)
    def test_predicted_and_memory_diagnoses_excluded(self):
        q,t=public_projection(self.obs,self.binding)
        self.assertNotIn('PRIVATE_PREDICTION',t['query']);self.assertNotIn('HISTORICAL_DIAGNOSIS',t['query'])
    def test_unreleased_binding_never_appears_in_query(self):
        self.obs['objective_findings']=[];q,t=public_projection(self.obs,self.binding,self.runtime)
        self.assertNotIn('Hemoglobin',t['query']);self.assertEqual(t['released_measurements'],[])
    def test_actual_ask_history_required_and_unknown_answers_excluded(self):
        r=deepcopy(self.binding['rows'][0]);r.update(kind='Ask',target='TEST_ONLY_ASK');b={'rows':[r]}
        self.obs['objective_findings']=[];self.obs['history']=[{'question':r['target'],'answer':r['evidence'],'answer_status':'answered'}]
        self.runtime.question_history=[]
        with self.assertRaisesRegex(ValueError,'ASK_NOT_EXECUTED'):public_projection(self.obs,b,self.runtime)
        self.runtime.question_history=deepcopy(self.obs['history']);public_projection(self.obs,b,self.runtime)
        self.obs['history'][0]['answer_status']='unknown'
        self.assertEqual(public_projection(self.obs,b)[1]['released_measurements'],[])
    def test_external_knowledge_separate_from_patient_evidence(self):
        before=deepcopy(self.obs);p,_=knowledge_projection(MedicalKnowledgeIndex(corpus()),self.obs,self.binding,runtime=self.runtime)
        self.assertEqual(self.obs,before);self.assertEqual(p['scope'],'EXTERNAL_KNOWLEDGE_NOT_PATIENT_EVIDENCE')
        self.assertLessEqual(len(p['results']),3);self.assertNotIn('evidence_indices',p)
        self.assertTrue(all(x['references'] for x in p['results']))
    def test_empty_query_knowledge_unavailable(self):
        self.obs['objective_findings']=[];p,_=knowledge_projection(MedicalKnowledgeIndex(corpus()),self.obs,self.binding)
        self.assertEqual(p['status'],'KNOWLEDGE_UNAVAILABLE');self.assertEqual(p['results'],[])

if __name__=='__main__':unittest.main()
