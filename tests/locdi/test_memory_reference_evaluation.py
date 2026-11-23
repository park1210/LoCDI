"""TEST_ONLY review/visibility fixtures; never clinical judgments or real labels."""
import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha
from app.locdi.memory_reference_evaluation import (VERSION,PROCEDURE,released_query,validate_packet,
    validate_review,score_packet,load_bundle,evaluate)
from app.locdi.episode_exporter import _write

def fixture():
    event={'event_id':'TEST_ONLY_E','action':'Ask','target':'What was the pain score?', 'available_minute':0,
        'evidence':json.dumps({'facts':[{'code':'TEST_ONLY','description':'pain score','value':'7',
            'units':'score','recorded_at':'2020-01-02T00:00:00+00:00'}]})}
    history=[{'question':event['target'],'answer':event['evidence'],'answer_status':'provided','observed_minute':2}]
    query,_=released_query(history,snapshot_minute=2,source_event=event)
    record={'memory_id':'TEST_ONLY_R1','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_PAST',
        'visit_started_at':'2020-01-01T00:00:00+00:00','observed_at':'2020-01-01T01:00:00+00:00',
        'available_at':'2020-01-01T02:00:00+00:00','source_id':'TEST_ONLY_SOURCE','kind':'history','text':'pain score','supersedes':[]}
    p={'snapshot_id':'Snapshot001','private_identity':{'patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_NOW',
        'started_at':'2020-01-02T00:00:00+00:00'},'decision_minute':2,'decision_time':'2020-01-02T00:02:00+00:00',
        'query':query,'query_fingerprint':digest(query),'released_history':history,'released_source_event':event,
        'eligible_memory':[record,dict(record,memory_id='TEST_ONLY_R2',text='unrelatedword')],
        'review_status':'UNREVIEWED','model_prediction_included':False,'retrieval_output_included':False}
    row={'snapshot_id':'Snapshot001','packet_sha256':'TEST_ONLY_HASH','query_fingerprint':digest(query),
        'review_status':'REVIEWED','reviewer_independent':True,'reviewer_private_ref':'TEST_ONLY_REVIEWER',
        'reviewed_at':'2020-01-03T00:00:00+00:00','procedure_version':PROCEDURE,'query_suitable':True,
        'query_visibility_reviewed':True,'whole_eligible_pool_reviewed':True,'model_or_retrieval_outputs_seen':False,
        'visibility_scope':'DEVELOPMENT_REPLAY_ONLY','recorded_time_proxy_explicitly_reviewed':True,'rationale':'TEST_ONLY',
        'records':[{'memory_id':r['memory_id'],'relevant':i==0,'stale':i==1,'conflict':False,
            'not_evaluable_reason':None,'rationale':'TEST_ONLY'} for i,r in enumerate(p['eligible_memory'])]}
    m={'fingerprint':'TEST_ONLY_MANIFEST','packets':[{'snapshot_id':'Snapshot001','sha256':'TEST_ONLY_HASH'}]}
    review={'schema_version':VERSION,'manifest_fingerprint':m['fingerprint'],'snapshots':[row]}
    return p,m,review

class MemoryReferenceTests(unittest.TestCase):
    def test_value_preserved_label_not_promoted_to_complaint(self):
        p,_,_=fixture();q,clean=released_query(p['released_history'],snapshot_minute=2,source_event=p['released_source_event'])
        self.assertIn('7',q);self.assertNotIn('chief_complaint',clean);self.assertNotIn('current_symptoms',clean)
    def test_future_release_rejected(self):
        p,_,_=fixture();p['released_history'][0]['observed_minute']=3
        with self.assertRaisesRegex(ValueError,'ANSWER_NOT_YET_RELEASED'):validate_packet(p)
    def test_unreleased_source_rejected(self):
        p,_,_=fixture();p['released_source_event']['available_minute']=3
        with self.assertRaises(ValueError):validate_packet(p)
    def test_gold_and_hidden_fields_rejected(self):
        p,_,_=fixture()
        for key in ('gold','evaluation','source_event_store','predicted_diagnosis'):
            h=deepcopy(p['released_history']);h[0][key]='TEST_ONLY_SECRET'
            with self.assertRaisesRegex(ValueError,'HIDDEN_FIELD_NOT_ALLOWED'):
                released_query(h,snapshot_minute=2,source_event=p['released_source_event'])
    def test_label_without_value_rejected(self):
        p,_,_=fixture();e=p['released_source_event'];v=json.loads(e['evidence']);v['facts'][0]['value']=None
        e['evidence']=json.dumps(v);p['released_history'][0]['answer']=e['evidence']
        with self.assertRaisesRegex(ValueError,'SOURCE_FACT_FIELDS'):validate_packet(p)
    def test_nullable_source_fields_and_replay_metadata_not_in_query(self):
        p,_,_=fixture();e=p['released_source_event'];value=json.loads(e['evidence'])
        value.update(kind='TEST_ONLY_KIND',replay_bin='TEST_ONLY_BIN');value['facts'][0].update(onset=None,duration=None,active_status=None)
        e['evidence']=json.dumps(value);p['released_history'][0]['answer']=e['evidence']
        q,_=released_query(p['released_history'],snapshot_minute=2,source_event=e)
        self.assertIn('7',q);self.assertNotIn('TEST_ONLY_BIN',q);self.assertNotIn('active_status',q)
    def test_future_fact_and_decision_time_binding(self):
        p,_,_=fixture();e=p['released_source_event'];value=json.loads(e['evidence']);value['facts'][0]['recorded_at']='2020-01-03T00:00:00+00:00'
        e['evidence']=json.dumps(value);p['released_history'][0]['answer']=e['evidence']
        p['query'],_=released_query(p['released_history'],snapshot_minute=2,source_event=e);p['query_fingerprint']=digest(p['query'])
        with self.assertRaisesRegex(ValueError,'FUTURE_SOURCE_FACT'):validate_packet(p)
        p,_,_=fixture();p['decision_time']='2020-01-03T00:02:00+00:00'
        with self.assertRaisesRegex(ValueError,'DECISION_TIME_BINDING'):validate_packet(p)
    def test_query_tamper_rejected(self):
        p,_,_=fixture();p['query']='TEST_ONLY_OTHER';p['query_fingerprint']=digest(p['query'])
        with self.assertRaisesRegex(ValueError,'QUERY_PROJECTION_MISMATCH'):validate_packet(p)
    def test_other_patient_current_visit_and_future_memory_rejected(self):
        original,_,_=fixture()
        for changes in ({'patient_id':'OTHER'},{'visit_id':'TEST_ONLY_NOW'},{'available_at':'2020-01-03T00:00:00+00:00'}):
            p=deepcopy(original);p['eligible_memory'][0].update(changes)
            with self.assertRaisesRegex(ValueError,'PACKET_MEMORY_BOUNDARY'):validate_packet(p)
    def test_valid_complete_review_and_paired_denominators(self):
        p,m,r=fixture();validate_packet(p);rows=validate_review(m,[p],r);out=score_packet(p,rows[0])
        self.assertEqual(out[0]['precision']['value'],.5);self.assertEqual(out[1]['precision']['value'],1)
        self.assertEqual(out[1]['recall_at_5']['value'],1);self.assertEqual(out[1]['returned_record_count'],1)
        self.assertEqual(out[0]['stale_retrieval_rate']['value'],.5);self.assertEqual(out[1]['stale_retrieval_rate']['value'],0)
        self.assertEqual(out[0]['query_fingerprint'],out[1]['query_fingerprint'])
        self.assertEqual(out[0]['patient_leakage_count']+out[1]['future_cutoff_leakage_count'],0)
    def test_missing_or_uncertain_annotation_fails_closed(self):
        p,m,r=fixture()
        for key in ('relevant','stale','conflict'):
            bad=deepcopy(r);bad['snapshots'][0]['records'][0][key]=None
            with self.assertRaisesRegex(ValueError,'UNREVIEWED_OR_UNCERTAIN_RECORD'):validate_review(m,[p],bad)
    def test_incomplete_record_pool_not_dropped_to_make_reference(self):
        p,m,r=fixture();r['snapshots'][0]['records'].pop()
        with self.assertRaisesRegex(ValueError,'ANNOTATION_COVERAGE_OR_ORDER'):validate_review(m,[p],r)
    def test_independence_or_query_review_failure(self):
        p,m,r=fixture()
        for changes in ({'reviewer_independent':False},{'query_suitable':None},{'query_visibility_reviewed':False},
                        {'model_or_retrieval_outputs_seen':True},{'recorded_time_proxy_explicitly_reviewed':None}):
            bad=deepcopy(r);bad['snapshots'][0].update(changes)
            with self.assertRaises(ValueError):validate_review(m,[p],bad)
    def test_cohort_and_query_hash_binding(self):
        p,m,r=fixture()
        for changes in ({'snapshot_id':'Snapshot002'},{'query_fingerprint':'0'*64},{'packet_sha256':'0'*64}):
            bad=deepcopy(r);bad['snapshots'][0].update(changes)
            with self.assertRaises(ValueError):validate_review(m,[p],bad)
    def test_true_empty_relevant_set_is_na_recall_not_missing(self):
        p,_,r=fixture();row=r['snapshots'][0]
        for x in row['records']:x['relevant']=False
        out=score_packet(p,row)
        self.assertIsNone(out[1]['recall_at_5']['value']);self.assertEqual(out[1]['recall_at_5']['status'],'not_applicable')
        self.assertEqual(out[1]['precision']['value'],0)
    def test_missing_reference_never_retrieves(self):
        with patch('app.locdi.memory_reference_evaluation.PatientMemoryStore.retrieve') as retrieve:
            with self.assertRaisesRegex(ValueError,'REVIEWED_REFERENCE_REQUIRED'):evaluate('UNUSED',None,None)
            retrieve.assert_not_called()
    def make_bundle(self,root):
        p,m,r=fixture();source=root/'TEST_ONLY_SOURCE.txt';source.write_text('TEST_ONLY',encoding='utf8')
        pp=root/'Snapshot001.json';pp.write_text(json.dumps(p),encoding='utf8')
        m={'packets':[{'snapshot_id':'Snapshot001','file':'Snapshot001.json','sha256':sha(pp)}],
            'source_hashes':{str(source):sha(source)}};m['fingerprint']=digest(m)
        (root/'manifest.json').write_text(json.dumps(m),encoding='utf8')
        r['manifest_fingerprint']=m['fingerprint'];r['snapshots'][0]['packet_sha256']=sha(pp)
        ref=root/'TEST_ONLY_REVIEW.json';ref.write_text(json.dumps(r),encoding='utf8');return source,ref
    def test_source_hash_mismatch_before_retrieval(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/private') as directory:
            root=Path(directory);source,ref=self.make_bundle(root);source.write_text('CHANGED',encoding='utf8')
            with patch('app.locdi.memory_reference_evaluation.PatientMemoryStore.retrieve') as retrieve:
                with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):evaluate(root,ref,sha(ref))
                retrieve.assert_not_called()
    def test_reference_hash_mismatch_denied(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/private') as directory:
            root=Path(directory);_,ref=self.make_bundle(root)
            with self.assertRaisesRegex(ValueError,'REVIEWED_REFERENCE_HASH_MISMATCH'):evaluate(root,ref,'0'*64)
    def test_test_only_end_to_end_public_has_no_private_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/private') as directory:
            root=Path(directory);_,ref=self.make_bundle(root);out=evaluate(root,ref,sha(ref));text=json.dumps(out)
            self.assertEqual(len(out['rows']),2);self.assertNotIn('TEST_ONLY_P',text);self.assertNotIn('TEST_ONLY_R1',text)
    def test_existing_output_preserved_on_duplicate_write(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root=Path(directory);_write({'TEST_ONLY':1},'reports/result.json',root,'reports');original=(root/'reports/result.json').read_bytes()
            with self.assertRaises(FileExistsError):_write({'TEST_ONLY':2},'reports/result.json',root,'reports')
            self.assertEqual(original,(root/'reports/result.json').read_bytes())
