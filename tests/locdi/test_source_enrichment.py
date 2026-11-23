"""TEST_ONLY source/release/partial metric contracts. No clinical labels."""
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
import unittest

from app.locdi.llm_setup import ROOT,digest
from app.locdi.source_enrichment import (fact_row,validate_fact,iso_ms,released_projection,validate_candidate,
    make_pool,judged_metrics,validate_partial_review,PARTIAL_VERSION,PARTIAL_PROCEDURE)
from scripts.enrich_full_cohort_sources import resource_entries,write,prepare,extract_past_context,partial_evaluate
from test_memory_reference_evaluation import fixture


class SourceEnrichmentTests(unittest.TestCase):
    def source(self):
        doc={'attributes':{'id':'TEST_ONLY_P'},'record':{'encounters':[{'uuid':'TEST_ONLY_V','start':1000,
             'observations':[{'uuid':'TEST_ONLY_O','start':2000,'value':7}]}]}}
        case={'dataset':'TEST_ONLY','case_ref':'TEST_ONLY_CASE','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_V'}
        row=fact_row(case,'TEST_ONLY.json','TEST_ONLY_SHA','/record/encounters/0/observations/0',
                     doc['record']['encounters'][0]['observations'][0],'Observation','/record/encounters/0')
        return doc,row

    def release(self):
        p,_,_=fixture();ask=p['released_source_event'];ask['event_id']='TEST_ONLY_ASK'
        test={'event_id':'TEST_ONLY_TEST','action':'Test','target':'TEST_ONLY_LAB','available_minute':3,
              'evidence':json.dumps({'records':[{'code':'TEST_ONLY_C','description':'TEST_ONLY_LAB','event_type':'observations',
                'recorded_at':'2020-01-02T00:01:00+00:00','recorded_evidence':{'kind':'recorded_observation','category':'laboratory','units':'u','value':3}}],
                'replay_bin':'TEST_ONLY_PRIVATE_DISCRETIZATION'})}
        obs={'elapsed_minutes':5,'history':p['released_history'],'objective_findings':[test['target']+': '+test['evidence']],
             'working_assessment':'TEST_ONLY_SECRET','gold':'TEST_ONLY_SECRET'}
        results=[{'test_name':test['target'],'observed_minute':4}]
        return p['private_identity'],obs,[ask,test],results

    def test_actual_source_fact_and_no_recorded_to_available_copy(self):
        doc,row=self.source();validate_fact(row,doc,'TEST_ONLY_SHA')
        self.assertIsNotNone(row['recorded_at']);self.assertIsNone(row['available_at'])
        self.assertEqual(row['projection_role'],'PRIVATE_SOURCE_ONLY_NOT_AGENT_INPUT')

    def test_explicit_source_available_at_preserved(self):
        doc,row=self.source();value=doc['record']['encounters'][0]['observations'][0]
        value['available_at']='1970-01-01T00:01:00+00:00'
        row=fact_row({'dataset':'TEST_ONLY','case_ref':'TEST_ONLY','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_V'},
            'TEST_ONLY.json','TEST_ONLY_SHA','/record/encounters/0/observations/0',value,'Observation','/record/encounters/0')
        self.assertEqual(row['available_at'],value['available_at'])
        self.assertNotEqual(row['available_at'],row['recorded_at'])

    def test_source_changed_or_hash_mismatch_rejected(self):
        doc,row=self.source()
        with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):validate_fact(row,doc,'OTHER')
        doc['record']['encounters'][0]['observations'][0]['value']=8
        with self.assertRaisesRegex(ValueError,'SOURCE_FACT_FINGERPRINT'):validate_fact(row,doc,'TEST_ONLY_SHA')

    def test_patient_visit_link_rejected(self):
        doc,row=self.source();row['patient_reference']='OTHER'
        with self.assertRaisesRegex(ValueError,'SOURCE_PATIENT_MISMATCH'):validate_fact(row,doc,'TEST_ONLY_SHA')
        doc,row=self.source();row['visit_reference']='OTHER'
        with self.assertRaisesRegex(ValueError,'SOURCE_VISIT_MISMATCH'):validate_fact(row,doc,'TEST_ONLY_SHA')

    def test_reason_and_condition_are_not_automatically_complaints(self):
        enc={'uuid':'TEST_ONLY_V','reasons':[{'code':'TEST_ONLY_CONDITION'}],'conditions':[{'uuid':'TEST_ONLY_C'}],
             'medications':[{'uuid':'TEST_ONLY_M'}]}
        types=[k for k,_,_ in resource_entries(enc,'/record/encounters/0')]
        self.assertIn('EncounterReason',types);self.assertIn('Condition',types);self.assertIn('Medication',types)
        self.assertNotIn('MedicationRequest',types);self.assertNotIn('ChiefComplaint',types)

    def test_pre_epoch_timestamp_supported(self):
        self.assertTrue(iso_ms(-1000).startswith('1969-12-31'))

    def test_exact_released_values_and_source_reproduction(self):
        identity,obs,events,results=self.release()
        c=released_projection(obs,events,results,identity);validate_candidate(c,events,identity)
        self.assertEqual(len(c['source_links']),2);self.assertEqual(len(c['source_facts']),2)
        self.assertEqual(c['source_facts'][1]['value'],3)
        self.assertNotIn('TEST_ONLY_SECRET',c['query']);self.assertNotIn('PRIVATE_DISCRETIZATION',c['query'])
        self.assertNotIn('chief_complaint',c['projection'])

    def test_unreceived_test_value_blocked(self):
        identity,obs,events,results=self.release()
        with self.assertRaisesRegex(ValueError,'TEST_NOT_RECEIVED'):released_projection(obs,events,[],identity)
        results[0]['observed_minute']=6
        with self.assertRaisesRegex(ValueError,'TEST_NOT_RECEIVED'):released_projection(obs,events,results,identity)

    def test_test_before_source_available_blocked(self):
        identity,obs,events,results=self.release();events[1]['available_minute']=5
        with self.assertRaisesRegex(ValueError,'TEST_NOT_RECEIVED'):released_projection(obs,events,results,identity)

    def test_future_source_fact_blocked(self):
        identity,obs,events,results=self.release();raw=json.loads(events[1]['evidence'])
        raw['records'][0]['recorded_at']='2021-01-01T00:00:00+00:00';events[1]['evidence']=json.dumps(raw)
        obs['objective_findings']=[events[1]['target']+': '+events[1]['evidence']]
        with self.assertRaisesRegex(ValueError,'FUTURE_SOURCE_FACT'):released_projection(obs,events,results,identity)

    def test_gold_and_condition_in_source_evidence_blocked(self):
        for change in ('gold','condition'):
            identity,obs,events,results=self.release();raw=json.loads(events[1]['evidence'])
            if change=='gold':raw['gold']='TEST_ONLY_SECRET'
            else:raw['records'][0]['event_type']='conditions'
            events[1]['evidence']=json.dumps(raw);obs['objective_findings']=[events[1]['target']+': '+events[1]['evidence']]
            with self.assertRaises(ValueError):released_projection(obs,events,results,identity)

    def test_altered_query_or_source_fingerprint_blocked(self):
        identity,obs,events,results=self.release();c=released_projection(obs,events,results,identity)
        c['query']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'QUERY_CANDIDATE_REPRODUCTION'):validate_candidate(c,events,identity)
        c=released_projection(obs,events,results,identity);c['source_links'][0]['source_fingerprint']='0'*64
        with self.assertRaisesRegex(ValueError,'RELEASE_SOURCE_FINGERPRINT'):validate_candidate(c,events,identity)

    def test_all_memory_retained_and_pool_deterministic(self):
        p,_,_=fixture();records=[]
        for i in range(40):records.append(dict(p['eligible_memory'][0],memory_id='TEST_ONLY_%03d'%i,text='TEST_ONLY code '+str(i)))
        before=deepcopy(records);one=make_pool(records,p['private_identity'],'TEST_ONLY code 7',[{'code':'7','description':'TEST_ONLY code'}],1)
        two=make_pool(records,p['private_identity'],'TEST_ONLY code 7',[{'code':'7','description':'TEST_ONLY code'}],1)
        self.assertEqual(one,two);self.assertEqual(records,before);self.assertEqual(one[2]['eligible_count'],40)
        self.assertEqual(len(one[1]),5);self.assertLessEqual(one[2]['pooled_count'],25)
        self.assertEqual(one[2]['unjudged_count'],40)

    def test_pool_excludes_other_patient_and_future(self):
        p,_,_=fixture();records=p['eligible_memory']+[dict(p['eligible_memory'][0],memory_id='OTHER',patient_id='OTHER')]
        selected,_,counts=make_pool(records,p['private_identity'],'pain',[],1)
        self.assertEqual(counts['eligible_count'],2);self.assertNotIn('OTHER',[r['memory_id'] for r in selected])

    def test_empty_patient_pool_is_preserved(self):
        p,_,_=fixture();selected,_,counts=make_pool([],p['private_identity'],'pain',[],30)
        self.assertEqual(selected,[]);self.assertEqual(counts['eligible_count'],0)
        self.assertIsNone(counts['unpooled_fraction'])

    def test_partial_metrics_use_judged_denominator_not_full_recall(self):
        judgments=[{'memory_id':'a','relevant':True,'stale':False},{'memory_id':'b','relevant':False,'stale':True},
                   {'memory_id':'c','relevant':None,'stale':None}]
        r=judged_metrics(['a','b','c','outside'],['a','b','c'],judgments)
        self.assertEqual(r['judged_set_precision']['value'],.5);self.assertEqual(r['judged_coverage']['value'],.5)
        self.assertEqual(r['unjudged_returned_count'],2);self.assertEqual(r['known_relevant_retrieval_count'],1)
        self.assertEqual(r['full_pool_recall'],'NOT_IDENTIFIABLE_NOT_COMPUTED')

    def test_all_unjudged_is_not_false_and_empty_denominator_is_na(self):
        r=judged_metrics(['a'],['a'],[{'memory_id':'a','relevant':None,'stale':None}])
        self.assertIsNone(r['judged_set_precision']['value']);self.assertEqual(r['judged_coverage']['value'],0)
        self.assertEqual(r['unjudged_returned_count'],1)
        self.assertIsNone(judged_metrics([],[],[])['judged_coverage']['value'])

    def test_partial_reviewer_metadata_required(self):
        with self.assertRaisesRegex(ValueError,'INDEPENDENT_PARTIAL_REVIEW_REQUIRED'):
            p={'query_candidate':{'query_fingerprint':'TEST_ONLY'}}
            validate_partial_review(p,{'schema_version':PARTIAL_VERSION,'packet_fingerprint':digest(p),'query_fingerprint':'TEST_ONLY'})

    def test_partial_reference_hash_binding(self):
        with self.assertRaisesRegex(ValueError,'PARTIAL_REFERENCE_BINDING'):
            validate_partial_review({'query_candidate':{'query_fingerprint':'TEST_ONLY'}},{})

    def test_partial_incomplete_judged_pool_cannot_fake_full_truth(self):
        with self.assertRaisesRegex(ValueError,'JUDGED_POOL_COVERAGE'):judged_metrics(['a'],['a','b'],[{'memory_id':'a','relevant':True,'stale':False}])

    def test_reviewed_partial_fixture_keeps_uncertainty_and_rejects_wrong_types(self):
        packet={'query_candidate':{'query_fingerprint':'TEST_ONLY','decision_time':'2020-01-01T00:00:00+00:00'},
                'candidate_memories':[{'memory_id':'TEST_ONLY_R'}]}
        review={'schema_version':PARTIAL_VERSION,'packet_fingerprint':digest(packet),'query_fingerprint':'TEST_ONLY',
            'review_status':'REVIEWED','reviewer_independent':True,'reviewer_private_ref':'TEST_ONLY_REVIEWER',
            'procedure_version':PARTIAL_PROCEDURE,'query_suitable':True,'visibility_reviewed':True,
            'recorded_time_proxy_reviewed':True,'methods_or_predictions_seen':False,'reviewed_at':'2020-01-02T00:00:00+00:00',
            'rationale':'TEST_ONLY','records':[{'memory_id':'TEST_ONLY_R','relevant':None,'stale':None,'conflict':None,
                'rationale':'TEST_ONLY uncertain','not_evaluable_reason':'TEST_ONLY'}]}
        result=validate_partial_review(packet,review);self.assertIsNone(result[0]['relevant'])
        for key in ('relevant','stale','conflict'):
            bad=deepcopy(review);bad['records'][0][key]='true'
            with self.assertRaisesRegex(ValueError,'INVALID_PARTIAL_JUDGMENT'):validate_partial_review(packet,bad)

    def test_past_context_extracted_without_memory_or_current_status_promotion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            root=Path(folder);doc={'attributes':{'id':'TEST_ONLY_P'},'record':{'encounters':[
                {'uuid':'TEST_ONLY_PAST','start':1000,'allergies':[{'uuid':'TEST_ONLY_A','start':2000}],
                 'medications':[{'uuid':'TEST_ONLY_M','start':3000,'stop':1000000}]},
                {'uuid':'TEST_ONLY_NOW','start':600000,'allergies':[{'uuid':'TEST_ONLY_CURRENT','start':600000}]},
                {'uuid':'TEST_ONLY_FUTURE','start':900000,'allergies':[{'uuid':'TEST_ONLY_FUTURE_A','start':900000}]}]}}
            path=root/'TEST_ONLY.json';write(path,doc);before=path.read_bytes()
            case={'dataset':'Original50','case_ref':'TEST_ONLY','patient_id':'TEST_ONLY_P','visit_id':'TEST_ONLY_NOW',
                  'decision_time':'1970-01-01T00:10:00+00:00','raw_path':str(path)}
            result=extract_past_context([case],root,root)
            self.assertEqual(result['record_occurrences'],2);self.assertEqual(path.read_bytes(),before)
            records=[json.loads(x) for x in (root/'historical_context_source_facts.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['visit_reference']=='TEST_ONLY_PAST' for r in records))
            self.assertTrue(all(r['available_at'] is None for r in records))
            self.assertTrue(any(r['contains_future_stop_field'] for r in records))

    def test_overwrite_and_namespace_escape_blocked(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            p=Path(folder)/'TEST_ONLY.json';write(p,{'a':1});before=p.read_bytes()
            with self.assertRaises(FileExistsError):write(p,{'a':2})
            self.assertEqual(p.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'INVALID_NAMESPACE'):prepare('../unsafe')

    def test_partial_missing_reference_and_output_escape_blocked(self):
        with self.assertRaisesRegex(ValueError,'REVIEWED_PARTIAL_REFERENCE_REQUIRED'):
            partial_evaluate(ROOT/'outputs/private/UNUSED',None,None,ROOT/'outputs/research_evaluation/UNUSED')
        with self.assertRaisesRegex(ValueError,'PUBLIC_OUTPUT_PATH_REQUIRED'):
            partial_evaluate(ROOT/'outputs/private/UNUSED',ROOT/'outputs/private/UNUSED.json','0'*64,ROOT/'UNSAFE_OUTPUT')


if __name__=='__main__':unittest.main()
