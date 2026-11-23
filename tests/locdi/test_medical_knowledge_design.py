"""TEST_ONLY knowledge, temporal display, rules and closed citation contracts."""
from copy import deepcopy
import json,socket,unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator,ValidationError
from app.locdi.medical_knowledge_catalog import corpus
from app.locdi.medical_knowledge import MedicalKnowledgeIndex,historical_labels,public_query
from app.locdi.evidence_selection_wire import encode,decode
from app.locdi.literature_rule_router import select_rule,LiteratureRuleRegistry
from app.locdi.current_ablation_llm_backend import _strict_object

class KnowledgeContracts(unittest.TestCase):
    def setUp(self):
        p=patch.object(socket.socket,'connect',side_effect=AssertionError('NO_NETWORK_TEST'));p.start();self.addCleanup(p.stop)
        self.index=MedicalKnowledgeIndex(corpus())
    def test_fifteen_and_reproducible_common_pack(self):
        self.assertEqual(len(self.index.cards),15)
        self.assertEqual(self.index.common(),MedicalKnowledgeIndex(corpus()).common())
        self.assertNotIn('evidence_indices',self.index.common())
    def test_search_diabetes_and_no_match(self):
        r=self.index.search({'chief_complaint':'diabetes A1c glucose'})
        self.assertEqual(r['status'],'AVAILABLE');self.assertTrue(any('Diabetes' in x['card']['disease'] for x in r['results']))
        self.assertEqual(self.index.search({'chief_complaint':'zzzzzzzzq'})['status'],'KNOWLEDGE_UNAVAILABLE')
    def test_reject_gold_nested_and_prediction_not_used(self):
        with self.assertRaises(ValueError):self.index.search({'history':[{'gold':'TEST_ONLY'}]})
        self.assertEqual(public_query({'chief_complaint':'TEST_ONLY','working_assessment':'ALZHEIMER'}).strip(),'TEST_ONLY')
    def test_untrusted_and_forged_summary_blocked(self):
        c=corpus();c['sources']['S01']['url']='https://untrusted.invalid/source'
        with self.assertRaises(ValueError):MedicalKnowledgeIndex(c)
        c=corpus();c['cards'][0]['common_note']='Correct answer for patient TEST_ONLY'
        with self.assertRaises(ValueError):MedicalKnowledgeIndex(c)
    def test_missing_or_unverified_blocked(self):
        with self.assertRaises(ValueError):MedicalKnowledgeIndex({})
        c=corpus();c['cards'][0]['status']='REVIEWED'
        with self.assertRaises(ValueError):MedicalKnowledgeIndex(c)
    def test_temporal_display_never_changes_source_or_judgments(self):
        o={'longitudinal_memory':{'records':[{'text':'TEST_ONLY','visit_started_at':'2000-01-01','status_as_of_visit':'historical'}]}}
        before=deepcopy(o);v=historical_labels(o)
        self.assertEqual(o,before);self.assertEqual(v['historical_records'][0]['scope'],'HISTORICAL_NOT_CURRENT_EVIDENCE')
        self.assertNotIn('relevant',str(v))

class EvidenceSetContracts(unittest.TestCase):
    def fixture(self,action='diagnose'):
        props={'action':{'enum':[action]},'evidence_indices':{'type':'array','uniqueItems':True,'maxItems':4,'items':{'type':'integer','minimum':0,'maximum':4}}}
        schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
        return {'evidence':['TEST_ONLY']*5,'instruction':''},schema
    def test_valid_decode_and_original_schema(self):
        p,s=self.fixture();q,w=encode(p,s);v={'action':'diagnose','evidence_selection':{'4':True,'0':True}}
        Draft202012Validator(w).validate(v);self.assertEqual(decode(v,s)['evidence_indices'],[0,4])
        self.assertIn('evidence_indices',s['properties']);self.assertNotIn('evidence_selection',s['properties'])
    def test_duplicates_raw_keys_never_repaired(self):
        with self.assertRaises(ValueError):_strict_object('{"action":"diagnose","evidence_selection":{"4":true,"4":true}}')
        _,s=self.fixture()
        with self.assertRaises(ValueError):decode({'action':'diagnose','evidence_indices':[4,4,4,4]},s)
    def test_bad_indices_false_and_five_rejected(self):
        p,s=self.fixture();_,w=encode(p,s)
        for selection in ({'9':True},{'1':False},{str(i):True for i in range(5)}):
            with self.assertRaises(ValidationError):Draft202012Validator(w).validate({'action':'diagnose','evidence_selection':selection})
    def test_non_diagnostic_empty_only(self):
        p,s=self.fixture('request_test');_,w=encode(p,s)
        Draft202012Validator(w).validate({'action':'request_test','evidence_selection':{}})
        with self.assertRaises(ValidationError):Draft202012Validator(w).validate({'action':'request_test','evidence_selection':{'0':True}})
    def test_zero_evidence_cannot_invent_index(self):
        p,s=self.fixture();p['evidence']=[];_,w=encode(p,s)
        with self.assertRaises(ValidationError):Draft202012Validator(w).validate({'action':'diagnose','evidence_selection':{'0':True}})

class RuleContracts(unittest.TestCase):
    def fixture(self,value='7',unit='%',code='4548-4'):
        ev=json.dumps({'value':value,'units':unit})
        row={'kind':'Test','target':'TEST_ONLY','description':'TEST_ONLY','code':code,'unit':unit,'evidence':ev,'available_minute':5}
        text='TEST_ONLY: '+ev
        return {'elapsed_minutes':5,'objective_findings':[text],'history':[]}, {'rows':[row]}
    def test_current_a1c_signal_is_not_diagnosis(self):
        o,b=self.fixture();r=select_rule(o,b,['endocrinology','nephrology'])
        self.assertEqual(r['selected_specialty'],'endocrinology');self.assertFalse(r['clinical_routing_validated'])
    def test_no_signal_normal_values_and_history_alone_abstain(self):
        o,b=self.fixture('5');o['longitudinal_memory']={'records':[{'text':'diabetes memory loss chest pain'}]}
        self.assertEqual(select_rule(o,b,['endocrinology','neurology','cardiology'])['decision_status'],'abstained')
    def test_conflict_abstains(self):
        o,b=self.fixture();o['chief_complaint']='chest pain'
        self.assertEqual(select_rule(o,b,['endocrinology','cardiology'])['decision_status'],'abstained')
    def test_negation_and_family_history_not_current_signal(self):
        for text in ('no chest pain','denies memory loss','family history of wheezing'):
            r=select_rule({'chief_complaint':text,'elapsed_minutes':0,'objective_findings':[],'history':[]},{'rows':[]},['cardiology','neurology','pulmonology'])
            self.assertEqual(r['decision_status'],'abstained')
    def test_unreceived_and_future_cannot_trigger(self):
        o,b=self.fixture();o['objective_findings']=[]
        self.assertEqual(select_rule(o,b,['endocrinology'])['decision_status'],'abstained')
        o,b=self.fixture();o['elapsed_minutes']=0
        with self.assertRaises(ValueError):select_rule(o,b,['endocrinology'])
    def test_units_must_match_and_unregistered_candidate_not_selected(self):
        o,b=self.fixture('70','mg/dL')
        self.assertEqual(select_rule(o,b,['endocrinology'])['decision_status'],'abstained')
        o,b=self.fixture();self.assertIsNone(select_rule(o,b,['cardiology'])['selected_specialty'])
    def test_gold_cannot_drive_router(self):
        o,b=self.fixture();o['gold']={'department':'TEST_ONLY'}
        with self.assertRaises(ValueError):select_rule(o,b,['endocrinology'])

if __name__=='__main__':unittest.main()
