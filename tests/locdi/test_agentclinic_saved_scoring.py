"""TEST_ONLY saved-score authorization, denominator and visibility contracts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha
from app.locdi.agentclinic_scenario_adapter import adapt_test_record
from app.locdi.agentclinic_benchmark_scoring import normalize,evaluate_test_only
from scripts.analyze_agentclinic_saved_pilot import scoring_plan,PinnedSavedScoringApproval,score_saved,trace_analysis
from agentclinic_offline_support import fixture


class SavedScoringTests(unittest.TestCase):
    def pair(self,i=1,prediction='TEST_ONLY_PRIVATE_KEY'):
        s,k=adapt_test_record(fixture(),i)
        e={'scenario_ref':s.scenario_id,'source':deepcopy(s.source),'trace':[],
           'result':{'status':'completed','completed':True,'predicted_diagnosis':prediction,
                     'diagnosis_unresolved':not bool(prediction),'turn_count':1,'action_count':1}}
        return s,k,e

    def authority(self,directory,sc,ep,keys,**changes):
        r={'schema_version':'agentclinic.saved_scoring_approval.v1','scope':'OFFLINE_AGENTCLINIC_SAVED_PILOT_LEXICAL_ONLY',
           'plan_fingerprint':digest(scoring_plan(sc,ep,keys)),
           'approval_source':'CURRENT_USER_C3_14_ANALYSIS_REQUEST','approved_by':'USER_REQUESTING_OFFLINE_ANALYSIS',
           'permitted_metric':'EXACT_MATCH_LEXICAL_ONLY','clinical_equivalence_approved':False,
           'alias_approved':False,'generation_budget':0,'grading_budget':0,'request_evidence':'TEST_ONLY'}
        r.update(changes);p=Path(directory)/'TEST_ONLY_approval.json';p.write_text(json.dumps(r),encoding='utf8')
        return PinnedSavedScoringApproval(p,sha(p))

    def score(self,sc,ep,keys,**changes):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            return score_saved(sc,ep,keys,authority=self.authority(d,sc,ep,keys,**changes))

    def test_no_permission_or_bool_never_scores(self):
        s,k,e=self.pair()
        for a in (None,True,'APPROVED'):
            with self.assertRaises(ValueError):score_saved([s],[e],[k],authority=a)

    def test_scope_clinical_alias_or_generation_expansion_denied(self):
        s,k,e=self.pair()
        for changes in ({'scope':'LIVE'},{'clinical_equivalence_approved':True},{'alias_approved':True},{'generation_budget':1}):
            with self.assertRaises(ValueError):self.score([s],[e],[k],**changes)

    def test_episode_fingerprint_change_denied(self):
        s,k,e=self.pair()
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            a=self.authority(d,[s],[e],[k]);e['result']['predicted_diagnosis']='OTHER_TEST_ONLY'
            with self.assertRaises(ValueError):score_saved([s],[e],[k],authority=a)

    def test_exact_rules_unchanged_not_synonyms(self):
        s,k,e=self.pair(prediction=' \nTEST_ONLY_PRIVATE_KEY.  ')
        self.assertEqual(self.score([s],[e],[k])['metrics']['exact_match_accuracy']['value'],1)
        self.assertEqual(normalize('e\u0301.'),normalize('\u00e9'))
        self.assertNotEqual(normalize('x..'),normalize('x'))
        e['result']['predicted_diagnosis']='TEST_ONLY_PRIVATE'
        r=self.score([s],[e],[k]);self.assertEqual(r['metrics']['review_required_count'],1)
        self.assertEqual(r['clinical_equivalence'],'CLINICAL_EQUIVALENCE_NOT_VERIFIED')

    def test_missing_episode_keeps_fixed_denominator_not_clinical_wrong(self):
        s,k,e=self.pair();s2,k2,_=self.pair(2)
        r=self.score([s,s2],[e],[k,k2])
        self.assertEqual(r['metrics']['exact_match_accuracy']['denominator'],2)
        self.assertEqual(r['metrics']['exact_match_accuracy']['value'],0.5)
        self.assertEqual(r['private_rows'][1]['state'],'not_executed')
        self.assertIsNone(r['private_rows'][1]['exact_match'])
        self.assertEqual(r['metrics']['failed_rate']['numerator'],0)

    def test_duplicate_wrong_source_and_missing_keys_rejected(self):
        s,k,e=self.pair()
        for episodes,keys in (([e,e],[k]),([e],[])):
            with self.assertRaises(ValueError):self.score([s],episodes,keys)
        e['source']['record_fingerprint']='TEST_ONLY_WRONG'
        with self.assertRaises(ValueError):self.score([s],[e],[k])

    def test_unresolved_failed_incomplete_are_distinct(self):
        for status,pred,state in [('completed','','diagnosis_unresolved'),('failed',None,'failed'),('incomplete',None,'incomplete')]:
            s,k,e=self.pair(prediction=pred);e['result']['status']=status
            self.assertEqual(self.score([s],[e],[k])['private_rows'][0]['state'],state)

    def test_group_leaf_repeat_and_future_observation(self):
        s,_,_=self.pair();t=s.exam_targets[0]
        obs={'request_id':'R0001','value':t.value,'observation_ref':'O0001'}
        event={'action':'EXAMINE','target':t.target_id,'observation':obs,'repeat_of':None,
               'status':'EXECUTED','validation':'VALID','turn_index':1}
        repeat=dict(event,repeat_of='R0001',turn_index=2)
        before=[{'request_id':'Request0001','payload':{'observations':[],'remaining_requests':10},'counters':{'doctor_request_attempts':0}},
                {'request_id':'Request0002','payload':{'observations':[obs],'remaining_requests':9},'counters':{'doctor_request_attempts':1}}]
        r=trace_analysis(s,{'trace':[event,repeat]},before)
        self.assertEqual(r[0]['node_kind'],t.node_kind);self.assertTrue(r[1]['repeated_target'])
        self.assertTrue(r[1]['prior_observation_provided']);self.assertFalse(r[1]['new_observation'])
        before[0]['payload']['observations']=[obs]
        with self.assertRaises(ValueError):trace_analysis(s,{'trace':[event,repeat]},before)

    def test_old_real_guard_still_blocked(self):
        s,k,e=self.pair();s.source['test_only']=False
        self.assertEqual(evaluate_test_only([s],[e],[k])['status'],'NOT_EVALUATED')

    def test_zero_request_denominator_not_applicable(self):
        s,k,_=self.pair();r=self.score([s],[],[k])
        self.assertEqual(r['metrics']['repeated_action_rate']['status'],'NOT_APPLICABLE')
        self.assertIsNone(r['metrics']['turns']['mean'])
