"""TEST_ONLY exact-ID availability and real Source replay contracts."""
import json
import unittest
from copy import deepcopy
from app.locdi.agentclinic_state_actions import ActionAvailability
from app.locdi.agentclinic_state_runtime import run_state_episode
from app.locdi.agentclinic_doctor_client import FakeDoctorClient,OfflineJsonTransport
from app.locdi.agentclinic_scenario_adapter import adapt_test_record,action_catalog
from agentclinic_offline_support import fixture
from unittest.mock import patch

class Script(OfflineJsonTransport):
    def __init__(self,actions):self.actions=iter(actions);self.payloads=[]
    def chat_json(self,system,user,model=None,*,schema=None):
        self.payloads.append(json.loads(user));return json.dumps(next(self.actions))

class StateActionsTests(unittest.TestCase):
    def setUp(self):
        self.s,_=adapt_test_record(fixture());self.catalog=action_catalog(self.s)
        self.target=self.catalog['EXAMINE'][0]['target']
    def run_actions(self,actions,enabled=True,**kw):
        t=Script(actions);e,p=run_state_episode(self.s,FakeDoctorClient(t),enabled=enabled,**kw)
        return e,p,t
    def test_exact_repeat_rejected_without_execution_and_return_to_diagnosis(self):
        a={'action':'EXAMINE','target':self.target}
        e,p,t=self.run_actions([a,a,{'action':'DIAGNOSE','predicted_diagnosis':''}])
        self.assertEqual(e['result']['status'],'completed');self.assertEqual(len(e['trace']),2)
        self.assertEqual(p['events'][1]['rejection_reason'],'ALREADY_OBSERVED')
        self.assertEqual(p['events'][1]['new_observation_count'],0)
        self.assertEqual(p['decision_turn_count'],3);self.assertEqual(e['doctor_request_attempts'],3)
        self.assertNotIn(self.target,[x['target'] for x in t.payloads[1]['catalog']['EXAMINE']])
    def test_baseline_preserves_repeated_source_event(self):
        a={'action':'EXAMINE','target':self.target}
        e,p,_=self.run_actions([a,a,{'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY'}],False)
        self.assertEqual(len(e['trace']),3);self.assertTrue(p['events'][1]['repeated_proposal'])
        self.assertIsNotNone(e['trace'][1]['repeat_of'])
    def test_rejections_consume_ten_decision_budget(self):
        a={'action':'EXAMINE','target':self.target}
        e,p,_=self.run_actions([a]*10)
        self.assertEqual(e['result']['status'],'incomplete');self.assertEqual(len(e['trace']),1)
        self.assertEqual(p['decision_turn_count'],10);self.assertEqual(e['doctor_request_attempts'],10)
    def test_failed_and_unavailable_not_consumed(self):
        p=ActionAvailability('TEST_ONLY',self.catalog,enabled=True)
        for status in ('FAILED','UNAVAILABLE'):
            p.observe({'action':'EXAMINE','target':self.target,'status':status,'validation':'VALID','observation_ref':'TEST_ONLY','observation':{}})
        self.assertEqual(p.snapshot(),[])
    def test_group_does_not_consume_descendants_and_scope_is_local(self):
        p=ActionAvailability('TEST_ONLY',self.catalog,enabled=True)
        target=self.catalog['TEST'][0]['target']
        p.observe({'action':'TEST','target':target,'status':'EXECUTED','validation':'VALID','observation_ref':'TEST_ONLY','observation':{}})
        self.assertEqual(len(p.snapshot()),1)
        self.assertEqual(len(p.available_catalog()['TEST']),len(self.catalog['TEST'])-1)
        q=ActionAvailability('OTHER',self.catalog,enabled=True);self.assertEqual(q.snapshot(),[])
    def test_malformed_not_repaired_or_retargeted(self):
        e,p,_=self.run_actions([{'action':'EXAMINE','target':'nonexistent'}])
        self.assertEqual(e['result']['status'],'failed');self.assertEqual(e['trace'],[])
    def test_context_failure_blocks_source_execution(self):
        e,p,_=self.run_actions([{'action':'EXAMINE','target':self.target}],context_observer=lambda:{'status':'INVALID'})
        self.assertEqual(e['result']['status'],'failed');self.assertEqual(e['trace'],[])
    def test_checkpoint_failure_prevents_call(self):
        t=Script([{'action':'DIAGNOSE','predicted_diagnosis':''}])
        def fail(phase,data):
            if phase=='before':raise OSError('TEST_ONLY')
        e,p=run_state_episode(self.s,FakeDoctorClient(t),enabled=True,checkpoint=fail)
        self.assertEqual(t.payloads,[]);self.assertEqual(e['result']['status'],'failed')
    def test_private_key_absent_from_requests_and_sidecar(self):
        e,p,t=self.run_actions([{'action':'DIAGNOSE','predicted_diagnosis':''}])
        self.assertNotIn('TEST_ONLY_PRIVATE_KEY',json.dumps(t.payloads)+json.dumps(p))
        self.assertTrue(e['result']['diagnosis_unresolved'])
    def test_unchanged_prompt_required(self):
        with self.assertRaises(ValueError):run_state_episode(self.s,FakeDoctorClient(Script([]),prompt_profile='B'),enabled=True)
    def test_no_study_authority_no_backend(self):
        from app.locdi.agentclinic_state_pilot import run_live
        with patch('app.locdi.agentclinic_state_pilot.create_live_client') as factory:
            for condition in ('A','C'):
                with self.assertRaises(ValueError):run_live(condition,execute=True,allow_llm=True)
            factory.assert_not_called()
    def test_empty_information_catalog_allows_unresolved(self):
        p=ActionAvailability('TEST_ONLY',{'ASK':[],'EXAMINE':[],'TEST':[]},enabled=True)
        self.assertEqual(p.classify({'action':'DIAGNOSE','predicted_diagnosis':''}),(True,None))
    def test_sidecar_links_actual_episode_and_order(self):
        from app.locdi.llm_setup import digest
        e,p,_=self.run_actions([{'action':'EXAMINE','target':self.target},{'action':'DIAGNOSE','predicted_diagnosis':''}])
        self.assertEqual(p['episode_fingerprint'],digest(e))
        self.assertEqual([r['turn_index'] for r in p['events']],[1,2])
        self.assertEqual([r['source_event_index'] for r in p['events']],[0,1])
