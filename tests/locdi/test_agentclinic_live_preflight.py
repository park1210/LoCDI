from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT
from app.locdi.agentclinic_live_preflight import proposal,preflight
from app.locdi.agentclinic_doctor_client import create_live_client,LiveDoctorClient
from app.locdi.agentclinic_turn_runtime import initial_observation
from app.locdi.agentclinic_turn_runtime import TurnOptions
from app.locdi.agentclinic_doctor_runtime import run_doctor_scenario
from agentclinic_offline_support import bundle
from agentclinic_json_support import fixture_approval


class AgentClinicPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_llm_setup as fixtures
        cls.fixtures=fixtures
        fixtures.LLMSetupContracts.setUpClass()
        cls.definition=proposal();cls.definition.update(scenario_ordinals=[1],total_generation_budget=20)
        cls.definition['legacy_config']=deepcopy(fixtures.LLMSetupContracts.config)

    @classmethod
    def tearDownClass(cls):
        cls.fixtures.LLMSetupContracts.tearDownClass()

    def test_no_approval_or_bool_approval_never_constructs_backend(self):
        with patch('app.locdi.ollama_adapter._backend',side_effect=AssertionError('TEST_ONLY_NO_BACKEND')) as backend:
            r=preflight(self.definition)
            self.assertFalse(r['live_execution_ready']);self.assertEqual(r['observed_server_digest'],'UNKNOWN_NOT_CHECKED')
            for kwargs in ({},{'execute':True,'allow_llm':True},{'execute':True,'allow_llm':True,'authority':'APPROVED'}):
                with self.assertRaises(ValueError):create_live_client(self.definition,**kwargs)
            with self.assertRaises(ValueError):LiveDoctorClient(self.definition,None)
        backend.assert_not_called()

    def test_source_prompt_budget_model_mismatch_blocks_before_backend(self):
        cases=[('source_revision','bad'),('source_sha256','0'*64),('prompt_sha256','0'*64),
               ('scenario_ordinals',[108]),('retry_budget',1),('total_generation_budget',999)]
        with patch('app.locdi.ollama_adapter._backend') as backend:
            for key,value in cases:
                d=deepcopy(self.definition);d[key]=value
                self.assertTrue(preflight(d)['static_blockers'])
            d=deepcopy(self.definition);d['legacy_config']['model_identifier']='sha256:'+'0'*64
            self.assertTrue(preflight(d)['static_blockers'])
        backend.assert_not_called()

    def test_pinned_approval_plan_and_context_review_cannot_be_status_only(self):
        plan=preflight(self.definition)['plan']
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory:
            authority=fixture_approval(directory,plan);self.assertFalse(authority.verify(plan)['clinical_reference_approved'])
            changed=deepcopy(plan);changed['definition']['max_turns_per_case']=1
            with self.assertRaises(ValueError):authority.verify(changed)
            Path(authority.path).write_text('{"status":"APPROVED"}',encoding='utf8')
            with self.assertRaises(ValueError):authority.verify(plan)

    def test_production_local_adapter_identity_and_role_call_with_fake_sdk_only(self):
        plan=preflight(self.definition)['plan'];case=self.fixtures.LLMSetupContracts()
        reply=json.dumps({'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_TRANSPORT_RESULT'})
        _,calls,options,http_options,sdk,http=case.backend(failure=reply)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory,sdk,http:
            client=create_live_client(self.definition,execute=True,allow_llm=True,authority=fixture_approval(directory,plan))
            obs=initial_observation(bundle().scenarios[0]);obs['remaining_requests']=10
            action=client.choose_action(obs)
            self.assertEqual(action['action'],'DIAGNOSE')
            self.assertEqual(client.transport.identity_check_count,1)
            self.assertEqual(client.transport.request_count,1)
            self.assertEqual(len(calls),1);self.assertEqual(options[0]['max_retries'],0)
            self.assertFalse(http_options[0]['follow_redirects'])
            client.close()

    def test_wrong_observed_digest_causes_zero_fake_generation(self):
        plan=preflight(self.definition)['plan'];case=self.fixtures.LLMSetupContracts()
        _,calls,_,_,sdk,http=case.backend(mismatch=True)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory,sdk,http:
            client=create_live_client(self.definition,execute=True,allow_llm=True,authority=fixture_approval(directory,plan))
            obs=initial_observation(bundle().scenarios[0]);obs['remaining_requests']=10
            with self.assertRaises(Exception):client.choose_action(obs)
            self.assertEqual(calls,[]);self.assertEqual(client.transport.request_count,0)
            client.close()

    def test_scenario_binding_and_prompt_file_change_rejected(self):
        plan=preflight(self.definition)['plan'];case=self.fixtures.LLMSetupContracts()
        _,calls,_,_,sdk,http=case.backend()
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory,sdk,http:
            client=create_live_client(self.definition,execute=True,allow_llm=True,authority=fixture_approval(directory,plan))
            obs=initial_observation(bundle().scenarios[0]);obs['remaining_requests']=10;obs['objective']='TEST_ONLY_CHANGED'
            with self.assertRaises(ValueError):client.choose_action(obs)
            with patch('app.locdi.agentclinic_doctor_client.PROMPT_SHA256','0'*64):
                self.assertTrue(preflight(self.definition)['static_blockers'])
            self.assertEqual(calls,[]);client.close()

    def test_live_case_cannot_exceed_pinned_turn_or_request_limit(self):
        plan=preflight(self.definition)['plan'];case=self.fixtures.LLMSetupContracts()
        _,calls,_,_,sdk,http=case.backend()
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory,sdk,http:
            client=create_live_client(self.definition,execute=True,allow_llm=True,authority=fixture_approval(directory,plan))
            for options in (TurnOptions(11,10),TurnOptions(10,11)):
                with self.assertRaises(ValueError):run_doctor_scenario(bundle().scenarios[0],client,options=options)
            self.assertEqual(calls,[]);client.close()
