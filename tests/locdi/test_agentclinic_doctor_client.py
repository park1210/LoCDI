from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT
from app.locdi.agentclinic_doctor_client import FakeDoctorClient,build_request
from app.locdi.agentclinic_doctor_runtime import run_doctor_scenario
from app.locdi.agentclinic_doctor_exporter import export_doctor_episode,public_doctor_episode
from app.locdi.agentclinic_scenario_adapter import adapt_test_record
from app.locdi.agentclinic_turn_runtime import initial_observation,TurnOptions
from agentclinic_offline_support import bundle,fixture,e2e_actions
from agentclinic_json_support import JsonScript
from scripts.audit_agentclinic_dataset import lexical_hit


class AgentClinicDoctorTests(unittest.TestCase):
    def test_all_107_initial_prompts_exclude_private_answers_and_stores(self):
        for s,key in zip(bundle().scenarios,bundle().scoring_keys):
            obs=initial_observation(s);obs['remaining_requests']=20
            system,payload,schema=build_request(obs)
            self.assertFalse(lexical_hit(key.correct_diagnosis,system+json.dumps(payload,ensure_ascii=False)))
            self.assertNotIn('Correct_Diagnosis',json.dumps(payload))
            self.assertEqual(payload['observations'],[])

    def test_actual_107_json_source_runtime_and_prompt_updates(self):
        for s in bundle().scenarios:
            actions=e2e_actions(s)
            if not s.test_targets:
                actions=[a for a in actions if a['action']!='TEST']
            transport=JsonScript(actions);doctor=FakeDoctorClient(transport)
            e=run_doctor_scenario(s,doctor)
            self.assertTrue(e['result']['completed'])
            self.assertEqual(len(transport.seen),len(actions))
            self.assertEqual(transport.seen[1]['payload']['observations'][0]['value'],s.patient_facts['History'])
            self.assertEqual(len(transport.seen[1]['payload']['history']),1)
            self.assertEqual(e['doctor_request_attempts'],len(actions))

    def test_unknown_store_cannot_be_passed_as_prompt_metadata(self):
        s,_=adapt_test_record(fixture());obs=initial_observation(s);obs['remaining_requests']=3
        obs['Correct_Diagnosis']='TEST_ONLY_PRIVATE'
        with self.assertRaises(ValueError):build_request(obs)

    def test_json_duplicate_keys_unknown_target_and_action_fail_without_retry(self):
        s,_=adapt_test_record(fixture())
        for invalid in ('not-json','{"action":"ASK","action":"TEST","target":"x"}',
                         {'action':'TEST','target':'absent'}, {'action':'CONSULT','target':'x'}):
            transport=JsonScript([invalid]);doctor=FakeDoctorClient(transport)
            e=run_doctor_scenario(s,doctor)
            self.assertEqual(e['result']['status'],'failed')
            self.assertEqual(len(transport.seen),1)
            self.assertEqual(len(doctor.audit),1)
            self.assertFalse(doctor.audit[0]['fallback_used'])

    def test_global_budget_and_per_case_incomplete_are_separate(self):
        s,_=adapt_test_record(fixture())
        doctor=FakeDoctorClient(JsonScript([{'action':'ASK','target':'history'}]*4),request_budget=1)
        e=run_doctor_scenario(s,doctor)
        self.assertEqual(e['result']['status'],'failed')
        self.assertEqual(doctor.request_count,1)
        e=run_doctor_scenario(s,FakeDoctorClient(JsonScript([{'action':'ASK','target':'history'}]*3)),options=TurnOptions(2,2))
        self.assertEqual(e['result']['status'],'incomplete')

    def test_unresolved_invalid_prediction_and_timeout(self):
        s,_=adapt_test_record(fixture())
        e=run_doctor_scenario(s,FakeDoctorClient(JsonScript([{'action':'DIAGNOSE','predicted_diagnosis':''}])))
        self.assertTrue(e['result']['diagnosis_unresolved'])
        for bad in ({'action':'DIAGNOSE','predicted_diagnosis':4},TimeoutError('TEST_ONLY_SECRET')):
            e=run_doctor_scenario(s,FakeDoctorClient(JsonScript([bad])))
            self.assertEqual(e['result']['status'],'failed')
            self.assertNotIn('TEST_ONLY_SECRET',json.dumps(public_doctor_episode(e)))

    def test_missing_results_not_visible_before_request_and_repeats_are_recorded(self):
        s,_=adapt_test_record(fixture());plan=[{'action':'TEST','target':'test/0002'}]*2
        plan += [{'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_OTHER'}]
        tr=JsonScript(plan);e=run_doctor_scenario(s,FakeDoctorClient(tr))
        self.assertNotIn('TEST_ONLY_A',json.dumps(tr.seen[0]))
        self.assertIn('TEST_ONLY_A',json.dumps(tr.seen[1]))
        self.assertNotIn('TEST_ONLY_B',json.dumps(tr.seen[1]))
        self.assertEqual(e['trace'][1]['repeat_of'],'R0001')

    def test_public_export_no_raw_prediction_audit_or_original_text_and_no_clobber(self):
        s,_=adapt_test_record(fixture());e=run_doctor_scenario(s,FakeDoctorClient(JsonScript(e2e_actions(s))))
        out=public_doctor_episode(e)
        self.assertNotIn('TEST_ONLY_HISTORY',json.dumps(out))
        self.assertNotIn('TEST_ONLY_FINAL',json.dumps(out))
        self.assertNotIn('invalid_response',json.dumps(out))
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/research_evaluation')) as pub, tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as priv:
            export_doctor_episode(e,public_root=pub,private_root=priv,name='TEST_ONLY')
            before=(Path(pub)/'episodes/TEST_ONLY.json').read_bytes()
            with self.assertRaises(FileExistsError):export_doctor_episode(e,public_root=pub,private_root=priv,name='TEST_ONLY')
            self.assertEqual((Path(pub)/'episodes/TEST_ONLY.json').read_bytes(),before)

    def test_failed_public_export_keeps_private_episode(self):
        s,_=adapt_test_record(fixture());e=run_doctor_scenario(s,FakeDoctorClient(JsonScript(e2e_actions(s))))
        from app.locdi.episode_exporter import _write
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/research_evaluation')) as pub, tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as priv:
            def write(document,destination,root,category):
                if str(root)==pub:raise OSError('TEST_ONLY_DISK')
                return _write(document,destination,root,category)
            with patch('app.locdi.agentclinic_doctor_exporter._write',side_effect=write):
                with self.assertRaises(OSError):export_doctor_episode(e,public_root=pub,private_root=priv,name='TEST_ONLY')
            self.assertTrue((Path(priv)/'episodes/TEST_ONLY.json').exists())

    def test_prompt_byte_limit_fails_before_transport(self):
        s,_=adapt_test_record(fixture());tr=JsonScript(e2e_actions(s))
        e=run_doctor_scenario(s,FakeDoctorClient(tr,max_prompt_bytes=256))
        self.assertEqual(e['result']['status'],'failed');self.assertEqual(tr.seen,[])
