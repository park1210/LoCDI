import json
import unittest
from app.locdi.agentclinic_scenario_adapter import adapt_test_record
from app.locdi.agentclinic_turn_runtime import run_scenario, TurnOptions
from agentclinic_offline_support import bundle, fixture, ScriptedDoctor, e2e_actions


class AgentClinicRuntimeTests(unittest.TestCase):
    def test_all_107_e2e_real_source_lookup(self):
        for ordinal, s in enumerate(bundle().scenarios, 1):
            with self.subTest(ordinal=ordinal):
                episode = run_scenario(s, ScriptedDoctor(e2e_actions(s)))
                self.assertTrue(episode['result']['completed'])
                self.assertEqual([e['turn_index'] for e in episode['trace']], [1,2,3,4])
                self.assertEqual(episode['trace'][0]['observation']['value'], s.patient_facts['History'])
                expected = 'UNAVAILABLE' if ordinal in (69,106) else 'EXECUTED'
                self.assertEqual(episode['trace'][2]['status'], expected)
                self.assertEqual(episode['clinical_metrics'], 'NOT_EVALUATED')

    def test_duplicate_question_and_test_link_to_original_observation(self):
        s, _ = adapt_test_record(fixture())
        actions = [{'action':'ASK','target':'history'}]*2 + [{'action':'TEST','target':'test/0002'}]*2
        actions += [{'action':'DIAGNOSE','predicted_diagnosis':''}]
        e = run_scenario(s, ScriptedDoctor(actions))
        for first, repeat in ((0,1),(2,3)):
            self.assertEqual(e['trace'][repeat]['repeat_of'], e['trace'][first]['request_id'])
            self.assertEqual(e['trace'][repeat]['observation_ref'], e['trace'][first]['observation_ref'])
        self.assertTrue(e['result']['diagnosis_unresolved'])

    def test_invalid_and_unavailable_requests_are_not_success(self):
        s, _ = adapt_test_record(fixture())
        c = ScriptedDoctor([{'action':'ASK','target':'free words'}, {'action':'CONSULT','target':'fake'},
                            {'action':'TEST','target':[]}, {'action':'DIAGNOSE','predicted_diagnosis':''}])
        e = run_scenario(s,c)
        self.assertEqual([x['status'] for x in e['trace'][:3]], ['UNAVAILABLE','UNSUPPORTED','INVALID'])
        self.assertEqual(e['result']['invalid_action_count'], 3)
        self.assertEqual(c.seen[1]['last_action']['status'], 'UNAVAILABLE')

    def test_request_and_turn_budgets_bound_all_attempts(self):
        s, _ = adapt_test_record(fixture())
        for opt, reason, count in ((TurnOptions(5,1),'REQUEST_BUDGET_EXHAUSTED',1),
                                    (TurnOptions(2,5),'TURN_BUDGET_EXHAUSTED',2)):
            c = ScriptedDoctor([{'action':'ASK','target':'unknown'}]*5)
            e = run_scenario(s,c,options=opt)
            self.assertEqual(len(c.seen),count)
            self.assertEqual(e['result']['stop_reason'],reason)
            self.assertEqual(e['result']['status'],'incomplete')
        for value in (0, True, 101):
            with self.assertRaises(ValueError):TurnOptions(value,10)

    def test_client_failure_is_exportable_partial_and_sanitized(self):
        s, _ = adapt_test_record(fixture())
        e = run_scenario(s,ScriptedDoctor([{'action':'ASK','target':'history'},RuntimeError('TEST_ONLY_SECRET')]))
        self.assertEqual(e['result']['status'],'failed')
        self.assertEqual(e['result']['turn_count'],2)
        self.assertNotIn('TEST_ONLY_SECRET',json.dumps(e))

    def test_unresolved_and_free_text_finalization_are_distinct_from_accuracy(self):
        s, _ = adapt_test_record(fixture())
        for prediction in ('', 'TEST_ONLY_OPEN_ENDED_NOT_IN_CATALOG'):
            e = run_scenario(s,ScriptedDoctor([{'action':'DIAGNOSE','predicted_diagnosis':prediction}]))
            self.assertTrue(e['result']['completed'])
            self.assertEqual(e['result']['diagnosis_unresolved'],not bool(prediction))
            self.assertEqual(e['result']['predicted_diagnosis'],prediction)
            self.assertEqual(e['grading_calls'],0)

    def test_deterministic_replay_and_no_non_offline_client(self):
        s, _ = adapt_test_record(fixture())
        self.assertEqual(run_scenario(s,ScriptedDoctor(e2e_actions(s))),
                         run_scenario(s,ScriptedDoctor(e2e_actions(s))))
        with self.assertRaises(TypeError):run_scenario(s,object())

    def test_offline_client_cannot_open_network(self):
        import socket
        s,_ = adapt_test_record(fixture())
        class NetworkAttempt(ScriptedDoctor):
            def choose_action(self,observation):
                socket.create_connection(('127.0.0.1',1))
        e = run_scenario(s,NetworkAttempt([]))
        self.assertEqual(e['result']['status'],'failed')
        self.assertEqual(e['result']['error_type'],'PermissionError')
