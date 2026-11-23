"""TEST_ONLY terminal-request scope; no model, Source or Utility fitting."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from app.locdi.terminal_only_recovery import (
    TerminalOnlyClient,TerminalOnlyInformationPolicy,terminal_request,terminal_only_scope)
from app.locdi.m5_common_source import NativeM5InformationPolicy
from app.locdi.llm_setup import digest
import app.locdi.common_menu_backend as legacy


class TerminalContracts(unittest.TestCase):
    def setup_policy(self,callback,observation=None):
        observed={'elapsed_minutes':7,'objective_findings':['TEST_ONLY']}
        p=object.__new__(TerminalOnlyInformationPolicy);p.requested=False
        p.terminal_candidate=lambda:{'observation_fingerprint':digest(observation or observed),'terminal_request':callback}
        runtime=SimpleNamespace(voi_state={'history':[{'minute':7,'selected_action':'diagnose','clinical_support':False}]})
        runtime.to_summary=lambda:deepcopy(runtime.voi_state)
        return p,runtime,observed
    def test_stop_makes_exactly_one_terminal_request(self):
        calls=[];candidate={'action':'diagnose','certainty':'unresolved','target':'Unresolved'}
        p,r,o=self.setup_policy(lambda obs:calls.append(obs) or candidate)
        with patch.object(NativeM5InformationPolicy,'propose',return_value=candidate):
            self.assertEqual(p.propose(r,o),candidate)
            with self.assertRaises(ValueError):p.propose(r,o)
        self.assertEqual(calls,[o]);self.assertEqual(r.voi_state['history'][-1]['terminal_extra_requests'],1)
    def test_continue_does_not_call_terminal(self):
        p,r,o=self.setup_policy(lambda _:self.fail('UNEXPECTED_TERMINAL'))
        action={'action':'request_test','target':'TEST_ONLY'}
        with patch.object(NativeM5InformationPolicy,'propose',return_value=action):self.assertEqual(p.propose(r,o),action)
        self.assertFalse(p.requested)
    def test_stale_snapshot_fails_before_callback(self):
        p,r,o=self.setup_policy(lambda _:self.fail('STALE_REQUEST'),{'elapsed_minutes':6})
        with patch.object(NativeM5InformationPolicy,'propose',return_value={'action':'diagnose'}):
            with self.assertRaisesRegex(ValueError,'CURRENT_SINGLE'):p.propose(r,o)
    def test_runtime_mutation_during_terminal_fails(self):
        p,r,o=self.setup_policy(lambda _:None)
        def corrupt(_):r.voi_state['history'][0]['minute']=8;return {'action':'diagnose'}
        p.terminal_candidate=lambda:{'observation_fingerprint':digest(o),'terminal_request':corrupt}
        with patch.object(NativeM5InformationPolicy,'propose',return_value={'action':'diagnose'}):
            with self.assertRaisesRegex(ValueError,'RUNTIME_OR_TIME'):p.propose(r,o)
    def test_scoped_legacy_rollback(self):
        old=(legacy.CommonMenuNativeClient,legacy.StopThenLLMInformationPolicy)
        with terminal_only_scope():self.assertIs(legacy.CommonMenuNativeClient,TerminalOnlyClient)
        self.assertEqual(old,(legacy.CommonMenuNativeClient,legacy.StopThenLLMInformationPolicy))
    def test_duplicate_terminal_client_request_is_blocked(self):
        client=object.__new__(TerminalOnlyClient);client.terminal_only_calls=1;client._current_call=None
        with self.assertRaisesRegex(ValueError,'ONE_TERMINAL'):client.request_terminal({})
    def test_no_terminal_permission_is_invented(self):
        s={'properties':{'action':{'enum':['diagnose']}}}
        with self.assertRaisesRegex(ValueError,'PERMISSION'):terminal_request({'allowed_actions':['request_test']},s)
    def test_terminal_does_not_add_supported_certainty(self):
        branch={'properties':{'action':{'enum':['diagnose']},'certainty':{'enum':['unresolved']},'target':{'enum':['Unresolved']}}}
        p={'allowed_actions':['diagnose'],'observation':{'history':[]},'evidence':[],'supported_diagnosis_eligible_indices':[]}
        new,s=terminal_request(p,branch)
        self.assertEqual(s,branch);self.assertEqual(new['evidence'],[])
        self.assertEqual(s['properties']['certainty']['enum'],['unresolved'])


if __name__=='__main__':unittest.main()
