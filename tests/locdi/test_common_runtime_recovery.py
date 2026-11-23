"""Nonclinical, offline recovery-contract tests. No inference or fitting."""
from copy import deepcopy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from app.locdi.common_runtime_recovery import released_measurement_view,StopThenLLMInformationPolicy
from app.locdi.m5_common_source import NativeM5InformationPolicy
from app.locdi.llm_setup import digest

class RecoveryContracts(unittest.TestCase):
    def fixture(self,minute=6):
        r={'kind':'Test','target':'TEST_ID','evidence':'{"value":"7.0","units":"TEST_UNIT"}',
           'available_minute':5,'description':'TEST_ONLY_MEASUREMENT','code':'TEST_CODE','unit':'TEST_UNIT'}
        text=r['target']+': '+r['evidence']
        p={'observation':{'elapsed_minutes':minute,'objective_findings':[text],'history':[]},
           'evidence':[text],'evidence_records':[{'text':text,'supports_finalization':True}],
           'supported_diagnosis_eligible_indices':[0],'instruction':'TEST_ONLY'}
        return p,{'rows':[r]}

    def test_exact_released_names_value_units_and_indices_preserved(self):
        p,b=self.fixture();saved=deepcopy((p,b));q=released_measurement_view(p,b)
        self.assertEqual((p,b),saved);self.assertEqual(q['observation'],p['observation'])
        self.assertEqual(q['evidence'],p['evidence']);self.assertEqual(q['evidence_records'],p['evidence_records'])
        self.assertEqual(q['supported_diagnosis_eligible_indices'],[0]);self.assertEqual(q['released_measurements'][0]['value'],'7.0')
        self.assertEqual(q['released_measurements'][0]['evidence_index'],0)

    def test_no_unreceived_hidden_value_in_display(self):
        p,b=self.fixture();p['observation']['objective_findings']=[];p['evidence']=[]
        self.assertEqual(released_measurement_view(p,b)['released_measurements'],[])

    def test_future_or_value_mismatch_not_promoted(self):
        p,b=self.fixture(1)
        with self.assertRaisesRegex(ValueError,'NOT_YET'):released_measurement_view(p,b)
        p,b=self.fixture();p['evidence'][0]+='WRONG'
        self.assertEqual(released_measurement_view(p,b)['released_measurements'],[])

    def test_ask_stays_patient_report_and_needs_question_history(self):
        p,b=self.fixture();r=b['rows'][0];r['kind']='Ask';p['evidence']=[r['evidence']];p['observation']['objective_findings']=[]
        self.assertEqual(released_measurement_view(p,b)['released_measurements'],[])
        p['observation']['history']=[{'question':r['target'],'answer':r['evidence'],'answer_status':'known'}]
        self.assertEqual(released_measurement_view(p,b)['released_measurements'][0]['kind'],'Ask')

    def wrapper(self,obs,candidate):
        policy=object.__new__(StopThenLLMInformationPolicy)
        policy.terminal_candidate=lambda:{'observation_fingerprint':digest(obs),'decision':candidate}
        rt=SimpleNamespace(voi_state={'history':[{'clinical_support':False,'selected_action':'diagnose'}]})
        return policy,rt

    def test_stop_uses_current_llm_without_inventing_m5_support(self):
        obs={'TEST_ONLY':1};candidate={'action':'diagnose','certainty':'supported','target':'TEST_LABEL','evidence_indices':[0]}
        policy,rt=self.wrapper(obs,candidate)
        with patch.object(NativeM5InformationPolicy,'propose',return_value={'action':'diagnose','certainty':'unresolved'}):
            out=policy.propose(rt,obs)
        self.assertEqual(out,candidate);self.assertIsNot(out,candidate)
        self.assertFalse(rt.voi_state['history'][-1]['m5_clinical_support'])
        self.assertIn('PENDING_CORE_REVALIDATION',rt.voi_state['history'][-1]['terminal_origin'])

    def test_information_actions_unchanged(self):
        policy,rt=self.wrapper({},{});original={'action':'request_test','target':'TEST'}
        with patch.object(NativeM5InformationPolicy,'propose',return_value=original):self.assertEqual(policy.propose(rt,{}),original)

    def test_unresolved_and_nonterminal_never_coerced(self):
        for c in ({'action':'request_test'},{'action':'diagnose','certainty':'unresolved','evidence_indices':[]}):
            policy,rt=self.wrapper({},c);original={'action':'diagnose','certainty':'unresolved'}
            with patch.object(NativeM5InformationPolicy,'propose',return_value=original):
                out=policy.propose(rt,{})
            self.assertEqual(out.get('certainty'),'unresolved')

    def test_stale_snapshot_blocks(self):
        policy,rt=self.wrapper({'TEST_ONLY':'old'},{'action':'diagnose'})
        with patch.object(NativeM5InformationPolicy,'propose',return_value={'action':'diagnose'}):
            with self.assertRaisesRegex(ValueError,'SNAPSHOT'):policy.propose(rt,{'TEST_ONLY':'new'})

if __name__=='__main__':unittest.main()
