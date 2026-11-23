"""Offline tests only. Saved cases are read, never regenerated or overwritten."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator,ValidationError
from app.locdi.common_prediction_contract import extract_core_prediction,shared_selection

ROOT=Path(__file__).resolve().parents[2]

class FollowupContract(unittest.TestCase):
    def setUp(self):
        self.no_net=patch.object(socket.socket,'connect',side_effect=AssertionError('OFFLINE_ONLY'))
        self.no_net.start();self.addCleanup(self.no_net.stop)

    def test_41_supported_predictions_equal_preserved_reconstruction(self):
        path=ROOT/'outputs/private/research_evaluation/B34_COMPLETED_RUN_ANALYSIS_20261121_v3/recovered_predictions_and_references.json'
        if not path.is_file():self.skipTest('Private saved-run regression data not distributed')
        rows=json.loads(path.read_text(encoding='utf-8'));supported=0;unresolved=0
        for r in rows:
            with self.subTest(run=r['plan_id'],episode=r['episode_ordinal']):
                folder=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1'/r['plan_id']/('run_'+r['plan_id'])/'requests'/('Episode%04d'%r['episode_ordinal'])
                decisions=json.loads((folder/'actual_core_decisions.json').read_text(encoding='utf-8'))
                pred=extract_core_prediction(decisions,'completed')
                self.assertEqual(pred['diagnosis'],r['recovered_diagnosis'])
                self.assertEqual(pred['department'],r['recovered_department'])
                self.assertEqual(pred['certainty'],r['certainty'])
                supported+=pred['certainty']=='supported';unresolved+=pred['certainty']=='unresolved'
        self.assertEqual(supported,41);self.assertEqual(unresolved,11)

    def test_failed_and_missing_terminal_not_promoted(self):
        t={'action':'er_finalize','certainty':'supported','target':'TEST_ONLY','target_department':'neurology'}
        self.assertIsNone(extract_core_prediction([t],'failed')['diagnosis'])
        self.assertEqual(extract_core_prediction([],'completed')['certainty'],'not_available')

    def test_unresolved_target_never_becomes_diagnosis(self):
        p=extract_core_prediction([{'action':'er_finalize','certainty':'unresolved','target':'Unresolved','target_department':'neurology'}],'completed')
        self.assertIsNone(p['diagnosis']);self.assertEqual(p['certainty'],'unresolved')

    def fixture(self):
        props={'action':{'enum':['diagnose']},'target':{'enum':['TEST_D1','TEST_D2','Unresolved']},
               'department':{'enum':['neurology','endocrinology','general_medicine']},
               'certainty':{'enum':['supported','unresolved']},
               'evidence_indices':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1},'maxItems':1}}
        schema={'type':'object','properties':props,'required':list(props),'additionalProperties':False}
        return {'diagnosis_catalog':['TEST_D1','TEST_D2'],'instruction':'TEST_ONLY'},schema,{'TEST_D1':'neurology','TEST_D2':'endocrinology'}

    def test_same_common_lists_and_no_mutation(self):
        p,s,t=self.fixture();before=deepcopy((p,s,t));q,w=shared_selection(p,s,t)
        self.assertEqual((p,s,t),before);self.assertEqual(q['department_catalog'],['endocrinology','neurology'])
        self.assertEqual(q['diagnosis_catalog'],['TEST_D1','TEST_D2'])
        self.assertEqual(q['diagnosis_department_catalog'],t);Draft202012Validator.check_schema(w)

    def test_out_of_list_labels_and_departments_rejected(self):
        p,s,t=self.fixture();_,w=shared_selection(p,s,t);v={'action':'diagnose','target':'TEST_D1','department':'neurology','certainty':'supported','evidence_indices':[0]}
        Draft202012Validator(w).validate(v)
        for key,bad in [('target','Other spelling'),('department','general_medicine'),('certainty','guessed')]:
            with self.subTest(key=key),self.assertRaises(ValidationError):Draft202012Validator(w).validate({**v,key:bad})

    def test_unresolved_and_evidence_constraints_preserved(self):
        p,s,t=self.fixture();_,w=shared_selection(p,s,t)
        Draft202012Validator(w).validate({'action':'diagnose','target':'Unresolved','department':'neurology','certainty':'unresolved','evidence_indices':[]})
        with self.assertRaises(ValidationError):Draft202012Validator(w).validate({'action':'diagnose','target':'TEST_D1','department':'neurology','certainty':'supported','evidence_indices':[999]})
        s['properties']['certainty']['enum']=['unresolved'];_,w=shared_selection(p,s,t)
        self.assertNotIn('supported',json.dumps(w))

    def test_information_action_targets_unchanged(self):
        p,s,t=self.fixture();b=deepcopy(s);b['properties']['action']['enum']=['request_test'];b['properties']['target']['enum']=['TEST_PANEL']
        _,w=shared_selection(p,{'anyOf':[b,s]},t)
        self.assertEqual(w['anyOf'][0]['properties']['target']['enum'],['TEST_PANEL'])

    def test_common_lists_do_not_use_patient_reference(self):
        p,s,t=self.fixture()
        for secret in ('PRIVATE_A','PRIVATE_B'):
            q,w=shared_selection({**p,'observation':{'chief_complaint':secret}},s,t)
            self.assertEqual(q['diagnosis_catalog'],list(t));self.assertNotIn(secret,json.dumps(w))

    def test_native_opt_in_reaches_transport_without_network(self):
        from app.locdi.common_menu_backend import CommonMenuNativeClient
        from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient
        p,s,t=self.fixture();p['allowed_actions']=['diagnose']
        client=object.__new__(CommonMenuNativeClient)
        client.source_binding={'rows':[]};client.selection_taxonomy=t
        with patch.object(BudgetedNativeERClient,'chat_json',return_value='TEST_ONLY') as call:
            client.chat_json('TEST_ONLY',json.dumps(p),'TEST_ONLY',schema=s)
            sent=json.loads(call.call_args.args[1]);schema=call.call_args.kwargs['schema']
            self.assertEqual(sent['department_catalog'],sorted(set(t.values())))
            self.assertEqual(sent['diagnosis_catalog'],list(t))
            self.assertEqual(schema['anyOf'][0]['properties']['department']['enum'],sorted(set(t.values())))

if __name__=='__main__':unittest.main()
