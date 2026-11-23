from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT
from app.locdi.agentclinic_scenario_adapter import adapt_test_record
from app.locdi.agentclinic_turn_runtime import run_scenario
from app.locdi.agentclinic_episode_exporter import export_episode, public_episode
from scripts.audit_agentclinic_dataset import lexical_hit
from agentclinic_offline_support import bundle, fixture, ScriptedDoctor, e2e_actions


class AgentClinicBoundaryTests(unittest.TestCase):
    def test_all_15_literal_label_results_hidden_until_exact_test_request(self):
        covered = []
        for ordinal, (s,key) in enumerate(zip(bundle().scenarios,bundle().scoring_keys),1):
            hits = [t for t in s.test_targets if isinstance(t.value,str) and lexical_hit(key.correct_diagnosis,t.value)]
            if not hits:continue
            covered.append(ordinal)
            c = ScriptedDoctor([{'action':'ASK','target':'history'}, {'action':'TEST','target':hits[0].target_id},
                                {'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_FINAL'}])
            e = run_scenario(s,c)
            self.assertNotIn(key.correct_diagnosis.casefold(),json.dumps(c.seen[:2],ensure_ascii=False).casefold())
            self.assertTrue(lexical_hit(key.correct_diagnosis,c.seen[2]['observations'][-1]['value']))
            self.assertNotIn('Correct_Diagnosis',json.dumps(c.seen))
            self.assertNotIn(key.correct_diagnosis.casefold(),json.dumps(public_episode(e),ensure_ascii=False).casefold())
        self.assertEqual(covered,[2,3,11,14,18,20,23,39,48,52,62,86,87,102,107])

    def test_leaf_request_reveals_no_sibling_and_group_is_explicit(self):
        s,_ = adapt_test_record(fixture())
        c = ScriptedDoctor([{'action':'TEST','target':'test/0002'}, {'action':'DIAGNOSE','predicted_diagnosis':''}])
        e = run_scenario(s,c)
        text=json.dumps(c.seen[1]['observations'])
        self.assertIn('TEST_ONLY_A',text);self.assertNotIn('TEST_ONLY_B',text)
        e = run_scenario(s,ScriptedDoctor([{'action':'TEST','target':'test/0001'},{'action':'DIAGNOSE','predicted_diagnosis':''}]))
        self.assertEqual(e['trace'][0]['observation']['value'],s.test_results['Group'])

    def test_client_cannot_mutate_source_or_force_future_release(self):
        s,_ = adapt_test_record(fixture());before=deepcopy(s)
        class Mutator(ScriptedDoctor):
            def choose_action(self,obs):
                obs['observations'].append({'value':'TEST_ONLY_FORGED'})
                return super().choose_action(obs)
        e=run_scenario(s,Mutator(e2e_actions(s)))
        self.assertEqual(s,before);self.assertNotIn('TEST_ONLY_FORGED',json.dumps(e))

    def test_public_export_hides_all_raw_values_prediction_and_unresolved_reason(self):
        s,key=adapt_test_record(fixture())
        e=run_scenario(s,ScriptedDoctor([{'action':'ASK','target':'history'},
                       {'action':'DIAGNOSE','predicted_diagnosis':key.correct_diagnosis}]))
        public=public_episode(e)
        for private in (key.correct_diagnosis,'TEST_ONLY_HISTORY'):
            self.assertNotIn(private,json.dumps(public))
        self.assertIn(key.correct_diagnosis,json.dumps(e))

    def test_atomic_export_no_clobber_or_path_escape_and_private_scoring_outside_episode(self):
        s,_=adapt_test_record(fixture());e=run_scenario(s,ScriptedDoctor(e2e_actions(s)))
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/research_evaluation')) as pub, tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as priv:
            export_episode(e,public_root=pub,private_root=priv,name='TEST_ONLY')
            path=Path(pub)/'episodes/TEST_ONLY.json';before=path.read_bytes()
            with self.assertRaises(FileExistsError):export_episode(e,public_root=pub,private_root=priv,name='TEST_ONLY')
            with self.assertRaises(ValueError):export_episode(e,public_root=pub,private_root=priv,name='../source')
            self.assertEqual(path.read_bytes(),before)
            self.assertNotIn('correct_diagnosis',(Path(priv)/'episodes/TEST_ONLY.json').read_text(encoding='utf8'))

    def test_public_failure_preserves_private_partial_export(self):
        s,_=adapt_test_record(fixture());e=run_scenario(s,ScriptedDoctor(e2e_actions(s)))
        from app.locdi.episode_exporter import _write
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/research_evaluation')) as pub, tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as priv:
            def fail_public(doc,destination,root,category):
                if Path(root)==Path(pub):raise OSError('TEST_ONLY_DISK')
                return _write(doc,destination,root,category)
            with patch('app.locdi.agentclinic_episode_exporter._write',side_effect=fail_public):
                with self.assertRaises(OSError):export_episode(e,public_root=pub,private_root=priv,name='TEST_ONLY')
            self.assertTrue((Path(priv)/'episodes/TEST_ONLY.json').is_file())
            self.assertFalse((Path(pub)/'episodes/TEST_ONLY.json').exists())
