"""TEST_ONLY real Runner/Scheduler/policy integration; no SDK/model calls."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.dataset_loader import load_cases
from app.locdi.integrated_runtime import run_integrated
from app.locdi.llm_setup import ROOT, digest
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.voi_planning import plan_voi
from app.core.diagnostic_belief import posterior
from app.llm.prompt_loader import er_system_prompt, specialty_system_prompt
from test_simulation import SourceActionClient
from runtime_integration_support import options_for


class RecordingClient(SourceActionClient):
    def __init__(self):
        super().__init__();self.systems=[]
    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        self.systems.append(system_prompt)
        return super().chat_json(system_prompt,user_prompt,model,schema=schema)


class RuntimeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
        cls.results={};cls.options={}
        for name,r,v in [('A',False,False),('B',True,False),('C',False,True),('D',True,True)]:
            opts=replace(options_for(cls.case,routing=r,voi=v),client=RecordingClient())
            with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner, \
                 patch('app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler, \
                 patch('app.core.information_value.plan_voi',wraps=plan_voi) as planner, \
                 patch('app.core.information_outcomes.posterior',wraps=posterior) as bayes:
                result=run_integrated(cls.case,options=opts,test_only=True)
            cls.results[name]=result;cls.options[name]=opts
            cls.results[name]['TEST_ONLY_calls']={'runner':runner.call_count,'scheduler':scheduler.call_count,'planner':planner.call_count,'posterior':bayes.call_count}

    def test_er_only_real_engine_memory_action_export(self):
        r=self.results['A']
        self.assertEqual(r['status'],'completed')
        self.assertEqual(r['TEST_ONLY_calls']['runner'],1)
        self.assertEqual(r['TEST_ONLY_calls']['scheduler'],1)
        self.assertTrue(r['readiness']['runtime_integration_ready'])
        self.assertTrue(r['episode']['memory']['retrievals'])
        self.assertTrue(all(x['cutoff_compliant'] for x in r['episode']['memory']['retrievals']))
        self.assertTrue(all(s==er_system_prompt() for s in self.options['A'].client.systems))
        self.assertTrue(r['episode']['result']['diagnosis_unresolved'])

    def test_adaptive_role_binding_uses_selected_prompt_model(self):
        r=self.results['B'];self.assertEqual(r['status'],'completed')
        self.assertTrue(all(x['status']=='selected' and x['selected_specialty']=='neurology' for x in r['audit']['routing']))
        self.assertTrue(all(s==specialty_system_prompt('neurology') for s in self.options['B'].client.systems))
        self.assertTrue(all(c['model']=='TEST_ONLY_NO_MODEL' for c in self.options['B'].client.calls))
        self.assertEqual(r['audit']['source_backed_consult_event_count'],0)
        self.assertTrue(r['readiness']['runtime_integration_ready'])

    def test_voi_selects_then_observation_updates_and_terminal(self):
        r=self.results['C'];v=r['audit']['voi']
        self.assertGreater(r['TEST_ONLY_calls']['planner'],0)
        self.assertGreater(r['TEST_ONLY_calls']['posterior'],0)
        self.assertEqual([x['selected_action'] for x in v],['ask_patient','request_test','diagnose'])
        self.assertEqual([x['matched_outcome_count'] for x in v],[0,1,2])
        self.assertEqual([x['belief_changed'] for x in v],[False,True,True])
        self.assertTrue(all(x['horizon']==2 and x['decision_sequences'] for x in v))
        self.assertEqual(r['status'],'completed')
        self.assertTrue(r['readiness']['runtime_integration_ready'])

    def test_combined_routing_and_voi_are_both_executed(self):
        r=self.results['D'];self.assertTrue(r['audit']['routing'] and r['audit']['voi'])
        self.assertTrue(r['readiness']['runtime_integration_ready'])
        self.assertTrue(all(s==specialty_system_prompt('neurology') for s in self.options['D'].client.systems))
        self.assertGreater(r['TEST_ONLY_calls']['planner'],0)

    def test_abstention_continues_er_without_specialty(self):
        client=RecordingClient();opts=replace(options_for(self.case,routing=True,abstain=True),client=client)
        r=run_integrated(self.case,options=opts,test_only=True)
        self.assertEqual(r['status'],'completed')
        self.assertTrue(all(x['status']=='abstained' and x['selected_specialty'] is None for x in r['audit']['routing']))
        self.assertTrue(all(s==er_system_prompt() for s in client.systems))

    def test_missing_utility_blocks_before_client_and_er_only_still_works(self):
        opts=replace(options_for(self.case,routing=True),routing_utility=None)
        r=run_integrated(self.case,options=opts,test_only=True)
        self.assertEqual(r['blocker_codes'],['ROUTING_UTILITY_MISSING'])
        self.assertEqual(opts.client.calls,[])
        self.assertIsNone(r['episode'])
        self.assertTrue(self.results['A']['readiness']['runtime_integration_ready'])

    def test_real_artifact_readiness_not_replaced_by_test_success(self):
        opts=options_for(self.case,routing=True,voi=True)
        r=run_integrated(self.case,options=opts)
        self.assertEqual(r['status'],'blocked')
        self.assertFalse(r['readiness']['clinical_reference_ready'] or r['readiness']['research_evaluation_ready'])
        self.assertEqual(opts.client.calls,[])
        for r in self.results.values():
            self.assertIsNone(r['correctness'])
            self.assertFalse(r['readiness']['clinical_reference_ready'] or r['readiness']['research_evaluation_ready'])

    def test_registry_utility_catalog_and_artifact_mismatch_block(self):
        r=run_integrated(self.case,options=options_for(self.case,routing=True),test_only=True,expected_registry_fingerprint='0'*64)
        self.assertEqual(r['blocker_codes'],['REGISTRY_FINGERPRINT_MISMATCH'])
        opts=options_for(self.case,routing=True);opts.routing_utility['registry']['agents'][0]['version']='TEST_ONLY_WRONG'
        self.assertEqual(run_integrated(self.case,options=opts,test_only=True)['status'],'blocked')
        opts=replace(options_for(self.case,voi=True),diagnosis_catalog=['TEST_ONLY_WRONG'])
        self.assertEqual(run_integrated(self.case,options=opts,test_only=True)['status'],'blocked')
        opts=replace(opts,diagnosis_departments={'TEST_ONLY_WRONG':'general_medicine'})
        self.assertEqual(run_integrated(self.case,options=opts,test_only=True)['blocker_codes'],['VOI_CATALOG_OR_TEST_ONLY_PROVENANCE_MISMATCH'])
        r=run_integrated(self.case,options=options_for(self.case,voi=True),test_only=True,expected_voi_fingerprint='0'*64)
        self.assertEqual(r['blocker_codes'],['VOI_FINGERPRINT_MISMATCH'])

    def test_future_test_not_visible_before_source_and_delivery(self):
        for name in self.results:
            r=self.results[name];raw=r['raw'];final=raw['execution']['visits'][0]['final_state']
            received={e['payload']['requested_test']:e['minute'] for e in final['event_log'] if e['event_type']=='result_received'}
            for call in self.options[name].client.calls:
                payload=call['payload'];obs=payload['observation'];texts=json.dumps(obs['objective_findings'])
                self.assertFalse(any(k in payload for k in ('gold','evaluation','provenance','source_event_store')))
                for event in self.case['environment']['current_encounter_events']:
                    if event['action']=='Test' and event['target']+': '+event['evidence'] in texts:
                        self.assertIn(event['target'],received)
                        self.assertGreaterEqual(obs['elapsed_minutes'],received[event['target']])
                        self.assertGreaterEqual(received[event['target']],event['available_minute'])

    def test_episode_audit_action_time_export_consistency(self):
        for r in self.results.values():
            self.assertEqual(r['audit']['configuration_fingerprint'],r['episode']['configuration_fingerprint'])
            self.assertEqual(r['audit']['workflow'],r['episode']['workflow'])
            self.assertTrue(all(d['execution_success'] is True for d in r['episode']['workflow']['decisions']))
            self.assertEqual(r['audit']['terminal'],r['episode']['result'])
            copy=deepcopy(r['audit']);fp=copy.pop('fingerprint');self.assertEqual(fp,digest(copy))
            text=json.dumps([r['audit'],r['episode']])
            for value in (self.case['patient_id'],self.case['case_id'],self.case['agent_visible']['index_encounter_id']):
                self.assertNotIn(value,text)

    def test_fallback_and_error_remain_observed_not_success(self):
        r=run_integrated(self.case,options=options_for(self.case,voi=True,fallback=True),test_only=True)
        self.assertIn('VOI_FALLBACK_OBSERVED',r['blocker_codes'])
        self.assertFalse(r['readiness']['runtime_integration_ready'])
        self.assertTrue(any(x['fallback'] for x in r['audit']['voi']))
        r=run_integrated(self.case,options=options_for(self.case,failure='timeout'),test_only=True)
        self.assertEqual(r['status'],'failed')
        self.assertFalse(r['readiness']['runtime_integration_ready'])
        self.assertIn('RoleCallError',r['episode']['result']['error_types'])

    def test_saved_episode_summary_runtime_audit_reload(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            r=run_integrated(self.case,options=options_for(self.case,voi=True),test_only=True,output_root=directory,export_id='TEST_ONLY_RUNTIME')
            self.assertEqual(json.loads((Path(directory)/'episodes/TEST_ONLY_RUNTIME.json').read_text(encoding='utf-8')),r['episode'])
            audit=json.loads((Path(directory)/'reports/TEST_ONLY_RUNTIME_runtime.json').read_text(encoding='utf-8'))
            self.assertEqual(audit,r['audit'])
            summary=json.loads((Path(directory)/'metrics/TEST_ONLY_RUNTIME.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(summary['episodes'][0]['action_counts'],r['episode']['result']['action_counts'])
            before=(Path(directory)/'episodes/TEST_ONLY_RUNTIME.json').read_bytes()
            duplicate=run_integrated(self.case,options=options_for(self.case,voi=True),test_only=True,output_root=directory,export_id='TEST_ONLY_RUNTIME')
            self.assertIn('EXPORT_OR_AUDIT_CONTRACT_ERROR',duplicate['blocker_codes'])
            self.assertEqual(before,(Path(directory)/'episodes/TEST_ONLY_RUNTIME.json').read_bytes())


if __name__=='__main__':
    unittest.main()
