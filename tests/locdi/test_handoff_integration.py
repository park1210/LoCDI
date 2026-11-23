"""TEST_ONLY real Core integration; no synthetic Runner success or live SDK."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.handoff_integration import run_handoff_integrated,IntegratedSequentialClient
from app.locdi.handoff_audit import audit_saved_success,project_audit
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT
from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.voi_planning import plan_voi
from app.core.diagnostic_belief import posterior
from app.locdi.sequential_handoff import HandoffStopped
from handoff_integration_support import integrated_options
import app.locdi.live_handoff as live


class HandoffIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0];cls.results={};cls.options={}
        for name,routing,voi in [('A',False,False),('B',True,False),('C',False,True),('D',True,True)]:
            options=integrated_options(cls.case,routing=routing,voi=voi)
            with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner,patch(
                    'app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler,patch(
                    'app.core.information_value.plan_voi',wraps=plan_voi) as planner,patch(
                    'app.core.information_outcomes.posterior',wraps=posterior) as bayes,patch(
                    'app.core.hospital_llm_policy.OllamaClient',side_effect=AssertionError('NO_REAL_MODEL')):
                result=run_handoff_integrated(cls.case,options=options,test_only=True,
                    routing='adaptive_test_only' if routing else 'static_development')
            result['TEST_ONLY_calls']={'runner':runner.call_count,'scheduler':scheduler.call_count,'planner':planner.call_count,'posterior':bayes.call_count}
            cls.results[name]=result;cls.options[name]=options

    def test_A_separate_three_stages_core_and_terminal(self):
        r=self.results['A'];self.assertEqual(r['status'],'completed');self.assertTrue(r['integration_contract_ready'])
        self.assertEqual([x['stage'] for x in r['audit']['requests'][:3]],['ER_INITIAL_DECISION','SPECIALTY_ASSESSMENT','ER_REASSESSMENT'])
        self.assertEqual(r['TEST_ONLY_calls']['runner'],1);self.assertEqual(r['TEST_ONLY_calls']['scheduler'],1)
        self.assertEqual(r['audit']['handoff_status'],'completed');self.assertEqual(r['audit']['source_consult_count'],0)

    def test_B_adaptive_selects_specialty_on_outer_snapshot_only(self):
        r=self.results['B'];self.assertTrue(r['integration_contract_ready'])
        self.assertEqual(r['audit']['routing_mode'],'adaptive_test_only');self.assertFalse(r['audit']['routing_performance_evaluated'])
        self.assertEqual(r['audit']['routing'][0]['selected_specialty'],'neurology')
        self.assertEqual([c['payload']['handoff_stage'] for c in self.options['B'].client.calls[:3]],['ER_INITIAL_DECISION','SPECIALTY_ASSESSMENT','ER_REASSESSMENT'])
        self.assertEqual(r['audit']['core_routing_status'],'disabled')

    def test_C_handoff_before_voi_then_real_source_bayes(self):
        r=self.results['C'];self.assertTrue(r['integration_contract_ready']);v=r['audit']['voi']
        self.assertGreater(r['TEST_ONLY_calls']['planner'],0);self.assertGreater(r['TEST_ONLY_calls']['posterior'],0)
        self.assertEqual([x['selected_action'] for x in v],['ask_patient','request_test','diagnose'])
        self.assertEqual([x['matched_outcome_count'] for x in v],[0,1,2])
        self.assertEqual([x['belief_changed'] for x in v],[False,True,True])
        self.assertTrue(all(x['preceding_request_stage']=='ER_REASSESSMENT' and x['execution_event_sequences'] for x in v))
        self.assertTrue(all(not x['specialty_opinion_used_as_source'] and x['horizon']==2 for x in v))

    def test_D_all_features_and_distinct_role_calls(self):
        r=self.results['D'];self.assertTrue(r['integration_contract_ready'])
        self.assertTrue(r['audit']['routing'] and r['audit']['voi']);self.assertEqual(r['audit']['handoff_status'],'completed')
        self.assertGreater(r['audit']['offline_dispatch_count'],r['audit']['core_action_decision_count'])
        self.assertEqual(r['audit']['actual_generation_request_count'],0);self.assertEqual(r['audit']['identity_check_count'],0)

    def test_voi_replacement_is_not_claimed_to_be_the_llm_selected_action(self):
        rows=self.results['C']['audit']['requests']
        replaced=[r for r in rows if r['selection_relation']=='proposal_replaced_by_voi']
        self.assertTrue(replaced)
        self.assertTrue(all(r['model_proposal_action']!=r['actual_selected_action'] for r in replaced))
        self.assertTrue(all(r['action_id'] is None and not r['execution_event_sequences'] for r in rows[:2]))

    def test_binding_and_identity_are_preserved_without_claiming_fake_identity(self):
        actual=audit_saved_success()['audit_v2']['requests']
        self.assertTrue(all(r['identity_validation']=='MATCHED' and r['prompt_fingerprint'] and r['response_fingerprint'] for r in actual))
        self.assertTrue(all(r['identity_validation']=='NOT_CHECKED' for r in self.results['A']['audit']['requests']))

    def test_E_abstention_continues_er_without_fabricated_specialty(self):
        options=integrated_options(self.case,routing=True,abstain=True)
        r=run_handoff_integrated(self.case,options=options,test_only=True,routing='adaptive_test_only')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['audit']['handoff_status'],'not_performed_abstained')
        self.assertEqual(r['audit']['routing'][0]['status'],'abstained')
        self.assertFalse(any(c['payload']['handoff_stage']=='SPECIALTY_ASSESSMENT' for c in options.client.calls))

    def test_disabled_is_distinct_from_abstention(self):
        r=run_handoff_integrated(self.case,options=integrated_options(self.case),test_only=True,routing='disabled')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['audit']['handoff_status'],'not_performed_disabled')
        self.assertEqual(r['audit']['routing'][0]['status'],'disabled')

    def test_F_missing_voi_blocks_before_runner(self):
        options=replace(integrated_options(self.case,voi=True),voi_model=None)
        with patch('app.locdi.handoff_integration.run_case') as runner:
            r=run_handoff_integrated(self.case,options=options,test_only=True)
        runner.assert_not_called();self.assertEqual(r['blocker_codes'],['VOI_ARTIFACT_MISSING']);self.assertEqual(options.client.calls,[])

    def test_missing_routing_utility_is_not_abstention(self):
        options=replace(integrated_options(self.case,routing=True),routing_utility=None)
        r=run_handoff_integrated(self.case,options=options,test_only=True,routing='adaptive_test_only')
        self.assertEqual(r['blocker_codes'],['ROUTING_UTILITY_MISSING']);self.assertEqual(options.client.calls,[])

    def test_G_timeout_semantic_repair_success_and_failure(self):
        for mode,status in [('timeout','failed'),('semantic_repair','completed'),('semantic_failure','failed')]:
            options=integrated_options(self.case,mode=mode)
            r=run_handoff_integrated(self.case,options=options,test_only=True)
            self.assertEqual(r['status'],status)
            self.assertEqual(r['audit']['specialty_semantic_repair_count'],0 if mode=='timeout' else 1)
            if status=='failed':self.assertNotEqual(r['audit']['handoff_status'],'completed')

    def test_H_global_budget_includes_specialty_and_repairs(self):
        for budget in (1,2,3):
            options=integrated_options(self.case,mode='semantic_repair')
            r=run_handoff_integrated(self.case,options=options,test_only=True,budget=budget)
            self.assertEqual(r['status'],'failed');self.assertEqual(len(options.client.calls),budget)
            self.assertIn('GLOBAL_REQUEST_BUDGET_EXHAUSTED',r['blocker_codes'])

    def test_I_future_test_never_released_before_delivery(self):
        for name,r in self.results.items():
            final=r['raw']['execution']['visits'][0]['final_state']
            received={e['payload']['requested_test']:e['minute'] for e in final['event_log'] if e['event_type']=='result_received'}
            for c in self.options[name].client.calls:
                obs=c['payload']['observation'];text=json.dumps(obs['objective_findings'])
                for e in self.case['environment']['current_encounter_events']:
                    if e['action']=='Test' and e['target']+': '+e['evidence'] in text:
                        self.assertGreaterEqual(obs['elapsed_minutes'],received[e['target']])
                        self.assertGreaterEqual(received[e['target']],e['available_minute'])

    def test_future_evidence_memory_gold_block_before_any_dispatch(self):
        base=self.options['A'].client.calls[0]['payload']
        for kind in ('gold','source_event_store','future_memory','future_evidence'):
            options=integrated_options(self.case);p=deepcopy(base)
            if kind in ('gold','source_event_store'):p[kind]={}
            elif kind=='future_memory':p['observation']['longitudinal_memory']['records'][0]['available_at']='2099-01-01T00:00:00Z'
            else:
                p['observation']['objective_findings']=['TEST_ONLY_FUTURE'];p['evidence']=['TEST_ONLY_FUTURE']
                p['evidence_records']=[{'source_id':'finding:0','kind':'objective_finding','text':'TEST_ONLY_FUTURE','supports_finalization':True,'available_minute':99}]
                p['supported_diagnosis_eligible_indices']=[0]
            c=IntegratedSequentialClient(options.client,AgentRegistry(options.registry_manifest),episode_id='TEST_ONLY',
                patient_id=self.case['patient_id'],visit_id=self.case['agent_visible']['index_encounter_id'],cutoff=self.case['agent_visible']['index_encounter_start'],budget=10)
            with self.assertRaises(HandoffStopped):c.chat_json('TEST_ONLY',json.dumps(p),model=options.registry_manifest['agents'][0]['model'],schema={})
            self.assertEqual(options.client.calls,[])

    def test_J_export_episode_audit_summary_match_and_no_clobber(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            options=integrated_options(self.case,routing=True,voi=True)
            r=run_handoff_integrated(self.case,options=options,test_only=True,routing='adaptive_test_only',output_root=d,export_id='TEST_ONLY_D')
            self.assertEqual(r['export_status'],'EXPORTED')
            audit=json.loads((Path(d)/'reports/TEST_ONLY_D_handoff_v2.json').read_text(encoding='utf-8'))
            episode=json.loads((Path(d)/'episodes/TEST_ONLY_D.json').read_text(encoding='utf-8'))
            summary=json.loads((Path(d)/'metrics/TEST_ONLY_D.jsonl').read_text(encoding='utf-8'))['episodes'][0]
            self.assertEqual(audit['workflow'],episode['workflow']);self.assertEqual(audit['terminal'],episode['result'])
            self.assertEqual(audit['configuration_fingerprint'],summary['configuration_fingerprint'])
            self.assertNotIn(self.case['patient_id'],json.dumps(audit));self.assertNotIn('TEST_ONLY assessment',json.dumps(audit))
            before=(Path(d)/'episodes/TEST_ONLY_D.json').read_bytes()
            again=run_handoff_integrated(self.case,options=integrated_options(self.case,routing=True,voi=True),test_only=True,routing='adaptive_test_only',output_root=d,export_id='TEST_ONLY_D')
            self.assertEqual(again['export_status'],'FAILED_OR_PARTIAL');self.assertEqual(before,(Path(d)/'episodes/TEST_ONLY_D.json').read_bytes())

    def test_handoff_completed_can_precede_later_episode_failure(self):
        r=run_handoff_integrated(self.case,options=integrated_options(self.case,mode='later_failure'),test_only=True)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['audit']['handoff_status'],'completed')
        self.assertEqual(r['audit']['episode_status'],'failed')

    def test_fallback_is_observed_not_promoted_to_integration_ready(self):
        r=run_handoff_integrated(self.case,options=integrated_options(self.case,voi=True,fallback=True),test_only=True)
        self.assertIn('VOI_FALLBACK_OBSERVED',r['blocker_codes']);self.assertFalse(r['integration_contract_ready'])

    def test_live_and_research_boundaries_remain_blocked(self):
        options=integrated_options(self.case,routing=True,voi=True)
        with patch('app.locdi.ollama_adapter._backend') as backend:
            self.assertEqual(run_handoff_integrated(self.case,options=options)['status'],'blocked')
            self.assertEqual(run_handoff_integrated(self.case,options=replace(options,purpose='development_llm',allow_llm=True),test_only=True)['status'],'blocked')
            c=live.load_config();c['routing']='adaptive_test_only'
            self.assertTrue(live.preflight(run_id='f'*32,config=c)['blocker_codes'])
        backend.assert_not_called();self.assertEqual(options.client.calls,[])

    def test_saved_actual_success_and_unknown_initial_memory_time(self):
        r=audit_saved_success();self.assertEqual(r['status'],'PASS')
        audit=r['audit_v2'];self.assertEqual(audit['actual_generation_request_count'],4)
        self.assertEqual(audit['core_action_decision_count'],2);self.assertEqual(audit['logical_judgment_request_count'],2)
        self.assertIsNone(audit['memory']['initial_retrieval_minute']);self.assertEqual(audit['core_routing_status'],'disabled')
        self.assertEqual(r['raw_wire_payload_validation'],'NOT_CHECKED_NOT_SAVED')

    def test_partial_export_is_not_reported_complete(self):
        from app.locdi.episode_exporter import _write
        def fail_summary(*args,**kwargs):raise OSError('TEST_ONLY storage')
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs')) as d,patch('app.locdi.handoff_integration.export_summary',side_effect=fail_summary):
            r=run_handoff_integrated(self.case,options=integrated_options(self.case),test_only=True,output_root=d,export_id='TEST_ONLY_PARTIAL')
            self.assertEqual(r['export_status'],'FAILED_OR_PARTIAL');self.assertTrue((Path(d)/'episodes/TEST_ONLY_PARTIAL.json').exists())
            self.assertFalse(r['integration_contract_ready'])


if __name__=='__main__':unittest.main()
