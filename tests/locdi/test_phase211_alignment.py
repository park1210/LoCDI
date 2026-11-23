"""TEST_ONLY prerequisites and decision math; no clinical review or fitting."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_loader import load_cases
from app.locdi.integrated_approval import preflight_integrated,request_integrated_approval
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from app.core.agent_registry import AgentRegistry
from app.core.diagnostic_belief import posterior,terminal_choice
from app.core.information_outcomes import InformationOutcomes
from app.core.voi_planning import plan_voi
from validation.research_artifact_gate import assess_artifacts
from test_routing_voi_readiness import test_only_registry,test_only_routing,test_only_voi
from test_routing_correctness_gate import test_only_reviewed_data
from phase211_support import select_cases,execute_case,boundary_check


class ResearchArtifactAdmissionTests(unittest.TestCase):
    def gate(self,**kw):
        return assess_artifacts(test_only_registry(),**kw)

    def test_absent_artifacts_block_separately_without_fit_or_client(self):
        with patch('app.core.routing_utility.fit_utility',side_effect=AssertionError('NO_FIT')),patch(
                'app.locdi.ollama_adapter._backend',side_effect=AssertionError('NO_MODEL')):
            r=self.gate()
        self.assertFalse(r['research_evaluation_ready'])
        self.assertTrue(all(a['status']=='BLOCKED' for a in r['artifacts'].values()))

    def test_actual_gate_reexecuted_and_75_remain_unreviewed(self):
        r=self.gate(correctness_inputs=actual_gate_inputs())
        self.assertEqual(r['correctness']['complete_train_groups'],0)
        self.assertEqual(r['correctness']['validation_group_count'],0)
        self.assertFalse(r['correctness']['clinical_review_ready'])

    def test_precomputed_forged_report_is_not_gate_input(self):
        r=self.gate(correctness_inputs={'utility_fit_ready':True,'complete_train_groups':10})
        self.assertEqual(r['correctness'],{'status':'NOT_CHECKED'})

    def test_toy_review_contract_cannot_become_research_review(self):
        data,opts=test_only_reviewed_data()
        r=assess_artifacts(opts['registry_manifest'],correctness_inputs=(data,opts))
        self.assertEqual(r['correctness']['complete_train_groups'],10)
        self.assertIn('CORRECTNESS_GATE',r['artifacts']['routing']['blocker_codes'])
        self.assertFalse(r['research_evaluation_ready'])

    def test_self_declared_clinical_likelihood_review_does_not_promote(self):
        a=test_only_voi();a['provenance'].update(synthetic=False,review_status='expert_reviewed',research_validated=True)
        r=self.gate(voi_artifact=a)
        states={x['code']:x['status'] for x in r['artifacts']['voi']['checks']}
        self.assertEqual(states['NON_TOY_PROVENANCE'],'READY')
        self.assertEqual(states['LIKELIHOOD_PROVENANCE_REVIEW'],'NOT_CHECKED')
        self.assertEqual(r['artifacts']['voi']['status'],'BLOCKED')

    def test_external_binding_hash_and_prompt_change_without_schema_change(self):
        a=test_only_voi();binding={'registry_fingerprint':digest(test_only_registry()),'prompt_fingerprint':'a'*64,
            'model_identifier':'TEST_ONLY','memory_fingerprint':'b'*64,'action_catalog_fingerprint':'c'*64}
        declarations={'voi':{'artifact_fingerprint':digest(a),'research_binding':deepcopy(binding)}}
        r=self.gate(voi_artifact=a,binding=binding,declarations=declarations)
        states=lambda r:{x['code']:x['status'] for x in r['artifacts']['voi']['checks']}
        self.assertEqual(states(r)['CORE_STRUCTURAL_CONTRACT'],'READY')
        self.assertEqual(states(r)['MODEL_ACTION_MEMORY_BINDING'],'READY')
        binding['prompt_fingerprint']='d'*64
        self.assertEqual(states(self.gate(voi_artifact=a,binding=binding,declarations=declarations))['MODEL_ACTION_MEMORY_BINDING'],'BLOCKED')
        binding['prompt_fingerprint']='a'*64;declarations['voi']['artifact_fingerprint']='e'*64
        self.assertEqual(states(self.gate(voi_artifact=a,binding=binding,declarations=declarations))['MODEL_ACTION_MEMORY_BINDING'],'BLOCKED')
        self.assertNotIn('research_binding',a)

    def test_invalid_prior_or_likelihood_blocked_by_existing_core(self):
        a=test_only_voi();a['prior']['TEST_ONLY_A']=.99
        r=self.gate(voi_artifact=a)
        self.assertIn('CORE_STRUCTURAL_CONTRACT',r['artifacts']['voi']['blocker_codes'])

    def test_cross_artifact_validation_overlap_is_explicit(self):
        data,opts=test_only_reviewed_data();a=test_only_voi()
        a['provenance']['training_patients']=data['split_manifest']['patients']['validation'][:1]
        r=self.gate(voi_artifact=a,correctness_inputs=(data,opts))
        self.assertEqual(r['routing_validation_voi_train_overlap_count'],1)
        self.assertIn('ROUTING_VALIDATION_VOI_TRAIN_DISJOINT',r['artifacts']['voi']['blocker_codes'])

    def test_toy_valid_gate_does_not_approve_unrelated_artifact_samples(self):
        data,opts=test_only_reviewed_data()
        r=assess_artifacts(opts['registry_manifest'],correctness_inputs=(data,opts))
        self.assertIn('FIT_ROWS_ARTIFACT_LINEAGE',r['artifacts']['routing']['blocker_codes'])


class IntegratedApprovalTests(unittest.TestCase):
    def plan(self,**options):
        return preflight_integrated(run_id='a'*32,mode=options.pop('mode','adaptive_and_voi'),**options)

    def test_all_three_live_modes_block_before_backend(self):
        with patch('app.locdi.ollama_adapter._backend',side_effect=AssertionError('NO_MODEL')):
            for mode in ('adaptive_routing','bayesian_voi','adaptive_and_voi'):
                r=request_integrated_approval(run_id='a'*32,mode=mode,execute=True,allow_llm=True,accept_release_assumption=True)
                self.assertEqual(r['status'],'blocked');self.assertEqual(r['actual_generation_calls'],0)
                self.assertIn('LIVE_INTEGRATED_EXECUTOR_NOT_IMPLEMENTED',r['blocker_codes'])

    def test_plan_binds_current_contract_and_is_deterministic(self):
        a=self.plan();b=self.plan();self.assertIsNotNone(a['plan'])
        self.assertEqual(a['plan_fingerprint'],b['plan_fingerprint'])
        for k in ('stages','registry_fingerprint','model_identifier','utility_artifact_hash','voi_artifact_hash',
                  'dataset_source_sha256','current_binding','runtime_configuration','global_request_budget',
                  'action_catalog_fingerprint','release_contract','research_evaluation_ready','output_namespace','code_sha256'):
            self.assertIn(k,a['plan'])
        self.assertFalse(a['plan']['runtime_configuration']['pre_decision_hook'])

    def test_fingerprint_or_budget_mutation_cannot_authorize(self):
        a=self.plan()
        b=request_integrated_approval(run_id='a'*32,mode='adaptive_and_voi',budget=9,
            execute=True,allow_llm=True,accept_release_assumption=True,expected_plan_fingerprint=a['plan_fingerprint'])
        self.assertIn('APPROVED_PLAN_FINGERPRINT_MISMATCH',b['blocker_codes'])
        self.assertEqual(b['status'],'blocked')

    def test_used_actual_run_and_unsafe_paths_are_rejected(self):
        for value in ('20261030000000000000000000002903','../escape'):
            self.assertIsNone(preflight_integrated(run_id=value,mode='adaptive_and_voi')['plan'])

    def test_invalid_boolean_and_budget_no_execution(self):
        for budget in (True,0,11):self.assertIn('MODE_OR_BUDGET_INVALID',self.plan(budget=budget)['blocker_codes'])
        r=request_integrated_approval(run_id='a'*32,mode='adaptive_and_voi',execute=1,allow_llm=True,accept_release_assumption=True)
        self.assertIn('EXPLICIT_LIVE_APPROVAL_REQUIRED',r['blocker_codes'])


class DecisionMathAlignmentTests(unittest.TestCase):
    def test_jaccard_beta_accuracy_formula_and_abstention(self):
        m=test_only_registry();registry=AgentRegistry(m,test_only_routing(m))
        d=registry.route({'chief_complaint':'test_only_signal'},['neurology','cardiology'],mode='adaptive')
        self.assertAlmostEqual(d['specialty_scores']['neurology'],(1+2)/(2+2))
        self.assertEqual(registry.route({},['neurology','cardiology'],mode='adaptive')['decision_status'],'abstained')

    def test_bayes_repeat_and_duplicate_evidence_are_not_incremental_reupdates(self):
        m=InformationOutcomes(test_only_voi());obs={'objective_findings':['TEST_ONLY_T: TEST_ONLY_A']}
        b,matched=m.observe(obs);self.assertAlmostEqual(b['TEST_ONLY_A'],.9)
        self.assertEqual(m.observe(obs),(b,matched))
        self.assertEqual(m.observe({'objective_findings':obs['objective_findings']*2}),(b,matched))
        self.assertEqual(m.artifact['prior'],{'TEST_ONLY_A':.5,'TEST_ONLY_B':.5})

    def test_bellman_expected_utility_is_not_entropy_objective(self):
        a=test_only_voi();p=plan_voi(a['prior'],a['actions'],a['terminal_utility'],horizon=2)
        self.assertAlmostEqual(terminal_choice(a['prior'],a['terminal_utility'])[1],.5)
        self.assertAlmostEqual(p['action_values'][0]['q_value'],.9)
        self.assertAlmostEqual(p['action_values'][0]['voi'],.4)
        self.assertEqual(p['cost_model'],'none')


class MultiCaseAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
        cls.selected=select_cases(cls.cases);cls.results=[]
        with patch('app.core.hospital_llm_policy.OllamaClient',side_effect=AssertionError('NO_MODEL')):
            for row in cls.selected:
                c=cls.cases[row['ordinal']-1]
                for scenario in ('A','B','C','D'):
                    result,client,record=execute_case(c,row,scenario)
                    cls.results.append((c,row,scenario,result,client,record))

    def test_selection_varies_memory_and_ignores_gold(self):
        self.assertEqual(len(self.selected),5)
        self.assertGreater(len({r['past_record_count'] for r in self.selected}),1)
        altered=deepcopy(self.cases)
        for c in altered:c['evaluation']={};c['task_annotations']={}
        self.assertEqual([r['ordinal'] for r in select_cases(altered)],[r['ordinal'] for r in self.selected])

    def test_real_runner_scheduler_and_supported_or_blocked_truthfully(self):
        for _,_,scenario,r,_,record in self.results:
            self.assertEqual(record['calls']['runner'],1);self.assertEqual(record['calls']['scheduler'],1)
            # This fixture is NOT guaranteed to have similarity support per case.
            self.assertIn(r['status'],{'completed','incomplete','failed'})
            self.assertEqual(record['dispatch_count'],r['audit']['offline_dispatch_count'])
            if scenario in {'C','D'}:self.assertGreater(record['calls']['plan_voi'],0)

    def test_observed_belief_refresh_follows_er_proposal_on_encounter_path(self):
        for _,_,scenario,r,_,record in self.results:
            if scenario not in {'C','D'}:continue
            order=record['order'];updates=[i for i,x in enumerate(order) if x['kind']=='BELIEF_RECOMPUTED_FROM_PRIOR']
            self.assertTrue(updates)
            self.assertEqual(order[updates[0]-1]['stage'],'ER_REASSESSMENT')
            self.assertTrue(all(v['preceding_request_stage']=='ER_REASSESSMENT' for v in r['audit']['voi']))

    def test_actual_result_release_memory_isolation_and_no_source_opinion(self):
        for case,_,_,r,client,_ in self.results:
            self.assertTrue(boundary_check(case,client,r))
            self.assertTrue(all(x['cutoff_compliant'] for x in r['audit']['memory']['retrievals']))
            self.assertTrue(all(not x['specialty_opinion_used_as_source'] for x in r['audit']['voi']))
            self.assertEqual(r['audit']['source_consult_count'],0)

    def test_public_audit_and_terminal_not_clinical_gold(self):
        for case,_,_,r,_,_ in self.results:
            text=json.dumps(r['audit']);self.assertNotIn(case['patient_id'],text)
            self.assertNotIn(case['agent_visible']['index_encounter_id'],text)
            self.assertEqual(r['audit']['terminal'],r['episode']['result'])
            self.assertFalse(r['research_evaluation_ready'])


if __name__=='__main__':unittest.main()
