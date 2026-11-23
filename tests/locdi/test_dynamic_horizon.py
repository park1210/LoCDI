"""TEST_ONLY state-conditioned Horizon contracts; no clinical labels/fitting."""
from copy import deepcopy
import math
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.dynamic_horizon import DynamicHorizonPolicy,policy_spec
from app.locdi.dynamic_horizon_evaluation import (run_dynamic,run_planner,action_interface,
    common_value,q_diagnostics,patient_summaries,choose_dynamic,restore_model_belief,complete_identity_recall)
from app.locdi.dataset_freeze import sha
from app.locdi.routing_two_decision_collection import verify_pins
from app.core.information_value import select_action
from app.core.information_outcomes import InformationOutcomes
from app.core.voi_planning import plan_voi
from test_routing_voi_readiness import test_only_voi
from test_voi_stop_consistency import interface


def menu(n):
    return [{'action_id':'TEST_ONLY_T%d'%i,'type':'Test','action':'request_test',
        'target':'TEST_ONLY_T%d'%i,'outcome_model':'TEST_ONLY_T%d'%i} for i in range(n)]


def context():
    return {'decision_minute':3,'remaining_minutes':None,'resources':None,'terminal_utility':.5,
        'terminal_tie_count':2,'terminal_action_available':True,'used_action_count':0}


def model_three():
    a=test_only_voi();action=a['actions'][0];a['actions']=[]
    for i in range(3):
        c=deepcopy(action);c['id']='TEST_ONLY_T%d'%i;c['target']=c['id']
        for o in c['outcomes']:o['observation']['text']=c['target']+': TEST_ONLY_'+o['id']
        a['actions'].append(c)
    return InformationOutcomes(a).artifact


class HorizonPolicyTests(unittest.TestCase):
    def setUp(self):self.policy=DynamicHorizonPolicy();self.b={'TEST_ONLY_A':.5,'TEST_ONLY_B':.5}
    def select(self,b=None,n=3,ctx=None,variant=None):
        return (DynamicHorizonPolicy(variant) if variant else self.policy).select(
            belief=self.b if b is None else b,available_actions=menu(n),observable_context=context() if ctx is None else ctx)
    def test_uniform_entropy_normalization(self):
        f=self.select()['observable_feature_summary'];self.assertEqual(f['entropy_nats'],math.log(2))
        self.assertEqual(f['normalized_entropy'],1);self.assertEqual(f['top1_top2_margin'],0)
    def test_low_entropy_high_probability_selects_one(self):
        r=self.select({'TEST_ONLY_A':.99,'TEST_ONLY_B':.01});self.assertEqual(r['selected_horizon'],1)
        self.assertEqual(r['observable_feature_summary']['max_posterior'],.99)
    def test_moderate_uncertainty_selects_two(self):self.assertEqual(self.select({'TEST_ONLY_A':.85,'TEST_ONLY_B':.15})['selected_horizon'],2)
    def test_high_entropy_narrow_margin_selects_three(self):self.assertEqual(self.select()['selected_horizon'],3)
    def test_exact_probability_one(self):
        r=self.select({'TEST_ONLY_A':1.,'TEST_ONLY_B':0.});self.assertEqual(r['selected_horizon'],1)
        self.assertEqual(r['observable_feature_summary']['normalized_entropy'],0)
    def test_margin_is_actual_top_two_difference(self):
        f=self.select({'TEST_ONLY_A':.6,'TEST_ONLY_B':.3,'TEST_ONLY_C':.1})['observable_feature_summary']
        self.assertEqual(f['top1_top2_margin'],.3)
    def test_one_remaining_action_cap(self):self.assertEqual(self.select(n=1)['selected_horizon'],1)
    def test_two_remaining_actions_cap(self):self.assertEqual(self.select(n=2)['selected_horizon'],2)
    def test_no_actions_is_terminal_not_horizon_zero(self):
        r=self.select(n=0);self.assertIsNone(r['selected_horizon']);self.assertEqual(r['effective_information_depth'],0)
        self.assertEqual(r['eligibility_status'],'TERMINAL_ONLY')
    def test_full_supported_menu_range_and_cap(self):
        for n in range(1,13):
            r=self.select(n=n);self.assertIn(r['selected_horizon'],[1,2,3]);self.assertLessEqual(r['selected_horizon'],n)
    def test_too_many_actions_blocked(self):self.assertEqual(self.select(n=13)['eligibility_status'],'BLOCKED')
    def test_duplicate_action_ids_blocked(self):
        m=menu(2);m[1]['action_id']=m[0]['action_id']
        self.assertEqual(self.policy.select(belief=self.b,available_actions=m,observable_context=context())['eligibility_status'],'BLOCKED')
    def test_wrong_action_type_not_counted_as_test(self):
        m=menu(1);m[0]['type']='Ask'
        self.assertEqual(self.policy.select(belief=self.b,available_actions=m,observable_context=context())['eligibility_status'],'BLOCKED')
    def test_unsupported_action_rejected(self):
        m=menu(1);m[0]['action']='treat'
        self.assertEqual(self.policy.select(belief=self.b,available_actions=m,observable_context=context())['eligibility_status'],'BLOCKED')
    def test_invalid_normalization_falls_back_not_renormalized(self):
        r=self.select({'TEST_ONLY_A':.5,'TEST_ONLY_B':.6});self.assertEqual(r['eligibility_status'],'FALLBACK')
        self.assertEqual(r['selected_horizon'],2);self.assertEqual(r['fallback_reason'],'BELIEF_FEATURES_INVALID')
    def test_nan_belief_without_canonical_fingerprint_is_blocked(self):
        r=self.select({'TEST_ONLY_A':float('nan'),'TEST_ONLY_B':.5})
        self.assertEqual(r['eligibility_status'],'BLOCKED');self.assertIsNone(r['selected_horizon'])
    def test_missing_context_fallback(self):
        r=self.policy.select(belief=self.b,available_actions=menu(3),observable_context=None)
        self.assertEqual(r['selected_horizon'],2);self.assertEqual(r['fallback_reason'],'OBSERVABLE_CONTEXT_MISSING')
    def test_invalid_time_fallback_explicit(self):
        ctx=context();ctx['remaining_minutes']=-1
        self.assertEqual(self.select(ctx=ctx)['fallback_reason'],'REMAINING_TIME_INVALID')
    def test_time_resource_unknown_not_imputed(self):
        f=self.select()['observable_feature_summary'];self.assertEqual(f['time_status'],'UNKNOWN')
        self.assertIsNone(f['remaining_minutes']);self.assertEqual(f['resource_status'],'UNKNOWN')
    def test_observed_time_does_not_create_delay_cost_policy(self):
        ctx=context();ctx['remaining_minutes']=0
        self.assertEqual(self.select(ctx=ctx)['selected_horizon'],self.select()['selected_horizon'])
        self.assertFalse(self.select(ctx=ctx)['observable_feature_summary']['time_or_resources_used_in_rule'])
    def test_gold_future_and_q_values_blocked_not_fallback(self):
        for key in ['gold','source_conditions','future_outcomes','q_value','optimal_value','outcomes']:
            ctx=context();ctx[key]={'TEST_ONLY':'TEST_ONLY'}
            with self.subTest(key=key):
                r=self.select(ctx=ctx);self.assertEqual(r['eligibility_status'],'BLOCKED');self.assertIsNone(r['fallback_reason'])
    def test_json_encoded_hidden_label_blocked(self):
        ctx=context();ctx['resources']='{"gold_department":"TEST_ONLY"}'
        self.assertEqual(self.select(ctx=ctx)['eligibility_status'],'BLOCKED')
    def test_changing_future_only_does_not_change_current_selection(self):
        envelope={'current':{'belief':self.b,'actions':menu(3),'context':context()},'future_outcome':'TEST_ONLY_A'}
        def choose(e):
            c=e['current'];return self.policy.select(belief=c['belief'],available_actions=c['actions'],observable_context=c['context'])
        r=choose(envelope);envelope['future_outcome']='TEST_ONLY_B';self.assertEqual(r,choose(envelope))
    def test_policy_deterministic_no_mutation(self):
        before=deepcopy(self.b);self.assertEqual(self.select(),self.select());self.assertEqual(before,self.b)
    def test_policy_fingerprint_changes_only_with_variant(self):
        self.assertEqual(self.policy.fingerprint,DynamicHorizonPolicy().fingerprint)
        self.assertNotEqual(self.policy.fingerprint,DynamicHorizonPolicy('entropy_only').fingerprint)
    def test_no_output_optimization_or_clinical_threshold(self):
        self.assertFalse(policy_spec()['clinical_thresholds']);self.assertFalse(policy_spec()['live_injection'])
    def test_entropy_removal_can_change_test_only_horizon(self):
        b={'TEST_ONLY_A':.9,'TEST_ONLY_B':.1}
        self.assertEqual(self.select(b)['selected_horizon'],2);self.assertEqual(self.select(b,variant='margin_only')['selected_horizon'],1)
    def test_margin_removal_can_change_test_only_horizon(self):
        b={'TEST_ONLY_%d'%i:(.66 if i==0 else .34 if i==1 else 0.) for i in range(32)}
        self.assertEqual(self.select(b)['selected_horizon'],3);self.assertEqual(self.select(b,variant='entropy_only')['selected_horizon'],1)
    def test_feasibility_removal_explicit_fallback_keeps_safety_cap(self):
        r=self.select(n=1,variant='feasibility_removed_fixed_fallback')
        self.assertEqual(r['selected_horizon'],1);self.assertEqual(r['eligibility_status'],'FALLBACK')
    def test_unknown_variant_not_silently_defaulted(self):
        with self.assertRaises(ValueError):DynamicHorizonPolicy('TEST_ONLY_UNKNOWN')


class AdapterAndMeasurementTests(unittest.TestCase):
    def setUp(self):self.a=model_three();self.b=self.a['prior'];self.cs=interface(self.a['actions'])
    def test_opt_in_required_before_core_call(self):
        with patch('app.locdi.dynamic_horizon_evaluation.plan_voi') as planner:
            with self.assertRaisesRegex(ValueError,'OPT_IN'):run_dynamic(self.b,self.cs,self.a,context())
            planner.assert_not_called()
    def test_dynamic_runs_real_planner_and_selector(self):
        r=run_dynamic(self.b,self.cs,self.a,context(),development_opt_in=True)
        self.assertEqual(r['planner_calls'],1);self.assertEqual(r['horizon_policy']['selected_horizon'],3)
        self.assertAlmostEqual(r['plan']['optimal_value'],.972);self.assertGreater(r['planner_evaluated_states'],0)
    def test_fixed_baseline_full_result_same(self):
        for h in (1,2,3):
            a=deepcopy(self.a);a['horizon']=h
            r=run_planner(self.b,self.cs,self.a,h)
            self.assertEqual(r['selection'],select_action(self.cs,selector='bayesian_voi',belief=self.b,artifact=a))
            self.assertEqual(r['plan'],plan_voi(self.b,self.a['actions'],self.a['terminal_utility'],horizon=h))
    def test_zero_info_terminal_core_sentinel(self):
        r=run_dynamic(self.b,[self.cs[-1]],self.a,context(),development_opt_in=True)
        self.assertIsNone(r['horizon_policy']['selected_horizon']);self.assertTrue(r['terminal_only_h1_api_sentinel'])
        self.assertEqual(r['plan']['action_values'],[]);self.assertEqual(r['selection']['selected_action']['type'],'Diagnose')
    def test_invalid_belief_fallback_cannot_execute_core(self):
        with patch('app.locdi.dynamic_horizon_evaluation.plan_voi') as planner:
            with self.assertRaises(ValueError):run_dynamic({'TEST_ONLY_A':.7,'TEST_ONLY_B':.7},self.cs,self.a,context(),development_opt_in=True)
            planner.assert_not_called()
    def test_gold_input_not_executes_fixed_fallback(self):
        ctx=context();ctx['gold']='TEST_ONLY_A'
        with patch('app.locdi.dynamic_horizon_evaluation.plan_voi') as planner:
            with self.assertRaisesRegex(ValueError,'INPUT_BLOCKED'):run_dynamic(self.b,self.cs,self.a,ctx,development_opt_in=True)
            planner.assert_not_called()
    def test_common_horizon_stop_value_is_terminal_not_v3(self):
        p=plan_voi(self.b,self.a['actions'],self.a['terminal_utility'],horizon=3)
        self.assertEqual(common_value({'type':'Diagnose'},p),.5);self.assertNotEqual(p['optimal_value'],.5)
    def test_common_horizon_info_value_matches_action(self):
        p=plan_voi(self.b,self.a['actions'],self.a['terminal_utility'],horizon=3)
        self.assertEqual(common_value({'type':'Test','action_id':self.a['actions'][0]['id']},p),p['action_values'][0]['q_value'])
    def test_near_tie_diagnostic_does_not_change_plan(self):
        p=plan_voi(self.b,self.a['actions'],self.a['terminal_utility'],horizon=2);before=deepcopy(p)
        q_diagnostics(p);self.assertEqual(p,before)
    def test_artifact_and_core_observer_preserved(self):
        from app.core import information_value
        before=deepcopy(self.a);original=information_value.plan_voi
        run_dynamic(self.b,self.cs,self.a,context(),development_opt_in=True)
        self.assertEqual(before,self.a);self.assertIs(information_value.plan_voi,original)
    def test_fingerprint_tamper_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY';p.write_text('TEST_ONLY',encoding='utf-8');pins={str(p):sha(p)}
            p.write_text('TEST_ONLY_CHANGED',encoding='utf-8')
            with self.assertRaises(ValueError):verify_pins(pins)
    def test_patient_clustering_macro_differs_from_micro_and_zero_is_null(self):
        def row(ref,changed):
            return {'case_ref':ref,'results':{'dynamic':{'horizon_policy':{'selected_horizon':2,'fallback_reason':None}}},
                'derived':{'dynamic':{'binding':('Test','TEST_ONLY_T') if changed else ('diagnose',None),'stop':not changed,'q_diagnostics':{'near_information_tie':False}},'H2':{'binding':('diagnose',None),'stop':True}}}
        r=patient_summaries([row('TEST_ONLY_P1',True),row('TEST_ONLY_P1',False),row('TEST_ONLY_P2',False)],['TEST_ONLY_P1','TEST_ONLY_P2','TEST_ONLY_P3'])
        self.assertEqual(r['patients_with_decisions'],2);self.assertEqual(r['patient_macro_action_change_rate_over_active'],.25)
        self.assertEqual(r['decision_micro_action_change_rate'],1/3);self.assertIsNone(r['rows'][2]['action_change_rate'])
    def test_unregistered_patient_rejected(self):
        with self.assertRaisesRegex(ValueError,'UNREGISTERED'):patient_summaries([{'case_ref':'TEST_ONLY_OTHER'}],['TEST_ONLY_P'])
    def test_duplicate_registered_patient_rejected(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):patient_summaries([],['TEST_ONLY_P','TEST_ONLY_P'])
    def test_jsonl_sorted_keys_restore_original_model_order_without_value_change(self):
        a=test_only_voi();a['prior']=dict(reversed(list(a['prior'].items())))
        model=InformationOutcomes(a);obs={};b,m=model.observe(obs)
        stored=json.loads(json.dumps(b,sort_keys=True));self.assertNotEqual(list(stored),list(b))
        restored,_=restore_model_belief(model,obs,stored,m)
        self.assertEqual(restored,stored);self.assertEqual(list(restored),list(b))
    def test_restore_cannot_change_sealed_belief_values(self):
        model=InformationOutcomes(test_only_voi());b,m=model.observe({});b={d:0. for d in b}
        with self.assertRaisesRegex(ValueError,'SOURCE_OBSERVATION'):restore_model_belief(model,{},b,m)
    def test_query_present_does_not_mean_reference_complete(self):
        r={'cases':[{'query_complete':True,'modes':{'selective':{'recall_status':'COMPUTED_COMPLETE_POOL','positive_returned':2,'full_pool_positive':5}}},
                    {'query_complete':True,'modes':{'selective':{'recall_status':'NOT_RUN_INCOMPLETE_OR_NO_POSITIVE_REFERENCE','positive_returned':3,'full_pool_positive':100}}}]}
        result=complete_identity_recall(r);self.assertEqual(result['complete_queries'],1)
        self.assertEqual(result['numerator'],2);self.assertEqual(result['denominator'],5)
    def test_empty_reference_recall_is_not_zero(self):
        r=complete_identity_recall({'cases':[]});self.assertIsNone(r['value'])
    def test_complete_reference_recall_aggregates_counts_not_rates(self):
        r={'cases':[{'modes':{'selective':{'recall_status':'COMPUTED_COMPLETE_POOL','positive_returned':1,'full_pool_positive':2}}},
                    {'modes':{'selective':{'recall_status':'COMPUTED_COMPLETE_POOL','positive_returned':1,'full_pool_positive':8}}}]}
        self.assertEqual(complete_identity_recall(r)['value'],.2)


if __name__=='__main__':unittest.main()
