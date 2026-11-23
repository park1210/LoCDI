"""C4-32 offline TEST_ONLY boundary tests and read-only frozen Source contracts."""
from copy import deepcopy
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch
from app.locdi.dynamic_horizon import DynamicHorizonPolicy,policy_spec
from app.locdi.dynamic_horizon_evaluation import choose_dynamic
from app.locdi.dynamic_horizon_source_evaluation import *
from app.locdi.voi_sequential_policy_evaluation import episode
from app.locdi.voi_source_replay_environment import SourceReplayEnvironment
from scripts.audit_research_gaps import read,ROOT


class DynamicSourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.evaluate_voi_sequential_policies import load_panel
        cls.inputs,cls.identity,cls.truth,cls.models,cls.values,cls.verified=load_panel()

    def setUp(self):
        self.m=deepcopy(self.models[1,'M5']);self.v=deepcopy(self.values[1]);self.ids=[a['id'] for a in self.m['actions']]

    def test_frozen_spec_matches_c419(self):
        self.assertEqual(policy_spec(),read(ROOT/'outputs/research_evaluation/c4_19_v1/dynamic_horizon_policy_spec.json'))

    def test_policy_fingerprint(self):
        self.assertEqual(DynamicHorizonPolicy().fingerprint,digest({'spec':policy_spec(),'variant':'entropy_margin_feasibility'}))

    def test_92_736_and_fold_sizes(self):
        self.assertEqual((len(self.inputs),self.verified['verified_source_outcomes']),(92,736))
        self.assertEqual(list(self.verified['fold_visits'].values()),[21,12,23,19,17])

    def test_fold_train_patient_disjoint(self):
        for p in self.inputs:self.assertNotIn(self.identity[p['scenario_ordinal']]['patient_id'],self.models[p['fold'],'M5']['provenance']['training_patients'])

    def test_three_horizons_separate(self):
        s=horizon_selection(self.m['prior'],self.m,self.ids,1)
        self.assertEqual((s['requested_horizon'],s['selected_horizon'],s['effective_horizon']),(3,3,1));self.assertTrue(s['budget_cap']);self.assertFalse(s['action_cap'])

    def test_action_cap_distinct(self):
        s=horizon_selection(self.m['prior'],self.m,self.ids[:2],5)
        self.assertEqual((s['requested_horizon'],s['selected_horizon'],s['effective_horizon']),(3,2,2));self.assertTrue(s['action_cap']);self.assertFalse(s['budget_cap'])

    def test_zero_budget_no_planner(self):
        meter=PlannerMeter();r=run_episode(self.m,self.v,'dynamic',0,meter)
        self.assertEqual(meter.stats['calls'],0);self.assertEqual(r['actual_actions'],0);self.assertEqual(r['terminal_only_reason'],'BUDGET_ZERO')

    def test_empty_actions_terminal(self):
        s=horizon_selection(self.m['prior'],self.m,[],5)
        self.assertIsNone(s['selected_horizon']);self.assertEqual(s['effective_horizon'],0);self.assertFalse(s['actual_planner_execution'])

    def test_high_confidence_requests_H1_not_budget(self):
        b={d:(.99 if i==0 else .01/14) for i,d in enumerate(self.m['prior'])}
        s=horizon_selection(b,self.m,self.ids,5)
        self.assertEqual(s['requested_horizon'],1);self.assertFalse(s['budget_cap'])

    def test_medium_features_H2(self):
        b={d:(.6 if i==0 else .2 if i==1 else .2/13) for i,d in enumerate(self.m['prior'])}
        s=horizon_selection(b,self.m,self.ids,5);self.assertEqual(s['requested_horizon'],2)

    def test_uniform_H3(self):
        b={d:1/15 for d in self.m['prior']};self.assertEqual(horizon_selection(b,self.m,self.ids,5)['requested_horizon'],3)

    def test_thresholds_unchanged(self):self.assertEqual(policy_spec()['thresholds'],{'low':1/3,'high':2/3})

    def test_missing_context_original_fallback(self):
        a=self.m['actions'][0];interface=[{'action_id':a['id'],'type':'Ask','action':a['type'],'target':a['target'],'outcome_model':a['id']}]
        s=choose_dynamic(self.m['prior'],interface,None,development_opt_in=True)
        self.assertEqual(s['eligibility_status'],'FALLBACK');self.assertEqual(s['selected_horizon'],1)

    def test_explicit_opt_in_original_adapter(self):
        with self.assertRaises(ValueError):choose_dynamic(self.m['prior'],[],{})

    def test_gold_unsafe_not_fallback(self):
        b={**self.m['prior'],'gold':0};s=DynamicHorizonPolicy().select(belief=b,available_actions=[],observable_context={})
        self.assertEqual(s['eligibility_status'],'BLOCKED');self.assertIsNone(s['fallback_reason'])

    def test_no_source_reference_selector_arguments(self):
        self.assertEqual(set(inspect.signature(horizon_selection).parameters),{'belief','model','remaining','remaining_budget'})

    def test_interface_has_no_outcomes(self):
        s=horizon_selection(self.m['prior'],self.m,self.ids,5)
        self.assertEqual(s['selection']['eligibility_status'],'ELIGIBLE')

    def test_invalid_remaining_action(self):
        with self.assertRaises(ValueError):horizon_selection(self.m['prior'],self.m,['hidden'],1)

    def test_duplicate_remaining(self):
        with self.assertRaises(ValueError):horizon_selection(self.m['prior'],self.m,[self.ids[0]]*2,1)

    def test_native_stop_preserved(self):
        m=deepcopy(self.m)
        for a in m['actions']:
            for o in a['outcomes']:o['likelihood']={d:1/3 for d in m['prior']}
        r=run_episode(m,self.v,'dynamic',2,PlannerMeter());self.assertEqual(r['actual_actions'],0);self.assertEqual(r['stop_reason'],'NATIVE_STOP')

    def test_posterior_and_source_runner_reused(self):
        a=run_episode(self.m,self.v,'voi_h1',1,PlannerMeter());b=episode(self.m,self.v,'voi_h1',1,PlannerMeter(),[])
        self.assertEqual(normalized_trace(a),normalized_trace(b))

    def test_fixed_three_horizons_identical_parent_case(self):
        old=ROOT/'outputs/private/research_evaluation/c4_31_v1/episode_trace.jsonl'
        from scripts.audit_research_gaps import rows
        ss={r['policy']:r for r in rows(old) if r['section']=='native' and r['scenario_ordinal']==1 and r['budget']==2 and r['policy'] in POLICIES[:3]}
        for p in POLICIES[:3]:self.assertEqual(normalized_trace(run_episode(self.m,self.v,p,2,PlannerMeter())),normalized_trace(ss[p]))

    def test_future_outcome_does_not_choose_depth(self):
        s=horizon_selection(self.m['prior'],self.m,self.ids,5);v=deepcopy(self.v);v.clear()
        self.assertEqual(s['requested_horizon'],horizon_selection(self.m['prior'],self.m,self.ids,5)['requested_horizon'])

    def test_missing_source_blocks(self):
        with self.assertRaisesRegex(ValueError,'MISSING_SOURCE'):run_episode(self.m,{},'dynamic',1,PlannerMeter())

    def test_source_clock(self):
        for v in self.v.values():v['available_minute']=45
        r=run_episode(self.m,self.v,'dynamic',1,PlannerMeter());self.assertTrue(all(e['released_at_minute']>=45 for e in r['releases']))

    def test_independent_budgets(self):
        a=run_episode(self.m,self.v,'dynamic',1,PlannerMeter());b=run_episode(self.m,self.v,'dynamic',2,PlannerMeter())
        self.assertEqual(a['decisions'][0]['belief_before'],b['decisions'][0]['belief_before']);self.assertEqual(a['decisions'][0]['effective_horizon'],1)
        self.assertEqual(b['decisions'][0]['effective_horizon'],2)

    def test_no_repeat_and_budget(self):
        r=run_episode(self.m,self.v,'dynamic',3,PlannerMeter());ids=[e['action_id'] for e in r['releases']]
        self.assertEqual(len(ids),len(set(ids)));self.assertLessEqual(len(ids),3)

    def test_cache_results_identical(self):
        a=run_episode(self.m,self.v,'dynamic',2,PlannerMeter(cached=False));meter=PlannerMeter(cached=True)
        b=run_episode(self.m,self.v,'dynamic',2,meter);c=run_episode(self.m,self.v,'dynamic',2,meter)
        self.assertEqual(normalized_trace(a),normalized_trace(b));self.assertEqual(normalized_trace(b),normalized_trace(c));self.assertGreater(meter.stats['hits'],0)

    def test_separate_policy_caches(self):
        a=PlannerMeter(cached=True);b=PlannerMeter(cached=True);run_episode(self.m,self.v,'voi_h1',1,a)
        self.assertFalse(b.cache);self.assertEqual(b.stats['calls'],0)

    def test_logical_states_same_cached_uncached(self):
        a=PlannerMeter(cached=False);b=PlannerMeter(cached=True)
        for _ in range(2):run_episode(self.m,self.v,'voi_h1',1,a);run_episode(self.m,self.v,'voi_h1',1,b)
        self.assertEqual(a.stats['logical_states'],b.stats['logical_states']);self.assertLess(b.stats['executed_states'],a.stats['executed_states'])

    def test_compute_budget_blocks(self):
        m=PlannerMeter(budget=ComputeBudget(max_calls=0))
        with self.assertRaisesRegex(ValueError,'COMPUTE'):run_episode(self.m,self.v,'dynamic',1,m)

    def test_core_max_states_not_increased(self):
        with self.assertRaises(ValueError):PlannerMeter()(self.m['prior'],self.m['actions'],self.m['terminal_utility'],max_states=20001)

    def test_paired_denominator_refusal(self):
        from scripts.analyze_voi_sequential_policies import paired_comparison
        with self.assertRaises(ValueError):paired_comparison([],[],self.identity)

    def test_patient_bootstrap_reproducible(self):
        from scripts.analyze_voi_sequential_policies import paired_comparison
        r=run_episode(self.m,self.v,'dynamic',1,PlannerMeter());score_episode(r,self.truth[1]);r['scenario_ordinal']=1
        self.assertEqual(paired_comparison([r],[r],self.identity,repetitions=10),paired_comparison([r],[r],self.identity,repetitions=10))

    def test_namespace_existing_refusal(self):
        from scripts.evaluate_dynamic_horizon_source_policy import prepare
        with self.assertRaisesRegex(ValueError,'NO_OVERWRITE'):prepare('c4_31_v1')

    def test_private_input_preserved(self):
        before=deepcopy(self.v);m=deepcopy(self.m);run_episode(self.m,self.v,'dynamic',1,PlannerMeter());self.assertEqual(before,self.v);self.assertEqual(m,self.m)


if __name__=='__main__':unittest.main()
