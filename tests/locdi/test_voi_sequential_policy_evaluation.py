"""TEST_ONLY nonclinical contracts; real frozen panel tests are read-only."""
from copy import deepcopy
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.diagnostic_belief import posterior, validation_utility
from app.core.voi_planning import plan_voi
from app.core.information_value import select_action
from app.locdi.voi_information_improvement import bucket
from app.locdi.voi_sequential_policy_evaluation import *
from app.locdi.voi_source_replay_environment import SourceReplayEnvironment


class SequentialContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.evaluate_voi_sequential_policies import load_panel
        cls.inputs,cls.identity,cls.truth,cls.models,cls.values,cls.verification=load_panel()

    def setUp(self):
        self.p=self.inputs[0];self.model=deepcopy(self.models[1,'M5']);self.vals=deepcopy(self.values[1]);self.cache=PlannerCache()
        self.order=global_eig_order(self.model,self.cache)

    def run_episode(self,p='voi_h1',k=2,**kw):
        return episode(self.model,self.vals,p,k,self.cache,self.order,**kw)

    def test_panel_92_63(self):
        self.assertEqual((self.verification['visits'],self.verification['patients'],self.verification['verified_source_outcomes']),(92,63,736))

    def test_same_fold_patient_never_train(self):
        for p in self.inputs:
            i=p['scenario_ordinal'];self.assertNotIn(self.identity[i]['patient_id'],self.models[p['fold'],'M5']['provenance']['training_patients'])

    def test_serialized_threshold_strict_boundary(self):
        for bs in self.model['provenance']['binning_thresholds'].values():
            for b in bs:
                self.assertEqual(bucket(b,bs),str(sum(b>x for x in bs)))
                self.assertGreater(int(bucket(math.nextafter(b,math.inf),bs)),int(bucket(b,bs)))

    def test_environment_selected_only(self):
        e=SourceReplayEnvironment(self.model,self.vals,2);a=self.order[0];r=e.acquire(a)
        self.assertEqual(set(r),{'action_id','outcome_id','released_at_minute'});self.assertEqual(len(e.releases),1)

    def test_repeat_rejected(self):
        e=SourceReplayEnvironment(self.model,self.vals,2);e.acquire(self.order[0])
        with self.assertRaisesRegex(ValueError,'REPEATED'):e.acquire(self.order[0])

    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):SourceReplayEnvironment(self.model,self.vals,2).acquire('invalid')

    def test_missing_not_sampled(self):
        self.vals.clear()
        with self.assertRaisesRegex(ValueError,'MISSING'):self.run_episode('fixed',1)

    def test_budget_zero_never_releases(self):
        for p in POLICIES:
            if p!='random':self.assertEqual(self.run_episode(p,0)['actual_actions'],0)

    def test_budget_cap(self):
        e=SourceReplayEnvironment(self.model,self.vals,1);e.acquire(self.order[0])
        with self.assertRaisesRegex(ValueError,'BUDGET'):e.acquire(self.order[1])

    def test_bad_budget_types(self):
        for k in (-1,6,True,1.5):
            with self.assertRaises(ValueError):SourceReplayEnvironment(self.model,self.vals,k)

    def test_future_clock_advance(self):
        for v in self.vals.values():v['available_minute']=45
        r=self.run_episode('fixed',2)
        self.assertEqual(r['elapsed_proxy_minutes'],45)
        self.assertTrue(all(e['released_at_minute']>=e['available_minute'] for e in r['releases']))

    def test_unreviewed_release_basis_rejected(self):
        for v in self.vals.values():v['availability_basis']='UNREVIEWED'
        with self.assertRaises(ValueError):self.run_episode('fixed',1)

    def test_invalid_time(self):
        for v in self.vals.values():v['available_minute']=float('nan')
        with self.assertRaises(ValueError):self.run_episode('fixed',1)

    def test_policy_signature_no_source_or_gold(self):
        names=set(inspect.signature(choose).parameters)
        self.assertFalse(names & {'values','reference','patient_id','source','environment','scorer'})

    def test_scorer_does_not_change_trajectory(self):
        a=self.run_episode('fixed',2);before=deepcopy(a['decisions']);score_episode(a,list(self.model['prior'])[0]);self.assertEqual([r['selected_action'] for r in before],[r['selected_action'] for r in a['decisions']])

    def test_future_value_cannot_change_first_choice(self):
        a=self.run_episode('voi_h1',1);self.vals={k:{**v,'value':v['value']+1000 if v['data_type']=='numeric' else v['value']} for k,v in self.vals.items()};b=self.run_episode('voi_h1',1)
        self.assertEqual(a['decisions'][0]['selected_action'],b['decisions'][0]['selected_action'])

    def test_independent_budget_states(self):
        a=self.run_episode('voi_h3',1);b=self.run_episode('voi_h3',3)
        self.assertEqual(a['decisions'][0]['belief_before'],b['decisions'][0]['belief_before'])
        self.assertEqual(a['decisions'][0]['effective_horizon'],1);self.assertEqual(b['decisions'][0]['effective_horizon'],3)

    def test_horizon_receding_cap(self):
        r=self.run_episode('voi_h3',2,fixed_k=True)
        self.assertEqual([d['effective_horizon'] for d in r['decisions']],[2,1])

    def test_native_planner_selection(self):
        for h in (1,2,3):
            selected,p,e=choose('voi_h'+str(h),self.model['prior'],self.model,[a['id'] for a in self.model['actions']],3,canonical_order(self.model),self.order,self.cache)
            self.assertEqual(selected,plan_voi(self.model['prior'],self.model['actions'],self.model['terminal_utility'],horizon=h)['selected_action_id'])

    def test_fixed_k_does_not_mutate_native_stop(self):
        m=deepcopy(self.model)
        for a in m['actions']:
            for o in a['outcomes']:o['likelihood']={d:1/3 for d in m['prior']}
        native=episode(m,self.vals,'voi_h1',2,self.cache,self.order)
        probe=episode(m,self.vals,'voi_h1',2,self.cache,self.order,fixed_k=True)
        self.assertEqual(native['actual_actions'],0);self.assertEqual(probe['actual_actions'],2)
        self.assertTrue(all(d['root_stop'] for d in probe['decisions']))
        self.assertNotEqual(native['mode'],probe['mode'])

    def test_ig_existing_selector(self):
        chosen,_,_=choose('information_gain',self.model['prior'],self.model,[a['id'] for a in self.model['actions']],2,[],[],self.cache)
        p=plan_voi(self.model['prior'],self.model['actions'],self.model['terminal_utility'])
        expected=sorted(p['action_values'],key=lambda r:(-r['information_gain_nats'],r['action_id']))[0]['action_id']
        self.assertEqual(chosen,expected)

    def test_global_order_no_validation(self):
        order=global_eig_order(self.model,self.cache)
        self.assertEqual(order,self.order);self.assertEqual(set(order),{a['id'] for a in self.model['actions']})

    def test_fixed_order_canonical_keys(self):
        self.assertEqual(canonical_order(self.model),self.p['fixed_action_order'])

    def test_cache_equal_uncached(self):
        args=(self.model['prior'],self.model['actions'],self.model['terminal_utility'])
        a=self.cache(*args,horizon=2);b=self.cache(*args,horizon=2);c=plan_voi(*args,horizon=2)
        self.assertEqual(a,c);self.assertEqual(b,c);self.assertGreater(self.cache.hits,0)

    def test_cache_mutation_isolated(self):
        args=(self.model['prior'],self.model['actions'],self.model['terminal_utility'])
        a=self.cache(*args);a['action_values'].clear();self.assertTrue(self.cache(*args)['action_values'])

    def test_cache_distinguishes_horizons(self):
        args=(self.model['prior'],self.model['actions'],self.model['terminal_utility'])
        self.cache(*args,horizon=2);n=self.cache.misses;self.cache(*args,horizon=3);self.assertEqual(self.cache.misses,n+1)

    def test_cache_limits_fail_closed(self):
        c=PlannerCache(max_calls=0)
        with self.assertRaisesRegex(ValueError,'RESOURCE'):c(self.model['prior'],self.model['actions'],self.model['terminal_utility'])

    def test_random_preregistered_count(self):
        r=random_expectation(self.model,self.vals,2,self.truth[1],self.cache,self.order)
        self.assertEqual(r['enumeration_paths'],32);self.assertEqual(r['actual_actions'],2)

    def test_random_repeat_identical(self):
        a=random_expectation(self.model,self.vals,1,self.truth[1],self.cache,self.order);b=random_expectation(self.model,self.vals,1,self.truth[1],self.cache,self.order)
        self.assertEqual(a,b)

    def test_random_is_mean_scores(self):
        paths=[];r=random_expectation(self.model,self.vals,1,self.truth[1],self.cache,self.order,paths.append)
        self.assertEqual(r['score']['nll'],sum(p['score']['nll'] for p in paths)/32)

    def test_no_repeated_actions_all_native(self):
        for p in POLICIES:
            if p=='random':continue
            r=self.run_episode(p,3);ids=[x['action_id'] for x in r['releases']];self.assertEqual(len(ids),len(set(ids)))

    def test_priors_same_all_policies(self):
        for p in POLICIES:
            if p in ('prior','random'):continue
            r=self.run_episode(p,1);self.assertEqual(r['decisions'][0]['belief_before'],self.model['prior'])

    def test_failure_not_imputed(self):
        rs=[{'scenario_ordinal':1,'status':'failed'}];m=mean_metrics(rs,self.identity)
        self.assertEqual(m['failed_visits'],1);self.assertIsNone(m['visit_macro']['top1'])

    def test_paired_missing_blocks(self):
        r=score_episode(self.run_episode('fixed',0),self.truth[1]);r['scenario_ordinal']=1
        with self.assertRaises(ValueError):paired_comparison([r],[],self.identity)

    def test_bootstrap_reproducible(self):
        a=score_episode(self.run_episode('fixed',1),self.truth[1]);a['scenario_ordinal']=1
        b=score_episode(self.run_episode('fixed',0),self.truth[1]);b['scenario_ordinal']=1
        self.assertEqual(paired_comparison([a],[b],self.identity,30),paired_comparison([a],[b],self.identity,30))

    def test_fractional_random_joint_counts(self):
        a=score_episode(self.run_episode('fixed',1),self.truth[1]);a['scenario_ordinal']=1
        b=deepcopy(a);b['score']['top1']=.25
        r=paired_comparison([a],[b],self.identity,10)
        self.assertAlmostEqual(sum(r['correctness_joint_counts'].values()),1)

    def test_aubc_requires_all_budgets(self):
        from scripts.evaluate_voi_sequential_policies import aubc_rows
        r=score_episode(self.run_episode('fixed',1),self.truth[1]);r['scenario_ordinal']=1
        with self.assertRaises(ValueError):aubc_rows([r],'fixed')

    def test_bad_namespace(self):
        from scripts.evaluate_voi_sequential_policies import places
        with self.assertRaises(ValueError):places('../escape')

    def test_old_namespace_rejected(self):
        from scripts.evaluate_voi_sequential_policies import prepare
        with self.assertRaisesRegex(ValueError,'ALREADY_USED'):prepare('c4_30_v2')

    def test_m1_alias_original_likelihood(self):
        from scripts.evaluate_voi_sequential_policies import OLD,read
        original=read(OLD/'fold01_probability_artifact.json');view=self.models[1,'M1']
        self.assertEqual(original['prior'],view['prior'])
        for a,b in zip(original['actions'],view['actions']):
            for x,y in zip(a['outcomes'],b['outcomes']):self.assertEqual(x['likelihood'],y['likelihood'])

    def test_raw_values_unchanged(self):
        before=deepcopy(self.vals);self.run_episode('fixed',5);self.assertEqual(before,self.vals)

    def test_posterior_order_invariance(self):
        ids=self.order[:3];a=self.run_episode('random',3,forced_order=ids);b=self.run_episode('random',3,forced_order=list(reversed(ids)))
        self.assertEqual(a['terminal_choice'],b['terminal_choice'])
        for d in a['belief']:self.assertAlmostEqual(a['belief'][d],b['belief'][d],places=12)


if __name__=='__main__':unittest.main()
