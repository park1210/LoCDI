"""TEST_ONLY comparator boundaries. No clinical/source experiment execution."""
from copy import deepcopy
import unittest
from app.locdi.source_runtime_comparison import compare_plan,compare_episode,ComparisonMismatch
from scripts.run_current_source_time_probe import reference_episode
from app.locdi.source_time_hospital_runtime import run_source_time_visit
from app.core.voi_planning import plan_voi
from app.core.diagnostic_belief import validation_utility
from test_frozen_m5_runtime_adapter import fixture


class Comparator(unittest.TestCase):
    def setUp(self):
        self.m,self.v=fixture();self.plans=[]
        self.reference=reference_episode(self.m,self.v['values'],lambda:None,self.plans,budget=3)
        self.actual=run_source_time_visit(self.m,self.v,[],max_actions=3)
        self.a=deepcopy(self.actual['decisions'][0]['plans'][0]);self.b=deepcopy(self.a)

    def test_ID_alignment_ignores_only_representation_order(self):
        self.b['action_values'].reverse();r=compare_plan(self.a,self.b)
        self.assertEqual(r['maximum_Q_difference'],0)

    def test_changed_Q_fails_after_reorder(self):
        self.b['action_values'].reverse();self.b['action_values'][0]['q_value']+=1e-8
        with self.assertRaisesRegex(ComparisonMismatch,'Q_MISMATCH'):compare_plan(self.a,self.b)

    def test_changed_VoI_fails(self):
        self.b['action_values'][0]['voi']+=1e-8
        with self.assertRaisesRegex(ComparisonMismatch,'VOI_MISMATCH'):compare_plan(self.a,self.b)

    def test_missing_ID(self):
        del self.b['action_values'][0]['action_id']
        with self.assertRaisesRegex(ComparisonMismatch,'ACTION_ID_MISSING'):compare_plan(self.a,self.b)

    def test_duplicate_before_dictionary(self):
        self.b['action_values'].append(deepcopy(self.b['action_values'][0]))
        with self.assertRaisesRegex(ComparisonMismatch,'DUPLICATE'):compare_plan(self.a,self.b)

    def test_different_ID_set(self):
        self.b['action_values'][0]['action_id']='TEST_ONLY_NEW_ID'
        with self.assertRaisesRegex(ComparisonMismatch,'ID_SET'):compare_plan(self.a,self.b)

    def test_STOP_mismatch(self):
        self.b['selected_action_id']=None if self.a['selected_action_id'] is not None else 'TEST_ONLY'
        with self.assertRaisesRegex(ComparisonMismatch,'STOP_MISMATCH'):compare_plan(self.a,self.b)

    def test_horizon_mismatch(self):
        self.b['horizon']=1 if self.a['horizon']!=1 else 2
        with self.assertRaisesRegex(ComparisonMismatch,'HORIZON'):compare_plan(self.a,self.b)

    def test_budget_mismatch(self):
        self.actual['decisions'][0]['remaining_budget']+=1
        with self.assertRaisesRegex(ComparisonMismatch,'BUDGET'):
            compare_episode(self.actual,self.reference,self.plans,self.m,budget=3)

    def test_actual_fixture_full_episode_and_extra_terminal(self):
        r=compare_episode(self.actual,self.reference,self.plans,self.m,budget=3)
        self.assertEqual(r['status'],'MATCH');self.assertTrue(r['terminal_choice_match'])
        self.assertEqual(r['runtime_decisions']-r['reference_decisions'],r['extra_terminal_decisions'])

    def test_changed_posterior_fails(self):
        k=next(iter(self.actual['belief']));self.actual['belief'][k]+=1e-7
        with self.assertRaisesRegex(ComparisonMismatch,'POSTERIOR'):
            compare_episode(self.actual,self.reference,self.plans,self.m,budget=3)

    def test_terminal_choice_mismatch(self):
        self.actual['algorithmic_choice']='TEST_ONLY_DIFFERENT'
        with self.assertRaisesRegex(ComparisonMismatch,'TERMINAL_CHOICE'):
            compare_episode(self.actual,self.reference,self.plans,self.m,budget=3)

    def test_original_Core_tie_breaking_not_candidate_reorder(self):
        b={'TEST_ONLY_D1':.5,'TEST_ONLY_D2':.5};u=validation_utility(b)
        base={'outcomes':[{'id':'x','likelihood':{'TEST_ONLY_D1':1.,'TEST_ONLY_D2':0.}},
                          {'id':'y','likelihood':{'TEST_ONLY_D1':0.,'TEST_ONLY_D2':1.}}]}
        actions=[dict(deepcopy(base),id='b'),dict(deepcopy(base),id='a')]
        before=deepcopy(actions)
        for horizon in [1,2,3]:
            p=plan_voi(b,actions,u,horizon=horizon)
            q=plan_voi(b,list(reversed(actions)),u,horizon=horizon)
            self.assertEqual(p['selected_action_id'],'a');self.assertEqual(q['selected_action_id'],'a')
            self.assertEqual(compare_plan(p,q)['maximum_Q_difference'],0)
        self.assertEqual(actions,before)


if __name__=='__main__':unittest.main()
