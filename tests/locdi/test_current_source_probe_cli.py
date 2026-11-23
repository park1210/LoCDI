"""TEST_ONLY regression for actual C4-38 reference CLI protocol mismatch."""
import unittest
from unittest.mock import patch
from test_frozen_m5_runtime_adapter import fixture
from scripts.run_current_source_time_probe import reference_episode
from app.locdi.dynamic_horizon_source_evaluation import run_episode
from app.locdi.source_time_hospital_runtime import run_source_time_visit


class SourceProbeCLI(unittest.TestCase):
    def test_original_plain_callable_failure_reproduced(self):
        m,v=fixture()
        with self.assertRaisesRegex(AttributeError,'snapshot'):
            run_episode(m,v['values'],'voi_h2',1,lambda *a,**k:None)

    def test_corrected_reference_protocol_and_actual_plans(self):
        m,v=fixture();plans=[];checks=[]
        result=reference_episode(m,v['values'],lambda:checks.append(True),plans,budget=3)
        actual=run_source_time_visit(m,v,[],max_actions=3)
        self.assertEqual(result['terminal_choice'],actual['algorithmic_choice'])
        self.assertEqual(result['belief'],actual['belief'])
        self.assertEqual(len(plans),result['resources']['calls'])
        self.assertEqual(len(checks),len(plans)*2)
        for i,p in enumerate(plans):
            a=actual['decisions'][i]['plans'][0]
            self.assertEqual(a['selected_action_id'],p['selected_action_id'])
            self.assertEqual(a['action_values'],p['action_values'])

    def test_resource_failure_never_calls_reference_planner(self):
        m,v=fixture();plans=[]
        def blocked():raise ValueError('TEST_ONLY_RESOURCE_CAP')
        with patch('app.locdi.dynamic_horizon_source_evaluation.plan_voi') as planner:
            with self.assertRaisesRegex(ValueError,'RESOURCE_CAP'):
                reference_episode(m,v['values'],blocked,plans,budget=1)
            planner.assert_not_called()
        self.assertEqual(plans,[])

    def test_zero_budget_keeps_empty_plans_without_fake_actions(self):
        m,v=fixture();plans=[]
        result=reference_episode(m,v['values'],lambda:None,plans,budget=0)
        self.assertEqual(plans,[])
        self.assertEqual(result['releases'],[])
        self.assertEqual(result['belief'],m['prior'])


if __name__=='__main__':unittest.main()
