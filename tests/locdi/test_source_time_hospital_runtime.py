"""TEST_ONLY runtime-interface equivalence, unchanged empirical M5."""
from copy import deepcopy
import unittest
from test_frozen_m5_runtime_adapter import fixture,memory_fixture
from app.locdi.source_time_hospital_runtime import run_source_time_visit,validate_development_encoding,public_visit,SourceBoundOutcomes
from app.locdi.source_memory_projection import project_past_observations
from app.locdi.frozen_m5_runtime_adapter import candidate_case
from app.locdi.dynamic_horizon_source_evaluation import run_episode,PlannerMeter
from app.locdi.llm_setup import digest
from app.core.diagnostic_belief import terminal_choice


class SourceTimeHospitalRuntime(unittest.TestCase):
    def test_real_runtime_and_policy_vs_existing_source_reference(self):
        model,v=fixture();before=digest(model)
        got=run_source_time_visit(model,v,[],max_actions=3)
        expected=run_episode(model,v['values'],'voi_h2',3,PlannerMeter())
        self.assertEqual(got['status'],'completed');self.assertTrue(got['hospital_unresolved'])
        self.assertIsNone(got['hospital_supported_diagnosis'])
        self.assertEqual(got['algorithmic_choice'],expected['terminal_choice'])
        self.assertLessEqual(max(abs(got['belief'][d]-expected['belief'][d]) for d in got['belief']),1e-12)
        releases=[e for e in got['actions'] if 'action_id' in e]
        self.assertEqual([r['action_id'] for r in releases],[r['action_id'] for r in expected['releases']])
        self.assertEqual([r['outcome_id'] for r in releases],[r['outcome_id'] for r in expected['releases']])
        self.assertTrue(all(e['released_at_minute']>=e['available_minute'] for e in releases))
        self.assertEqual(before,digest(model))

    def test_horizon_remaining_budget_clipped_without_artifact_change(self):
        m,v=fixture();result=run_source_time_visit(m,v,[],max_actions=1)
        self.assertEqual(result['decisions'][0]['plans'][0]['horizon'],1)
        self.assertEqual(m['horizon'],2)

    def test_zero_budget_terminal_and_no_fake_source(self):
        m,v=fixture();r=run_source_time_visit(m,v,[],max_actions=0)
        self.assertEqual(len(r['actions']),1);self.assertEqual(r['actions'][0]['kind'],'terminal_unresolved')
        self.assertEqual(r['belief'],m['prior']);self.assertEqual(r['LLM_generation'],0)

    def test_hidden_reference_removed_from_runtime(self):
        m,v=fixture();v['diagnosis']='TEST_ONLY_PRIVATE_REFERENCE'
        self.assertNotIn('diagnosis',public_visit(v))
        c=candidate_case(m,v);bad=deepcopy(c);bad['evaluation']={'gold':'TEST_ONLY'}
        with self.assertRaises(ValueError):validate_development_encoding(m,v,bad)

    def test_exact_semantic_source_gate_no_string_only_permission(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['source_record']['UNITS']='OTHER'
        with self.assertRaises(ValueError):run_source_time_visit(m,v,[],max_actions=1)

    def test_raw_unknown_or_unreleased_observation_rejected(self):
        m,v=fixture();view=SourceBoundOutcomes(m,v);view.events=candidate_case(m,v)['current_encounter_events']
        with self.assertRaisesRegex(ValueError,'UNRELEASED'):view.observe({'objective_findings':['TEST_ONLY_FUTURE']})

    def test_memory_context_without_new_likelihood(self):
        m,v=fixture();i,e,o=memory_fixture();i['visit_id']=v['visit_id'];i['patient_id']=v['patient_id']
        e[0][1]['PATIENT']=v['patient_id'];o[0][1]['PATIENT']=v['patient_id']
        memory=project_past_observations(i,e,o,accept_recorded_time_proxy=True)['past_visits']
        a=run_source_time_visit(m,v,memory,max_actions=2);b=run_source_time_visit(m,v,[],max_actions=2)
        self.assertEqual(a['belief'],b['belief'])
        self.assertFalse(any(x['type']=='retrieve_memory' for x in m['actions']))

    def test_scope_never_marks_research_ready(self):
        m,v=fixture();r=run_source_time_visit(m,v,[],max_actions=0)
        self.assertFalse(r['development_gate']['research_evaluation_ready'])
        self.assertIn('not_resource_simulation',r['scheduler_mode'])

    def test_source_support_cannot_be_fabricated(self):
        m,v=fixture();m['actions'][0]['outcomes'][0]['supports_diagnoses']=[next(iter(m['diagnoses']))]
        with self.assertRaisesRegex(ValueError,'SUPPORT'):run_source_time_visit(m,v,[],max_actions=0)


if __name__=='__main__':unittest.main()
