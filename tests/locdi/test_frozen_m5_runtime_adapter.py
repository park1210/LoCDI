"""TEST_ONLY Source rows with unchanged existing M5; no performance evaluation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
from app.locdi.llm_setup import ROOT,digest
from app.locdi.frozen_m5_runtime_adapter import bind_value,ReleasedCategoryView,candidate_case,runtime_readiness
from app.locdi.source_memory_projection import project_past_observations
from app.locdi.voi_information_improvement import discrete_belief,bucket
from app.core.information_outcomes import InformationOutcomes
from app.core.voi_planning import plan_voi
from app.core.information_value import select_action
from app.core.diagnostic_belief import terminal_choice
from app.core.longitudinal_memory import PatientMemoryStore


def fixture():
    model=json.loads((ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json').read_text(encoding='utf-8'))
    visit={'patient_id':'TEST_ONLY_PATIENT','visit_id':'TEST_ONLY_CURRENT','case_id':'TEST_ONLY_CASE',
           'decision_time':'2000-01-02T00:00:00Z','values':{}}
    for i,key in enumerate(model['provenance']['action_key_mapping']):
        kind,system,code,unit,_=json.loads(key)
        number=model['provenance']['binning_thresholds'][key][0]
        raw={'PATIENT':visit['patient_id'],'ENCOUNTER':visit['visit_id'],'CODE':code,'UNITS':unit,
             'VALUE':str(number),'TYPE':'numeric','DATE':visit['decision_time'],'DESCRIPTION':'TEST_ONLY','CATEGORY':'TEST_ONLY'}
        visit['values'][key]={'source_record':raw,'source_record_fingerprint':digest(raw),'data_type':'numeric',
            'value':number,'availability_basis':'RECORDED_TIME_PROXY','available_minute':0 if kind=='Ask' else 4,
            'recorded_at':raw['DATE'],'actual_available_at':None,'event_id':'TEST_ONLY_EVENT_'+str(i)}
    return model,visit


def memory_fixture():
    identity={'patient_id':'TEST_ONLY_PATIENT','visit_id':'TEST_ONLY_CURRENT','started_at':'2000-01-02T00:00:00Z'}
    enc=[(2,{'Id':'TEST_ONLY_PAST','PATIENT':identity['patient_id'],'START':'2000-01-01T00:00:00Z','STOP':'2000-01-01T01:00:00Z'})]
    obs=[(2,{'PATIENT':identity['patient_id'],'ENCOUNTER':'TEST_ONLY_PAST','DATE':'2000-01-01T00:30:00Z',
         'CODE':'TEST_ONLY_CODE','VALUE':'42','UNITS':'TEST_ONLY_UNIT','TYPE':'numeric','DESCRIPTION':'TEST_ONLY','CATEGORY':'TEST_ONLY'})]
    return identity,enc,obs


class FrozenM5RuntimeAdapter(unittest.TestCase):
    def test_exact_eight_bindings_and_model_immutable(self):
        m,v=fixture();before=digest(m)
        self.assertEqual(len([bind_value(m,v,k) for k in v['values']]),8)
        self.assertEqual(digest(m),before)

    def test_units_cannot_be_aliased(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['source_record']['UNITS']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'UNIT_MISMATCH'):bind_value(m,v,k)

    def test_system_cannot_be_inferred(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['source_record']['SYSTEM']='http://loinc.org'
        with self.assertRaisesRegex(ValueError,'SEMANTIC_MISMATCH'):bind_value(m,v,k)

    def test_wrong_patient(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['source_record']['PATIENT']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'SOURCE_IDENTITY'):bind_value(m,v,k)

    def test_source_fingerprint(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['source_record']['DESCRIPTION']='TEST_ONLY_CHANGE'
        with self.assertRaisesRegex(ValueError,'SOURCE_FINGERPRINT'):bind_value(m,v,k)

    def test_source_missing(self):
        m,v=fixture();k=next(iter(v['values']));del v['values'][k]
        with self.assertRaisesRegex(ValueError,'SOURCE_MISSING'):bind_value(m,v,k)

    def test_raw_value_mismatch_and_nonfinite(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['value']=float('nan')
        with self.assertRaisesRegex(ValueError,'SOURCE_VALUE_MISMATCH'):bind_value(m,v,k)

    def test_boundary_equality_and_next_float(self):
        m,v=fixture()
        for key,bounds in m['provenance']['binning_thresholds'].items():
            for bound in bounds:
                self.assertEqual(bucket(bound,bounds),str(sum(bound>x for x in bounds)))
                above=math.nextafter(bound,math.inf)
                self.assertEqual(bucket(above,bounds),str(sum(above>x for x in bounds)))

    def test_time_proxy_preserved(self):
        m,v=fixture();k=next(iter(v['values']));v['values'][k]['actual_available_at']='TEST_ONLY_FAKE'
        with self.assertRaisesRegex(ValueError,'SOURCE_TIME'):bind_value(m,v,k)

    def test_release_requires_selected_action(self):
        m,v=fixture();a=next(iter(m['provenance']['action_key_mapping'].values()));view=ReleasedCategoryView(m,v)
        with self.assertRaisesRegex(ValueError,'BEFORE_SELECTION'):view.release(a,selected='TEST_ONLY_OTHER',minute=4)
        self.assertFalse(view.released)

    def test_future_and_repeat_blocked(self):
        m,v=fixture();a=next(a['id'] for a in m['actions'] if a['type']=='request_test');view=ReleasedCategoryView(m,v)
        with self.assertRaisesRegex(ValueError,'FUTURE'):view.release(a,selected=a,minute=3)
        view.release(a,selected=a,minute=4)
        with self.assertRaisesRegex(ValueError,'REPEATED'):view.release(a,selected=a,minute=4)

    def test_reference_not_in_engine_candidate(self):
        m,v=fixture();v['diagnosis']='TEST_ONLY_PRIVATE_GOLD';v['reference_candidates']=['TEST_ONLY_PRIVATE_GOLD']
        candidate=candidate_case(m,v)
        self.assertNotIn('TEST_ONLY_PRIVATE_GOLD',json.dumps(candidate))
        self.assertEqual(candidate['current_initial_state']['doctor_visible_info'],{})
        self.assertNotIn('resource_profile',candidate)

    def test_categorical_posterior_and_terminal_correspondence(self):
        m,v=fixture();view=ReleasedCategoryView(m,v);keys=list(v['values'])[:3]
        for key in keys:
            a=m['provenance']['action_key_mapping'][key];view.release(a,selected=a,minute=4)
        b,_=InformationOutcomes(m).observe(view.observation());expected,_=discrete_belief(m,v['values'],only=keys)
        self.assertLessEqual(max(abs(b[d]-expected[d]) for d in b),1e-12)
        self.assertEqual(terminal_choice(b,m['terminal_utility'])[0],terminal_choice(expected,m['terminal_utility'])[0])

    def test_planner_q_voi_and_selector_root(self):
        m,v=fixture();view=ReleasedCategoryView(m,v);key=next(iter(v['values']));a=m['provenance']['action_key_mapping'][key]
        view.release(a,selected=a,minute=4);b,_=InformationOutcomes(m).observe(view.observation())
        expected,_=discrete_belief(m,v['values'],only=[key]);remaining=[x for x in m['actions'] if x['id']!=a]
        p=plan_voi(b,remaining,m['terminal_utility'],horizon=2);q=plan_voi(expected,remaining,m['terminal_utility'],horizon=2)
        self.assertEqual(p['selected_action_id'],q['selected_action_id'])
        for x,y in zip(p['action_values'],q['action_values']):
            self.assertEqual(x['action_id'],y['action_id']);self.assertAlmostEqual(x['q_value'],y['q_value'],delta=1e-12)
            self.assertAlmostEqual(x['voi'],y['voi'],delta=1e-12)
        interface=[{'action_id':x['id'],'outcome_model':x['id'],'type':'Ask' if x['type']=='ask_patient' else 'Test',
                    'action':x['type'],'target':x['target']} for x in remaining]
        interface.append({'action_id':'diagnose','outcome_model':None,'type':'Diagnose','action':'diagnose','target':None})
        selection=select_action(interface,selector='bayesian_voi',belief=b,artifact=m)['selected_action']
        self.assertEqual(None if selection['type']=='Diagnose' else selection['action_id'],p['selected_action_id'])

    def test_gate_encoding_and_terminal_not_promoted(self):
        m,_=fixture();r=runtime_readiness(m)
        self.assertFalse(r['hospital_execute_allowed']);self.assertFalse(r['category_view_installed'])
        self.assertEqual(r['artifact_gate']['development_status'],'BLOCKED')
        self.assertEqual(m['provenance']['observation_encoding'],'c4_30_source_discrete_v2')
        self.assertTrue(all(not o['supports_diagnoses'] for a in m['actions'] for o in a['outcomes']))

    def test_actual_frozen_model_sha(self):
        import hashlib
        p=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c')

    def test_memory_source_projection_context_only(self):
        i,e,o=memory_fixture();p=project_past_observations(i,e,o,accept_recorded_time_proxy=True)
        self.assertEqual(p['summary']['records'],1);r=p['past_visits'][0]['records'][0]
        self.assertEqual(r['available_at'],o[0][1]['DATE']);self.assertEqual(json.loads(r['text'])['VALUE'],'42')
        self.assertEqual(len(r),10);self.assertFalse(p['summary']['bayesian_memory_evidence'])
        store=PatientMemoryStore([r]);self.assertEqual(store.retrieve(i,'TEST_ONLY_CODE','selective',5)['selected_count'],1)

    def test_memory_proxy_requires_explicit_scope(self):
        with self.assertRaisesRegex(ValueError,'PROXY'):project_past_observations(*memory_fixture())

    def test_cross_patient_memory_rejected(self):
        i,e,o=memory_fixture();o[0][1]['PATIENT']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'CROSS_PATIENT'):project_past_observations(i,e,o,accept_recorded_time_proxy=True)

    def test_current_and_future_memory_excluded(self):
        i,e,o=memory_fixture();e[0][1]['Id']=i['visit_id'];o[0][1]['ENCOUNTER']=i['visit_id']
        self.assertEqual(project_past_observations(i,e,o,accept_recorded_time_proxy=True)['summary']['records'],0)
        i,e,o=memory_fixture();o[0][1]['DATE']='2000-01-03T00:00:00Z'
        self.assertEqual(project_past_observations(i,e,o,accept_recorded_time_proxy=True)['summary']['records'],0)

    def test_unknown_or_unended_past_visit_not_used(self):
        i,e,o=memory_fixture();e[0][1]['STOP']=''
        self.assertEqual(project_past_observations(i,e,o,accept_recorded_time_proxy=True)['summary']['records'],0)


if __name__=='__main__':unittest.main()
