"""TEST_ONLY Source fixtures. Written for user execution; not run by Codex."""
from copy import deepcopy
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_frozen_m5_runtime_adapter import fixture
from app.locdi.llm_setup import digest
from app.locdi.m5_common_source import bind_case,CommonM5SourceView,NativeM5Outcomes,NativeM5InformationPolicy
from app.locdi.common_menu_backend import routing_admission,make_common_backend
from app.core.information_outcomes import InformationOutcomes


def source_fixture(folder):
    model,visit=fixture();events=[];raws=[];date='2000-01-02T00:00:30Z'
    lower=int(datetime.fromisoformat(date.replace('Z','+00:00')).timestamp()*1000)
    path=Path(folder)/'TEST_ONLY_observations.csv'
    for key,value in visit['values'].items():
        kind,system,code,unit,_=json.loads(key);raw=deepcopy(value['source_record'])
        raw.update(DATE=date,CATEGORY='survey' if kind=='Ask' else 'laboratory')
        raws.append(raw)
        events.append({'event_id':'TEST_ONLY_'+code,'event_type':'observations','patient_id':visit['patient_id'],
            'encounter_id':visit['visit_id'],'target':{'code':code,'description':raw['DESCRIPTION'],'system':None},
            'evidence':{'value':raw['VALUE'],'units':unit,'kind':'recorded_observation','category':raw['CATEGORY']},
            'timestamp':date,'timestamp_lower_epoch_ms':lower,'timestamp_upper_epoch_ms':lower+999,'timing_precision':'second',
            'source_metadata':{'source_file':str(path),'source_row':len(raws)+1,'source_record':raw}})
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raws[0]));writer.writeheader();writer.writerows(raws)
    case={'case_id':visit['case_id'],'patient_id':visit['patient_id'],'agent_visible':{
        'index_encounter_id':visit['visit_id'],'index_encounter_start':visit['decision_time'],
        'longitudinal':{'patient_id':visit['patient_id'],'visit_id':visit['visit_id'],'started_at':visit['decision_time']}},
        'environment':{'source_event_store':events}}
    return model,case


class CommonMenuConnection(unittest.TestCase):
    def test_exact_eight_raw_bindings_without_model_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d);before=digest((m,c));b=bind_case(c,m)
            self.assertEqual(len(b['rows']),8);self.assertEqual(b['missing_actions'],[])
            self.assertEqual(before,digest((m,c)))

    def test_patient_and_unit_mismatch_remain_missing(self):
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d);c['environment']['source_event_store'][0]['patient_id']='TEST_ONLY_OTHER'
            b=bind_case(c,m);self.assertEqual(len(b['rows']),7);self.assertEqual(len(b['missing_actions']),1)

    def test_future_source_not_selected(self):
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d)
            for e in c['environment']['source_event_store']:
                e['timestamp']='2000-01-02T01:00:00Z';e['source_metadata']['source_record']['DATE']=e['timestamp']
                lower=int(datetime(2000,1,2,1,tzinfo=timezone.utc).timestamp()*1000)
                e['timestamp_lower_epoch_ms']=lower;e['timestamp_upper_epoch_ms']=lower+999
            b=bind_case(c,m);self.assertEqual(b['rows'],[]);self.assertEqual(len(b['missing_actions']),8)

    def test_raw_csv_change_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d);path=Path(c['environment']['source_event_store'][0]['source_metadata']['source_file'])
            path.write_text('WRONG\nRAW\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'RAW_CSV'):bind_case(c,m)

    def test_gold_not_consulted(self):
        class ProtectedCase(dict):
            def __getitem__(self,key):
                if key in ('evaluation','task_annotations'):raise AssertionError('GOLD_ACCESS')
                return super().__getitem__(key)
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d);b=bind_case(ProtectedCase(c),m);self.assertEqual(len(b['rows']),8)

    def test_new_source_authority_required(self):
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d);b=bind_case(c,m)
            with self.assertRaises(ValueError):CommonM5SourceView(b,{})

    def test_pain_stays_ask_and_all_conditions_same_menu(self):
        with tempfile.TemporaryDirectory() as d:
            m,c=source_fixture(d);b=bind_case(c,m)
            a={'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY','source_binding_fingerprint':digest(b),
               'accept_recorded_time_proxy':True,'accept_common_m5_menu':True,'clinical_release_approved':False}
            v=CommonM5SourceView(b,a)
            adapted=SimpleNamespace(runtime_case={'longitudinal':c['agent_visible']['longitudinal']},environment=c['environment'])
            menus=[]
            from app.locdi.current_ablation_runner import features_for
            for condition in ('B0','B1','B2','B3','B4'):
                result,_=v.apply(adapted,features=features_for(condition),purpose='development_llm')
                menus.append(result['current_encounter_events'])
                self.assertEqual(sum(e['action']=='Ask' for e in menus[-1]),1)
            self.assertTrue(all(menus[0]==x for x in menus))

    def binding(self,folder):
        m,c=source_fixture(folder);return m,bind_case(c,m)

    def test_memory_does_not_create_a_likelihood(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);view=NativeM5Outcomes(m,b)
            prior,_=InformationOutcomes(m).observe({})
            got,matched=view.observe({'longitudinal_memory':{'records':[]}})
            self.assertEqual(prior,got);self.assertEqual(matched,[])

    def test_result_requires_actual_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);view=NativeM5Outcomes(m,b);r=next(x for x in b['rows'] if x['kind']=='Test')
            observed={'elapsed_minutes':5,'objective_findings':[r['target']+': '+r['evidence']]}
            with self.assertRaisesRegex(ValueError,'UNRELEASED'):view.observe(observed)

    def test_received_raw_observation_matches_original_m5_posterior(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);before=digest(m);view=NativeM5Outcomes(m,b)
            r=next(x for x in b['rows'] if x['kind']=='Test');text=r['target']+': '+r['evidence']
            a=next(a for a in m['actions'] if a['id']==r['action_id']);o=next(o for o in a['outcomes'] if o['id']==r['outcome_id'])
            expected,_=InformationOutcomes(m).observe({'objective_findings':[o['observation']['text']]})
            for include_ask_history in (True,False):
                with self.subTest(include_ask_history=include_ask_history):
                    # Match the real Runtime's empty history, and separately
                    # verify that a Test-only observation does not inspect Ask.
                    view.runtime=SimpleNamespace(completed_tests=[r['target']],objective_findings=[text],current_time_min=5)
                    if include_ask_history:view.runtime.question_history=[]
                    posterior,matched=view.observe({'elapsed_minutes':5,'objective_findings':[text]})
                    self.assertEqual(posterior,expected);self.assertEqual(len(matched),1);self.assertEqual(digest(m),before)

    def test_claimed_ask_requires_runtime_question_history(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);view=NativeM5Outcomes(m,b);r=next(x for x in b['rows'] if x['kind']=='Ask')
            observed={'elapsed_minutes':5,'history':[{'question':r['target'],'answer':r['evidence'],'answer_status':'known'}]}
            view.runtime=SimpleNamespace(current_time_min=5)
            with self.assertRaisesRegex(ValueError,'ASK_RUNTIME_HISTORY_REQUIRED'):view.observe(observed)
            for history in ([],[{'question':r['target'],'answer':'NOT_SOURCE'}],
                            [{'question':r['target'],'answer':r['evidence'],'answer_status':'unknown'}]):
                with self.subTest(history=history):
                    view.runtime.question_history=history
                    with self.assertRaisesRegex(ValueError,'UNRELEASED'):view.observe(observed)

    def test_ask_requires_matching_source_and_actual_history(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);view=NativeM5Outcomes(m,b);r=next(x for x in b['rows'] if x['kind']=='Ask')
            history=[{'question':r['target'],'answer':r['evidence'],'answer_status':'known'}]
            view.runtime=SimpleNamespace(current_time_min=5,question_history=history)
            posterior,matched=view.observe({'elapsed_minutes':5,'history':history})
            a=next(a for a in m['actions'] if a['id']==r['action_id']);o=next(o for o in a['outcomes'] if o['id']==r['outcome_id'])
            expected,_=InformationOutcomes(m).observe({'history':[{'question':r['target'],
                'answer':o['observation']['text'],'answer_status':'known'}]})
            self.assertEqual(posterior,expected);self.assertEqual(len(matched),1)
            prior,_=InformationOutcomes(m).observe({})
            wrong=[{'question':r['target'],'answer':'NOT_SOURCE','answer_status':'known'}]
            got,matched=view.observe({'elapsed_minutes':5,'history':wrong})
            self.assertEqual(got,prior);self.assertEqual(matched,[])

    def test_wrong_value_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);view=NativeM5Outcomes(m,b);r=next(x for x in b['rows'] if x['kind']=='Test')
            with self.assertRaisesRegex(ValueError,'UNBOUND'):view.observe({'objective_findings':[r['target']+': WRONG']})

    def test_missing_review_stops_before_transport(self):
        with patch('app.locdi.common_menu_backend.BudgetedNativeERClient',side_effect=AssertionError('NO_TRANSPORT')):
            with self.assertRaisesRegex(ValueError,'REVIEWED_ROUTING'):routing_admission(None,{},[],{})

    def test_failed_existing_gate_cannot_be_overridden_by_ready_flag(self):
        e={'training_data':{},'validator_options':{'registry_manifest':{},'allow_test_only':False},'ready':True}
        with patch('app.locdi.common_menu_backend.to_policy_fit_rows',return_value={'report':{'utility_fit_ready':False},'rows':None}):
            with self.assertRaisesRegex(ValueError,'EXISTING_CORRECTNESS'):routing_admission(e,{},[],{})

    def test_no_source_policy_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            m,b=self.binding(d);p=NativeM5InformationPolicy(m,b)
            with patch('app.core.information_value.InformationPolicy.propose',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'NO_SILENT'):p.propose(SimpleNamespace(),{})


if __name__=='__main__':unittest.main()
