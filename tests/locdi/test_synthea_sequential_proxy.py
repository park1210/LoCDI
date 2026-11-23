"""TEST_ONLY admissions/clients; real existing Runner/Scheduler, generation0."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator

from app.locdi.synthea_sequential_proxy import (stratified_panel,policy_contract,
    SequentialSyntheticSourceView,validate_permission)
from app.locdi.run_simulation import run_case
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.longitudinal_runner import run_with_recovery
from test_simulation import SourceActionClient,smoke_options

ROOT=Path(__file__).resolve().parents[2]


def fixture_permission(spec):
    return {'scope':'SYNTHETIC_SEQUENTIAL_ER_PILOT_ONLY','plan_fingerprint':'TEST_ONLY_PLAN',
        'policy_fingerprint':spec['fingerprint'],'evidence_type':'TEST_ONLY','user_statement':'TEST_ONLY_NOT_REAL_APPROVAL',
        'accept_source_record_proxy_cutoff':True,'accept_fixed_five_minute_synthetic_turnaround':True,
        'authorize_sequential_source_and_pilot':True,'accept_reference_timing_limits':True,
        'clinical_release_approved':False,'v1_grants_created':0}


class CheckingClient(SourceActionClient):
    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        user_prompt=json.dumps(SequentialSyntheticSourceView.compact_payload(json.loads(user_prompt)),sort_keys=True,ensure_ascii=False)
        result=super().chat_json(system_prompt,user_prompt,model=model,schema=schema)
        Draft202012Validator(schema).validate(json.loads(result))
        return result


class SequentialProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[10]
        allrows=json.loads((ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_COMPLETION_v1/proposed_source_rows_NOT_AUTHORIZED.json').read_text())
        cls.rows=[r for r in allrows if r['patient_ordinal']==1][:1]
        cls.catalog=json.loads((ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY/Episode001.json').read_text())['preexisting_catalog']
    def view(self,rows=None):
        rows=deepcopy(self.rows if rows is None else rows);spec=policy_contract(rows,test_only=True)
        return SequentialSyntheticSourceView(rows,spec=spec,authorization=fixture_permission(spec),
            plan_fingerprint='TEST_ONLY_PLAN',catalog=self.catalog)
    def test_selection_no_outcomes_and_reproducible(self):
        candidates=[{'patient_ordinal':i,'stratum':'endocrinology' if i<=10 else 'neurology'} for i in range(1,21)]
        a=stratified_panel(candidates);self.assertEqual(a,stratified_panel(list(reversed(candidates))))
        self.assertEqual(len([x for x in a if x<=10]),5);self.assertEqual(len([x for x in a if x>10]),5)
        candidates[0]['D4_correct']=True
        with self.assertRaises(ValueError):stratified_panel(candidates)
    def test_missing_permission(self):
        with self.assertRaises(ValueError):validate_permission(None,policy_contract([]),None,test_only=False)
    def test_candidate_compaction_preserves_all_clinical_and_actions(self):
        p={'observation':{'full_memory':'TEST_ONLY_UNTRUNCATED'},'evidence':['TEST_ONLY_VALUE'],
            'candidate_actions':[{'action':'request_test','target':'FULL_TEST_NAME','action_id':'REDUNDANT',
                'type':'Test','cost':None,'outcome_model':None}]}
        before=deepcopy(p);q=SequentialSyntheticSourceView.compact_payload(p)
        self.assertEqual(p,before);self.assertEqual(q['observation'],p['observation'])
        self.assertEqual(q['evidence'],p['evidence']);self.assertEqual(q['candidate_actions'],[{'action':'request_test','target':'FULL_TEST_NAME'}])
    def test_nonnull_voi_metadata_not_stripped(self):
        with self.assertRaises(ValueError):SequentialSyntheticSourceView.compact_payload({'candidate_actions':[{'cost':1}]})
    def test_static_permission_not_reused(self):
        spec=policy_contract(self.rows,test_only=True);a=fixture_permission(spec);a['scope']='SYNTHETIC_STATIC_D2_D4_ONLY'
        with self.assertRaises(ValueError):validate_permission(a,spec,'TEST_ONLY_PLAN',test_only=True)
    def test_test_only_never_live(self):
        with self.assertRaisesRegex(ValueError,'NEVER_LIVE'):
            self.view().apply(adapt_case(self.case),features=smoke_options().runner_options,purpose='development_llm')
    def test_original_case_and_adapter_not_mutated(self):
        original=deepcopy(self.case);a=adapt_case(self.case);before=deepcopy(a.runtime_case)
        runtime,added=self.view().apply(a,features=smoke_options().runner_options,purpose='development_smoke')
        self.assertEqual(a.runtime_case,before);self.assertEqual(self.case,original)
        self.assertEqual(len(runtime['available_tests']),5);self.assertEqual(len(added),1)
        self.assertEqual(runtime['past_visits'],before['past_visits'])
        self.assertFalse(runtime['doctor_visible_info'].get('objective_findings'))
    def test_voi_and_routing_rejected(self):
        for key,value in [('voi_enabled',True),('routing_mode','adaptive'),('memory_mode','full_history')]:
            f=deepcopy(smoke_options().runner_options);f[key]=value
            with self.assertRaises(ValueError):self.view().apply(adapt_case(self.case),features=f,purpose='development_smoke')
    def test_changed_source_hash_rejected(self):
        rows=deepcopy(self.rows);rows[0]['source_event']['evidence']['value']='999'
        with self.assertRaisesRegex(ValueError,'EXACT_ORIGINAL'):
            self.view(rows).apply(adapt_case(self.case),features=smoke_options().runner_options,purpose='development_smoke')
    def test_future_source_rejected(self):
        rows=deepcopy(self.rows);e=rows[0]['source_event'];e['timestamp_upper_epoch_ms']+=60*60*1000
        with self.assertRaises(ValueError):self.view(rows).apply(adapt_case(self.case),features=smoke_options().runner_options,purpose='development_smoke')
    def test_pain_not_promoted(self):
        rows=deepcopy(self.rows);rows[0]['source_event']['target']['code']='72514-3'
        with self.assertRaises(ValueError):self.view(rows).apply(adapt_case(self.case),features=smoke_options().runner_options,purpose='development_smoke')
    def test_wrong_available_offset_rejected(self):
        rows=deepcopy(self.rows);rows[0]['proxy_available_minute']=0
        with self.assertRaisesRegex(ValueError,'OFFSET'):
            self.view(rows).apply(adapt_case(self.case),features=smoke_options().runner_options,purpose='development_smoke')
    def test_real_runner_scheduler_release_and_memory_pair(self):
        for mode in ('none','selective'):
            client=CheckingClient();features=deepcopy(smoke_options().runner_options);features['memory_mode']=mode
            options=smoke_options(client,runner_options=features,diagnosis_catalog=self.catalog,sequential_source_view=self.view())
            with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner, \
                 patch('app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler:
                result=run_case(self.case,options=options)
            self.assertEqual(result['status'],'completed');runner.assert_called_once();scheduler.assert_called_once()
            self.assertTrue(result['mock_execution'])
            name=result['synthetic_source_view_audit'][0]['added_tests'][0]['test_name']
            events=result['execution']['visits'][0]['final_state']['event_log']
            started=[e for e in events if e['event_type']=='test_started' and e['payload'].get('requested_test')==name]
            received=[e for e in events if e['event_type']=='result_received' and e['payload'].get('requested_test')==name]
            self.assertEqual(len(started),1);self.assertEqual(len(received),1)
            self.assertGreaterEqual(received[0]['minute'],started[0]['minute']+5)
            for call in client.calls:
                p=call['payload'];o=p['observation'];has_result=any(x.startswith(name+':') for x in o['objective_findings'])
                if has_result:self.assertGreaterEqual(o['elapsed_minutes'],received[0]['minute'])
                if o['elapsed_minutes']<received[0]['minute']:self.assertFalse(has_result)
                self.assertEqual(p['diagnosis_catalog'],self.catalog)
                self.assertNotIn('evaluation',p);self.assertNotIn('source_event_store',p)
                memory=o.get('longitudinal_memory',{}).get('records',[])
                if mode=='none':self.assertEqual(memory,[])
                else:self.assertLessEqual(len(memory),5)
    def test_missing_source_not_fabricated(self):
        rows=deepcopy(self.rows);rows[0]['source_event']['event_id']='MISSING'
        with self.assertRaises(ValueError):self.view(rows).apply(adapt_case(self.case),features=smoke_options().runner_options,purpose='development_smoke')


if __name__=='__main__':unittest.main()
