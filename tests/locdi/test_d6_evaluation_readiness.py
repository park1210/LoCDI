"""TEST_ONLY grades/points; no real clinical evaluation or model requests."""
from copy import deepcopy
import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from app.locdi.llm_setup import ROOT
from app.locdi.healthbench_grader import (TestOnlyGrader,run_grading,parse_grade,adapted_score,Ledger,grader_messages,GRADER_CONFIG)
from app.locdi.dataset_release_readiness import readiness,dependency_inventory,split_audit
from test_d4_reference_acquisition import fixture


class FakeGrader(TestOnlyGrader):
    def __init__(self,mode=None):self.mode=mode;self.calls=[];self.identity_calls=0
    def identity(self):
        self.identity_calls+=1
        return {'model':'TEST_ONLY_GRADER','version':'TEST_ONLY_v1','digest':'TEST_ONLY_NO_MODEL_ARTIFACT' if self.mode!='identity' else 'TEST_ONLY_WRONG'}
    def grade(self,messages,*,model,timeout):
        self.calls.append(deepcopy(messages))
        if self.mode=='timeout':raise TimeoutError('TEST_ONLY_PRIVATE_ERROR')
        if self.mode=='model_error':raise ConnectionError('TEST_ONLY_PRIVATE_ERROR')
        if self.mode=='partial' and len(self.calls)>1:return 'invalid TEST_ONLY'
        if self.mode=='invalid':return 'invalid TEST_ONLY'
        if self.mode=='missing':return '{"criteria_met": true}'
        return json.dumps({'criteria_met':True,'explanation':'TEST_ONLY protocol response, not a clinical judgment'})


def rows_with_negative():
    row=fixture();row['rubrics'].append({'criterion':'TEST_ONLY_NEGATIVE','points':-3,'tags':[]})
    return [row]


class D6BoundedGraderContracts(unittest.TestCase):
    def run_fake(self,mode=None,**kwargs):
        client=FakeGrader(mode)
        result=run_grading(rows_with_negative(),['TEST_ONLY_COMPLETION'],client=client,execute=True,test_only=True,**kwargs)
        return result,client

    def test_default_dry_run_and_real_execution_approval_cannot_call_client(self):
        client=FakeGrader()
        result=run_grading(rows_with_negative(),['TEST_ONLY_COMPLETION'],client=client)
        self.assertEqual(result['status'],'DRY_RUN');self.assertEqual(client.calls,[]);self.assertEqual(client.identity_calls,0)
        for allow in (False,True):
            result=run_grading(rows_with_negative(),['TEST_ONLY_COMPLETION'],client=client,execute=True,allow_grader=allow)
            self.assertEqual(result['status'],'BLOCKED');self.assertEqual(client.calls,[])

    def test_all_rubrics_and_negative_points_preserved_no_official_score(self):
        result,client=self.run_fake()
        self.assertEqual(result['status'],'TEST_ONLY_COMPLETED');self.assertEqual(len(client.calls),2)
        self.assertEqual([r['points'] for r in result['records']],[2,-3]);self.assertEqual(result['rubrics_recorded'],2)
        self.assertIsNone(result['official_healthbench_score']);self.assertEqual(result['clinical_gold_generated'],0)
        self.assertEqual(result['actual_grader_calls'],0);self.assertEqual(result['actual_generation_calls'],0)

    def test_parser_bool_explanation_and_markdown_contract(self):
        self.assertTrue(parse_grade('```json\n{"criteria_met": true, "explanation": "TEST_ONLY"}\n```')['criteria_met'])
        for text in ('[]','{"criteria_met": 1,"explanation":"TEST_ONLY"}','{"criteria_met":true}',
                     '{"criteria_met": true, "criteria_met": false,"explanation":"TEST_ONLY"}',
                     '{"criteria_met": true,"explanation":"TEST_ONLY","extra":1}'):
            with self.assertRaises(ValueError):parse_grade(text)

    def test_invalid_json_missing_fields_timeout_and_model_errors_finitely_stop(self):
        for mode,expected in (('invalid','INVALID_JSON_OR_SCHEMA'),('missing','INVALID_JSON_OR_SCHEMA'),('timeout','TIMEOUT'),('model_error','MODEL_OR_STORAGE_ERROR')):
            result,client=self.run_fake(mode,max_attempts=2)
            self.assertEqual(result['status'],'FAILED');self.assertEqual(len(client.calls),2)
            self.assertEqual(result['records'][0]['attempts'][-1]['status'],expected)
            self.assertEqual(result['blocker_codes'],['RUBRIC_MAX_ATTEMPTS_REACHED'])
            self.assertNotIn('TEST_ONLY_PRIVATE_ERROR',json.dumps(result))

    def test_partial_success_not_scored_and_not_changed_to_incorrect(self):
        result,client=self.run_fake('partial')
        self.assertEqual(result['status'],'PARTIAL');self.assertEqual(result['rubrics_recorded'],1)
        self.assertEqual(result['unrecorded_rubrics'],1);self.assertIsNone(result['official_healthbench_score'])
        self.assertNotIn('criteria_met',result['records'][1])

    def test_global_and_per_sample_ledger_checks_before_each_request(self):
        ledger=Ledger(global_cap=3,per_sample_cap=10,reserved_generation=1)
        ledger.reserve(1);ledger.reserve(1)
        with self.assertRaisesRegex(RuntimeError,'GLOBAL_REQUEST_BUDGET_EXHAUSTED'):ledger.reserve(1)
        self.assertEqual(ledger.grading_requests,2)
        ledger=Ledger(global_cap=80,per_sample_cap=2,reserved_generation=1);ledger.reserve(1)
        with self.assertRaisesRegex(RuntimeError,'PER_SAMPLE_BUDGET_EXHAUSTED'):ledger.reserve(1)

    def test_planned_budget_and_invalid_options_block_without_calls(self):
        result,client=self.run_fake(global_cap=2);self.assertEqual(result['status'],'BLOCKED');self.assertEqual(client.calls,[])
        result,client=self.run_fake(per_sample_cap=2);self.assertEqual(result['status'],'BLOCKED');self.assertEqual(client.calls,[])
        for value in (0,4,True):
            with self.assertRaises(ValueError):self.run_fake(max_attempts=value)

    def test_identity_mismatch_stops_before_grading(self):
        result,client=self.run_fake('identity')
        self.assertEqual(result['blocker_codes'],['GRADER_IDENTITY_MISMATCH']);self.assertEqual(client.calls,[])
        self.assertEqual(result['identity_check_requests'],1)

    def test_grader_receives_prompt_completion_rubric_without_reference_canary(self):
        row=fixture();messages=grader_messages(row,'TEST_ONLY_COMPLETION',row['rubrics'][0]);text=json.dumps(messages)
        self.assertIn('TEST_ONLY_HIDDEN_RUBRIC',text);self.assertIn('TEST_ONLY_COMPLETION',text)
        for token in ('TEST_ONLY_HIDDEN_REFERENCE','TEST_ONLY_HIDDEN_CANARY','TEST_ONLY_SAMPLE'):self.assertNotIn(token,text)
        from app.locdi.external_dialogue import messages_for
        agent=json.dumps(messages_for(row));self.assertNotIn('TEST_ONLY_HIDDEN_RUBRIC',agent)

    def test_deterministic_run_id_for_same_plan(self):
        a,_=self.run_fake();b,_=self.run_fake();self.assertEqual(a['run_id'],b['run_id'])

    def test_private_partial_results_and_no_clobber(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory:
            root=Path(directory)/'TEST_ONLY';result,client=self.run_fake('partial',output_root=root)
            self.assertEqual(result['status'],'PARTIAL');self.assertTrue((root/'result.json').exists())
            before={p.name:p.read_bytes() for p in root.iterdir()};another=FakeGrader()
            with self.assertRaises(FileExistsError):run_grading(rows_with_negative(),['TEST_ONLY_COMPLETION'],client=another,execute=True,test_only=True,output_root=root)
            self.assertEqual(another.calls,[]);self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_checkpoint_failure_before_dispatch_does_not_count_a_fake_call(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory:
            from app.locdi.healthbench_grader import atomic_rows
            def fail(root,name,document,**kwargs):
                if name.endswith('_before.json'):raise OSError('TEST_ONLY private storage detail')
                return atomic_rows(root,name,document,**kwargs)
            with patch('app.locdi.healthbench_grader.atomic_rows',side_effect=fail):
                result,client=self.run_fake(output_root=Path(directory)/'TEST_ONLY')
            self.assertEqual(result['fake_grading_calls'],0);self.assertEqual(client.calls,[])
            self.assertEqual(result['grading_budget_reservations'],1)

    def test_response_storage_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory:
            from app.locdi.healthbench_grader import atomic_rows
            def fail(root,name,document,**kwargs):
                if name.endswith('_after.json'):raise OSError('TEST_ONLY private storage detail')
                return atomic_rows(root,name,document,**kwargs)
            with patch('app.locdi.healthbench_grader.atomic_rows',side_effect=fail):
                result,client=self.run_fake(max_attempts=2,output_root=Path(directory)/'TEST_ONLY')
            self.assertEqual(len(client.calls),1);self.assertEqual(result['blocker_codes'],['GRADER_STORAGE_FAILED'])

    def test_adapted_arithmetic_positive_negative_none_and_missing(self):
        rubrics=rows_with_negative()[0]['rubrics']
        self.assertEqual(adapted_score(rubrics,[{'criteria_met':True},{'criteria_met':True}]),-0.5)
        self.assertIsNone(adapted_score([{'points':-1}],[{'criteria_met':True}]))
        with self.assertRaises(ValueError):adapted_score(rubrics,[{'criteria_met':True}])


class D6EnvironmentAndReadinessContracts(unittest.TestCase):
    def test_official_unmodified_module_and_pure_functions_under_isolated_python(self):
        interpreter=ROOT/'.venv-healthbench/Scripts/python.exe';self.assertTrue(interpreter.is_file())
        report=json.loads(subprocess.check_output([str(interpreter),'-B',str(ROOT/'scripts/smoke_healthbench_official.py')],text=True))
        self.assertTrue(report['python'].startswith('3.11.'));self.assertEqual(report['official_module_import'],'PASS')
        self.assertFalse(report['grade_sample_called']);self.assertEqual(report['actual_grader_calls'],0)
        self.assertTrue(report['dataset_mean_clipped_after_averaging'])
        for r in report['test_only_arithmetic_fixtures']:
            expected=adapted_score([{'points':p} for p in r['points']],[{'criteria_met':v} for v in r['criteria_met']])
            self.assertEqual(expected,r['test_only_score'])

    def test_environment_does_not_promote_clinical_or_external_readiness(self):
        states=readiness(software_pass=True,development_valid=True,environment_import_pass=True)
        self.assertTrue(states['SOFTWARE_READY']['ready']);self.assertTrue(states['DATASET_READY_FOR_DEVELOPMENT']['ready'])
        self.assertFalse(states['EXTERNAL_BENCHMARK_EVALUATION_READY']['ready']);self.assertFalse(states['CLINICAL_RESEARCH_EVALUATION_READY']['ready'])
        self.assertIsNone(GRADER_CONFIG['model']);self.assertFalse(GRADER_CONFIG['quality_validated'])

    def test_pinned_dependency_modules_and_two_unbounded_retry_layers_inspected(self):
        records=dependency_inventory();self.assertEqual(len(records),4);self.assertTrue(all(r['status']=='FOUND' for r in records))
        self.assertTrue(next(r for r in records if r['module']=='healthbench_eval.py')['unbounded_while_lines'])
        self.assertTrue(next(r for r in records if r['module']=='sampler/chat_completion_sampler.py')['unbounded_while_lines'])

    def test_actual_patient_splits_and_validation_voi_overlap_not_independent(self):
        report=split_audit();self.assertEqual(report['patient_counts']['holdout'],50);self.assertEqual(report['patient_counts']['D3_replay'],29)
        self.assertFalse(report['routing_validation_independent_of_voi_training']);self.assertEqual(report['routing_patient_split_conflicts'],0)
        self.assertFalse(report['external_blind_test_approved'])


if __name__=='__main__':unittest.main()
