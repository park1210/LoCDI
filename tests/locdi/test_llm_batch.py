"""Batch orchestration tests: status fixtures, no LLM responses or networking."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import load_setup, validate_setup, digest
from scripts import run_development_llm_batch as batch


class LLMBatchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(batch.ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl')[:5]
        cls.config = load_setup()
        cls.prepared = validate_setup(cls.config)['prepared']
        cls.old_episode = json.loads((batch.ROOT / 'outputs/episodes/fa86db9c632343ccb0382ac23f5dc7ad_real_case001.json').read_text(encoding='utf-8'))

    def fixture(self, case, status, requests=1, errors=()):
        """Only status/counter signals for persistence tests; no clinical facts."""
        context = adapt_case(case)
        metadata = {**self.old_episode['llm_execution'], 'request_count': requests,
                    'identity_check_count': requests, 'real_model_execution': requests > 0, 'error_types': list(errors)}
        configuration = deepcopy(self.old_episode['configuration'])
        configuration['llm_settings_fingerprint'] = self.prepared['fingerprint']
        result = {'purpose':'development_llm', 'development_only':True, 'mock_execution':False,
                  'simulation_ready':False, 'clinical_metrics':'NOT_EVALUATED', 'research_evaluation_eligible':False,
                  'case_count':1, 'status':status, 'errors':[], 'telemetry':{'decisions':[], 'retrievals':[]},
                  'llm_execution':metadata,
                  'execution_metadata':{'configuration':configuration, 'configuration_fingerprint':digest(configuration),
                      'started_at':'2026-10-28T00:00:00+00:00', 'ended_at':'2026-10-28T00:00:01+00:00'},
                  'execution':{'status':status, 'visits':[{'identity':context.runtime_case['longitudinal'],
                      'stop_reason':None if status=='completed' else 'visit_minute_limit_reached' if status=='incomplete' else 'role_call_failed',
                      'final_state':{'current_minute':1, 'event_log':[], 'cases':{context.case_id:{
                          'is_completed':status=='completed', 'alerts':[], 'llm_audit':[]}}}}]}}
        return result, metadata, None

    def test_dry_run_selects_first_five_in_order_without_transport_or_files(self):
        output = io.StringIO()
        with patch.object(batch, '_execute_case', side_effect=AssertionError('No execution')) as execute, \
             patch('app.locdi.ollama_adapter._backend', side_effect=AssertionError('No network')) as backend, \
             patch.object(batch, 'inspect_development_llm', wraps=batch.inspect_development_llm) as inspect, \
             patch.object(batch, '_write') as write, redirect_stdout(output):
            self.assertEqual(batch.main([]), 0)
        execute.assert_not_called(); backend.assert_not_called(); write.assert_not_called()
        self.assertEqual([call.args[0]['case_id'] for call in inspect.call_args_list], [c['case_id'] for c in self.cases])
        report = json.loads(output.getvalue())
        self.assertEqual([c['case_ref'] for c in report['cases']], ['Case001','Case002','Case003','Case004','Case005'])
        self.assertEqual(report['request_count'], 0)

    def test_missing_approval_and_case_limit_prevent_execution(self):
        with patch.object(batch, '_execute_case') as execute:
            for kwargs in ({'execute':True}, {'execute':True,'allow_llm':True}, {'execute':True,'accept_release_assumption':True}):
                with self.assertRaises(ValueError):batch.run_batch(self.cases, self.config, **kwargs)
            for cases in (self.cases[:4], self.cases+[self.cases[0]]):
                with self.assertRaises(ValueError):batch.run_batch(cases, self.config)
            for args in (['--execute'], ['--execute','--allow-llm'], ['--execute','--accept-release-assumption']):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):batch.main(args)
        execute.assert_not_called()

    def test_mixed_outcomes_keep_all_cases_unique_files_and_exact_counters(self):
        outcomes = [self.fixture(self.cases[0], 'completed', 2),
                    (None, {'request_count':0,'identity_check_count':1,'real_model_execution':False,'error_types':['ModelIdentityError']}, 'ModelIdentityError'),
                    self.fixture(self.cases[2], 'incomplete', 1), self.fixture(self.cases[3], 'completed', 3),
                    self.fixture(self.cases[4], 'failed', 1, ['LLMResponseError'])]
        first = outcomes[0][0]
        state = next(iter(first['execution']['visits'][0]['final_state']['cases'].values()))
        state['llm_audit'] = [{'status':'error', 'error_type':'ContractValidationError'}]
        first['telemetry']['decisions'] = [{'case_id':self.cases[0]['case_id'], 'minute':1, 'role':'er', 'audit_index':0}]
        with tempfile.TemporaryDirectory() as root, patch.object(batch, '_execute_case', side_effect=outcomes) as execute:
            report = batch.run_batch(self.cases, self.config, execute=True, allow_llm=True, accept_release_assumption=True, output_root=root)
            self.assertEqual(execute.call_count, 5)
            self.assertEqual(report['status_counts'], {'completed':2,'incomplete':1,'failed':2})
            self.assertEqual(report['total_request_count'], 7)
            self.assertEqual(report['total_identity_check_count'], 8)
            self.assertEqual(report['error_type_case_counts'], {'ContractValidationError':1,'ModelIdentityError':1,'LLMResponseError':1})
            self.assertEqual(report['summary_episode_count'], 4)
            self.assertNotIn('episode', report['cases'][1]['files'])
            self.assertTrue(all('report' in r['files'] for r in report['cases']))
            files = list(Path(root).rglob('*.json')) + list(Path(root).rglob('*.jsonl'))
            self.assertEqual(len(files), len({p.name for p in files}))
            self.assertEqual(len(list((Path(root)/'reports').glob('*case*_real_run.json'))), 5)
            summary = json.loads((Path(root)/report['files']['summary']).read_text(encoding='utf-8'))
            self.assertEqual(summary['episode_count'], 4)
            text = '\n'.join(p.read_text(encoding='utf-8') for p in files)
            for c in self.cases:self.assertNotIn(c['patient_id'], text)

    def test_storage_failure_keeps_core_status_and_continues(self):
        real_export = batch.export_episode
        def export(*args, **kwargs):
            if kwargs['case_ordinal'] == 2:raise OSError('private evidence must not be recorded')
            return real_export(*args, **kwargs)
        outcomes = [self.fixture(c, 'completed') for c in self.cases]
        with tempfile.TemporaryDirectory() as root, patch.object(batch, '_execute_case', side_effect=outcomes), patch.object(batch, 'export_episode', side_effect=export):
            report = batch.run_batch(self.cases, self.config, execute=True, allow_llm=True, accept_release_assumption=True, output_root=root)
            self.assertEqual(report['status_counts']['completed'], 5)
            self.assertEqual(report['storage_failed_cases'], 1)
            self.assertEqual(report['cases'][1]['status'], 'completed')
            self.assertEqual(report['cases'][1]['storage_status'], 'failed')
            self.assertEqual(len(report['cases']), 5)
            self.assertEqual(report['status'], 'issues_recorded')
            self.assertNotIn('private evidence', json.dumps(report))

    def test_unexpected_runner_exception_preserves_client_counts_and_close(self):
        metadata = {**self.old_episode['llm_execution'], 'request_count':2, 'identity_check_count':2}
        prepared = self.prepared
        closed = []
        class Client:
            def __init__(self, config, authorized):self.prepared = prepared
            def public_metadata(self):return metadata
            def close(self):closed.append(True)
        with patch.object(batch, 'LocalOllamaAdapter', Client), patch.object(batch, 'run_case', side_effect=RuntimeError('private patient message')):
            result, observed, error = batch._execute_case(self.cases[0], self.config, prepared['fingerprint'])
        self.assertIsNone(result)
        self.assertEqual(observed['request_count'], 2)
        self.assertEqual(error, 'RuntimeError')
        self.assertEqual(closed, [True])
