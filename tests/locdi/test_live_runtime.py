"""TEST_ONLY fake backend; exercise the actual adapter and Core, no network."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app.locdi.live_runtime as live
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT, load_setup, digest
from app.locdi.ollama_adapter import LLMRequestError, LLMTimeoutError, ModelIdentityError, LLMRequestLimitError
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from scripts.run_controlled_runtime_pilot import main
from test_simulation import SourceActionClient
from test_first_real_run import keys, strings


class LiveRuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(live.DATASET)[0]
        cls.config = load_setup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=str(ROOT / 'outputs'))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'pilot'
        self.addCleanup(patch.stopall)
        patch.object(live, 'PILOT_ROOT', self.root).start()
        patch.object(live, 'load_cases', return_value=[self.case]).start()
        # TEST_ONLY disk-availability fixture, not a model or probability artifact.
        patch('app.locdi.llm_setup.inspect_local_model', return_value={
            'status': 'READY', 'model_identifier': self.config['model_identifier']}).start()
        self.run_id = 'a' * 32
        self.calls = []

    def backend(self, failure=None, mismatch=False):
        source = SourceActionClient()
        calls = self.calls
        identifier = self.config['model_identifier']
        class HTTP:
            def get(self, url):
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                    'models': [{'name': 'qwen2.5:1.5b-instruct',
                                'digest': 'sha256:' + '0' * 64 if mismatch else identifier}]})
            def close(self):
                pass
        class Legacy:
            def chat_json(self, system, user, model=None, *, schema=None):
                calls.append(json.loads(user))
                if failure == 'timeout':
                    raise TimeoutError('PRIVATE SOURCE SECRET')
                return source.chat_json(system, user, model=model, schema=schema)
        http = HTTP()
        return patch('app.locdi.ollama_adapter._backend', return_value=(Legacy(), http, http, http))

    def execute(self, **kwargs):
        plan = live.preflight(run_id=self.run_id, config=self.config)
        return live.run_pilot(run_id=self.run_id, config=self.config, execute=True, allow_llm=True,
            accept_release_assumption=True, expected_plan_fingerprint=plan['plan_fingerprint'], **kwargs)

    def adapter(self):
        config = deepcopy(self.config); config['max_requests'] = 5
        return live.PilotOllamaAdapter(config, authorized=True, output_root=self.root)

    def test_default_static_never_creates_client_backend_or_output(self):
        with patch.object(live, 'PilotOllamaAdapter') as client, self.backend() as backend:
            result = live.run_pilot(run_id=self.run_id, config=self.config)
        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['request_count'], 0)
        self.assertEqual(result['server_availability'], 'NOT_CHECKED')
        self.assertEqual(result['server_model_identity'], 'NOT_CHECKED')
        self.assertEqual(result['plan']['generation']['max_requests'], 5)
        self.assertEqual(result['plan']['features'], live.FEATURES)
        self.assertEqual(self.config['max_requests'], 12)
        client.assert_not_called(); backend.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_missing_approval_blocks_before_client_and_preflight(self):
        for flags in ({'execute': True}, {'execute': True, 'allow_llm': True},
                      {'execute': True, 'accept_release_assumption': True}, {'execute': 1}):
            with patch.object(live, 'PilotOllamaAdapter') as client, patch.object(live, 'preflight') as pre:
                result = live.run_pilot(run_id=self.run_id, **flags)
                self.assertEqual(result['status'], 'blocked')
                client.assert_not_called(); pre.assert_not_called()

    def test_approved_fingerprint_required_before_client(self):
        with patch.object(live, 'PilotOllamaAdapter') as client:
            result = live.run_pilot(run_id=self.run_id, config=self.config, execute=True,
                allow_llm=True, accept_release_assumption=True, expected_plan_fingerprint='0'*64)
        self.assertEqual(result['blocker_codes'], ['APPROVED_PLAN_FINGERPRINT_MISMATCH'])
        client.assert_not_called()

    def test_model_registry_prompt_mismatch_block_before_transport(self):
        for key, value in [('model_identifier', 'sha256:'+'0'*64), ('registry_path', 'missing.json')]:
            config = deepcopy(self.config); config[key] = value
            with self.backend() as backend:
                result = live.preflight(run_id=self.run_id, config=config)
            self.assertIn('STATIC_SETUP_NOT_READY', result['blocker_codes'])
            backend.assert_not_called()
        config = deepcopy(self.config)
        config['agent_bindings'][0]['prompt'] = 'prompts/patient_system.txt'
        self.assertIn('STATIC_SETUP_NOT_READY', live.preflight(run_id=self.run_id, config=config)['blocker_codes'])
        approved = live.preflight(run_id=self.run_id, config=self.config)['plan_fingerprint']
        config = deepcopy(self.config); config['temperature'] = 0.1
        with patch.object(live, 'PilotOllamaAdapter') as client:
            result = live.run_pilot(run_id=self.run_id, config=config, execute=True, allow_llm=True,
                accept_release_assumption=True, expected_plan_fingerprint=approved)
        self.assertEqual(result['status'], 'blocked'); client.assert_not_called()

    def test_server_digest_mismatch_zero_generation_and_durable_failure(self):
        with self.backend(mismatch=True):
            result = self.execute()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['llm_execution']['request_count'], 0)
        self.assertEqual(result['llm_execution']['identity_check_count'], 1)
        self.assertIn('ModelIdentityError', result['llm_execution']['error_types'])
        self.assertEqual(self.calls, [])
        self.assertTrue((self.root / self.run_id / 'reports/pilot.json').is_file())

    def test_global_five_request_limit_real_core_and_partial_counts(self):
        with self.backend(), patch('app.locdi.run_simulation.run_with_recovery', wraps=run_with_recovery) as runner, \
                patch('app.core.longitudinal_runner.HospitalParallelScheduler', wraps=HospitalParallelScheduler) as scheduler:
            result = self.execute()
        runner.assert_called_once(); scheduler.assert_called_once()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(result['llm_execution']['request_count'], 5)
        self.assertEqual(result['llm_execution']['identity_check_count'], 5)
        self.assertIn('REQUEST_LIMIT_REACHED', result['blocker_codes'])
        saved = json.loads((self.root / self.run_id / 'reports/pilot.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, result)
        self.assertFalse(result['research_evaluation_ready'])

    def test_timeout_preserves_actual_count_and_never_retries(self):
        with self.backend(failure='timeout'):
            result = self.execute()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['llm_execution']['request_count'], 1)
        self.assertIn('LLMTimeoutError', result['llm_execution']['error_types'])
        self.assertNotIn('PRIVATE SOURCE SECRET', json.dumps(result))

    def test_completed_and_incomplete_are_preserved_without_promotion(self):
        original = self.config
        try:
            for minute_limit, expected, run_id in ((20, 'completed', 'b'*32), (1, 'incomplete', 'c'*32)):
                self.config = deepcopy(original); self.config['minute_limit'] = minute_limit
                self.run_id = run_id
                with self.backend():
                    result = self.execute()
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['core_status'], expected)
                self.assertLessEqual(result['llm_execution']['request_count'], 5)
                self.assertFalse(result['research_evaluation_ready'])
                self.assertEqual(result['clinical_metrics'], 'NOT_EVALUATED')
        finally:
            self.config = original

    def test_storage_failure_after_request_blocks_next_request(self):
        original = live._write
        def fail_after(document, destination, *args):
            if destination == 'private/001_after.json':
                raise OSError('PRIVATE SOURCE SECRET')
            return original(document, destination, *args)
        with self.backend(), patch.object(live, '_write', side_effect=fail_after):
            result = self.execute()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['llm_execution']['request_count'], 1)
        self.assertIn('PilotStorageError', result['llm_execution']['error_types'])
        self.assertTrue((self.root / self.run_id / 'private/001_before.json').is_file())
        self.assertNotIn('PRIVATE SOURCE SECRET', json.dumps(result))

    def test_storage_failure_before_request_blocks_transport(self):
        with self.backend() as backend, patch.object(live, '_write', side_effect=OSError('private')):
            result = self.execute()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('FINAL_REPORT_STORAGE_FAILED', result['blocker_codes'])
        backend.assert_not_called()

    def test_checkpoints_fail_closed_even_if_caller_attempts_again(self):
        client = self.adapter()
        system = __import__('app.llm.prompt_loader', fromlist=['er_system_prompt']).er_system_prompt()
        with self.backend(failure='timeout'):
            with self.assertRaises(LLMTimeoutError):
                client.chat_json(system, '{}', model=self.config['model'], schema={})
            with self.assertRaises(LLMRequestError):
                client.chat_json(system, '{}', model=self.config['model'], schema={})
        self.assertEqual(client.request_count, 1)
        client.close()

    def test_output_no_clobber_and_path_escape_before_calls(self):
        existing = self.root / self.run_id
        existing.mkdir(parents=True)
        marker = existing / 'existing.json'; marker.write_text('unchanged')
        with self.backend() as backend:
            result = self.execute()
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(marker.read_text(), 'unchanged')
        backend.assert_not_called()
        for value in ('../escape', '', 'A'*32):
            self.assertIn('EXPLICIT_32_HEX_RUN_ID_REQUIRED', live.preflight(run_id=value)['blocker_codes'])

    def test_public_private_separation_and_actual_payload_boundary(self):
        with self.backend():
            result = self.execute()
        documents = [json.loads(p.read_text(encoding='utf-8')) for p in (self.root / self.run_id).rglob('*.json')]
        forbidden = {'gold', 'evaluation', 'provenance', 'source_event_store', 'patient_id', 'visit_id',
                     'case_id', 'task_annotations', 'system_prompt', 'user_prompt', 'raw_response'}
        self.assertFalse(set().union(*(keys(d) for d in documents)) & forbidden)
        leaves = [text for d in documents for text in strings(d)]
        for needle in (self.case['patient_id'], self.case['case_id'], self.case['agent_visible']['index_encounter_id']):
            self.assertFalse(any(needle in text for text in leaves))
        for event in self.case['environment']['current_encounter_events']:
            self.assertFalse(any(event['evidence'] in text for text in leaves))
        # Internal Memory carries patient/visit linkage; public files above do not.
        self.assertTrue(all(not keys(payload) & (forbidden - {'patient_id', 'visit_id'}) for payload in self.calls))
        episode = json.loads((self.root / self.run_id / result['files']['episode']).read_text(encoding='utf-8'))
        audit = json.loads((self.root / self.run_id / result['files']['audit']).read_text(encoding='utf-8'))
        self.assertEqual(audit['workflow'], episode['workflow'])
        self.assertEqual(audit['terminal'], episode['result'])
        self.assertEqual(audit['roles'], episode['decision_audit'])
        self.assertEqual(audit['configuration_fingerprint'], episode['configuration_fingerprint'])

    def test_unsupported_live_modes_block_without_toy_promotion(self):
        for mode in ('specialty', 'adaptive', 'adaptive_voi', 'research', 'TEST_ONLY'):
            with patch.object(live, 'PilotOllamaAdapter') as client:
                result = live.run_pilot(run_id=self.run_id, config=self.config, mode=mode,
                    execute=True, allow_llm=True, accept_release_assumption=True)
            self.assertEqual(result['status'], 'blocked'); client.assert_not_called()
        config = deepcopy(self.config); config['routing_utility'] = {'TEST_ONLY': True}
        self.assertIn('STATIC_SETUP_NOT_READY', live.preflight(run_id=self.run_id, config=config)['blocker_codes'])
        config = deepcopy(self.config); config['voi_model'] = {'TEST_ONLY': True}
        self.assertIn('STATIC_SETUP_NOT_READY', live.preflight(run_id=self.run_id, config=config)['blocker_codes'])

    def test_cli_default_and_missing_flags(self):
        output = io.StringIO()
        with redirect_stdout(output), patch('app.locdi.ollama_adapter._backend') as backend:
            self.assertEqual(main(['--run-id', self.run_id]), 0)
        backend.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['status'], 'dry_run')
        with redirect_stderr(io.StringIO()), patch('scripts.run_controlled_runtime_pilot.load_setup') as load:
            with self.assertRaises(SystemExit):
                main(['--run-id', self.run_id, '--execute'])
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
