"""TEST_ONLY clinical-free transport fixtures. No Ollama or fitting calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.llm_setup import digest, ROOT
from app.locdi.routing_collection_preparation import (DEPARTMENTS, ASSUMPTIONS, select_candidates, project_input,
    response_schema, _snapshot)
from app.locdi.routing_collector import (collect, validate_preparation, save_preparation, load_preparation, atomic_json)
from app.locdi.ollama_adapter import LLMTimeoutError, LLMResponseError, ModelIdentityError
from validation.routing_training_data import _leaves
from scripts.collect_routing_outcomes import main


def test_only_bundle():
    agents = [{'agent_id': 'TEST_ONLY_ER', 'role': 'er', 'department': '', 'model': 'TEST_ONLY_MODEL', 'version': 'TEST_ONLY', 'keywords': []}]
    agents += [{'agent_id': 'TEST_ONLY_' + d, 'role': 'specialty', 'department': d,
                'model': 'TEST_ONLY_MODEL', 'version': 'TEST_ONLY', 'keywords': []} for d in DEPARTMENTS]
    registry = {'schema_version': 1, 'agents': agents}
    generation = {'model_identifier': 'sha256:' + '1' * 64, 'temperature': 0, 'seed': 42, 'max_output_tokens': 1024,
                  'prompt_fingerprints': {a['agent_id']: '2' * 64 for a in agents[1:]}}
    candidates, assignments, patients = [], [], {'train': [], 'validation': [], 'test': ['TEST_ONLY_HOLDOUT']}
    for i in range(15):
        split = 'validation' if i % 3 == 2 else 'train'
        observation = {'chief_complaint': 'TEST_ONLY_SIGNAL', 'demographics': {}, 'elapsed_minutes': 1,
                       'current_symptoms': [], 'history': [], 'objective_findings': [], 'longitudinal_memory': {'mode': 'selective', 'records': []}}
        decision = {'patient_id': 'TEST_ONLY_PRIVATE_P%d' % i, 'visit_id': 'TEST_ONLY_PRIVATE_V%d' % i,
                    'decision_id': 'TEST_ONLY_PRIVATE_D%d' % i, 'source_ref': 'TEST_ONLY_PRIVATE_S%d' % i,
                    'minute': 1, 'split': split, 'observation': observation, 'registry_fingerprint': digest(registry),
                    'tokens': ['test_only_signal'], 'generation': deepcopy(generation), 'allowed_actions': ['diagnose']}
        payload = project_input(observation, ['TEST_ONLY_A'])
        candidates.append({'case_ref': 'Candidate%03d' % (i + 1), 'stratum': DEPARTMENTS[i // 3], 'decision': decision,
            'visibility': {p: {'available_minute': 0} for p in _leaves(observation)},
            'shared_payload': payload, 'shared_input_fingerprint': digest(payload)})
        assignments.append({k: decision[k] for k in ('patient_id', 'decision_id', 'split')})
        patients[split].append(decision['patient_id'])
    bundle = {'schema_version': 'locdi.routing_preparation.v1', 'purpose': 'TEST_ONLY', 'assumptions': ASSUMPTIONS,
        'registry': registry, 'catalog': ['TEST_ONLY_A'], 'response_schema': response_schema(['TEST_ONLY_A']),
        'candidates': candidates, 'selection_reasons': {}, 'llm_config': {'model_identifier': generation['model_identifier'], 'max_requests': 75},
        'split_manifest': {'schema_version': 'locdi.routing_split.v1', 'dataset_fingerprint': '3' * 64,
                           'holdout_fingerprint': digest(patients['test']), 'patients': patients, 'decisions': assignments}}
    bundle['fingerprint'] = digest(bundle)
    setup = {'static_readiness': 'READY', 'prepared': {'registry': registry,
             'prompt_fingerprints': {a['agent_id']: '2' * 64 for a in agents}}}
    return bundle, setup


class TestOnlyClient:
    def __init__(self, errors=None, unresolved=False):
        self.errors = errors or {}; self.request_count = 0; self.identity_check_count = 0
        self.calls = []; self.closed = False; self.unresolved = unresolved
    def chat_json(self, system, payload, *, model, schema):
        self.identity_check_count += 1; index = len(self.calls); self.calls.append(json.loads(payload))
        failure = self.errors.get(index)
        if failure is ModelIdentityError:
            raise failure('TEST_ONLY_PRIVATE_ERROR')
        self.request_count += 1
        if failure:
            raise failure('TEST_ONLY_PRIVATE_ERROR')
        return json.dumps({'action': 'diagnose', 'diagnosis': 'Unresolved' if self.unresolved else 'TEST_ONLY_A',
                           'certainty': 'unresolved' if self.unresolved else 'supported', 'rationale': 'TEST_ONLY_NOT_CLINICAL'})
    def public_metadata(self):
        return {'request_count': self.request_count, 'identity_check_count': self.identity_check_count}
    def close(self):
        self.closed = True


class RoutingCollectionContracts(unittest.TestCase):
    def setUp(self):
        self.bundle, self.setup = test_only_bundle()
        self.patches = [patch('app.locdi.routing_collector.build_preparation', return_value=(deepcopy(self.bundle), [])),
                        patch('app.locdi.routing_collector.validate_setup', return_value=self.setup)]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)

    def execute(self, client=None, budget=75):
        client = client or TestOnlyClient()
        with tempfile.TemporaryDirectory(dir=str(ROOT / 'tests')) as root:
            with patch('app.locdi.routing_collector.LocalOllamaAdapter', return_value=client):
                report = collect(self.bundle, execute=True, allow_llm=True, accept_development_assumptions=True,
                    run_id='TEST_ONLY_RUN', max_requests=budget, private_root=Path(root)/'private', public_root=Path(root)/'public')
            private = [json.loads(p.read_text(encoding='utf-8')) for p in (Path(root)/'private/TEST_ONLY_RUN').glob('Candidate*.json')]
            public = json.loads((Path(root)/'public/TEST_ONLY_RUN_routing_collection.json').read_text(encoding='utf-8'))
        return report, client, private, public

    def test_dry_run_never_constructs_backend_or_writes(self):
        with patch('app.locdi.routing_collector.LocalOllamaAdapter') as client, patch('app.locdi.routing_collector.atomic_json') as writer:
            report = collect(self.bundle)
        client.assert_not_called(); writer.assert_not_called()
        self.assertEqual(report['generation_requests'], 0)
        self.assertEqual(report['shared_input_checks'], 75)

    def test_all_approval_flags_required_before_transport_and_storage(self):
        with self.assertRaises(ValueError):
            collect(self.bundle, execute='false', allow_llm=True, accept_development_assumptions=True)
        for allow, accept in [(False, False), (True, False), (False, True)]:
            with patch('app.locdi.routing_collector.LocalOllamaAdapter') as client:
                with self.assertRaises(ValueError):
                    collect(self.bundle, execute=True, allow_llm=allow, accept_development_assumptions=accept)
                client.assert_not_called()

    def test_selection_balanced_unique_holdout_and_deterministic(self):
        rows, taxonomy = [], {}
        for d in DEPARTMENTS:
            label = 'TEST_ONLY_LABEL_' + d; taxonomy[label] = d
            for i in range(5):
                rows.append({'training_patient_id': 'TEST_ONLY_' + d + str(i), 'training_case_id': 'TEST_ONLY_C_' + d + str(i), 'training_label': label})
        rows.append(deepcopy(rows[0]))
        selected, reasons, _ = select_candidates(rows, taxonomy, {rows[1]['training_patient_id']})
        selected2, _, _ = select_candidates(list(reversed(rows)), taxonomy, {rows[1]['training_patient_id']})
        self.assertEqual(selected, selected2)
        self.assertEqual(len({r['bank']['training_patient_id'] for r in selected}), 15)
        self.assertEqual(sum(r['split'] == 'train' for r in selected), 10)
        self.assertIn('DUPLICATE_PATIENT_CONTEXT', reasons)
        self.assertIn('HOLDOUT_PATIENT', reasons)

    def test_insufficient_unique_patients_not_filled(self):
        with self.assertRaises(ValueError):
            select_candidates([], {}, set())

    def test_shared_input_same_five_agents_system_department_separate(self):
        report, client, rows, _ = self.execute()
        first = [dict(p) for p in client.calls[:5]]
        for payload in first:
            payload.pop('department')
        self.assertTrue(all(p == first[0] for p in first))
        self.assertEqual(report['generation_requests'], 75)
        self.assertEqual(len(rows), 75)
        self.assertTrue(client.closed)

    def test_request_cap_is_global_across_cases_and_agents(self):
        report, client, rows, _ = self.execute(budget=7)
        self.assertEqual(report['generation_requests'], 7)
        self.assertEqual(len(rows), 7)
        self.assertEqual(report['stop_reason'], 'GLOBAL_REQUEST_LIMIT')

    def test_invalid_request_limits_rejected(self):
        for value in (0, 76, True):
            with self.assertRaises(ValueError):
                collect(self.bundle, max_requests=value)

    def test_timeout_and_schema_errors_preserve_missing_not_false(self):
        report, client, rows, _ = self.execute(TestOnlyClient({0: LLMTimeoutError, 1: LLMResponseError}), budget=7)
        self.assertEqual(report['generation_requests'], 7)
        self.assertEqual(sum(r['correctness_status'] == 'NOT_EVALUABLE' for r in rows), 2)
        self.assertTrue(all(r['correct'] is None for r in rows))

    def test_identity_error_stops_all_subsequent_calls(self):
        report, client, rows, public = self.execute(TestOnlyClient({0: ModelIdentityError}))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(report['generation_requests'], 0)
        self.assertEqual(report['stop_reason'], 'IDENTITY_OR_CLIENT_ERROR')
        self.assertNotIn('TEST_ONLY_PRIVATE_ERROR', json.dumps(public))

    def test_unresolved_not_incorrect_and_no_automatic_reference_review(self):
        report, _, rows, _ = self.execute(TestOnlyClient(unresolved=True), budget=5)
        self.assertEqual(report['complete_train_groups'], 0)
        self.assertTrue(all(r['correct'] is None and r['correctness_status'] == 'NOT_EVALUABLE' for r in rows))

    def test_private_original_ids_never_appear_in_public_audit(self):
        _, _, private, public = self.execute(budget=5)
        self.assertIn('TEST_ONLY_PRIVATE_P0', json.dumps(private))
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(public))
        self.assertNotIn('TEST_ONLY_NOT_CLINICAL', json.dumps(public))

    def test_tampered_payload_registry_or_gold_blocks_before_backend(self):
        for change in ('gold', 'registry', 'memory'):
            bundle = deepcopy(self.bundle)
            if change == 'gold':
                bundle['candidates'][0]['shared_payload']['gold'] = 'TEST_ONLY_PRIVATE'
            elif change == 'registry':
                bundle['registry']['agents'][1]['version'] = 'TEST_ONLY_OTHER'
            else:
                bundle['candidates'][0]['decision']['observation']['longitudinal_memory']['records'] = [{'text': 'TEST_ONLY_OTHER'}]
            with patch('app.locdi.routing_collector.LocalOllamaAdapter') as backend:
                with self.assertRaises(ValueError):
                    collect(bundle)
                backend.assert_not_called()

    def test_future_observation_and_missing_ledger_block(self):
        bundle = deepcopy(self.bundle)
        bundle['candidates'][0]['visibility']['/chief_complaint']['available_minute'] = 2
        bundle.pop('fingerprint'); bundle['fingerprint'] = digest(bundle)
        with patch('app.locdi.routing_collector.build_preparation', return_value=(bundle, [])):
            with self.assertRaises(ValueError):
                validate_preparation(bundle)
        source_row = {'training_patient_id': 'TEST_ONLY_P', 'training_case_id': 'TEST_ONLY_C',
                      'source_provenance': {'observation_event_ids': ['TEST_ONLY_FUTURE']}}
        source = {'source_events': [{'event_id': 'TEST_ONLY_FUTURE', 'patient_id': 'TEST_ONLY_P', 'encounter_id': 'TEST_ONLY_V',
                  'timestamp_upper_epoch_ms': 1793145720000}], 'symptom_sources': []}
        context = {'demographics': {}, 'index_encounter': {'encounter_id': 'TEST_ONLY_V', 'start_time': '2026-10-28T00:00:00+00:00'}}
        observation, ledger, excluded = _snapshot(source_row, source, context, {'mode': 'selective', 'records': []})
        self.assertEqual(excluded, 1)
        self.assertEqual(observation['objective_findings'], [])
        bundle = deepcopy(self.bundle)
        bundle['candidates'][0]['visibility'].pop('/chief_complaint')
        bundle.pop('fingerprint'); bundle['fingerprint'] = digest(bundle)
        with patch('app.locdi.routing_collector.build_preparation', return_value=(bundle, [])):
            with self.assertRaises(ValueError):
                validate_preparation(bundle)

    def test_atomic_no_clobber_and_invalid_serialization_preserve_original(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as root:
            path = Path(root)/'old.json'; atomic_json(path, {'TEST_ONLY': 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                atomic_json(path, {'TEST_ONLY': 2})
            with self.assertRaises(ValueError):
                atomic_json(path, {'TEST_ONLY': float('nan')})
            self.assertEqual(before, path.read_bytes())

    def test_save_reload_manifest_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as root:
            save_preparation(self.bundle, [], 'TEST_ONLY', root=root)
            self.assertEqual(load_preparation('TEST_ONLY', root=root), self.bundle)
            with self.assertRaises(FileExistsError):
                save_preparation(self.bundle, [], 'TEST_ONLY', root=root)
            with self.assertRaises(ValueError):
                save_preparation(self.bundle, [], '../escape', root=root)

    def test_storage_failure_stops_requests_immediately(self):
        client = TestOnlyClient()
        import app.locdi.routing_collector as module
        original = module.atomic_json
        def fail_on_outcome(path, document):
            if path.name.startswith('Candidate'):
                raise OSError('TEST_ONLY_STORAGE_ERROR')
            return original(path, document)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as root:
            with patch.object(module, 'LocalOllamaAdapter', return_value=client), patch.object(module, 'atomic_json', side_effect=fail_on_outcome):
                with self.assertRaises(OSError):
                    collect(self.bundle, execute=True, allow_llm=True, accept_development_assumptions=True, run_id='TEST_ONLY',
                            private_root=Path(root)/'private', public_root=Path(root)/'public')
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.closed)

    def test_cli_missing_authorization_does_not_load_or_call(self):
        with patch('scripts.collect_routing_outcomes.load_preparation') as loader, patch('builtins.print'):
            self.assertEqual(main(['--preparation', 'TEST_ONLY', '--execute']), 2)
        loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
