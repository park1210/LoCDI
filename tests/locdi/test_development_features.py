"""Static setup tests; TEST_ONLY toys never enter a model or Runner."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.locdi.action_audit import audit_selectors, read_decision_audit
from app.locdi.development_features import check_development_features, read_artifact, DEFAULT_FEATURE_CONFIG
from app.locdi.llm_setup import ROOT, load_setup, validate_setup
from validation.artifact_readiness import validate_routing_artifact, validate_voi_artifact
from test_artifact_readiness import voi_fixture, source_fixture
from test_routing_voi_readiness import test_only_registry, test_only_routing
from scripts.check_development_features import main


class DevelopmentFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_setup(DEFAULT_FEATURE_CONFIG)
        cls.registry = json.loads((ROOT / cls.config['llm']['registry_path']).read_text(encoding='utf-8'))

    def static(self, config):
        # No synthetic model files. Only mock the local manifest inspector.
        identifier = self.config['llm']['model_identifier']
        with patch('app.locdi.llm_setup.inspect_local_model', return_value={'status': 'READY', 'model_identifier': identifier}):
            return validate_setup(config)

    def test_six_roles_prompts_shared_model_and_version(self):
        self.assertEqual(len(self.registry['agents']), 6)
        self.assertEqual({a['department'] for a in self.registry['agents'] if a['role'] == 'specialty'},
                         {'cardiology', 'endocrinology', 'nephrology', 'neurology', 'pulmonology'})
        setup = self.static(self.config['llm'])
        self.assertEqual(setup['static_readiness'], 'READY')
        self.assertEqual(len(setup['prepared']['prompt_fingerprints']), 6)
        self.assertEqual(setup['checks']['server']['status'], 'NOT_CHECKED')
        registry = AgentRegistry(self.registry)
        for agent in self.registry['agents']:
            self.assertEqual(registry.get(agent['role'], agent['department'])['model'], self.config['llm']['model'])

    def test_duplicate_bindings_and_prompt_digest_model_version_rejected(self):
        for field, value in [('model', 'TEST_ONLY_WRONG:tag'), ('model_version', 'TEST_ONLY_WRONG'),
                             ('model_identifier', 'sha256:' + '0' * 64)]:
            config = deepcopy(self.config['llm']); config[field] = value
            self.assertNotEqual(self.static(config)['static_readiness'], 'READY')
        for mutation in ('duplicate', 'prompt'):
            config = deepcopy(self.config['llm'])
            if mutation == 'duplicate':
                config['agent_bindings'][1] = deepcopy(config['agent_bindings'][0])
            else:
                config['agent_bindings'][1]['prompt'] = 'prompts/missing.txt'
            self.assertNotEqual(self.static(config)['static_readiness'], 'READY')
        registry = deepcopy(self.registry)
        registry['agents'][1] = deepcopy(registry['agents'][0])
        with self.assertRaises(ValueError):
            AgentRegistry(registry)

    def test_default_er_only_has_no_artifact_dependencies_or_backend(self):
        with patch('app.locdi.ollama_adapter._backend', side_effect=AssertionError('No backend')) as backend:
            report = check_development_features()
        backend.assert_not_called()
        self.assertEqual(report['specialties'], [])
        self.assertEqual(report['routing_utility']['status'], 'NOT_REQUIRED')
        self.assertEqual(report['voi_artifact']['status'], 'NOT_REQUIRED')
        self.assertFalse(report['ready'] or report['research_readiness'])
        self.assertEqual(report['generation_requests'], 0)
        self.assertEqual(report['server'], 'NOT_CHECKED')

    def test_specialty_missing_utility_blocks_and_voi_not_read_for_adaptive(self):
        with patch('app.locdi.development_features.read_artifact', side_effect=ValueError('TEST_ONLY_PRIVATE')) as reader:
            report = check_development_features('adaptive_routing', config=self.config)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(report['development_readiness'], 'BLOCKED')
        self.assertIn('ROUTING_CONTRACT_BLOCKED', report['blockers'])
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(report))
        self.assertEqual(report['runtime_audit']['effective_department'], 'NOT_EXECUTED')

    def test_actual_external_voi_source_coverage_and_missing_catalog_binding(self):
        report = check_development_features('bayesian_voi', config=self.config)
        self.assertEqual(report['voi_artifact']['structural_status'], 'STRUCTURAL_VALID')
        self.assertEqual(report['voi_artifact']['development_status'], 'DEVELOPMENT_CONTRACT_READY')
        self.assertEqual(report['voi_artifact']['details']['replay_bin_match_count'], 250)
        self.assertEqual(report['voi_artifact']['details']['train_evaluation_overlap_count'], 0)
        self.assertIn('SHARED_DIAGNOSIS_CATALOG_MISSING_OR_INVALID', report['blockers'])
        self.assertIn('VOI_MODEL_BINDING_NOT_CONFIRMED', report['blockers'])
        text = json.dumps(report)
        for forbidden in ('source_event_store', 'evidence', 'gold_label', 'task_annotations', 'TEST_ONLY_TRAIN'):
            self.assertNotIn(forbidden, text)

    def test_horizon_null_bool_and_one_block(self):
        for horizon in (None, True, 1, 4):
            config = deepcopy(self.config); config['horizon'] = horizon
            report = check_development_features('bayesian_voi', config=config)
            self.assertIn('MULTISTEP_HORIZON_INVALID', report['blockers'])

    def test_routing_snapshot_support_abstention_using_existing_test_only_toys(self):
        manifest = test_only_registry(); utility = test_only_routing(manifest)
        utility['min_samples'] = 999
        report = validate_routing_artifact(manifest, utility, evaluation_patient_ids=['TEST_ONLY_EVAL'], observations=[{}])
        self.assertEqual(report['details']['selection_status'], 'VALID_ABSTENTION')
        self.assertFalse(report['research_validated'])
        utility['registry']['agents'][0]['version'] = 'TEST_ONLY_WRONG'
        self.assertEqual(validate_routing_artifact(manifest, utility)['structural_status'], 'BLOCKED')

    def test_source_coverage_and_catalog_mismatch_block_test_only(self):
        for sources, catalog in [([{'action': 'Test', 'target': 'TEST_ONLY_MISSING', 'evidence': 'TEST_ONLY'}], None),
                                 (source_fixture(), ['TEST_ONLY_WRONG'])]:
            report = validate_voi_artifact(voi_fixture(), source_actions=sources, diagnosis_candidates=catalog,
                                          expected_horizon=2, evaluation_patient_ids=['TEST_ONLY_EVAL'])
            self.assertEqual(report['development_status'], 'BLOCKED')

    def test_external_path_restrictions_and_malformed_envelope(self):
        for path in (None, '../secrets.json', '../../escape.json', 'configs/development_llm.txt'):
            with self.assertRaises(ValueError):
                read_artifact(path)
        config = deepcopy(self.config); config['ready'] = True
        self.assertIn('FEATURE_CONFIGURATION_INVALID', check_development_features('bayesian_voi', config=config)['blockers'])
        with self.assertRaises(ValueError):
            check_development_features('unknown')

    def test_cli_no_execution_flags_default_dry_run(self):
        with patch('builtins.print'):
            self.assertIn(main(['--mode', 'adaptive_routing']), (0, 2))
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                main(['--execute', '--allow-llm'])

    def test_selectors_no_inferred_success_private_text_or_early_model_error(self):
        self.assertEqual(audit_selectors({}, requested_selector='bayesian_voi')['voi_selection_status'], 'NOT_RECORDED')
        self.assertEqual(audit_selectors({'status': 'error'}, requested_selector='bayesian_voi')['voi_selection_status'], 'NOT_RECORDED')
        row = {'action_selector': 'specialist_heuristic_no_supported_voi',
               'selector_status': {'status': 'unavailable', 'reason': 'TEST_ONLY_PRIVATE'}}
        report = audit_selectors(row, requested_selector='bayesian_voi', artifact_present=True)
        self.assertEqual(report['voi_selection_status'], 'UNAVAILABLE')
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(report))
        row = {'action_selector': 'explicit_outcome_voi', 'selector_status': {'status': 'available'},
               'decision': {'action': 'request_test'}}
        self.assertEqual(audit_selectors(row, requested_selector='bayesian_voi')['voi_selection_status'], 'SUCCESS')

    def test_existing_mock_and_real_documents_remain_readable(self):
        paths = list((ROOT / 'outputs/episodes').glob('*.json'))
        found = set()
        for path in paths:
            document = json.loads(path.read_text(encoding='utf-8'))
            if 'mock_execution' in document:
                found.add(document['mock_execution'])
                self.assertIsInstance(read_decision_audit(document)['records'], list)
        self.assertEqual(found, {True, False})


if __name__ == '__main__':
    unittest.main()
