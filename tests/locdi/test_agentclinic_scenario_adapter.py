import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
import unittest
from app.locdi.agentclinic_scenario_adapter import load_scenarios, adapt_test_record, action_catalog
from app.locdi.agentclinic_scenario_schema import ScenarioContractError, SOURCE_SHA256
from app.locdi.agentclinic_turn_runtime import initial_observation
from agentclinic_offline_support import bundle, fixture, SOURCE


class AgentClinicAdapterTests(unittest.TestCase):
    def test_all_107_originals_and_keys_are_separate(self):
        data = bundle()
        self.assertEqual((len(data.scenarios), len(data.scoring_keys)), (107, 107))
        self.assertEqual(len({s.scenario_id for s in data.scenarios}), 107)
        for s, key in zip(data.scenarios, data.scoring_keys):
            self.assertEqual(s.scenario_id, key.scenario_id)
            self.assertNotIn('Correct_Diagnosis', json.dumps(initial_observation(s)))
            self.assertNotIn(key.correct_diagnosis.casefold(), json.dumps(initial_observation(s), ensure_ascii=False).casefold())
            self.assertEqual(s.source['original_sha256'], SOURCE_SHA256)

    def test_original_source_and_all_store_types_preserved(self):
        before = SOURCE.read_bytes()
        rows = [json.loads(l)['OSCE_Examination'] for l in before.decode('utf8').splitlines()]
        for original, s, key in zip(rows, bundle().scenarios, bundle().scoring_keys):
            self.assertEqual(s.patient_facts, original['Patient_Actor'])
            self.assertEqual(s.physical_exams, original['Physical_Examination_Findings'])
            self.assertEqual(s.test_results, original['Test_Results'])
            self.assertEqual(key.correct_diagnosis, original['Correct_Diagnosis'])
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)

    def test_empty_test_cases_keep_empty_catalog(self):
        empty = [i for i, s in enumerate(bundle().scenarios, 1) if not s.test_results]
        self.assertEqual(empty, [69, 106])
        for i in empty:
            self.assertEqual(action_catalog(bundle().scenarios[i-1])['TEST'], [])

    def test_wrong_hash_and_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'source.jsonl'
            p.write_bytes(SOURCE.read_bytes()+b'\n')
            with self.assertRaisesRegex(ScenarioContractError, '^SOURCE_HASH_MISMATCH$'):
                load_scenarios(p)
        with self.assertRaisesRegex(ScenarioContractError, '^SOURCE_REVISION_MISMATCH$'):
            load_scenarios(SOURCE, revision='wrong')

    def test_fixture_copy_isolation_and_type_variation(self):
        row = fixture(); before = deepcopy(row)
        scenario, key = adapt_test_record(row)
        scenario.patient_facts['History'] = 'mutated-copy'
        self.assertEqual(row, before)
        self.assertIsInstance(scenario.patient_facts['Past_Medical_History'], list)
        self.assertIsInstance(scenario.patient_facts['Social_History'], dict)

    def test_wrong_source_container_type_and_private_key_are_blocked(self):
        row = fixture(); row['OSCE_Examination']['Test_Results'] = []
        with self.assertRaisesRegex(ScenarioContractError, '^OSCE_FIELD_TYPES$'):
            adapt_test_record(row)
        row = fixture(); row['OSCE_Examination']['Patient_Actor']['Symptoms']['Correct_Diagnosis'] = 'PRIVATE'
        with self.assertRaisesRegex(ScenarioContractError, '^PRIVATE_KEY_IN_FACT_STORE$'):
            adapt_test_record(row)

    def test_source_target_with_answer_label_cannot_leak_via_catalog(self):
        row = fixture()
        row['OSCE_Examination']['Test_Results'] = {'TEST_ONLY_PRIVATE_KEY': 'value'}
        with self.assertRaisesRegex(ScenarioContractError, '^INITIAL_OR_CATALOG_REFERENCE_EXPOSURE$'):
            adapt_test_record(row)
