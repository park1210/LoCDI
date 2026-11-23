"""Structural and provenance checks using copies of real dataset records."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from app.locdi.dataset_loader import load_cases
from validation.schema_validator import validate_schema
from validation.provenance_validator import validate_provenance
from validation.run_validation import VALIDATORS

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets/raw/longitudinal_benchmark_50.jsonl"


class SchemaContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_cases(DATASET)[0]

    def test_declared_validation_order(self):
        self.assertEqual([fn.__name__ for fn in VALIDATORS], [
            "validate_schema", "validate_temporal", "validate_actions", "validate_provenance"])

    def test_actual_schema_and_clinical_scope(self):
        result = validate_schema(self.case)
        self.assertEqual(result.status, "PASS")
        self.assertIn("clinical_correctness", result.not_reviewed)
        self.assertEqual(result.case_id, self.case["case_id"])

    def test_missing_version_type_identity_and_duplicate_ids(self):
        mutations = [lambda c: c.pop("evaluation"),
                     lambda c: c.update(schema_version="unsupported"),
                     lambda c: c["agent_visible"].update(past_visits={}),
                     lambda c: c["agent_visible"]["longitudinal"].update(patient_id=c["case_id"]),
                     lambda c: c["agent_visible"]["past_visits"].append(deepcopy(c["agent_visible"]["past_visits"][0])),
                     lambda c: c["environment"]["current_encounter_events"].append(deepcopy(c["environment"]["current_encounter_events"][0]))]
        for mutate in mutations:
            case = deepcopy(self.case)
            mutate(case)
            result = validate_schema(case)
            self.assertEqual(result.status, "FAIL")
            self.assertTrue(all(i.path and i.message for i in result.issues))
            self.assertNotIn(self.case["patient_id"], " ".join(i.message for i in result.issues))

    def test_bad_schema_blocks_provenance(self):
        self.assertEqual(validate_schema({}).status, "FAIL")
        self.assertEqual(validate_provenance({}).status, "BLOCKED")

    def test_readiness_without_gold_and_unknown_memory_annotation(self):
        case = deepcopy(self.case)
        case["task_annotations"]["metric_readiness"]["diagnosis_accuracy"] = True
        self.assertIn("READINESS_WITHOUT_GOLD", {i.code for i in validate_schema(case).issues})
        case = deepcopy(self.case)
        case["task_annotations"]["gold"]["relevant_memory_ids"] = [case["case_id"]]
        self.assertIn("UNKNOWN_MEMORY_ANNOTATION", {i.code for i in validate_schema(case).issues})

    def test_hash_format_is_not_byte_verification(self):
        result = validate_provenance(self.case)
        self.assertEqual(result.status, "NOT_REVIEWED")
        self.assertEqual(result.verified_source_hashes, 0)
        self.assertIsNone(result.source_integrity_verified)
        case = deepcopy(self.case)
        key = next(iter(case["provenance"]["source_sha256"]))
        case["provenance"]["source_sha256"][key] = "invalid"
        self.assertIn("SOURCE_HASH_FORMAT", {i.code for i in validate_provenance(case).issues})

    def test_explicit_available_source_hash_is_recomputed(self):
        case = deepcopy(self.case)
        name = "app/data/hospital_scene_layout.json"
        case["provenance"]["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
        result = validate_provenance(case, source_root=ROOT)
        self.assertIs(result.source_integrity_verified, True)
        self.assertEqual(result.verified_source_hashes, 1)
        self.assertNotEqual(result.status, "PASS")  # clinical/row reconstruction remains unreviewed
        case["provenance"]["source_sha256"][name] = "0" * 64
        self.assertIn("SOURCE_HASH_MISMATCH", {i.code for i in validate_provenance(case, source_root=ROOT).issues})

    def test_missing_and_unsafe_original_source(self):
        result = validate_provenance(self.case, source_root=ROOT)
        self.assertIsNone(result.source_integrity_verified)
        self.assertIn("SOURCE_FILE_MISSING", {i.code for i in result.issues})
        case = deepcopy(self.case)
        case["provenance"]["source_sha256"] = {"../outside.json": "0" * 64}
        self.assertIn("UNSAFE_SOURCE_PATH", {i.code for i in validate_provenance(case, source_root=ROOT).issues})
