"""Real-client wiring tests with fake SDK/HTTP only; zero actual model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT, load_setup, validate_setup, local_endpoint, normalize_model_identifier
from app.locdi.ollama_adapter import LocalOllamaAdapter, LLMRequestError, LLMResponseError, LLMTimeoutError, LLMRequestLimitError, ModelIdentityError
from app.locdi.run_simulation import run_case, run_development_llm, SimulationOptions, SimulationConfigurationError
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.episode_exporter import build_episode, export_episode, export_summary
from test_simulation import SourceActionClient, smoke_options
from smoke_support import check_boundary


class LLMSetupContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=str(ROOT / "tests"))
        cls.root = Path(cls.directory.name)
        cls.case = load_cases(ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
        cls.context = adapt_case(cls.case)
        cls.config = load_setup()
        manifest = json.loads((ROOT / cls.config["registry_path"]).read_text(encoding="utf-8"))
        for agent in manifest["agents"]:
            agent["version"] = "test-only-v1"
        path = cls.root / "registry.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        cls.config["registry_path"] = path.relative_to(ROOT).as_posix()
        model_root = cls.root / "models"
        manifest_path = model_root / "manifests/registry.ollama.ai/library/qwen2.5/1.5b-instruct"
        manifest_path.parent.mkdir(parents=True)
        blob = b"TEST-ONLY-NOT-A-MODEL"
        value = hashlib.sha256(blob).hexdigest()
        (model_root / "blobs").mkdir()
        (model_root / "blobs" / ("sha256-" + value)).write_bytes(blob)
        raw = json.dumps({"config": {"digest": "sha256:"+value, "size": len(blob)}, "layers": []}).encode()
        manifest_path.write_bytes(raw)
        cls.identifier = "sha256:" + hashlib.sha256(raw).hexdigest()
        cls.config.update(model_version="test-only-v1", model_identifier=cls.identifier,
                          temperature=0, seed=0, max_output_tokens=768, timeout_seconds=1,
                          request_retries=0, max_requests=20, minute_limit=120, models_root=str(model_root))
        for binding in cls.config["agent_bindings"]:
            binding.update(available=True, model_identifier=cls.identifier)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def backend(self, failure=None, mismatch=False, bare_digest=False):
        source = SourceActionClient()
        calls, options, http_options = [], [], []
        identifier = self.identifier.removeprefix("sha256:") if bare_digest else self.identifier
        class HTTP:
            def __init__(self, **kwargs):
                http_options.append(kwargs)
            def get(self, url):
                data = {"models": [{"name": "qwen2.5:1.5b-instruct", "digest": "sha256:"+"0"*64 if mismatch else identifier}]}
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
            def close(self):
                pass
        class SDK:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            def with_options(self, **kwargs):
                options.append(kwargs)
                return self
            def create(self, **kwargs):
                calls.append(kwargs)
                if failure == "timeout":
                    raise TimeoutError("PRIVATE patient evidence")
                if failure == "request":
                    raise ConnectionError("PRIVATE patient evidence")
                content = (failure if failure not in (None, "timeout", "request") else
                    source.chat_json(kwargs["messages"][0]["content"], kwargs["messages"][1]["content"], model=kwargs["model"],
                                     schema=kwargs["response_format"]["json_schema"]["schema"]))
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])
            def close(self):
                pass
        return source, calls, options, http_options, patch("app.llm.ollama_client.OpenAI", SDK), patch("httpx.Client", HTTP)

    def test_default_is_static_dry_run_with_no_client_or_network(self):
        with patch("app.locdi.ollama_adapter._backend", side_effect=AssertionError("No backend")) as backend:
            report = run_development_llm(self.case)
        backend.assert_not_called()
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["static_readiness"], validate_setup(load_setup())["static_readiness"])
        self.assertEqual(report["checks"]["server"]["status"], "NOT_CHECKED")
        self.assertEqual(report["request_count"], 0)
        self.assertEqual(report["dataset_validation"]["schema"], "PASS")

    def test_unauthorized_real_and_wrong_client_types_are_blocked(self):
        with self.assertRaises(SimulationConfigurationError):
            run_development_llm(self.case, config=self.config, execute=True)
        with self.assertRaises(ValueError):
            LocalOllamaAdapter(self.config)
        with self.assertRaises(SimulationConfigurationError):
            run_case(self.case, options=SimulationOptions(purpose="development_llm", dry_run=False,
                        allow_llm=True, client=SourceActionClient(), llm_config=self.config))
        real = LocalOllamaAdapter(self.config, authorized=True)
        try:
            with self.assertRaises(SimulationConfigurationError):
                run_case(self.case, options=smoke_options(client=real))
        finally:
            real.close()

    def test_registry_model_prompt_and_unsupported_options(self):
        self.assertEqual(validate_setup(self.config)["static_readiness"], "READY")
        mutations = [("seed", None), ("registry_path", "missing.json"), ("temperature", True),
                     ("model_identifier", "sha256:"+"0"*64), ("request_retries", 1), ("top_p", 0.5)]
        for key, value in mutations:
            config = deepcopy(self.config); config[key] = value
            with self.subTest(key=key):
                self.assertNotEqual(validate_setup(config)["static_readiness"], "READY")
                with self.assertRaises(ValueError):
                    LocalOllamaAdapter(config, authorized=True)
        config = deepcopy(self.config)
        config["agent_bindings"][0]["prompt"] = "../secret.txt"
        self.assertEqual(validate_setup(config)["checks"]["prompts"]["status"], "INVALID")
        for endpoint in ("https://example.com/v1", "http://host.docker.internal:11434/v1", "http://127.0.0.1:11434/v1?x=1", "http://user@localhost:11434/v1"):
            with self.assertRaises(ValueError):
                local_endpoint(endpoint)

    def test_fake_sdk_full_core_path_and_exporter(self):
        source, calls, options, http_options, sdk, http = self.backend()
        with sdk, http:
            result = run_development_llm(self.case, config=self.config, execute=True, allow_llm=True,
                        runner_options=smoke_options().runner_options, accept_recorded_release_assumption=True)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["mock_execution"] or result["research_evaluation_eligible"])
        self.assertEqual(result["llm_execution"]["request_count"], 6)
        self.assertTrue(check_boundary(self.case, self.context, result, source)["release"])
        self.assertTrue(all(k["seed"] == 0 and k["temperature"] == 0 and k["max_tokens"] == 768 for k in calls))
        self.assertEqual(options[0]["timeout"], 1)
        self.assertEqual(options[0]["max_retries"], 0)
        self.assertFalse(http_options[0]["trust_env"] or http_options[0]["follow_redirects"])
        document = build_episode(result, context=self.context)
        self.assertFalse(document["mock_execution"])
        self.assertEqual(document["schema_version"], "locdi.episode.v1")
        self.assertEqual(document["llm_execution"]["model_identifier"], self.identifier)
        self.assertNotIn(self.case["patient_id"], json.dumps(document))
        with tempfile.TemporaryDirectory() as root:
            export_episode(result, "episodes/fake_transport.json", context=self.context, output_root=root)
            export_summary({"episodes": [document]}, "metrics/real.json", output_root=root)
            old = json.loads(next((ROOT / 'outputs/episodes').glob('*case001.json')).read_text(encoding='utf-8'))
            with self.assertRaises(ValueError):
                export_summary({"episodes": [old, document]}, "metrics/mixed.json", output_root=root)

    def test_json_schema_timeout_and_request_failures(self):
        schema = {"type": "object", "required": ["action"], "properties": {"action": {"enum": ["ask_patient"]}}, "additionalProperties": False}
        for response, exception in (("not-json", LLMResponseError), ('{"action":"unsupported"}', LLMResponseError),
                ('{"action":"ask_patient","gold":"PRIVATE"}', LLMResponseError), ("timeout", LLMTimeoutError), ("request", LLMRequestError)):
            _, _, _, _, sdk, http = self.backend(response)
            with self.subTest(response=response), sdk, http:
                client = LocalOllamaAdapter(self.config, authorized=True)
                try:
                    with self.assertRaises(exception) as caught:
                        client.chat_json("system", "{}", model=self.config["model"], schema=schema)
                    self.assertNotIn("PRIVATE", str(caught.exception))
                finally:
                    client.close()

    def test_server_model_digest_mismatch_and_request_budget(self):
        _, calls, _, _, sdk, http = self.backend(mismatch=True)
        with sdk, http:
            client = LocalOllamaAdapter(self.config, authorized=True)
            try:
                with self.assertRaises(ModelIdentityError):
                    client.chat_json("system", "{}", model=self.config["model"], schema={})
                self.assertEqual(calls, [])
            finally:
                client.close()

        config = deepcopy(self.config); config["max_requests"] = 1
        _, _, _, _, sdk, http = self.backend('{"action":"ask_patient"}')
        with sdk, http:
            client = LocalOllamaAdapter(config, authorized=True)
            try:
                client.chat_json("system", "{}", model=config["model"], schema={"type":"object"})
                with self.assertRaises(LLMRequestLimitError):
                    client.chat_json("system", "{}", model=config["model"], schema={"type":"object"})
            finally:
                client.close()

    def test_server_bare_digest_matches_without_weakening_validation(self):
        self.assertEqual(normalize_model_identifier(self.identifier[7:]), self.identifier)
        self.assertEqual(normalize_model_identifier(self.identifier), self.identifier)
        for value in (None, "", "sha256:" + "g"*64, "sha512:" + "0"*64, " " + self.identifier):
            with self.assertRaises(ValueError):
                normalize_model_identifier(value)
        _, calls, _, _, sdk, http = self.backend('{"action":"ask_patient"}', bare_digest=True)
        with sdk, http:
            client = LocalOllamaAdapter(self.config, authorized=True)
            try:
                client.chat_json("system", "{}", model=self.config["model"], schema={"type":"object"})
                self.assertEqual(len(calls), 1)
            finally:
                client.close()
