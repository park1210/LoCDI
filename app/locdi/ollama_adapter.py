"""Opt-in local Ollama adapter using the unchanged legacy OllamaClient."""
import json
from types import SimpleNamespace
from app.llm.ollama_client import OllamaClient
from app.locdi.llm_setup import validate_setup, normalize_model_identifier


class LLMRequestError(RuntimeError):
    pass


class LLMTimeoutError(TimeoutError):
    pass


class LLMResponseError(ValueError):
    pass


class LLMRequestLimitError(RuntimeError):
    pass


class ModelIdentityError(RuntimeError):
    pass


def _backend(prepared):
    """No connections until identity GET or chat; proxy env/redirects disabled."""
    import httpx
    config = prepared["config"]
    http = httpx.Client(timeout=config["timeout_seconds"], trust_env=False, follow_redirects=False)
    legacy = OllamaClient(base_url=prepared["endpoint"], api_key="ollama")
    if legacy.client is None:
        http.close()
        raise LLMRequestError("Installed client backend is unavailable")
    original = legacy.client
    sdk = original.with_options(timeout=config["timeout_seconds"], max_retries=0, http_client=http)
    def create(**kwargs):
        kwargs.update(temperature=config["temperature"], max_tokens=config["max_output_tokens"], seed=config["seed"])
        return sdk.chat.completions.create(**kwargs)
    legacy.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return legacy, http, original, sdk


class LocalOllamaAdapter:
    """A real-client type, deliberately unrelated to OfflineClient.

    Construction requires authorization and complete static validation. Even
    construction is lazy: network calls occur only through explicit chat_json.
    Tests patch _backend; no test transport can be configured in user JSON.
    """
    def __init__(self, config, *, authorized=False):
        if authorized is not True:
            raise ValueError("Explicit real-model authorization is required")
        report = validate_setup(config)
        if report["static_readiness"] != "READY":
            raise ValueError("Local LLM configuration is not statically ready")
        self.prepared = report["prepared"]
        self.request_count = 0
        self.identity_check_count = 0
        self.errors = []
        self._resources = None
        self.closed = False
        self.last_observed_model_identifier = None
        self.last_response_content = None

    def close(self):
        self.closed = True
        if self._resources is not None:
            for resource in self._resources[1:]:
                try:
                    resource.close()
                except Exception:
                    pass

    def public_metadata(self):
        config = self.prepared["config"]
        return {"real_model_execution": self.request_count > 0, "request_count": self.request_count,
                "identity_check_count": self.identity_check_count, "model_identifier": config["model_identifier"],
                "model_version": config["model_version"], "settings_fingerprint": self.prepared["fingerprint"],
                "error_types": list(self.errors)}

    def chat_json(self, system_prompt, user_prompt, model=None, *, schema=None):
        self.last_response_content = None
        config = self.prepared["config"]
        if self.closed:
            raise LLMRequestError("Client has been closed")
        if model != config["model"] or schema is None:
            raise LLMRequestError("Requested model or response contract is not authorized")
        if self.request_count >= config["max_requests"]:
            self.errors.append("LLMRequestLimitError")
            raise LLMRequestLimitError("Explicit model request budget exhausted")
        try:
            # Revalidate file bindings/prompts and disk manifest before each call.
            current = validate_setup(config)
            if current["static_readiness"] != "READY" or current["prepared"]["fingerprint"] != self.prepared["fingerprint"]:
                raise ModelIdentityError("Static model or prompt binding changed")
            if self._resources is None:
                self._resources = _backend(self.prepared)
            legacy, http, _, _ = self._resources
            self.identity_check_count += 1
            self.last_observed_model_identifier = None
            response = http.get(self.prepared["endpoint"][:-3] + "/api/tags")
            response.raise_for_status()
            entries = response.json()["models"]
            try:
                observed = {normalize_model_identifier(entry.get("digest")) for entry in entries
                            if entry.get("name") == model or entry.get("model") == model}
                self.last_observed_model_identifier = next(iter(observed)) if len(observed) == 1 else None
                matches = [(entry.get("name") == model or entry.get("model") == model)
                           and normalize_model_identifier(entry.get("digest")) == config["model_identifier"]
                           for entry in entries if entry.get("name") == model or entry.get("model") == model]
            except ValueError:
                raise ModelIdentityError("Server model digest format is invalid") from None
            if not any(matches):
                raise ModelIdentityError("Server model identifier does not match the pinned manifest")
            self.request_count += 1
            content = legacy.chat_json(system_prompt, user_prompt, model=model, schema=schema)
            self.last_response_content = content
        except ModelIdentityError:
            self.errors.append("ModelIdentityError")
            raise
        except Exception as exc:
            # Do not copy server bodies, URLs, evidence, or SDK exception text.
            timeout = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
            kind = LLMTimeoutError if timeout else LLMRequestError
            self.errors.append(kind.__name__)
            raise kind("Local model request failed; raw details withheld") from None
        try:
            from jsonschema import Draft202012Validator
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError()
                    result[key] = value
                return result
            def constant(_):
                raise ValueError()
            value = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(value)
            if not isinstance(value, dict):
                raise ValueError()
        except Exception:
            self.errors.append("LLMResponseError")
            raise LLMResponseError("Model response violates the requested JSON contract") from None
        return content
