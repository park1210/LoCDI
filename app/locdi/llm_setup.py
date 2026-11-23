"""Static-only development LLM setup checks. Never probes a server."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
from app.core.agent_registry import AgentRegistry
from app.llm.prompt_loader import BASE_PHYSICIAN_GUARD, er_system_prompt, specialty_system_prompt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/development_llm.json"
REQUIRED = {"schema_version", "purpose", "ready", "endpoint", "model", "model_version", "model_identifier",
            "allowed_models", "temperature", "seed", "max_output_tokens", "timeout_seconds", "request_retries",
            "max_requests", "minute_limit", "models_root", "registry_path", "agent_bindings", "patient_prompt"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def normalize_model_identifier(value):
    """Accept only lowercase 64-hex SHA-256, optionally prefixed by sha256:."""
    if not isinstance(value, str) or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value):
        raise ValueError("Invalid model SHA-256 identifier")
    return value if value.startswith("sha256:") else "sha256:" + value


def local_endpoint(value):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.path != "/v1" or parsed.port is None or not 1 <= parsed.port <= 65535):
            raise ValueError()
        host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
        return "http://{}:{}/v1".format(host, parsed.port)
    except (TypeError, ValueError, AttributeError):
        raise ValueError("Only explicit loopback HTTP Ollama /v1 endpoints are allowed") from None


def project_file(value):
    if not isinstance(value, str):
        raise ValueError("Project-relative file path is required")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("File path must remain inside the project")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        raise ValueError("File path escapes project") from None
    if not path.is_file():
        raise ValueError("Required project file is missing")
    return path


def load_setup(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_local_model(model, models_root=None):
    """Manifest identifier + referenced-file existence/size; no blob hash claim."""
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", model):
        raise ValueError("Use an explicit library model:tag")
    root = Path(models_root or os.environ.get("OLLAMA_MODELS", str(Path.home() / ".ollama/models"))).resolve()
    name, tag = model.split(":")
    path = root / "manifests/registry.ollama.ai/library" / name / tag
    if not path.is_file():
        return {"status": "BLOCKED", "reason": "manifest_missing", "model_identifier": None}
    raw = path.read_bytes()
    manifest = json.loads(raw)
    entries = [manifest["config"]] + manifest["layers"]
    for entry in entries:
        value = entry["digest"]
        if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("Invalid local manifest digest")
        blob = root / "blobs" / value.replace(":", "-")
        if not blob.is_file() or blob.stat().st_size != entry["size"]:
            return {"status": "BLOCKED", "reason": "blob_missing_or_size_mismatch", "model_identifier": None}
    # Recent Ollama may append the byte-exact upstream manifest as one layer
    # in its local wrapper. /api/tags identifies that upstream manifest. Accept
    # it only after hashing the small blob and proving all config/layer entries
    # identical to the outer manifest (except that single wrapper layer).
    manifest_layers = [e for e in manifest['layers']
                       if e.get('mediaType') == 'application/vnd.docker.distribution.manifest.v2+json']
    if manifest_layers:
        if len(manifest_layers) != 1 or manifest_layers[0]['size'] > 1024**2:
            raise ValueError('Invalid upstream manifest wrapper')
        entry = manifest_layers[0]
        upstream_raw = (root / 'blobs' / entry['digest'].replace(':', '-')).read_bytes()
        if 'sha256:' + hashlib.sha256(upstream_raw).hexdigest() != entry['digest']:
            raise ValueError('Upstream manifest hash mismatch')
        upstream = json.loads(upstream_raw)
        if (set(upstream) != {'schemaVersion', 'mediaType', 'config', 'layers'}
                or upstream['schemaVersion'] != 2 or upstream['config'] != manifest['config']
                or upstream['mediaType'] != manifest['mediaType']
                or upstream['layers'] != [e for e in manifest['layers'] if e != entry]):
            raise ValueError('Upstream manifest content differs from wrapper')
        return {'status': 'READY', 'reason': 'upstream_manifest_hash_and_wrapper_content_checked',
                'model_identifier': entry['digest'],
                'local_wrapper_sha256': hashlib.sha256(raw).hexdigest(),
                'blob_content_integrity': 'UPSTREAM_MANIFEST_ONLY; model weight blobs size-checked',
                'server_availability': 'NOT_CHECKED'}
    return {"status": "READY", "reason": "manifest_and_blob_sizes_checked",
            "model_identifier": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "blob_content_integrity": "NOT_CHECKED", "server_availability": "NOT_CHECKED"}


def validate_setup(config):
    """Return sanitized states and private prepared bindings, without networking.

    READY here is static contract readiness, not server/model quality readiness.
    Template nulls are BLOCKED; malformed or unsupported settings are INVALID.
    """
    states, prepared = {}, None
    def check(key, operation):
        try:
            value = operation()
            states[key] = {"status": "READY"}
            return value
        except FileNotFoundError:
            states[key] = {"status": "BLOCKED", "reason": "required_file_missing"}
        except Exception:
            states[key] = {"status": "INVALID", "reason": "contract_invalid"}
        return None

    if not isinstance(config, dict) or set(config) != REQUIRED or config.get("schema_version") != 1 or config.get("purpose") != "development_llm" or type(config.get("ready")) is not bool:
        return {"static_readiness": "INVALID", "execution_readiness": "BLOCKED", "checks": {"configuration": {"status": "INVALID"}}, "prepared": None}
    config = deepcopy(config)
    endpoint = check("endpoint", lambda: local_endpoint(config["endpoint"]))
    def settings():
        pending = ("model_version", "model_identifier", "temperature", "seed", "max_output_tokens", "timeout_seconds", "request_retries", "max_requests", "minute_limit")
        if any(config[k] is None for k in pending):
            return False
        if (not isinstance(config["allowed_models"], list) or not config["allowed_models"]
                or config["model"] not in config["allowed_models"]
                or not isinstance(config["model_version"], str) or config["model_version"] == "UNCONFIRMED"
                or not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,120}", config["model_version"])
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", config["model_identifier"])):
            raise ValueError()
        for key in ("seed", "max_output_tokens", "max_requests", "minute_limit", "request_retries"):
            if type(config[key]) is not int:
                raise ValueError()
        if config["request_retries"] != 0 or not 1 <= config["max_requests"] <= 1000 or not 1 <= config["max_output_tokens"] <= 32768 or not 1 <= config["minute_limit"] <= 1440:
            raise ValueError()
        for key, lower, upper in (("temperature", 0, 2), ("timeout_seconds", 0.1, 600)):
            value = config[key]
            if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError()
        return True
    complete = check("generation_settings", settings)
    if complete is False:
        states["generation_settings"] = {"status": "BLOCKED", "reason": "user_settings_unconfirmed"}
    manifest = check("registry", lambda: AgentRegistry(json.loads(project_file(config["registry_path"]).read_text(encoding="utf-8"))).snapshot())
    local = check("local_model_files", lambda: inspect_local_model(config["model"], config["models_root"]))
    if local is not None:
        states["local_model_files"] = local
    def prompts():
        if manifest is None or not isinstance(config["agent_bindings"], list):
            raise ValueError()
        agents = {a["agent_id"]: a for a in manifest["agents"]}
        if len(config["agent_bindings"]) != len(agents):
            raise ValueError()
        hashes, seen = {}, set()
        for binding in config["agent_bindings"]:
            if set(binding) != {"agent_id", "prompt", "available", "model_identifier", "shared_base_model"}:
                raise ValueError()
            agent = agents[binding["agent_id"]]
            expected = "prompts/er_department_system.txt" if agent["role"] == "er" else "prompts/specialty/{}.txt".format(agent["department"])
            if binding["agent_id"] in seen or binding["prompt"] != expected or binding["shared_base_model"] is not True:
                raise ValueError()
            seen.add(binding["agent_id"])
            text = project_file(binding["prompt"]).read_text(encoding="utf-8").strip()
            actual = er_system_prompt() if agent["role"] == "er" else specialty_system_prompt(agent["department"])
            if not text or actual != BASE_PHYSICIAN_GUARD + "\n\n" + text:
                raise ValueError("Prompt cache or path mismatch")
            hashes[binding["agent_id"]] = hashlib.sha256(actual.encode("utf-8")).hexdigest()
        if config["patient_prompt"] != "prompts/patient_system.txt" or not project_file(config["patient_prompt"]).read_text(encoding="utf-8").strip():
            raise ValueError()
        return hashes
    prompt_hashes = check("prompts", prompts)
    def bindings():
        if manifest is None or complete is not True or local is None or local["status"] != "READY":
            return False
        if local["model_identifier"] != config["model_identifier"]:
            raise ValueError()
        return all(a["model"] == config["model"] and a["version"] == config["model_version"] for a in manifest["agents"]) and all(
            b["available"] is True and b["model_identifier"] == config["model_identifier"] for b in config["agent_bindings"])
    bound = check("agent_bindings", bindings)
    if bound is False:
        states["agent_bindings"] = {"status": "BLOCKED", "reason": "binding_not_confirmed"}
    def libraries():
        from app.llm.ollama_client import OpenAI
        if OpenAI is None or importlib.util.find_spec("jsonschema") is None:
            return False
        import inspect
        from openai.resources.chat.completions import Completions
        return {"seed", "temperature", "max_tokens", "response_format"}.issubset(inspect.signature(Completions.create).parameters)
    libraries_ready = check("client_interface", libraries)
    if libraries_ready is False:
        states["client_interface"] = {"status": "BLOCKED", "reason": "installed_client_capability_missing"}
    states.update(server={"status": "NOT_CHECKED"}, server_model_identity={"status": "NOT_CHECKED"}, response_quality={"status": "NOT_CHECKED"})
    static = "INVALID" if any(s["status"] == "INVALID" for s in states.values()) else "BLOCKED" if any(s["status"] == "BLOCKED" for s in states.values()) else "READY"
    if static == "READY":
        prepared = {"config": config, "endpoint": endpoint, "registry": manifest, "prompt_fingerprints": prompt_hashes,
                    "fingerprint": digest({"config": config, "registry": manifest, "prompts": prompt_hashes})}
    return {"static_readiness": static, "execution_readiness": "NOT_CHECKED" if static == "READY" else "BLOCKED",
            "research_readiness": "BLOCKED", "checks": states, "prepared": prepared}
