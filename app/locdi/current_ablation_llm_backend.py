"""C4-38 opt-in Native Ollama backend for development ER-only conditions.

No transport is constructed during static preflight. Every real generation is
reserved against the paired harness global budget before its POST. Clinical
admission remains outside this development-only adapter.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import patch

from app.locdi.current_ablation_runner import (
    CurrentAblationBlocked, _atomic_json, features_for, reserve_generation,
)
from app.locdi.llm_setup import digest, validate_setup, normalize_model_identifier
from app.locdi.ollama_adapter import (
    LocalOllamaAdapter, LLMRequestError, LLMTimeoutError, LLMResponseError,
    LLMRequestLimitError, ModelIdentityError,
)

CONTEXT = 32768
OUTPUT = 1024
MARGIN = 512
TIMEOUT = 120
REPAIR_FIELDS = {'output_contract_repair', 'repair', 'invalid_output', 'validation_error', 'original_request'}


class InputContextBudgetError(ValueError):
    pass


def static_preflight(config):
    """Read local setup only. No /api calls, no model loading, no client."""
    result = validate_setup(deepcopy(config))
    valid = result['static_readiness'] == 'READY'
    fixed = (config.get('temperature') == 0 and config.get('seed') == 42
             and config.get('max_output_tokens') == OUTPUT and config.get('timeout_seconds') == TIMEOUT
             and config.get('request_retries') == 0)
    return {'status': 'STATIC_CONTRACT_READY' if valid and fixed else 'BLOCKED',
        'checks': result.get('checks'), 'settings_fingerprint': result['prepared']['fingerprint'] if result.get('prepared') else None,
        'context': CONTEXT, 'context_applied': 'NOT_CHECKED_NO_MODEL_REQUEST',
        'native_client': 'lazy /api/chat, streamfalse, JSON Schema, retry/repair0',
        'fixed_request_options_match': fixed, 'LLM_generation': 0, 'server_requests': 0}


def input_guard(system, user, schema, *, tokenizer_counter=None):
    """Conservative byte bound includes both messages AND the JSON Schema.

    The256-byte operational framing allowance is explicitly additional, not
    a claim of HF/server tokenizer equivalence. A counter can tighten checks,
    but never override a failing conservative bound or truncate FullHistory.
    """
    if not isinstance(system, str) or not isinstance(user, str) or not isinstance(schema, dict):
        raise InputContextBudgetError('EXPLICIT_TEXT_AND_SCHEMA_REQUIRED')
    text = json.dumps({'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                       'format': schema}, sort_keys=True, ensure_ascii=False, allow_nan=False)
    byte_bound = len(text.encode('utf-8')) + 256
    if byte_bound + OUTPUT + MARGIN > CONTEXT:
        raise InputContextBudgetError('FULL_INPUT_CONSERVATIVE_CONTEXT_LIMIT')
    counted = None
    if tokenizer_counter is not None:
        counted = tokenizer_counter.count(system, user)
        if type(counted) is not int or counted < 0 or counted + OUTPUT + MARGIN > CONTEXT:
            raise InputContextBudgetError('TOKENIZER_CONTEXT_LIMIT')
    return {'conservative_input_token_bound': byte_bound, 'optional_HF_prompt_tokens': counted,
            'schema_UTF8_bytes': len(json.dumps(schema, ensure_ascii=False).encode('utf-8')),
            'context': CONTEXT, 'max_output': OUTPUT, 'margin': MARGIN, 'input_truncated': False}


def _strict_object(content):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    def constant(_):
        raise ValueError('NONFINITE_JSON')
    obj = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(obj, dict):
        raise ValueError('JSON_OBJECT_REQUIRED')
    return obj


def safe_input_bound(system, user, schema):
    """Transport-free exact preflight for the same conservative runtime guard."""
    return input_guard(system, user, schema)


def _native_transport(prepared):
    import httpx
    return httpx.Client(timeout=TIMEOUT, trust_env=False, follow_redirects=False)


class BudgetedNativeERClient(LocalOllamaAdapter):
    """Existing real-client type, explicit Native request contract, no retries."""
    def __init__(self, config, *, authorized=False, contract_fingerprint=None,
                 checkpoint_root=None, tokenizer_counter=None, response_contract=None):
        if (authorized is not True or not isinstance(contract_fingerprint, str)
                or len(contract_fingerprint) != 64 or checkpoint_root is None):
            raise CurrentAblationBlocked('EXPLICIT_EXECUTION_AUTHORITY_AND_PLAN_REQUIRED')
        super().__init__(config, authorized=authorized)
        if static_preflight(config)['status'] != 'STATIC_CONTRACT_READY':
            raise CurrentAblationBlocked('NATIVE_FIXED_OPTIONS_NOT_READY')
        self.contract_fingerprint = contract_fingerprint
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.checkpoint_root.mkdir(parents=True, exist_ok=False)
        self.tokenizer_counter = tokenizer_counter
        self.native_http = None
        self.context_verified = False
        self.last_server_usage = None
        self.repair_requests_blocked = 0
        self.fatal_failure = False
        if response_contract not in (None, 'memory_citation_index_set.v1', 'memory_citation_subset_code.v1'):
            raise CurrentAblationBlocked('UNKNOWN_RESPONSE_WIRE_CONTRACT')
        self.response_contract = response_contract

    def close(self):
        if self.native_http is not None:
            self.native_http.close()
        super().close()

    def public_metadata(self):
        result = super().public_metadata()
        result.update(transport='OLLAMA_NATIVE_CHAT', context=CONTEXT,
                      context_verified=self.context_verified, retry=0, repair=0,
                      repair_requests_blocked=self.repair_requests_blocked,
                      contract_fingerprint=self.contract_fingerprint)
        if self.response_contract is not None:
            result['response_wire_contract'] = self.response_contract
        return result

    def _get(self, path):
        response = self.native_http.get(self.prepared['endpoint'][:-3] + path)
        response.raise_for_status()
        if len(response.content) > 1024**2:
            raise ModelIdentityError('METADATA_BYTE_LIMIT')
        return response.json()

    def _identity(self, model, *, loaded=False):
        data = self._get('/api/ps' if loaded else '/api/tags')
        self.identity_check_count += 1
        entries = [r for r in data.get('models', []) if r.get('name') == model or r.get('model') == model]
        try:
            matched = len(entries) == 1 and normalize_model_identifier(entries[0].get('digest')) == self.prepared['config']['model_identifier']
        except ValueError:
            matched = False
        if not matched:
            raise ModelIdentityError('SERVER_MODEL_DIGEST_MISMATCH')
        if loaded and entries[0].get('context_length') != CONTEXT:
            raise ModelIdentityError('ACTUAL_LOADED_CONTEXT_MISMATCH')
        self.last_observed_model_identifier = normalize_model_identifier(entries[0]['digest'])
        if loaded:
            self.context_verified = True
        return {k: entries[0].get(k) for k in ('name', 'model', 'digest', 'context_length', 'size', 'size_vram')}

    def chat_json(self, system_prompt, user_prompt, model=None, *, schema=None):
        self.last_response_content = None
        config = self.prepared['config']
        if self.closed:
            raise LLMRequestError('CLIENT_CLOSED')
        if model != config['model'] or schema is None:
            raise LLMRequestError('MODEL_OR_SCHEMA_NOT_PINNED')
        try:
            payload = _strict_object(user_prompt)
        except (ValueError, TypeError):
            raise LLMResponseError('ROLE_PAYLOAD_MUST_BE_JSON_OBJECT') from None
        if REPAIR_FIELDS & set(payload):
            self.repair_requests_blocked += 1
            self.errors.append('LLMRequestLimitError')
            raise LLMRequestLimitError('REPAIR_NOT_AUTHORIZED')
        original_schema = schema
        citation_mapping = None
        if self.response_contract is not None:
            if self.response_contract == 'memory_citation_subset_code.v1':
                from app.locdi.memory_subset_wire import encode
            else:
                from app.locdi.memory_citation_wire import encode
            payload, schema, citation_mapping = encode(payload, schema)
            user_prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if self.fatal_failure:
            raise LLMRequestLimitError('PREVIOUS_REQUEST_FAILURE_NO_RESUME')
        if self.request_count >= config['max_requests']:
            raise LLMRequestLimitError('PER_CLIENT_REQUEST_CAP')
        try:
            guard = input_guard(system_prompt, user_prompt, schema, tokenizer_counter=self.tokenizer_counter)
        except InputContextBudgetError:
            self.fatal_failure = True
            self.errors.append('InputContextBudgetError')
            raise
        current = validate_setup(config)
        if current['static_readiness'] != 'READY' or current['prepared']['fingerprint'] != self.prepared['fingerprint']:
            self.fatal_failure = True
            raise ModelIdentityError('STATIC_BINDING_CHANGED')
        request = {'model': model, 'stream': False, 'format': schema,
            'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
            'options': {'num_ctx': CONTEXT, 'temperature': 0, 'seed': 42, 'num_predict': OUTPUT}}
        number = self.request_count + 1
        before = self.checkpoint_root / ('Request%04d_before.json' % number)
        after = self.checkpoint_root / ('Request%04d_after.json' % number)
        if self.native_http is None:
            self.native_http = _native_transport(self.prepared)
        try:
            identity = self._identity(model)
        except ModelIdentityError:
            self.fatal_failure = True
            self.errors.append('ModelIdentityError')
            raise
        before_record = {'status': 'BEFORE', 'contract': self.contract_fingerprint,
            'request_fingerprint': digest(request), 'guard': guard, 'identity_before': identity,
            'request': request, 'at': datetime.now(timezone.utc).isoformat()}
        if self.response_contract is not None:
            before_record.update(response_wire_contract=self.response_contract,
                                 internal_schema_fingerprint=digest(original_schema))
        _atomic_json(before, before_record)
        start = time.monotonic()
        # Includes failing POSTs; no reservation means no generation request.
        try:
            reserve_generation()
        except Exception as exc:
            self.fatal_failure = True
            _atomic_json(after, {'status': 'BLOCKED_BEFORE_GENERATION', 'request_number': number,
                'generation_attempted': False, 'error_type': type(exc).__name__, 'request_fingerprint': digest(request)})
            raise
        self.request_count += 1
        record = {'status': 'FAILED', 'request_number': number, 'request_fingerprint': digest(request),
                  'response_received': False, 'HTTP_status': None}
        try:
            response = self.native_http.post(self.prepared['endpoint'][:-3] + '/api/chat', json=request)
            record['HTTP_status'] = response.status_code
            response.raise_for_status()
            if len(response.content) > 2*1024**2:
                raise LLMResponseError('RESPONSE_BYTE_LIMIT')
            envelope = response.json(); record['response_received'] = True
            record['response_envelope'] = envelope
            if envelope.get('model') != model:
                raise ModelIdentityError('RESPONSE_MODEL_MISMATCH')
            content = envelope['message']['content']
            self.last_response_content = content
            loaded = self._identity(model, loaded=True)
            record['identity_after'] = loaded
            if envelope.get('done') is not True or envelope.get('done_reason') == 'length':
                raise LLMResponseError('INCOMPLETE_OR_TRUNCATED_RESPONSE')
            count = envelope.get('prompt_eval_count')
            if type(count) is not int or count < 0 or count + OUTPUT + MARGIN > CONTEXT:
                raise LLMResponseError('SERVER_CONTEXT_TOKEN_BOUND_UNVERIFIED')
            if count > guard['conservative_input_token_bound']:
                raise LLMResponseError('SERVER_EXCEEDS_CONSERVATIVE_PROMPT_BOUND')
            if type(envelope.get('eval_count')) is not int or not 0 <= envelope['eval_count'] <= OUTPUT:
                raise LLMResponseError('SERVER_OUTPUT_TOKEN_BOUND')
            value = _strict_object(content)
            from jsonschema import Draft202012Validator
            try:
                Draft202012Validator.check_schema(schema); Draft202012Validator(schema).validate(value)
            except Exception:
                raise LLMResponseError('JSON_SCHEMA_INVALID') from None
            if self.response_contract is not None:
                if self.response_contract == 'memory_citation_subset_code.v1':
                    from app.locdi.memory_subset_wire import decode
                else:
                    from app.locdi.memory_citation_wire import decode
                value = decode(value, citation_mapping, original_schema)
                content = json.dumps(value, ensure_ascii=False, sort_keys=True)
                record['response_wire_contract'] = self.response_contract
                record['internal_response_fingerprint'] = digest(value)
                record['internal_response'] = value
            self.last_server_usage = {'prompt_tokens': count, 'completion_tokens': envelope['eval_count']}
            from app.llm.role_call import response_usage, response_metadata
            response_usage.set(self.last_server_usage)
            response_metadata.set({'finish_reason': 'stop', 'native_done_reason': envelope.get('done_reason'),
                                   'context': CONTEXT, 'model_identifier': self.last_observed_model_identifier})
            record['status'] = 'VALID'
            return content
        except Exception as exc:
            self.fatal_failure = True
            self.errors.append(type(exc).__name__)
            record['error_type'] = type(exc).__name__
            if isinstance(exc, ModelIdentityError):
                raise
            if isinstance(exc, TimeoutError) or 'timeout' in type(exc).__name__.casefold():
                raise LLMTimeoutError('NATIVE_REQUEST_TIMEOUT') from None
            if isinstance(exc, (ValueError, KeyError, TypeError)):
                raise LLMResponseError('NATIVE_RESPONSE_CONTRACT_FAILED') from None
            raise LLMRequestError('NATIVE_REQUEST_FAILED') from None
        finally:
            record['wall_seconds'] = time.monotonic() - start
            _atomic_json(after, record)


def make_er_backend(cases_by_ref, *, config, contract_fingerprint, checkpoint_root,
                    authorized=False, tokenizer_counter=None, diagnosis_catalog=None,
                    sequential_source_view=None, response_contract=None, history_policy=None):
    """Closure keeps complete LoCDI wrappers away from the harness input surface.

    The harness passes {'case_ref': ...}; no reference is returned to model
    observation construction. Existing run_cases validates/project the retained
    original wrapper internally, without Memory truncation.
    """
    if authorized is not True:
        raise CurrentAblationBlocked('EXPLICIT_BACKEND_AUTHORITY_REQUIRED')
    if static_preflight(config)['status'] != 'STATIC_CONTRACT_READY':
        raise CurrentAblationBlocked('STATIC_BACKEND_PREFLIGHT_BLOCKED')
    # Explicit opt-in output contract. None preserves the historical free-text
    # path; the same immutable copy is used for every episode in this closure.
    if diagnosis_catalog is not None:
        if (not isinstance(diagnosis_catalog, (list, tuple)) or not diagnosis_catalog
                or any(not isinstance(x, str) or not x.strip() for x in diagnosis_catalog)
                or len(set(diagnosis_catalog)) != len(diagnosis_catalog)):
            raise CurrentAblationBlocked('INVALID_EXPLICIT_DIAGNOSIS_CATALOG')
        diagnosis_catalog = tuple(diagnosis_catalog)
    originals = deepcopy(cases_by_ref)
    count = [0]
    def run_one(execution_case, features):
        if set(execution_case) != {'case_ref'} or execution_case['case_ref'] not in originals:
            raise CurrentAblationBlocked('OPAQUE_CASE_REFERENCE_REQUIRED')
        if features not in [features_for(c) for c in ('B0', 'B1', 'B2')]:
            raise CurrentAblationBlocked('ER_ONLY_B0_B1_B2_BACKEND')
        from app.locdi.run_simulation import SimulationOptions, run_case, _SOCKET_LOCK
        from app.locdi.case_adapter import adapt_case
        from app.locdi.episode_exporter import build_episode
        from app.llm.role_call import role_call
        import app.core.hospital_llm_policy as policy_module
        case = deepcopy(originals[execution_case['case_ref']])
        count[0] += 1
        folder = Path(checkpoint_root) / ('Episode%04d' % count[0])
        client_options = dict(authorized=True, contract_fingerprint=contract_fingerprint,
                              checkpoint_root=folder, tokenizer_counter=tokenizer_counter)
        if response_contract is not None:
            client_options['response_contract'] = response_contract
        client = BudgetedNativeERClient(config, **client_options)
        decisions = []
        def observed_call(*args, **kwargs):
            if sequential_source_view is not None and kwargs.get('role') == 'er':
                kwargs = dict(kwargs)
                kwargs['payload'] = sequential_source_view.compact_payload(kwargs['payload'])
            value = role_call(*args, **kwargs)
            if kwargs.get('role') == 'er' and isinstance(value, dict):
                decisions.append(deepcopy(value))
            return value
        start = time.monotonic()
        try:
            options = SimulationOptions(purpose='development_llm', allow_llm=True, dry_run=False,
                client=client, llm_config=config, minute_limit=config['minute_limit'],
                registry_manifest=client.prepared['registry'], runner_options=deepcopy(features),
                accept_recorded_release_assumption=True, max_retries=0, max_memory_chars=None,
                diagnosis_catalog=diagnosis_catalog, sequential_source_view=sequential_source_view,
                history_policy=history_policy if features['memory_mode']=='full_history' else None)
            with _SOCKET_LOCK, patch.object(policy_module, 'role_call', new=observed_call):
                raw = run_case(case, options=options)
            _atomic_json(folder/'raw_runner_result.json', raw)
            _atomic_json(folder/'validated_ER_decisions.json', decisions)
            if raw['status'] == 'failed':
                raise CurrentAblationBlocked('LLM_EPISODE_TECHNICAL_FAILURE')
            context = adapt_case(case, registry_manifest=client.prepared['registry'])
            episode = build_episode(raw, context=context)
            terminal = next((r for r in reversed(decisions) if r.get('action') == 'diagnose'), None)
            success = raw['status'] == 'completed' and terminal is not None
            supported = success and terminal.get('certainty') == 'supported'
            prediction = {'certainty': terminal.get('certainty') if success else 'not_available',
                'diagnosis': terminal.get('target') if supported else None,
                'department': terminal.get('department') if success else None,
                'algorithmic_diagnosis_choice': None}
            counts = episode['result']['action_counts']
            actions = {'Ask': counts.get('ask_history', 0), 'Test': counts.get('order_test', 0),
                'Consult': counts.get('request_specialty', 0),
                'Diagnose': sum(counts.get(k, 0) for k in ('er_finalize', 'specialty_finalize', 'specialist_finalize'))}
            visits = (raw.get('execution') or {}).get('visits', [])
            memory = [r['memory_id'] for r in visits[0]['memory_context']['records']] if visits else None
            _atomic_json(folder/'episode_and_raw.json', {'raw': raw, 'episode': episode, 'validated_ER_decisions': decisions})
            return {'status': raw['status'], 'prediction': prediction, 'request_count': client.request_count,
                'action_counts': actions if raw.get('execution') is not None else None,
                'memory_selected_ids': memory, 'routing_call_count': 0, 'routing_abstention_count': 0,
                'identity_check_count': client.identity_check_count,
                'computation': {'wall_seconds': time.monotonic()-start, 'planner_calls': 0, 'evaluated_states': 0},
                'client_metadata': client.public_metadata(), 'backend_private_episode_fingerprint': digest(episode),
                'clinical_metrics': 'NOT_EVALUATED'}
        finally:
            client.close()
    return run_one
