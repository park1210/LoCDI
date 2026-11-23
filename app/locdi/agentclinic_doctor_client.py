"""Versioned JSON doctor. Fake transport stays offline; live construction is gated."""
from abc import ABC, abstractmethod
from copy import deepcopy
import hashlib
import json
from jsonschema import Draft202012Validator
from app.locdi.llm_setup import ROOT, digest
from app.locdi.reference_acquisition import strict_json
from app.locdi.run_simulation import _offline_network_guard
from app.llm.role_call import role_call

PROMPT_VERSION = 'agentclinic.doctor.v1'
PROMPT_PATH = ROOT/'prompts/agentclinic/v1/doctor_system.txt'
PROMPT_SHA256 = '201d5b54620b700c90da9992312314f762a4f8b1df2cb5d2a5f12fed3a5fe1bf'
CLIENT_VERSION = 'agentclinic.json_doctor.v1'
CATALOG_POLICY = 'SOURCE_BRANCH_LABELS_EXACT_IDS_V1'


def prompt_binding(profile=None):
    if profile is None or profile=='A':return PROMPT_VERSION,PROMPT_PATH,PROMPT_SHA256
    if profile=='B':return ('agentclinic.doctor.repetition_aware.v1',
        ROOT/'prompts/agentclinic/v2_repetition_aware/doctor_system.txt',
        '26ec93b61568c67ccf9233972f9c3927d7363fa0dce4a318a1ea89b4843d88bd')
    raise ValueError('FIXED_PROMPT_PROFILE_REQUIRED')


def system_prompt(profile=None):
    _,path,expected=prompt_binding(profile)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('DOCTOR_PROMPT_HASH_MISMATCH')
    return raw.decode('utf8')


def action_schema(catalog):
    branches = []
    for kind in ('ASK', 'EXAMINE', 'TEST'):
        targets = [t['target'] for t in catalog[kind]]
        if targets:
            branches.append({'type': 'object', 'properties': {
                'action': {'const': kind}, 'target': {'type': 'string', 'enum': targets}},
                'required': ['action', 'target'], 'additionalProperties': False})
    branches.append({'type': 'object', 'properties': {
        'action': {'const': 'DIAGNOSE'}, 'predicted_diagnosis': {'type': 'string', 'maxLength': 4096},
        'unresolved_reason': {'type': 'string', 'maxLength': 1000}},
        'required': ['action', 'predicted_diagnosis'], 'additionalProperties': False})
    return {'oneOf': branches}


def build_request(observation, *, prompt_profile=None):
    allowed = {'scenario_ref', 'objective', 'catalog', 'observations', 'turn_index',
               'remaining_requests', 'last_action', 'history'}
    if not isinstance(observation, dict) or set(observation) - allowed:
        raise ValueError('DOCTOR_OBSERVATION_FIELDS')
    if not {'scenario_ref', 'objective', 'catalog', 'observations', 'turn_index', 'remaining_requests'} <= set(observation):
        raise ValueError('DOCTOR_OBSERVATION_REQUIRED')
    def private(v):
        if isinstance(v, dict):
            return any(k.casefold() in {'correct_diagnosis', 'gold', 'scoring_key', 'evaluation',
                                        'test_results', 'physical_examination_findings'} or private(x)
                       for k, x in v.items())
        return isinstance(v, list) and any(private(x) for x in v)
    if private(observation):
        raise ValueError('PRIVATE_STORE_IN_DOCTOR_REQUEST')
    c = observation['catalog']
    if not isinstance(c, dict) or set(c) != {'ASK', 'EXAMINE', 'TEST'}:
        raise ValueError('DOCTOR_CATALOG')
    for kind, rows in c.items():
        if not isinstance(rows, list) or any(set(t) != {'target', 'label', 'node_kind'}
                or not all(isinstance(v, str) for v in t.values()) for t in rows):
            raise ValueError('DOCTOR_CATALOG')
    schema = action_schema(c)
    payload = deepcopy(observation)
    payload.update(catalog_policy=CATALOG_POLICY, response_schema=schema)
    return system_prompt(prompt_profile), payload, schema


def validate_action(raw, schema):
    try:
        value = strict_json(raw)
        Draft202012Validator(schema).validate(value)
    except Exception:
        raise ValueError('DOCTOR_JSON_ACTION_CONTRACT') from None
    return value


class OfflineJsonTransport(ABC):
    @abstractmethod
    def chat_json(self, system, user, model=None, *, schema=None):
        """TEST_ONLY transport. Network is prohibited by FakeDoctorClient."""


class _JsonDoctor:
    def _initialize(self, transport, model, request_budget, max_prompt_bytes, prompt_profile=None):
        if type(request_budget) is not int or not 1 <= request_budget <= 1000:
            raise ValueError('DOCTOR_REQUEST_BUDGET')
        if type(max_prompt_bytes) is not int or not 256 <= max_prompt_bytes <= 262144:
            raise ValueError('DOCTOR_INPUT_BYTE_BUDGET')
        self.transport, self.model = transport, model
        self.request_budget, self.max_prompt_bytes = request_budget, max_prompt_bytes
        self.request_count, self.audit = 0, []
        self.prompt_profile=prompt_profile
        self.prompt_sha256=prompt_binding(prompt_profile)[2]

    def _choose(self, observation):
        if self.request_count >= self.request_budget:
            raise ValueError('DOCTOR_GLOBAL_BUDGET_EXHAUSTED')
        system, payload, schema = build_request(observation,prompt_profile=self.prompt_profile)
        size = len(system.encode('utf8')) + len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf8'))
        if size > self.max_prompt_bytes:
            raise ValueError('DOCTOR_PROMPT_BYTE_LIMIT')
        self.request_count += 1
        return role_call(self.transport, role='agentclinic_doctor', model=self.model, system=system,
                         payload=payload, audit=self.audit, schema=schema,
                         validate=lambda raw: validate_action(raw, schema), repair_codes=())


class FakeDoctorClient(_JsonDoctor):
    def __init__(self, transport, *, request_budget=1000, max_prompt_bytes=131072, prompt_profile=None):
        if not isinstance(transport, OfflineJsonTransport):
            raise TypeError('EXPLICIT_TEST_ONLY_TRANSPORT_REQUIRED')
        self._initialize(transport, 'TEST_ONLY_MODEL', request_budget, max_prompt_bytes,prompt_profile)

    def choose_action(self, observation):
        with _offline_network_guard():
            return self._choose(observation)


_LIVE_CAPABILITY = object()


class LiveDoctorClient(_JsonDoctor):
    def __init__(self, definition, authority, *, _capability=None):
        if _capability is not _LIVE_CAPABILITY:
            raise ValueError('PINNED_LIVE_APPROVAL_REQUIRED')
        from app.locdi.agentclinic_live_preflight import preflight
        from app.locdi.ollama_adapter import LocalOllamaAdapter
        report = preflight(definition)
        if report['static_blockers']:
            raise ValueError('LIVE_STATIC_PREFLIGHT_BLOCKED')
        authority.verify(report['plan'])
        self.definition, self.authority = deepcopy(definition), authority
        self.plan_fingerprint = digest(report['plan'])
        self.plan = deepcopy(report['plan'])
        self._initialize(LocalOllamaAdapter(definition['legacy_config'], authorized=True),
                         definition['legacy_config']['model'], definition['total_generation_budget'],
                         definition['max_prompt_bytes'],definition.get('prompt_profile'))

    def choose_action(self, observation):
        from app.locdi.agentclinic_live_preflight import preflight
        current = preflight(self.definition)
        if current['static_blockers'] or digest(current['plan']) != self.plan_fingerprint:
            raise ValueError('LIVE_BINDING_CHANGED')
        self.authority.verify(current['plan'])
        binding = current['plan']['scenarios'].get(observation.get('scenario_ref'))
        if (binding is None or digest(observation.get('objective')) != binding['objective_fingerprint']
                or digest(observation.get('catalog')) != binding['catalog_fingerprint']):
            raise ValueError('LIVE_SCENARIO_BINDING')
        return self._choose(observation)

    def close(self):
        self.transport.close()


def create_live_client(definition, *, execute=False, allow_llm=False, authority=None):
    from app.locdi.agentclinic_live_preflight import preflight, PinnedDoctorApproval
    report = preflight(definition)
    if not execute or not allow_llm or not isinstance(authority, PinnedDoctorApproval):
        raise ValueError('EXPLICIT_PINNED_LIVE_APPROVAL_REQUIRED')
    if report['static_blockers']:
        raise ValueError('LIVE_STATIC_PREFLIGHT_BLOCKED')
    authority.verify(report['plan'])
    return LiveDoctorClient(definition, authority, _capability=_LIVE_CAPABILITY)
