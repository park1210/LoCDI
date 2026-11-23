"""Opt-in finalization at an unchanged Frozen M5 STOP; legacy is untouched.

An extra, budgeted request is NOT a repair of the preceding generic proposal.
Its output returns to HospitalLLMPolicy's original semantic validator.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from unittest.mock import patch

from app.locdi.common_menu_backend import CommonMenuNativeClient
from app.locdi.m5_common_source import NativeM5InformationPolicy
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.run_simulation import _SOCKET_LOCK

VERSION = 'frozen_m5_stop_terminal_only.v1'


def terminal_request(payload, schema):
    """Narrow the existing Core contract, without creating evidence/permission."""
    branches = schema.get('anyOf', [schema])
    terminal = [deepcopy(b) for b in branches
                if b.get('properties', {}).get('action', {}).get('enum') == ['diagnose']]
    if not terminal:
        raise ValueError('CORE_TERMINAL_BRANCH_REQUIRED')
    p = deepcopy(payload)
    if 'diagnose' not in p.get('allowed_actions', []):
        raise ValueError('CORE_TERMINAL_PERMISSION_REQUIRED')
    p['allowed_actions'] = ['diagnose']
    p['candidate_actions'] = [c for c in p.get('candidate_actions', []) if c['action'] == 'diagnose']
    p['terminal_request_contract'] = VERSION
    p['instruction'] = p.get('instruction', '') + (
        ' Information acquisition has stopped. Return diagnose with supported certainty only '
        'when the currently supplied eligible evidence justifies it; otherwise return diagnose '
        'with unresolved certainty, if allowed by the schema. Do not request more information. '
        'Historical Memory is not current objective evidence. Do not invent citations.')
    return p, {'anyOf': terminal} if len(terminal) > 1 else terminal[0]


class TerminalOnlyClient(CommonMenuNativeClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.terminal_only_calls = 0
        self._current_call = None

    def chat_json(self, system_prompt, user_prompt, model=None, *, schema=None):
        self._current_call = (system_prompt, deepcopy(json.loads(user_prompt)), model, deepcopy(schema))
        content = super().chat_json(system_prompt, user_prompt, model, schema=schema)
        if self.last_terminal_candidate is not None:
            # Host-only callable. Never serialized or sent to a model.
            self.last_terminal_candidate['terminal_request'] = self.request_terminal
        return content

    def request_terminal(self, expected_observation):
        if self.terminal_only_calls or self._current_call is None:
            raise ValueError('ONE_TERMINAL_REQUEST_PER_EPISODE')
        system, original, model, schema = self._current_call
        if digest(original['observation']) != digest(expected_observation):
            raise ValueError('TERMINAL_SNAPSHOT_MISMATCH')
        p, s = terminal_request(original, schema)
        frozen = digest({k: original.get(k) for k in
                         ('observation', 'evidence', 'supported_diagnosis_eligible_indices')})
        if frozen != digest({k: p.get(k) for k in
                             ('observation', 'evidence', 'supported_diagnosis_eligible_indices')}):
            raise ValueError('TERMINAL_PATIENT_INFORMATION_CHANGED')
        self.terminal_only_calls += 1
        number = self.request_count + 1
        record = {'contract': VERSION, 'request_number': number, 'frozen_snapshot_fingerprint': frozen,
                  'original_observation_fingerprint': digest(expected_observation),
                  'extra_generation_budgeted': True, 'retry': 0, 'repair': 0,
                  'semantic_status': 'PENDING_ORIGINAL_CORE_REVALIDATION'}
        atomic_json(self.checkpoint_root / 'terminal_only_request.json', record)
        # Call the base directly: the terminal call cannot recursively schedule another.
        content = CommonMenuNativeClient.chat_json(self, system, json.dumps(p, ensure_ascii=False), model, schema=s)
        candidate = json.loads(content)
        if candidate.get('action') != 'diagnose':
            raise ValueError('TERMINAL_ONLY_ACTION_REQUIRED')
        return candidate


class TerminalOnlyInformationPolicy(NativeM5InformationPolicy):
    def __init__(self, model, binding, selector, terminal_candidate):
        super().__init__(model, binding, selector)
        self.terminal_candidate = terminal_candidate
        self.requested = False

    def propose(self, runtime, observed, **kwargs):
        proposal = super().propose(runtime, observed, **kwargs)
        if proposal['action'] != 'diagnose':
            return proposal
        record = self.terminal_candidate()
        if (not record or record['observation_fingerprint'] != digest(observed)
                or not callable(record.get('terminal_request')) or self.requested):
            raise ValueError('CURRENT_SINGLE_TERMINAL_REQUEST_REQUIRED')
        self.requested = True
        # No action is executed while this request is outstanding.
        before = digest(runtime.to_summary())
        candidate = record['terminal_request'](deepcopy(observed))
        if before != digest(runtime.to_summary()):
            raise ValueError('TERMINAL_RUNTIME_OR_TIME_CHANGED')
        trace = runtime.voi_state['history'][-1]
        trace.update(terminal_contract=VERSION, terminal_extra_requests=1,
                     terminal_snapshot_fingerprint=digest(observed),
                     terminal_origin='CURRENT_LLM_DIAGNOSE_PENDING_CORE_REVALIDATION',
                     m5_clinical_support=trace.get('clinical_support', False),
                     llm_terminal_certainty=candidate.get('certainty'))
        return candidate


@contextmanager
def terminal_only_scope():
    """Synchronous scoped installation; all old plans retain the legacy classes."""
    with _SOCKET_LOCK, patch('app.locdi.common_menu_backend.CommonMenuNativeClient', TerminalOnlyClient), \
            patch('app.locdi.common_menu_backend.StopThenLLMInformationPolicy', TerminalOnlyInformationPolicy):
        yield
