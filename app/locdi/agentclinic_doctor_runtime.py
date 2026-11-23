"""Doctor orchestration over the unchanged deterministic C3-11 source executor.

Prefix replay computes state from actual accepted actions. Its input sentinel is
never exported as a failure. Only final source execution and real doctor errors
become episode events. No backend is called inside the offline source executor.
"""
from copy import deepcopy
from app.locdi.agentclinic_turn_runtime import OfflineDoctorClient, TurnOptions, run_scenario
from app.locdi.agentclinic_doctor_client import FakeDoctorClient, LiveDoctorClient, PROMPT_SHA256
from app.locdi.llm_setup import digest


class _NeedInput(Exception):
    pass


class _Replay(OfflineDoctorClient):
    def __init__(self, actions, failure=None):
        self.actions, self.failure, self.position = actions, failure, 0
        self.observation = None

    def choose_action(self, observation):
        if self.position < len(self.actions):
            action = deepcopy(self.actions[self.position]); self.position += 1
            return action
        if self.failure is not None:
            raise self.failure
        self.observation = deepcopy(observation)
        raise _NeedInput()


def run_doctor_scenario(scenario, doctor, *, options=None):
    if not isinstance(doctor, (FakeDoctorClient, LiveDoctorClient)):
        raise TypeError('GATED_JSON_DOCTOR_REQUIRED')
    if isinstance(doctor, LiveDoctorClient):
        limit = doctor.definition['max_turns_per_case']
        options = options or TurnOptions(limit, limit)
        if not isinstance(options, TurnOptions) or options.max_turns > limit or options.max_requests > limit:
            raise ValueError('LIVE_CASE_BUDGET_EXCEEDS_PINNED_PLAN')
    else:
        options = options or TurnOptions()
    scenario = deepcopy(scenario)
    if isinstance(doctor, LiveDoctorClient):
        from app.locdi.agentclinic_live_preflight import SOURCE
        from app.locdi.agentclinic_scenario_adapter import load_scenarios
        canonical = next((s for s in load_scenarios(SOURCE).scenarios if s.scenario_id==scenario.scenario_id), None)
        if canonical != scenario:
            raise ValueError('LIVE_SOURCE_PROJECTION_CHANGED')
    actions, failure = [], None
    started, audit_start = doctor.request_count, len(doctor.audit)
    generation_start = doctor.transport.request_count if isinstance(doctor, LiveDoctorClient) else 0
    while True:
        replay = _Replay(actions, failure)
        episode = run_scenario(scenario, replay, options=options)
        if replay.observation is None:
            break
        observation = replay.observation
        observation['history'] = [{k:e[k] for k in ('request_id','turn_index','action','target',
                     'validation','status','error_code','observation_ref','repeat_of')}
                     for e in episode['trace'][:-1]]
        observation['remaining_requests'] = min(observation['remaining_requests'],
                                               doctor.request_budget-doctor.request_count)
        try:
            actions.append(doctor.choose_action(observation))
        except Exception as exc:
            failure = exc
    live = isinstance(doctor, LiveDoctorClient)
    if failure is not None:
        episode['trace'][-1]['error_code'] = 'DOCTOR_CLIENT_ERROR'
        episode['result']['doctor_error_code'] = (doctor.audit[-1].get('failure_category','client_contract')
                                                   if len(doctor.audit)>audit_start else 'pre_request_contract')
    episode.update(mock_client=not live, prompt_fingerprint=getattr(doctor,'prompt_sha256',PROMPT_SHA256),
                   doctor_audit=deepcopy(doctor.audit[audit_start:]),
                   doctor_request_attempts=doctor.request_count-started)
    episode['config'].update(mode='GATED_LIVE_DOCTOR' if live else 'OFFLINE_JSON_DOCTOR',
                              runtime_version='agentclinic.prefix_source_runtime.v2',
                              doctor_global_request_budget=doctor.request_budget)
    episode['config_fingerprint'] = digest(episode['config'])
    if live:
        episode['real_generation_calls'] = doctor.transport.request_count-generation_start
        episode['model_metadata'] = doctor.transport.public_metadata()
    return episode
