"""Budgeted OFFLINE external scenario runtime. No hospital clock or scoring access."""
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from app.locdi.agentclinic_scenario_schema import Scenario, POLICY_TEXT, POLICY_VERSION
from app.locdi.agentclinic_scenario_adapter import action_catalog
from app.locdi.llm_setup import digest
from app.locdi.run_simulation import _offline_network_guard


class OfflineDoctorClient(ABC):
    @abstractmethod
    def choose_action(self, observation):
        """Return one dict action, using no remote backend."""


@dataclass(frozen=True)
class TurnOptions:
    max_turns: int = 20
    max_requests: int = 20

    def __post_init__(self):
        if any(type(v) is not int or not 1 <= v <= 100 for v in (self.max_turns, self.max_requests)):
            raise ValueError('FINITE_OFFLINE_BUDGET_REQUIRED')


def initial_observation(scenario):
    return {'scenario_ref': scenario.scenario_id, 'objective': scenario.objective,
            'catalog': action_catalog(scenario), 'observations': [], 'turn_index': 0}


def run_scenario(scenario, client, *, options=None):
    if not isinstance(scenario, Scenario) or not isinstance(client, OfflineDoctorClient):
        raise TypeError('EXPLICIT_OFFLINE_SCENARIO_AND_CLIENT_REQUIRED')
    options = options or TurnOptions()
    if not isinstance(options, TurnOptions):
        raise TypeError('TURN_OPTIONS_REQUIRED')
    scenario = deepcopy(scenario)
    targets = {kind: {t.target_id: t for t in ts} for kind, ts in
               (('ASK', scenario.ask_targets), ('EXAMINE', scenario.exam_targets), ('TEST', scenario.test_targets))}
    current = initial_observation(scenario)
    released, trace = {}, []
    result = {'status': 'incomplete', 'completed': False, 'predicted_diagnosis': None,
              'unresolved_reason': None, 'stop_reason': None, 'error_type': None}
    requests = 0
    with _offline_network_guard():
        for turn in range(1, options.max_turns + 1):
            if requests >= options.max_requests:
                result['stop_reason'] = 'REQUEST_BUDGET_EXHAUSTED'
                break
            request_id = 'R%04d' % (requests + 1)
            current['turn_index'] = turn - 1
            current['remaining_requests'] = options.max_requests - requests
            requests += 1
            event = {'request_id': request_id, 'turn_index': turn, 'action': None, 'target': None,
                     'validation': 'INVALID', 'status': 'INVALID', 'error_code': None,
                     'observation_ref': None, 'observation': None, 'repeat_of': None}
            try:
                action = client.choose_action(deepcopy(current))
            except Exception as exc:
                event.update(status='FAILED', error_code='OFFLINE_CLIENT_ERROR')
                trace.append(event)
                result.update(status='failed', stop_reason='CLIENT_FAILED', error_type=type(exc).__name__)
                break
            if not isinstance(action, dict) or not isinstance(action.get('action'), str):
                event['error_code'] = 'ACTION_SHAPE'
            else:
                kind = action['action']
                event['action'] = kind if kind in {'ASK', 'EXAMINE', 'TEST', 'DIAGNOSE'} else 'UNSUPPORTED'
                if kind not in {'ASK', 'EXAMINE', 'TEST', 'DIAGNOSE'}:
                    event.update(status='UNSUPPORTED', error_code='ACTION_UNSUPPORTED')
                elif kind == 'DIAGNOSE':
                    if set(action) - {'action', 'predicted_diagnosis', 'unresolved_reason'} or not isinstance(action.get('predicted_diagnosis'), str):
                        event['error_code'] = 'DIAGNOSIS_SHAPE'
                    elif not isinstance(action.get('unresolved_reason', ''), str) or len(action['predicted_diagnosis']) > 4096:
                        event['error_code'] = 'DIAGNOSIS_SHAPE'
                    else:
                        predicted = action['predicted_diagnosis']
                        event.update(validation='VALID', status='EXECUTED')
                        trace.append(event)
                        result.update(status='completed', completed=True, predicted_diagnosis=predicted,
                                      stop_reason='DIAGNOSIS_FINALIZED' if predicted.strip() else 'UNRESOLVED',
                                      unresolved_reason=(action.get('unresolved_reason') or 'NO_DIAGNOSIS') if not predicted.strip() else None)
                        break
                elif set(action) != {'action', 'target'} or not isinstance(action.get('target'), str):
                    event['error_code'] = 'ACTION_SHAPE'
                elif action['target'] not in targets[kind]:
                    # Unknown free-text targets are never echoed into public records.
                    event.update(status='UNAVAILABLE', error_code='TARGET_UNAVAILABLE')
                else:
                    target = targets[kind][action['target']]
                    event.update(target=target.target_id, validation='VALID', status='EXECUTED')
                    key = (kind, target.target_id)
                    if key in released:
                        obs = released[key]
                        event['repeat_of'] = obs['request_id']
                    else:
                        obs = {'observation_ref': 'O%04d' % (len(released) + 1),
                               'action': kind, 'target': target.target_id, 'request_id': request_id,
                               'released_turn': turn, 'value': deepcopy(target.value)}
                        released[key] = obs
                        current['observations'].append(deepcopy(obs))
                    event.update(observation_ref=obs['observation_ref'], observation=deepcopy(obs))
            trace.append(event)
            current['last_action'] = {k: event[k] for k in
                                      ('request_id', 'action', 'target', 'status', 'error_code', 'observation_ref')}
        else:
            result['stop_reason'] = 'TURN_BUDGET_EXHAUSTED'
    if result['stop_reason'] is None:
        result['stop_reason'] = 'TURN_BUDGET_EXHAUSTED'
    result.update(turn_count=len(trace), action_count=len(trace),
                  test_count=sum(e['action'] == 'TEST' for e in trace),
                  invalid_action_count=sum(e['status'] in {'INVALID', 'UNSUPPORTED', 'UNAVAILABLE'} for e in trace),
                  request_count=requests, diagnosis_unresolved=not bool((result['predicted_diagnosis'] or '').strip()))
    config = {'policy_version': POLICY_VERSION, 'max_turns': options.max_turns,
              'max_requests': options.max_requests, 'mode': 'OFFLINE_FAKE_CLIENT',
              'clinical_minutes': 'NOT_SUPPORTED'}
    return {'schema_version': 'agentclinic.external_episode.v1', 'scenario_ref': scenario.scenario_id,
            'source': deepcopy(scenario.source), 'config': config, 'config_fingerprint': digest(config),
            'prompt_fingerprint': digest(POLICY_TEXT), 'mock_client': True,
            'trace': trace, 'result': result, 'clinical_metrics': 'NOT_EVALUATED',
            'real_generation_calls': 0, 'grading_calls': 0}
