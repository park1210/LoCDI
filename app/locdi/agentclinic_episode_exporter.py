"""External-only exports; original values/prediction stay private, no score computed."""
from copy import deepcopy
import re
from pathlib import Path
from app.locdi.episode_exporter import _write
from app.locdi.llm_setup import ROOT, digest


def public_episode(episode):
    if episode.get('schema_version') != 'agentclinic.external_episode.v1':
        raise ValueError('EXTERNAL_EPISODE_SCHEMA')
    result = deepcopy(episode['result'])
    # All free-text predictions and reasons can reproduce source reference text.
    result['prediction_recorded'] = bool((result.pop('predicted_diagnosis') or '').strip())
    result['unresolved_reason_recorded'] = bool(result.pop('unresolved_reason'))
    result['error_type'] = 'OFFLINE_CLIENT_ERROR' if result['error_type'] else None
    trace = [{k: deepcopy(e[k]) for k in ('request_id', 'turn_index', 'action', 'target',
              'validation', 'status', 'error_code', 'observation_ref', 'repeat_of')} for e in episode['trace']]
    return {'schema_version': 'agentclinic.external_public_episode.v1',
            'scenario_ref': episode['scenario_ref'], 'source': deepcopy(episode['source']),
            'config': deepcopy(episode['config']), 'config_fingerprint': episode['config_fingerprint'],
            'prompt_fingerprint': episode['prompt_fingerprint'], 'mock_client': True,
            'trace': trace, 'result': result, 'clinical_metrics': 'NOT_EVALUATED',
            'real_generation_calls': 0, 'grading_calls': 0}


def export_episode(episode, *, public_root, private_root, name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', name):
        raise ValueError('EXPORT_NAME')
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if (not public_root.is_relative_to(ROOT/'outputs/research_evaluation') or
            not private_root.is_relative_to(ROOT/'outputs/private')):
        raise ValueError('EXTERNAL_EXPORT_ROOT')
    # Preflight BOTH paths before the first write. Atomicity is per file only;
    # a disk failure can leave an identifiable private partial export.
    for root in (public_root, private_root):
        if (root/'episodes'/(name+'.json')).exists():
            raise FileExistsError('EXTERNAL_EPISODE_EXISTS')
    public = public_episode(episode)
    public['private_episode_fingerprint'] = digest(episode)
    _write(episode, 'episodes/'+name+'.json', private_root, 'episodes')
    _write(public, 'episodes/'+name+'.json', public_root, 'episodes')
    return public
