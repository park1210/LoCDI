"""JSON-doctor versioned public projection; raw role audit/scoring remains private."""
from copy import deepcopy
from pathlib import Path
import re
from app.locdi.agentclinic_episode_exporter import public_episode
from app.locdi.episode_exporter import _write
from app.locdi.llm_setup import ROOT, digest


def public_doctor_episode(episode):
    out = public_episode(episode)
    out['schema_version'] = 'agentclinic.json_doctor_public_episode.v1'
    out['mock_client'] = episode['mock_client']
    out['real_generation_calls'] = episode['real_generation_calls']
    out['doctor_request_attempts'] = episode['doctor_request_attempts']
    out['result']['error_type'] = 'DOCTOR_CLIENT_ERROR' if episode['result']['error_type'] else None
    out['requests'] = [{k:deepcopy(r[k]) for k in ('role','status','input_sha256','error_type','failure_category') if k in r}
                       for r in episode['doctor_audit']]
    for row in out['requests']:
        if 'error_type' in row:
            row['error_type'] = 'DOCTOR_REQUEST_FAILED'
    out['private_episode_fingerprint'] = digest(episode)
    return out


def export_doctor_episode(episode, *, public_root, private_root, name):
    public_root, private_root = Path(public_root).resolve(), Path(private_root).resolve()
    if (not public_root.is_relative_to(ROOT/'outputs/research_evaluation')
            or not private_root.is_relative_to(ROOT/'outputs/private')
            or not re.fullmatch('[A-Za-z0-9_-]{1,80}',name)):
        raise ValueError('DOCTOR_EXPORT_SCOPE')
    for root in (public_root,private_root):
        if (root/'episodes'/(name+'.json')).exists():
            raise FileExistsError('DOCTOR_EXPORT_EXISTS')
    out = public_doctor_episode(episode)
    _write(episode,'episodes/'+name+'.json',private_root,'episodes')
    _write(out,'episodes/'+name+'.json',public_root,'episodes')
    return out
