"""TEST_ONLY JSON transport and private approval fixtures, never research authority."""
from copy import deepcopy
import json
from pathlib import Path
from app.locdi.agentclinic_doctor_client import OfflineJsonTransport
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import sha


class JsonScript(OfflineJsonTransport):
    def __init__(self, actions):
        self.actions, self.seen = deepcopy(actions), []

    def chat_json(self, system, user, model=None, *, schema=None):
        self.seen.append({'system':system,'payload':json.loads(user),'schema':schema})
        value = self.actions.pop(0)
        if isinstance(value,Exception):
            raise value
        return json.dumps(value) if isinstance(value,dict) else value


def fixture_approval(directory, plan):
    from app.locdi.agentclinic_live_preflight import PinnedDoctorApproval
    directory=Path(directory)
    review={'plan_fingerprint':digest(plan),'reviewed_by':'TEST_ONLY_REVIEWER',
            'review_status':'APPROVED','exposure_policy':plan['definition']['exposure_policy'],
            'verified_server_context_tokens':plan['definition']['context_tokens'],
            'server_context_configuration_reference':'TEST_ONLY_CONTEXT_NOT_REAL_SERVER'}
    rp=directory/'TEST_ONLY_review.json';rp.write_text(json.dumps(review),encoding='utf8')
    approval={'schema_version':'agentclinic.doctor_approval.v1','scope':'LIVE_AGENTCLINIC_DOCTOR_ONLY',
              'plan_fingerprint':digest(plan),'approved_by':'TEST_ONLY_APPROVER',
              'issued_at':'2000-01-01T00:00:00+00:00','expires_at':'2100-01-01T00:00:00+00:00',
              'review_path':str(rp),'review_sha256':sha(rp)}
    path=directory/'TEST_ONLY_approval.json';path.write_text(json.dumps(approval),encoding='utf8')
    return PinnedDoctorApproval(path,sha(path))
