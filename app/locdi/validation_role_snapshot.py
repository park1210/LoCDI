"""Single terminal judgment on a frozen Validation Snapshot, NOT Native skill.

Replay the prespecified Cardiology acquisition prefix. Change only the final
model role/System Prompt; the original Core semantic validator still runs.
"""
from copy import deepcopy
from contextlib import contextmanager
import json
from unittest.mock import patch
from app.core.hospital_llm_policy import HospitalLLMPolicy
from app.locdi.common_menu_backend import CommonMenuNativeClient
from app.locdi.terminal_only_recovery import terminal_request
from app.locdi.llm_setup import digest
from app.llm.prompt_loader import er_system_prompt,specialty_system_prompt

VERSION='validation_same_snapshot_six_roles.v1'


@contextmanager
def diagnostic_task_budget(final_request_number):
    """Declare zero further acquisition for the NEW diagnostic-only task.

Use the existing Core terminal-budget path to allow legitimate unresolved;
never add a certainty branch behind the Core validator's back. This is not a
claim that the original Native episode had exhausted its budget.
"""
    original=HospitalLLMPolicy._decision;count=[0]
    def decide(policy,runtime,role,department=''):
        count[0]+=1;saved=policy.config
        if count[0]==final_request_number:
            prior=sum(a['role'] not in {'patient','router'} and a['status']=='success' for a in runtime.llm_audit)
            policy.config=deepcopy(saved)
            policy.config['max_decisions']=prior+runtime.voi_state.get('decision_count',0)+1
        try:return original(policy,runtime,role,department)
        finally:policy.config=saved
    with patch.object(HospitalLLMPolicy,'_decision',new=decide):yield


def narrow_role(payload,schema,*,role,department,registry):
    bound=[a for a in registry['agents'] if a['role']==role and (role=='er' or a['department']==department)]
    if len(bound)!=1:raise ValueError('EXACT_REGISTERED_ROLE_REQUIRED')
    p,s=terminal_request(payload,schema)
    p['role']=role;p['department']=department if role=='specialty' else ''
    p['terminal_request_contract']=VERSION
    # All roles get the same task extension; original System Prompt is unchanged.
    system=er_system_prompt() if role=='er' else specialty_system_prompt(department)
    return p,s,system,bound[0]['model']


def client_type(*,final_request_number,role,department,registry):
    class RoleSnapshotClient(CommonMenuNativeClient):
        def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
            if self.request_count+1>final_request_number:raise ValueError('ONE_FINAL_SNAPSHOT_ONLY')
            if self.request_count+1==final_request_number:
                p,s,system,bound_model=narrow_role(json.loads(user_prompt),schema,role=role,department=department,registry=registry)
                if bound_model!=model:raise ValueError('MODEL_BINDING_CHANGED')
                return super().chat_json(system,json.dumps(p,ensure_ascii=False),model,schema=s)
            return super().chat_json(system_prompt,user_prompt,model,schema=schema)
    return RoleSnapshotClient


class Reply:
    status_code=200
    def __init__(self,obj):self.obj=deepcopy(obj);self.content=json.dumps(obj).encode()
    def json(self):return deepcopy(self.obj)
    def raise_for_status(self):pass


class ValidationPrefixTransport:
    def __init__(self,before,after,*,prepared,factory,role,department):
        if not before or len(before)!=len(after):raise ValueError('COMPLETE_SAVED_NATIVE_REQUESTS_REQUIRED')
        for b,a in zip(before,after):
            if digest(b['request'])!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or a['status']!='VALID' or a['HTTP_status']!=200:raise ValueError('SAVED_REQUEST_INTEGRITY')
        self.before=before;self.after=after;self.prefix_count=len(before)-1;self.index=0
        self.pending=False;self.prepared=prepared;self.factory=factory;self.actual=None
        self.role=role;self.department=department;self.posts=0;self.attempts=0
    def get(self,url):
        if url.endswith('/api/ps') and self.pending:
            self.pending=False;return Reply({'models':[self.after[self.index-1]['identity_after']]})
        if url.endswith('/api/tags') and self.index<self.prefix_count:
            return Reply({'models':[self.before[self.index]['identity_before']]})
        if self.actual is None:self.actual=self.factory(self.prepared)
        return self.actual.get(url)
    def reserve(self,reserve):
        if self.index<self.prefix_count:return
        reserve();self.attempts+=1
    def post(self,url,*,json):
        if self.index<self.prefix_count:
            if digest(json)!=self.before[self.index]['request_fingerprint']:raise ValueError('PREFIX_WIRE_CHANGED')
            a=self.after[self.index];self.index+=1;self.pending=True;return Reply(a['response_envelope'])
        p=__import__('json').loads(json['messages'][1]['content'])
        original=__import__('json').loads(self.before[-1]['request']['messages'][1]['content'])
        if self.posts or p.get('terminal_request_contract')!=VERSION or p['role']!=self.role or p['department']!=(self.department if self.role=='specialty' else ''):raise ValueError('ONE_REGISTERED_ROLE_TERMINAL_REQUEST')
        for key in ('observation','evidence','supported_diagnosis_eligible_indices'):
            if digest(p.get(key))!=digest(original.get(key)):raise ValueError('VALIDATION_SNAPSHOT_CHANGED')
        self.posts+=1
        if self.actual is None:self.actual=self.factory(self.prepared)
        return self.actual.post(url,json=json)
    def close(self):
        if self.actual is not None:self.actual.close()
