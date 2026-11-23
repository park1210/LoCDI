"""TEST_ONLY authorization, metadata and synchronous context stopping tests."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha
from app.locdi.agentclinic_context_bootstrap import (
    verify_operator_evidence,observe_context,bootstrap_plan,PinnedBootstrapApproval,run_bootstrap,verify_basis)
from app.locdi.agentclinic_live_preflight import proposal


class ContextBootstrapTests(unittest.TestCase):
    def evidence(self,directory,**changes):
        now=datetime.now(timezone.utc)
        r={'schema_version':'agentclinic.operator_authorization.v1',
           'authorization_source':'EXPLICIT_USER_INSTRUCTION','operator_role':'LOCAL_SERVER_OPERATOR',
           'operator_reference':'USER_LOCAL_SERVER_OPERATOR','operator_authorized':True,
           'permitted_scopes':['LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY','LIVE_AGENTCLINIC_DOCTOR_ONLY','LIVE_AGENTCLINIC_PILOT_5_ONLY'],
           'user_statement':'TEST_ONLY_EXPLICIT_PERMISSION','scope_confirmation_statement':'TEST_ONLY',
           'research_approval':{'status':'USER_REPORTED_VERBAL_APPROVAL','signed_document_present':False},
           'issued_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat(),
           'total_generation_limit':50,'bootstrap_generation_limit':1,
           'bootstrap_namespace':'TEST_ONLY_bootstrap','pilot_namespace':'TEST_ONLY_pilot'}
        r.update(changes);p=Path(directory)/'TEST_ONLY_evidence.json';p.write_text(json.dumps(r),encoding='utf8')
        return p,sha(p)

    def test_verbal_and_operator_evidence_distinct(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            p,h=self.evidence(d)
            r=verify_operator_evidence(p,h,scope='LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY')
            self.assertEqual(r['research_approval']['status'],'USER_REPORTED_VERBAL_APPROVAL')
            self.assertFalse(r['research_approval']['signed_document_present'])

    def test_professor_signature_cannot_be_invented(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            p,h=self.evidence(d,research_approval={'status':'USER_REPORTED_VERBAL_APPROVAL','signed_document_present':True})
            with self.assertRaises(ValueError):verify_operator_evidence(p,h,scope='LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY')

    def test_missing_operator_or_scope_cannot_authorize(self):
        for change in ({'operator_authorized':False},{'permitted_scopes':[]},{'total_generation_limit':51},
                       {'expires_at':(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()}):
            with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
                p,h=self.evidence(d,**change)
                with self.assertRaises(ValueError):verify_operator_evidence(p,h,scope='LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY')

    def test_hash_mismatch_denied(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            p,h=self.evidence(d)
            with self.assertRaises(ValueError):verify_operator_evidence(p,'0'*64,scope='LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY')

    def test_no_bootstrap_authority_never_constructs_client(self):
        with patch('app.locdi.ollama_adapter.LocalOllamaAdapter') as client:
            for a in (None,True,'approved'):
                with self.assertRaises(ValueError):run_bootstrap('TEST_ONLY',authority=a,execute=True,allow_llm=True)
            client.assert_not_called()

    def test_bootstrap_namespace_and_plan_pinned(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            p,h=self.evidence(d);plan=bootstrap_plan('TEST_ONLY_bootstrap','TEST_ONLY','0'*64)
            a={'schema_version':'agentclinic.bootstrap_approval.v1','scope':'LOCAL_CONTEXT_BOOTSTRAP_ONE_ONLY',
               'plan_fingerprint':digest(plan),'approved_by':'USER_LOCAL_SERVER_OPERATOR',
               'operator_evidence_path':str(p),'operator_evidence_sha256':h}
            ap=Path(d)/'TEST_ONLY_approval.json';ap.write_text(json.dumps(a),encoding='utf8')
            auth=PinnedBootstrapApproval(ap,sha(ap));auth.verify(plan)
            with self.assertRaises(ValueError):auth.verify(dict(plan,namespace='OTHER'))

    def test_actual_context_digest_memory_required(self):
        cfg=proposal()['legacy_config']
        model={'name':cfg['model'],'digest':cfg['model_identifier'],'context_length':32768,'size':100,'size_vram':80}
        def observe(rows):
            with patch('app.locdi.agentclinic_context_bootstrap.metadata',side_effect=lambda k: {'models':rows} if k=='ps' else {'version':'TEST_ONLY'}),patch('app.locdi.agentclinic_context_bootstrap.memory_snapshot',return_value={'available_physical_bytes':100,'gpu_status':'OBSERVED'}):
                return observe_context(cfg)
        self.assertEqual(observe([model])['status'],'VALID')
        for rows in ([],[dict(model,context_length=4096)],[dict(model,digest='sha256:'+'0'*64)],[model,model]):
            self.assertEqual(observe(rows)['status'],'INVALID')

    def test_status_only_or_professor_approver_cannot_pass_basis(self):
        with self.assertRaises(ValueError):verify_basis({'approved':True},'LIVE_AGENTCLINIC_DOCTOR_ONLY')
        with self.assertRaises(ValueError):verify_basis({'approval_basis_type':'USER_OPERATOR_WITH_REPORTED_VERBAL_RESEARCH_APPROVAL','approved_by':'INVENTED_PROFESSOR'},'LIVE_AGENTCLINIC_DOCTOR_ONLY')

    def test_context_hook_stops_before_action_return_and_saves_response(self):
        from app.locdi.agentclinic_live_pilot_runner import _Monitor,PilotStop
        from app.locdi.agentclinic_doctor_client import LiveDoctorClient
        from app.locdi.agentclinic_pilot_readiness import prepare_plan
        import app.locdi.agentclinic_live_pilot_runner as runner
        from app.locdi.agentclinic_turn_runtime import initial_observation
        from app.locdi.agentclinic_scenario_adapter import load_scenarios
        from app.locdi.agentclinic_live_preflight import SOURCE
        cfg=proposal()['legacy_config'];transport=SimpleNamespace(request_count=0,identity_check_count=0,last_observed_model_identifier=cfg['model_identifier'])
        def chat(*args,**kwargs):
            transport.request_count+=1;return '{"action":"DIAGNOSE","predicted_diagnosis":"TEST_ONLY"}'
        transport.chat_json=chat
        client=object.__new__(LiveDoctorClient);client.transport=transport;client.request_count=0;client.request_budget=49;client.max_prompt_bytes=16384
        def choose(obs):
            client.request_count+=1;return json.loads(client.transport.chat_json('TEST_ONLY','TEST_ONLY'))
        client.choose_action=choose
        class Authority:
            path='TEST_ONLY';sha256='0'*64
            def verify(self,*args,**kwargs):return None
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            with patch.object(runner,'PinnedPilotApproval',Authority),patch('app.locdi.external_reference_readiness.pinned_private_json',return_value={}):
                monitor=_Monitor(client,Path(d)/'public',Path(d)/'private',prepare_plan('TEST_ONLY_context'),SimpleNamespace(count=lambda *a:100),authority=Authority())
                monitor.context_observer=lambda:{'status':'INVALID','blocker_codes':['TEST_ONLY_CONTEXT_MISMATCH']}
                obs=initial_observation(load_scenarios(SOURCE).scenarios[0]);obs.update(history=[],remaining_requests=10)
                with self.assertRaises(PilotStop):monitor.choose(obs)
                self.assertEqual(transport.request_count,1)
                after=json.loads((Path(d)/'private/checkpoints/Request0001_after.json').read_text())
                self.assertEqual(after['error_code'],'ACTUAL_CONTEXT_UNVERIFIED_OR_MISMATCH')
                self.assertIsNotNone(after['response_content'])

    def test_bootstrap_is_bounded_nonclinical_and_source_free(self):
        p=bootstrap_plan('TEST_ONLY','TEST_ONLY','0'*64)
        self.assertEqual(p['max_generations'],1);self.assertEqual(p['config']['max_requests'],1)
        self.assertEqual(p['pilot_remaining_after_attempt'],49)
        self.assertEqual(p['retry'],0);self.assertEqual(p['repair'],0)
        self.assertNotIn('gold',json.dumps(p).casefold())
