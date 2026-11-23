"""TEST_ONLY prompt difference, fixed paired settings and fresh authority contracts."""
from datetime import datetime,timezone,timedelta
import json,tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha
from app.locdi.agentclinic_doctor_client import system_prompt,prompt_binding,build_request,FakeDoctorClient,OfflineJsonTransport
from app.locdi.agentclinic_doctor_runtime import run_doctor_scenario
from app.locdi.agentclinic_scenario_adapter import adapt_test_record
from app.locdi.agentclinic_turn_runtime import initial_observation
from app.locdi.agentclinic_live_preflight import proposal,preflight,PinnedDoctorApproval
from app.locdi.agentclinic_pilot_readiness import prepare_plan,PinnedPilotApproval
from app.locdi.agentclinic_prompt_ablation import verify_experiment,verify_condition_authorization
from agentclinic_offline_support import fixture


class PromptAblationTests(unittest.TestCase):
    def test_exact_one_sentence_appended_v1_preserved(self):
        a=system_prompt('A');b=system_prompt('B')
        sentence='Do not request the same Action and Target again after its result has been provided; if you need more information choose a different valid Target, and if you have enough information give a final diagnosis or abstain.'
        self.assertEqual(b,a+'\n'+sentence+'\n');self.assertEqual(system_prompt(),a)
        self.assertEqual(sha(prompt_binding('B')[1]),prompt_binding('B')[2])

    def test_only_prompt_differs_in_fixed_condition_definitions(self):
        a=proposal('A');b=proposal('B')
        diff={k for k in a if a[k]!=b[k]}
        self.assertEqual(diff,{'prompt_profile','prompt_version','prompt_sha256'})
        for d in (a,b):
            self.assertEqual(d['total_generation_budget'],50);self.assertEqual(d['max_turns_per_case'],10)
            self.assertEqual(d['context_tokens'],32768);self.assertEqual(d['legacy_config']['seed'],42)
            self.assertFalse(preflight(d)['static_blockers'])

    def test_identical_payload_catalog_schema_no_gold(self):
        s,_=adapt_test_record(fixture());obs=initial_observation(s);obs.update(remaining_requests=10,history=[])
        a=build_request(obs,prompt_profile='A');b=build_request(obs,prompt_profile='B')
        self.assertEqual(a[1:],b[1:]);self.assertNotEqual(a[0],b[0])
        obs['gold']='TEST_ONLY'
        with self.assertRaises(ValueError):build_request(obs,prompt_profile='B')

    def test_arbitrary_prompt_or_definition_rejected(self):
        with self.assertRaises(ValueError):system_prompt('C')
        b=proposal('B');b['legacy_config']['seed']=43
        with self.assertRaises(ValueError):prepare_plan('TEST_ONLY_B',b)
        b=proposal('B');b['prompt_sha256']='0'*64
        self.assertTrue(preflight(b)['static_blockers'])

    def test_repeated_actions_not_masked_and_profile_episode_fingerprint(self):
        s,_=adapt_test_record(fixture());target=s.exam_targets[0].target_id
        class Script(OfflineJsonTransport):
            def __init__(self):self.calls=0
            def chat_json(self,system,user,model=None,*,schema=None):
                self.calls+=1
                return json.dumps({'action':'EXAMINE','target':target} if self.calls<=2 else {'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY'})
        e=run_doctor_scenario(s,FakeDoctorClient(Script(),prompt_profile='B'))
        self.assertIsNotNone(e['trace'][1]['repeat_of'])
        self.assertEqual(e['prompt_fingerprint'],prompt_binding('B')[2])

    def evidence(self,directory,**changes):
        now=datetime.now(timezone.utc);plans={c:prepare_plan('TEST_ONLY_'+c,proposal(c)) for c in ('A','B')}
        e={'schema_version':'agentclinic.prompt_ablation_authorization.v1','approval_source':'EXPLICIT_CURRENT_USER_C3_15_APPROVAL',
           'approved_by':'USER_LOCAL_SERVER_OPERATOR','execute_authorized':True,'user_statement':'TEST_ONLY',
           'order':['A','B'],'cohort':[1,2,3,4,5],'condition_budget':50,'combined_generation_budget':100,
           'retry_budget':0,'repair_budget':0,'scoring_scope':'OFFLINE_EXACT_MATCH_LEXICAL_ONLY',
           'exposure_policy':'DEVELOPMENT_EXPOSED_NOT_INDEPENDENT_TEST','issued_at':now.isoformat(),
           'expires_at':(now+timedelta(hours=1)).isoformat(),
           'conditions':{c:{'namespace':plans[c]['namespace'],'runner_plan_fingerprint':digest(plans[c]),
              'doctor_plan_fingerprint':plans[c]['doctor_plan_fingerprint'],'prompt_sha256':prompt_binding(c)[2]} for c in plans}}
        e.update(changes);p=Path(directory)/'TEST_ONLY_evidence.json';p.write_text(json.dumps(e),encoding='utf8')
        return p,plans

    def test_new_authority_required_for_both_profiles(self):
        for c in ('A','B'):
            with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
                p=Path(d)/'TEST_ONLY_old.json';p.write_text('{"approved":true}',encoding='utf8')
                plan=prepare_plan('TEST_ONLY_'+c,proposal(c))
                with self.assertRaises((ValueError,KeyError)):PinnedDoctorApproval(p,sha(p)).verify(plan['doctor_plan'])
                with self.assertRaises((ValueError,KeyError)):PinnedPilotApproval(p,sha(p)).verify(plan)

    def test_condition_namespace_plan_and_hash_cannot_swap(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            p,plans=self.evidence(d)
            record={'approved_by':'USER_LOCAL_SERVER_OPERATOR','ablation_evidence_path':str(p),'ablation_evidence_sha256':sha(p)}
            verify_condition_authorization(record,plans['A'])
            with self.assertRaises(ValueError):verify_condition_authorization(record,dict(plans['A'],namespace='OTHER'))
            with self.assertRaises(ValueError):verify_experiment(p,'0'*64)

    def test_expired_order_budget_or_missing_permission_denied(self):
        for changes in ({'execute_authorized':False},{'order':['B','A']},{'condition_budget':51},
                        {'combined_generation_budget':101},{'expires_at':(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()}):
            with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
                p,_=self.evidence(d,**changes)
                with self.assertRaises(ValueError):verify_experiment(p,sha(p))

    def test_no_authority_never_constructs_backend(self):
        from app.locdi.agentclinic_live_pilot_runner import run_live
        with patch('app.locdi.agentclinic_live_pilot_runner.create_live_client') as factory:
            for c in ('A','B'):
                with self.assertRaises(ValueError):run_live('TEST_ONLY_'+c,definition=proposal(c),execute=True,allow_llm=True)
            factory.assert_not_called()

    def test_B_cannot_generate_before_this_experiments_A_summary(self):
        from app.locdi.agentclinic_prompt_ablation import verify_execution_order
        from app.locdi.dataset_freeze import FreezeError
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            p,plans=self.evidence(d)
            record={'approved_by':'USER_LOCAL_SERVER_OPERATOR','ablation_evidence_path':str(p),'ablation_evidence_sha256':sha(p)}
            rp=Path(d)/'TEST_ONLY_pilot.json';rp.write_text(json.dumps(record),encoding='utf8')
            authority=SimpleNamespace(path=rp,sha256=sha(rp))
            with self.assertRaisesRegex(FreezeError,'MISSING_SOURCE'):verify_execution_order(authority,plans['B'])
            with patch('app.locdi.dataset_freeze.load',side_effect=[plans['A'],{'mode':'LIVE'}]):
                verify_execution_order(authority,plans['B'])
