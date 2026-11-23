"""TEST_ONLY runner, offline authority and mocked model transport contracts."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha,load
from app.locdi.agentclinic_live_pilot_runner import run_fake,run_live
from app.locdi.agentclinic_pilot_readiness import (readiness,prepare_plan,PinnedPilotApproval,
    output_paths,prompt_sizes,tokenizer_tree,LocalTokenCounter)
from app.locdi.ollama_adapter import ModelIdentityError
from app.locdi.agentclinic_scenario_adapter import load_scenarios
from app.locdi.agentclinic_live_preflight import SOURCE
from scripts.prepare_agentclinic_doctor import ScriptedJson,source_plan
from agentclinic_json_support import fixture_approval


class Counter:
    def __init__(self,value=100):self.value=value
    def count(self,system,user):return self.value


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        self.names=[]

    def name(self):
        n='TEST_ONLY_c313a_'+uuid.uuid4().hex;self.names.append(n);return n

    def tearDown(self):
        for n in self.names:
            for p in output_paths(n):
                if p.exists():shutil.rmtree(str(p))

    def plans(self):
        return [a for s in load_scenarios(SOURCE).scenarios[:5] for a in source_plan(s)]

    def test_five_in_order_one_client_and_checkpoint_links(self):
        n=self.name();r=run_fake(n,ScriptedJson(self.plans()))
        self.assertEqual(r['status'],'COMPLETED');self.assertEqual(r['planned_denominator'],5)
        self.assertEqual([x['ordinal'] for x in r['rows']],[1,2,3,4,5])
        self.assertEqual(r['counters']['fake_transport_calls'],20)
        self.assertEqual(r['counters']['doctor_request_attempts'],20)
        pub,priv=output_paths(n)
        for i in range(1,21):
            before=load(priv/'checkpoints'/('Request%04d_before.json'%i))
            after=load(priv/'checkpoints'/('Request%04d_after.json'%i))
            event=load(priv/'checkpoints'/('Request%04d_event.json'%i))
            link=load(pub/'checkpoints'/('Request%04d_link.json'%i))
            self.assertEqual(before['request_sha256'],after['request_sha256'])
            self.assertIsNotNone(after['response_sha256'])
            self.assertEqual(link['private_link_fingerprint'],digest(event))
        self.assertEqual(load(priv/'checkpoints/Request0020_after.json')['counters']['doctor_request_attempts'],20)

    def test_global_budget_not_reset_per_scenario_and_denominator_preserved(self):
        r=run_fake(self.name(),ScriptedJson(self.plans()),generation_budget=5)
        self.assertEqual(r['status'],'BUDGET_EXCEEDED')
        self.assertEqual(r['counters']['fake_transport_calls'],5)
        self.assertEqual(r['rows'][0]['status'],'COMPLETED')
        self.assertEqual(r['rows'][1]['episode_export'],'EXPORTED')
        self.assertEqual(r['not_executed'],3);self.assertEqual(len(r['rows']),5)

    def test_timeout_preserves_prior_completed_and_partial_failed_episode(self):
        actions=self.plans()[:5]+[TimeoutError('TEST_ONLY_PRIVATE_SECRET')]
        n=self.name();r=run_fake(n,ScriptedJson(actions))
        self.assertEqual(r['rows'][0]['status'],'COMPLETED')
        self.assertEqual(r['rows'][1]['terminal_status'],'failed')
        self.assertEqual(r['rows'][1]['episode_export'],'EXPORTED')
        self.assertEqual(r['not_executed'],3)
        self.assertNotIn('TEST_ONLY_PRIVATE_SECRET',json.dumps(r))

    def test_identity_mismatch_stops_and_never_completes(self):
        r=run_fake(self.name(),ScriptedJson([ModelIdentityError('TEST_ONLY')]))
        self.assertEqual(r['status'],'IDENTITY_MISMATCH');self.assertEqual(r['not_executed'],4)

    def test_prompt_hash_mutation_blocks_before_transport(self):
        tr=ScriptedJson(self.plans())
        with patch('app.locdi.agentclinic_doctor_client.PROMPT_SHA256','0'*64):
            with self.assertRaises(ValueError):run_fake(self.name(),tr)
        self.assertEqual(tr.seen,[])

    def test_invalid_json_and_absent_source_target_stop(self):
        for a,code in [('not JSON','JSON_RESPONSE_INVALID'),({'action':'TEST','target':'absent'},'SOURCE_ACTION_ERROR')]:
            r=run_fake(self.name(),ScriptedJson([a]))
            self.assertEqual(r['status'],code);self.assertEqual(r['counters']['fake_transport_calls'],1)

    def test_actual_context_limit_denies_before_transport(self):
        tr=ScriptedJson(self.plans());r=run_fake(self.name(),tr,token_counter=Counter(40000))
        self.assertEqual(r['status'],'CONTEXT_LIMIT_EXCEEDED');self.assertEqual(tr.seen,[])

    def test_byte_limit_is_separate_from_tokens(self):
        tr=ScriptedJson(self.plans());r=run_fake(self.name(),tr,max_prompt_bytes=256)
        self.assertEqual(r['status'],'CONTEXT_LIMIT_EXCEEDED');self.assertEqual(tr.seen,[])
        self.assertIsNone(prompt_sizes()[0]['input_tokens'])

    def test_unresolved_and_incomplete_have_distinct_terminal_states(self):
        r=run_fake(self.name(),ScriptedJson([{'action':'DIAGNOSE','predicted_diagnosis':''}]*5))
        self.assertEqual(r['status'],'UNRESOLVED');self.assertEqual(r['not_executed'],0)
        r=run_fake(self.name(),ScriptedJson([{'action':'ASK','target':'history'}]),max_turns=1)
        self.assertEqual(r['status'],'INCOMPLETE');self.assertEqual(r['not_executed'],4)

    def test_namespace_collision_and_resume_refused(self):
        n=self.name();run_fake(n,ScriptedJson([{'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY'}]*5))
        path=output_paths(n)[0]/'metrics/summary.json';old=path.read_bytes()
        tr=ScriptedJson(self.plans())
        with self.assertRaises(ValueError):run_fake(n,tr)
        self.assertEqual(tr.seen,[]);self.assertEqual(path.read_bytes(),old)

    def test_gold_not_in_checkpoint_request_or_public_results(self):
        n=self.name();run_fake(n,ScriptedJson(self.plans()));pub,priv=output_paths(n)
        for p in priv.glob('checkpoints/*_before.json'):
            text=p.read_text(encoding='utf8')
            self.assertNotIn('Correct_Diagnosis',text);self.assertNotIn('scoring_key',text)
        for p in pub.rglob('*.json'):
            text=p.read_text(encoding='utf8')
            self.assertNotIn('predicted_diagnosis',text);self.assertNotIn('Correct_Diagnosis',text)
        self.assertEqual(load(pub/'metrics/summary.json')['benchmark_scores'],'NOT_EVALUATED')

    def test_no_authority_or_string_never_constructs_live_client(self):
        with patch('app.locdi.agentclinic_live_pilot_runner.create_live_client') as backend:
            for kwargs in ({},{'execute':True,'allow_llm':True,'authority':'APPROVED'}):
                with self.assertRaises(ValueError):run_live(self.name(),**kwargs)
        backend.assert_not_called()

    def test_fake_never_constructs_real_backend(self):
        with patch('app.locdi.ollama_adapter._backend',side_effect=AssertionError('TEST_ONLY_NO_BACKEND')) as backend:
            run_fake(self.name(),ScriptedJson(self.plans()))
        backend.assert_not_called()

    def test_export_failure_keeps_private_and_planned_denominator(self):
        from app.locdi.episode_exporter import _write
        def fail(doc,dest,root,category):
            if 'outputs/research_evaluation' in Path(root).as_posix():raise OSError('TEST_ONLY_DISK')
            return _write(doc,dest,root,category)
        n=self.name()
        with patch('app.locdi.agentclinic_doctor_exporter._write',side_effect=fail):
            r=run_fake(n,ScriptedJson(self.plans()))
        self.assertEqual(r['status'],'EXPORT_FAILED');self.assertEqual(r['not_executed'],4)
        self.assertTrue((output_paths(n)[1]/'episodes/Scenario001.json').is_file())

    def test_checkpoint_before_write_failure_prevents_generation(self):
        from app.locdi.episode_exporter import _write
        def fail(doc,dest,root,category):
            if dest.endswith('_before.json'):raise OSError('TEST_ONLY_DISK')
            return _write(doc,dest,root,category)
        tr=ScriptedJson(self.plans())
        with patch('app.locdi.agentclinic_live_pilot_runner._write',side_effect=fail):
            r=run_fake(self.name(),tr)
        self.assertEqual(r['status'],'EXPORT_FAILED');self.assertEqual(tr.seen,[])

    def test_dry_readiness_no_network_no_approval_and_unknown_tokens(self):
        with patch('app.locdi.ollama_adapter._backend') as backend:
            r=readiness(self.name())
        self.assertFalse(r['LIVE_PILOT_READY']);self.assertFalse(r['LIVE_SERVER_READY'])
        self.assertFalse(r['LIVE_EXECUTION_APPROVED']);self.assertEqual(len(r['initial_prompt_sizes']),5)
        self.assertTrue(all(row['input_tokens'] is None for row in r['initial_prompt_sizes']))
        backend.assert_not_called()

    def test_source_cohort_namespace_and_budget_cannot_change_approved_plan(self):
        from app.locdi.agentclinic_live_preflight import proposal
        d=proposal();d['scenario_ordinals']=[2,1,3,4,5]
        with self.assertRaises(ValueError):prepare_plan(self.name(),d)
        d=proposal();d['total_generation_budget']=100
        with self.assertRaises(ValueError):prepare_plan(self.name(),d)
        with self.assertRaises(ValueError):prepare_plan('../escape')

    def make_authority(self,directory,n):
        directory=Path(directory);plan=prepare_plan(n)
        doctor=fixture_approval(directory,plan['doctor_plan'])
        tokenizer=directory/'tokenizer';tokenizer.mkdir();(tokenizer/'tokenizer_config.json').write_text('{}',encoding='utf8')
        now=datetime.now(timezone.utc)
        cfg=plan['doctor_plan']['definition']['legacy_config']
        review={'plan_fingerprint':digest(plan),'review_status':'APPROVED','reviewed_by':'TEST_ONLY',
            'observed_at':(now-timedelta(minutes=1)).isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat(),
            'model_identifier':cfg['model_identifier'],'model_name':cfg['model'],'model_version':cfg['model_version'],
            'context_tokens':32768,'max_context_tokens':32768,'server_version':'TEST_ONLY',
            'context_evidence_reference':'TEST_ONLY','identity_evidence_reference':'TEST_ONLY',
            'available_memory_bytes':1,'tokenizer_model_identifier':cfg['model_identifier'],
            'tokenizer_server_equivalence_review_reference':'TEST_ONLY',
            'tokenizer_tree_sha256':tokenizer_tree(tokenizer)}
        rp=directory/'TEST_ONLY_server.json';rp.write_text(json.dumps(review),encoding='utf8')
        approval={'schema_version':'agentclinic.pilot_approval.v1','scope':'LIVE_AGENTCLINIC_PILOT_5_ONLY',
            'plan_fingerprint':digest(plan),'approved_by':'TEST_ONLY',
            'doctor_approval_path':str(doctor.path),'doctor_approval_sha256':doctor.sha256,
            'server_context_review_path':str(rp),'server_context_review_sha256':sha(rp),'tokenizer_path':str(tokenizer)}
        p=directory/'TEST_ONLY_pilot_approval.json';p.write_text(json.dumps(approval),encoding='utf8')
        return PinnedPilotApproval(p,sha(p)),plan,review

    def test_pinned_pilot_authority_cannot_be_reused_for_new_namespace(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            authority,plan,_=self.make_authority(d,self.name())
            with patch('app.locdi.agentclinic_pilot_readiness.LocalTokenCounter',return_value=Counter()):
                authority.verify(plan)
                with self.assertRaises(ValueError):authority.verify(prepare_plan(self.name()))
            Path(authority.path).write_text('{"approved":true}',encoding='utf8')
            with self.assertRaises(ValueError):authority.verify(plan)

    def test_unverified_context_and_tokenizer_hash_fail_before_backend(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            n=self.name();authority,plan,review=self.make_authority(d,n)
            rp=Path(d)/'TEST_ONLY_server.json';rp.write_text('{}',encoding='utf8')
            with patch('app.locdi.agentclinic_live_pilot_runner.create_live_client') as backend:
                with self.assertRaises(ValueError):run_live(n,execute=True,allow_llm=True,authority=authority)
            backend.assert_not_called()

    def test_live_path_real_adapter_with_mock_transport_only(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as d:
            n=self.name();authority,plan,_=self.make_authority(d,n)
            cfg=plan['doctor_plan']['definition']['legacy_config'];generations=[]
            def generate(*args,**kwargs):
                generations.append(1)
                return json.dumps({'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_MOCK_TRANSPORT'})
            response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'models':[{'name':cfg['model'],'digest':cfg['model_identifier']}]})
            http=SimpleNamespace(get=lambda *args:response,close=lambda:None)
            legacy=SimpleNamespace(chat_json=generate)
            with patch('app.locdi.agentclinic_pilot_readiness.LocalTokenCounter',return_value=Counter()),patch('app.locdi.ollama_adapter._backend',return_value=(legacy,http,http,http)):
                result=run_live(n,execute=True,allow_llm=True,authority=authority)
            self.assertEqual(result['status'],'COMPLETED')
            self.assertEqual(len(generations),5);self.assertEqual(result['counters']['identity_count'],5)
            self.assertEqual(result['counters']['generation_count'],5)
            self.assertEqual(result['benchmark_scores'],'NOT_EVALUATED')

    def test_tokenizer_missing_local_files_fails_without_download(self):
        with self.assertRaises(ValueError):LocalTokenCounter(ROOT/'NOT_PRESENT_TEST_ONLY','0'*64)

    def test_installed_local_tokenizer_template_and_hash_mutation_no_download(self):
        from tokenizers import Tokenizer,models,pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private')) as directory:
            backend=Tokenizer(models.WordLevel({'[UNK]':0,'TEST_ONLY':1},unk_token='[UNK]'))
            backend.pre_tokenizer=pre_tokenizers.Whitespace()
            tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]')
            tok.chat_template="{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
            tok.save_pretrained(directory)
            counter=LocalTokenCounter(directory,tokenizer_tree(directory))
            self.assertGreater(counter.count('TEST_ONLY','TEST_ONLY'),0)
            (Path(directory)/'TEST_ONLY_changed.txt').write_text('TEST_ONLY',encoding='utf8')
            with self.assertRaises(ValueError):counter.count('TEST_ONLY','TEST_ONLY')
