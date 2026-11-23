"""TEST_ONLY synthetic symbols/HTTP, zero real server or generation access."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import digest
from app.locdi.answer_grounded_routing import _pair,bind_agents,build_plan
from app.locdi.answer_grounded_live_collection import (MODEL,DIGEST,PinnedCollectionApproval,validate_bundle,
    server_state,request_payload,run_collection,selection_potential,atomic_json)
from app.locdi.answer_grounded_routing_evaluation import evaluate


def fixture():
    departments=['cardiology','neurology','nephrology','endocrinology','pulmonology']
    agents=[{'agent_id':'TEST_ER','role':'er','department':'','model':MODEL,'version':'TEST_V','keywords':[]}]
    agents += [{'agent_id':'TEST_'+d,'role':'specialty','department':d,'model':MODEL,'version':'TEST_V','keywords':[]} for d in departments]
    cfg={'model':MODEL,'model_identifier':DIGEST,'model_version':'TEST_V',
        'agent_bindings':[{'agent_id':a['agent_id'],'model_identifier':DIGEST} for a in agents]}
    bs=bind_agents({'schema_version':1,'agents':agents},cfg,lambda d:'TEST_ONLY '+d)
    pairs=[_pair({'question':'TEST_ONLY static problem %d'%i,'options':[]},{'TEST_ONLY_SOURCE':i},
        'TEST_ONLY','TEST_REVISION',i,'UNKNOWN','TEST_ONLY','exact_text','TEST_TARGET',None) for i in range(1,6)]
    qs=[q for q,r in pairs];rs=[r for q,r in pairs]
    settings={'temperature':0,'seed':42,'context':32768,'max_output_tokens':1024,'timeout_seconds':120,'retry':0,'repair':0}
    p=build_plan(qs,rs,bs,settings,run_id='c4_24_pilot_001')
    bundle={'plan':p,'questions':qs,'references':rs,'bindings':bs,'source_pins':{}}
    e={'parent_plan_fingerprint':p['fingerprint'],'run_id':p['run_id'],'namespace':'TEST_ONLY',
        'model':MODEL,'model_digest':DIGEST,'settings':settings,'context_margin':512,'input_bytes_max':65536,
        'max_generations':25,'endpoint':'http://127.0.0.1:11434'}
    e['fingerprint']=digest(e)
    return bundle,cfg,e


class Counter:
    def __init__(self,n=50):self.n=n
    def count(self,system,user):return self.n


class FakeApproval(PinnedCollectionApproval):
    def __init__(self):pass
    def verify(self,execution,now=None):return {'approval_sha256':'TEST_ONLY','basis':'NONCLINICAL_TEST_FIXTURE'}


class FakeHTTP:
    TEST_ONLY=True
    def __init__(self,mode='valid'):
        self.mode=mode;self.loaded=False;self.calls=[];self.metadata_requests=0;self.closed=False
    def get(self,path):
        self.metadata_requests+=1
        if path=='/api/version':return {'version':'TEST_ONLY'}
        if path=='/api/tags':return {'models':[{'name':MODEL,'digest':'b'*64 if self.mode=='digest' else DIGEST}]}
        if path=='/api/ps':return {'models':[{'name':MODEL,'digest':DIGEST,'context_length':4096 if self.mode=='context' else 32768}] if self.loaded else []}
        raise AssertionError(path)
    def generate(self,payload):
        self.calls.append(deepcopy(payload));self.loaded=True
        if self.mode=='timeout':raise TimeoutError('TEST_ONLY')
        content={'status':'supported','answer_id':None,'diagnosis':'TEST_TARGET','rationale':'TEST_ONLY'}
        if self.mode=='unresolved':content.update(status='unresolved',diagnosis=None)
        if self.mode=='wrong':content['diagnosis']='TEST_WRONG'
        if self.mode=='semantic':content.update(status='unresolved',diagnosis='TEST_TARGET')
        env={'model':MODEL,'done':True,'message':{'content':'{broken' if self.mode=='invalid' else json.dumps(content)},
             'eval_count':2000 if self.mode=='output' else 25,'prompt_eval_count':50,'done_reason':'length' if self.mode=='length' else 'stop'}
        return {'http_status':500 if self.mode=='http' else 200,'body':json.dumps(env).encode()}
    def close(self):self.closed=True


class LiveCollectionTests(unittest.TestCase):
    def run_fake(self,mode='valid',counter=None,execution_mutator=None):
        b,c,e=fixture()
        if execution_mutator:
            execution_mutator(e);e['fingerprint']=digest({k:v for k,v in e.items() if k!='fingerprint'})
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)/'TEST_ONLY_RUN';client=FakeHTTP(mode)
        r=run_collection(b,c,e,root,execute=True,allow_llm=True,approval=FakeApproval(),
            transport_factory=lambda:client,counter=counter or Counter(),test_only=True)
        return r,client,root,b,e

    def test_fixed_plan(self):
        b,c,e=fixture();self.assertEqual(len(validate_bundle(b,c)['tasks']),25)

    def test_plan_fingerprint(self):
        b,c,e=fixture();b['plan']['run_id']='CHANGED'
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate_bundle(b,c)

    def test_schema_or_task_order_changed(self):
        b,c,e=fixture();b['plan']['tasks'].reverse();b['plan']['fingerprint']=digest({k:v for k,v in b['plan'].items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'TASK_ORDER'):validate_bundle(b,c)

    def test_prompt_changed(self):
        b,c,e=fixture();b['bindings'][0]['system_prompt']='CHANGED'
        with self.assertRaises(ValueError):validate_bundle(b,c)

    def test_dry_run_no_factory_or_directory(self):
        b,c,e=fixture();calls=[]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'UNUSED';r=run_collection(b,c,e,p,transport_factory=lambda:calls.append(True))
            self.assertEqual(r['generation_requests'],0);self.assertEqual(calls,[]);self.assertFalse(p.exists())

    def test_no_approval_before_client(self):
        b,c,e=fixture();calls=[]
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(b,c,e,'UNUSED',execute=True,allow_llm=True,transport_factory=lambda:calls.append(True))
        self.assertEqual(calls,[])

    def test_allow_llm_flag_required(self):
        b,c,e=fixture()
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(b,c,e,'UNUSED',execute=True,approval=FakeApproval())

    def test_model_digest_preflight(self):
        r,f,p,b,e=self.run_fake('digest');self.assertEqual(r['attempted'],0);self.assertEqual(len(f.calls),0)
        self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],25)

    def test_unloaded_context_is_pending_not_verified(self):
        b,c,e=fixture();self.assertEqual(server_state(FakeHTTP(),e)['loaded_context_status'],'PENDING_FIRST_FIXED_TASK_LOAD')

    def test_loaded_wrong_context_blocks(self):
        b,c,e=fixture();f=FakeHTTP('context');f.loaded=True
        with self.assertRaisesRegex(ValueError,'CONTEXT_MISMATCH'):server_state(f,e)

    def test_post_request_context_stops(self):
        r,f,p,b,e=self.run_fake('context');self.assertEqual(r['attempted'],1)
        self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],24)

    def test_token_context_insufficient_no_generation(self):
        r,f,p,b,e=self.run_fake(counter=Counter(33000));self.assertEqual(len(f.calls),0)

    def test_budget_25(self):
        r,f,p,b,e=self.run_fake();self.assertEqual(r['attempted'],25);self.assertEqual(len(f.calls),25)
        self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['actual_llm_generations'],0)

    def test_smaller_budget_keeps_remaining_slots(self):
        r,f,p,b,e=self.run_fake(execution_mutator=lambda e:e.update(max_generations=2))
        self.assertEqual(r['attempted'],2);self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],23)

    def test_request_options_no_retry_repair(self):
        r,f,p,b,e=self.run_fake();payload=f.calls[0]
        self.assertEqual(payload['options'],{'temperature':0,'seed':42,'num_ctx':32768,'num_predict':1024})
        self.assertFalse(payload['stream']);self.assertEqual(len(f.calls),25)

    def test_timeout_consumes_budget_no_retry(self):
        r,f,p,b,e=self.run_fake('timeout');self.assertEqual(len(f.calls),1);self.assertEqual(r['attempted'],1)
        self.assertEqual(r['evaluation']['all_agents']['counts']['GENERATION_FAILED'],1)

    def test_http_failure_stops(self):
        r,f,p,b,e=self.run_fake('http');self.assertEqual(len(f.calls),1);self.assertEqual(r['http_success'],0)
        self.assertTrue((p/'Request001_response_body.bin').exists())

    def test_invalid_json_distinct_from_generation_failure(self):
        r,f,p,b,e=self.run_fake('invalid');self.assertEqual(r['evaluation']['all_agents']['counts']['INVALID_RESPONSE'],1)
        self.assertEqual(r['evaluation']['all_agents']['counts']['GENERATION_FAILED'],0)

    def test_semantic_schema_error_stops(self):
        r,f,p,b,e=self.run_fake('semantic');self.assertEqual(r['attempted'],1);self.assertEqual(r['schema_valid'],0)

    def test_valid_unresolved_continues(self):
        r,f,p,b,e=self.run_fake('unresolved');self.assertEqual(r['attempted'],25)
        self.assertEqual(r['evaluation']['complete_collection_groups'],5);self.assertEqual(r['evaluation']['complete_scorable_groups'],0)

    def test_exact_match_unchanged(self):
        r,f,p,b,e=self.run_fake();self.assertEqual(r['evaluation']['all_agents']['counts']['CORRECT_ANSWER'],25)

    def test_mismatch_not_reclassified(self):
        r,f,p,b,e=self.run_fake('wrong');self.assertEqual(r['evaluation']['all_agents']['counts']['INCORRECT_ANSWER'],25)

    def test_server_output_budget_stops(self):
        r,f,p,b,e=self.run_fake('output');self.assertEqual(r['attempted'],1);self.assertEqual(r['stop_reason'],'SERVER_OUTPUT_TOKEN_BUDGET_EXCEEDED')

    def test_output_truncation_invalid(self):
        r,f,p,b,e=self.run_fake('length');self.assertEqual(r['evaluation']['all_agents']['counts']['INVALID_RESPONSE'],1)

    def test_checkpoints_and_manifest(self):
        r,f,p,b,e=self.run_fake();self.assertTrue((p/'Request001_before.json').exists());self.assertTrue((p/'Request025_after.json').exists())
        self.assertTrue((p/'completion_manifest.json').exists());self.assertTrue(f.closed)

    def test_no_answer_reference_in_request(self):
        r,f,p,b,e=self.run_fake()
        for payload in f.calls:
            user=json.loads(payload['messages'][1]['content']);self.assertEqual(set(user),{'question','options'})
            self.assertNotIn('TEST_TARGET',payload['messages'][1]['content'])

    def test_no_run_overwrite(self):
        r,f,p,b,e=self.run_fake();before=(p/'collection.json').read_bytes();calls=[]
        with self.assertRaises(FileExistsError):run_collection(b,fixture()[1],e,p,execute=True,allow_llm=True,approval=FakeApproval(),
            transport_factory=lambda:calls.append(True),counter=Counter(),test_only=True)
        self.assertEqual(calls,[]);self.assertEqual((p/'collection.json').read_bytes(),before)

    def test_after_checkpoint_failure_preserves_attempt(self):
        original=atomic_json
        def broken(path,data):
            if Path(path).name=='Request001_after.json':raise OSError('TEST_ONLY_EXPORT')
            return original(path,data)
        with patch('app.locdi.answer_grounded_live_collection.atomic_json',side_effect=broken):r,f,p,b,e=self.run_fake()
        self.assertEqual(r['attempted'],1);self.assertEqual(r['evaluation']['all_agents']['counts']['CORRECT_ANSWER'],1)
        self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],24)

    def test_atomic_write_refuses_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST.json';atomic_json(p,{'TEST':1})
            with self.assertRaises(FileExistsError):atomic_json(p,{'TEST':2})
            self.assertEqual(json.loads(p.read_text())['TEST'],1)

    def test_atomic_serialization_failure_leaves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST.json';atomic_json(p,{'TEST':1})
            with self.assertRaises(ValueError):atomic_json(p,{'TEST':float('nan')})
            self.assertEqual(json.loads(p.read_text())['TEST'],1)

    def test_paired_oracle_gap(self):
        b,c,e=fixture();rows=[]
        for i,q in enumerate(b['questions']):
            for j,a in enumerate(b['bindings']):
                state=('CORRECT_ANSWER' if j==i else 'INCORRECT_ANSWER') if i<2 else 'NOT_ATTEMPTED'
                rows.append({'question_id':q['question_id'],'agent_id':a['agent_id'],'state':state})
        r=selection_potential(b,rows);self.assertEqual(r['complete_paired_questions'],2)
        self.assertEqual(r['oracle_posthoc_upper_bound']['value'],1);self.assertEqual(r['best_fixed_posthoc']['value'],.5)
        self.assertEqual(r['oracle_gap'],.5);self.assertEqual(r['uniform_random_expected']['value'],.2)

    def test_unattempted_agent_never_oracle_imputed(self):
        b,c,e=fixture();rows=[]
        for q in b['questions']:
            for j,a in enumerate(b['bindings']):rows.append({'question_id':q['question_id'],'agent_id':a['agent_id'],'state':'CORRECT_ANSWER' if j==0 else 'NOT_ATTEMPTED'})
        r=selection_potential(b,rows);self.assertEqual(r['complete_paired_questions'],0);self.assertIsNone(r['oracle_gap'])

    def test_zero_oracle_gap_for_identical_agents(self):
        r,f,p,b,e=self.run_fake();self.assertEqual(selection_potential(b,r['rows'])['oracle_gap'],0)

    def test_protocol_hash_changed(self):
        b,c,e=fixture();e['max_generations']=100
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):run_collection(b,c,e,'UNUSED')

    def test_test_fixture_authority_cannot_authorize_live(self):
        b,c,e=fixture()
        with self.assertRaisesRegex(ValueError,'LIVE_PINNED_APPROVAL_TYPE'):
            run_collection(b,c,e,'UNUSED',execute=True,allow_llm=True,approval=FakeApproval())

    def test_approval_binding_and_validity(self):
        now=datetime.now(timezone.utc);e={'fingerprint':'TEST_PROTOCOL','parent_plan_fingerprint':'TEST_PLAN','run_id':'TEST_RUN',
            'namespace':'TEST_NAMESPACE','source_sha256':'TEST_SOURCE','task_version':'TEST_TASK','model_digest':'TEST_DIGEST',
            'task_prompt_version':'TEST_PROMPT','endpoint':'TEST_ENDPOINT'}
        a={'scope':'LIVE_ANSWER_GROUNDED_PILOT_25_ONLY','execution_protocol_fingerprint':e['fingerprint'],
            'pilot_plan_fingerprint':e['parent_plan_fingerprint'],'run_id':e['run_id'],'namespace':e['namespace'],
            'source_sha256':e['source_sha256'],'task_version':e['task_version'],'model_digest':e['model_digest'],
            'task_prompt_version':e['task_prompt_version'],'endpoint':e['endpoint'],'max_generations':25,'retry':0,'repair':0,
            'authorization_source':'EXPLICIT_USER_STATEMENT','allow_first_fixed_task_context_load':True,
            'user_statement':'TEST_ONLY symbolic evidence; not real approval','operator_reference':'USER_LOCAL_SERVER_OPERATOR',
            'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        with patch('app.locdi.answer_grounded_live_collection.pinned_private_json',return_value=a):
            p=PinnedCollectionApproval('TEST_ONLY','TEST_ONLY');self.assertIn('approval_sha256',p.verify(e,now))
            a['model_digest']='CHANGED'
            with self.assertRaisesRegex(ValueError,'SCOPE'):p.verify(e,now)


if __name__=='__main__':unittest.main()
