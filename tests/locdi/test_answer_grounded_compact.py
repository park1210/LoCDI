"""TEST_ONLY symbolic compact responses; no real model or network."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_answer_grounded_live_collection import fixture as parent_fixture,Counter,FakeHTTP
from app.locdi.answer_grounded_routing import build_plan,model_input
from app.locdi.answer_grounded_compact import (VERSION,PROMPT_VERSION,build_bundle,parse,schema,score,repetition)
from app.locdi.answer_grounded_compact_collection import (CompactApproval,run_collection,request_payload,
    stability,validate,NAMESPACE,MODEL,DIGEST,selection_potential)
from app.locdi.answer_grounded_live_collection import PinnedCollectionApproval,atomic_json,sha
from app.locdi.llm_setup import digest


def fixture():
    p,c,_=parent_fixture()
    for a in p['bindings']:
        a['system_prompt']='TEST_ONLY old common rationale.\nYou are the '+a['department']+' department.\nFocus:\n- TEST_ONLY focus.\n- If certainty is limited, return a syndrome.'
        a['benchmark_prompt_fingerprint']=digest(a['system_prompt'])
    p['plan']=build_plan(p['questions'],p['references'],p['bindings'],p['plan']['settings'])
    b=build_bundle(p)
    ep={'plan_fingerprint':b['plan']['fingerprint'],'parent_plan_fingerprint':p['plan']['fingerprint'],
        'namespace':NAMESPACE,'run_id':'c4_26_compact_pilot_001','max_generations':10,'settings':b['plan']['settings'],
        'context_margin':512,'input_bytes_max':65536,'endpoint':'http://127.0.0.1:11434',
        'model':MODEL,'model_digest':DIGEST,'source_sha256':'TEST_ONLY_SOURCE','schema_fingerprint':digest(schema())}
    ep['fingerprint']=digest(ep)
    return b,c,ep,p


class CompactFakeHTTP(FakeHTTP):
    def generate(self,payload):
        self.calls.append(deepcopy(payload));self.loaded=True
        if self.mode=='timeout':raise TimeoutError('TEST_ONLY')
        content={'status':'supported','diagnosis':'TEST_TARGET'}
        if self.mode=='unresolved':content={'status':'unresolved','diagnosis':None}
        if self.mode=='semantic':content={'status':'unresolved','diagnosis':'TEST_TARGET'}
        if self.mode=='extra':content['rationale']='TEST_ONLY'
        env={'model':MODEL,'done':True,'done_reason':'length' if self.mode=='length' else 'stop',
            'message':{'content':'{broken' if self.mode=='invalid' else json.dumps(content)},
            'eval_count':1025 if self.mode=='overbudget' else 20,'prompt_eval_count':50}
        return {'http_status':500 if self.mode=='http' else 200,'body':json.dumps(env).encode()}


class CompactTests(unittest.TestCase):
    def setUp(self):self.b,self.c,self.e,self.parent=fixture();self.q=self.b['questions'][0];self.ref=self.b['references'][0]
    def response(self,value):return score(self.q,self.ref,value)
    def run_fake(self,mode='valid',counter=None,transport=None):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)/'TEST_ONLY_RUN'
        f=transport or CompactFakeHTTP(mode)
        with patch.object(CompactApproval,'verify',return_value={'TEST_ONLY':'NOT_REAL_APPROVAL'}):
            r=run_collection(self.b,self.c,self.e,root,execute=True,allow_llm=True,
                approval=CompactApproval('TEST_ONLY','TEST_ONLY'),counter=counter or Counter(),
                transport_factory=lambda:f,test_only=True)
        return r,f,root

    def test_supported(self):self.assertEqual(self.response({'status':'supported','diagnosis':'TEST_TARGET'})['state'],'CORRECT_ANSWER')
    def test_unresolved(self):self.assertEqual(self.response({'status':'unresolved','diagnosis':None})['state'],'UNRESOLVED')
    def test_supported_null(self):self.assertEqual(self.response({'status':'supported','diagnosis':None})['state'],'INVALID_RESPONSE')
    def test_supported_empty(self):self.assertEqual(self.response({'status':'supported','diagnosis':'  '})['state'],'INVALID_RESPONSE')
    def test_unresolved_nonnull(self):self.assertEqual(self.response({'status':'unresolved','diagnosis':'TEST_TARGET'})['state'],'INVALID_RESPONSE')
    def test_extra_rationale(self):self.assertEqual(self.response({'status':'unresolved','diagnosis':None,'rationale':'TEST_ONLY'})['state'],'INVALID_RESPONSE')
    def test_extra_answer_id(self):self.assertEqual(self.response({'status':'unresolved','diagnosis':None,'answer_id':None})['state'],'INVALID_RESPONSE')
    def test_duplicate_key(self):self.assertEqual(self.response('{"status":"supported","diagnosis":"TEST_TARGET","diagnosis":null}')['state'],'INVALID_RESPONSE')
    def test_truncated(self):self.assertEqual(self.response('{"status":"supported","diagnosis":"TEST')['state'],'INVALID_RESPONSE')
    def test_markdown(self):self.assertEqual(self.response('```json\n{"status":"unresolved","diagnosis":null}\n```')['state'],'INVALID_RESPONSE')
    def test_extra_explanation(self):self.assertEqual(self.response('{"status":"unresolved","diagnosis":null} explanation')['state'],'INVALID_RESPONSE')
    def test_status_enum(self):self.assertEqual(self.response({'status':'certain','diagnosis':'TEST_TARGET'})['state'],'INVALID_RESPONSE')
    def test_alternative_diagnoses(self):self.assertEqual(self.response({'status':'supported','diagnosis':'TEST_A or TEST_B'})['state'],'INVALID_RESPONSE')
    def test_diagnosis_list(self):self.assertEqual(self.response({'status':'supported','diagnosis':['TEST_A','TEST_B']})['state'],'INVALID_RESPONSE')
    def test_exact_normalization(self):self.assertTrue(self.response({'status':'supported','diagnosis':'  test_target. '})['answer_correct'])
    def test_lexical_mismatch_not_clinical(self):self.assertIs(self.response({'status':'supported','diagnosis':'TEST_OTHER'})['answer_correct'],False)
    def test_unresolved_correctness_null(self):self.assertIsNone(self.response({'status':'unresolved','diagnosis':None})['answer_correct'])
    def test_failed_correctness_null(self):self.assertIsNone(score(self.q,self.ref,error_type='TEST_ERROR')['answer_correct'])
    def test_notattempted(self):self.assertEqual(score(self.q,self.ref,attempted=False)['state'],'NOT_ATTEMPTED')
    def test_no_fabricated_rationale(self):self.assertNotIn('rationale',parse('{"status":"unresolved","diagnosis":null}'))
    def test_first_two_parent_only(self):self.assertEqual(self.b['questions'],self.parent['questions'][:2])
    def test_task_order(self):self.assertEqual([(t['question_id'],t['agent_id']) for t in self.b['plan']['tasks']],[(q['question_id'],a['agent_id']) for q in self.parent['questions'][:2] for a in self.parent['bindings']])
    def test_parent_inputs_unchanged(self):self.assertEqual([model_input(q) for q in self.b['questions']],[model_input(q) for q in self.parent['questions'][:2]])
    def test_new_prompt_no_legacy_prefix(self):
        for a in self.b['bindings']:
            self.assertNotIn('TEST_ONLY old common rationale',a['system_prompt']);self.assertNotIn('If certainty is limited',a['system_prompt']);self.assertIn('TEST_ONLY focus',a['system_prompt'])
    def test_roles_preserved(self):
        for a in self.b['bindings']:self.assertIn('You are the '+a['department']+' department.',a['system_prompt'])
    def test_plan_binding(self):self.assertEqual(self.b['plan']['fingerprint'],digest({k:v for k,v in self.b['plan'].items() if k!='fingerprint'}))
    def test_plan_tamper(self):
        self.b['plan']['tasks'].reverse()
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate(self.b,self.c,self.e,True)
    def test_protocol_tamper(self):
        self.e['max_generations']=11
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate(self.b,self.c,self.e,True)
    def test_budget_above_ten_rejected(self):
        self.e['max_generations']=11;self.e['fingerprint']=digest({k:v for k,v in self.e.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'BINDING'):validate(self.b,self.c,self.e,True)
    def test_dry_run_offline(self):
        calls=[];r=run_collection(self.b,self.c,self.e,'UNUSED',test_only=True,transport_factory=lambda:calls.append(1))
        self.assertEqual(r['generation_requests'],0);self.assertEqual(calls,[])
    def test_no_approval(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,allow_llm=True,test_only=True)
    def test_previous_approval_class_rejected(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=PinnedCollectionApproval('TEST','TEST'),test_only=True)
    def test_allow_llm_required(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,approval=CompactApproval('TEST','TEST'),test_only=True)
    def test_native_schema_sent(self):
        p,n=request_payload(self.q,self.b['bindings'][0],self.e,Counter());self.assertEqual(p['format'],schema());self.assertEqual(set(p['format']['properties']),{'status','diagnosis'})
    def test_native_options_unchanged(self):
        p,n=request_payload(self.q,self.b['bindings'][0],self.e,Counter());self.assertEqual(p['options'],{'temperature':0,'seed':42,'num_ctx':32768,'num_predict':1024});self.assertFalse(p['stream'])
    def test_no_reference_payload(self):
        p,n=request_payload(self.q,self.b['bindings'][0],self.e,Counter());self.assertNotIn('TEST_TARGET',json.dumps(p));self.assertEqual(json.loads(p['messages'][1]['content']),model_input(self.q))
    def test_insufficient_context(self):
        with self.assertRaisesRegex(ValueError,'CONTEXT'):request_payload(self.q,self.b['bindings'][0],self.e,Counter(32768))
    def test_ten_budget(self):
        r,f,p=self.run_fake();self.assertEqual(r['attempted'],10);self.assertEqual(len(f.calls),10);self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['actual_llm_generations'],0)
    def test_normal_unresolved_both_groups(self):
        r,f,p=self.run_fake('unresolved');self.assertEqual(len(f.calls),10);self.assertEqual(r['evaluation']['complete_collection_groups'],2);self.assertEqual(r['evaluation']['complete_scorable_groups'],0)
    def test_first_invalid_stops_before_group2(self):
        r,f,p=self.run_fake('invalid');self.assertEqual(len(f.calls),1);self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],9)
    def test_schema_extra_stops(self):
        r,f,p=self.run_fake('extra');self.assertEqual(r['attempted'],1);self.assertEqual(r['stop_reason'],'INVALID_RESPONSE')
    def test_semantic_stops(self):
        r,f,p=self.run_fake('semantic');self.assertEqual(r['attempted'],1)
    def test_http_stops(self):
        r,f,p=self.run_fake('http');self.assertEqual(r['attempted'],1);self.assertTrue((p/'Request001_response_body.bin').exists())
    def test_timeout_no_retry(self):
        r,f,p=self.run_fake('timeout');self.assertEqual(r['attempted'],1);self.assertEqual(len(f.calls),1);self.assertEqual(r['evaluation']['all_agents']['counts']['GENERATION_FAILED'],1)
    def test_truncation_stops(self):
        r,f,p=self.run_fake('length');self.assertEqual(r['stop_reason'],'OUTPUT_LIMIT_TRUNCATION');self.assertEqual(r['attempted'],1)
    def test_server_output_overbudget(self):
        r,f,p=self.run_fake('overbudget');self.assertEqual(r['attempted'],1);self.assertEqual(r['stop_reason'],'SERVER_OUTPUT_TOKEN_BUDGET_EXCEEDED')
    def test_model_digest(self):
        r,f,p=self.run_fake('digest');self.assertEqual(r['attempted'],0);self.assertEqual(len(f.calls),0)
    def test_context_after_first(self):
        r,f,p=self.run_fake('context');self.assertEqual(r['attempted'],1);self.assertIn('CONTEXT',r['stop_reason'])
    def test_checkpoints(self):
        r,f,p=self.run_fake();self.assertTrue((p/'Request010_after.json').exists());self.assertTrue((p/'completion_manifest.json').exists());self.assertTrue(f.closed)
    def test_run_no_overwrite(self):
        r,f,p=self.run_fake();old=(p/'collection.json').read_bytes()
        with patch.object(CompactApproval,'verify',return_value={'TEST_ONLY':True}):
            with self.assertRaises(FileExistsError):run_collection(self.b,self.c,self.e,p,execute=True,allow_llm=True,approval=CompactApproval('TEST','TEST'),test_only=True,counter=Counter(),transport_factory=lambda:f)
        self.assertEqual(old,(p/'collection.json').read_bytes())
    def test_zero_paired_oracle_null(self):
        r,f,p=self.run_fake('invalid');self.assertIsNone(selection_potential(self.b,r['rows'])['oracle_gap'])
    def test_all_unresolved_oracle_zero_not_clinical_false(self):
        r,f,p=self.run_fake('unresolved');v=selection_potential(self.b,r['rows']);self.assertEqual(v['complete_paired_questions'],2);self.assertEqual(v['oracle_gap'],0);self.assertTrue(all(row['answer_correct'] is None for row in r['rows']))
    def test_different_correct_agent_oracle_same_denominator(self):
        rows=[]
        for i,q in enumerate(self.b['questions']):
            for j,a in enumerate(self.b['bindings']):rows.append({'question_id':q['question_id'],'agent_id':a['agent_id'],'state':'CORRECT_ANSWER' if j==i else 'INCORRECT_ANSWER'})
        v=selection_potential(self.b,rows);self.assertEqual(v['oracle_gap'],.5);self.assertEqual(v['best_fixed_posthoc']['denominator'],2)
    def test_repetition_rule(self):self.assertTrue(repetition('one two three four five six '*5)['repetitive'])
    def test_no_repetition(self):self.assertFalse(repetition('{"status":"unresolved","diagnosis":null}')['repetitive'])
    def test_null_attempted_denominator(self):
        r,f,p=self.run_fake('digest');v=stability(self.b,r,[]);self.assertIsNone(v['attempted_denominator_rates']['schema_valid']['value']);self.assertEqual(v['registered_denominator_rates']['schema_valid']['denominator'],10)
    def test_registered_denominator_after_failure(self):
        r,f,p=self.run_fake('invalid');rec=json.loads((p/'Request001_after.json').read_text());v=stability(self.b,r,[rec]);self.assertEqual(v['all_slots_metrics']['registered'],10);self.assertEqual(v['attempted_denominator_rates']['http_success']['denominator'],1)
    def test_old_c425_code_pins_unchanged(self):
        m=json.loads(Path('outputs/research_evaluation/c4_25_v1/completion_manifest.json').read_text(encoding='utf-8'))
        for name in ['app/locdi/answer_grounded_live_collection.py','scripts/collect_answer_grounded_routing.py','tests/locdi/test_answer_grounded_live_collection.py']:self.assertEqual(sha(name),m['files'][name])
    def test_approval_new_scope_and_timing(self):
        now=datetime.now(timezone.utc);a={'scope':'LIVE_COMPACT_DIAGNOSTIC_MINI_PILOT_10_ONLY','authorization_source':'EXPLICIT_USER_STATEMENT','operator_reference':'USER_LOCAL_SERVER_OPERATOR',
            'execution_protocol_fingerprint':self.e['fingerprint'],'pilot_plan_fingerprint':self.e['plan_fingerprint'],'parent_plan_fingerprint':self.e['parent_plan_fingerprint'],
            'run_id':self.e['run_id'],'namespace':NAMESPACE,'source_sha256':self.e['source_sha256'],'task_version':VERSION,'prompt_version':PROMPT_VERSION,
            'schema_fingerprint':self.e['schema_fingerprint'],'model_digest':DIGEST,'endpoint':self.e['endpoint'],'max_generations':10,'retry':0,'repair':0,'warmup':0,
            'allow_first_fixed_task_context_load':True,'user_statement':'TEST_ONLY_NOT_REAL_APPROVAL','recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        with patch('app.locdi.answer_grounded_compact_collection.pinned_private_json',return_value=a):
            app=CompactApproval('TEST_ONLY','TEST_ONLY');self.assertIn('approval_sha256',app.verify(self.e,now))
            a['scope']='LIVE_ANSWER_GROUNDED_PILOT_25_ONLY'
            with self.assertRaisesRegex(ValueError,'SCOPE'):app.verify(self.e,now)
    def test_live_fake_subclass_approval_rejected(self):
        class Fake(CompactApproval):pass
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=Fake('TEST','TEST'),test_only=True)


if __name__=='__main__':unittest.main()
