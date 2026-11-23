"""TEST_ONLY remaining15/combination contracts, zero real generation."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_answer_grounded_compact import fixture as mini_fixture,CompactFakeHTTP,Counter
from app.locdi.answer_grounded_compact_extension import (build_bundle,combine,run_collection,validate,
    ExtensionApproval,VERSION,PROMPT_VERSION,NAMESPACE,DIGEST,request_payload,model_input,schema,score,sha)
from app.locdi.answer_grounded_compact_collection import CompactApproval
from app.locdi.answer_grounded_live_collection import PinnedCollectionApproval
from app.locdi.answer_grounded_routing_evaluation import evaluate
from app.locdi.llm_setup import digest


def fixture():
    old,c,oe,parent=mini_fixture();new=build_bundle(parent)
    common={'task_version':VERSION,'prompt_version':PROMPT_VERSION,'schema_fingerprint':digest(schema()),
        'task_prompt_fingerprint':'TEST_PROMPT','model_version':'TEST_V','tokenizer_tree_sha256':'TEST_TREE',
        'retry':0,'repair':0,'warmup':0,'source_sha256':'TEST_SOURCE'}
    oe.update(common);oe['plan_fingerprint']=old['plan']['fingerprint']
    oe['fingerprint']=digest({k:v for k,v in oe.items() if k!='fingerprint'})
    e={**common,'extension_plan_fingerprint':new['plan']['fingerprint'],'task_order_fingerprint':new['plan']['task_order_fingerprint'],
        'c4_26_protocol_fingerprint':oe['fingerprint'],'parent_plan_fingerprint':parent['plan']['fingerprint'],
        'namespace':NAMESPACE,'run_id':'c4_27_compact_extension_001','max_generations':15,'settings':new['plan']['settings'],
        'context_margin':512,'input_bytes_max':65536,'endpoint':oe['endpoint'],'model':oe['model'],'model_digest':DIGEST}
    e['fingerprint']=digest(e)
    return parent,old,oe,new,c,e


def rows(bundle,mode='unresolved'):
    output=[]
    for q,ref in zip(bundle['questions'],bundle['references']):
        for a in bundle['bindings']:
            value={'status':'unresolved','diagnosis':None} if mode=='unresolved' else {'status':'supported','diagnosis':ref['answer']}
            output.append({**score(q,ref,value),'agent_id':a['agent_id']})
    return output


class ExtensionTests(unittest.TestCase):
    def setUp(self):self.parent,self.old,self.oe,self.b,self.c,self.e=fixture()
    def run_fake(self,mode='valid',counter=None):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)/'TEST_ONLY_EXTENSION';f=CompactFakeHTTP(mode)
        with patch.object(ExtensionApproval,'verify',return_value={'TEST_ONLY':'NOT_REAL_APPROVAL'}):
            r=run_collection(self.b,self.c,self.e,root,execute=True,allow_llm=True,approval=ExtensionApproval('TEST','TEST'),
                counter=counter or Counter(),transport_factory=lambda:f,test_only=True)
        return r,f,root
    def combined(self,oldrows=None,newrows=None):
        return combine(self.parent,self.old,self.oe,oldrows or rows(self.old),self.b,self.e,newrows or rows(self.b))

    def test_original_questions345(self):self.assertEqual(self.b['questions'],self.parent['questions'][2:5])
    def test_no_old_questions(self):self.assertFalse({q['question_id'] for q in self.old['questions']} & {q['question_id'] for q in self.b['questions']})
    def test_same_binding_prompts(self):self.assertEqual(self.b['bindings'],self.old['bindings'])
    def test_same_task_schema(self):self.assertTrue(all(t['schema_fingerprint']==digest(schema()) for t in self.b['plan']['tasks']))
    def test_user_input_unchanged(self):self.assertEqual([model_input(q) for q in self.b['questions']],[model_input(q) for q in self.parent['questions'][2:]])
    def test_task_order(self):self.assertEqual([(t['question_id'],t['agent_id']) for t in self.b['plan']['tasks']],[(q['question_id'],a['agent_id']) for q in self.parent['questions'][2:] for a in self.parent['bindings']])
    def test_plan_fingerprint(self):self.assertEqual(self.b['plan']['fingerprint'],digest({k:v for k,v in self.b['plan'].items() if k!='fingerprint'}))
    def test_plan_tamper(self):
        self.b['plan']['tasks'].reverse()
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate(self.b,self.c,self.e,True)
    def test_protocol_tamper(self):
        self.e['max_generations']=16
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate(self.b,self.c,self.e,True)
    def test_budget16_rejected(self):
        self.e['max_generations']=16;self.e['fingerprint']=digest({k:v for k,v in self.e.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'BINDING'):validate(self.b,self.c,self.e,True)
    def test_dryrun_no_transport(self):
        calls=[];v=run_collection(self.b,self.c,self.e,'UNUSED',test_only=True,transport_factory=lambda:calls.append(1));self.assertEqual(calls,[]);self.assertEqual(v['generation_requests'],0)
    def test_no_approval(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,allow_llm=True,test_only=True)
    def test_old_compact_approval_rejected(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=CompactApproval('TEST','TEST'),test_only=True)
    def test_old25_approval_rejected(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=PinnedCollectionApproval('TEST','TEST'),test_only=True)
    def test_allow_llm_required(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.c,self.e,'UNUSED',execute=True,approval=ExtensionApproval('TEST','TEST'),test_only=True)
    def test_budget15(self):
        r,f,p=self.run_fake();self.assertEqual(r['attempted'],15);self.assertEqual(len(f.calls),15);self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['actual_llm_generations'],0)
    def test_no_first2_recall(self):
        r,f,p=self.run_fake();self.assertTrue(all(json.loads(x['messages'][1]['content']) not in [model_input(q) for q in self.old['questions']] for x in f.calls))
    def test_unresolved_all_three_groups_continue(self):
        r,f,p=self.run_fake('unresolved');self.assertEqual(r['attempted'],15);self.assertEqual(r['evaluation']['complete_collection_groups'],3);self.assertEqual(r['evaluation']['complete_scorable_groups'],0)
    def test_invalid_stops(self):
        r,f,p=self.run_fake('invalid');self.assertEqual(r['attempted'],1);self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],14)
    def test_http_stops(self):
        r,f,p=self.run_fake('http');self.assertEqual(len(f.calls),1);self.assertTrue((p/'Request001_response_body.bin').exists())
    def test_timeout_budget_consumed(self):
        r,f,p=self.run_fake('timeout');self.assertEqual(r['attempted'],1);self.assertEqual(r['evaluation']['all_agents']['counts']['GENERATION_FAILED'],1)
    def test_no_repair_on_length(self):
        r,f,p=self.run_fake('length');self.assertEqual(len(f.calls),1);self.assertEqual(r['stop_reason'],'OUTPUT_LIMIT_TRUNCATION')
    def test_schema_extra_rejected(self):
        r,f,p=self.run_fake('extra');self.assertEqual(r['attempted'],1);self.assertEqual(r['stop_reason'],'INVALID_RESPONSE')
    def test_digest_mismatch_blocks(self):
        r,f,p=self.run_fake('digest');self.assertEqual(r['attempted'],0)
    def test_postcontext_blocks(self):
        r,f,p=self.run_fake('context');self.assertEqual(r['attempted'],1);self.assertIn('CONTEXT',r['stop_reason'])
    def test_insufficient_tokens_no_generate(self):
        r,f,p=self.run_fake(counter=Counter(32768));self.assertEqual(r['attempted'],0)
    def test_checkpoint_parent_order(self):
        r,f,p=self.run_fake();a=json.loads((p/'Request001_before.json').read_text());z=json.loads((p/'Request015_after.json').read_text());self.assertEqual(a['parent_task_index'],11);self.assertEqual(z['parent_task_index'],25)
    def test_no_gold_payload(self):
        r,f,p=self.run_fake();self.assertTrue(all('TEST_TARGET' not in json.dumps(x) for x in f.calls))
    def test_scorer_reused(self):
        r,f,p=self.run_fake();self.assertEqual(r['evaluation']['all_agents']['counts']['CORRECT_ANSWER'],15)
    def test_partial_registered_denominator(self):
        r,f,p=self.run_fake('invalid');self.assertEqual(r['evaluation']['all_agents']['registered'],15)
    def test_run_no_overwrite(self):
        r,f,p=self.run_fake();original=(p/'collection.json').read_bytes()
        with patch.object(ExtensionApproval,'verify',return_value={'TEST_ONLY':True}):
            with self.assertRaises(FileExistsError):run_collection(self.b,self.c,self.e,p,execute=True,allow_llm=True,approval=ExtensionApproval('TEST','TEST'),test_only=True,counter=Counter(),transport_factory=lambda:f)
        self.assertEqual((p/'collection.json').read_bytes(),original)
    def test_combine25(self):self.assertEqual(self.combined()['evaluation']['all_agents']['registered'],25)
    def test_combined_original_order(self):self.assertEqual([r['question_id'] for r in self.combined()['rows']][::5],[q['question_id'] for q in self.parent['questions']])
    def test_no_clinical_false_for_abstention(self):self.assertTrue(all(r['answer_correct'] is None for r in self.combined()['rows']))
    def test_combined_agent_denominator5(self):self.assertTrue(all(v['registered']==5 for v in self.combined()['evaluation']['per_agent'].values()))
    def test_complete5_but_scorable0(self):
        e=self.combined()['evaluation'];self.assertEqual(e['complete_collection_groups'],5);self.assertEqual(e['complete_scorable_groups'],0)
    def test_duplicate_run_rows_rejected(self):
        new=rows(self.b);new[0]=deepcopy(rows(self.old)[0])
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):self.combined(newrows=new)
    def test_different_prompt_rejected(self):
        self.b['bindings'][0]['system_prompt']='CHANGED'
        with self.assertRaisesRegex(ValueError,'INCOMPATIBLE'):self.combined()
    def test_different_schema_rejected(self):
        self.e['schema_fingerprint']='OTHER'
        with self.assertRaisesRegex(ValueError,'INCOMPATIBLE'):self.combined()
    def test_v1_results_rejected(self):
        self.oe['task_version']='OLD_V1'
        with self.assertRaisesRegex(ValueError,'INCOMPATIBLE'):self.combined()
    def test_different_settings_rejected(self):
        self.e['settings']=dict(self.e['settings'],seed=99)
        with self.assertRaisesRegex(ValueError,'INCOMPATIBLE'):self.combined()
    def test_different_source_rejected(self):
        self.e['source_sha256']='OTHER'
        with self.assertRaisesRegex(ValueError,'INCOMPATIBLE'):self.combined()
    def test_prior_protocol_pin_rejected(self):
        self.e['c4_26_protocol_fingerprint']='OTHER'
        with self.assertRaisesRegex(ValueError,'PRIOR_PROTOCOL'):self.combined()
    def test_missing_registered_rows_rejected(self):
        with self.assertRaisesRegex(ValueError,'REGISTERED'):self.combined(newrows=rows(self.b)[:-1])
    def test_incomplete_cannot_impute_oracle(self):
        new=rows(self.b);new[0].update(state='NOT_ATTEMPTED',answer_correct=None)
        v=self.combined(newrows=new)['potential'];self.assertEqual(v['complete_paired_questions'],4)
    def test_zero_denominator_oracle_null(self):
        old=rows(self.old);new=rows(self.b)
        for r in old+new:r.update(state='NOT_ATTEMPTED',answer_correct=None)
        self.assertIsNone(self.combined(old,new)['potential']['oracle_gap'])
    def test_positive_oracle_same_denominator(self):
        old=rows(self.old);new=rows(self.b);allrows=old+new
        for i in range(5):allrows[i*5+i].update(state='CORRECT_ANSWER',answer_correct=True)
        v=self.combined(old,new)['potential'];self.assertEqual(v['oracle_gap'],.8);self.assertEqual(v['best_fixed_posthoc']['denominator'],5)
    def test_prior_completed_code_unchanged(self):
        m=json.loads(Path('outputs/research_evaluation/c4_26_v1/completion_manifest.json').read_text(encoding='utf-8'))
        for n in ['app/locdi/answer_grounded_compact.py','app/locdi/answer_grounded_compact_collection.py','scripts/collect_compact_answer_grounded.py','tests/locdi/test_answer_grounded_compact.py']:self.assertEqual(sha(n),m['files'][n])
    def test_new_approval_scope(self):
        now=datetime.now(timezone.utc)
        a={'scope':'LIVE_COMPACT_EXTENSION_REMAINING_15_ONLY','authorization_source':'EXPLICIT_USER_STATEMENT','operator_reference':'USER_LOCAL_SERVER_OPERATOR',
            'execution_protocol_fingerprint':self.e['fingerprint'],'parent_plan_fingerprint':self.e['parent_plan_fingerprint'],
            'c4_26_protocol_fingerprint':self.e['c4_26_protocol_fingerprint'],'extension_plan_fingerprint':self.e['extension_plan_fingerprint'],
            'task_order_fingerprint':self.e['task_order_fingerprint'],'source_sha256':self.e['source_sha256'],'model_digest':DIGEST,
            'task_version':VERSION,'prompt_version':PROMPT_VERSION,'schema_fingerprint':self.e['schema_fingerprint'],'endpoint':self.e['endpoint'],
            'max_generations':15,'run_id':self.e['run_id'],'namespace':NAMESPACE,'retry':0,'repair':0,'warmup':0,
            'allow_first_fixed_task_context_load':True,'user_statement':'TEST_ONLY_NOT_REAL_APPROVAL','recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        with patch('app.locdi.answer_grounded_compact_extension.pinned_private_json',return_value=a):
            app=ExtensionApproval('TEST','TEST');self.assertIn('approval_sha256',app.verify(self.e,now));a['max_generations']=10
            with self.assertRaisesRegex(ValueError,'SCOPE'):app.verify(self.e,now)
            a.update(max_generations=15,recorded_at=(now-timedelta(hours=2)).isoformat(),expires_at=(now-timedelta(hours=1)).isoformat())
            with self.assertRaisesRegex(ValueError,'VALIDITY'):app.verify(self.e,now)
    def test_missing_approval_evidence(self):
        with patch('app.locdi.answer_grounded_compact_extension.pinned_private_json',return_value={}):
            with self.assertRaises(ValueError):ExtensionApproval('TEST','TEST').verify(self.e)


if __name__=='__main__':unittest.main()
