"""TEST_ONLY symbolic model switching, common settings, and menu denominators."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from test_answer_grounded_compact import fixture as compact_fixture,Counter
from app.locdi.weight_differentiated_agent import (TAGS,EXPECTED_DIGESTS,NAMESPACE,RUN_ID,VERSION,PROMPT_VERSION,
    build_bundle,gguf_metadata,token_budget,run_collection,WeightApproval,validate,selection_menus,role_transitions,
    schema,score,model_input,sha,digest,model_state,verify_local_artifact,research_decision)
from app.locdi.answer_grounded_compact_extension import ExtensionApproval
from app.locdi.answer_grounded_compact_collection import CompactApproval


def fixture():
    old,c,e,parent=compact_fixture();models={}
    for tag in TAGS:
        models[tag]={'manifest_digest':EXPECTED_DIGESTS[tag],'gguf_sha256':'TEST_BLOB_'+tag,'quantization':'TEST_Q4' if tag==TAGS[0] else 'TEST_F16',
            'template_fingerprint':digest('TEST_TEMPLATE_'+tag),'parameters_fingerprint':digest('TEST_PARAMETERS_'+tag)}
    contract={'context':4096,'output_tokens':1024,'native_template_token_reserve':1024,'input_bytes_max':8192,
        'options':{'num_ctx':4096,'num_predict':1024,'temperature':0,'seed':42,'top_p':.9,'top_k':40,'min_p':0.0,'repeat_penalty':1.1,'repeat_last_n':64,'mirostat':0,'stop':['<|im_start|>','<|im_end|>','<|endoftext|>']}}
    artifacts={'models':models};bundle=build_bundle(parent,models,contract)
    ep={'namespace':NAMESPACE,'run_id':RUN_ID,'plan_fingerprint':bundle['plan']['fingerprint'],'parent_plan_fingerprint':parent['plan']['fingerprint'],
        'max_generations':20,'context':4096,'contract':contract,'model_bindings':models,'endpoint':'http://127.0.0.1:11434',
        'source_sha256':'TEST_SOURCE','task_order_fingerprint':digest(bundle['plan']['tasks']),'schema_fingerprint':digest(schema())}
    ep['fingerprint']=digest(ep)
    return bundle,artifacts,contract,ep


class FakeHTTP:
    TEST_ONLY=True
    def __init__(self,mode='unresolved',preload=None):
        self.mode=mode;self.current=preload;self.calls=[];self.unloads=[];self.context=4096;self.metadata_requests=0;self.closed=False
    def get(self,path):
        self.metadata_requests+=1
        if path=='/api/version':return {'version':'TEST_ONLY'}
        if path=='/api/tags':return {'models':[{'name':t,'digest':'b'*64 if self.mode=='digest' and t==TAGS[1] else EXPECTED_DIGESTS[t]} for t in TAGS]}
        if path=='/api/ps':return {'models':[] if not self.current else [{'name':self.current,'digest':EXPECTED_DIGESTS.get(self.current,'TEST_FOREIGN'),
            'context_length':self.context,'size':1000,'size_vram':900}]}
        raise AssertionError(path)
    def show_model(self,tag):
        self.metadata_requests+=1
        return {'template':'CHANGED' if self.mode=='template' and tag==TAGS[1] else 'TEST_TEMPLATE_'+tag,'parameters':'TEST_PARAMETERS_'+tag}
    def unload(self,tag):
        self.metadata_requests+=1;self.unloads.append(tag)
        if self.mode=='unload_error':raise ValueError('UNLOAD_NOT_NON_GENERATIVE')
        self.current=None;return {'done':True,'done_reason':'unload','response':''}
    def generate(self,payload):
        self.calls.append(deepcopy(payload));self.current=payload['model'];self.context=2048 if self.mode=='context' else 4096
        if self.mode=='timeout':raise TimeoutError('TEST_ONLY')
        value={'status':'supported','diagnosis':'TEST_TARGET'} if self.mode=='correct' else {'status':'unresolved','diagnosis':None}
        if self.mode=='extra':value['rationale']='TEST_ONLY'
        raw='{bad' if self.mode=='invalid' else json.dumps(value)
        env={'model':TAGS[2] if self.mode=='wrong_reply' else self.current,'done':True,'message':{'content':raw},
            'prompt_eval_count':5000 if self.mode=='input_overflow' else 50,'eval_count':1025 if self.mode=='output_overflow' else 12,
            'done_reason':'length' if self.mode=='length' else 'stop'}
        return {'http_status':500 if self.mode=='http' else 200,'body':json.dumps(env).encode()}
    def close(self):self.closed=True


def all_rows(bundle):
    rs=[]
    for q,ref in zip(bundle['questions'],bundle['references']):
        for a in bundle['bindings']:rs.append({**score(q,ref,{'status':'unresolved','diagnosis':None}),'agent_id':a['agent_id']})
    return rs


class WeightTests(unittest.TestCase):
    def setUp(self):self.b,self.a,self.c,self.e=fixture()
    def fake(self,mode='unresolved',preload=None,counter=None):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)/'TEST_ONLY_WEIGHT_RUN';http=FakeHTTP(mode,preload)
        with patch.object(WeightApproval,'verify',return_value={'TEST_ONLY':'NOT_REAL_APPROVAL'}):
            r=run_collection(self.b,self.a,self.c,self.e,root,execute=True,allow_llm=True,approval=WeightApproval('TEST','TEST'),
                counter=counter or Counter(),transport_factory=lambda:http,test_only=True)
        return r,http,root
    def test_four_bindings(self):self.assertEqual([a['agent_id'] for a in self.b['bindings']],list('ABCD'))
    def test_three_models(self):self.assertEqual({a['model'] for a in self.b['bindings']},set(TAGS))
    def test_neurology_prompt_equal(self):self.assertEqual(self.b['bindings'][0]['system_prompt'],self.b['bindings'][1]['system_prompt'])
    def test_pulmonology_prompt_equal(self):self.assertEqual(self.b['bindings'][2]['system_prompt'],self.b['bindings'][3]['system_prompt'])
    def test_five_questions(self):self.assertEqual(len(self.b['questions']),5)
    def test_fixed20_order(self):self.assertEqual([(t['agent_id'],t['parent_question_ordinal']) for t in self.b['plan']['tasks']],[(a,i) for a in 'ABCD' for i in range(1,6)])
    def test_plan_fingerprint(self):self.assertEqual(self.b['plan']['fingerprint'],digest({k:v for k,v in self.b['plan'].items() if k!='fingerprint'}))
    def test_plan_tamper(self):
        self.b['plan']['tasks'].reverse()
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate(self.b,self.a,self.c,self.e,True)
    def test_protocol_tamper(self):
        self.e['max_generations']=21
        with self.assertRaisesRegex(ValueError,'FINGERPRINT'):validate(self.b,self.a,self.c,self.e,True)
    def test_budget21_rejected(self):
        self.e['max_generations']=21;self.e['fingerprint']=digest({k:v for k,v in self.e.items() if k!='fingerprint'})
        with self.assertRaisesRegex(ValueError,'FIXED20'):validate(self.b,self.a,self.c,self.e,True)
    def test_dryrun_no_state_change(self):
        calls=[];v=run_collection(self.b,self.a,self.c,self.e,'UNUSED',test_only=True,transport_factory=lambda:calls.append(1));self.assertEqual(calls,[]);self.assertEqual(v['generation_requests'],0);self.assertEqual(v['unloads'],0)
    def test_approval_required(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.a,self.c,self.e,'UNUSED',execute=True,allow_llm=True,test_only=True)
    def test_old15_approval_rejected(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.a,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=ExtensionApproval('TEST','TEST'),test_only=True)
    def test_old10_approval_rejected(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.a,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=CompactApproval('TEST','TEST'),test_only=True)
    def test_allow_llm_required(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.a,self.c,self.e,'UNUSED',execute=True,approval=WeightApproval('TEST','TEST'),test_only=True)
    def test_subclass_authority_rejected(self):
        class Fake(WeightApproval):pass
        with self.assertRaisesRegex(ValueError,'APPROVAL'):run_collection(self.b,self.a,self.c,self.e,'UNUSED',execute=True,allow_llm=True,approval=Fake('TEST','TEST'),test_only=True)
    def test_budget20(self):
        r,h,p=self.fake();self.assertEqual(len(h.calls),20);self.assertEqual(r['generation_budget_consumed'],20);self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['actual_task_generation_requests'],0)
    def test_unresolved_continues_switches(self):
        r,h,p=self.fake();self.assertEqual(r['evaluation']['all_agents']['counts']['UNRESOLVED'],20);self.assertEqual(r['evaluation']['complete_collection_groups'],5)
    def test_three_switch_unloads(self):
        r,h,p=self.fake();self.assertEqual(h.unloads,[TAGS[0],TAGS[1],TAGS[0]]);self.assertEqual(r['unload_requests'],3)
    def test_initial_loaded_model_unloaded(self):
        r,h,p=self.fake(preload=TAGS[0]);self.assertEqual(len(h.unloads),4);self.assertEqual(r['unload_requests'],4)
    def test_foreign_model_not_unloaded(self):
        r,h,p=self.fake(preload='TEST_FOREIGN');self.assertEqual(len(h.calls),0);self.assertEqual(h.unloads,[]);self.assertEqual(r['stop_reason'],'OTHER_OR_MULTIPLE_MODELS_LOADED')
    def test_unload_error_stops_and_charges_uncertainty(self):
        r,h,p=self.fake('unload_error');self.assertEqual(len(h.calls),5);self.assertEqual(r['uncertain_or_unexpected_unload_budget_charges'],1);self.assertEqual(r['generation_budget_consumed'],6)
    def test_model_digest_change_blocks(self):
        r,h,p=self.fake('digest');self.assertEqual(len(h.calls),5);self.assertIn('DIGEST',r['stop_reason'])
    def test_native_template_change_blocks(self):
        r,h,p=self.fake('template');self.assertEqual(len(h.calls),5);self.assertIn('TEMPLATE',r['stop_reason'])
    def test_actual_context_mismatch_stops(self):
        r,h,p=self.fake('context');self.assertEqual(len(h.calls),1);self.assertIn('CONTEXT',r['stop_reason'])
    def test_context_failed_score_excluded_from_menu(self):
        r,h,p=self.fake('context');self.assertFalse(r['rows'][0]['execution_contract_valid']);self.assertEqual(selection_menus(self.b,r['rows'])['shared_complete_question_denominator'],0)
    def test_conservative_input_overflow_stops(self):
        r,h,p=self.fake('input_overflow');self.assertEqual(len(h.calls),1);self.assertEqual(r['stop_reason'],'SERVER_INPUT_EXCEEDS_CONSERVATIVE_BOUND')
    def test_output_overflow_stops(self):
        r,h,p=self.fake('output_overflow');self.assertEqual(len(h.calls),1);self.assertEqual(r['stop_reason'],'SERVER_OUTPUT_BUDGET')
    def test_wrong_model_reply_stops(self):
        r,h,p=self.fake('wrong_reply');self.assertEqual(len(h.calls),1);self.assertEqual(r['stop_reason'],'RESPONSE_MODEL_OR_DONE')
    def test_invalid_json_stops(self):
        r,h,p=self.fake('invalid');self.assertEqual(len(h.calls),1);self.assertEqual(r['evaluation']['all_agents']['counts']['INVALID_RESPONSE'],1)
    def test_extra_field_stops(self):
        r,h,p=self.fake('extra');self.assertEqual(len(h.calls),1)
    def test_truncation_no_repair(self):
        r,h,p=self.fake('length');self.assertEqual(len(h.calls),1);self.assertEqual(r['stop_reason'],'OUTPUT_LIMIT_TRUNCATION')
    def test_http_stops_and_preserves_entity(self):
        r,h,p=self.fake('http');self.assertEqual(len(h.calls),1);self.assertTrue((p/'Request001_response_body.bin').exists())
    def test_timeout_no_retry(self):
        r,h,p=self.fake('timeout');self.assertEqual(len(h.calls),1);self.assertEqual(r['generation_budget_consumed'],1)
    def test_common_options(self):
        r,h,p=self.fake();self.assertTrue(all(v['options']==self.c['options'] and v['keep_alive']==120 and v['format']==schema() for v in h.calls))
    def test_exact_model_binding(self):
        r,h,p=self.fake();self.assertEqual([x['model'] for x in h.calls],[TAGS[0]]*5+[TAGS[1]]*5+[TAGS[0]]*5+[TAGS[2]]*5)
    def test_no_reference_payload(self):
        r,h,p=self.fake();self.assertTrue(all('TEST_TARGET' not in json.dumps(x) for x in h.calls))
    def test_role_matched_payload(self):
        r,h,p=self.fake()
        for offset in (0,10):
            for i in range(5):self.assertEqual(h.calls[offset+i]['messages'],h.calls[offset+i+5]['messages'])
    def test_checkpoints_and_manifest(self):
        r,h,p=self.fake();self.assertTrue((p/'Request020_after.json').exists());self.assertTrue((p/'Unload003_after.json').exists());self.assertTrue(h.closed)
    def test_no_overwrite(self):
        r,h,p=self.fake();raw=(p/'collection.json').read_bytes()
        with patch.object(WeightApproval,'verify',return_value={'TEST_ONLY':True}):
            with self.assertRaises(FileExistsError):run_collection(self.b,self.a,self.c,self.e,p,execute=True,allow_llm=True,approval=WeightApproval('TEST','TEST'),test_only=True,counter=Counter(),transport_factory=lambda:h)
        self.assertEqual((p/'collection.json').read_bytes(),raw)
    def test_scorer_exact(self):
        r,h,p=self.fake('correct');self.assertEqual(r['evaluation']['all_agents']['counts']['CORRECT_ANSWER'],20)
    def test_partial_registered20(self):
        r,h,p=self.fake('invalid');self.assertEqual(r['evaluation']['all_agents']['registered'],20);self.assertEqual(r['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],19)
    def test_per_condition_denominator5(self):
        r,h,p=self.fake('invalid');self.assertTrue(all(v['registered']==5 for v in r['evaluation']['per_agent'].values()))
    def test_token_margin_budget(self):
        user,n,bound=token_budget(self.b['questions'][0],self.b['bindings'][0],self.c,Counter(800));self.assertEqual(bound,1824)
    def test_insufficient_context_no_generation(self):
        r,h,p=self.fake(counter=Counter(3000));self.assertEqual(len(h.calls),0)
    def test_menu_shared_denominator(self):
        rs=all_rows(self.b);rs[-1].update(state='NOT_ATTEMPTED',answer_correct=None);v=selection_menus(self.b,rs)
        self.assertEqual(v['shared_complete_question_denominator'],4);self.assertTrue(all(m['complete_paired_questions']==4 for m in v['menus'].values()))
    def test_menu_zero_denominator_null(self):
        rs=all_rows(self.b)
        for r in rs:r.update(state='NOT_ATTEMPTED',answer_correct=None)
        v=selection_menus(self.b,rs);self.assertTrue(all(m['oracle_gap'] is None for m in v['menus'].values()))
    def test_menu_unresolved_not_clinical_false(self):
        rs=all_rows(self.b);v=selection_menus(self.b,rs);self.assertTrue(all(r['answer_correct'] is None for r in rs));self.assertEqual(v['menus']['mixed']['oracle_gap'],0)
    def test_positive_menu_gap_same_cohort(self):
        rs=all_rows(self.b);rs[0].update(state='CORRECT_ANSWER',answer_correct=True);rs[5].update(state='CORRECT_ANSWER',answer_correct=True)
        v=selection_menus(self.b,rs);self.assertAlmostEqual(v['menus']['mixed']['oracle_gap'],.2);self.assertEqual(v['menus']['base_only']['oracle_gap'],0);self.assertEqual(v['menus']['distinct_weight']['oracle_gap'],0)
    def test_menu_size_random_expectation(self):
        rs=all_rows(self.b);rs[0].update(state='CORRECT_ANSWER',answer_correct=True);v=selection_menus(self.b,rs)
        self.assertEqual(v['menus']['base_only']['uniform_random_expected']['value'],.1);self.assertEqual(v['menus']['mixed']['uniform_random_expected']['value'],.05)
    def test_role_prompt_only_gap_not_weight_selection_gain(self):
        rs=all_rows(self.b)
        for i in (0,6,1,7):rs[i].update(state='CORRECT_ANSWER',answer_correct=True)
        ev=__import__('app.locdi.answer_grounded_routing_evaluation',fromlist=['evaluate']).evaluate(self.b['questions'],self.b['bindings'],rs)
        self.assertEqual(research_decision({'stop_reason':None},ev,selection_menus(self.b,rs)),'STOP_CURRENT_ROUTING_TRACK')
    def test_distinct_weights_add_correct_question_go(self):
        rs=all_rows(self.b)
        for i in (1,7):rs[i].update(state='CORRECT_ANSWER',answer_correct=True)
        ev=__import__('app.locdi.answer_grounded_routing_evaluation',fromlist=['evaluate']).evaluate(self.b['questions'],self.b['bindings'],rs)
        self.assertEqual(research_decision({'stop_reason':None},ev,selection_menus(self.b,rs)),'GO_DISTINCT_WEIGHT_AGENT_SELECTION_EXPLORATORY')
    def test_one_better_fixed_model_not_routing_gain(self):
        rs=all_rows(self.b);rs[1].update(state='CORRECT_ANSWER',answer_correct=True)
        ev=__import__('app.locdi.answer_grounded_routing_evaluation',fromlist=['evaluate']).evaluate(self.b['questions'],self.b['bindings'],rs)
        self.assertEqual(research_decision({'stop_reason':None},ev,selection_menus(self.b,rs)),'MODEL_IMPROVEMENT_WITHOUT_ROUTING_GAIN')
    def test_duplicate_rows_rejected(self):
        rs=all_rows(self.b)
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):selection_menus(self.b,rs+[rs[0]])
    def test_transitions_unresolved_to_correct(self):
        rs=all_rows(self.b);rs[1].update(state='CORRECT_ANSWER',answer_correct=True);v=role_transitions(self.b,rs);self.assertEqual(v['neurology']['rows'][0]['transition'],'UNRESOLVED -> CORRECT_ANSWER')
    def test_transitions_same_outcome(self):self.assertEqual(role_transitions(self.b,all_rows(self.b))['pulmonology']['rows'][0]['transition'],'SAME_OUTCOME')
    def test_transitions_invalid_not_imputed(self):
        rs=all_rows(self.b);rs[1].update(state='INVALID_RESPONSE',answer_correct=None);self.assertEqual(role_transitions(self.b,rs)['neurology']['paired_question_denominator'],4)
    def test_failed_execution_contract_pair_excluded(self):
        rs=all_rows(self.b);rs[1]['execution_contract_valid']=False;self.assertEqual(role_transitions(self.b,rs)['neurology']['paired_question_denominator'],4)
    def test_gguf_minimal_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.gguf';s=b'TEST_ONLY_KEY';value=b'TEST_ONLY_VALUE'
            p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,1)+struct.pack('<Q',len(s))+s+struct.pack('<IQ',8,len(value))+value)
            self.assertEqual(gguf_metadata(p)['values']['TEST_ONLY_KEY'],'TEST_ONLY_VALUE')
    def test_gguf_bad_magic(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.gguf';p.write_bytes(b'BAD!')
            with self.assertRaisesRegex(ValueError,'MAGIC'):gguf_metadata(p)
    def test_gguf_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.gguf';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,1))
            with self.assertRaisesRegex(ValueError,'TRUNCATED'):gguf_metadata(p)
    def test_gguf_oversized_key_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.gguf';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,100001))
            with self.assertRaisesRegex(ValueError,'KEY_LIMIT'):gguf_metadata(p)
    def test_deployment_blob_hash_verified(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);mp=p/'TEST_MANIFEST';bp=p/'TEST_BLOB';mp.write_bytes(b'TEST_ONLY_MANIFEST');bp.write_bytes(b'TEST_ONLY_BLOB')
            m={'manifest_path':str(mp),'manifest_digest':sha(mp),'blob_path':str(bp),'gguf_bytes':bp.stat().st_size,'gguf_sha256':sha(bp)}
            self.assertTrue(verify_local_artifact(m,True))
    def test_deployment_blob_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);mp=p/'TEST_MANIFEST';bp=p/'TEST_BLOB';mp.write_bytes(b'TEST_ONLY');bp.write_bytes(b'TEST_ONLY')
            m={'manifest_path':str(mp),'manifest_digest':sha(mp),'blob_path':str(bp),'gguf_bytes':bp.stat().st_size,'gguf_sha256':'0'*64}
            with self.assertRaisesRegex(ValueError,'MODEL_BLOB_CHANGED'):verify_local_artifact(m,True)
    def test_deployment_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);mp=p/'TEST_MANIFEST';bp=p/'TEST_BLOB';mp.write_bytes(b'TEST_ONLY');bp.write_bytes(b'TEST_ONLY')
            m={'manifest_path':str(mp),'manifest_digest':'0'*64,'blob_path':str(bp),'gguf_bytes':bp.stat().st_size,'gguf_sha256':sha(bp)}
            with self.assertRaisesRegex(ValueError,'MANIFEST_CHANGED'):verify_local_artifact(m,True)
    def test_old_core_and_compact_hashes(self):
        p=Path('outputs/private/research_evaluation/c4_27_v1/source_hashes.json');pins=json.loads(p.read_text(encoding='utf-8'))
        for n,h in pins.items():
            if n.replace('\\','/').endswith(('app/core/routing_utility.py','app/core/voi_planning.py')):self.assertEqual(sha(n),h)
    def test_approval_model_unload_and20_scope(self):
        now=datetime.now(timezone.utc);a={'scope':'LIVE_WEIGHT_COMPARISON_20_ONLY','authorization_source':'EXPLICIT_USER_STATEMENT','operator_reference':'USER_LOCAL_SERVER_OPERATOR',
            'protocol_fingerprint':self.e['fingerprint'],'plan_fingerprint':self.e['plan_fingerprint'],'parent_plan_fingerprint':self.e['parent_plan_fingerprint'],
            'source_sha256':self.e['source_sha256'],'task_order_fingerprint':self.e['task_order_fingerprint'],'model_bindings_fingerprint':digest(self.e['model_bindings']),
            'contract_fingerprint':digest(self.e['contract']),'task_version':VERSION,'prompt_version':PROMPT_VERSION,'schema_fingerprint':self.e['schema_fingerprint'],
            'endpoint':self.e['endpoint'],'namespace':NAMESPACE,'run_id':RUN_ID,'max_generations':20,'retry':0,'repair':0,'warmup':0,
            'allow_experiment_model_unloads':True,'max_unloads':4,'no_other_ollama_work_confirmed':True,'user_statement':'TEST_ONLY_NOT_REAL_APPROVAL',
            'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat()}
        with patch('app.locdi.weight_differentiated_agent.pinned_private_json',return_value=a):
            authority=WeightApproval('TEST','TEST');self.assertIn('approval_sha256',authority.verify(self.e,now));a['allow_experiment_model_unloads']=False
            with self.assertRaisesRegex(ValueError,'SCOPE'):authority.verify(self.e,now)
            a['allow_experiment_model_unloads']=True;a['expires_at']=(now-timedelta(seconds=1)).isoformat()
            with self.assertRaisesRegex(ValueError,'VALIDITY'):authority.verify(self.e,now)


if __name__=='__main__':unittest.main()
