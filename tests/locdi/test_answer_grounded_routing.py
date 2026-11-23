"""TEST_ONLY abstract symbols, never medical correctness or real model evidence."""
from copy import deepcopy
import json
from pathlib import Path
import ast
import unittest
from app.locdi.llm_setup import digest
from app.locdi.answer_grounded_routing import (task,model_input,score,split_audit,bind_agents,build_plan,
    DEPARTMENTS,_pair,read_medqa,validate_pair)
from app.locdi.answer_grounded_routing_evaluation import metrics,evaluate,collect_test_only,support_audit
import tempfile


def pair(i=1):
    return task({'id':'TEST_%d'%i,'question':'TEST_ONLY problem %d: choose the designated symbol.'%i,
        'options':{'A':'TEST_ALPHA','B':'TEST_BETA','C':'TEST_GAMMA','D':'TEST_DELTA'},'answer_idx':'B'},
        dataset='MedQA',version='TEST_ONLY',ordinal=i,original_split='train',exposure='TEST_ONLY')


def response(answer='B',status='supported',diagnosis=None):
    return {'status':status,'answer_id':answer,'diagnosis':diagnosis,'rationale':'TEST_ONLY explanation'}


def config_registry():
    agents=[{'agent_id':'TEST_ER','role':'er','department':'','model':'TEST:MODEL','version':'TEST_V','keywords':[]}]
    agents += [{'agent_id':'TEST_'+d,'role':'specialty','department':d,'model':'TEST:MODEL','version':'TEST_V','keywords':[]} for d in sorted(DEPARTMENTS)]
    reg={'schema_version':1,'agents':agents}
    config={'model':'TEST:MODEL','model_version':'TEST_V','model_identifier':'sha256:'+'a'*64,
        'agent_bindings':[{'agent_id':a['agent_id'],'model_identifier':'sha256:'+'a'*64} for a in agents]}
    return reg,config


def bundle():
    pairs=[pair(i) for i in range(1,6)];qs=[q for q,r in pairs];refs=[r for q,r in pairs]
    reg,cfg=config_registry();agents=bind_agents(reg,cfg,lambda d:'TEST_ONLY '+d)
    settings={'temperature':0,'seed':42,'context':32768,'max_output_tokens':1024,'timeout_seconds':120,'retry':0,'repair':0}
    plan=build_plan(qs,refs,agents,settings,run_id='TEST_ONLY_RUN')
    return qs,refs,agents,plan


class Fake:
    TEST_ONLY=True
    def __init__(self,mode='valid'):self.mode=mode;self.calls=[]
    def generate(self,system,public,schema):
        self.calls.append(deepcopy(public))
        if self.mode=='timeout':raise TimeoutError('TEST_ONLY')
        if self.mode=='invalid':return '{broken'
        if self.mode=='unresolved':return response(None,'unresolved')
        return response()


class AnswerGroundedTests(unittest.TestCase):
    def test_explicit_reference_required(self):
        q,r=pair();r['answer']=None
        with self.assertRaisesRegex(ValueError,'REFERENCE_REQUIRED'):validate_pair(q,r)

    def test_option_answer_must_exist(self):
        q,r=pair();r['answer']='Z'
        with self.assertRaisesRegex(ValueError,'OPTION_ID'):validate_pair(q,r)

    def test_correct_option(self):
        q,r=pair();s=score(q,r,response());self.assertEqual(s['state'],'CORRECT_ANSWER');self.assertIs(s['answer_correct'],True)

    def test_incorrect_option(self):
        q,r=pair();s=score(q,r,response('A'));self.assertEqual(s['state'],'INCORRECT_ANSWER');self.assertIs(s['answer_correct'],False)

    def test_unresolved_not_false(self):
        q,r=pair();s=score(q,r,response(None,'unresolved'));self.assertEqual(s['state'],'UNRESOLVED');self.assertIsNone(s['answer_correct'])

    def test_unresolved_nonnull_invalid(self):
        q,r=pair();self.assertEqual(score(q,r,response('B','unresolved'))['state'],'INVALID_RESPONSE')

    def test_missing_supported_answer_invalid(self):
        q,r=pair();self.assertEqual(score(q,r,response(None))['state'],'INVALID_RESPONSE')

    def test_schema_extra_field_invalid(self):
        q,r=pair();v=response();v['extra']='TEST';self.assertEqual(score(q,r,v)['state'],'INVALID_RESPONSE')

    def test_malformed_json_invalid(self):
        q,r=pair();self.assertEqual(score(q,r,'{x')['state'],'INVALID_RESPONSE')

    def test_duplicate_json_key_invalid(self):
        q,r=pair();self.assertEqual(score(q,r,'{"status":"supported","status":"unresolved"}')['state'],'INVALID_RESPONSE')

    def test_generation_failure(self):
        q,r=pair();s=score(q,r,error_type='TimeoutError');self.assertEqual(s['state'],'GENERATION_FAILED');self.assertIsNone(s['answer_correct'])

    def test_not_attempted(self):
        q,r=pair();self.assertEqual(score(q,r,attempted=False)['state'],'NOT_ATTEMPTED')

    def test_not_attempted_cannot_have_response(self):
        q,r=pair()
        with self.assertRaisesRegex(ValueError,'NOT_ATTEMPTED'):score(q,r,response(),attempted=False)

    def test_option_order_preserved(self):
        q,r=pair();self.assertEqual([o['id'] for o in model_input(q)['options']],list('ABCD'))

    def test_reference_source_hash_mismatch(self):
        q,r=pair();r['source_fingerprint']='b'*64
        with self.assertRaisesRegex(ValueError,'BINDING'):validate_pair(q,r)

    def test_input_fingerprint(self):
        q,r=pair();q['input']['question']='CHANGED'
        with self.assertRaisesRegex(ValueError,'BINDING'):model_input(q)

    def test_model_projection_excludes_reference(self):
        q,r=pair();q['reference_private']=r
        self.assertEqual(set(model_input(q)),{'question','options'});self.assertNotIn('answer',model_input(q))

    def test_gold_key_input_blocked(self):
        q,r=pair();q['input']['gold']='B';q['input_fingerprint']=digest(q['input'])
        with self.assertRaisesRegex(ValueError,'BINDING'):model_input(q)

    def test_embedded_answer_input_blocked(self):
        q,r=pair();q['input']['question']='TEST_ONLY answer: B';q['input_fingerprint']=digest(q['input'])
        with self.assertRaises(ValueError):model_input(q)

    def test_duplicate_id(self):
        q,r=pair();self.assertEqual(split_audit([q,q])['status'],'BLOCKED')

    def test_duplicate_text_different_id(self):
        q,r=pair();b=deepcopy(q);b['question_id']='TEST_OTHER'
        self.assertTrue(any(i['code']=='DUPLICATE_QUESTION' for i in split_audit([q,b])['issues']))

    def test_same_case_cross_split(self):
        q,r=pair();b,_=pair(2);b['case_group']=q['case_group'];b['role']='validation'
        self.assertTrue(any(i['cross_split'] for i in split_audit([q,b])['issues']))

    def test_train_exposure_preserved(self):
        q,r=pair();out=split_audit([q]);self.assertEqual(out['rows'][0]['exposure'],'TEST_ONLY')
        self.assertFalse(out['independent_test_ready']);self.assertIn('UNKNOWN',out['rows'][0]['training_overlap'])

    def test_development_not_independent_test(self):
        q,r=pair();q['role']='test';self.assertEqual(split_audit([q])['status'],'BLOCKED')

    def test_known_training_overlap_blocks_validation(self):
        q,r=pair();q['role']='validation'
        self.assertEqual(split_audit([q],training_inputs=[digest(q['input'])])['status'],'BLOCKED')

    def test_registry_shared_model_not_distinct_training(self):
        reg,cfg=config_registry();bs=bind_agents(reg,cfg,lambda d:'TEST_'+d)
        self.assertEqual(len(bs),5);self.assertEqual(len({b['model_identifier'] for b in bs}),1)
        self.assertEqual(len({b['benchmark_prompt_fingerprint'] for b in bs}),5)

    def test_registry_digest_binding(self):
        reg,cfg=config_registry();cfg['agent_bindings'][1]['model_identifier']='b'*64
        with self.assertRaisesRegex(ValueError,'BINDING'):bind_agents(reg,cfg,lambda d:d)

    def test_default_dry_run_constructs_no_client(self):
        qs,rs,bs,p=bundle();calls=[]
        out=collect_test_only(p,qs,rs,bs,lambda:calls.append(True))
        self.assertEqual(calls,[]);self.assertEqual(out['generation_attempts'],0)

    def test_live_path_blocked_without_authority(self):
        qs,rs,bs,p=bundle();calls=[]
        with self.assertRaisesRegex(ValueError,'LIVE_EXECUTION_DISABLED'):collect_test_only(p,qs,rs,bs,lambda:calls.append(True),execute=True)
        self.assertEqual(calls,[])

    def test_same_public_input_all_five_agents(self):
        qs,rs,bs,p=bundle();f=Fake();out=collect_test_only(p,qs,rs,bs,lambda:f,execute=True,test_only=True)
        self.assertEqual(len(f.calls),25)
        for i in range(0,25,5):self.assertTrue(all(x==f.calls[i] for x in f.calls[i:i+5]))
        self.assertEqual(out['evaluation']['complete_scorable_groups'],5)

    def test_budget_limit_retains_unattempted(self):
        qs,rs,bs,p=bundle();out=collect_test_only(p,qs,rs,bs,Fake,execute=True,test_only=True,max_requests=3)
        self.assertEqual(out['generation_attempts'],3)
        self.assertEqual(out['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],22)

    def test_budget_above_25_refused(self):
        qs,rs,bs,p=bundle()
        with self.assertRaisesRegex(ValueError,'BUDGET'):collect_test_only(p,qs,rs,bs,Fake,execute=True,test_only=True,max_requests=26)

    def test_invalid_stops_not_transport_failure(self):
        qs,rs,bs,p=bundle();out=collect_test_only(p,qs,rs,bs,lambda:Fake('invalid'),execute=True,test_only=True)
        self.assertEqual(out['generation_attempts'],1)
        self.assertEqual(out['rows'][0]['state'],'INVALID_RESPONSE')
        self.assertEqual(out['evaluation']['all_agents']['counts']['NOT_ATTEMPTED'],24)

    def test_timeout_first_error_stops(self):
        qs,rs,bs,p=bundle();out=collect_test_only(p,qs,rs,bs,lambda:Fake('timeout'),execute=True,test_only=True)
        self.assertEqual(out['rows'][0]['state'],'GENERATION_FAILED');self.assertEqual(out['generation_attempts'],1)

    def test_unresolved_collects_complete_not_scorable(self):
        qs,rs,bs,p=bundle();out=collect_test_only(p,qs,rs,bs,lambda:Fake('unresolved'),execute=True,test_only=True)
        self.assertEqual(out['evaluation']['complete_collection_groups'],5)
        self.assertEqual(out['evaluation']['complete_scorable_groups'],0)

    def test_zero_denominator(self):
        self.assertIsNone(metrics([])['selective_answer_accuracy']['value'])

    def test_accuracy_coverage_denominators(self):
        rs=[{'state':s} for s in ('CORRECT_ANSWER','INCORRECT_ANSWER','UNRESOLVED','INVALID_RESPONSE')]
        m=metrics(rs);self.assertEqual(m['answer_coverage']['value'],.5)
        self.assertEqual(m['selective_answer_accuracy']['value'],.5)
        self.assertEqual(m['correct_supported_answer_rate']['value'],.25)

    def test_oracle_evaluation_only(self):
        qs,rs,bs,p=bundle();out=collect_test_only(p,qs,rs,bs,Fake,execute=True,test_only=True)
        e=out['evaluation'];self.assertIn('Posthoc',e['evaluation_only_oracle']['scope'])
        self.assertFalse(e['utility_fit_ready']);self.assertEqual(e['evaluation_only_oracle']['success_upper_bound']['value'],1)

    def test_missing_group_kept(self):
        qs,rs,bs,p=bundle();e=evaluate(qs,bs,[])
        self.assertEqual(e['all_agents']['registered'],25);self.assertEqual(e['complete_scorable_groups'],0)
        self.assertIsNone(e['evaluation_only_oracle']['success_upper_bound']['value'])

    def test_duplicate_response_group_refused(self):
        qs,rs,bs,p=bundle();r={**score(qs[0],rs[0],response()),'agent_id':bs[0]['agent_id']}
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):evaluate(qs,bs,[r,r])

    def test_unresolved_bool_cannot_enter_metrics(self):
        qs,rs,bs,p=bundle();r={'question_id':qs[0]['question_id'],'agent_id':bs[0]['agent_id'],'state':'UNRESOLVED','answer_correct':False}
        with self.assertRaisesRegex(ValueError,'MEANING'):evaluate(qs,bs,[r])

    def test_free_text_lexical_only(self):
        q,r=_pair({'question':'TEST_ONLY static symbol problem','options':[]},{'TEST_SOURCE':'X'},'TEST','V',1,'UNKNOWN','TEST_ONLY','exact_text','TEST_TARGET',None)
        self.assertEqual(score(q,r,response(None,diagnosis='test_target.'))['state'],'CORRECT_ANSWER')
        self.assertEqual(score(q,r,response(None,diagnosis='TEST_POSSIBLE_SYNONYM'))['state'],'INCORRECT_ANSWER')

    def test_reader_parses_original_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'TEST_ONLY.jsonl';p.write_text(json.dumps({'question':'TEST_ONLY choose symbol','options':dict(zip('ABCD',['TEST_A','TEST_B','TEST_C','TEST_D'])),'answer_idx':'A'})+'\n',encoding='utf-8')
            pairs=read_medqa(p,version='TEST_ONLY',split='train',exposure='TEST_ONLY');self.assertEqual(len(pairs),1)

    def test_similarity_support_not_fitting(self):
        qs,rs,bs,p=bundle();out=support_audit(qs)
        self.assertTrue(all(r['other_questions_with_positive_jaccard']==4 for r in out['rows']));self.assertEqual(out['fit_calls'],0)

    def test_plan_references_changed_blocked_before_client(self):
        qs,rs,bs,p=bundle();rs[0]['answer']='C';calls=[]
        with self.assertRaisesRegex(ValueError,'BINDING'):collect_test_only(p,qs,rs,bs,lambda:calls.append(True),execute=True,test_only=True)
        self.assertEqual(calls,[])

    def test_no_fitting_or_clinical_gate_injection(self):
        root=Path(__file__).resolve().parents[2]
        for name in ('answer_grounded_routing.py','answer_grounded_routing_evaluation.py'):
            tree=ast.parse((root/'app/locdi'/name).read_text(encoding='utf-8'))
            calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            self.assertNotIn('fit_utility',calls);self.assertNotIn('to_policy_fit_rows',calls)

    def test_original_question_id_stable_across_ordinal(self):
        row={'id':'TEST_SOURCE_ID','question':'TEST_ONLY problem','options':dict(zip('ABCD',['TEST_A','TEST_B','TEST_C','TEST_D'])),'answer_idx':'A'}
        a,_=task(row,dataset='MedQA',version='TEST_ONLY',ordinal=1,original_split='train',exposure='TEST_ONLY')
        b,_=task(row,dataset='MedQA',version='TEST_ONLY',ordinal=99,original_split='train',exposure='TEST_ONLY')
        self.assertEqual(a['question_id'],b['question_id'])

    def test_source_id_duplicate_even_when_text_differs(self):
        a,_=pair(1);b,_=pair(2)
        b['question_id']=a['question_id']
        self.assertTrue(any(i['code']=='DUPLICATE_ID' for i in split_audit([a,b])['issues']))

    def test_invalid_original_question_id_refused(self):
        row={'id':7,'question':'TEST_ONLY problem','options':dict(zip('ABCD',['TEST_A','TEST_B','TEST_C','TEST_D'])),'answer_idx':'A'}
        with self.assertRaisesRegex(ValueError,'SOURCE_QUESTION_ID'):
            task(row,dataset='MedQA',version='TEST_ONLY',ordinal=1,original_split='train',exposure='TEST_ONLY')


if __name__=='__main__':unittest.main()
