"""TEST_ONLY symbols/bytes; no external cases, model requests or clinical verdicts."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha
from app.locdi.reference_acquisition import (acquire,fingerprint,validate_plan,PinnedApproval,NoRedirect)
from app.locdi.external_reference_task import (adapt_exam,index_contract,model_projection,public_projection,
    validate_task,require_use,inspect_case_candidates,qa_metrics,execute_model)
from app.locdi.external_reference_readiness import evaluate_gate


def rows():
    return [{'id':'TEST_ONLY_PRIVATE_ID','question':'TEST_ONLY symbolic question?',
      'options':{'A':'SYMBOL_ALPHA','B':'SYMBOL_BETA','C':'SYMBOL_GAMMA','D':'SYMBOL_DELTA'},
      'answer_idx':'B','explanation':'TEST_ONLY_PRIVATE_EXPLANATION'}]


def adapted(data=None,dataset='MedQA',encoding='medqa_letter',split='validation'):
    data=rows() if data is None else data
    contract=index_contract(data,dataset,'a'*40,split,encoding)
    return adapt_exam(data,contract,verified_contract_fingerprint=digest(contract))


def mcqa(cop=1):
    return [{'id':'TEST_ONLY_ID','question':'TEST_ONLY question?', 'opa':'SYM_A','opb':'SYM_B','opc':'SYM_C','opd':'SYM_D',
      'cop':cop,'choice_type':'single','exp':'TEST_ONLY_PRIVATE_EXPLANATION','subject_name':'PRIVATE_SUBJECT','topic_name':'PRIVATE_TOPIC'}]


class Response(io.BytesIO):
    def __init__(self,data):super().__init__(data);self.headers={'Content-Length':str(len(data)),'Content-Type':'application/jsonl'}


class ExternalReferenceTests(unittest.TestCase):
    def test_medqa_letter_and_model_projection(self):
        tasks,keys=adapted();p=model_projection(tasks[0]);d=json.loads(p[0]['content'])
        self.assertEqual(set(d),{'question','options'});self.assertEqual(keys['AnswerKey000001']['answer_label'],'B')
        self.assertNotIn('TEST_ONLY_PRIVATE',json.dumps(p));self.assertNotIn('answer_idx',json.dumps(p))

    def test_question_answer_injection_blocked(self):
        r=rows();r[0]['question']='TEST_ONLY answer_idx=B'
        with self.assertRaises(ValueError):adapted(r)

    def test_quoted_answer_marker_blocked(self):
        r=rows();r[0]['question']='TEST_ONLY {"answer_idx":"B"}'
        with self.assertRaises(ValueError):adapted(r)

    def test_medqa_dev_split_preserved(self):
        tasks,_=adapted(split='dev');self.assertEqual(tasks[0]['original_split'],'dev')

    def test_cli_download_switches_remain_phase_blocked(self):
        from scripts.prepare_c3_06 import main
        with patch('builtins.print') as out,patch('app.locdi.reference_acquisition.https_transport') as transport:
            self.assertEqual(main(['--execute','--allow-download']),2)
            transport.assert_not_called();self.assertIn('C3_06_REAL_DOWNLOAD_FORBIDDEN',str(out.call_args))

    def test_current_plans_do_not_invent_bytes_or_approve_rights(self):
        from app.locdi.public_reference_preparation import acquisition_plans
        for plan in acquisition_plans():
            self.assertEqual(validate_plan(plan)['status'],'ACQUISITION_BLOCKED')
            self.assertIsNone(plan['maximum_bytes']);self.assertIsNone(plan['accepted_license_use_policy_evidence'])

    def test_question_explanation_injection_blocked(self):
        r=rows();r[0]['question']+=' '+r[0]['explanation']
        with self.assertRaises(ValueError):adapted(r)

    def test_options_reordered_after_index_pin_blocked(self):
        r=rows();c=index_contract(r,'MedQA','a'*40,'validation','medqa_letter');pin=digest(c)
        r[0]['options']['A'],r[0]['options']['B']=r[0]['options']['B'],r[0]['options']['A']
        with self.assertRaises(ValueError):adapt_exam(r,c,verified_contract_fingerprint=pin)

    def test_noncanonical_option_order_blocked(self):
        r=rows();r[0]['options']=dict(reversed(list(r[0]['options'].items())))
        with self.assertRaises(ValueError):adapted(r)

    def test_answer_key_absent(self):
        r=rows();del r[0]['answer_idx']
        with self.assertRaises(ValueError):adapted(r)

    def test_medmcqa_test_with_or_without_placeholder_blocked(self):
        for val in (None,0,1):
            with self.assertRaisesRegex(ValueError,'WITHHELD'):adapted(mcqa(val),'MedMCQA','medmcqa_hf_zero_based','test')

    def test_medmcqa_explicit_index_conventions(self):
        _,one=adapted(mcqa(1),'MedMCQA','medmcqa_csv_one_based')
        _,zero=adapted(mcqa(0),'MedMCQA','medmcqa_hf_zero_based')
        self.assertEqual(one['AnswerKey000001']['answer_label'],zero['AnswerKey000001']['answer_label'])

    def test_medmcqa_zero_is_not_one_based(self):
        with self.assertRaises(ValueError):adapted(mcqa(0),'MedMCQA','medmcqa_csv_one_based')

    def test_medmcqa_four_is_not_zero_based(self):
        with self.assertRaises(ValueError):adapted(mcqa(4),'MedMCQA','medmcqa_hf_zero_based')

    def test_medmcqa_bool_is_not_integer_label(self):
        with self.assertRaises(ValueError):adapted(mcqa(True),'MedMCQA','medmcqa_csv_one_based')

    def test_medmcqa_unknown_index_not_guessed(self):
        with self.assertRaises(ValueError):adapted(mcqa(),'MedMCQA','auto')

    def test_choice_count_and_missing_option(self):
        r=rows();r[0]['options'].pop('D')
        with self.assertRaises(ValueError):adapted(r)
        r=mcqa();r[0]['opd']=''
        with self.assertRaises(ValueError):adapted(r,'MedMCQA','medmcqa_csv_one_based')

    def test_multiple_choice_type_requires_new_policy(self):
        r=mcqa();r[0]['choice_type']='multi'
        with self.assertRaises(ValueError):adapted(r,'MedMCQA','medmcqa_csv_one_based')

    def test_explanation_subject_topic_not_in_model_input(self):
        tasks,_=adapted(mcqa(),'MedMCQA','medmcqa_csv_one_based')
        s=json.dumps(model_projection(tasks[0]))
        self.assertTrue(all(v not in s for v in ('PRIVATE','topic','subject','exp','cop')))

    def test_input_gold_extra_key_blocked_even_rehashed(self):
        tasks,_=adapted();t=tasks[0];p=json.loads(t['input_messages'][0]['content']);p['answer_idx']='B'
        t['input_messages'][0]['content']=json.dumps(p);t['prompt_fingerprint']=digest(t['input_messages'])
        with self.assertRaises(ValueError):model_projection(t)

    def test_rubric_canary_injection_blocked(self):
        for token in ('rubric','canary','gold'):
            r=rows();r[0]['question']='TEST_ONLY '+token+'=PRIVATE'
            with self.assertRaises(ValueError):adapted(r)

    def test_source_fingerprint_tamper(self):
        r=rows();c=index_contract(r,'MedQA','a'*40,'validation','medqa_letter');c['rows_fingerprint']='0'*64
        with self.assertRaises(ValueError):adapt_exam(r,c,verified_contract_fingerprint=digest(c))

    def test_prompt_fingerprint_tamper(self):
        tasks,_=adapted();tasks[0]['prompt_fingerprint']='0'*64
        with self.assertRaises(ValueError):model_projection(tasks[0])

    def test_duplicate_question_or_id(self):
        for change in ('id','question'):
            r=rows();other=deepcopy(r[0]);other['id']='TEST_ONLY_2';other['question']='TEST_ONLY other question?'
            other[change]=r[0][change]
            with self.assertRaises(ValueError):adapted(r+[other])

    def test_medcase_shared_pmcid_requires_grouped_review(self):
        r=[{'id':'ONE','case_prompt':'TEST_ONLY symbolic context','pmcid':'TEST_ONLY_PMC','final_diagnosis':'SYMBOL_KEY'},
           {'id':'TWO','case_prompt':'TEST_ONLY another context','pmcid':'TEST_ONLY_PMC','final_diagnosis':'SYMBOL_KEY'}]
        with self.assertRaisesRegex(ValueError,'SHARED_PMCID'):inspect_case_candidates(r)

    def test_medcase_answer_in_prompt_is_review_trigger(self):
        with self.assertRaises(ValueError):inspect_case_candidates([{'case_prompt':'TEST_ONLY SYMBOL_KEY','pmcid':'TEST_ONLY_PMC','final_diagnosis':'SYMBOL_KEY'}])

    def test_public_projection_has_no_item_id_source_path_answer(self):
        tasks,_=adapted();p=public_projection(tasks);s=json.dumps(p)
        self.assertNotIn('task_id',s);self.assertNotIn('TEST_ONLY',s);self.assertNotIn('answer_key_reference',s)

    def test_test_item_cannot_become_training(self):
        task=adapted(split='test')[0][0]
        with self.assertRaises(ValueError):require_use(task,purpose='training')

    def test_development_exposure_cannot_become_independent_test(self):
        task=adapted(split='train')[0][0]
        with self.assertRaises(ValueError):require_use(task,purpose='external_evaluation',independent_test=True)

    def test_no_clinical_gold_fit_rows_or_voi_conversion(self):
        task=adapted()[0][0]
        for purpose in ('clinical_gold','routing_fit_rows','voi_likelihood'):
            with self.assertRaises(ValueError):require_use(task,purpose=purpose)

    def test_no_answer_key_selective_accuracy_not_evaluated(self):
        tasks,_=adapted();m=qa_metrics(tasks,['B'])
        self.assertEqual(m['selective_accuracy']['status'],'NOT_EVALUATED')

    def test_empty_answer_key_is_missing_not_wrong_answer(self):
        tasks,_=adapted();m=qa_metrics(tasks,['B'],{'AnswerKey000001':{}})
        self.assertEqual(m['selective_accuracy']['status'],'NOT_EVALUATED')
        self.assertEqual(m['overall_answer_accuracy']['status'],'NOT_EVALUATED')

    def test_duplicate_json_keys_rejected(self):
        from app.locdi.reference_acquisition import strict_json
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):strict_json('{"cop":0,"cop":1}')

    def test_nonfinite_json_rejected(self):
        from app.locdi.reference_acquisition import strict_json
        for value in ('NaN','Infinity','-Infinity'):
            with self.assertRaisesRegex(ValueError,'NONFINITE_JSON_VALUE'):strict_json('{"cop":'+value+'}')

    def test_duplicate_model_json_fields_rejected(self):
        tasks,_=adapted();task=tasks[0]
        payload=task['input_messages'][0]['content']
        task['input_messages'][0]['content']=payload[:-1]+',"question":"TEST_ONLY other?"}'
        task['prompt_fingerprint']=digest(task['input_messages'])
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):model_projection(task)

    def test_coverage_abstention_invalid_and_selective_denominators(self):
        r=rows()
        for i in range(2):q=deepcopy(r[0]);q['id']='TEST_ONLY_%d'%i;q['question']='TEST_ONLY q%d?'%i;r.append(q)
        tasks,keys=adapted(r);m=qa_metrics(tasks,['B','ABSTAIN','bad'],keys)
        self.assertEqual(m['coverage']['value'],1/3);self.assertEqual(m['selective_accuracy']['value'],1)
        self.assertEqual(m['overall_answer_accuracy']['value'],1/3)

    def test_zero_denominator_not_zero_accuracy(self):
        self.assertIsNone(qa_metrics([],[],{})['coverage']['value'])
        tasks,keys=adapted();self.assertEqual(qa_metrics(tasks,['ABSTAIN'],keys)['selective_accuracy']['status'],'NOT_EVALUATED')

    def test_wrong_answer_key_source_rejected(self):
        tasks,keys=adapted();keys['AnswerKey000001']['source_fingerprint']='0'*64
        with self.assertRaises(ValueError):qa_metrics(tasks,['ABSTAIN'],keys)

    def test_model_and_grader_calls_forbidden(self):
        client=Mock()
        with self.assertRaises(RuntimeError):execute_model(client)
        client.assert_not_called()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private'))
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.data=('\n'.join(json.dumps(r) for r in rows())+'\n').encode('utf8')
        self.plan={'schema_version':'locdi.reference_acquisition_plan.v1','dataset_id':'MedQA',
          'canonical_url':'https://huggingface.co','pinned_revision':'a'*40,'expected_split':'validation',
          'destination_private_namespace':'TEST_ONLY','maximum_bytes':len(self.data),'network_request_budget':1,'timeout_seconds':2,
          'download_attempt_limit':1,'approval_reference':'TEST_ONLY_APPROVAL',
          'expected_files':[{'name':'data.jsonl','url':'https://huggingface.co/TEST_ONLY/'+('a'*40)+'/data.jsonl','expected_bytes':len(self.data),'sha256':hashlib.sha256(self.data).hexdigest()}]}
        evidence=self.root/'license.txt';evidence.write_text('TEST_ONLY_NOT_REAL_LICENSE_APPROVAL',encoding='utf8')
        self.plan['accepted_license_use_policy_evidence']={'reference':evidence.relative_to(ROOT).as_posix(),'sha256':sha(evidence)}
        self.plan['plan_fingerprint']=fingerprint(self.plan)
        approval={'schema_version':'locdi.acquisition_approval.v1','approval_reference':'TEST_ONLY_APPROVAL',
          'plan_fingerprint':fingerprint(self.plan),'license_evidence':self.plan['accepted_license_use_policy_evidence'],
          'scope':'PRIVATE_RAW_ACQUISITION_ONLY','approved_by':'TEST_ONLY_OPERATOR','issued_at':'2000-01-01T00:00:00+00:00','expires_at':'2099-01-01T00:00:00+00:00'}
        self.approval=self.root/'approval.json';self.approval.write_text(json.dumps(approval),encoding='utf8')
        self.authority=PinnedApproval(self.approval,sha(self.approval))
        self.transport=Mock(side_effect=lambda url,timeout:Response(self.data))
        self.patcher=patch('app.locdi.reference_acquisition.PRIVATE_ROOT',self.root/'raw');self.patcher.start();self.addCleanup(self.patcher.stop)

    def dispatch(self,**kw):return acquire(self.plan,execute=True,allow_download=True,authority=self.authority,transport=self.transport,**kw)

    def test_dry_run_no_network_or_write(self):
        self.assertEqual(acquire(self.plan,transport=self.transport)['mode'],'DRY_RUN');self.transport.assert_not_called()
        self.assertFalse((self.root/'raw').exists())

    def test_no_explicit_approval_dispatches_nothing(self):
        with self.assertRaises(ValueError):acquire(self.plan,execute=True,transport=self.transport)
        self.transport.assert_not_called()

    def test_license_unconfirmed_blocks_before_transport(self):
        self.plan['accepted_license_use_policy_evidence']=None;self.plan['plan_fingerprint']=fingerprint(self.plan)
        with self.assertRaises(ValueError):self.dispatch()
        self.transport.assert_not_called()

    def test_pinned_revision_mismatch_blocks(self):
        self.plan['pinned_revision']='b'*40;self.plan['plan_fingerprint']=fingerprint(self.plan)
        with self.assertRaises(ValueError):self.dispatch()
        self.transport.assert_not_called()

    def test_success_receipt_hash_size_budget_and_no_clobber(self):
        receipt=self.dispatch();self.assertTrue(receipt['download_complete']);self.assertEqual(receipt['request_count'],1)
        before=deepcopy(receipt)
        with self.assertRaises(FileExistsError):self.dispatch()
        self.assertEqual(receipt,before);self.assertEqual(self.transport.call_count,1)

    def test_unknown_size_blocks(self):
        self.plan['expected_files'][0]['expected_bytes']=None;self.plan['plan_fingerprint']=fingerprint(self.plan)
        self.assertIn('EXPECTED_FILE_SIZE_REQUIRED',validate_plan(self.plan)['blocker_codes'])

    def test_max_bytes_plan_overflow(self):
        self.plan['maximum_bytes']=1;self.plan['plan_fingerprint']=fingerprint(self.plan)
        with self.assertRaises(ValueError):self.dispatch()
        self.transport.assert_not_called()

    def test_stream_overflow_preserves_failure_not_raw_success(self):
        def oversized(url,timeout):r=Response(self.data+b'X');r.headers.pop('Content-Length');return r
        self.transport.side_effect=oversized
        with self.assertRaises(ValueError):self.dispatch()
        dest=self.root/'raw/MedQA'/('a'*40)
        self.assertFalse((dest/'data.jsonl').exists());self.assertEqual(json.loads((dest/'receipt.json').read_text())['status'],'FAILED')

    def test_no_retry_after_transport_error(self):
        self.transport.side_effect=TimeoutError('TEST_ONLY_PRIVATE_ERROR')
        with self.assertRaises(ValueError):self.dispatch()
        self.assertEqual(self.transport.call_count,1)

    def test_request_budget_blocks_multiple_files(self):
        self.plan['expected_files'].append({**self.plan['expected_files'][0],'name':'second.jsonl'})
        self.plan['maximum_bytes']*=2;self.plan['plan_fingerprint']=fingerprint(self.plan)
        self.assertIn('REQUEST_BUDGET_EXCEEDED',validate_plan(self.plan)['blocker_codes'])

    def test_redirect_not_followed(self):
        with self.assertRaises(ValueError):NoRedirect().redirect_request(None,None,302,'redirect',{},'https://other.test')

    def test_path_escape_blocked(self):
        self.plan['expected_files'][0]['name']='../outside';self.plan['plan_fingerprint']=fingerprint(self.plan)
        with self.assertRaises(ValueError):self.dispatch()
        self.transport.assert_not_called()

    def test_approval_hash_not_accepted_status(self):
        self.approval.write_text('{"status":"APPROVED"}',encoding='utf8')
        with self.assertRaises(ValueError):self.dispatch()
        self.transport.assert_not_called()

    def test_status_only_json_does_not_unlock_gate(self):
        p=self.root/'fake.json';p.write_text('{"status":"RAW_ACQUIRED","download_complete":true}',encoding='utf8')
        gate=evaluate_gate(self.plan,raw_evidence={'receipt_path':str(p),'receipt_sha256':sha(p)})
        self.assertFalse(gate['states']['RAW_ACQUIRED']);self.assertFalse(gate['states']['EXTERNAL_EVALUATION_READY'])

    def test_full_fake_receipt_adapter_gate_roundtrip_never_execution_ready(self):
        self.dispatch();dest=self.root/'raw/MedQA'/('a'*40)
        c=index_contract(rows(),'MedQA','a'*40,'validation','medqa_letter')
        p=self.root/'index.json';p.write_text(json.dumps(c),encoding='utf8')
        note=self.root/'metadata.json';note.write_text(json.dumps({'dataset_id':'MedQA','observations':['TEST_ONLY'],
          'official_urls':['https://example.test'],'current_documentary_review':'CONFIRMED'}),encoding='utf8')
        gate=evaluate_gate(self.plan,metadata={'path':str(note),'sha256':sha(note)},
          raw_evidence={'receipt_path':str(dest/'receipt.json'),'receipt_sha256':sha(dest/'receipt.json'),
            'approval_path':str(self.approval),'approval_sha256':sha(self.approval)},
          index_evidence={'path':str(p),'sha256':sha(p),'fingerprint':digest(c)})
        self.assertTrue(gate['states']['EVALUATION_CONTRACT_READY']);self.assertFalse(gate['states']['EXTERNAL_EVALUATION_READY'])
        (dest/'data.jsonl').write_bytes(b'TEST_ONLY_TAMPER')
        broken=evaluate_gate(self.plan,raw_evidence={'receipt_path':str(dest/'receipt.json'),'receipt_sha256':sha(dest/'receipt.json'),
          'approval_path':str(self.approval),'approval_sha256':sha(self.approval)})
        self.assertFalse(broken['states']['RAW_ACQUIRED'])

    def test_frozen_manifest_hash_change_detected_read_only(self):
        from app.locdi.routing_policy_analysis import verify_manifest
        source=ROOT/'outputs/research_evaluation/c3_05_v1/completion_manifest.json'
        m=json.loads(source.read_text(encoding='utf8'));m['files_fingerprint']='0'*64
        p=self.root/'TEST_ONLY_changed_manifest.json';p.write_text(json.dumps(m),encoding='utf8')
        with self.assertRaises(ValueError):verify_manifest(p)
        self.assertEqual(verify_manifest(source)['status'],'PASS')
        from app.locdi.dataset_freeze import PUBLIC,PRIVATE,load,verify_pair
        public,private=load(ROOT/PUBLIC),load(ROOT/PRIVATE)
        public['manifest_fingerprint']='0'*64
        with self.assertRaises(ValueError):verify_pair(public,private)

    def test_wrong_download_hash_never_commits_raw(self):
        self.transport.side_effect=lambda url,timeout:Response(b'Z'*len(self.data))
        with self.assertRaises(ValueError):self.dispatch()
        self.assertFalse((self.root/'raw/MedQA'/('a'*40)/'data.jsonl').exists())

    def test_html_confirmation_page_not_accepted_as_raw(self):
        def html(url,timeout):r=Response(self.data);r.headers['Content-Type']='text/html';return r
        self.transport.side_effect=html
        with self.assertRaises(ValueError):self.dispatch()

    def test_expired_approval_never_dispatches(self):
        a=json.loads(self.approval.read_text());a['expires_at']='2001-01-01T00:00:00+00:00'
        self.approval.write_text(json.dumps(a),encoding='utf8');self.authority=PinnedApproval(self.approval,sha(self.approval))
        with self.assertRaises(ValueError):self.dispatch()
        self.transport.assert_not_called()


if __name__=='__main__':unittest.main()
