"""TEST_ONLY fixtures, fake transports and static source audit. No network."""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import app.locdi.external_dialogue as external
from app.locdi.llm_setup import ROOT,load_setup,digest
from scripts.run_healthbench_pilot import main
from scripts.inspect_healthbench_grader import inspect


def fixture_rows():
    return [{'prompt_id':'TEST_ONLY_PRIVATE_ID_'+str(i),'prompt':[
        {'role':'user','content':'  TEST_ONLY initial\nquestion  '},
        {'role':'assistant','content':'TEST_ONLY historical assistant context'},
        {'role':'user','content':'TEST_ONLY final user '+str(i)}],
        'rubrics':[{'criterion':'TEST_ONLY_HIDDEN_RUBRIC','points':2,'tags':['TEST_ONLY_HIDDEN_TAG']} for _ in range(i+1)],
        'ideal_completions_data':{'response':'TEST_ONLY_HIDDEN_REFERENCE'},'canary':'TEST_ONLY_HIDDEN_CANARY',
        'example_tags':['TEST_ONLY_HIDDEN_TAG'],'unknown_evaluation_metadata':'TEST_ONLY_HIDDEN_OTHER'} for i in range(1,6)]


class FakeDialogue(external.OfflineDialogueClient):
    def __init__(self,failure=None):self.calls=[];self.failure=failure
    def chat_messages(self,messages,*,model):
        self.calls.append(deepcopy(messages))
        if self.failure=='timeout' and len(self.calls)==2:raise TimeoutError('TEST_ONLY_PRIVATE_ERROR_EVIDENCE')
        if self.failure=='empty':return ''
        return 'TEST_ONLY_PRIVATE_RESPONSE: free prose, not Action JSON.\nSecond line.'


class ExternalDialogueContracts(unittest.TestCase):
    def setUp(self):
        self.private=tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/private'))
        self.public=tempfile.TemporaryDirectory(dir=str(ROOT/'outputs/reports'))
        self.addCleanup(self.private.cleanup);self.addCleanup(self.public.cleanup);self.addCleanup(patch.stopall)
        self.private_root=Path(self.private.name)/'D5_TEST_ONLY';self.public_root=Path(self.public.name)/'D5_TEST_ONLY'
        patch.object(external,'PRIVATE',self.private_root).start();patch.object(external,'PUBLIC',self.public_root).start()
        self.rows=fixture_rows();self.manifest={'manifest_fingerprint':external.EXPECTED_MANIFEST,
            'files':{external.DATA_NAME:{'sha256':'TEST_ONLY_HASH'}},'inventory':{'sample_count':5000}}
        patch.object(external,'verify_source',return_value=(self.manifest,self.rows)).start()
        self.config=load_setup();self.identifier=self.config['model_identifier']
        patch('app.locdi.llm_setup.inspect_local_model',return_value={'status':'READY','model_identifier':self.identifier}).start()
        self.run_id='d5'*16

    def prepare(self):return external.prepare_pilot()

    def execute(self,client=None):
        client=client or FakeDialogue();check=external.preflight(self.run_id)
        result=external.run_pilot(self.run_id,execute=True,test_only=True,client=client,expected_plan_fingerprint=check['plan_fingerprint'])
        return result,client

    def test_lossless_multiturn_role_content_order_and_final_user(self):
        row=self.rows[0];before=deepcopy(row);messages=external.messages_for(row)
        self.assertEqual(messages,row['prompt']);self.assertEqual(row,before);self.assertEqual(messages[-1]['role'],'user')
        self.assertEqual([m['role'] for m in messages],['user','assistant','user'])
        messages[0]['content']='TEST_ONLY_MUTATION';self.assertEqual(row,before)

    def test_rubric_reference_canary_tags_and_unknown_metadata_never_in_input(self):
        text=json.dumps(external.messages_for(self.rows[0]))
        for sentinel in ('RUBRIC','REFERENCE','CANARY','HIDDEN_TAG','HIDDEN_OTHER','PRIVATE_ID'):self.assertNotIn(sentinel,text)

    def test_invalid_last_role_and_message_extra_fields_rejected(self):
        row=deepcopy(self.rows[0]);row['prompt'][-1]['role']='assistant'
        with self.assertRaises(ValueError):external.messages_for(row)
        row=deepcopy(self.rows[0]);row['prompt'][0]['rubric']='TEST_ONLY'
        with self.assertRaises(ValueError):external.messages_for(row)

    def test_selection_precedes_responses_and_preserves_test_role(self):
        selected=self.prepare();self.assertEqual([s['source_row'] for s in selected['samples']],[1,2,3])
        self.assertEqual([s['sample_id'] for s in selected['samples']],[r['prompt_id'] for r in self.rows[:3]])
        self.assertTrue(selected['selected_before_model_responses']);self.assertEqual(selected['original_split'],'test')
        self.assertIsNone(selected['assigned_internal_split']);self.assertFalse(selected['independent_test_approved'])
        self.assertFalse((self.private_root/'runs').exists())
        before=(self.private_root/'selection_manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertEqual(before,(self.private_root/'selection_manifest.json').read_bytes())

    def test_selection_hash_and_expansion_or_split_changes_blocked(self):
        self.prepare();path=self.private_root/'selection_manifest.json';original=path.read_text(encoding='utf-8')
        for key,value in [('assigned_internal_split','train'),('samples',[]),('selection_fingerprint','0'*64)]:
            altered=json.loads(original);altered[key]=value
            if key!='selection_fingerprint':altered['selection_fingerprint']=digest({k:v for k,v in altered.items() if k!='selection_fingerprint'})
            path.write_text(json.dumps(altered),encoding='utf-8')
            self.assertTrue(external.preflight(self.run_id)['blocker_codes'])

    def test_default_dry_run_never_constructs_client_or_backend(self):
        self.prepare()
        with patch.object(external,'LocalDialogueClient') as client,patch('app.locdi.ollama_adapter._backend') as backend:
            result=external.run_pilot(self.run_id)
        self.assertEqual(result['status'],'DRY_RUN');self.assertEqual(result['model_calls'],0);self.assertEqual(result['grader_calls'],0)
        client.assert_not_called();backend.assert_not_called();self.assertFalse((self.private_root/'runs').exists())

    def test_missing_approval_client_injection_and_grading_always_blocked(self):
        for kwargs in ({'execute':True},{'execute':1},{'execute':True,'grade':True},{'execute':True,'allow_llm':True,'client':FakeDialogue()},
                       {'execute':True,'test_only':True,'client':object()}):
            with patch.object(external,'LocalDialogueClient') as client:
                result=external.run_pilot(self.run_id,**kwargs)
            self.assertEqual(result['status'],'BLOCKED');client.assert_not_called()
        self.prepare()
        result=external.run_pilot(self.run_id,execute=True,allow_llm=True,expected_plan_fingerprint='0'*64)
        self.assertIn('PLAN_APPROVAL_MISMATCH',result['blocker_codes'])

    def test_budget_includes_every_rubric_retry_repair_and_full_dataset_blocks(self):
        b=external.budget(3,31);self.assertEqual(b['total_llm_max_requests'],34);self.assertTrue(b['within_cap'])
        full=external.budget(5000,57237);self.assertEqual(full['total_llm_max_requests'],62237);self.assertFalse(full['within_cap'])
        b=external.budget(3,31,generation_retries=1,generation_repairs=1,grader_retries=1,grader_repairs=1)
        self.assertEqual(b['total_llm_max_requests'],102);self.assertFalse(b['within_cap'])
        for args in ((True,31),(3,2),(0,0)):
            with self.assertRaises(ValueError):external.budget(*args)

    def test_free_text_private_responses_public_audit_and_reuse_history(self):
        self.prepare();result,client=self.execute();self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(client.calls,[r['prompt'] for r in self.rows[:3]])
        self.assertEqual(result['model_calls'],0);self.assertEqual(result['fake_dispatches'],3);self.assertEqual(result['grader_calls'],0)
        root=self.private_root/'runs'/self.run_id
        response=json.loads((root/'Sample001_response.json').read_text(encoding='utf-8'))
        self.assertIn('free prose',response['response']);self.assertIn('sample_id',response)
        public=(self.public_root/(self.run_id+'.json')).read_text(encoding='utf-8')
        for secret in ('PRIVATE_ID','PRIVATE_RESPONSE','HIDDEN_RUBRIC','HIDDEN_REFERENCE','HIDDEN_CANARY'):self.assertNotIn(secret,public)
        history=json.loads((root/'reuse_history.json').read_text(encoding='utf-8'))
        self.assertEqual(history['exposure_kind'],'TEST_ONLY_FAKE');self.assertTrue(history['excluded_from_independent_aggregate'])
        self.assertIsNone(result['official_score']);self.assertFalse(result['research_evaluation_ready'])

    def test_error_stops_without_retry_preserves_previous_response_and_no_raw_error(self):
        self.prepare();result,client=self.execute(FakeDialogue('timeout'))
        self.assertEqual(result['status'],'FAILED');self.assertEqual(len(client.calls),2);self.assertEqual(result['uncollected_samples'],1)
        root=self.private_root/'runs'/self.run_id;self.assertTrue((root/'Sample001_response.json').exists())
        self.assertTrue((root/'Sample002_error.json').exists());self.assertFalse((root/'Sample003_before.json').exists())
        self.assertNotIn('PRIVATE_ERROR_EVIDENCE',json.dumps(result))

    def test_empty_output_is_failed_not_synthesized(self):
        self.prepare();result,client=self.execute(FakeDialogue('empty'))
        self.assertEqual(result['status'],'FAILED');self.assertEqual(len(client.calls),1)
        self.assertFalse((self.private_root/'runs'/self.run_id/'Sample001_response.json').exists())

    def test_existing_run_and_path_escape_block_before_dispatch(self):
        self.prepare();self.execute();root=self.private_root/'runs'/self.run_id
        before={p.name:p.read_bytes() for p in root.iterdir()};again,client=self.execute()
        self.assertEqual(again['status'],'BLOCKED');self.assertEqual(client.calls,[])
        self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
        self.assertTrue(external.preflight('../escape')['blocker_codes'])
        with patch.object(external,'PRIVATE',ROOT/'datasets/TEST_ONLY_ESCAPE'):
            self.assertIn('OUTPUT_PATH_ESCAPE',external.preflight(self.run_id)['blocker_codes'])

    def test_checkpoint_storage_failure_before_model_stops_without_call(self):
        self.prepare();write=external.save
        def fail(root,name,document):
            if name=='Sample001_before.json':raise OSError('TEST_ONLY_PRIVATE_ERROR_EVIDENCE')
            return write(root,name,document)
        with patch.object(external,'save',side_effect=fail):result,client=self.execute()
        self.assertEqual(result['status'],'FAILED');self.assertEqual(client.calls,[])

    def test_real_facade_sdk_receives_original_messages_no_system_or_json_format(self):
        self.prepare();calls=[];identifier=self.identifier
        class HTTP:
            def get(self,url):return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'models':[{'name':'qwen2.5:1.5b-instruct','digest':identifier}]})
            def close(self):pass
        def create(**kwargs):
            calls.append(kwargs);return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='TEST_ONLY free prose'))])
        http=HTTP();legacy=SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        check=external.preflight(self.run_id)
        with patch('app.locdi.ollama_adapter._backend',return_value=(legacy,http,http,http)),patch('socket.create_connection',side_effect=AssertionError('NO_NETWORK')):
            result=external.run_pilot(self.run_id,execute=True,allow_llm=True,expected_plan_fingerprint=check['plan_fingerprint'])
        self.assertEqual(result['status'],'COMPLETED');self.assertEqual(result['model_calls'],3)
        self.assertEqual([c['messages'] for c in calls],[r['prompt'] for r in self.rows[:3]])
        self.assertTrue(all(set(c)=={'model','messages'} for c in calls));self.assertEqual(result['identity_check_count'],3)

    def test_model_identity_failure_blocks_generation(self):
        self.prepare();calls=[]
        http=SimpleNamespace(get=lambda url:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'models':[]}),close=lambda:None)
        legacy=SimpleNamespace(client=None)
        check=external.preflight(self.run_id)
        with patch('app.locdi.ollama_adapter._backend',return_value=(legacy,http,http,http)):
            result=external.run_pilot(self.run_id,execute=True,allow_llm=True,expected_plan_fingerprint=check['plan_fingerprint'])
        self.assertEqual(result['status'],'FAILED');self.assertEqual(result['model_calls'],0);self.assertEqual(result['identity_check_count'],1)
        self.assertEqual(result['observed_server_digest'],'NOT_RECORDED')

    def test_cli_default_dry_run_and_missing_execution_flags(self):
        self.prepare();out=io.StringIO()
        with redirect_stdout(out):self.assertEqual(main(['--run-id',self.run_id]),0)
        self.assertEqual(json.loads(out.getvalue())['status'],'DRY_RUN')
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):main(['--run-id',self.run_id,'--execute'])

    def test_hospital_runner_is_never_used_for_external_dialogue(self):
        self.prepare()
        with patch('app.locdi.run_simulation.run_case',side_effect=AssertionError('NO_HOSPITAL_CASE')) as hospital:
            result,client=self.execute()
        hospital.assert_not_called();self.assertEqual(result['status'],'COMPLETED')

    def test_changed_selected_input_blocks_even_with_old_approved_plan(self):
        self.prepare();check=external.preflight(self.run_id);self.rows[0]['prompt'][0]['content']='TEST_ONLY_CHANGED_SOURCE'
        client=FakeDialogue()
        result=external.run_pilot(self.run_id,execute=True,test_only=True,client=client,expected_plan_fingerprint=check['plan_fingerprint'])
        self.assertEqual(result['status'],'BLOCKED');self.assertEqual(client.calls,[])


class ExternalActualSourceContracts(unittest.TestCase):
    def test_all_5000_samples_lossless_and_hash_manifest_immutable(self):
        manifest,rows=external.verify_source();self.assertEqual(len(rows),5000)
        self.assertEqual(manifest['manifest_fingerprint'],external.EXPECTED_MANIFEST)
        for row in rows:self.assertEqual(external.messages_for(row),row['prompt'])
        self.assertEqual(sum(len(r['rubrics']) for r in rows[:3]),31)

    def test_grader_static_environment_incompatibility_no_import_or_execution(self):
        report=inspect();self.assertEqual(report['builtin_probes']['pep604'],'UNSUPPORTED_TYPE_ERROR')
        self.assertEqual(report['builtin_probes']['zip_strict'],'UNSUPPORTED_TYPE_ERROR')
        self.assertEqual(report['unbounded_retry_lines'],[415]);self.assertFalse(report['evaluator_imported'])
        self.assertEqual(report['grader_calls'],0);self.assertFalse(report['separate_environment_spec']['environment_created'])

    def test_corrupted_raw_or_manifest_hash_fails_closed(self):
        with patch.object(external,'sha',return_value='TEST_ONLY_WRONG_HASH'):
            with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):external.verify_source()
        manifest=external.read(external.MANIFEST);manifest['inventory']['sample_count']=1
        manifest['manifest_fingerprint']=digest({k:v for k,v in manifest.items() if k!='manifest_fingerprint'})
        with patch.object(external,'read',return_value=manifest):
            with self.assertRaisesRegex(ValueError,'MANIFEST_FINGERPRINT_MISMATCH'):external.verify_source()


if __name__=='__main__':unittest.main()
