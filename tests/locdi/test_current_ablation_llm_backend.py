"""TEST_ONLY Native transport contracts. No server or real model calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.locdi.current_ablation_runner import GenerationBudget, _BUDGET, CurrentAblationBlocked
from app.locdi.current_ablation_llm_backend import (
    BudgetedNativeERClient, InputContextBudgetError, input_guard, static_preflight, make_er_backend,
)
from app.locdi.ollama_adapter import ModelIdentityError, LLMResponseError, LLMRequestLimitError

MODEL='TEST_ONLY_MODEL'
DIGEST='sha256:'+'a'*64
SCHEMA={'type':'object','properties':{'action':{'enum':['diagnose']}},'required':['action'],'additionalProperties':False}
CONFIG={'model':MODEL,'model_identifier':DIGEST,'model_version':'TEST_ONLY',
    'max_requests':2,'temperature':0,'seed':42,'max_output_tokens':1024,'timeout_seconds':120,'request_retries':0,'minute_limit':90}
PREPARED={'config':CONFIG,'fingerprint':'TEST_ONLY_PREPARED','endpoint':'http://127.0.0.1:11434/v1','registry':{'agents':[]}}
REPORT={'static_readiness':'READY','prepared':PREPARED,'checks':{}}


class Response:
    def __init__(self,data,status=200):self.data=data;self.status_code=status;self.content=json.dumps(data).encode()
    def json(self):return deepcopy(self.data)
    def raise_for_status(self):
        if self.status_code!=200:raise RuntimeError('TEST_ONLY_HTTP_ERROR')


class Transport:
    def __init__(self):
        self.gets=[];self.posts=[];self.context=32768;self.digest=DIGEST;self.closed=False
        self.envelope={'model':MODEL,'done':True,'done_reason':'stop','prompt_eval_count':40,'eval_count':4,
                       'message':{'content':'{"action":"diagnose"}'}}
    def get(self,url):
        self.gets.append(url)
        return Response({'models':[{'name':MODEL,'digest':self.digest,'context_length':self.context,'size':1,'size_vram':1}]})
    def post(self,url,*,json):
        self.posts.append((url,deepcopy(json)));return Response(self.envelope)
    def close(self):self.closed=True


class CurrentAblationLLMBackend(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.transport=Transport()
        self.patches=[patch('app.locdi.ollama_adapter.validate_setup',return_value=deepcopy(REPORT)),
            patch('app.locdi.current_ablation_llm_backend.validate_setup',return_value=deepcopy(REPORT)),
            patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=self.transport)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.token=_BUDGET.set(GenerationBudget(2,2));self.addCleanup(_BUDGET.reset,self.token)
    def client(self,**kwargs):
        return BudgetedNativeERClient(CONFIG,authorized=True,contract_fingerprint='b'*64,
            checkpoint_root=Path(self.tmp.name)/'requests',**kwargs)
    def call(self,client,user=None):
        return client.chat_json('TEST_ONLY SYSTEM',json.dumps(user or {'role':'er','observation':{}}),model=MODEL,schema=SCHEMA)

    def test_static_preflight_no_transport_or_generation(self):
        with patch('app.locdi.current_ablation_llm_backend._native_transport') as factory:
            result=static_preflight(CONFIG)
            factory.assert_not_called();self.assertEqual(result['LLM_generation'],0)

    def test_client_construction_lazy(self):
        c=self.client();self.assertEqual(c.request_count,0);self.assertEqual(self.transport.gets,[]);self.assertEqual(self.transport.posts,[])

    def test_approval_required(self):
        with self.assertRaises(CurrentAblationBlocked):BudgetedNativeERClient(CONFIG)

    def test_native_numctx_and_global_budget(self):
        c=self.client();content=self.call(c)
        self.assertEqual(json.loads(content),{'action':'diagnose'});self.assertEqual(c.request_count,1)
        request=self.transport.posts[0][1]
        self.assertEqual(request['options']['num_ctx'],32768);self.assertEqual(request['options']['num_predict'],1024)
        self.assertEqual(request['options']['temperature'],0);self.assertEqual(request['options']['seed'],42)
        self.assertFalse(request['stream']);self.assertEqual(request['format'],SCHEMA)
        self.assertEqual(_BUDGET.get().used,1);self.assertTrue(c.context_verified);self.assertEqual(c.identity_check_count,2)

    def test_before_after_checkpoints(self):
        c=self.client();self.call(c);p=Path(self.tmp.name)/'requests'
        before=json.loads((p/'Request0001_before.json').read_text(encoding='utf-8'))
        after=json.loads((p/'Request0001_after.json').read_text(encoding='utf-8'))
        self.assertEqual(before['request_fingerprint'],after['request_fingerprint']);self.assertEqual(after['status'],'VALID')
        self.assertTrue(after['response_received'])

    def test_full_history_not_truncated_no_post(self):
        c=self.client()
        with self.assertRaises(InputContextBudgetError):self.call(c,{'role':'er','observation':{'memory':'x'*40000}})
        self.assertEqual(c.request_count,0);self.assertEqual(self.transport.posts,[]);self.assertEqual(self.transport.gets,[])

    def test_schema_bytes_in_context_bound(self):
        schema={'type':'object','description':'x'*40000}
        with self.assertRaises(InputContextBudgetError):input_guard('a','{}',schema)

    def test_hf_counter_cannot_override_byte_limit(self):
        class Counter:
            def count(self,*args):return 1
        with self.assertRaises(InputContextBudgetError):input_guard('a','x'*40000,SCHEMA,tokenizer_counter=Counter())

    def test_repair_rejected_without_second_generation(self):
        c=self.client();self.call(c)
        with self.assertRaises(LLMRequestLimitError):self.call(c,{'output_contract_repair':{'error':'TEST_ONLY'}})
        self.assertEqual(len(self.transport.posts),1);self.assertEqual(c.repair_requests_blocked,1)

    def test_request_limit_before_generation(self):
        c=self.client();_BUDGET.get().limit=0
        with self.assertRaises(CurrentAblationBlocked):self.call(c)
        self.assertEqual(c.request_count,0);self.assertFalse(self.transport.posts)
        after=json.loads((Path(self.tmp.name)/'requests/Request0001_after.json').read_text(encoding='utf-8'))
        self.assertFalse(after['generation_attempted'])

    def test_digest_failure_no_generation(self):
        c=self.client();self.transport.digest='sha256:'+'f'*64
        with self.assertRaises(ModelIdentityError):self.call(c)
        self.assertEqual(c.request_count,0);self.assertFalse(self.transport.posts)

    def test_actual_loaded_context_failure_blocks_next(self):
        c=self.client();self.transport.context=4096
        with self.assertRaises(ModelIdentityError):self.call(c)
        self.assertEqual(c.request_count,1);self.assertFalse(c.context_verified)
        self.transport.context=32768
        with self.assertRaises(LLMRequestLimitError):self.call(c)
        self.assertEqual(len(self.transport.posts),1)

    def test_response_model_mismatch(self):
        c=self.client();self.transport.envelope['model']='TEST_ONLY_OTHER'
        with self.assertRaises(ModelIdentityError):self.call(c)

    def test_schema_failure_not_repaired(self):
        c=self.client();self.transport.envelope['message']['content']='{"action":"invented"}'
        with self.assertRaises(LLMResponseError):self.call(c)
        self.assertEqual(c.request_count,1)

    def test_duplicate_keys_rejected(self):
        c=self.client();self.transport.envelope['message']['content']='{"action":"diagnose","action":"diagnose"}'
        with self.assertRaises(LLMResponseError):self.call(c)

    def test_truncated_response_not_success(self):
        c=self.client();self.transport.envelope['done_reason']='length'
        with self.assertRaises(LLMResponseError):self.call(c)

    def test_server_prompt_token_boundary(self):
        c=self.client();self.transport.envelope['prompt_eval_count']=32000
        with self.assertRaises(LLMResponseError):self.call(c)

    def test_missing_server_token_counts_not_invented(self):
        c=self.client();del self.transport.envelope['prompt_eval_count']
        with self.assertRaises(LLMResponseError):self.call(c)

    def test_source_case_closure_whitelist(self):
        backend=make_er_backend({'TEST_ONLY':{}},config=CONFIG,contract_fingerprint='b'*64,
            checkpoint_root=Path(self.tmp.name)/'backend',authorized=True)
        with self.assertRaisesRegex(ValueError,'OPAQUE'):backend({'case_ref':'TEST_ONLY','reference':'GOLD'}, {})
        with self.assertRaisesRegex(ValueError,'ER_ONLY'):backend({'case_ref':'TEST_ONLY'}, {'voi_enabled':True})

    def test_close(self):
        c=self.client();self.call(c);c.close();self.assertTrue(self.transport.closed)


if __name__=='__main__':unittest.main()
