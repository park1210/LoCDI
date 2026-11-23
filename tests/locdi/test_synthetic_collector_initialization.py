"""TEST_ONLY real Native-client lifecycle; all HTTP replaced, sockets denied."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from app.locdi.synthetic_pilot_collection import read,PRIVATE,collect,MARKERS
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient

class Reply:
    status_code=200
    def __init__(self,payload):self.payload=payload;self.content=json.dumps(payload).encode()
    def json(self):return self.payload
    def raise_for_status(self):pass

class NativeFake:
    def __init__(self,config):self.config=config;self.posts=[];self.gets=[]
    def get(self,url):
        self.gets.append(url)
        return Reply({'models':[{'name':self.config['model'],'model':self.config['model'],
            'digest':self.config['model_identifier'],'context_length':32768,'size':1,'size_vram':1}]})
    def post(self,url,*,json):
        self.posts.append(deepcopy(json))
        return Reply({'model':self.config['model'],'done':True,'done_reason':'stop','prompt_eval_count':1,'eval_count':1,
          'message':{'content':__import__('json').dumps({'action':'diagnose','diagnosis':'Unresolved','certainty':'unresolved','rationale':'TEST_ONLY_NOT_MODEL_OUTPUT'})}})
    def close(self):pass

class CollectorInitialization(unittest.TestCase):
    def fixture(self):
        p=deepcopy(read(PRIVATE/'syn15_initial_full_002/plan.json'))
        p.update(plan_id='TEST_ONLY',run_id='TEST_ONLY_RUN',fingerprint='1'*64,tasks=p['tasks'][:1],generation_cap=5)
        now=datetime.now(timezone.utc)
        a={**MARKERS,'scope':'AUTHORED_SYNTHETIC_5AGENT_COLLECTION_ONLY','plan_fingerprint':p['fingerprint'],
            'generation_cap':5,'operator_approved':True,'accept_author_reference_proxy':True,
            'accept_synthetic_acquisition_timing':True,'no_other_ollama_work':True,
            'operator_ref':'TEST_ONLY_NOT_AUTHORITY','statement':'TEST_ONLY fake transport only',
            'recorded_at':now.isoformat(),'expires_at':(now+timedelta(hours=1)).isoformat(),'TEST_ONLY':True}
        return p,a
    def test_old_precreated_leaf_reproduces_before_any_transport(self):
        p,_=self.fixture()
        with tempfile.TemporaryDirectory() as d,patch('app.locdi.current_ablation_llm_backend._native_transport') as transport:
            path=Path(d)/'existing';path.mkdir()
            with self.assertRaises(FileExistsError):BudgetedNativeERClient(p['config'],authorized=True,contract_fingerprint=p['fingerprint'],checkpoint_root=path)
            transport.assert_not_called()
    def test_five_agents_real_client_fake_http_checkpoint_lifecycle(self):
        p,a=self.fixture();fake=NativeFake(p['config'])
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
            root=Path(d);pub=root/'public';pr=root/'private';pub.mkdir();pr.mkdir();approval=root/'TEST_ONLY_approval.json';approval.write_text(json.dumps(a))
            stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN')))
            stack.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN')))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.check',return_value=p))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.locations',return_value=(pub,pr)))
            stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
            report=collect('TEST_ONLY',execute=True,allow_llm=True,approval=approval)
            self.assertEqual(report['status'],'COLLECTED');self.assertEqual(len(fake.posts),5)
            self.assertEqual(report['generation_attempts'],5);self.assertEqual(report['not_attempted'],0)
            self.assertEqual(len(list(pr.rglob('Request0001_before.json'))),5)
            self.assertEqual(len(list(pr.rglob('Request0001_after.json'))),5)
            self.assertEqual(len(list(pr.rglob('response.json'))),5)
            self.assertEqual(len(fake.gets),10)
            self.assertEqual(len({json.dumps({k:v for k,v in json.loads(x['messages'][1]['content']).items() if k!='department'},sort_keys=True) for x in fake.posts}),1)
            with self.assertRaises(FileExistsError):collect('TEST_ONLY',execute=True,allow_llm=True,approval=approval)
    def test_constructor_error_preserves_marker_and_zero_attempts(self):
        p,a=self.fixture()
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
            root=Path(d);pub=root/'public';pr=root/'private';pub.mkdir();pr.mkdir();approval=root/'TEST_ONLY_approval.json';approval.write_text(json.dumps(a))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.check',return_value=p))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.locations',return_value=(pub,pr)))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.BudgetedNativeERClient',side_effect=ValueError('TEST_ONLY constructor failure')))
            report=collect('TEST_ONLY',execute=True,allow_llm=True,approval=approval)
            self.assertEqual(report['status'],'PARTIAL_OR_FAILED');self.assertEqual(report['generation_attempts'],0)
            marker=read(next(pr.rglob('response.json')));self.assertEqual(marker['generation_attempts'],0)
            self.assertIsNone(marker['response']);self.assertEqual(report['not_attempted'],4)
    def test_cli_partial_collection_is_nonzero_exit(self):
        from scripts import synthetic_pilot_utility as cli
        with patch('sys.argv',['TEST_ONLY','collect','--plan-id','TEST_ONLY']),patch.object(cli,'collect',return_value={'status':'PARTIAL_OR_FAILED'}),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:cli.main()
            self.assertEqual(exc.exception.code,2)

    def test_strict_schema_accepts_only_consistent_pairs(self):
        from jsonschema import Draft202012Validator,ValidationError
        from app.locdi.synthetic_pilot_collection import response_schema
        p,_=self.fixture();validator=Draft202012Validator(response_schema(p['catalog']))
        for diagnosis in p['catalog']:
            validator.validate({'action':'diagnose','diagnosis':diagnosis,'certainty':'supported','rationale':'TEST_ONLY'})
            with self.assertRaises(ValidationError):validator.validate({'action':'diagnose','diagnosis':diagnosis,'certainty':'unresolved','rationale':'TEST_ONLY'})
        validator.validate({'action':'diagnose','diagnosis':'Unresolved','certainty':'unresolved','rationale':'TEST_ONLY'})
        with self.assertRaises(ValidationError):validator.validate({'action':'diagnose','diagnosis':'Unresolved','certainty':'supported','rationale':'TEST_ONLY'})

    def test_original_schema_and_semantic_gate_unchanged(self):
        from app.locdi.routing_collection_preparation import response_schema as legacy
        from app.locdi.synthetic_pilot_collection import response_schema
        from app.locdi.synthetic_pilot_source import score_response
        from jsonschema import Draft202012Validator
        p,_=self.fixture();before=legacy(p['catalog']);response_schema(p['catalog']);self.assertEqual(before,legacy(p['catalog']))
        invalid={'action':'diagnose','diagnosis':next(iter(p['catalog'])),'certainty':'unresolved','rationale':'TEST_ONLY'}
        Draft202012Validator(before).validate(invalid)
        with self.assertRaisesRegex(ValueError,'CERTAINTY_DIAGNOSIS_CONTRADICTION'):score_response(invalid,{},p['catalog'])

    def test_schema_fingerprint_stable_across_plan_json_key_sorting(self):
        from app.locdi.synthetic_pilot_collection import response_schema
        from app.locdi.llm_setup import digest
        catalog={'Z_TEST_ONLY':'nephrology','A_TEST_ONLY':'cardiology'}
        persisted=json.loads(json.dumps(catalog,sort_keys=True))
        self.assertEqual(digest(response_schema(catalog)),digest(response_schema(persisted)))

    def test_contradiction_is_preserved_failed_without_retry_or_coercion(self):
        class Contradictory(NativeFake):
            def post(self,url,*,json):
                r=super().post(url,json=json)
                row=__import__('json').loads(r.payload['message']['content']);row.update(diagnosis='Asthma (disorder)',certainty='unresolved')
                r.payload['message']['content']=__import__('json').dumps(row);return Reply(r.payload)
        p,a=self.fixture();fake=Contradictory(p['config'])
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack:
            root=Path(d);pub=root/'public';pr=root/'private';pub.mkdir();pr.mkdir();approval=root/'TEST_ONLY_approval.json';approval.write_text(json.dumps(a))
            stack.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN')))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.check',return_value=p))
            stack.enter_context(patch('app.locdi.synthetic_pilot_collection.locations',return_value=(pub,pr)))
            stack.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
            report=collect('TEST_ONLY',execute=True,allow_llm=True,approval=approval)
            self.assertEqual(len(fake.posts),1);self.assertEqual(report['normal_responses'],0);self.assertEqual(report['not_attempted'],4)
            marker=read(next(pr.rglob('response.json')));self.assertIsNone(marker['response'])
            after=read(next(pr.rglob('Request0001_after.json')));self.assertEqual(after['status'],'FAILED')
            raw=json.loads(after['response_envelope']['message']['content']);self.assertEqual(raw['certainty'],'unresolved');self.assertNotEqual(raw['diagnosis'],'Unresolved')

if __name__=='__main__':unittest.main()
