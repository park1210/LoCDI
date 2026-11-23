"""TEST_ONLY fake adapters; no model calls, real approvals or clinical labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.locdi import routing_two_decision_collection as two
from app.locdi.dataset_freeze import load
from app.locdi.llm_setup import ROOT, digest
from app.locdi.ollama_adapter import LLMTimeoutError, LLMRequestError, LLMResponseError
from app.locdi.routing_collection_preparation import response_schema
from app.llm.prompt_loader import specialty_system_prompt


def fixture():
    registry = load(ROOT / 'configs/agent_registry.specialty_development.json')
    agents = [a for a in registry['agents'] if a['role'] == 'specialty']
    candidates = []
    for i in range(1, 16):
        payload = {'task': 'TEST_ONLY', 'observation': {'chief_complaint': 'TEST_ONLY_%d' % i},
                   'allowed_actions': ['diagnose'], 'diagnosis_catalog': ['TEST_ONLY_A']}
        candidates.append({'case_ref': 'Candidate%03d' % i, 'shared_payload': payload,
            'shared_input_fingerprint': digest(payload), 'source_fingerprint': digest(['TEST_ONLY_SOURCE', i]),
            'memory_fingerprint': digest(['TEST_ONLY_MEMORY', i]), 'decision': {
                'patient_id': 'TEST_ONLY_P%d' % i, 'visit_id': 'TEST_ONLY_V%d' % i, 'decision_id': 'TEST_ONLY_D%d' % i,
                'split': 'train' if i <= 10 else 'validation', 'generation': {'prompt_fingerprints': {
                    a['agent_id']: hashlib.sha256(specialty_system_prompt(a['department']).encode('utf-8')).hexdigest() for a in agents}}}})
    bundle = {'registry': registry, 'catalog': ['TEST_ONLY_A'], 'candidates': candidates,
        'response_schema': response_schema(['TEST_ONLY_A']), 'fingerprint': digest('TEST_ONLY_PREPARATION'),
        'split_manifest': {'patients': {'validation': ['TEST_ONLY_P%d' % i for i in range(11,16)]}},
        'llm_config': {**two.GENERATION, 'request_retries': 0, 'model': agents[0]['model'],
                      'model_identifier': 'TEST_ONLY_DIGEST', 'model_version': agents[0]['version']}}
    selected = [{k: c[k] for k in ('case_ref', 'shared_input_fingerprint', 'memory_fingerprint', 'source_fingerprint')}
                | {'prompt_fingerprints': c['decision']['generation']['prompt_fingerprints']} for c in candidates[:2]]
    parent = two.signed({'selected': selected, 'agents': agents, 'registry_fingerprint': digest(registry),
        'preparation_fingerprint': bundle['fingerprint'], 'catalog_fingerprint': digest(bundle['catalog']),
        **{k: bundle['llm_config'][k] for k in ('model', 'model_identifier', 'model_version')}})
    with patch.object(two, 'PARENT_FP', parent['fingerprint']), patch.object(two, 'validate_preparation', return_value=agents):
        plan = two.build_plan(bundle, parent)
    return bundle, parent, plan, {'binding_pins': {}, 'source_pins': {}}


class FakeAdapter:
    fail_at = None
    mode = 'success'
    calls = []
    def __init__(self, cfg, authorized):
        assert authorized and cfg['max_requests'] <= 10
        self.request_count = self.identity_check_count = 0
        self.last_response_content = None; self.last_post_status = None
        self.last_observed_model_identifier = 'TEST_ONLY_DIGEST'
    def close(self): pass
    def chat_json(self, system, user, model=None, *, schema=None):
        self.request_count += 1; self.identity_check_count += 1
        self.last_response_content = None; self.last_post_status = 200
        self.calls.append({'system': system, 'user': json.loads(user), 'model': model})
        if self.request_count == self.fail_at:
            if self.mode == 'timeout': raise LLMTimeoutError('TEST_ONLY_PRIVATE_NO_PUBLIC')
            if self.mode == 'http':
                self.last_post_status = 500; raise LLMRequestError('TEST_ONLY_PRIVATE_HTTP_BODY')
            if self.mode == 'client':
                self.last_post_status = None; raise LLMRequestError('TEST_ONLY')
            self.last_response_content = '{"TEST_ONLY_INVALID":true}' if self.mode == 'schema' else 'TEST_ONLY_INVALID_JSON'
            raise LLMResponseError('TEST_ONLY')
        self.last_response_content = json.dumps({'action': 'diagnose', 'diagnosis': 'Unresolved' if self.mode == 'unresolved' else 'TEST_ONLY_A',
            'certainty': 'unresolved' if self.mode == 'unresolved' else 'supported', 'rationale': 'TEST_ONLY'})
        return self.last_response_content


class TwoDecisionContracts(unittest.TestCase):
    def run_fake(self, root, mode='success', fail_at=None, cap=10, run_id='TEST_ONLY_RUN'):
        bundle, parent, plan, bindings = fixture()
        with patch.object(two, 'ROOT', Path(root)), patch.object(two, 'load_protocol', return_value=(bundle, plan, bindings)), patch.object(two, 'ObservedAdapter', FakeAdapter), patch.object(FakeAdapter, 'mode', mode), patch.object(FakeAdapter, 'fail_at', fail_at), patch.object(FakeAdapter, 'calls', []):
            result = two.collect('TEST_ONLY_PROTOCOL', run_id, execute=True, allow_llm=True, accept_proxy=True,
                acknowledge_timeout=True, expected_plan_fingerprint=plan['fingerprint'], max_requests=cap)
            calls = deepcopy(FakeAdapter.calls)
        private = Path(root) / 'outputs/private/research_evaluation/TEST_ONLY_PROTOCOL/collection' / run_id
        return result, calls, private

    def test_ten_calls_two_fixed_decisions_agent_order_and_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            result, calls, private = self.run_fake(root)
            self.assertEqual(result['generation_requests'], 10); self.assertEqual(result['status'], 'COLLECTION_COMPLETE_NOT_CLINICALLY_REVIEWED')
            _, _, plan, _ = fixture()
            saved = [load(p) for p in sorted(private.glob('Request*_after.json'))]
            self.assertEqual([r['case_ref'] for r in saved], ['Candidate001'] * 5 + ['Candidate002'] * 5)
            self.assertEqual([r['agent_id'] for r in saved], [a['agent_id'] for a in plan['agents']] * 2)
            for offset in (0, 5):
                common = [{k:v for k,v in c['user'].items() if k != 'department'} for c in calls[offset:offset+5]]
                self.assertTrue(all(x == common[0] for x in common))
            self.assertEqual(len(list(private.glob('*_before.json'))), 10)
            self.assertTrue(all(r['correct'] is None for r in saved))
            self.assertEqual(result['independently_evaluable_responses'], 0)
            self.assertEqual(result['schema_valid_responses'], 10)

    def test_first_middle_last_fail_stop_all_error_types(self):
        for mode, category in [('timeout','TIMEOUT'), ('http','SERVER_HTTP_ERROR'), ('schema','SCHEMA_OR_JSON_ERROR'), ('json','SCHEMA_OR_JSON_ERROR')]:
            for index in (1, 6, 10):
                with self.subTest(mode=mode, index=index), tempfile.TemporaryDirectory() as root:
                    result, calls, private = self.run_fake(root, mode, index)
                    self.assertEqual(len(calls), index); self.assertEqual(result['generation_requests'], index)
                    self.assertEqual(result['stop_reason'], category); self.assertEqual(result['status'], 'COLLECTION_PARTIAL_OR_FAILED')
                    self.assertEqual(len(list(private.glob('*_before.json'))), index)
                    self.assertEqual(len(list(private.glob('*_after.json'))), index)
                    last = load(private / ('Request%03d_after.json' % index))
                    self.assertIsNone(last['correct'])
                    self.assertEqual(last['raw_response'] is not None, mode in ('schema', 'json'))
                    self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(result))

    def test_received_json_schema_and_evaluable_counts_separate(self):
        for mode, received, valid_json in [('timeout',0,0), ('http',0,0), ('schema',1,1), ('json',1,0)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                r, _, _ = self.run_fake(root, mode, 1)
                self.assertEqual(r['received_responses'], received); self.assertEqual(r['json_valid_responses'], valid_json)
                self.assertEqual(r['schema_valid_responses'], 0); self.assertEqual(r['independently_evaluable_responses'], 0)

    def test_global_cap_across_decisions(self):
        with tempfile.TemporaryDirectory() as root:
            r, calls, _ = self.run_fake(root, cap=7)
            self.assertEqual(len(calls), 7); self.assertEqual(r['not_attempted_tasks'], 3)
            self.assertEqual(r['stop_reason'], 'GLOBAL_REQUEST_LIMIT')

    def test_unresolved_preserved_not_false(self):
        with tempfile.TemporaryDirectory() as root:
            r, _, private = self.run_fake(root, 'unresolved')
            self.assertEqual(r['unresolved_responses'], 10); self.assertEqual(r['schema_valid_responses'], 10)
            self.assertEqual(r['supported_responses'], 0); self.assertEqual(r['independently_evaluable_responses'], 0)
            for p in private.glob('*_after.json'):
                row=load(p); self.assertEqual(row['response']['diagnosis'], 'Unresolved'); self.assertIsNone(row['correct'])

    def test_approval_required_before_protocol_or_client(self):
        with patch.object(two, 'load_protocol') as protocol, patch.object(two, 'ObservedAdapter') as client:
            for flags in ({}, {'allow_llm':True}, {'allow_llm':True,'accept_proxy':True}, {'allow_llm':True,'accept_proxy':True,'acknowledge_timeout':True}):
                with self.assertRaises(ValueError): two.collect('TEST_ONLY', 'TEST_ONLY', execute=True, **flags)
            protocol.assert_not_called(); client.assert_not_called()

    def test_dry_run_no_client_or_files(self):
        b, _, p, bindings=fixture()
        with tempfile.TemporaryDirectory() as root, patch.object(two,'ROOT',Path(root)), patch.object(two,'load_protocol',return_value=(b,p,bindings)), patch.object(two,'ObservedAdapter') as client:
            r=two.collect('TEST_ONLY','TEST_ONLY')
            self.assertEqual(len(r['ordered_tasks']),10); self.assertFalse(r['client_constructed']); client.assert_not_called()
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_budget_invalid(self):
        for cap in (0,11,True):
            with self.assertRaises(ValueError): two.collect('TEST_ONLY','TEST_ONLY',max_requests=cap)

    def test_no_resume_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            _,_,private=self.run_fake(root); before={p.name:p.read_bytes() for p in private.glob('*')}
            with self.assertRaisesRegex(ValueError,'NO_RESUME'): self.run_fake(root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in private.glob('*')})

    def test_before_checkpoint_failure_zero_calls(self):
        original=two.atomic_json
        def fail(path, value):
            if path.name.endswith('_before.json'): raise OSError('TEST_ONLY_STORAGE')
            return original(path,value)
        with tempfile.TemporaryDirectory() as root, patch.object(two,'atomic_json',side_effect=fail), patch.object(FakeAdapter,'chat_json') as chat:
            with self.assertRaises(OSError): self.run_fake(root)
            chat.assert_not_called()

    def test_after_checkpoint_failure_no_next_request(self):
        original=two.atomic_json
        def fail(path,value):
            if path.name=='Request006_after.json': raise OSError('TEST_ONLY_STORAGE')
            return original(path,value)
        with tempfile.TemporaryDirectory() as root, patch.object(two,'atomic_json',side_effect=fail):
            with self.assertRaises(OSError): self.run_fake(root)
            private=next(Path(root).rglob('failed.json'))
            self.assertEqual(load(private)['generation_requests'],6)
            self.assertEqual(len(list(private.parent.glob('*_after.json'))),5)
            self.assertTrue((private.parent/'Request006_before.json').exists())

    def test_dispatch_registry_model_prompt_and_input_mismatch(self):
        for kind in ('registry','model','prompt','input','source'):
            b,_,p,bindings=fixture(); c=b['candidates'][0]; a=deepcopy(p['agents'][0])
            if kind=='registry': b['registry']['agents'][0]['version']='TEST_ONLY_CHANGED'
            if kind=='model': a['model']='TEST_ONLY_OTHER'
            if kind=='prompt': p['selected'][0]['prompt_fingerprints'][a['agent_id']]='0'*64
            if kind=='input': c['shared_payload']['observation']['gold']='TEST_ONLY_FORBIDDEN'
            if kind=='source': c['source_fingerprint']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError): two.validate_dispatch(b,p,c,a,bindings)

    def test_source_file_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'TEST_ONLY';path.write_bytes(b'TEST_ONLY')
            with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'): two.verify_pins({str(path):'0'*64})

    def test_frozen_model_generation_and_selection_not_replaceable(self):
        for kind in ('model','registry','generation','source','selection'):
            b,parent,_,_=fixture(); agents=[a for a in b['registry']['agents'] if a['role']=='specialty']
            if kind=='model': b['llm_config']['model_identifier']='TEST_ONLY_CHANGED'
            if kind=='registry': b['registry']['agents'][0]['version']='TEST_ONLY_CHANGED'
            if kind=='generation': b['llm_config']['timeout_seconds']=121
            if kind=='source': b['candidates'][0]['source_fingerprint']='TEST_ONLY_CHANGED'
            if kind=='selection': parent['selected'][0]['case_ref']='Candidate004';parent=two.signed({k:v for k,v in parent.items() if k!='fingerprint'})
            with patch.object(two,'PARENT_FP',parent['fingerprint']),patch.object(two,'validate_preparation',return_value=agents),self.subTest(kind=kind),self.assertRaises(ValueError): two.build_plan(b,parent)

    def test_duplicate_json_not_approved(self):
        with self.assertRaises(ValueError): two.strict_json('{"x":1,"x":2}')
        with self.assertRaises(ValueError): two.strict_json('{"x":NaN}')

    def test_followup_no_review_never_fits(self):
        b,_,plan,bindings=fixture()
        with tempfile.TemporaryDirectory() as root:
            self.run_fake(root)
            with patch.object(two,'ROOT',Path(root)),patch.object(two,'load_protocol',return_value=(b,plan,bindings)),patch('app.core.routing_utility.fit_utility') as fit:
                r=two.assess_run('TEST_ONLY_PROTOCOL','TEST_ONLY_RUN')
            fit.assert_not_called();self.assertEqual(r['status'],'BLOCKED')
            self.assertIn('INDEPENDENT_REFERENCE_AND_RESPONSE_REVIEW_PACKAGE_MISSING',r['blocker_codes'])

    def test_client_error_not_falsely_http_error(self):
        with tempfile.TemporaryDirectory() as root:
            r,_,_=self.run_fake(root,'client',1)
            self.assertEqual(r['stop_reason'],'CLIENT_ERROR_HTTP_STATUS_NOT_OBSERVED')

    def test_real_adapter_http_hook_with_test_only_transport(self):
        import httpx
        from app.locdi import ollama_adapter as base
        cfg={'model':'TEST_ONLY_MODEL','model_identifier':'sha256:'+'a'*64,'max_requests':10,'timeout_seconds':120}
        prepared={'config':cfg,'endpoint':'http://127.0.0.1:11434/v1','fingerprint':'TEST_ONLY'}
        def transport(request):
            if request.method=='GET': return httpx.Response(200,json={'models':[{'name':cfg['model'],'digest':cfg['model_identifier']}]})
            return httpx.Response(503,json={'error':'TEST_ONLY_PRIVATE'})
        http=httpx.Client(transport=httpx.MockTransport(transport))
        def chat(*args,**kwargs): http.post(prepared['endpoint']+'/chat/completions').raise_for_status()
        legacy=SimpleNamespace(chat_json=chat)
        with patch.object(base,'validate_setup',return_value={'static_readiness':'READY','prepared':prepared}),patch.object(two,'_backend',return_value=(legacy,http,http,http)):
            client=two.ObservedAdapter(cfg,authorized=True)
            with self.assertRaises(LLMRequestError):client.chat_json('TEST_ONLY','TEST_ONLY',model=cfg['model'],schema={'type':'object'})
            self.assertEqual(client.last_post_status,503);self.assertEqual(client.request_count,1)
            client.close()

    def test_protocol_freezes_unknown_and_proxy_without_approval(self):
        p=two.evaluation_protocol()
        self.assertEqual(p['proxy_scope_status'],'NOT_ACCEPTED')
        self.assertIn('KEEP_NULL',p['unknown_reference'])
        self.assertIn('NOT_EVALUATED',p['missing_model_response'])
        self.assertFalse(p['clinical_review_generated'])

    def test_plan_approval_fingerprint_mismatch_before_client(self):
        b,_,plan,bindings=fixture()
        with patch.object(two,'load_protocol',return_value=(b,plan,bindings)),patch.object(two,'ObservedAdapter') as client:
            with self.assertRaisesRegex(ValueError,'APPROVED_PLAN_MISMATCH'):
                two.collect('TEST_ONLY','TEST_ONLY',execute=True,allow_llm=True,accept_proxy=True,acknowledge_timeout=True,expected_plan_fingerprint='0'*64)
            client.assert_not_called()


if __name__=='__main__':unittest.main()
