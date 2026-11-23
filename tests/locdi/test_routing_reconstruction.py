"""TEST_ONLY reconstruction/collector contracts; no clinical judgments/model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility
from app.locdi.dataset_freeze import load
from app.locdi.llm_setup import ROOT,digest
from app.locdi.routing_utility_reconstruction import (source_code_bindings,reference_facts,response_evidence,
    build_development,require_clinical_artifact,fit_verified,execute_conditions)
from app.locdi import routing_utility_pilot as pilot
from app.locdi.ollama_adapter import LLMResponseError,LLMTimeoutError
from app.llm.prompt_loader import specialty_system_prompt


def manifest():return load(ROOT/'configs/agent_registry.specialty_development.json')


def proxy_rows():
    rows=[]
    for i in range(2):
        for agent in manifest()['agents']:
            if agent['role']!='specialty':continue
            answer={'diagnosis':'TEST_ONLY','certainty':'supported'};source={'TEST_ONLY':True}
            rows.append({'patient_id':'TEST_ONLY_TRAIN_%d'%i,'decision_id':'TEST_ONLY_D%d'%i,
                'agent_id':agent['agent_id'],'model':agent['model'],'version':agent['version'],'split':'train',
                'reference_type':'TEST_ONLY','response_status':'SUPPORTED','proxy_correct':agent['department']=='neurology',
                'observation':{'chief_complaint':'TEST_ONLY_signal'},'source_scope_status':'REVIEWED_FOR_SOURCE_PROXY_ONLY',
                'scope_review_ref':'TEST_ONLY_NOT_A_REAL_REVIEW','reference_timing_status':'CURRENT_DECISION_SOURCE_PROXY_VALID',
                'source_start_after_decision':False,'response':answer,'response_fingerprint':digest(answer),
                'source_evidence':source,'source_fingerprint':digest(source)})
    return rows


def bundle():
    m=manifest();agents=[a for a in m['agents'] if a['role']=='specialty']
    payload={'observation':{'chief_complaint':'TEST_ONLY'},'diagnosis_catalog':['TEST_ONLY_A']}
    candidate={'case_ref':'Candidate001','shared_payload':payload,'shared_input_fingerprint':digest(payload),
        'visit_started_at':'2020-01-01T00:00:00Z','source_fingerprint':digest('TEST_ONLY'),
        'decision':{'patient_id':'TEST_ONLY_PATIENT','visit_id':'TEST_ONLY_VISIT','decision_id':'TEST_ONLY_DECISION','split':'train',
            'generation':{'prompt_fingerprints':{a['agent_id']:hashlib.sha256(specialty_system_prompt(a['department']).encode('utf8')).hexdigest() for a in agents}}}}
    b={'registry':m,'candidates':[candidate],'fingerprint':digest('TEST_ONLY_BUNDLE'),'assumptions':['TEST_ONLY_PROXY'],
        'llm_config':{'model':'qwen2.5:1.5b-instruct','model_identifier':'TEST_ONLY_MODEL','model_version':'TEST_ONLY',
            'max_output_tokens':1024,'timeout_seconds':120,'request_retries':0,'temperature':0,'seed':42},
        'response_schema':{'type':'object'}}
    return b,agents


class ReconstructionContracts(unittest.TestCase):
    def test_development_complete_groups_use_unchanged_core_score(self):
        a=build_development(proxy_rows(),manifest(),test_only=True)
        reg=AgentRegistry(manifest(),a)
        d=reg.route({'chief_complaint':'TEST_ONLY_signal'},[x['department'] for x in manifest()['agents'] if x['role']=='specialty'],mode='adaptive')
        self.assertEqual(d['selected_specialty'],'neurology')
        self.assertEqual(d['utility_estimates']['neurology']['score'],.75)
        self.assertFalse(a['clinical_fit_approved'])

    def test_serialization_fingerprint_roundtrip(self):
        a=build_development(proxy_rows(),manifest(),test_only=True);b=json.loads(json.dumps(a))
        reg=AgentRegistry(manifest())
        self.assertEqual(RoutingUtility(a,reg).fingerprint,RoutingUtility(b,reg).fingerprint)

    def test_unresolved_never_false_or_true(self):
        for value in (False,True):
            rows=proxy_rows();rows[0].update(response_status='UNRESOLVED',proxy_correct=value)
            with self.assertRaisesRegex(ValueError,'SUPPORTED_EXPLICIT'):build_development(rows,manifest(),test_only=True)

    def test_null_not_false(self):
        rows=proxy_rows();rows[0]['proxy_correct']=None
        with self.assertRaises(ValueError):build_development(rows,manifest(),test_only=True)

    def test_partial_group_not_filled_or_dropped(self):
        with self.assertRaisesRegex(ValueError,'COMPLETE_GROUP'):build_development(proxy_rows()[:-1],manifest(),test_only=True)

    def test_duplicate_response_not_selectively_replaced(self):
        rows=proxy_rows();rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):build_development(rows,manifest(),test_only=True)

    def test_eval_patient_overlap(self):
        with self.assertRaisesRegex(ValueError,'OVERLAP'):build_development(proxy_rows(),manifest(),evaluation_patients=['TEST_ONLY_TRAIN_0'],test_only=True)

    def test_version_mismatch(self):
        rows=proxy_rows();rows[0]['version']='TEST_ONLY_WRONG'
        with self.assertRaisesRegex(ValueError,'VERSION'):build_development(rows,manifest(),test_only=True)

    def test_response_and_source_hash_mismatch(self):
        for field in ('response_fingerprint','source_fingerprint'):
            rows=proxy_rows();rows[0][field]='BAD'
            with self.assertRaisesRegex(ValueError,'PROVENANCE'):build_development(rows,manifest(),test_only=True)

    def test_future_reference(self):
        rows=proxy_rows();rows[0]['source_start_after_decision']=True
        with self.assertRaises(ValueError):build_development(rows,manifest(),test_only=True)

    def test_same_snapshot_required(self):
        rows=proxy_rows();rows[0]['observation']['chief_complaint']='TEST_ONLY_OTHER'
        with self.assertRaisesRegex(ValueError,'COMPLETE_GROUP'):build_development(rows,manifest(),test_only=True)

    def test_min_samples_not_lowered(self):
        for n in (0,True,3):
            with self.assertRaises(ValueError):build_development(proxy_rows(),manifest(),min_samples=n,test_only=True)

    def test_no_real_proxy_rows_means_no_artifact(self):
        with self.assertRaisesRegex(ValueError,'NO_ELIGIBLE'):build_development([],manifest())

    def test_test_only_not_promoted_to_source_proxy(self):
        with self.assertRaises(ValueError):build_development(proxy_rows(),manifest())

    def test_development_not_clinical(self):
        a=build_development(proxy_rows(),manifest(),test_only=True)
        with self.assertRaisesRegex(ValueError,'NOT_CLINICAL'):require_clinical_artifact(a)

    def test_native_tie_and_insufficient_support(self):
        rows=proxy_rows()
        for r in rows:r['proxy_correct']=True
        a=build_development(rows,manifest(),test_only=True);reg=AgentRegistry(manifest(),a)
        deps=[r['department'] for r in manifest()['agents'] if r['role']=='specialty']
        d=reg.route({'chief_complaint':'TEST_ONLY_signal'},deps,mode='adaptive')
        self.assertEqual(d['decision_status'],'abstained');self.assertEqual(len(d['tied_departments']),5)
        d=reg.route({'chief_complaint':'TEST_ONLY_ZERO_OVERLAP'},deps,mode='adaptive')
        self.assertTrue(all(v is None for v in d['utility_estimates'].values()))

    def test_gold_cannot_enter_development_tokens(self):
        rows=proxy_rows();rows[0]['observation']['gold_department']='TEST_ONLY'
        with self.assertRaises(ValueError):build_development(rows,manifest(),test_only=True)

    def test_gate_blocked_never_calls_fit(self):
        from test_routing_correctness_gate import test_only_reviewed_data
        data,options=test_only_reviewed_data()
        with patch('app.core.routing_utility.fit_utility') as f:
            artifact,report=fit_verified(data,options)
        f.assert_not_called();self.assertIsNone(artifact);self.assertFalse(report['utility_fit_ready'])

    def test_existing_evaluator_rejects_test_only_even_if_marker_forged(self):
        from test_routing_correctness_gate import test_only_reviewed_data
        from app.locdi.routing_evaluation import validate_utility_provenance
        data,opts=test_only_reviewed_data()
        artifact=build_development(proxy_rows(),manifest(),test_only=True);artifact['reference_type']='INDEPENDENT_REVIEWED'
        with self.assertRaisesRegex(ValueError,'GATE_BLOCKED'):
            validate_utility_provenance(AgentRegistry(manifest(),artifact),{'data':data,'options':opts})

    def test_exact_code_fact_not_correctness(self):
        b,_=bundle();c=b['candidates'][0]
        ref={'declared_label':'TEST_ONLY_A','source_condition_rows':[{'source_record':{'PATIENT':'TEST_ONLY_PATIENT',
            'ENCOUNTER':'TEST_ONLY_VISIT','START':'2020-01-01','SYSTEM':'TEST_ONLY_SYSTEM','CODE':'TEST_ONLY_CODE','DESCRIPTION':'TEST_ONLY_A'}}]}
        facts=reference_facts(c,ref);binding=source_code_bindings([ref])
        answer={'diagnosis':'TEST_ONLY_A','certainty':'supported'}
        row={'agent_id':'TEST_ONLY_AGENT','department':'neurology','split':'train','model':'TEST_ONLY','version':'TEST_ONLY',
            'response':answer,'response_fingerprint':digest(answer),'shared_input_fingerprint':c['shared_input_fingerprint'],
            'execution_status':'SUCCEEDED','response_contract_status':'VALID'}
        e=response_evidence(row,c,facts,binding)
        self.assertTrue(e['exact_reference_code_match']);self.assertIsNone(e['correct']);self.assertIsNone(e['proxy_correct'])

    def test_source_description_alias_not_inferred(self):
        refs=[{'source_condition_rows':[{'source_record':{'SYSTEM':'TEST_ONLY','CODE':'A','DESCRIPTION':'TEST_ONLY_TERM'}}]}]
        self.assertNotIn('test_only_term',source_code_bindings(refs))


class FakeClient:
    invalid=False
    def __init__(self,cfg,authorized):
        assert authorized and cfg['max_requests']<=5
        self.request_count=self.identity_check_count=0;self.last_response_content=None
        self.last_observed_model_identifier='TEST_ONLY_MODEL'
    def chat_json(self,*args,**kwargs):
        self.request_count+=1;self.identity_check_count+=1
        if self.invalid:self.last_response_content='TEST_ONLY_INVALID_RAW';raise LLMResponseError('TEST_ONLY')
        self.last_response_content=json.dumps({'action':'diagnose','diagnosis':'TEST_ONLY_A','certainty':'unresolved','rationale':'TEST_ONLY'})
        return self.last_response_content
    def public_metadata(self):return {'request_count':self.request_count,'identity_check_count':self.identity_check_count}
    def close(self):pass


class PilotContracts(unittest.TestCase):
    def test_approval_required_before_constructor(self):
        b,_=bundle()
        with patch.object(pilot,'LocalOllamaAdapter') as client:
            with self.assertRaisesRegex(ValueError,'ACCEPTANCE'):pilot.collect_pilot(b,execute=True)
        client.assert_not_called()

    def test_dry_run_no_client(self):
        b,agents=bundle()
        with patch.object(pilot,'validate_preparation',return_value=agents),patch.object(pilot,'LocalOllamaAdapter') as client:
            r=pilot.collect_pilot(b,namespace='TEST_ONLY_DRY')
        self.assertEqual(r['status'],'DRY_RUN');client.assert_not_called()

    def test_max_five_enforced(self):
        b,agents=bundle()
        with patch.object(pilot,'validate_preparation',return_value=agents):
            for n in (0,6,True):
                with self.assertRaises(ValueError):pilot.plan_pilot(b,n)

    def execute(self,root,cap=5):
        b,agents=bundle()
        with patch.object(pilot,'ROOT',Path(root)),patch.object(pilot,'validate_preparation',return_value=agents),patch.object(pilot,'LocalOllamaAdapter',FakeClient):
            return pilot.collect_pilot(b,namespace='TEST_ONLY',run_id='TEST_ONLY_PILOT',execute=True,allow_llm=True,accept_proxy=True,max_requests=cap)

    def test_real_shaped_five_checkpoints_no_correctness(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as root:
            r=self.execute(root);self.assertEqual(r['generation_requests'],5);self.assertEqual(r['unresolved'],5)
            private=Path(root)/'outputs/private/research_evaluation/TEST_ONLY/live_pilot/TEST_ONLY_PILOT'
            self.assertEqual(len(list(private.glob('Request*_before.json'))),5)
            for p in private.glob('Request*_after.json'):
                self.assertIsNone(load(p)['correct']);self.assertIsNotNone(load(p)['raw_response'])

    def test_budget_not_reset_between_agents(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as root:
            r=self.execute(root,1);self.assertEqual(r['generation_requests'],1);self.assertEqual(r['missing_agents'],4)

    def test_invalid_response_raw_preserved_no_retry(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as root,patch.object(FakeClient,'invalid',True):
            r=self.execute(root);self.assertEqual(r['generation_requests'],1);self.assertEqual(r['errors'],1)
            p=next((Path(root)/'outputs/private').rglob('Request001_after.json'))
            self.assertEqual(load(p)['raw_response'],'TEST_ONLY_INVALID_RAW')
            self.assertEqual(r['received_model_responses'],1);self.assertEqual(r['schema_valid_responses'],0)

    def test_timeout_attempt_is_not_received_model_response(self):
        def timeout(client,*args,**kwargs):
            client.request_count+=1;client.identity_check_count+=1;client.last_response_content=None
            raise LLMTimeoutError('TEST_ONLY')
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as root,patch.object(FakeClient,'chat_json',timeout):
            r=self.execute(root)
            self.assertEqual(r['generation_requests'],1);self.assertEqual(r['stored_attempt_records'],1)
            self.assertEqual(r['received_model_responses'],0);self.assertEqual(r['schema_valid_responses'],0)

    def test_no_resume_or_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as root:
            self.execute(root)
            with self.assertRaisesRegex(ValueError,'ALREADY_USED'):self.execute(root)

    def test_before_checkpoint_failure_prevents_generation(self):
        original=pilot.atomic_json;b,agents=bundle()
        def fail(path,doc):
            if path.name.endswith('_before.json'):raise OSError('TEST_ONLY_STORAGE')
            original(path,doc)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as root,patch.object(pilot,'ROOT',Path(root)),patch.object(pilot,'validate_preparation',return_value=agents),patch.object(pilot,'atomic_json',side_effect=fail),patch.object(pilot,'LocalOllamaAdapter',FakeClient),patch.object(FakeClient,'chat_json') as chat:
            with self.assertRaises(OSError):pilot.collect_pilot(b,namespace='TEST_ONLY',execute=True,allow_llm=True,accept_proxy=True)
            chat.assert_not_called()


if __name__=='__main__':unittest.main()
