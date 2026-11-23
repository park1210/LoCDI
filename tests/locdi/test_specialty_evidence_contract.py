"""TEST_ONLY source slots and response variants; no clinical judgments."""
from copy import deepcopy
from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
from app.locdi.specialty_evidence_contract import projection,validate_response,EvidenceScopeError,PARENT_ERROR
from app.locdi.sequential_handoff import run_sequential_handoff
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from test_sequential_handoff import HandoffClient,registry_fixture
from test_simulation import smoke_options
import app.locdi.live_handoff as live
from test_live_handoff import fake_backend


def payload():
    return {'observation':{'elapsed_minutes':3,'objective_findings':['TEST_ONLY_RELEASED_FINDING']},
        'evidence':['TEST_ONLY_RELEASED_FINDING','TEST_ONLY_HISTORY'],
        'evidence_records':[{'source_id':'finding:0','kind':'objective_finding','text':'TEST_ONLY_RELEASED_FINDING','supports_finalization':True},
            {'source_id':'history:0','kind':'patient_history','text':'TEST_ONLY_HISTORY','supports_finalization':False}],
        'supported_diagnosis_eligible_indices':[0]}


def response(indices,certainty='supported'):
    return {'assessment':'TEST_ONLY bounded opinion','certainty':certainty,'evidence_indices':indices,'completed':True}


class EvidenceClient(HandoffClient):
    def __init__(self,indices,certainty):super().__init__();self.indices=indices;self.certainty=certainty
    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        p=json.loads(user_prompt)
        if p['handoff_stage']=='SPECIALTY_ASSESSMENT':
            self.calls.append({'payload':p,'model':model,'system':system_prompt})
            return json.dumps(response(self.indices,self.certainty))
        return super().chat_json(system_prompt,user_prompt,model,schema=schema)


class SpecialtyEvidenceContracts(unittest.TestCase):
    def test_zero_based_table_preserves_source_and_valid_objective(self):
        p=payload();before=deepcopy(p);snapshot=projection(p)
        self.assertEqual(p,before);self.assertEqual(snapshot['evidence'],p['evidence'])
        self.assertEqual([r['index'] for r in snapshot['evidence_index_map']],[0,1])
        self.assertEqual(snapshot['evidence_contract']['allowed_indices'],[0,1])
        self.assertEqual(validate_response(response([0]),snapshot)['semantic_validation'],'VALID')

    def test_schema_valid_outside_and_one_based_last_index_rejected(self):
        snapshot=projection(payload())
        for indices in ([2],[99],[-1]):
            self.assertEqual(validate_response(response(indices),snapshot)['detail_code'],'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE')
        # In-range one-based intent is unknowable; never guess or remap indices.
        self.assertEqual(validate_response(response([1]),snapshot)['detail_code'],'SPECIALTY_CERTAINTY_EVIDENCE_CONTRADICTION')

    def test_supported_empty_requires_evidence_and_is_not_rewritten(self):
        value=response([]);before=deepcopy(value)
        result=validate_response(value,projection(payload()))
        self.assertEqual(result['detail_code'],'SPECIALTY_EVIDENCE_REQUIRED');self.assertEqual(value,before)

    def test_empty_list_unresolved_legal_supported_and_index_zero_illegal(self):
        p=payload();p.update(evidence=[],evidence_records=[],supported_diagnosis_eligible_indices=[]);p['observation']['objective_findings']=[]
        snapshot=projection(p)
        self.assertEqual(validate_response(response([],'unresolved'),snapshot)['semantic_validation'],'VALID')
        self.assertEqual(validate_response(response([]),snapshot)['detail_code'],'SPECIALTY_EVIDENCE_REQUIRED')
        self.assertEqual(validate_response(response([0]),snapshot)['detail_code'],'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE')

    def test_unresolved_can_cite_valid_released_history(self):
        self.assertEqual(validate_response(response([1],'unresolved'),projection(payload()))['semantic_validation'],'VALID')

    def test_future_evidence_or_not_observed_current_finding_blocked(self):
        for alteration in ('time','finding'):
            p=payload()
            if alteration=='time':p['evidence_records'][0]['available_minute']=4
            else:p['observation']['objective_findings']=[]
            with self.assertRaises(EvidenceScopeError):projection(p)

    def test_memory_private_source_kind_or_wrong_mapping_not_promoted(self):
        for alteration in ('memory','private','text','eligible','record_length'):
            p=payload()
            if alteration in ('memory','private'):p['evidence_records'][0]['kind']=alteration
            elif alteration=='text':p['evidence_records'][0]['text']='TEST_ONLY_WRONG_SOURCE'
            elif alteration=='eligible':p['supported_diagnosis_eligible_indices']=[1]
            else:p['evidence_records'].pop()
            with self.assertRaises(EvidenceScopeError):projection(p)

    def test_safe_audit_contains_no_evidence_text_or_original_indices(self):
        result=validate_response(response([0]),projection(payload()))
        self.assertNotIn('TEST_ONLY_RELEASED_FINDING',json.dumps(result));self.assertNotIn('evidence_indices',result)
        self.assertEqual(result['citation_count'],1)

    def execute(self,indices,certainty):
        case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0];client=EvidenceClient(indices,certainty)
        result=run_sequential_handoff(case,options=smoke_options(client,registry_manifest=registry_fixture()),
            test_only=True,routing='static_development',static_department='neurology')
        return result,client

    def test_real_core_schema_valid_semantic_failures_keep_parent_and_detail(self):
        for indices,certainty,detail in (([0],'unresolved','SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE'),
                ([], 'supported','SPECIALTY_EVIDENCE_REQUIRED')):
            result,client=self.execute(indices,certainty)
            self.assertEqual(result['status'],'failed');self.assertIn(PARENT_ERROR,result['blocker_codes'])
            row=result['audit']['requests'][1]
            self.assertEqual(row['schema_validation'],'VALID');self.assertEqual(row['status'],'response_received')
            self.assertEqual(row['specialty_evidence_validation']['detail_code'],detail)
            self.assertEqual(len(client.calls),2);self.assertFalse(any(r['stage']=='ER_REASSESSMENT' for r in result['audit']['requests']))

    def test_unresolved_empty_real_runner_scheduler_action_and_export_document(self):
        with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner,patch(
                'app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler:
            result,client=self.execute([],'unresolved')
        runner.assert_called_once();scheduler.assert_called_once();self.assertEqual(result['status'],'completed')
        self.assertEqual(result['audit']['handoffs'][0]['handoff_status'],'handoff_completed')
        initial,specialty,returned=[r['payload'] for r in client.calls[:3]]
        self.assertEqual(initial['observation'],specialty['observation']);self.assertEqual(initial['observation'],returned['observation'])
        self.assertNotIn('logical_specialty_assessment',specialty)
        self.assertEqual(returned['logical_specialty_assessment']['kind'],'MODEL_OPINION_NOT_SOURCE_RESULT')
        self.assertTrue(result['episode']['result']['action_counts'])
        self.assertEqual(initial['evidence'],returned['evidence'])
        self.assertNotIn(returned['logical_specialty_assessment']['assessment'],returned['evidence'])


class LiveSemanticEvidenceContracts(unittest.TestCase):
    def test_semantic_failure_checkpoint_and_final_audit_agree_no_raw_response(self):
        config=live.load_config();identifier=config['stage_bindings'][0]['model_identifier'];calls=[]
        original=HandoffClient.chat_json
        def variant(client,system,user,model=None,*,schema=None):
            if json.loads(user)['handoff_stage']=='SPECIALTY_ASSESSMENT':return json.dumps(response([0],'unresolved'))
            return original(client,system,user,model,schema=schema)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs')) as temporary,patch.object(live,'HANDOFF_ROOT',Path(temporary)/'TEST_ONLY'),patch(
                'app.locdi.llm_setup.inspect_local_model',return_value={'status':'READY','model_identifier':identifier}):
            check=live.preflight(run_id='b'*32)
            with fake_backend(calls,identifier),patch.object(HandoffClient,'chat_json',variant):
                result=live.run_live_handoff(run_id='b'*32,execute=True,allow_llm=True,accept_release_assumption=True,expected_plan_fingerprint=check['plan_fingerprint'])
            self.assertEqual(result['status'],'failed');self.assertEqual(len(calls),2)
            root=Path(temporary)/'TEST_ONLY'/('b'*32)
            schema=json.loads((root/'private/Request0002_after.json').read_text(encoding='utf-8'))
            semantic=json.loads((root/'private/Request0002_semantic.json').read_text(encoding='utf-8'))
            audit=json.loads((root/'reports/handoff.json').read_text(encoding='utf-8'))
            self.assertEqual(schema['request']['schema_validation'],'VALID')
            self.assertNotIn('specialty_evidence_validation',schema['request'])
            verdict=semantic['request']['specialty_evidence_validation']
            self.assertEqual(verdict['detail_code'],'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE')
            self.assertEqual(verdict,audit['requests'][1]['specialty_evidence_validation'])
            self.assertNotIn('TEST_ONLY bounded opinion',json.dumps(audit))
            self.assertNotIn('evidence_indices',json.dumps(semantic))

    def test_v1_prompt_is_preserved_and_v2_teaches_exact_scope(self):
        config=live.load_config();stage=config['stage_bindings'][1]
        self.assertEqual(stage['prompt'],'prompts/handoff/v2/specialty_assessment.txt')
        v1=(ROOT/'prompts/handoff/v1/specialty_assessment.txt').read_text(encoding='utf-8')
        self.assertIn('zero-based',v1)
        text=(ROOT/stage['prompt']).read_text(encoding='utf-8')
        for marker in ('evidence_index_map','supported_diagnosis_eligible_indices','evidence_indices=[]','Memory record positions'):
            self.assertIn(marker,text)


if __name__=='__main__':unittest.main()
