"""TEST_ONLY semantic repair; no socket, clinical reference or verdict generation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.locdi.llm_setup import ROOT
from app.locdi.sequential_handoff import run_sequential_handoff,SequentialClient,HandoffStopped
from app.locdi.specialty_evidence_contract import projection,repair_payload,validate_response,REPAIRABLE_CODES
from app.core.agent_registry import AgentRegistry
import app.locdi.live_handoff as live
from test_sequential_handoff import HandoffClient,registry_fixture
from test_simulation import smoke_options
from test_specialty_evidence_contract import payload,response
from test_live_handoff import fake_backend


class RepairClient(HandoffClient):
    def __init__(self,mode='success'):
        super().__init__();self.variant=mode;self.specialty_count=0

    def chat_json(self,system,user,model=None,*,schema=None):
        p=json.loads(user)
        if p['handoff_stage']=='SPECIALTY_ASSESSMENT':
            self.calls.append({'payload':p,'system':system,'model':model})
            self.specialty_count+=1
            if self.specialty_count==1:
                return json.dumps(response([] if self.variant=='required' else [0,1,2,3]))
            if self.variant=='schema':return json.dumps({'certainty':'unresolved'})
            if self.variant=='timeout':raise TimeoutError('TEST_ONLY private transport error')
            if self.variant=='failure':return json.dumps(response([0]))
            return json.dumps(response([],'unresolved'))
        return super().chat_json(system,user,model,schema=schema)


class SpecialtySemanticRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=live.load_cases(live.DATASET)[0]

    def execute(self,mode='success',budget=10,**kwargs):
        client=RepairClient(mode)
        with patch('socket.create_connection',side_effect=AssertionError('NO_NETWORK')):
            result=run_sequential_handoff(self.case,options=smoke_options(client,registry_manifest=registry_fixture()),
                test_only=True,routing='static_development',static_department='neurology',budget=budget,
                specialty_semantic_repairs=1,**kwargs)
        return result,client

    def test_empty_evidence_explicit_contract_and_no_answer_rewriting(self):
        p=payload();p.update(evidence=[],evidence_records=[],supported_diagnosis_eligible_indices=[])
        s=projection(p);self.assertFalse(s['evidence_contract']['supported_permitted'])
        self.assertEqual(s['evidence_contract']['permitted_certainties'],['unresolved'])
        good=response([],'unresolved');self.assertEqual(validate_response(good,s)['semantic_validation'],'VALID')
        bad=response([0]);before=deepcopy(bad)
        self.assertEqual(validate_response(bad,s)['detail_code'],'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE')
        self.assertEqual(before,bad)

    def test_repair_success_preserves_both_attempts_and_er_return(self):
        r,c=self.execute();self.assertEqual(r['status'],'completed')
        rows=r['audit']['requests'];self.assertEqual([x['stage'] for x in rows[:4]],
            ['ER_INITIAL_DECISION','SPECIALTY_ASSESSMENT','SPECIALTY_ASSESSMENT','ER_REASSESSMENT'])
        self.assertEqual(rows[1]['specialty_evidence_validation']['semantic_validation'],'INVALID')
        self.assertEqual(rows[2]['specialty_evidence_validation']['semantic_validation'],'VALID')
        self.assertTrue(rows[2]['repair']);self.assertFalse(rows[2]['retry'])
        self.assertEqual(rows[2]['repair_of_request_id'],rows[1]['request_id'])
        self.assertEqual(rows[2]['repair_kind'],'SPECIALTY_SEMANTIC')
        self.assertEqual(r['audit']['specialty_semantic_repair_count'],1)
        self.assertEqual(r['audit']['request_count'],len(c.calls))
        h=r['audit']['handoffs'][0];self.assertEqual(h['specialty_response_id'],rows[2]['decision_id'])
        self.assertEqual(h['return_to_er_decision_id'],rows[3]['decision_id'])
        self.assertEqual(h['handoff_status'],'handoff_completed')

    def test_repair_input_only_adds_safe_code_and_ranges_snapshot_identical(self):
        r,c=self.execute();original=c.calls[1]['payload'];fixed=c.calls[2]['payload']
        expected=deepcopy(original);expected['specialty_semantic_repair']={
            'error_code':'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE','allowed_evidence_indices':[],
            'supported_eligible_indices':[]}
        self.assertEqual(fixed,expected)
        self.assertEqual(original['observation'],c.calls[3]['payload']['observation'])
        self.assertEqual(original['evidence'],[])
        self.assertNotIn('TEST_ONLY bounded opinion',json.dumps(fixed))
        self.assertEqual(c.calls[1]['system'],c.calls[2]['system'])
        self.assertEqual(c.calls[1]['model'],c.calls[2]['model'])

    def test_required_error_is_repaired(self):
        r,c=self.execute('required');self.assertEqual(r['status'],'completed')
        self.assertEqual(c.calls[2]['payload']['specialty_semantic_repair']['error_code'],'SPECIALTY_EVIDENCE_REQUIRED')

    def test_three_error_codes_only_and_contradiction_feedback(self):
        s=projection(payload());s['supported_diagnosis_eligible_indices']=[]
        for code in REPAIRABLE_CODES:
            repaired=repair_payload(s,{'detail_code':code})
            self.assertEqual(repaired['specialty_semantic_repair']['error_code'],code)
            self.assertEqual(repaired['evidence'],s['evidence'])
        with self.assertRaises(ValueError):repair_payload(s,{'detail_code':'SPECIALTY_EVIDENCE_SCOPE_MISMATCH'})

    def test_second_semantic_failure_stops_after_one_repair(self):
        r,c=self.execute('failure');self.assertEqual(r['status'],'failed')
        self.assertIn('SPECIALTY_EVIDENCE_CONTRACT_ERROR',r['blocker_codes'])
        self.assertEqual(len(c.calls),3);self.assertEqual(c.specialty_count,2)
        self.assertFalse(any(x['stage']=='ER_REASSESSMENT' for x in r['audit']['requests']))
        self.assertEqual(r['audit']['handoffs'][0]['handoff_status'],'handoff_failed')

    def test_repair_response_schema_and_transport_are_not_repaired_again(self):
        for mode,code in [('schema','SCHEMA_INVALID'),('timeout','TIMEOUT')]:
            r,c=self.execute(mode);self.assertEqual(r['status'],'failed')
            self.assertIn(code,r['blocker_codes']);self.assertEqual(len(c.calls),3)
            self.assertNotIn('private transport error',json.dumps(r['audit']))

    def test_repair_budget_exhausted_before_transport(self):
        r,c=self.execute(budget=2);self.assertEqual(len(c.calls),2)
        self.assertIn('GLOBAL_REQUEST_BUDGET_EXHAUSTED',r['blocker_codes'])
        self.assertEqual(r['audit']['specialty_semantic_repair_count'],0)
        self.assertEqual(r['audit']['handoffs'][0]['handoff_status'],'handoff_failed')

    def test_repair_uses_last_budget_slot_no_er_return_over_budget(self):
        r,c=self.execute(budget=3);self.assertEqual(len(c.calls),3)
        self.assertEqual(r['audit']['specialty_semantic_repair_count'],1)
        self.assertIn('GLOBAL_REQUEST_BUDGET_EXHAUSTED',r['blocker_codes'])

    def test_memory_id_is_not_evidence_and_future_source_blocked(self):
        p=payload();p['evidence_records'][0]['kind']='memory'
        with self.assertRaises(ValueError):projection(p)
        p=payload();p['evidence_records'][0]['available_minute']=4
        with self.assertRaises(ValueError):projection(p)
        s=projection(payload())
        self.assertEqual(validate_response(response(['TEST_ONLY_MEMORY_ID']),s)['detail_code'],'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE')

    def test_gold_and_private_source_and_future_memory_before_requests(self):
        _,captured=self.execute();base=captured.calls[0]['payload']
        for forbidden in ('gold','source_event_store','reference','provenance','future_memory'):
            p=deepcopy(base)
            if forbidden=='future_memory':p['observation']['longitudinal_memory']['records'][0]['available_at']='2099-01-01T00:00:00Z'
            else:p[forbidden]={}
            transport=HandoffClient();engine=SequentialClient(transport,AgentRegistry(registry_fixture()),
                episode_id='TEST_ONLY',patient_id=self.case['patient_id'],visit_id=self.case['agent_visible']['index_encounter_id'],
                cutoff=self.case['agent_visible']['index_encounter_start'],budget=10,specialty_semantic_repairs=1)
            with self.assertRaises(HandoffStopped):engine.chat_json('TEST_ONLY',json.dumps(p),model=registry_fixture()['agents'][0]['model'],schema={})
            self.assertEqual(transport.calls,[])

    def test_episode_audit_and_saved_summary_match(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs')) as d:
            r,c=self.execute(output_root=d,export_id='TEST_ONLY_REPAIR')
            audit=json.loads((Path(d)/'reports/TEST_ONLY_REPAIR_handoff.json').read_text(encoding='utf-8'))
            episode=json.loads((Path(d)/'episodes/TEST_ONLY_REPAIR.json').read_text(encoding='utf-8'))
            summary=json.loads((Path(d)/'metrics/TEST_ONLY_REPAIR.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(audit,r['audit']);self.assertEqual(episode,r['episode'])
            self.assertEqual(summary['episodes'][0]['configuration_fingerprint'],episode['configuration_fingerprint'])
            self.assertEqual(sum(bool(x['action_id']) for x in audit['requests']),len(episode['workflow']['decisions']))
            self.assertNotIn(self.case['patient_id'],json.dumps(audit))

    def test_v3_plan_explicit_opt_in_old_v2_preserved(self):
        config=live.load_config(ROOT/'configs/development_handoff_empty_evidence_v3.json')
        # The real 02903 run is preserved. A fresh-plan test must not assume
        # its output directory is unused or weaken the no-overwrite guard.
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs')) as directory,patch.object(live,'HANDOFF_ROOT',Path(directory)/'TEST_ONLY'),patch('app.locdi.ollama_adapter._backend') as backend:
            p=live.preflight(run_id='a'*32,config=config)
            used=Path(directory)/'TEST_ONLY'/('a'*32);used.mkdir(parents=True)
            blocked=live.preflight(run_id='a'*32,config=config)
        self.assertEqual(p['blocker_codes'],[]);backend.assert_not_called()
        self.assertIn('RUN_ID_OR_OUTPUT_ALREADY_USED',blocked['blocker_codes'])
        self.assertEqual(p['plan']['specialty_semantic_repair']['max_attempts'],1)
        self.assertEqual(p['plan']['generation']['max_requests'],10)
        for invalid in (2,-1,True):
            c=deepcopy(config);c['specialty_semantic_repairs']=invalid
            self.assertIn('HANDOFF_CONFIGURATION_INVALID',live.preflight(run_id='a'*32,config=c)['blocker_codes'])
        self.assertNotIn('specialty_semantic_repairs',live.load_config())

    def test_live_adapter_test_only_repair_checkpoint_and_counter(self):
        config=live.load_config(ROOT/'configs/development_handoff_empty_evidence_v3.json')
        identifier=config['stage_bindings'][0]['model_identifier'];calls=[];original=HandoffClient.chat_json
        def variant(client,system,user,model=None,*,schema=None):
            p=json.loads(user)
            if p['handoff_stage']=='SPECIALTY_ASSESSMENT':
                return json.dumps(response([],'unresolved') if 'specialty_semantic_repair' in p else response([0,1,2,3]))
            return original(client,system,user,model,schema=schema)
        with tempfile.TemporaryDirectory(dir=str(ROOT/'outputs')) as d,patch.object(live,'HANDOFF_ROOT',Path(d)/'TEST_ONLY'),patch(
                'app.locdi.llm_setup.inspect_local_model',return_value={'status':'READY','model_identifier':identifier}):
            check=live.preflight(run_id='a'*32,config=config)
            with fake_backend(calls,identifier),patch.object(HandoffClient,'chat_json',variant),patch('socket.create_connection',side_effect=AssertionError('NO_NETWORK')):
                r=live.run_live_handoff(run_id='a'*32,config=config,execute=True,allow_llm=True,accept_release_assumption=True,expected_plan_fingerprint=check['plan_fingerprint'])
            self.assertEqual(r['status'],'completed');self.assertEqual(r['llm_execution']['request_count'],len(calls))
            self.assertEqual(r['llm_execution']['specialty_semantic_repair_count'],1)
            root=Path(d)/'TEST_ONLY'/('a'*32)
            audit=json.loads((root/'reports/handoff.json').read_text(encoding='utf-8'))
            for i in (2,3):
                saved=json.loads((root/('private/Request%04d_semantic.json'%i)).read_text(encoding='utf-8'))
                self.assertEqual(saved['request']['specialty_evidence_validation'],audit['requests'][i-1]['specialty_evidence_validation'])
            self.assertFalse(any(x['retry'] for x in audit['requests']))


if __name__=='__main__':unittest.main()
