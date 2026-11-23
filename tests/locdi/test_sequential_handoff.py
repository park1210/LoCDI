"""TEST_ONLY sequential calls; all patient evidence comes from existing sources."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.locdi.case_adapter import adapt_case
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT
from app.locdi.sequential_handoff import run_sequential_handoff,SequentialClient,HandoffStopped
from test_simulation import SourceActionClient,smoke_options
from test_routing_voi_readiness import test_only_registry,test_only_routing
from smoke_support import check_boundary


def registry_fixture():
    registry=test_only_registry()
    for a in registry['agents']:a['model']='TEST_ONLY_'+a['agent_id']
    return registry


class HandoffClient(SourceActionClient):
    def __init__(self,mode=None):
        super().__init__();self.mode=mode;self.repair_sent=False

    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        p=json.loads(user_prompt);stage=p['handoff_stage']
        self.calls.append({'payload':p,'model':model,'system':system_prompt})
        if stage=='ER_INITIAL_DECISION':
            decision=None
            if self.mode=='no_handoff':decision=json.loads(SourceActionClient().chat_json(system_prompt,user_prompt,model,schema=schema['properties']['decision']['anyOf'][0]))
            return json.dumps({'handoff_required':self.mode!='no_handoff','request_reason':'TEST_ONLY logical review request','decision':decision})
        if stage=='SPECIALTY_ASSESSMENT':
            if self.mode=='timeout':raise TimeoutError('TEST_ONLY private error evidence')
            if self.mode=='client_error':raise ConnectionError('TEST_ONLY private error evidence')
            if self.mode=='schema_invalid':return '{bad-json'
            return json.dumps({'assessment':'TEST_ONLY assessment: insufficient evidence; no new clinical fact.',
                'certainty':'unresolved','evidence_indices':[],'completed':self.mode!='unfinished'})
        raw=SourceActionClient().chat_json(system_prompt,user_prompt,model,schema=schema)
        if self.mode=='repair' and not self.repair_sent:
            self.repair_sent=True;r=json.loads(raw);r['rationale']='';return json.dumps(r)
        if self.mode=='invalid_action':
            r=json.loads(raw);r['target']='TEST_ONLY_UNAVAILABLE_TARGET';return json.dumps(r)
        return raw


class SequentialHandoffContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]

    def execute(self,mode=None,**kwargs):
        client=HandoffClient(mode);registry=registry_fixture()
        result=run_sequential_handoff(self.case,options=smoke_options(client,registry_manifest=registry),
            test_only=True,routing=kwargs.pop('routing','static_development'),
            static_department=kwargs.pop('static_department','neurology'),**kwargs)
        return result,client

    def test_separate_three_calls_real_runner_scheduler_core_action_and_terminal(self):
        with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner, \
             patch('app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler, \
             patch('app.core.hospital_llm_policy.OllamaClient',side_effect=AssertionError('NO_REAL_MODEL')):
            result,client=self.execute()
        self.assertEqual(result['status'],'completed');self.assertEqual(runner.call_count,1);self.assertEqual(scheduler.call_count,1)
        audit=result['audit'];self.assertEqual([r['stage'] for r in audit['requests'][:3]],['ER_INITIAL_DECISION','SPECIALTY_ASSESSMENT','ER_REASSESSMENT'])
        self.assertNotEqual(client.calls[0]['model'],client.calls[1]['model']);self.assertEqual(client.calls[0]['model'],client.calls[2]['model'])
        self.assertEqual(audit['request_count'],len(client.calls));self.assertEqual(audit['source_consult_count'],0)
        self.assertEqual(audit['handoffs'][0]['handoff_status'],'handoff_completed')
        self.assertTrue(result['episode']['result']['diagnosis_unresolved'])
        self.assertTrue(check_boundary(self.case,adapt_case(self.case),result['raw'],client)['release'])

    def test_scope_snapshot_only_completed_response_and_same_clock(self):
        result,client=self.execute();a,b,c=[x['payload'] for x in client.calls[:3]]
        self.assertEqual(a['observation'],b['observation']);self.assertEqual(a['observation'],c['observation'])
        self.assertNotIn('logical_specialty_assessment',a);self.assertNotIn('logical_specialty_assessment',b)
        self.assertIn('logical_specialty_assessment',c)
        h=result['audit']['handoffs'][0];self.assertEqual(h['request_minute'],h['response_minute'])
        self.assertEqual([r['minute'] for r in result['audit']['requests'][:3]],[h['request_minute']]*3)
        self.assertTrue(all(r['cutoff_compliant'] for r in result['raw']['telemetry']['retrievals']))

    def test_timeout_invalid_schema_and_incomplete_response_stop_without_er_return(self):
        for mode,reason in [('timeout','TIMEOUT'),('schema_invalid','SCHEMA_INVALID'),('unfinished','SCHEMA_INVALID'),('client_error','CLIENT_ERROR')]:
            with self.subTest(mode=mode):
                result,client=self.execute(mode);self.assertEqual(result['status'],'failed')
                self.assertIn(reason,result['blocker_codes']);self.assertEqual(len(client.calls),2)
                self.assertEqual(result['audit']['handoffs'][0]['handoff_status'],'handoff_failed')

    def test_global_budget_one_two_and_subsequent_er_calls_share_limit(self):
        for budget in (1,2,3):
            with self.subTest(budget=budget):
                result,client=self.execute(budget=budget);self.assertEqual(result['status'],'failed')
                self.assertEqual(len(client.calls),budget);self.assertIn('GLOBAL_REQUEST_BUDGET_EXHAUSTED',result['blocker_codes'])

    def test_core_repair_is_counted_does_not_repeat_escalation(self):
        result,client=self.execute('repair',budget=11)
        self.assertEqual(result['status'],'completed')
        rows=result['audit']['requests'];self.assertEqual(sum(r['repair'] for r in rows),1)
        self.assertEqual(result['audit']['handoffs'][0]['handoff_status'],'handoff_completed')
        self.assertTrue(next(r for r in rows if r['repair'])['repair_of_request_id'])
        self.assertEqual(sum(r['stage']=='SPECIALTY_ASSESSMENT' for r in rows),1)
        self.assertEqual(len(rows),len(client.calls))
        limited,_=self.execute('repair',budget=3)
        self.assertEqual(limited['status'],'failed');self.assertEqual(limited['audit']['request_count'],3)

    def test_adaptive_missing_utility_blocked_and_toy_abstention_not_selected(self):
        blocked,client=self.execute(routing='adaptive',static_department=None)
        self.assertEqual(blocked['status'],'blocked');self.assertEqual(client.calls,[])
        utility=test_only_routing(registry_fixture());utility['min_samples']=999
        result,client=self.execute(routing='adaptive',static_department=None,routing_utility=utility)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(client.calls),1)
        self.assertEqual(result['audit']['routing'][0]['status'],'abstained')
        self.assertIsNone(result['audit']['handoffs'][0]['target_department'])

    def test_toy_adaptive_selection_uses_registry_and_not_gold(self):
        utility=test_only_routing(registry_fixture())
        for row in utility['samples']:row['tokens']=['pain']
        result,client=self.execute(routing='adaptive',static_department=None,routing_utility=utility)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['audit']['routing'][0]['selected_specialty'],'neurology')

    def test_disabled_unavailable_and_no_handoff_are_distinct(self):
        result,client=self.execute(routing='disabled',static_department=None)
        self.assertIn('ROUTING_DISABLED',result['blocker_codes']);self.assertEqual(len(client.calls),1)
        unavailable,client=self.execute(static_department='dermatology')
        self.assertEqual(unavailable['status'],'blocked');self.assertEqual(client.calls,[])
        normal,client=self.execute('no_handoff')
        self.assertEqual(normal['status'],'completed');self.assertEqual(normal['audit']['handoffs'],[])
        self.assertFalse(any(c['payload']['handoff_stage']=='SPECIALTY_ASSESSMENT' for c in client.calls))

    def test_gold_future_memory_and_injected_specialty_response_block_before_request(self):
        _,client=self.execute();base=deepcopy(client.calls[0]['payload'])
        for key in ('gold','reference','source_event_store','logical_specialty_assessment'):
            payload=deepcopy(base);payload[key]={}
            transport=HandoffClient();c=self.direct_client(transport)
            with self.assertRaises(HandoffStopped):c.chat_json('TEST_ONLY',json.dumps(payload),model=registry_fixture()['agents'][0]['model'],schema={})
            self.assertEqual(transport.calls,[])
        payload=deepcopy(base);payload['observation']['longitudinal_memory']['records'][0]['available_at']='2099-01-01T00:00:00Z'
        transport=HandoffClient();c=self.direct_client(transport)
        with self.assertRaises(HandoffStopped):c.chat_json('TEST_ONLY',json.dumps(payload),model=registry_fixture()['agents'][0]['model'],schema={})
        self.assertEqual(transport.calls,[])

    def direct_client(self,transport):
        return SequentialClient(transport,AgentRegistry(registry_fixture()),episode_id='TEST_ONLY_EPISODE',
            patient_id=self.case['patient_id'],visit_id=self.case['agent_visible']['index_encounter_id'],
            cutoff=self.case['agent_visible']['index_encounter_start'],budget=10)

    def test_future_specialty_completion_is_rejected(self):
        from app.locdi.llm_setup import digest
        _,recorded=self.execute();payload=deepcopy(recorded.calls[0]['payload'])
        transport=HandoffClient();client=self.direct_client(transport);client.initial_done=True
        client._completed={'observation_fingerprint':digest(payload['observation']),
            'memory_fingerprint':digest(payload['observation']['longitudinal_memory']),
            'minute':payload['observation']['elapsed_minutes']+1}
        with self.assertRaises(HandoffStopped):client.chat_json('TEST_ONLY',json.dumps(payload),model=registry_fixture()['agents'][0]['model'],schema={})
        self.assertEqual(client.failure,'FUTURE_SPECIALTY_RESPONSE');self.assertEqual(transport.calls,[])

    def test_original_action_validator_not_bypassed(self):
        result,_=self.execute('invalid_action')
        self.assertEqual(result['status'],'failed');self.assertEqual(result['episode']['result']['action_counts'],{})
        self.assertEqual(result['audit']['handoffs'][0]['handoff_status'],'handoff_failed')

    def test_export_correlation_digests_privacy_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT/'tests')) as directory:
            result,_=self.execute(output_root=directory,export_id='TEST_ONLY_SEQ')
            audit=result['audit'];self.assertEqual(audit['export_status'],'EXPORTED')
            data=json.loads((Path(directory)/'reports/TEST_ONLY_SEQ_handoff.json').read_text(encoding='utf-8'))
            self.assertEqual(data,audit)
            self.assertNotIn(self.case['patient_id'],json.dumps(data));self.assertNotIn('TEST_ONLY assessment',json.dumps(data))
            self.assertTrue(all(r['pinned_model_digest']==r['observed_server_digest']=='NOT_RECORDED' for r in data['requests']))
            linked=[r for r in audit['requests'] if r['action_id']]
            self.assertEqual(len(linked),len(result['episode']['workflow']['decisions']))
            self.assertTrue(all(r['execution_event_sequences'] for r in linked))
            again,_=self.execute(output_root=directory,export_id='TEST_ONLY_SEQ')
            self.assertEqual(again['status'],'export_failed')
            self.assertEqual(data,json.loads((Path(directory)/'reports/TEST_ONLY_SEQ_handoff.json').read_text(encoding='utf-8')))

    def test_explicit_offline_boundary_and_determinism(self):
        client=HandoffClient();opts=smoke_options(client,registry_manifest=registry_fixture())
        self.assertEqual(run_sequential_handoff(self.case,options=opts)['status'],'blocked');self.assertEqual(client.calls,[])
        a,_=self.execute();b,_=self.execute()
        self.assertEqual(a['episode']['normalized_trace_fingerprint'],b['episode']['normalized_trace_fingerprint'])


if __name__=='__main__':unittest.main()
