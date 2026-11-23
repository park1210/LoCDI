"""TEST_ONLY backend exercises real adapter, orchestration and Core. No network."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app.locdi.live_handoff as live
from app.locdi.llm_setup import ROOT
from app.locdi.run_simulation import OfflineClient
from app.locdi.ollama_adapter import LocalOllamaAdapter
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from scripts.run_controlled_handoff_pilot import main
from test_sequential_handoff import HandoffClient
from test_first_real_run import keys, strings


def fake_backend(calls, identifier, mode=None, mismatch=False):
    """Only test code can select this transport; no config switch exists."""
    source=HandoffClient(mode)
    class HTTP:
        def get(self,url):
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'models':[
                {'name':'qwen2.5:1.5b-instruct','digest':'sha256:'+'0'*64 if mismatch else identifier}]})
        def close(self):pass
    class Legacy:
        def chat_json(self,system,user,model=None,*,schema=None):
            payload=json.loads(user);calls.append({'payload':payload,'system':system,'model':model,'schema':schema})
            if mode=='er_return_failure' and payload['handoff_stage']=='ER_REASSESSMENT':
                raise ConnectionError('TEST_ONLY PRIVATE ERROR EVIDENCE')
            return source.chat_json(system,user,model=model,schema=schema)
    http=HTTP()
    return patch('app.locdi.ollama_adapter._backend',return_value=(Legacy(),http,http,http))


class LiveHandoffContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=live.load_cases(live.DATASET)[0]
        cls.definition=live.load_config()
        cls.identifier=cls.definition['stage_bindings'][0]['model_identifier']

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=str(ROOT/'outputs'))
        self.addCleanup(self.directory.cleanup);self.addCleanup(patch.stopall)
        self.root=Path(self.directory.name)/'handoff'
        patch.object(live,'HANDOFF_ROOT',self.root).start()
        patch.object(live,'load_cases',return_value=[self.case]).start()
        patch('app.locdi.llm_setup.inspect_local_model',return_value={'status':'READY','model_identifier':self.identifier}).start()
        self.run_id='d'*32;self.config=deepcopy(self.definition);self.calls=[]

    def execute(self,mode=None,mismatch=False):
        check=live.preflight(run_id=self.run_id,config=self.config)
        with fake_backend(self.calls,self.identifier,mode,mismatch),patch('socket.create_connection',side_effect=AssertionError('NO_NETWORK')):
            return live.run_live_handoff(run_id=self.run_id,config=self.config,execute=True,allow_llm=True,
                accept_release_assumption=True,expected_plan_fingerprint=check['plan_fingerprint'])

    def saved(self,name):
        return json.loads((self.root/self.run_id/name).read_text(encoding='utf-8'))

    def test_default_cli_and_api_never_create_client_backend_or_output(self):
        with patch.object(live,'LiveSequentialClient') as client,patch('app.locdi.ollama_adapter._backend') as backend:
            result=live.run_live_handoff(run_id=self.run_id,config=self.config)
            output=io.StringIO()
            with redirect_stdout(output):self.assertEqual(main(['--run-id',self.run_id]),0)
        self.assertEqual(result['status'],'dry_run');self.assertEqual(result['blocker_codes'],[])
        self.assertEqual(result['request_count'],0);self.assertEqual(result['server_availability'],'NOT_CHECKED')
        self.assertEqual(result['plan']['generation']['max_requests'],10)
        client.assert_not_called();backend.assert_not_called();self.assertFalse(self.root.exists())
        self.assertEqual(json.loads(output.getvalue())['status'],'dry_run')

    def test_explicit_approval_and_fingerprint_required(self):
        for flags in ({'execute':True},{'execute':True,'allow_llm':True},{'execute':1}):
            with patch.object(live,'LiveSequentialClient') as client:
                result=live.run_live_handoff(run_id=self.run_id,config=self.config,**flags)
            self.assertEqual(result['status'],'blocked');client.assert_not_called()
        with patch.object(live,'LiveSequentialClient') as client:
            r=live.run_live_handoff(run_id=self.run_id,config=self.config,execute=True,allow_llm=True,
                accept_release_assumption=True,expected_plan_fingerprint='0'*64)
        self.assertIn('APPROVED_PLAN_FINGERPRINT_MISMATCH',r['blocker_codes']);client.assert_not_called()
        with redirect_stderr(io.StringIO()),patch('scripts.run_controlled_handoff_pilot.load_config') as load:
            with self.assertRaises(SystemExit):main(['--run-id',self.run_id,'--execute'])
        load.assert_not_called()

    def test_real_type_is_separate_from_original_offline_client(self):
        check=live.preflight(run_id=self.run_id,config=self.config)
        client=live.LiveSequentialClient(self.config,approved_plan=check['plan'],authorized=True,output_root=self.root/self.run_id)
        self.addCleanup(client.close)
        self.assertIsInstance(client,LocalOllamaAdapter);self.assertNotIsInstance(client,OfflineClient)
        with self.assertRaises(ValueError):live.LiveSequentialClient(self.config,approved_plan=check['plan'],output_root=self.root)

    def test_three_stages_use_real_core_distinct_bindings_and_prompts(self):
        with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner,patch(
                'app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler:
            result=self.execute()
        self.assertEqual(result['status'],'completed');runner.assert_called_once();scheduler.assert_called_once()
        audit=self.saved('reports/handoff.json');rows=audit['requests']
        self.assertEqual([r['stage'] for r in rows[:3]],list(live.STAGES))
        self.assertEqual([r['agent_id'] for r in rows[:3]],['dev-shared-er','dev-shared-neurology','dev-shared-er'])
        self.assertEqual(len(set(r['prompt_fingerprint'] for r in rows[:3])),3)
        self.assertEqual(audit['request_count'],len(self.calls));self.assertEqual(audit['identity_check_count'],len(self.calls))
        self.assertTrue(result['sequential_handoff_verified']);self.assertEqual(audit['source_consult_count'],0)
        self.assertEqual(result['clinical_metrics'],'NOT_EVALUATED');self.assertFalse(result['research_evaluation_ready'])
        self.assertTrue(all(r['observed_server_digest']==self.identifier for r in rows))
        semantic_count=sum('specialty_evidence_validation' in r for r in rows)
        self.assertEqual(len(list((self.root/self.run_id/'private').glob('*.json'))),2*len(rows)+semantic_count)

    def test_snapshots_memory_clock_and_completed_opinion_boundary(self):
        self.execute();a,b,c=[r['payload'] for r in self.calls[:3]]
        self.assertEqual(a['observation'],b['observation']);self.assertEqual(a['observation'],c['observation'])
        self.assertNotIn('logical_specialty_assessment',a);self.assertNotIn('logical_specialty_assessment',b)
        self.assertEqual(c['logical_specialty_assessment']['kind'],'MODEL_OPINION_NOT_SOURCE_RESULT')
        self.assertEqual(len({x['observation']['elapsed_minutes'] for x in (a,b,c)}),1)
        forbidden={'gold','reference','evaluation','provenance','source_event_store','current_encounter_events','task_annotations'}
        self.assertTrue(all(not keys(r['payload'])&forbidden for r in self.calls))
        for r in self.calls:
            for record in r['payload']['observation'].get('longitudinal_memory',{}).get('records',[]):
                self.assertEqual(record['patient_id'],self.case['patient_id'])
                self.assertNotEqual(record['visit_id'],self.case['agent_visible']['index_encounter_id'])

    def test_specialty_timeout_schema_and_unfinished_stop_without_return(self):
        for i,(mode,code) in enumerate((('timeout','TIMEOUT'),('schema_invalid','SCHEMA_INVALID'),('unfinished','SCHEMA_INVALID'))):
            self.run_id=format(i+1,'032x');self.calls=[];result=self.execute(mode)
            self.assertEqual(result['status'],'failed');self.assertIn(code,result['blocker_codes'])
            self.assertEqual(len(self.calls),2);self.assertEqual(result['llm_execution']['request_count'],2)
            self.assertEqual(self.saved('reports/handoff.json')['handoffs'][0]['handoff_status'],'handoff_failed')

    def test_er_return_failure_is_not_success_and_no_retry(self):
        result=self.execute('er_return_failure');self.assertEqual(result['status'],'failed')
        self.assertEqual(len(self.calls),3);self.assertIn('CLIENT_ERROR',result['blocker_codes'])
        self.assertFalse(result['sequential_handoff_verified'])
        self.assertNotIn('PRIVATE ERROR EVIDENCE',json.dumps(result))

    def test_global_budget_includes_all_stages_and_later_er(self):
        for budget in (1,2,3):
            self.run_id=format(budget,'032x');self.config['max_requests']=budget;self.calls=[]
            result=self.execute();self.assertEqual(result['status'],'failed')
            self.assertEqual(len(self.calls),budget);self.assertEqual(result['llm_execution']['request_count'],budget)
            self.assertIn('GLOBAL_REQUEST_BUDGET_EXHAUSTED',result['blocker_codes'])

    def test_core_repair_shares_budget_without_second_specialty_request(self):
        result=self.execute('repair');self.assertEqual(result['status'],'completed')
        rows=self.saved('reports/handoff.json')['requests']
        self.assertEqual(sum(r['repair'] for r in rows),1);self.assertEqual(sum(r['stage']=='SPECIALTY_ASSESSMENT' for r in rows),1)
        self.assertEqual(len(rows),len(self.calls));self.assertTrue(next(r for r in rows if r['repair'])['repair_of_request_id'])
        self.run_id='e'*32;self.calls=[];self.config['max_requests']=3
        limited=self.execute('repair');self.assertEqual(limited['status'],'failed');self.assertEqual(len(self.calls),3)

    def test_identity_mismatch_records_observed_not_pinned_zero_generation(self):
        result=self.execute(mismatch=True);self.assertEqual(result['status'],'failed')
        self.assertEqual(result['llm_execution']['request_count'],0);self.assertEqual(result['llm_execution']['identity_check_count'],1)
        self.assertEqual(self.calls,[]);self.assertIn('MODEL_IDENTITY_FAILED',result['blocker_codes'])
        row=self.saved('reports/handoff.json')['requests'][0]
        self.assertEqual(row['pinned_model_digest'],self.identifier);self.assertEqual(row['observed_server_digest'],'sha256:'+'0'*64)

    def test_registry_prompt_digest_and_unsupported_routing_block_before_calls(self):
        examples=[]
        for key,value,code in [('prompt_sha256','0'*64,'PROMPT_FINGERPRINT_MISMATCH'),
                ('agent_id','missing','REGISTRY_BINDING_MISMATCH'),('version','wrong','REGISTRY_BINDING_MISMATCH'),
                ('model_identifier','sha256:'+'0'*64,'BINDING_DIGEST_MISMATCH')]:
            c=deepcopy(self.config);c['stage_bindings'][1][key]=value;examples.append((c,code))
        c=deepcopy(self.config);c['target_department']='dermatology';examples.append((c,'SPECIALTY_UNAVAILABLE'))
        for route in ('adaptive','disabled'):
            c=deepcopy(self.config);c['routing']=route;examples.append((c,'HANDOFF_CONFIGURATION_INVALID'))
        c=deepcopy(self.config);c['max_requests']=11;examples.append((c,'HANDOFF_CONFIGURATION_INVALID'))
        for config,code in examples:
            with patch('app.locdi.ollama_adapter._backend') as backend:
                check=live.preflight(run_id=self.run_id,config=config)
            self.assertIn(code,check['blocker_codes']);backend.assert_not_called()

    def test_no_handoff_skips_specialty_and_is_not_claimed_verified(self):
        result=self.execute('no_handoff');self.assertEqual(result['status'],'completed')
        self.assertEqual(result['handoff_outcome'],'NOT_REQUESTED');self.assertFalse(result['sequential_handoff_verified'])
        self.assertFalse(any(r['payload']['handoff_stage']=='SPECIALTY_ASSESSMENT' for r in self.calls))
        self.assertEqual(self.saved('reports/handoff.json')['handoffs'],[])

    def test_public_private_exports_correlate_without_identifiers_or_source_text(self):
        result=self.execute();audit=self.saved('reports/handoff.json');episode=self.saved('episodes/Case001.json')
        runtime=self.saved('reports/runtime.json');summary=json.loads((self.root/self.run_id/'metrics/summary.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(runtime['workflow'],episode['workflow']);self.assertEqual(runtime['terminal'],episode['result'])
        self.assertEqual(audit['configuration_fingerprint'],episode['configuration_fingerprint'])
        self.assertEqual(summary['episodes'][0]['configuration_fingerprint'],episode['configuration_fingerprint'])
        self.assertEqual(len([r for r in audit['requests'] if r['action_id']]),len(episode['workflow']['decisions']))
        documents=[json.loads(p.read_text(encoding='utf-8')) for p in (self.root/self.run_id).rglob('*.json')]
        forbidden={'gold','reference','evaluation','provenance','source_event_store','patient_id','visit_id','case_id','raw_response','system_prompt','user_prompt'}
        self.assertFalse(set().union(*(keys(d) for d in documents))&forbidden)
        leaves=[s for d in documents for s in strings(d)]
        for value in (self.case['patient_id'],self.case['case_id'],self.case['agent_visible']['index_encounter_id'],'TEST_ONLY assessment'):
            self.assertFalse(any(value in s for s in leaves))
        for event in self.case['environment']['current_encounter_events']:
            self.assertFalse(any(event['evidence'] in s for s in leaves))
        self.assertEqual(self.saved('reports/pilot.json'),result)

    def test_path_escape_existing_run_and_outputs_never_overwritten(self):
        result=self.execute();before={str(p):p.read_bytes() for p in (self.root/self.run_id).rglob('*') if p.is_file()}
        self.calls=[];again=self.execute();self.assertEqual(again['status'],'blocked');self.assertEqual(self.calls,[])
        self.assertEqual(before,{str(p):p.read_bytes() for p in (self.root/self.run_id).rglob('*') if p.is_file()})
        for run_id in ('../escape','', 'A'*32):self.assertTrue(live.preflight(run_id=run_id,config=self.config)['blocker_codes'])

    def test_checkpoint_failure_stops_before_next_generation(self):
        write=live._write
        def broken(document,destination,*args):
            if destination=='private/Request0001_after.json':raise OSError('TEST_ONLY PRIVATE ERROR EVIDENCE')
            return write(document,destination,*args)
        with patch.object(live,'_write',side_effect=broken):result=self.execute()
        self.assertEqual(result['status'],'failed');self.assertEqual(len(self.calls),1)
        self.assertTrue((self.root/self.run_id/'private/Request0001_before.json').exists())
        self.assertNotIn('PRIVATE ERROR EVIDENCE',json.dumps(result))

    def test_untrusted_gold_and_future_memory_stop_before_transport(self):
        self.execute();payload=deepcopy(self.calls[0]['payload']);self.run_id='f'*32
        for kind in ('gold','future'):
            check=live.preflight(run_id=self.run_id,config=self.config)
            client=live.LiveSequentialClient(self.config,approved_plan=check['plan'],authorized=True,output_root=self.root/self.run_id)
            p=deepcopy(payload)
            if kind=='gold':p['gold']={}
            else:p['observation']['longitudinal_memory']['records'][0]['available_at']='2099-01-01T00:00:00Z'
            with patch('app.locdi.ollama_adapter._backend') as backend:
                with self.assertRaises(live.HandoffStopped):client.chat_json('TEST_ONLY',json.dumps(p),model=self.config['stage_bindings'][0]['model'],schema={})
            backend.assert_not_called();self.assertEqual(client.request_count,0);client.close()

    def test_approved_setup_change_between_requests_blocks_transport(self):
        self.execute();payload=deepcopy(self.calls[0]['payload']);self.run_id='f'*32
        check=live.preflight(run_id=self.run_id,config=self.config)
        client=live.LiveSequentialClient(self.config,approved_plan=check['plan'],authorized=True,output_root=self.root/self.run_id)
        self.addCleanup(client.close)
        client.live_config['stage_bindings'][1]['prompt_sha256']='0'*64
        with patch('app.locdi.ollama_adapter._backend') as backend:
            with self.assertRaises(live.HandoffStopped):client.chat_json('TEST_ONLY',json.dumps(payload),model=self.config['stage_bindings'][0]['model'],schema={})
        backend.assert_not_called();self.assertEqual(client.failure,'APPROVED_SETUP_CHANGED');self.assertEqual(client.request_count,0)

    def test_request_checkpoint_write_failure_before_transport(self):
        write=live._write
        def broken(document,destination,*args):
            if destination=='private/Request0001_before.json':raise OSError('TEST_ONLY PRIVATE ERROR EVIDENCE')
            return write(document,destination,*args)
        with patch.object(live,'_write',side_effect=broken):result=self.execute()
        self.assertEqual(result['status'],'failed');self.assertEqual(self.calls,[])
        self.assertIn('CHECKPOINT_STORAGE_FAILED',result['blocker_codes'])


if __name__=='__main__':unittest.main()
