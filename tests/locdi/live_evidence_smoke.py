"""TEST_ONLY persistent v2 evidence audit; backend is always fake, no network."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import app.locdi.live_handoff as live
from test_live_handoff import fake_backend
from test_sequential_handoff import HandoffClient
from test_specialty_evidence_contract import response


def main():
    output=live.ROOT/'outputs/test_only_handoff/phase29b_v2'
    output.mkdir(parents=True,exist_ok=False);config=live.load_config();identifier=config['stage_bindings'][0]['model_identifier']
    original=HandoffClient.chat_json;rows=[]
    for index,mode in enumerate(('unresolved_valid','index_out_of_range','supported_empty','unfinished'),1):
        calls=[];run_id=format(0x29b000+index,'032x')
        def variant(client,system,user,model=None,*,schema=None):
            if json.loads(user)['handoff_stage']=='SPECIALTY_ASSESSMENT' and mode in {'index_out_of_range','supported_empty'}:
                return json.dumps(response([0],'unresolved') if mode=='index_out_of_range' else response([]))
            return original(client,system,user,model,schema=schema)
        with patch.object(live,'HANDOFF_ROOT',output),patch('app.locdi.llm_setup.inspect_local_model',return_value={
                'status':'READY','model_identifier':identifier}),patch('socket.create_connection',side_effect=AssertionError('NO_NETWORK')):
            check=live.preflight(run_id=run_id)
            with fake_backend(calls,identifier,'unfinished' if mode=='unfinished' else None),patch.object(HandoffClient,'chat_json',variant):
                result=live.run_live_handoff(run_id=run_id,execute=True,allow_llm=True,accept_release_assumption=True,expected_plan_fingerprint=check['plan_fingerprint'])
        audit=json.loads((output/run_id/'reports/handoff.json').read_text(encoding='utf-8'))
        episode=json.loads((output/run_id/'episodes/Case001.json').read_text(encoding='utf-8'))
        semantic=audit['requests'][1].get('specialty_evidence_validation')
        rows.append({'scenario':mode,'run_id':run_id,'core_status':result['core_status'],'export_status':result['export_status'],
            'fake_dispatches':len(calls),'handoff_outcome':result['handoff_outcome'],'blocker_codes':result['blocker_codes'],
            'schema_validation':audit['requests'][1]['schema_validation'],'semantic_verdict':semantic,
            'er_return_requests':sum(r['stage']=='ER_REASSESSMENT' for r in audit['requests']),
            'elapsed_simulation_minutes':episode['result']['elapsed_simulation_minutes'],'action_counts':episode['result']['action_counts']})
    summary={'schema_version':'locdi.phase29b.TEST_ONLY_smoke.v1','test_only':True,'transport':'PATCHED_FAKE_BACKEND',
        'actual_model_generations':0,'actual_identity_queries':0,'utility_fit_count':0,'clinical_verdicts_generated':0,
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,'scenarios':rows}
    with (output/'TEST_ONLY_summary.json').open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
