"""TEST_ONLY persistent evidence. Always patches transport; never real Ollama.

Run: python -B tests/locdi/live_handoff_smoke.py
The adapter's counters measure fake dispatches, not actual model generations.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import app.locdi.live_handoff as live
from test_live_handoff import fake_backend


def main():
    destination=live.ROOT/'outputs/test_only_handoff/phase29a_v1'
    if destination.exists():raise FileExistsError('TEST_ONLY output already exists')
    destination.mkdir(parents=True,exist_ok=False)
    config=live.load_config();identifier=config['stage_bindings'][0]['model_identifier'];rows=[]
    modes=[('normal',None,10,False),('no_handoff','no_handoff',10,False),
        ('specialty_timeout','timeout',10,False),('specialty_schema_invalid','schema_invalid',10,False),
        ('er_return_failure','er_return_failure',10,False),('budget_exhausted',None,2,False),
        ('core_repair','repair',10,False),('identity_mismatch',None,10,True)]
    with patch.object(live,'HANDOFF_ROOT',destination),patch('app.locdi.llm_setup.inspect_local_model',return_value={
            'status':'READY','model_identifier':identifier}),patch('socket.create_connection',side_effect=AssertionError('NO_NETWORK')):
        for index,(name,mode,budget,mismatch) in enumerate(modes,1):
            definition=deepcopy(config);definition['max_requests']=budget;calls=[];run_id=format(index,'032x')
            check=live.preflight(run_id=run_id,config=definition)
            with fake_backend(calls,identifier,mode,mismatch):
                result=live.run_live_handoff(run_id=run_id,config=definition,execute=True,allow_llm=True,
                    accept_release_assumption=True,expected_plan_fingerprint=check['plan_fingerprint'])
            folder=destination/run_id
            audit=json.loads((folder/'reports/handoff.json').read_text(encoding='utf-8'))
            episode=json.loads((folder/'episodes/Case001.json').read_text(encoding='utf-8'))
            rows.append({'scenario':name,'run_id':run_id,'status':result['status'],
                'core_status':result['core_status'],'export_status':result['export_status'],
                'blocker_codes':result['blocker_codes'],'fake_generation_dispatches':len(calls),
                'adapter_generation_counter':audit['request_count'],'fake_identity_checks':audit['identity_check_count'],
                'handoff_outcome':result['handoff_outcome'],'sequential_handoff_verified':result['sequential_handoff_verified'],
                'result':episode['result'],'request_stages':[r['stage'] for r in audit['requests']],
                'repair_count':sum(r['repair'] for r in audit['requests']),'source_consult_count':audit['source_consult_count']})
    document={'schema_version':'locdi.test_only_live_handoff_smoke.v1','test_only':True,
        'transport':'PATCHED_FAKE_BACKEND_NO_NETWORK','actual_model_generations':0,'actual_server_identity_queries':0,
        'utility_fit_count':0,'clinical_verdicts_generated':0,'clinical_metrics':'NOT_EVALUATED',
        'research_evaluation_ready':False,'scenarios':rows}
    with (destination/'TEST_ONLY_summary.json').open('x',encoding='utf-8') as stream:json.dump(document,stream,ensure_ascii=True,indent=2)
    print(json.dumps(document,ensure_ascii=True));return 0


if __name__=='__main__':sys.exit(main())
