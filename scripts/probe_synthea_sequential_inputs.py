"""Recheck the same frozen selection with compact, fact-preserving wire data."""
from copy import deepcopy
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_synthea_sequential_pilot import (PUB,PR,CODE,DATA,load_protocol,test_authority)
from scripts.audit_research_gaps import read,write,sha
from app.locdi.synthea_sequential_proxy import SequentialSyntheticSourceView,policy_contract
from app.locdi.dataset_loader import load_cases
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.current_ablation_llm_backend import input_guard
from app.locdi.current_ablation_runner import features_for
from app.locdi.run_simulation import run_case


def main():
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from test_simulation import SourceActionClient,smoke_options
    rows=read(PR/'source_rows.json');catalog=read(CATALOG);cases=load_cases(DATA)
    panel=read(PR/'panel.json');index=read(PR/'case_index.json')
    spec=policy_contract(rows,test_only=True)
    view=SequentialSyntheticSourceView(rows,spec=spec,authorization=test_authority(spec),plan_fingerprint='TEST_ONLY_PROBE',catalog=catalog)
    class Client(SourceActionClient):
        def __init__(self):super().__init__();self.bounds=[];self.errors=[]
        def chat_json(self,system,user,model=None,*,schema=None):
            p=json.loads(user);q=view.compact_payload(p)
            assert q['observation']==p['observation'] and q['evidence_records']==p['evidence_records']
            assert [(x['action'],x['target']) for x in q['candidate_actions']]==[(x['action'],x['target']) for x in p['candidate_actions']]
            text=json.dumps(q,ensure_ascii=False,sort_keys=True)
            try:self.bounds.append(input_guard(system,text,schema))
            except Exception as exc:self.errors.append(type(exc).__name__)
            return super().chat_json(system,text,model=model,schema=schema)
    probes=[];started=time.monotonic()
    for ordinal,p in enumerate(panel,1):
        for condition in ('B0','B2'):
            client=Client();result=run_case(cases[index[p['case_ref']]-1],options=smoke_options(client,
                runner_options=features_for(condition),diagnosis_catalog=catalog,max_memory_chars=None,sequential_source_view=view))
            names={x['test_name'] for x in result.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])}
            events=result['execution']['visits'][0]['final_state']['event_log'] if result.get('execution') else []
            starts={e['payload']['requested_test']:e['minute'] for e in events if e['event_type']=='test_started' and e['payload'].get('requested_test') in names}
            received={e['payload']['requested_test']:e['minute'] for e in events if e['event_type']=='result_received' and e['payload'].get('requested_test') in names}
            violations=[]
            for name,minute in received.items():
                if name not in starts or minute<starts[name]+5:violations.append('REQUEST_START_OR_TAT')
            for call in client.calls:
                o=call['payload']['observation']
                for finding in o['objective_findings']:
                    for name in names:
                        if finding.startswith(name+':') and (name not in received or o['elapsed_minutes']<received[name]):violations.append('BEFORE_RECEIPT')
            write(PR/('TEST_ONLY_compact_Patient%02d_%s.json'%(ordinal,condition)),{'raw':result,'model_payloads':client.calls,
                'input_bounds':client.bounds,'input_errors':client.errors,'TEST_ONLY':True})
            probes.append({'selected_patient_ordinal':ordinal,'condition':condition,'TEST_ONLY':True,'core_status':result['status'],
                'additional_menu_count':len(names),'additional_started':len(starts),'additional_received':len(received),
                'decision_count':len(client.calls),'input_context_error_count':len(client.errors),
                'max_input_bound':max((x['conservative_input_token_bound'] for x in client.bounds),default=None),'temporal_violations':violations})
    write(PUB/'TEST_ONLY_compact_selected10_core_probe.json',probes)
    write(PUB/'sequential_source_policy_final.json',policy_contract(rows))
    p=read(PUB/'SYNTHEA_PILOT_PROTOCOL.json');old=p.pop('fingerprint')
    p['supersedes_initial_probe_protocol']=old;p['initial_probe_blocked_slots_preserved']=p['TEST_ONLY_probe_blocked_slots']
    p['TEST_ONLY_probe_blocked_slots']=[x for x in probes if x['input_context_error_count'] or x['temporal_violations'] or x['core_status']=='failed']
    p['policy_fingerprint']=policy_contract(rows)['fingerprint']
    p['candidate_wire_projection']='preserve full observation/Memory/evidence and action-target pairs; omit redundant action_id/type/null cost/model; no Prompt text changes'
    p['new_code_hashes']={f:sha(ROOT/f) for f in CODE+[str(Path(__file__).relative_to(ROOT)).replace('\\','/')]}
    p['fingerprint']=__import__('app.locdi.llm_setup',fromlist=['digest']).digest(p)
    write(PUB/'SYNTHEA_PILOT_PROTOCOL_FINAL.json',p)
    write(PUB/'compact_probe_timing.json',{'seconds':time.monotonic()-started,'LLM_generation':0})
    print(json.dumps({'status':'PASS' if not p['TEST_ONLY_probe_blocked_slots'] else 'BLOCKED','blocked_slots':p['TEST_ONLY_probe_blocked_slots'],
        'max_bound':max(x['max_input_bound'] or 0 for x in probes),'fingerprint':p['fingerprint']}))


if __name__=='__main__':main()
