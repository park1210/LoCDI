"""TEST_ONLY multi-case diagnostics using original clients/fixtures/Core."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
from app.locdi.handoff_integration import run_handoff_integrated
from app.locdi.llm_setup import digest
from app.locdi.episode_exporter import _write
from app.core.information_outcomes import InformationOutcomes
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.voi_planning import plan_voi
from handoff_integration_support import integrated_options, IntegrationClient


def features(case, ordinal):
    visible=case['agent_visible'];events=case['environment']['current_encounter_events']
    return {'case_ref':'Case%03d'%ordinal,'ordinal':ordinal,
        'past_record_count':sum(len(p['records']) for p in visible['past_visits']),
        'past_visit_count':len(visible['past_visits']),
        'ask_count':sum(e['action']=='Ask' for e in events),
        'test_count':sum(e['action']=='Test' for e in events),
        'consult_count':sum(e['action']=='Consult' for e in events),
        'initial_objective_count':len(visible['current_initial_state']['doctor_visible_info'].get('objective_findings',[])),
        'event_available_minute_range':[min(e['available_minute'] for e in events),max(e['available_minute'] for e in events)],
        'availability_basis':'RECORDED_DATASET_PROXY_NOT_CLINICAL_RELEASE_VERIFICATION',
        'specialty_candidate_basis':'REGISTERED_TEST_ONLY_CANDIDATES_NOT_GOLD',
        'source_fingerprint':digest(dict(case))}


def select_cases(cases):
    """Pre-outcome rule: first, shortest/longest memory, earliest/latest visit."""
    rows=[features(c,i+1) for i,c in enumerate(cases)]
    count=lambda i:rows[i]['past_record_count']
    start=lambda i:cases[i]['agent_visible']['index_encounter_start']
    indices=list(dict.fromkeys([0,min(range(len(cases)),key=count),max(range(len(cases)),key=count),
        min(range(len(cases)),key=start),max(range(len(cases)),key=start)]))
    for i in sorted(range(len(cases)),key=lambda i:(-count(i),i)):
        if len(indices)>=5:break
        if i not in indices:indices.append(i)
    return [rows[i] for i in indices]


class OrderedClient(IntegrationClient):
    def __init__(self, mode=None):
        super().__init__(mode);self.order=[]

    def chat_json(self,*args,**kwargs):
        import json
        payload=json.loads(args[1]);self.order.append({'kind':'LLM_PROPOSAL','stage':payload['handoff_stage'],
            'minute':payload['observation']['elapsed_minutes'],'call_sequence':len(self.calls)+1})
        return super().chat_json(*args,**kwargs)


def execute_case(case, row, scenario, *, output_root=None):
    routing=scenario in {'B','D','E'};voi=scenario in {'C','D'}
    mode={'F_TIMEOUT':'timeout','H_INVALID_EVIDENCE':'semantic_failure'}.get(scenario)
    client=OrderedClient(mode)
    options=replace(integrated_options(case,routing=routing,voi=voi,abstain=scenario=='E'),client=client)
    original=InformationOutcomes.observe
    def observe(model,obs):
        belief,matched=original(model,obs)
        client.order.append({'kind':'BELIEF_RECOMPUTED_FROM_PRIOR','minute':obs.get('elapsed_minutes'),
            'llm_proposals_seen':len(client.calls),'matched_outcome_count':len(matched),
            'belief_fingerprint':digest(belief),'observation_fingerprint':digest(obs)})
        return belief,matched
    with patch.object(InformationOutcomes,'observe',observe),patch(
            'app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner,patch(
            'app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler,patch(
            'app.core.information_value.plan_voi',wraps=plan_voi) as planner:
        result=run_handoff_integrated(case,options=options,test_only=True,
            routing='adaptive_test_only' if routing else 'static_development',
            budget=2 if scenario=='G_BUDGET' else 10,output_root=output_root,
            export_id='TEST_ONLY_'+row['case_ref']+'_'+scenario)
    record={'case_ref':row['case_ref'],'scenario':scenario,'status':result['status'],
        'blocker_codes':result['blocker_codes'],'integration_contract_ready':result['integration_contract_ready'],
        'export_status':result['export_status'],'calls':{'runner':runner.call_count,'scheduler':scheduler.call_count,'plan_voi':planner.call_count},
        'dispatch_count':len(client.calls),'order':client.order,'research_evaluation_ready':False}
    if result['audit']:
        record.update(handoff_status=result['audit']['handoff_status'],routing=result['audit']['routing'],
            terminal=result['audit']['terminal'],voi=result['audit']['voi'],memory=result['audit']['memory'])
    return result,client,record


def boundary_check(case,client,result):
    """Actual observed input vs actual source delivery and original memory IDs."""
    final=result['raw']['execution']['visits'][0]['final_state']
    received={e['payload']['requested_test']:e['minute'] for e in final['event_log'] if e['event_type']=='result_received'}
    allowed={r['memory_id'] for p in case['agent_visible']['past_visits'] for r in p['records']}
    import json
    safe=True
    for call in client.calls:
        p=call['payload'];obs=p['observation'];text=json.dumps(obs['objective_findings'])
        safe &= not any(key in p for key in ('gold','evaluation','provenance','source_event_store'))
        safe &= all(r['memory_id'] in allowed for r in obs.get('longitudinal_memory',{}).get('records',[]))
        for event in case['environment']['current_encounter_events']:
            if event['action']=='Test' and event['target']+': '+event['evidence'] in text:
                safe &= event['target'] in received and obs['elapsed_minutes']>=received[event['target']]>=event['available_minute']
    return bool(safe)
