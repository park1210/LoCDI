"""Additional read-only facts for the completed development comparison."""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NS='B34_COMPLETED_RUN_ANALYSIS_20261121_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/recovery_llm_all_001'
    evidence=ROOT/'outputs/private/research_evaluation/B34_LLM_TEST_READY_v1/development_proxy_fit_001/development_routing_evidence.json'
    ev=read(evidence);samples=ev['routing_artifact']['samples'];groups=defaultdict(list)
    for x in samples:groups[x['decision_id']].append(x)
    assert len(groups)==2 and all(len(g)==5 for g in groups.values())
    assert all(len({tuple(x['tokens']) for x in g})==1 for g in groups.values())
    role=defaultdict(Counter);dept=defaultdict(Counter);horizons=Counter();delays=Counter();symptoms=Counter()
    posterior_max=0.;routes=defaultdict(Counter);source_menu=Counter();cutoff_count=0
    all_ids=set();train={x['patient_id'] for x in samples}
    bindings=read(base/'bindings.json');panel=read(base/'panel.json')
    all_ids.update(x['patient_id'] for x in bindings.values());assert not train&all_ids
    for b in bindings.values():
        for x in b['rows']:source_menu[x['kind']]+=1
    complete=read(base/'run_recovery_llm_all_001/collection.json')
    for row in complete['rows']:
        co=row['condition'];f=base/'run_recovery_llm_all_001/requests'/('Episode%04d'%row['episode_ordinal'])
        rt=read(f/'runtime_snapshot_private.json');end=read(f/'actual_core_decisions.json')[-1];dept[co][end['target_department']]+=1
        for q in rt['voi_state'].get('primary_routing_history',[]):routes[co][q.get('selected_specialty') or 'NONE']+=1
        for b in sorted(f.glob('Request*_before.json')):
            payload=json.loads(read(b)['request']['messages'][1]['content'])
            role[co][payload['role']+':'+str(payload.get('department') or '')]+=1
        first=json.loads(read(f/'Request0001_before.json')['request']['messages'][1]['content'])['observation']
        symptoms[type(first.get('current_symptoms')).__name__]+=1
        for p in read(f/'planner_trace.json'):horizons[p['horizon']]+=1
    prior=read(ROOT/'outputs/research_evaluation'/NS/'analysis.json')
    import csv
    doc=ROOT/'docs'/'\uC5F0\uAD6C \uD3C9\uAC00'/NS
    with (doc/'EPISODE_RESULTS.csv').open(encoding='utf-8-sig') as f:
        for x in csv.DictReader(f):
            delays[x['export_elapsed_minus_terminal_minute']]+=1
            posterior_max=max(posterior_max,float(x['posterior_max_difference']))
    bad=[p for p,h in ev['origin_files'].items() if not Path(p).is_file() or sha(Path(p))!=h]
    assert not bad
    facts={'training_groups':len(groups),'training_rows':len(samples),
        'train_patient_count':len(train),'evaluation_patient_count':len(all_ids),'train_evaluation_patient_overlap':len(train&all_ids),
        'identical_training_tokens_across_five_agents_per_group':True,
        'train_positive_proxy_by_agent':dict(Counter(x['agent_id'] for x in samples if x['correct'])),
        'train_false_proxy_by_agent':dict(Counter(x['agent_id'] for x in samples if not x['correct'])),
        'min_samples':ev['routing_artifact']['min_samples'],'origin_files_checked':len(ev['origin_files']),
        'origin_hash_mismatches':bad,'utility_evidence_sha256':sha(evidence),
        'actual_LLM_roles':{k:dict(v) for k,v in role.items()},'final_departments':{k:dict(v) for k,v in dept.items()},
        'routing_selections':{k:dict(v) for k,v in routes.items()},'effective_planner_horizons':dict(horizons),
        'export_minus_terminal_minute_counts':dict(delays),'posterior_reconstruction_max_abs_difference':posterior_max,
        'current_symptoms_field_types_initial':dict(symptoms),'common_source_menu_rows':dict(source_menu),
        'evaluation_source_department_distribution':dict(Counter(d for x in panel for d in x['reference']['accepted_departments'])),
        'actual_new_generation':0,'fitting_calls':0}
    for p in (ROOT/'outputs/research_evaluation'/NS/'supplement.json',doc/'SUPPLEMENT.json'):
        with p.open('x',encoding='utf-8') as f:json.dump(facts,f,ensure_ascii=True,indent=2)
    print(json.dumps(facts,ensure_ascii=True))
if __name__=='__main__':main()
