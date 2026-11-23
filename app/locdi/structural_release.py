"""Read-only structural RC checks. No LLM backend, fitting or clinical labels."""
from collections import Counter
from pathlib import Path
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import _json
from app.core.clinical_observation import doctor_observation,initial_symptoms
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.information_value import candidate_actions
from app.core.longitudinal_memory import PatientMemoryStore,timestamp
from app.core.agent_registry import AgentRegistry

RUNS=('20261029000000000000000000002901','20261029000000000000000000002902','20261030000000000000000000002903')
STRUCTURAL_REQUIRED=('structural_contract_ready','offline_integration_ready','live_er_verified',
    'live_sequential_handoff_verified','all_case_static_validated','full_regression_passed',
    'source_boundary_verified','artifact_gate_enforced','output_integrity_verified','preservation_verified')


def evaluate_release(checks):
    """Only exact PASS meets a structural prerequisite. BLOCKED is not PASS."""
    blocked=[k for k in STRUCTURAL_REQUIRED if checks.get(k)!='PASS']
    research=('clinical_reference_ready','live_adaptive_voi_ready','research_evaluation_ready')
    return {'STRUCTURAL_RELEASE_CANDIDATE':'READY_DEVELOPMENT_ONLY' if not blocked else 'BLOCKED',
        'RESEARCH_EVALUATION_READY':'BLOCKED', 'structural_blocker_codes':blocked,
        'research_blocker_codes':[k for k in research if checks.get(k)!='PASS']+
            ['PAPER_EQUIVALENCE_NOT_CHECKED','INDEPENDENT_RESEARCH_APPROVAL_NOT_VERIFIED'],
        'criteria':checks,'scope':'DEVELOPMENT_CONTRACTS_NOT_CLINICAL_CLEARANCE',
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}


def preservation(baseline):
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    inventories={}
    for folder in ('outputs/development_handoff','outputs/development_pilot'):
        for directory in (ROOT/folder).iterdir():
            if not directory.is_dir():continue
            prefix=directory.relative_to(ROOT).as_posix()+'/'
            expected={p for p in baseline if p.startswith(prefix)}
            actual={p.relative_to(ROOT).as_posix() for p in directory.rglob('*') if p.is_file()}
            inventories[directory.name]={'file_count':len(actual),'unchanged':expected==actual and not expected.intersection(changed)}
    return {'status':'PASS' if not changed and all(v['unchanged'] for v in inventories.values()) else 'FAIL',
        'files_checked':len(baseline),'core_files_checked':sum(p.startswith('app/core/') for p in baseline),
        'changed_files':changed,'saved_runs':inventories}


def audit_handoff(run_id):
    """Audit saved schema/semantic statuses, never revalidate missing raw text."""
    if run_id not in RUNS:raise ValueError('UNREGISTERED_SAVED_RUN')
    root=ROOT/'outputs/development_handoff'/run_id
    names=['reports/'+n+'.json' for n in ('preflight','started','pilot','handoff','runtime')]+['episodes/Case001.json','metrics/summary.jsonl']
    missing=[n for n in names if not (root/n).is_file()]
    if missing:return {'status':'BLOCKED','run_id':run_id,'error_codes':['SAVED_RUN_INCOMPLETE'],'missing_files':missing}
    pre,start,pilot,audit,runtime,ep,summary=[load(root/n) for n in names]
    summary=summary['episodes'][0];plan=pre['plan'];requests=audit['requests'];expected='completed' if run_id==RUNS[2] else 'failed'
    checks={'run_identity':all(x['run_id']==run_id for x in (pre,start,pilot,audit,runtime,ep)),
        'plan_fingerprint':digest(plan)==pre['plan_fingerprint']==start['plan_fingerprint']==pilot['plan_fingerprint']==audit['plan_fingerprint'],
        'registry_and_prompt':all(all(r[k]==plan['stages'][r['stage']][k] for k in ('agent_id','model','version')) and
            r['prompt_fingerprint']==plan['stages'][r['stage']]['system_prompt_fingerprint'] for r in requests),
        'pinned_observed_digest':all(r['pinned_model_digest']==r['observed_server_digest']==plan['model_identifier'] for r in requests),
        'terminal_status':pilot['status']==pilot['core_status']==ep['result']['status']==summary['status']==expected,
        'export_status':pilot['export_status']=='EXPORTED',
        'runtime_episode':runtime['workflow']==ep['workflow'] and runtime['terminal']==ep['result'] and runtime['memory']==ep['memory'],
        'summary_consistency':all(summary[k]==ep['result'][k] for k in ('status','action_counts','elapsed_simulation_minutes','diagnosis_unresolved')),
        'configuration_fingerprint':audit['configuration_fingerprint']==ep['configuration_fingerprint']==summary['configuration_fingerprint'],
        'counters':all(x['request_count']==len(requests) and x['identity_check_count']==len(requests) for x in
            (pilot['llm_execution'],ep['llm_execution'],summary['llm_execution'])) and audit['request_count']==audit['identity_check_count']==len(requests),
        'checkpoint_links':True,'semantic_checkpoint':True,
        'memory_cutoff':all(r['cutoff_compliant'] is True for r in ep['memory']['retrievals'])}
    stable=('request_id','decision_id','stage','minute','input_fingerprint','prompt_fingerprint','schema_fingerprint','response_fingerprint',
        'generation_count_after','identity_check_count_after')
    for r in requests:
        before=load(root/('private/'+r['request_id']+'_before.json'))['request']
        after=load(root/('private/'+r['request_id']+'_after.json'))['request']
        checks['checkpoint_links'] &= all(after[k]==r[k] for k in stable) and all(before[k]==r[k] for k in stable[:7])
        if 'specialty_evidence_validation' in r:
            checks['semantic_checkpoint'] &= load(root/('private/'+r['request_id']+'_semantic.json'))['request']['specialty_evidence_validation']==r['specialty_evidence_validation']
    decisions=ep['workflow']['decisions'];events={e['sequence']:e for e in ep['workflow']['events']}
    checks['request_decision_action_event']=all(any(r.get('core_decision_sequence')==d['sequence'] and
        r.get('execution_event_sequences')==d['execution_event_sequences'] and r.get('action_id')=='Action%04d'%d['sequence'] for r in requests) and
        all(i in events and events[i]['minute']==d['minute'] for i in d['execution_event_sequences']) for d in decisions if d['execution_event_sequences'])
    checks['handoff_terminal']=(pilot['sequential_handoff_verified'] is (expected=='completed') and
        (bool([r for r in requests if r['stage']=='ER_REASSESSMENT']) is (expected=='completed')))
    if expected=='failed':
        checks['failure_preserved']=pilot['blocker_codes']==['SPECIALTY_EVIDENCE_CONTRACT_ERROR'] and not ep['result']['action_counts'] and not ep['result']['finalized']
    source_events=[{'event':e['event_type'],'minute':e['minute'],'sequence':e['sequence'],'target_ref':e['target_ref']}
        for e in events.values() if e['event_type'] in {'test_ordered','test_started','result_received','case_completed'}]
    checks['source_release_event_order']=all(any(o['event']=='test_ordered' and o['target_ref']==e['target_ref'] and o['minute']<=e['minute'] for o in source_events)
        and any(o['event']=='test_started' and o['target_ref']==e['target_ref'] and o['minute']<=e['minute'] for o in source_events)
        for e in source_events if e['event']=='result_received')
    return {'status':'PASS' if all(checks.values()) else 'FAIL','run_id':run_id,'checks':checks,
        'recorded_status':pilot['status'],'core_status':pilot['core_status'],'export_status':pilot['export_status'],
        'handoff_completed':pilot['sequential_handoff_verified'],'blocker_codes':pilot['blocker_codes'],
        'generation_count':audit['request_count'],'identity_check_count':audit['identity_check_count'],
        'semantic_repair_count':sum(r.get('repair_kind')=='SPECIALTY_SEMANTIC' for r in requests),
        'retry_count':sum(r['retry'] for r in requests),
        'plan_fingerprint':pre['plan_fingerprint'],'registry_fingerprint':plan['registry_fingerprint'],'model_identifier':plan['model_identifier'],
        'prompt_fingerprints':{s:p['system_prompt_fingerprint'] for s,p in plan['stages'].items()},
        'requests':[{k:r.get(k) for k in ('request_id','decision_id','stage','minute','core_decision_sequence','action_id','execution_event_sequences',
            'schema_validation','specialty_evidence_validation','input_fingerprint','response_fingerprint')} for r in requests],
        'legacy_semantic_detail':'NOT_CHECKED_NOT_RECORDED' if run_id==RUNS[0] else 'SAVED_SEMANTIC_STATUS',
        'terminal':ep['result'],'memory':ep['memory'],'source_release_events':source_events,
        'raw_wire':'NOT_CHECKED_NOT_SAVED','file_count':sum(p.is_file() for p in root.rglob('*')),
        'files_fingerprint':digest({p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()})}


def inspect_case(case,ordinal,registry_manifest):
    row={'case_ref':'Case%03d'%ordinal,'status':'FAIL','error_codes':[],'actual_generation_calls':0,
        'clinical_correctness':'NOT_CHECKED','live_execution':'NOT_CHECKED','actual_clinical_release':'NOT_CHECKED',
        'episode_export':'NOT_CHECKED_NOT_EXECUTED'}
    try:
        registry=AgentRegistry(registry_manifest);context=adapt_case(case,registry_manifest=registry_manifest)
        engine=context.runtime_case;identity=engine['longitudinal'];records=[r for p in engine['past_visits'] for r in p['records']]
        cutoff=timestamp(identity['started_at']);store=PatientMemoryStore(records)
        runtime=HospitalCaseRuntime(engine['case_id'],engine,0,ordinal-1,current_symptoms=initial_symptoms(engine))
        observed=doctor_observation(runtime)
        from app.core.clinical_observation import memory_query
        selected=store.retrieve(identity,memory_query(observed),'selective',5)
        runtime.memory_context=selected;observed=doctor_observation(runtime)
        choices=candidate_actions(runtime,['ask_patient','request_test','diagnose'],engine['available_tests'],[])
        allowed={r['memory_id'] for r in records}
        memory=all(r['memory_id'] in allowed and r['patient_id']==identity['patient_id'] and r['visit_id']!=identity['visit_id'] and
            timestamp(r['visit_started_at'])<cutoff and timestamp(r['observed_at'])<=timestamp(r['available_at'])<=cutoff for r in selected['records'])
        finite=bool(_json(engine))
        checks={'adapter_validated':True,'memory_cutoff_and_identity':memory,
            'initial_result_not_projected':not observed['objective_findings'] and not observed['history'],
            'source_candidate_creation':bool(choices) and all(c['action'] in {'ask_patient','request_test','diagnose'} for c in choices),
            'registry_er_binding':bool(registry.get('er','')),'finite_export_input_json':finite}
        counts=Counter(e['action'] for e in engine['current_encounter_events'])
        row.update(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
            validator_statuses={r.validator:r.status for r in context.validation_results},
            source_event_counts=dict(counts),past_memory_count=len(records),selected_memory_count=len(selected['records']),
            initial_objective_count=len(observed['objective_findings']),initial_history_count=len(observed['history']),
            candidate_counts=dict(Counter(c['action'] for c in choices)),
            available_minute_range=[min(e['available_minute'] for e in engine['current_encounter_events']),max(e['available_minute'] for e in engine['current_encounter_events'])],
            availability_basis='DEVELOPMENT_PROXY_NOT_ACTUAL_RECEIPT',export_input_status='FINITE_ADAPTER_INPUT_ONLY',
            source_fingerprint=digest(dict(case)))
    except Exception as exc:
        row['error_codes']=['STATIC_CONTRACT_REJECTED'];row['error_type']=type(exc).__name__
    return row


def verify_boundary(case,client,result):
    """Direct strings, NOT JSON-escaped substring tests; checks both Ask/Test."""
    events=result['raw']['execution']['visits'][0]['final_state']['event_log']
    delivered={e['payload']['requested_test']:e['minute'] for e in events if e['event_type']=='result_received'}
    # Scheduler stores the synchronous source answer in question_asked; it
    # does not emit a separate answer_received event on this encounter path.
    answered={e['payload'].get('question'):e['minute'] for e in events if e['event_type']=='question_asked'}
    sources=case['environment']['current_encounter_events'];memory={r['memory_id']:r for p in case['agent_visible']['past_visits'] for r in p['records']}
    index=timestamp(case['agent_visible']['index_encounter_start']);checks={'test_release':True,'ask_release':True,'memory_scope':True,'private_scope':True}
    def forbidden(value):
        if isinstance(value,dict):return bool(set(value)&{'gold','evaluation','provenance','task_annotations','source_event_store','current_encounter_events','hidden_test_results'}) or any(forbidden(v) for v in value.values())
        return isinstance(value,list) and any(forbidden(v) for v in value)
    for call in client.calls:
        payload=call['payload'];obs=payload['observation'];minute=obs['elapsed_minutes']
        checks['private_scope'] &= not forbidden(payload)
        for finding in obs.get('objective_findings',[]):
            matching=[e for e in sources if e['action']=='Test' and finding==e['target']+': '+e['evidence']]
            checks['test_release'] &= bool(matching) and all(e['target'] in delivered and minute>=delivered[e['target']]>=e['available_minute'] for e in matching)
        for answer in obs.get('history',[]):
            if answer.get('answer_status')=='unknown':continue
            matching=[e for e in sources if e['action']=='Ask' and e['target']==answer.get('question') and e['evidence']==answer.get('answer')]
            checks['ask_release'] &= bool(matching) and all(e['target'] in answered and minute>=answered[e['target']]>=e['available_minute'] for e in matching)
        for r in obs.get('longitudinal_memory',{}).get('records',[]):
            source=memory.get(r['memory_id'])
            checks['memory_scope'] &= source is not None and r['text']==source['text'] and \
                r['patient_id']==source['patient_id']==case['patient_id'] and r['visit_id']==source['visit_id'] and \
                r['available_at']==source['available_at'] and r['observed_at']==source['observed_at'] and \
                timestamp(r['observed_at'])<=timestamp(r['available_at'])<=index and timestamp(r['visit_started_at'])<index
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}


def validate_episode_links(episode,audit,summary):
    row=summary['episodes'][0]
    checks={'workflow':episode['workflow']==audit['workflow'],'terminal':episode['result']==audit['terminal'],
        'configuration':episode['configuration_fingerprint']==audit['configuration_fingerprint']==row['configuration_fingerprint'],
        'summary':all(row[k]==episode['result'][k] for k in ('status','diagnosis_unresolved','elapsed_simulation_minutes','action_counts')),
        'configuration_fingerprint':digest(episode['configuration'])==episode['configuration_fingerprint']}
    trace_keys=['case_ref','patient_ref','visit_ref','configuration_fingerprint','workflow','memory','capabilities','result']
    if 'llm_execution' in episode:trace_keys.append('llm_execution')
    checks['normalized_trace_fingerprint']=digest({k:episode[k] for k in trace_keys})==episode['normalized_trace_fingerprint']==row['normalized_trace_fingerprint']
    checks['audit_fingerprint']=digest({k:v for k,v in audit.items() if k!='fingerprint'})==audit['fingerprint']
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}
