"""Phase 3.0 read-only inventories. Never runs RC scenarios or a model."""
from collections import Counter
from pathlib import Path
import subprocess
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.dataset_loader import load_cases
from app.locdi.evaluation_gate_audit import actual_gate_inputs,audit_actual_gate
from app.locdi.development_features import read_artifact
from experiments.evaluation_protocol import metric_contracts,condition_contracts
from validation.official_evaluation_readiness import assess_official_evaluation


def anchored_sources(root, inventory, expected_fingerprint):
    files=inventory['files']
    anchored=digest(files)==expected_fingerprint==inventory['source_fingerprint']
    changed=sum(not (root/p).is_file() or sha(root/p)!=h for p,h in files.items())
    return {'status':'PASS' if anchored and not changed else 'FAIL',
        'anchored_source_count':len(files),'changed_or_missing':changed,
        'inventory_matches_rc_fingerprint':anchored,
        'new_source_files':'ALLOWED_ADDITIONS_NOT_PART_OF_ORIGINAL_RC_FINGERPRINT'}


def inspect_rc():
    from scripts.verify_simulator_release import verify_manifest,stored_offline
    root=ROOT/'outputs/structural_rc/phase212_rc_v1'
    manifest=load(root/'manifests/release_manifest.json')
    sources=anchored_sources(ROOT,load(ROOT/'outputs/private/phase30_rc_source_inventory_v1.json'),manifest['source_code_fingerprint'])
    prior=stored_offline()
    saved={k:load(root/'reports'/(k+'.json')) for k in ('all_case_static','additional_offline',
        'regression','release_decision','artifact_gate','live_er','live_handoff_1','live_handoff_2','live_handoff_3')}
    return {'status':'PASS' if verify_manifest(root,manifest) and sources['status']=='PASS' and prior['status']=='PASS' else 'FAIL',
        'scope':'HASH_AND_SAVED_EXPORT_AUDIT_ONLY_NO_RC_REPLAY','manifest_fingerprint':manifest['fingerprint'],
        'manifest_files':len(manifest['files']),'original_sources':sources,'prior_offline':prior,
        'additional_offline':{k:v for k,v in saved['additional_offline'].items() if k!='records'},
        'static_cases':saved['all_case_static']['case_count'],'stored_regression':saved['regression'],
        'release_decision':saved['release_decision']['decision'],
        'artifact_gate_status':saved['artifact_gate']['research_readiness'],
        'live_audit_statuses':{k:v['status'] for k,v in saved.items() if k.startswith('live_')},
        'raw_wire':'NOT_CHECKED_NOT_SAVED','actual_generation_calls':0}


def paper_inventory():
    result=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*.pdf','-g','*.docx',
        '-g','*.tex','-g','*.md','-g','!.git/**','-g','!**/site-packages/**',
        '-g','!**/node_modules/**','-g','!**/.venv*/**','-g','!**/__pycache__/**'],
        cwd=str(ROOT),capture_output=True,text=True,encoding='utf-8',check=False)
    if result.returncode not in (0,1):raise ValueError('PAPER_SEARCH_FAILED')
    files=sorted(p.replace('\\','/') for p in result.stdout.splitlines())
    # Inventory is evidence for a human to identify the definitive manuscript;
    # implementation reports are not silently accepted as the thesis.
    return {'status':'NOT_FOUND_CONFIRMED_THESIS','paper_equivalence':'NOT_CHECKED',
        'scope':'CURRENT_WORKSPACE_EXCLUDING_GIT_ENVIRONMENTS_AND_NODE_MODULES',
        'format_counts':dict(Counter(Path(p).suffix.lower() for p in files)),
        'files':files,'required_original':['Definitive manuscript and version/date',
        'Memory/Routing/Belief/VoI equations and execution-order sections',
        'B0-B4 table, Handoff policy, common budgets/model/prompt/split',
        'Metric denominators, censoring/abstention policy and evaluation approvals']}


def blocker_plan(codes):
    rows=[]
    for code in sorted(set(codes)):
        if any(k in code for k in ('OVERLAP','DISJOINT','SPLIT','SOURCE_ACTION','CASE','ORDER','PATIENT_SCOPE')):phase,area,need=3,'Dataset independence','Private patient/encounter lineage and disjoint manifests; temporal review'
        elif any(k in code for k in ('PAPER','PROTOCOL','FEATURES')):phase,area,need=1,'Paper alignment','Definitive manuscript and approved condition/order table'
        elif any(k in code for k in ('REFERENCE','REVIEW','RELEVANCE','APPROVAL','METRIC_READINESS')):phase,area,need=2,'Clinical review','Independent source/time/relevance judgments and authenticated authority'
        elif any(k in code for k in ('ARTIFACT','BINDING','UTILITY','VOI','MODEL','REGISTRY','GROUP','SIMILARITY','FITTING','PROVENANCE','OUTCOME','HORIZON','ENCODING','TRANSPORT','STRUCTURAL_CONTRACT')):phase,area,need=4,'Research artifacts','Reviewed fit rows, positive similarity support, validated outcome bindings'
        else:phase,area,need=5,'Official evaluator','Approved common configuration, runner/metric adapters and failure ledger'
        if code in ('ACTUAL_PAIRED_EXECUTION_CONSISTENCY','PAIRED_EXECUTION_APPROVAL_AND_BUDGET_MISSING'):phase,area,need=6,'Paired experiment','All upstream approvals and fixed complete execution manifest'
        rows.append({'blocker_code':code,'priority':phase,'affected_module':area,
            'required_input':need,'responsible_workstream':area,
            'prerequisite':[] if phase==1 else ['PHASE_3_'+str(phase-1)+'_ACCEPTED'],
            'verification_method':'Offline contract/hash checks plus independent signed review where applicable; no self-declared READY',
            'estimated_effort':'UNKNOWN_PENDING_INPUT_AND_REVIEWER_CAPACITY',
            'blocks_official_evaluation':True,'planned_phase':'3.'+str(phase)})
    return sorted(rows,key=lambda r:(r['priority'],r['blocker_code']))


def scoped_executions():
    runs=[('development_pilot','20261029000000000000000000002601'),
        ('development_handoff','20261029000000000000000000002901'),
        ('development_handoff','20261029000000000000000000002902'),
        ('development_handoff','20261030000000000000000000002903')]
    rows=[]
    for folder,run in runs:
        e=load(ROOT/'outputs'/folder/run/'episodes/Case001.json')
        r=e['result']
        rows.append({'run_id':run,'status':r['status'],'finalized':r['finalized'],
            'diagnosis_unresolved_flag':r['diagnosis_unresolved'],
            'request_count':e['llm_execution']['request_count'],
            'identity_check_count':e['llm_execution']['identity_check_count'],
            'memory_retrieval_count':len(e['memory']['retrievals']),
            'actual_action_count':sum(r['action_counts'].values()),
            'elapsed_simulation_minutes':r['elapsed_simulation_minutes'],
            'diagnosis_event_minutes':[x['minute'] for x in e['workflow']['events'] if x['event_type']=='case_completed']})
    return {'scope':'EXACT_RC_HISTORICAL_LIVE_RUNS_NOT_ALL_PROJECT_RUNS_OR_A_PAIRED_BATCH',
        'unique_dataset_case_count':1,'run_count':len(rows),'rows':rows,
        'unrecorded_raw_wire':'NOT_CHECKED','clinical_correctness_evaluable':0}


def audit():
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    configs={n:load(ROOT/'experiments/configs'/(n+'.json')) for n in ('B0','B1','B2','B3','B4')}
    data,options=actual_gate_inputs()
    actual_gate=audit_actual_gate()
    config=load(ROOT/'configs/development_specialty_llm.json')
    registry=load(ROOT/config['llm']['registry_path'])
    voi=read_artifact(config['voi_model_path']) if config['voi_model_path'] else None
    artifact_inputs=dict(registry_manifest=registry,
        routing_artifact=read_artifact(config['routing_utility_path']) if config['routing_utility_path'] else None,
        voi_artifact=voi,correctness_inputs=(data,options),
        evaluation_patient_ids=[c['patient_id'] for c in cases],
        source_actions=[e for c in cases for e in c['environment']['current_encounter_events']],
        diagnosis_candidates=None)
    split={k:list(v) for k,v in data['split_manifest']['patients'].items()}
    split['test']=[c['patient_id'] for c in cases]
    gate=assess_official_evaluation(configs,cases=cases,case_sets={n:cases for n in configs},
        split_manifest=split,artifact_inputs=artifact_inputs)
    events=[e for c in cases for e in c['environment']['current_encounter_events']]
    inventory={'locdi_cases':len(cases),'locdi_patients':len({c['patient_id'] for c in cases}),
        'source_sha256':sha(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'),
        'past_memory_records':sum(len(v['records']) for c in cases for v in c['agent_visible']['past_visits']),
        'source_action_counts':dict(Counter(e['action'] for e in events)),
        'available_minute_counts':dict(Counter(str(e['available_minute']) for e in events)),
        'independent_metric_ready_cases':{k:sum(c['task_annotations']['metric_readiness'].get(k) is True for c in cases)
            for k in ('diagnosis_accuracy','memory_retrieval','routing_accuracy','voi_regret')},
        'split_counts':{k:len(v) for k,v in split.items()},
        'specialty_responses':len(data['outcomes']),
        'certainty_counts':dict(Counter(o['response']['certainty'] for o in data['outcomes'])),
        'clinical_correctness_nonnull':sum(o['correct'] is not None for o in data['outcomes']),
        'collection_groups':len(data['decisions']),
        'reviewed_complete_train_groups':actual_gate['complete_train_groups'],
        'reviewed_validation_groups':actual_gate['validation_group_count'],
        'positive_similarity_support':actual_gate['positive_similarity_support'],
        'candidate010_timing':[{'case_ref':r['case_ref'],'reference_timing_status':r['reference_timing_status'],
            'source_condition_start_after_decision_days':r['source_condition_start_after_decision_days']}
            for r in actual_gate['groups'] if r['case_ref']=='Candidate010'],
        'clinical_readiness':'BLOCKED','timing_scope':'SOURCE_EVENT_PROXY_NOT_INDEPENDENTLY_REVIEWED_ACTUAL_RELEASE'}
    observed=scoped_executions()
    contracts=metric_contracts()
    for row in contracts:
        row['official_eligible_data_count']=0
        row['observed_data_scope']='RC historical ER + three Handoff run records; Case001 only; separate collection 75 responses'
        m=row['metric_id']
        row['reference_available_count']=(0 if m in ('diagnosis_correctness','memory_precision_at_k',
            'memory_recall_at_k','memory_mrr','strict_routing_accuracy','acceptable_specialty_accuracy',
            'routing_success','action_quality','information_gain','unnecessary_test_rate')
            else 'NOT_APPLICABLE_DESCRIPTIVE_METRIC')
        row['descriptive_observation_count']=(7 if m.startswith('memory_') else 2 if m in ('time_to_final_diagnosis','unresolved_rate')
            else 3 if m in ('repair_count','retry_count','unnecessary_test_rate') else 5 if m=='action_quality'
            else 4 if m in ('finalization_rate','completion_rate','action_count','generation_requests','failure_rate','budget_exhaustion_rate') else 0)
        row['observation_unit']=('retrieval_snapshot' if m.startswith('memory_') else 'executed_action' if m=='action_quality'
            else 'test_order' if m=='unnecessary_test_rate' else 'historical_run_record')
    codes=gate['blocker_codes']+actual_gate['blocker_codes']+['ROUTING_VALIDATION_OVERLAPS_VOI_TRAINING',
        'PAIRED_EXECUTION_APPROVAL_AND_BUDGET_MISSING']
    for kind,a in gate['artifact_gate']['artifacts'].items():
        codes.extend(kind.upper()+'_'+c for c in a['blocker_codes'])
    return {'schema_version':'locdi.research_protocol_audit.v1','research_evaluation_ready':False,
        'rc':inspect_rc(),'paper':paper_inventory(),'conditions':condition_contracts(configs),
        'data':inventory,'scoped_executions':observed,'correctness_gate':actual_gate,'official_gate':gate,
        'metric_contracts':contracts,'blocker_plan':blocker_plan(codes),
        'current_order':'ER_LLM_PROPOSAL_THEN_BELIEF_VOI_THEN_VALIDATED_ACTION',
        'belief_conditioned_proposal_hook':False,'clinical_metrics':'NOT_EVALUATED',
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}
