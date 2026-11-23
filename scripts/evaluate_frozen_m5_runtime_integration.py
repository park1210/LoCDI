"""C4-37 read-only Source audit/projection. Actual Hospital probe remains gated."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.llm_setup import digest
from app.locdi.voi_patient_independence_review import SourceIndex
from app.locdi.frozen_m5_runtime_adapter import bind_value,candidate_case,runtime_readiness,M5_SHA
from app.locdi.source_memory_projection import project_past_observations

PUB=ROOT/'outputs/research_evaluation/c4_37_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_37_v1'
P35=ROOT/'outputs/private/research_evaluation/c4_35_v1'
M5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
RAW=ROOT/'outputs/private/research_evaluation/c4_34_v1/raw'
CODE=['app/locdi/frozen_m5_runtime_adapter.py','app/locdi/source_memory_projection.py',
      'scripts/evaluate_frozen_m5_runtime_integration.py','tests/locdi/test_frozen_m5_runtime_adapter.py']


def prepare():
    pins=check(read(PR/'protected_hashes.json'))
    if pins['changed_or_missing']:raise ValueError('PROTECTED_ASSET_CHANGED')
    write(PR/'initial_preservation.json',pins)
    if sha(M5)!=M5_SHA:raise ValueError('FROZEN_M5_HASH')
    admission=read(P35/'approved_admission.json');panel=sorted(admission['panel'],key=lambda v:(v['patient_id'],v['visit_id']))
    assert digest(admission['panel'])==admission['panel_fingerprint']
    assert len(panel)==71 and len({v['patient_id'] for v in panel})==35
    sources={str(M5.resolve()):sha(M5),str((P35/'approved_admission.json').resolve()):sha(P35/'approved_admission.json')}
    for p in [ROOT/'outputs/research_evaluation/c4_35_v1/bounded_evaluation_protocol.json',
              ROOT/'outputs/research_evaluation/c4_36_v1/memory_voi_compatibility.json']:
        sources[str(p.resolve())]=sha(p)
    protocol={'version':'locdi.frozen_m5.runtime_integration_audit.c4_37.v1',
        'model_sha256':M5_SHA,'admission_fingerprint':admission['panel_fingerprint'],
        'selection':'first5 of existing admitted panel sorted by patient_id,visit_id; no performance selection',
        'registered_panel_patients':35,'registered_panel_visits':71,'future_probe_max_visits':5,
        'horizon':'Frozen M5 scalar horizon2; not dynamic, not C4_35 H3 performance rerun',
        'action_menu':'unchanged8','max_information_actions_per_visit':5,
        'tolerance':{'probability_Q_VoI_T_absolute':1e-12,'relative':0,'decision_ids_and_STOP':'EXACT; original epsilon/tie'},
        'source_values':'exact Code/System/Unit/identity/Raw row fingerprint; no unit conversion',
        'release':'existing Recorded-Time Proxy; no clinical duration/resource imputation',
        'projection':'Observation-only ended past visits; recorded time proxy schema feasibility, not clinical release approval',
        'actual_policy_or_Hospital_execution_authorized':False,'new_probabilities':False,
        'code_hashes':{n:sha(ROOT/n) for n in CODE},'input_hashes':sources,
        'budgets':{'projection_records_cap':1000000,'private_projection_bytes_cap':1024**3,
            'future_probe_planner_calls':30,'Core_max_states_per_call':20000,'future_probe_executed_states':600000,
            'future_probe_wall_seconds':120,'future_probe_private_memory_bytes':512*1024**2,
            'LLM_generation':0,'fitting':0},'future_probe_status':'BLOCKED_EXISTING_GATE; needs compatible interface and new explicit approval'}
    protocol['fingerprint']=digest(protocol)
    write(PUB/'source_backed_runtime_probe_protocol.json',protocol)
    write(PR/'selected_future_probe_references.json',[{'patient_id':v['patient_id'],'visit_id':v['visit_id'],
         'C4_35_ordinal':i} for i,v in enumerate(panel[:5],1)])
    print(json.dumps({'status':'PREPARED_NO_EXECUTION','protocol':protocol['fingerprint']}))


def verify():
    protocol=read(PUB/'source_backed_runtime_probe_protocol.json')
    if protocol['fingerprint']!=digest({k:v for k,v in protocol.items() if k!='fingerprint'}):raise ValueError('PROTOCOL_HASH')
    for group in ['input_hashes','code_hashes']:
        if check(protocol[group])['changed_or_missing']:raise ValueError('SOURCE_OR_CODE_CHANGED')
    return protocol


def audit():
    protocol=verify();m=read(M5);admission=read(P35/'approved_admission.json')
    panel=sorted(admission['panel'],key=lambda v:(v['patient_id'],v['visit_id']))
    index=SourceIndex(ROOT/'outputs/private/research_evaluation/c4_34_v1/lossless_source_cache_v2.sqlite')
    bindings=[];projection_summaries=[];memory_total=0;candidate_count=0;native_count=0;native_visit_count=0
    native=read(P35/'native_lineage_private.json')['new'];cache={};raw_counts=[]
    projection_path=PR/'new_observation_memory_projection.jsonl'
    with projection_path.open('x',encoding='utf-8') as projected, (PR/'engine_private_candidates.jsonl').open('x',encoding='utf-8') as candidates:
        try:
            for ordinal,v in enumerate(panel,1):
                if v['patient_id'] not in cache:
                    tables=index.tables(v['patient_id']);cache[v['patient_id']]=tables
                    entry=native[v['patient_id']];native_file=Path(entry['source_file'])
                    n=read(native_file)
                    if n['attributes']['id']!=v['patient_id']:raise ValueError('NATIVE_PATIENT_LINK')
                    native_count+=1
                    raw_counts.append({'patient_id':v['patient_id'],'native_source_file':str(native_file),
                        'native_sha256':sha(native_file),'native_encounter_ids':[e['uuid'] for e in n['record']['encounters']]})
                tables=cache[v['patient_id']]
                encounter=next((r for _,r in tables['encounters'] if r['Id']==v['visit_id']),None)
                if encounter is None or encounter['START']!=v['decision_time']:raise ValueError('ENCOUNTER_LINK')
                lineage=next(x for x in raw_counts if x['patient_id']==v['patient_id'])
                if v['visit_id'] not in lineage['native_encounter_ids']:raise ValueError('NATIVE_ENCOUNTER_LINK')
                native_visit_count+=1
                references={r['DESCRIPTION'] for _,r in tables['conditions'] if r['ENCOUNTER']==v['visit_id'] and r['DESCRIPTION'] in m['diagnoses']}
                if references!=set(v['reference_candidates']) or references!={v['diagnosis']}:raise ValueError('REFERENCE_LINEAGE_CHANGED')
                for key in m['provenance']['action_key_mapping']:
                    row=bind_value(m,v,key);e=v['values'][key]
                    if Path(e['source_file']).resolve()!=(RAW/'csv/observations.csv').resolve():raise ValueError('RAW_SOURCE_FILE')
                    raw=index.db.execute('SELECT data FROM observations WHERE line=?',(e['source_row'],)).fetchone()
                    if raw is None or json.loads(raw[0])!=e['source_record']:raise ValueError('CSV_ROW_LINK')
                    bindings.append({'case_ordinal':ordinal,**row})
                identity={'patient_id':v['patient_id'],'visit_id':v['visit_id'],'started_at':v['decision_time']}
                projection=project_past_observations(identity,tables['encounters'],tables['observations'],accept_recorded_time_proxy=True)
                memory_total+=projection['summary']['records']
                if memory_total>protocol['budgets']['projection_records_cap']:raise ValueError('PROJECTION_ROW_CAP')
                projected.write(json.dumps({'case_ordinal':ordinal,'identity':identity,**projection},ensure_ascii=False,allow_nan=False)+'\n');projected.flush()
                if projected.tell()>protocol['budgets']['private_projection_bytes_cap']:raise ValueError('PROJECTION_STORAGE_CAP')
                candidate=candidate_case(m,v,projection['past_visits'])
                candidates.write(json.dumps({'case_ordinal':ordinal,'engine_candidate':candidate},ensure_ascii=False,allow_nan=False)+'\n')
                candidate_count+=1;projection_summaries.append({'case_ordinal':ordinal,**projection['summary']})
        finally:index.close()
    write(PR/'binding_lineage_private.json',bindings);write(PR/'native_coverage_private.json',raw_counts)
    matrix=[]
    for key,aid in m['provenance']['action_key_mapping'].items():
        kind,system,code,unit,resource=json.loads(key);a=next(a for a in m['actions'] if a['id']==aid)
        matrix.append({'code':code,'system':system,'unit':unit,'resource':resource,'type':a['type'],
            'target':a['target'],'action_id':aid,'rows':sum(r['source_key']==key for r in bindings),
            'status':'ADAPTER_COMPATIBLE','source_identity':'EXACT_BINDING','bounds':m['provenance']['binning_thresholds'][key],
            'availability':'RECORDED_TIME_PROXY','direct_original50_binding':'unchanged0/250',
            'clinical_system_verification':'CSV lacks explicit system, no LOINC namespace inferred'})
    write(PUB/'source_action_binding_matrix.json',{'registered_visits':71,'bindings':len(bindings),'actions':matrix,
        'raw_row_hash_links':'PASS','native_patient_links':native_count,'native_current_encounter_links':native_visit_count,
        'native_all_observation_value_lineage':'NOT_RECHECKED; raw CSV exact-row scope used for value/recorded-time projection',
        'classification_axes':'exact Source identity separate from proposed Hospital adapter compatibility'})
    readiness=runtime_readiness(m)
    write(PUB/'runtime_source_adapter_status.json',{**readiness,'engine_private_candidates_schema_pass':candidate_count,
        'full_LoCDI_wrapper_created':False,'initial_state':'empty unknown doctor/patient fields; no fabricated complaint/vitals',
        'test_timing':'MISSING_SOURCE_DURATION_RESOURCES; would fall back to simulation defaults, not permitted as source truth'})
    write(PUB/'outcome_encoding_compatibility.json',{'frozen_encoding':m['provenance']['observation_encoding'],
        'existing_runtime_tags':[None,'replay_bins_v1'],'category_rule':'str(sum(value>frozen_boundary))',
        'boundary_equality':'lower bin; equal cutpoints remain unchanged, no bin merging',
        'standalone_released_category_view':'proposed meaning-preserving finite-category view; TEST_ONLY math tested, not installed',
        'model_mutated':False,'gate_relaxed':False,'full_runtime_classification':'C_CORE_GATE_CONTRACT_BLOCKED',
        'future_option':'separately reviewed external observation-view interface; preserve model tag and require exact Source binding, no current activation'})
    write(PUB/'terminal_diagnosis_compatibility.json',{'replay':'terminal_choice maximizes unreviewed dimensionless1/0 Utility',
        'hospital':'diagnose passes catalog/certainty/evidence/allow_unresolved validators',
        'diagnostic_support_outcomes':sum(bool(o['supports_diagnoses']) for a in m['actions'] for o in a['outcomes']),
        'supported_diagnosis':'BLOCKED_NO_MODEL_OUTCOME_SUPPORT; source reference cannot be cited as evidence',
        'unresolved':'allowed only if original allow_unresolved contract permits, not forced or a correct diagnosis',
        'source_reference':'private unreviewed Source label only; timing not clinical attestation',
        'runtime_terminal_executed':False})
    totals=Counter()
    for row in projection_summaries:totals.update(row['excluded_rows'])
    write(PUB/'memory_source_projection_status.json',{'status':'MEMORY_CONTEXT_READY_ONLY_SCHEMA_VALIDATED',
        'visits':71,'native_patients_inspected':native_count,'native_current_encounters_linked':native_visit_count,
        'projected_record_instances':memory_total,'record_count_meaning':'visit-specific snapshots; same past row can recur across current visits',
        'visits_with_records':sum(r['records']>0 for r in projection_summaries),
        'by_visit':projection_summaries,'excluded_rows':dict(totals),'schema':'existing10fields PASS',
        'conditions_and_current_reference_projected':False,'LLM_context_delivery_executed':False,
        'actual_retrieval_or_performance_run':False,'availability':'RECORDED_TIME_PROXY actual clinical release NOT_VERIFIED',
        'original72590_changed':False,'concept_diverse_changed':False})
    write(PUB/'memory_role_contract.json',{'thesis_role':'THESIS_NOT_AVAILABLE',
        'context_only':'existing role input route implemented; new projections schema-ready only; no actual LLM effect measured',
        'Bayesian_evidence':'M5 has0 memory likelihood actions; not classified as a thesis defect without manuscript',
        'retrieve_memory_action':'M5 has0 such actions; no fabricated likelihood added',
        'D_MEMORY_EVIDENCE_MODEL_REQUIRED':'CONDITIONAL_ONLY_IF_FUTURE_VERIFIED_THESIS_REQUIRES_IT'})
    # Read one archived H2/budget5 path for each preregistered ordinal1..5.
    # No planner invocation, posterior replay, Reference scoring, or policy rerun.
    found=[]
    with (P35/'evaluation_run/trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);ordinal=row['scenario_ordinal']
            if ordinal<=5 and row['policy']=='voi_h2' and row['budget']==5:
                v=panel[ordinal-1];ids=[]
                for release in row['releases']:
                    key=next(k for k,a in m['provenance']['action_key_mapping'].items() if a==release['action_id'])
                    b=bind_value(m,v,key)
                    if b['outcome_id']!=release['outcome_id'] or release['released_at_minute']<b['available_minute']:raise ValueError('ARCHIVED_RELEASE_MISMATCH')
                    ids.append(release['action_id'])
                if len(ids)!=len(set(ids)):raise ValueError('ARCHIVED_ACTION_REPEAT')
                found.append({'case_ordinal':ordinal,'archived_release_count':len(ids),'frozen_category_links':'PASS',
                    'prior_fingerprint':digest(row['decisions'][0]['belief_before']) if row['decisions'] else digest(m['prior']),
                    'Q_values':'NOT_RECORDED_IN_ARCHIVED_PER_DECISION_TRACE','new_Hospital_counterpart':'NOT_RUN'})
    if len(found)!=5:raise ValueError('ARCHIVED_PATHS_MISSING')
    write(PUB/'runtime_replay_equivalence.json',{'status':'ACTUAL_HOSPITAL_COMPARISON_NOT_RUN_GATE_BLOCKED',
        'archived_path_checks':found,'scope':'read-only action/category/release links on5 existing paths; no policy score recomputation',
        'TEST_ONLY_math':'posterior/T/Q/VoI/root exact IDs tested at fixed1e-12 abs tolerance; see test_results',
        'actual_Runtime_math_equivalence':'NOT_ESTABLISHED','old_metrics_changed':False})
    write(PUB/'source_backed_runtime_probe_approval_status.json',{'status':'BLOCKED_CONTRACT_BEFORE_APPROVAL',
        'new_approval':'NOT_PROVIDED','prior_C4_35_approval_reused':False,'actual_source_runtime_execution':0,
        'reason':'encoding Gate, observer interface, Source resource/time and terminal contract unresolved; approval alone insufficient'})
    write(PUB/'source_backed_runtime_probe_results.json',{'status':'NOT_RUN','new_Hospital_invocations':0,
        'new_source_policy_scores':0,'source_adapter_audit_visits':71,'exact_source_bindings':len(bindings),
        'memory_schema_projections':71,'C4_36_TEST_ONLY_conditions_repeated':0,'LLM_generation':0})
    write(PUB/'thesis_contract_alignment.json',{'status':'THESIS_NOT_AVAILABLE','evidence':'user explicitly replied no manuscript after C4_36',
        'research_goal_questions_framework_memory_routing_Bayes_VoI_Dynamic_B0_B4_data_metrics_success_equation_numbers':'THESIS_NOT_AVAILABLE',
        'thesis_modified':False,'implementation_reports_substituted':False})
    write(PUB/'thesis_claim_evidence_mapping.json',{'status':'THESIS_NOT_AVAILABLE',
        'code_protocol_evidence_matrix':'outputs/research_evaluation/c4_36_v1/research_evidence_matrix.json',
        'audit_formula_mapping':'outputs/research_evaluation/c4_36_v1/formula_to_code_mapping.json',
        'paper_claim_or_equation_mapping':None,'Memory_role_A_B_C_D':None})
    rows=[]
    for i in range(3):
        config=read(ROOT/('experiments/configs/B%d.json'%i))
        rows.append({'condition':'B%d'%i,'Adaptive_utility_required':False,'VoI_model_required':False,
            'features':config['features'],'ready':config['ready'],
            'missing_common':[k for k,v in config['common'].items() if v is None],
            'blockers':['OFFICIAL_RUNNER_NOT_IMPLEMENTED','RESEARCH_PURPOSE_BLOCKED',
                'SHARED_MODEL_REGISTRY_PROMPT_DECODING_VERSION_NOT_PINNED_IN_OFFICIAL_CONFIG',
                'EVALUATION_REFERENCE_SPLIT_ACTION_TIME_REPEAT_CONTRACT_NOT_CLEARED'],
            'frozen_M5_adapter_is_not_prerequisite_for_B0_B2':True})
    write(PUB/'official_b0_b2_blocker_matrix.json',{'conditions':rows,
        'B3_B4':'additional MISSING_UTILITY, B4 compatible Source/VoI model/evidence and horizon contract',
        'official_files_changed':False,'Gate_changed':False})
    print(json.dumps({'status':'SOURCE_AND_MEMORY_AUDIT_COMPLETE_RUNTIME_BLOCKED','bindings':len(bindings),
        'projected_record_instances':memory_total,'candidate_schemas':candidate_count,'archived_paths_read':len(found)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','audit','execute'],nargs='?',default='prepare');a=p.parse_args()
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('OFFLINE_ONLY')
    sys.addaudithook(guard)
    if a.stage=='prepare':prepare()
    elif a.stage=='audit':audit()
    else:print(json.dumps({'status':'RUNTIME_CONTRACT_BLOCKED','executed':False,'requires':'separate reviewed interface/Source-time-terminal contract and new approval; C4_35 approval not reused'}));sys.exit(2)
