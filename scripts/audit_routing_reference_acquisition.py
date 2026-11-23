"""C4-16 offline, new-namespace reference acquisition audit. No new routing policy."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.routing_data_reuse import read,BANK
from app.locdi.source_enrichment import validate_fact,iso_ms
from app.core.longitudinal_memory import timestamp
from app.locdi.routing_two_decision_collection import verify_pins
from app.locdi.source_routing_reference import (blind_packet,second_stage,narrow_candidate,summarize,benchmark,rule_spec,assert_blind)
from scripts.enrich_full_cohort_sources import write,rows


def run(namespace):
    if namespace!='c4_16_v1':raise ValueError('FIXED_AUDIT_NAMESPACE_REQUIRED')
    pub=ROOT/'outputs/research_evaluation'/namespace;pri=ROOT/'outputs/private/research_evaluation'/namespace
    if not (pri/'initial_preservation.json').is_file():raise ValueError('PROTECTION_BASELINE_REQUIRED')
    if (pub/'verification.json').exists():raise ValueError('AUDIT_ALREADY_COMPLETED_NO_RESUME')
    pins=load(pri/'protected_hashes.json');sources=load(pri/'source_hashes.json')
    mapping=load(pub/'source_mapping_evidence_registry.json')
    write(pub/'narrow_reference_rule.json',rule_spec())
    parent=ROOT/'outputs/private/research_evaluation/c4_15_v1'
    cases=load(parent/'cases.json');source_packets=load(parent/'decision_source_packets.json')
    response_analysis=load(parent/'response_source_analysis.json')['rows']
    facts15={r['case_ref']:r for r in read(ROOT/'outputs/private/research_evaluation/c4_12_v1/source_decision_eligibility.jsonl')}
    native_index=read(ROOT/'outputs/private/research_evaluation/c4_05_v1/case_source_index.jsonl')
    packetfiles={};clinical_forms=[]
    blind_dir=pri/'track_a_blind';blind_dir.mkdir()
    for c in cases:
        f=facts15[c['case_ref']];t=timestamp(c['decision_time'])
        descriptor=next(x for x in native_index if x['dataset']=='TrainingBank' and x['patient_id']==c['patient_id'])
        native=load(descriptor['raw_path'])
        starts={e['uuid']:iso_ms(e['start']) for e in native['record']['encounters']}
        def projected(n):
            return {'source_fingerprint':n['source_fingerprint'],'source_codes':deepcopy(n['codes']),
                'recorded_onset':n['start'],'current_encounter':n['current_encounter'],
                'encounter_started_at':starts[n['encounter_id']],
                'actual_available_at':None,'reference_status':'NOT_REVIEWED'}
        current=[projected(n) for n in f['native_condition_inventory'] if n['current_encounter'] and timestamp(n['start'])<=t]
        past=[projected(n) for n in f['native_condition_inventory'] if not n['current_encounter'] and timestamp(n['start'])<=t and timestamp(starts[n['encounter_id']])<t]
        packet=blind_packet(c,current,past)
        packet['source_current_condition_count_including_quarantined']=f['current_encounter_condition_count']
        packet['future_conditions_excluded']=sum(timestamp(n['start'])>t for n in f['native_condition_inventory'])
        packet['private_source_ledger']=deepcopy(f['visibility_ledger'])
        assert_blind(packet)
        path=blind_dir/(c['case_ref']+'.json');write(path,packet);packetfiles[path.relative_to(ROOT).as_posix()]=sha(path)
        clinical_forms.append({'case_ref':c['case_ref'],'decision_id':c['decision_id'],
            'observable_input_fingerprint':c['observable_input_fingerprint'],'blind_packet_fingerprint':digest(packet),
            'observation_sufficient_for_referral':None,'accepted_specialties':None,'multiple_specialties_acceptable':None,
            'router_abstention_appropriate':None,'additional_questions_or_tests_needed':None,'not_evaluable':None,
            'source_timing_status':'NOT_REVIEWED','recorded_time_proxy_review':'NOT_REVIEWED',
            'reviewer_kind':None,'reviewer_qualification':None,'reviewer_private_ref':None,'reviewer_independent':None,
            'reviewed_at':None,'reviewer_rationale':None,'independent_review_status':'NOT_REVIEWED'})
    write(pri/'independent_reference_review_blank.json',{'version':'locdi.routing_independent_review.c4_16.v1','reviews':clinical_forms})
    stage2=second_stage(cases,response_analysis)
    path=pri/'track_a_second_stage_agent_responses.json';write(path,stage2)
    write(pub/'independent_reference_packet_manifest.json',{'stage1_blind_decision_packets':2,'stage2_agent_responses':10,
        'stage1_directory':'outputs/private/research_evaluation/c4_16_v1/track_a_blind/',
        'stage2_private_file':'outputs/private/research_evaluation/c4_16_v1/track_a_second_stage_agent_responses.json',
        'packet_file_sha256':packetfiles,'stage2_sha256':sha(path),'stage1_review_completed':False,
        'independent_clinical_reference_acquired':0,'reviewer_created':False,'router_and_prediction_hidden_from_stage1':True,
        'future_conditions_quarantined':True,'mapping_and_policy_results_hidden_from_stage1':True})

    descriptors=[r for r in read(ROOT/'outputs/private/research_evaluation/c4_05_v1/case_source_index.jsonl') if r['dataset'] in ['Original50','TrainingBank']]
    descriptor_by_ref={(r['dataset'],r['case_ref']):r for r in descriptors}
    grouped=defaultdict(list);bycase=defaultdict(list)
    # Existing full-cohort factual export, not Source Labels or model predictions.
    with (ROOT/'outputs/private/research_evaluation/c4_05_v1/source_facts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            f=json.loads(line)
            if f['dataset'] in ['Original50','TrainingBank'] and f['source_resource_type']=='Condition':
                grouped[f['source_file']].append(f)
    verified=0
    for i,(name,fs) in enumerate(sorted(grouped.items()),1):
        source=Path(name);expected=sources.get(str(source))
        if expected is None or sha(source)!=expected:raise ValueError('RAW_SOURCE_NOT_PINNED')
        raw=load(source)
        for f in fs:
            validate_fact(f,raw,expected)
            d=descriptor_by_ref[(f['dataset'],f['case_ref'])]
            if f['patient_reference']!=d['patient_id'] or f['visit_reference']!=d['visit_id']:raise ValueError('DESCRIPTOR_IDENTITY_MISMATCH')
            f['source_verified']=True
            bycase[(f['dataset'],f['case_ref'])].append(f);verified+=1
        if i%50==0:print('SOURCE_POINTER_FILES',i,flush=True)
    registry=load(parent/'registry.json');departments=sorted(a['department'] for a in registry['agents'] if a['role']=='specialty')
    cohort=[];allfacts={}
    for d in descriptors:
        c={'dataset':d['dataset'],'case_ref':d['case_ref'],'patient_id':d['patient_id'],'visit_id':d['visit_id'],
            'decision_time':d['decision_time'],'candidate_departments':departments,'development_exposed':True,
            'independent_evaluation_eligible':False}
        cohort.append(c);allfacts[c['case_ref']]=bycase[(d['dataset'],d['case_ref'])]
    bundle=load(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json')
    bankvisits={(d['patient_id'],d['visit_id']):d for d in descriptors if d['dataset']=='TrainingBank'}
    for c in bundle['candidates']:
        d=c['decision'];descriptor=bankvisits[(d['patient_id'],d['visit_id'])]
        cohort.append({'dataset':'Prepared15','case_ref':c['case_ref'],'patient_id':d['patient_id'],'visit_id':d['visit_id'],
            'decision_time':facts15[c['case_ref']]['decision_time'],'candidate_departments':departments,
            'development_exposed':True,'independent_evaluation_eligible':False})
        allfacts[c['case_ref']]=bycase[('TrainingBank',descriptor['case_ref'])]
    if Counter(c['dataset'] for c in cohort)!={'Original50':50,'TrainingBank':229,'Prepared15':15}:raise ValueError('FULL_COHORT_COUNT_MISMATCH')
    if len({c['case_ref'] for c in cohort})!=294:raise ValueError('DUPLICATE_REGISTERED_CASE')
    write(pri/'registered_cohort.json',cohort)
    audit=[narrow_candidate(c,allfacts[c['case_ref']],mapping) for c in cohort]
    rows(pri/'source_candidate_audit.jsonl',audit)
    refs=[r for r in audit if r['state'] in ['POSITIVE','NEGATIVE']]
    rows(pri/'source_grounded_reference.jsonl',refs)
    rows(pub/'source_grounded_reference.jsonl',[])  # No admitted labels in current actual source.
    if refs:raise ValueError('NEW_ADMITTED_REFERENCE_REQUIRES_PRIVATE_PUBLIC_EXPORT_REVIEW')
    coverage={ds:{**summarize([r for r in audit if r['dataset']==ds]),
        'unique_patients':len({c['patient_id'] for c in cohort if c['dataset']==ds}),
        'development_exposed_cases':sum(c['dataset']==ds for c in cohort),'independent_evaluation_eligible_cases':0} for ds in ['Prepared15','Original50','TrainingBank']}
    write(pub/'source_grounded_coverage.json',{'cohorts':coverage,'registered_scope_occurrences':len(cohort),
        'distinct_patient_encounter_pairs':len({(c['patient_id'],c['visit_id']) for c in cohort}),
        'unique_patients_across_scopes':len({c['patient_id'] for c in cohort}),
        'overlap_note':'Prepared15 is a subset of TrainingBank visits with minute1 decisions; cohort occurrence counts are not independent patient counts',
        'verified_original_current_condition_occurrences':verified,'raw_patient_files_checked':len(grouped),
        'actual_source_release_timestamps':sum(bool(f['available_at']) for fs in grouped.values() for f in fs)})
    write(pub/'label_quality_audit.json',{'admitted_reference_records':len(refs),'clinical_reference_records':0,
        'mapping_exact_code_entries':len(mapping['mappings']),'formal_clinical_code_to_specialty_crosswalk':False,
        'source_terminology_release':'UNKNOWN','source_condition_fact_scope':'Actual native code/system/encounter/onset; explicit clinical availability absent',
        'unmapped_is_not_negative':True,'multiple_conditions':'Unknown domain or timing blocks complete set admission',
        'source_field_pointer_values_and_hashes_verified':verified,'independent_clinical_verdicts_generated':0,
        'future_source_not_promoted':True,'silver_merged_with_independent_reference':False,'mapping_selected_for_measured_accuracy':False})
    # Read existing decisions; do not execute a new benchmark on no admitted labels.
    old=load(ROOT/'outputs/research_evaluation/c4_15_v1/policy_execution_trace.json')
    write(pub/'routing_benchmark_results.json',{**benchmark([c['case_ref'] for c in cohort],[],refs),
        'adaptive_accuracy':'MISSING_UTILITY','previous_c4_15_decision_trace':old['rows'],
        'previous_trace_is_not_c4_16_benchmark':True,'new_actual_benchmark_router_calls':0})
    orig=read(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    prepinputs=[c['shared_payload']['observation'] for c in bundle['candidates']]
    originalinfo=[c['agent_visible']['current_initial_state']['doctor_visible_info'] for c in orig]
    terms=[m['concept'] for m in mapping['mappings']]
    memory_mentions=sum(any(term.casefold() in r.get('text','').casefold() for term in terms) for o in prepinputs for r in o['longitudinal_memory']['records'])
    source_keywords={m['mapping_id']:{'specialty':m['specialties'][0],
        'concept_word_overlap_with_registry_keywords':[kw for a in registry['agents'] if a['department'] in m['specialties'] for kw in a['keywords'] if set(kw.casefold().split())<=set(m['concept'].casefold().split())]} for m in mapping['mappings']}
    write(pub/'circularity_and_leakage_audit.json',{'status':'CIRCULAR_EVALUATION_RISK','same_literal_label_function_as_router':False,
        'label_function_inputs':['exact_code_system','exact_code','patient_encounter_link','explicit_release_time','multi_condition_scope'],
        'policy_inputs':'Doctor-observable text and Registry keywords; no source condition appendices or review labels',
        'semantic_dependency':source_keywords,'prepared_memory_records':sum(len(o['longitudinal_memory']['records']) for o in prepinputs),
        'prepared_memory_records_with_exact_mapped_concept_phrase':memory_mentions,
        'historical_diagnosis_text_is_not_automatic_gold_injection':True,'current_condition_fields_not_in_prepared_policy_snapshot':True,
        'prepared_explicit_policy_inputs':15,'original_initial_doctor_views_inspected':len(originalinfo),
        'original_longitudinal_router_snapshot_not_newly_constructed':True,'trainingbank_full_router_snapshots_not_newly_constructed':True,
        'original50_development_exposed':True,'mapping_inventory_informed_development_scope':True,
        'new_blind_holdout_claimed':False,'independent_accuracy_computed':False,
        'c4_10_identity_reference_not_repurposed_as_specialty_gold':True,
        'future_memory_cutoff_policy_changed':False})
    previous_split=load(ROOT/'outputs/research_evaluation/c4_15_v1/split_and_leakage_audit.json')
    write(pub/'evaluation_readiness.json',{'final_status':'C4_16_REFERENCE_ACQUISITION_BLOCKED',
        'track_a':'BLIND_PACKET_READY_INDEPENDENT_REVIEW_NOT_RECEIVED','track_b':'SOURCE_TIME_AND_COMPLETE_MAPPING_BLOCKED',
        'independent_clinical_references':0,'admitted_source_grounded_references':0,'verified_accuracy_utility':0,
        'routing_single_module_validation':'Validation5 available only for development, not independent integrated evaluation',
        'integration_validation_voi_train_overlap':previous_split['routing_validation_voi_train_overlap'],
        'new_holdout_requirement':'New patients disjoint from all Routing and VoI train/development exposure, with frozen availability/condition/task reference',
        'llm_generation':0,'utility_fitting':0,'research_evaluation_ready':False})
    write(pub/'verification.json',{'status':'PASS','registered_scope_occurrences':len(cohort),'source_fact_occurrences_verified':verified,
        'blind_packets':2,'second_stage_actual_responses':len(stage2['rows']),'independent_review_created':False,
        'reference_records_admitted':len(refs),'new_routing_policy':False,'existing_core_or_gate_changed':False,
        'protected_files':len(pins),'source_files':len(sources),'initial_preservation':'PASS','llm_generation':0,'fitting':0})
    print('C4_16_REFERENCE_ACQUISITION_BLOCKED',coverage,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--namespace',default='c4_16_v1');a=p.parse_args()
    def guard(event,args):
        if event=='socket.connect':raise RuntimeError('OFFLINE_AUDIT_ONLY')
    sys.addaudithook(guard)
    run(a.namespace)
