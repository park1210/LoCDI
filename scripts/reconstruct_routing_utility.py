"""C4-11 offline evidence audit and conditional fitting. No model construction."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import load_preparation,PRIVATE_ROOT
from app.locdi.specialty_collection_analysis import analyze_run
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from app.locdi.routing_collection_preparation import BANK,SOURCE,CATALOG,TAXONOMY,CONFIG
from app.locdi.routing_training_setup import REGISTRY,HOLDOUT,HOSPITAL
from app.locdi.routing_data_reuse import read
from app.locdi.routing_utility_reconstruction import (VERSION,source_code_bindings,reference_facts,response_evidence,
    fit_verified,build_development,execute_conditions,execution_summary)
from scripts.evaluate_adaptive_routing import case_contract
from scripts.classify_pre_review_evidence import jsonl
from scripts.enrich_full_cohort_sources import write
from scripts.audit_memory_review_progress import check_hashes


def protected_assets():
    prior=ROOT/'outputs/private/research_evaluation/c4_10_v1'
    pins=load(prior/'protected_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_10_v1/output_manifest.json';m=load(mp)
    if digest({k:v for k,v in m.items() if k!='fingerprint'})!=m['fingerprint']:raise ValueError('PARENT_MANIFEST')
    ip=prior/'private_index.json'
    if sha(ip)!=m['private_index_sha256']:raise ValueError('PARENT_INDEX')
    pins.update(m['files']);pins.update(load(ip)['files'])
    for p in (mp,ip):pins[p.relative_to(ROOT).as_posix()]=sha(p)
    for folder in ('c4_05_reviews','c4_06_reviews'):
        for p in (ROOT/'outputs/private/research_evaluation'/folder).rglob('*'):
            if p.is_file():pins[p.relative_to(ROOT).as_posix()]=sha(p)
    check_hashes(pins)
    return pins


def run(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',namespace):raise ValueError('NAMESPACE')
    public=ROOT/'outputs/research_evaluation'/namespace;private=ROOT/'outputs/private/research_evaluation'/namespace
    if public.exists() or private.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    pins=protected_assets()
    sources=load(ROOT/'outputs/private/research_evaluation/c4_10_v1/source_hashes.json')
    sources.update(load(ROOT/'outputs/private/research_evaluation/c4_10_v1/terminology_source_paths.json')['files'])
    check_hashes(sources)
    print('Prior/source hashes verified',flush=True)
    bundle=load_preparation('phase24b_candidates_v2')
    for key,path in dict(bank=BANK,sources=SOURCE,catalog=CATALOG,taxonomy=TAXONOMY,config=CONFIG,registry=REGISTRY,holdout=HOLDOUT).items():
        if sha(path)!=bundle['source_pins'][key]:raise ValueError('SOURCE_PIN_MISMATCH')
        sources[str(path)]=sha(path)
    if load(REGISTRY)!=bundle['registry']:raise ValueError('REGISTRY_CHANGED')
    summary,_=analyze_run()
    if summary['state']!='COLLECTION_COMPLETE_NOT_CLINICALLY_EVALUATED' or summary['integrity_issue_codes']:
        raise ValueError('ORIGINAL_RESPONSE_INTEGRITY_BLOCKED')
    print('Existing collection provenance checked',flush=True)
    references=load(PRIVATE_ROOT/'phase24b_candidates_v2/reference_candidates.json')
    refs={r['case_ref']:r for r in references};bindings=source_code_bindings(references)
    records=[load(p) for p in sorted((PRIVATE_ROOT/'phase24b_collect_001').glob('Candidate*.json'))]
    candidates={c['case_ref']:c for c in bundle['candidates']}
    ref_facts={key:reference_facts(c,refs[key]) for key,c in candidates.items()}
    evidence=[response_evidence(r,candidates[r['case_ref']],ref_facts[r['case_ref']],bindings) for r in records]
    # Both unchanged gates execute. No caller-supplied ready boolean or manual rows can bypass them.
    data,options=actual_gate_inputs();options['min_samples']=2
    verified,gate=fit_verified(data,options)
    eligible_proxy=[]  # Only explicit eligible judgments could be projected, never code-match booleans.
    for r in evidence:
        if r['eligibility_for_training'] or r['proxy_correct'] is not None:raise ValueError('UNREVIEWED_PROXY_PROMOTED')
    development=None;development_blocker=None
    try:development=build_development(eligible_proxy,bundle['registry'],evaluation_patients=bundle['split_manifest']['patients']['validation'],min_samples=2)
    except ValueError as exc:development_blocker=str(exc)
    if verified is not None:write(private/'verified_routing_utility.json',verified)
    if development is not None:write(private/'development_routing_utility.json',development)
    # No missing utility is replaced with a fixture. Actual 15-case source routing remains observable.
    cases=case_contract(bundle)
    traces=execute_conditions(cases,bundle['registry'],verified or development)
    repeat=execute_conditions(cases,bundle['registry'],verified or development)
    if traces!=repeat:raise ValueError('NONDETERMINISTIC_ROUTING')
    for i in range(0,len(traces),3):
        group=traces[i:i+3]
        if any(any(r[k]!=group[0][k] for k in ('observation_fingerprint','candidate_fingerprint','memory_fingerprint')) for r in group):
            raise ValueError('PAIRED_INPUT_CHANGED')
    splits=bundle['split_manifest']['patients'];train=set(splits['train']);val=set(splits['validation']);hold=set(splits['test'])
    voi_path=BANK.parent/'voi_model.json';voi=load(voi_path);sources[str(voi_path)]=sha(voi_path)
    voi_train=set(voi['provenance']['training_patients']);bank=read(BANK)
    if voi_train!={r['training_patient_id'] for r in bank}:raise ValueError('VOI_TRAIN_SOURCE_CHANGED')
    if train&val or train&hold or val&hold:raise ValueError('PATIENT_SPLIT_OVERLAP')
    compatibility=[]
    for p in sorted((HOSPITAL/'app/data').glob('*/routing_utility*.json')):
        a=load(p);compatibility.append({'artifact_ref':'ArchivedUtility%03d'%(len(compatibility)+1),
            'sha256':sha(p),'registry_compatible':a.get('registry')==bundle['registry'],'sample_count':len(a.get('samples',[])),'admitted':False})
    agents=[a for a in bundle['registry']['agents'] if a['role']=='specialty']
    groups=[]
    for c in bundle['candidates']:
        ev=[r for r in evidence if r['decision_ref']==c['case_ref']]
        groups.append({'case_ref':c['case_ref'],'split':c['decision']['split'],'stored':len(ev),
            'supported':sum(r['response_status']=='SUPPORTED' for r in ev),'unresolved':sum(r['response_status']=='UNRESOLVED' for r in ev),
            'proxy_judged':sum(r['proxy_correct'] is not None for r in ev),'reference_reviewed':False,'complete_eligible':False})
    named=[r for r in evidence if r['response_status']=='SUPPORTED']
    extraction=Counter(r['answer_extraction_status'] for r in named)
    equality=Counter('SAME_EXPLICIT_CODE' if r['exact_reference_code_match'] is True else 'DIFFERENT_EXPLICIT_CODE' if r['exact_reference_code_match'] is False else 'UNMAPPED' for r in named)
    write(public/'registry_compatibility_audit.json',{'registry_path':REGISTRY.relative_to(ROOT).as_posix(),'registry_fingerprint':digest(bundle['registry']),
        'agents':agents,'specialty_count':len(agents),'historical_outcomes_model_version_match':all(r['model']==next(a['model'] for a in agents if a['agent_id']==r['agent_id']) and r['version']==next(a['version'] for a in agents if a['agent_id']==r['agent_id']) for r in records),
        'model_identifier':bundle['llm_config']['model_identifier'],'archived_artifacts':compatibility,
        'schema':{'schema_version':2,'method':'jaccard_beta_accuracy_v2','min_samples':2,'requires_same_snapshot_complete_agents':5},
        'tokens':'Unchanged Core observed_tokens over chief complaint, symptoms, objective findings, provided history and non-superseded Memory; regex word-set casefold',
        'score_formula':'(1 + sum_j J(tokens,tokens_j)*correct_j)/(2 + sum_j J(tokens,tokens_j)); only J>0; at least min_samples per candidate Agent',
        'abstention':'All candidates must be supported; best-score tie uses absolute tolerance1e-12; missing/insufficient/tie remain distinct',
        'score_calibrated':False,'core_formula_changed':False})
    write(public/'correctness_data_inventory.json',{'bank_rows':len(bank),'bank_patients':len({r['training_patient_id'] for r in bank}),
        'prepared_decisions':15,'original_response_count':len(evidence),'supported':len(named),'unresolved':len(evidence)-len(named),
        'clinical_correctness_nonnull':sum(r.get('correct') is not None for r in records),'independent_reference_reviews':len(options['review_results']['reference_reviews']),
        'independent_response_reviews':len(options['review_results']['response_reviews']),'eligible_training_rows':0,
        'source_proxy_judged_rows':0,'groups':groups,
        'specialties':{a['department']:dict(Counter(r['response_status'] for r in evidence if r['agent_id']==a['agent_id'])) for a in agents}})
    write(public/'response_collection_manifest.json',{'status':'BASELINE_AUDITED_NEW_PILOT_AUTHORIZED_PENDING_PREFLIGHT',
        'baseline_run':'phase24b_collect_001','preparation':'phase24b_candidates_v2','baseline_fingerprint':digest(records),
        'new_pilot_scope':{'case_ref':bundle['candidates'][0]['case_ref'],'selection':'FIRST_TRAIN_PREPARATION_ORDER','specialty_agents':5,
            'max_generation':5,'max_output_tokens':1024,'timeout_seconds':120,'seed':42,'temperature':0,'retry':0,'repair':0,'server_change':False},
        'new_live_results':'live_pilot/c4_11_collect_001/summary.json if actually executed',
        'deduplication_policy':'Original baseline frozen; new run kept separate, no merge/replacement or outcome-guided selection for fitting',
        'generation_at_offline_audit':0,'raw_wire_historical':'NOT_SAVED_NOT_RECONSTRUCTED'})
    write(public/'source_proxy_labeling_audit.json',{'reference_type':'SOURCE_PROXY_DEVELOPMENT_CANDIDATE_FACTS_ONLY','proxy_correctness_created':0,
        'named_response_extraction':dict(extraction),'code_comparisons_only':dict(equality),'code_match_is_correctness':False,
        'reference_scope_reviewed':0,'reference_time_scope_reviewed':0,'source_rows_are_independent_clinical_gold':False,
        'decision_suitability_and_multiple_diagnosis_review':'NOT_REVIEWED','unresolved_false_imputation':0,
        'future_source_condition_cases':[{'case_ref':k,'start_day_offset':max(r['start_day_offset'] for r in f['condition_rows'])} for k,f in ref_facts.items() if any(r['start_day_offset']>0 for r in f['condition_rows'])],
        'blocker_counts':dict(Counter(code for r in evidence for code in r['exclusion_reason']))})
    write(public/'utility_training_manifest.json',{'reference_levels':['TEST_ONLY','SOURCE_PROXY_DEVELOPMENT','INDEPENDENT_REVIEWED'],
        'verified_status':'VERIFIED_UTILITY_NOT_CREATED' if verified is None else 'VERIFIED_UTILITY_FITTED',
        'development_status':'DEVELOPMENT_UTILITY_NOT_CREATED' if development is None else 'DEVELOPMENT_UTILITY_FITTED',
        'min_samples':2,'eligible_complete_train_groups':gate['complete_train_groups'],'eligible_proxy_complete_groups':0,
        'actual_fitted_agents':0,'actual_fitted_specialties':0,'actual_training_patients':0,
        'verified_fit_calls':int(verified is not None),'development_builds':int(development is not None),
        'development_blocker':development_blocker,'dedup_policy':'No run merge or selective response replacement',
        'minimum_data_requirement':'At least2 complete eligible Train Decisions (10 supported judged Agent outcomes), plus disjoint evaluable validation inputs and positive-similarity support for every candidate Agent'})
    write(public/'utility_validation.json',{'existing_gate':{k:gate[k] for k in ('training_readiness','utility_fit_ready','source_reference_ready','clinical_review_ready','response_policy_ready','complete_train_groups','validation_group_count','positive_similarity_support','blocker_codes')},
        'verified_artifact_created':verified is not None,'development_artifact_created':development is not None,
        'real_artifact_roundtrip':'NOT_RUN_NO_ELIGIBLE_ARTIFACT','test_only_roundtrip':'SEE_TEST_RESULTS',
        'production_gate_unchanged':True,'development_artifact_clinical_injection':'REJECTED_BY_SEPARATE_MARKER_CHECK_AND_EXISTING_GATE_PROOF_REQUIREMENT'})
    execution=execution_summary(traces)
    bycase=defaultdict(dict)
    for r in traces:bycase[r['case_ref']][r['condition']]=r['decision']
    jointly=sum(v['rule_similarity']['decision_status']=='selected' and v['adaptive_native_abstention']['decision_status']=='selected' for v in bycase.values())
    agreement=sum(v['rule_similarity']['selected_specialty']==v['adaptive_native_abstention']['selected_specialty'] for v in bycase.values() if v['rule_similarity']['decision_status']=='selected' and v['adaptive_native_abstention']['decision_status']=='selected')
    write(public/'routing_execution_summary.json',{'evidence_level':'SOURCE_STRUCTURAL_NO_FITTED_UTILITY','conditions':execution,
        'repeat_trace_equal':True,'first_pass_decisions':len(traces),'repeat_decisions':len(repeat),'actual_native_rule_adaptive_calls':60,
        'rule_baseline_agreement':{'numerator':agreement,'denominator':jointly,'value':agreement/jointly if jointly else None},
        'development_proxy_agreement':'NOT_EVALUATED_NO_ADMITTED_PROXY_REFERENCE','independent_specialty_accuracy':'NOT_EVALUATED',
        'adaptive_forced_choice_variant':'NOT_IMPLEMENTED_NOT_REPORTED_AS_SEPARATE_CONDITION'})
    write(public/'split_leakage_audit.json',{'routing_train':len(train),'routing_validation':len(val),'original50':len(hold),
        'routing_train_validation_overlap':len(train&val),'routing_train_original50_overlap':len(train&hold),
        'routing_validation_original50_overlap':len(val&hold),'routing_validation_voi_train_overlap':len(val&voi_train),
        'routing_train_voi_train_overlap':len(train&voi_train),'new_utility_training_patients':0,'new_utility_training_evaluation_overlap':0,
        'new_train_eval_disjoint_status':'NOT_APPLICABLE_NO_NEW_FITTED_UTILITY','Original50':'DEVELOPMENT_EXPOSED_NOT_NEW_BLIND_HOLDOUT',
        'observable_input_boundary':'Existing observable whitelist, source-visibility ledger and identical shared input validated; raw-wire content not reconstructed',
        'clinical_visibility':'NOT_REVIEWED_RECORDED_TIME_PROXY'})
    jsonl(private/'response_evidence_inventory.jsonl',evidence)
    write(private/'source_reference_facts.json',list(ref_facts.values()));write(private/'source_description_code_bindings.json',bindings)
    write(private/'registry_snapshot.json',bundle['registry']);write(private/'source_cases.json',cases)
    write(private/'routing_traces.json',traces);write(private/'existing_gate_result.json',gate)
    write(private/'protected_hashes.json',pins);write(private/'source_hashes.json',sources)
    check_hashes(pins);check_hashes(sources)
    result={'status':'C4_11_REFERENCE_BLOCKED','verified_utility':'NOT_CREATED','development_utility':'NOT_CREATED',
        'baseline_responses':75,'eligible_training_rows':0,'fitted_agents':0,'fitted_specialties':0,'utility_fit_count':0,
        'offline_audit_llm_generations':0,'protected_files':len(pins),'changed_protected_files':0,'research_evaluation_ready':False}
    write(public/'offline_result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--namespace',default='c4_11_v1');args=p.parse_args()
    def guard(event,arguments):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('OFFLINE_AUDIT_ONLY')
    sys.addaudithook(guard)
    print(json.dumps(run(args.namespace),ensure_ascii=True))
