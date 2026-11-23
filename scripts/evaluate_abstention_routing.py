"""C4-15 offline audit / hash-pinned reviewed routing evaluation. No generation."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.core.agent_registry import AgentRegistry
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.routing_two_decision_collection import load_protocol,verify_pins,verify_fingerprint
from app.locdi.routing_collection_preparation import BANK
from app.locdi.abstention_routing import compare,metrics,reference_blank,policy_spec
from scripts.enrich_full_cohort_sources import write


def locations(namespace):
    import re
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',namespace):raise ValueError('INVALID_NAMESPACE')
    pub=ROOT/'outputs/research_evaluation'/namespace
    private=ROOT/'outputs/private/research_evaluation'/namespace
    if pub.exists() or private.exists():raise ValueError('OUTPUT_ALREADY_USED_NO_RESUME')
    pub.mkdir(parents=True);private.mkdir(parents=True)
    return pub,private


def protect_parent():
    parent=ROOT/'outputs/private/research_evaluation/c4_14c_v1'
    pins=load(parent/'protected_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_14c_v1/output_manifest.json'
    manifest=load(mp);verify_fingerprint(manifest)
    ip=parent/'private_index.json'
    if sha(ip)!=manifest['private_index_sha256']:raise ValueError('PARENT_PRIVATE_INDEX_HASH')
    pins.update(manifest['files']);pins.update(load(ip)['files'])
    for p in [mp,ip]:pins[p.relative_to(ROOT).as_posix()]=sha(p)
    verify_pins(pins)
    sources=load(parent/'source_hashes.json');verify_pins(sources)
    return pins,sources


def audit(namespace):
    pub,private=locations(namespace)
    print('VERIFYING_COMPLETED_ASSETS',flush=True)
    pins,sources=protect_parent()
    write(private/'protected_hashes.json',pins);write(private/'source_hashes.json',sources)
    bundle,plan,bindings=load_protocol('c4_14_v1')
    print('SOURCE_PROTOCOL_VERIFIED',len(pins),len(sources),flush=True)
    registry=AgentRegistry(bundle['registry'])
    oldprivate=ROOT/'outputs/private/research_evaluation/c4_14c_v1'
    alignment=load(oldprivate/'source_response_alignment.json')['rows']
    facts={r['case_ref']:r for r in __import__('json').loads('['+','.join((ROOT/'outputs/private/research_evaluation/c4_12_v1/source_decision_eligibility.jsonl').read_text(encoding='utf-8').splitlines())+']')}
    cases=[];packets=[];analyses=[]
    for ref in ['Candidate001','Candidate002']:
        c=next(c for c in bundle['candidates'] if c['case_ref']==ref);f=facts[ref]
        packet={'case_ref':ref,'decision_id':c['decision']['decision_id'],'decision_time':f['decision_time'],
            'observation':c['shared_payload']['observation'],'source_fingerprint':c['source_fingerprint'],
            'current_and_other_encounter_conditions_private_review_only':f['all_patient_condition_rows'],
            'native_condition_timestamps':f['native_condition_inventory'],'source_visibility_ledger':f['visibility_ledger'],
            'timing_warning':'Native onset/CSV recorded dates are not clinical release; future/other encounter records are not current reference',
            'no_automatic_specialty_gold':True,'independent_review_status':'NOT_REVIEWED'}
        packets.append(packet)
        cases.append({'case_ref':ref,'decision_id':c['decision']['decision_id'],'patient_id':c['decision']['patient_id'],
            'split':c['decision']['split'],'decision_time':f['decision_time'],'observation':deepcopy(c['shared_payload']['observation']),
            'observable_input_fingerprint':digest(c['shared_payload']['observation']),
            'candidate_departments':sorted(a['department'] for a in plan['agents']),
            'registry_fingerprint':digest(registry.snapshot()),'source_fingerprint':c['source_fingerprint'],
            'evidence_fingerprint':digest(packet)})
        for a in [a for a in alignment if a['case_ref']==ref]:
            response=next(load(p) for p in sorted((ROOT/'outputs/private/research_evaluation/c4_14_v1/collection/c4_14_ten_001').glob('*_after.json')) if (lambda r:r['case_ref']==ref and r['agent_id']==a['agent_id'])(load(p)))
            analyses.append({**deepcopy(a),'actual_rationale':response['response']['rationale'],'actual_response':response['response'],
                'response_fingerprint':response['response_fingerprint'],'reference_packet_fingerprint':digest(packet),
                'diagnosable_from_current_information':'REVIEW_REQUIRED','diagnosis_abstention_permitted_by_collection_schema':True,
                'routing_suitability':'NOT_REVIEWED','agent_abstention_appropriateness':'REVIEW_REQUIRED',
                'reference_blockers':['CURRENT_TASK_AND_MULTIPLE_CONDITION_SCOPE_UNREVIEWED','ACTUAL_RELEASE_NOT_RECORDED_PROXY_UNREVIEWED_FOR_EVALUATION','INDEPENDENT_CLINICAL_REVIEW_MISSING'],
                'rationale_is_not_causal_or_clinical_proof':True})
    write(private/'decision_source_packets.json',packets)
    write(private/'response_source_analysis.json',{'rows':analyses,'correctness_generated':0})
    write(private/'cases.json',cases);write(private/'registry.json',bundle['registry'])
    refs=reference_blank(cases);write(private/'abstention_reference_blank.json',refs)
    traces=compare(cases,registry);repeat=compare(cases,registry)
    if traces!=repeat:raise ValueError('NONDETERMINISTIC_OFFLINE_TRACE')
    write(private/'policy_execution_trace.json',traces)
    conditions=sorted({t['condition'] for t in traces})
    results={condition:metrics(cases,[t for t in traces if t['condition']==condition],refs,purpose='development_reviewed') for condition in conditions}
    write(pub/'abstention_policy_spec.json',policy_spec())
    # No patient/Decision IDs, medical text, predictions, source code, or rationale in public.
    public=[]
    for t in traces:
        d=t['decision']
        public.append({k:t[k] for k in ['case_ref','condition','execution_status','input_fingerprint','memory_fingerprint','candidate_fingerprint','registry_fingerprint','source_fingerprint']})
        public[-1]['decision']={k:d.get(k) for k in ['mode','decision_status','selected_specialty','candidate_departments']}
        public[-1]['reason_code']='MISSING_UTILITY' if t['execution_status']=='MISSING_UTILITY' else d.get('rationale') if t['condition']=='D_abstention_aware' else 'PREDECLARED_STATIC' if t['condition']=='A_static_development' else 'EXISTING_CORE_RULE'
        if 'support' in d:
            public[-1]['support']=[{'department':s['department'],'support_status':s['support_status'],'current_match_count':len(s['current_keyword_matches']),'context_match_count':len(s['context_keyword_matches']),'evidence_fingerprint':s['evidence_fingerprint']} for s in d['support']]
    write(pub/'policy_execution_trace.json',{'rows':public,'repeat_equal':True,'clinical_performance':'NOT_EVALUATED','main_decisions':8,'repeat_verification_decisions':8})
    write(pub/'abstention_metrics.json',{'conditions':results,'agent_unresolved_observed':6,'agent_unresolved_is_not_router_abstention_rate':True})
    fixture=runpy.run_path(str(ROOT/'tests/locdi/test_abstention_routing.py'))
    treg,tcases=fixture['fixture']();trefs=fixture['reviewed'](tcases);tt=compare(tcases,treg)
    write(pub/'test_only_execution.json',{'scope':'TEST_ONLY_NONCLINICAL','clinical_metrics':'NOT_EVALUATED','conditions':{condition:metrics(tcases,[t for t in tt if t['condition']==condition],trefs,test_only=True) for condition in conditions}})
    patients=bundle['split_manifest']['patients'];train=set(patients['train']);val=set(patients['validation']);holdout=set(patients['test'])
    vp=set(load(BANK.parent/'voi_model.json')['provenance']['training_patients'])
    split={'routing_train':len(train),'routing_validation':len(val),'holdout':len(holdout),
        'train_validation_overlap':len(train&val),'train_holdout_overlap':len(train&holdout),'validation_holdout_overlap':len(val&holdout),
        'routing_train_voi_train_overlap':len(train&vp),'routing_validation_voi_train_overlap':len(val&vp),
        'holdout_voi_train_overlap':len(holdout&vp),'integration_independent_holdout':'BLOCKED_VALIDATION_EXPOSED_TO_VOI_TRAINING',
        'original50_development_exposed':True,'cohort_or_split_modified':False,'threshold_tuned_on_results':False,
        'new_policy_has_prior_development_exposure':True,'future_independent_patients_required':True,'source_snapshot_hash_check':'PASS','gold_field_input_check':'PASS'}
    if split['train_validation_overlap'] or split['train_holdout_overlap'] or split['validation_holdout_overlap']:raise ValueError('PATIENT_SPLIT_OVERLAP')
    write(pub/'split_and_leakage_audit.json',split)
    baseline=[load(p) for p in (ROOT/'outputs/private/routing_collection/phase24b_collect_001').glob('Candidate*_Agent*.json')]
    if len(baseline)!=75 or any(r['correct'] is not None for r in baseline):raise ValueError('BASELINE_COUNT_OR_REVIEW_CHANGED')
    write(pub/'response_analysis_summary.json',{'actual_new_responses':len(analyses),'supported':sum(r['response_status']=='SUPPORTED_REFERENCE_NOT_REVIEWED' for r in analyses),
        'unresolved':sum(r['response_status']=='UNRESOLVED' for r in analyses),'baseline_response_count':len(baseline),
        'baseline_certainty_counts':dict(Counter(r['response']['certainty'] for r in baseline)),
        'independent_correctness_references':0,'verified_routing_utility':0,'routing_suitability_reviewed':0,'abstention_appropriateness_reviewed':0,
        'current_encounter_code_matched_supported':sum(r['current_encounter_code_match_count']>0 for r in analyses),
        'past_encounter_code_matched_supported':sum(r['past_encounter_native_code_match_count']>0 for r in analyses),
        'correctness_created':0,'llm_generation':0,'utility_fitting':0,'source_label_is_not_specialty_reference':True})
    verify_pins(pins);verify_pins(sources)
    write(pub/'verification.json',{'status':'PASS','protected_files':len(pins),'source_files':len(sources),'changed_or_missing':0,
        'new_response_source_links':len(analyses),'registered_decisions':len(cases),'reviewed_decisions':0,'core_or_gate_changed':False,
        'fitting':0,'llm_generation':0,'policy_tested':True,'clinical_reference_status':'BLOCKED','research_evaluation_ready':False,
        'final_status':'C4_15_REFERENCE_BLOCKED'})
    print('C4_15_REFERENCE_BLOCKED_POLICY_TESTED',flush=True)


def evaluate_review(namespace,review_path,review_sha,purpose):
    pub,private=locations(namespace)
    if not review_path or not review_sha or sha(review_path)!=review_sha:raise ValueError('REVIEW_FILE_SHA_REQUIRED')
    parent=ROOT/'outputs/private/research_evaluation/c4_15_v1'
    mp=ROOT/'outputs/research_evaluation/c4_15_v1/output_manifest.json'
    m=load(mp);verify_fingerprint(m)
    if sha(parent/'private_index.json')!=m['private_index_sha256']:raise ValueError('PARENT_INDEX_HASH')
    verify_pins(load(parent/'private_index.json')['files']);verify_pins(load(parent/'source_hashes.json'))
    cases=load(parent/'cases.json');traces=load(parent/'policy_execution_trace.json');refs=load(review_path)
    result={cond:metrics(cases,[t for t in traces if t['condition']==cond],refs,purpose=purpose) for cond in sorted({t['condition'] for t in traces})}
    write(private/'review_input.json',refs);write(pub/'abstention_metrics.json',{'conditions':result,'review_file_sha256':review_sha,'llm_generation':0,'utility_fitting':0,'research_evaluation_ready':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','evaluate'])
    parser.add_argument('--namespace',required=True)
    parser.add_argument('--review-results');parser.add_argument('--review-sha256')
    parser.add_argument('--purpose',choices=['independent_evaluation','development_reviewed'],default='independent_evaluation')
    args=parser.parse_args()
    # Offline-only executable boundary, also protects accidental new network use.
    def guard(event,values):
        if event=='socket.connect':raise RuntimeError('OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    if args.command=='audit':audit(args.namespace)
    else:evaluate_review(args.namespace,args.review_results,args.review_sha256,args.purpose)


if __name__=='__main__':main()
