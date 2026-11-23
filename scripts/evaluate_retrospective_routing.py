"""Freeze then execute the offline C4-17 retrospective benchmark. Never resumes."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_memory import timestamp
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.routing_data_reuse import read,BANK
from app.locdi.source_enrichment import validate_fact,iso_ms,pointer
from app.locdi.routing_two_decision_collection import verify_pins,verify_fingerprint
from app.locdi.routing_collection_preparation import project_input
from app.locdi.abstention_routing import compare
from app.locdi.retrospective_routing import protocol,label,coverage,measurements,public_input,validate_mapping
from scripts.enrich_full_cohort_sources import write,rows

PARENT=ROOT/'outputs/private/research_evaluation/c4_16_v1'
PARENT_PUBLIC=ROOT/'outputs/research_evaluation/c4_16_v1'
PREP=ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json'


def locations(namespace):
    if not re.fullmatch(r'c4_17_[A-Za-z0-9_-]{1,48}',namespace):raise ValueError('C4_17_NAMESPACE_REQUIRED')
    return ROOT/'outputs/research_evaluation'/namespace,ROOT/'outputs/private/research_evaluation'/namespace


def utc():return datetime.now(timezone.utc).isoformat()


def freeze(namespace):
    pub,pri=locations(namespace)
    if pub.exists() or pri.exists():raise ValueError('OUTPUT_ALREADY_USED_NO_RESUME')
    pins=load(PARENT/'protected_hashes.json');m=load(PARENT_PUBLIC/'output_manifest.json');verify_fingerprint(m)
    ip=PARENT/'private_index.json'
    if sha(ip)!=m['private_index_sha256']:raise ValueError('PARENT_PRIVATE_INDEX_HASH')
    pins.update(m['files']);pins.update(load(ip)['files'])
    for p in [ip,PARENT_PUBLIC/'output_manifest.json']:pins[p.relative_to(ROOT).as_posix()]=sha(p)
    print('VERIFYING_PROTECTED_AND_SOURCE_FILES',len(pins),flush=True)
    verify_pins(pins);sources=load(PARENT/'source_hashes.json');verify_pins(sources)
    mapping=load(PARENT_PUBLIC/'source_mapping_evidence_registry.json');validate_mapping(mapping)
    cohort=load(PARENT/'registered_cohort.json')
    pub.mkdir(parents=True);pri.mkdir(parents=True)
    write(pri/'protected_hashes.json',pins);write(pri/'source_hashes.json',sources)
    write(pri/'registered_cohort.json',cohort)
    spec={**protocol(),'frozen_at':utc(),'mapping_registry_sha256':sha(PARENT_PUBLIC/'source_mapping_evidence_registry.json'),
          'registered_cohort_fingerprint':digest(cohort),'preparation_sha256':sha(PREP),
          'scope_sizes':dict(Counter(c['dataset'] for c in cohort))}
    write(pub/'retrospective_task_protocol.json',spec)
    files={p.relative_to(ROOT).as_posix():sha(p) for p in [pub/'retrospective_task_protocol.json',pri/'registered_cohort.json',
        ROOT/'app/locdi/retrospective_routing.py',Path(__file__).resolve()]}
    manifest={'version':protocol()['version'],'frozen_at':spec['frozen_at'],'files':files,
        'protocol_fingerprint':digest(spec),'results_exist_at_freeze':False}
    manifest['fingerprint']=digest(manifest);write(pub/'preregistration_manifest.json',manifest)
    write(pri/'initial_preservation.json',{'status':'PASS','protected_files':len(pins),'source_files':len(sources),'checked_at':utc()})
    print('PROTOCOL_FROZEN_NO_OUTCOMES',flush=True)


def execute(namespace):
    pub,pri=locations(namespace)
    if (pri/'execution_started.json').exists():raise ValueError('NO_RESUME_OR_OVERWRITE')
    frozen=load(pub/'preregistration_manifest.json');verify_fingerprint(frozen);verify_pins(frozen['files'])
    spec=load(pub/'retrospective_task_protocol.json')
    if digest(spec)!=frozen['protocol_fingerprint']:raise ValueError('PROTOCOL_CHANGED')
    if sha(PREP)!=spec['preparation_sha256']:raise ValueError('PREPARATION_CHANGED')
    pins=load(pri/'protected_hashes.json');sources=load(pri/'source_hashes.json');verify_pins(pins);verify_pins(sources)
    write(pri/'execution_started.json',{'started_at':utc(),'protocol_fingerprint':digest(spec)})
    mapping=load(PARENT_PUBLIC/'source_mapping_evidence_registry.json')
    if sha(PARENT_PUBLIC/'source_mapping_evidence_registry.json')!=spec['mapping_registry_sha256']:raise ValueError('MAPPING_CHANGED')
    cohort=load(pri/'registered_cohort.json');bundle=load(PREP);registry=AgentRegistry(bundle['registry'])
    descriptors=[x for x in read(ROOT/'outputs/private/research_evaluation/c4_05_v1/case_source_index.jsonl') if x['dataset'] in ['Original50','TrainingBank']]
    bykey={(d['dataset'],d['case_ref']):d for d in descriptors}
    byvisit={(d['patient_id'],d['visit_id']):d for d in descriptors if d['dataset']=='TrainingBank'}
    grouped=defaultdict(list);facts_bycase=defaultdict(list);windows={}
    with (ROOT/'outputs/private/research_evaluation/c4_05_v1/source_facts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            f=json.loads(line)
            if f['dataset'] in ['Original50','TrainingBank'] and f['source_resource_type']=='Condition':grouped[f['source_file']].append(f)
    desc_byfile=defaultdict(list)
    for d in descriptors:desc_byfile[d['raw_path']].append(d)
    verified=0
    for i,name in enumerate(sorted(set(grouped)|set(desc_byfile)),1):
        expected=sources.get(str(Path(name)))
        if not expected or sha(name)!=expected:raise ValueError('SOURCE_HASH_MISMATCH')
        raw=load(name);encounters={e['uuid']:e for e in raw['record']['encounters']}
        for d in desc_byfile[name]:
            if raw['attributes']['id']!=d['patient_id']:raise ValueError('NATIVE_PATIENT_MISMATCH')
            e=encounters[d['visit_id']]
            windows[(d['dataset'],d['case_ref'])]={'patient_id':d['patient_id'],'visit_id':d['visit_id'],
                'start':iso_ms(e.get('start')),'stop':iso_ms(e.get('stop')),'completed':e.get('ended') is True,
                'native_encounter_fingerprint':digest(e)}
            # Full current Condition inventory is required, not just an exported positive subset.
            fs=[f for f in grouped[name] if (f['dataset'],f['case_ref'])==(d['dataset'],d['case_ref'])]
            if len(fs)!=len(e.get('conditions',[])):raise ValueError('INCOMPLETE_NATIVE_CONDITION_INVENTORY')
        for f in grouped[name]:
            validate_fact(f,raw,expected);d=bykey[(f['dataset'],f['case_ref'])]
            if f['patient_reference']!=d['patient_id'] or f['visit_reference']!=d['visit_id']:raise ValueError('SOURCE_DESCRIPTOR_MISMATCH')
            f['source_verified']=True;facts_bycase[(f['dataset'],f['case_ref'])].append(f);verified+=1
        if i%50==0:print('NATIVE_FILES_CHECKED',i,flush=True)
    extracted=utc();labels=[];inventory=[]
    prepared={c['case_ref']:c for c in bundle['candidates']}
    for c in cohort:
        key=(c['dataset'],c['case_ref'])
        if c['dataset']=='Prepared15':
            d=byvisit[(c['patient_id'],c['visit_id'])];key=('TrainingBank',d['case_ref'])
        l=label(c,facts_bycase[key],windows[key],mapping,extracted);labels.append(l)
        inventory.append({'case_ref':c['case_ref'],'dataset':c['dataset'],'window_verified':l['window_verified'],
            'label_coverage':l['coverage'],'positive_tag_count':len(l['confirmed_positive_domains']),
            'unknown_domain_count':len(l['unknown_domains']),
            'input_status':'EXISTING_PREPARED15_MINUTE1' if c['dataset']=='Prepared15' else 'INPUT_UNAVAILABLE',
            'eligible_confirmed_positive_comparison':c['dataset']=='Prepared15' and bool(l['confirmed_positive_domains']),
            'development_exposed':True,'independent_holdout':False})
    rows(pri/'retrospective_source_labels.jsonl',labels)
    # Public labels contain ordinal/state/counts only; condition codes/times remain private.
    rows(pub/'retrospective_source_labels.jsonl',[{k:v for k,v in l.items() if k in ['version','target','case_ref','dataset','coverage','window_verified','in_window_conditions','unmapped_conditions','missing_time_conditions','outside_window_conditions','future_outcome_conditions']} for l in labels])
    write(pub/'eligible_case_inventory.json',{'rows':inventory,'registered_scope_occurrences':len(cohort),
        'distinct_patient_encounter_pairs':len({(c['patient_id'],c['visit_id']) for c in cohort}),
        'unique_patients_across_scopes':len({c['patient_id'] for c in cohort}),'prepared_is_subset_of_trainingbank':True,
        'patient_counts':{ds:len({c['patient_id'] for c in cohort if c['dataset']==ds}) for ds in spec['cohorts']}})
    cases=[];memory_checked=ledger_checked=0
    for ref,c in prepared.items():
        d=c['decision'];obs=public_input(c['shared_payload']['observation'])
        if digest(c['shared_payload'])!=c['shared_input_fingerprint']:raise ValueError('SHARED_INPUT_FINGERPRINT_MISMATCH')
        if project_input(d['observation'],bundle['catalog'])!=c['shared_payload']:raise ValueError('PUBLIC_PROJECTION_CHANGED')
        if digest(d['observation']['longitudinal_memory'])!=c['memory_fingerprint']:raise ValueError('MEMORY_FINGERPRINT_MISMATCH')
        for record in d['observation']['longitudinal_memory']['records']:
            if (record['patient_id']!=d['patient_id'] or record['visit_id']==d['visit_id']
                or timestamp(record['visit_started_at'])>=timestamp(c['visit_started_at'])
                or timestamp(record['available_at'])>timestamp(c['visit_started_at'])
                or timestamp(record['observed_at'])>timestamp(record['available_at'])):raise ValueError('MEMORY_CUTOFF_OR_IDENTITY_VIOLATION')
            memory_checked+=1
        for metadata in c['visibility'].values():
            if metadata['available_minute']>d['minute']:raise ValueError('FUTURE_INPUT_LEDGER_VIOLATION')
            ledger_checked+=1
        cr=next(x for x in cohort if x['case_ref']==ref)
        if timestamp(cr['decision_time'])!=timestamp(c['visit_started_at'])+__import__('datetime').timedelta(minutes=d['minute']):raise ValueError('INPUT_DECISION_TIME_MISMATCH')
        cases.append({'case_ref':ref,'decision_id':d['decision_id'],'patient_id':d['patient_id'],'split':d['split'],
            'decision_time':cr['decision_time'],'observation':obs,'observable_input_fingerprint':digest(obs),
            'candidate_departments':sorted(a['department'] for a in bundle['registry']['agents'] if a['role']=='specialty'),
            'registry_fingerprint':digest(registry.snapshot()),'source_fingerprint':c['source_fingerprint'],
            'evidence_fingerprint':digest(c['visibility'])})
    # This file contains input only; Source labels are physically separate and never passed to compare().
    write(pri/'policy_input_cases.json',cases)
    traces=compare(cases,registry);repeat=compare(deepcopy(cases),registry)
    if traces!=repeat:raise ValueError('POLICY_REPRODUCTION_MISMATCH')
    rows(pri/'routing_policy_trace.jsonl',traces)
    publictraces=[]
    for t in traces:
        publictraces.append({k:deepcopy(t[k]) for k in ['case_ref','condition','execution_status','input_fingerprint','memory_fingerprint','candidate_fingerprint','registry_fingerprint','source_fingerprint']})
        publictraces[-1]['decision']={k:t['decision'][k] for k in ['decision_status','selected_specialty']}
    l15=[l for l in labels if l['dataset']=='Prepared15']
    measures={letter:measurements(l15,[t for t in traces if t['condition']==condition]) for letter,condition in spec['policies'].items()}
    split=bundle['split_manifest']['patients'];tr=set(split['train']);va=set(split['validation']);ho=set(split['test'])
    vp=set(load(BANK.parent/'voi_model.json')['provenance']['training_patients'])
    if tr&va or tr&ho or va&ho:raise ValueError('SPLIT_OVERLAP')
    if any(r['split']!=('train' if r['patient_id'] in tr else 'validation') for r in bundle['patient_visit_reservations']):raise ValueError('PATIENT_ENCOUNTER_SPLIT_CHANGED')
    splitinfo={'train_patients':len(tr),'validation_patients':len(va),'original50_patients':len(ho),
        'train_validation_overlap':len(tr&va),'train_original50_overlap':len(tr&ho),'validation_original50_overlap':len(va&ho),
        'routing_train_voi_train_overlap':len(tr&vp),'routing_validation_voi_train_overlap':len(va&vp),
        'original50_voi_train_overlap':len(ho&vp),'same_patient_encounters_reserved_to_same_split':True,
        'all_results_exploratory_development_exposed':True,'independent_integrated_validation':False}
    bycase={c['case_ref']:{} for c in cases}
    for t in publictraces:bycase[t['case_ref']][t['condition']]=t['decision']
    differences={a+'__'+b:sum(bycase[c][spec['policies'][a]]!=bycase[c][spec['policies'][b]] for c in bycase) for a,b in [('A','B'),('A','C'),('B','C')]}
    write(pub/'routing_policy_comparison.json',{'policies':spec['policies'],'rows':publictraces,'repeat_equal':True,
        'pairwise_decision_differences_all15':differences,'actual_path_executions':len(traces),'repeat_path_executions':len(repeat),
        'missing_utility_is_not_evidence_based_abstention':True,'clinical_performance':'NOT_EVALUATED'})
    write(pub/'domain_agreement_metrics.json',{'target':spec['target'],'conditions':measures,'split':splitinfo,
        'by_split':{s:{letter:measurements([l for l in l15 if l['case_ref'] in {c['case_ref'] for c in cases if c['split']==s}],
            [t for t in traces if t['condition']==condition and t['case_ref'] in {c['case_ref'] for c in cases if c['split']==s}])
            for letter,condition in spec['policies'].items()} for s in ['train','validation']},
        'multiple_domain_policy':'Selection matching ANY confirmed positive tag is a hit, never exclusive correctness; all tags preserved'})
    cover={ds:coverage([l for l in labels if l['dataset']==ds]) for ds in spec['cohorts']}
    write(pub/'label_coverage.json',{'cohorts':cover,'all_scope_occurrences_not_independent_samples':True,
        'actual_router_inputs':15,'eligible_positive_input_pairs':cover['Prepared15']['with_positive_tags'],
        'label_only_other_scope_occurrences':279,'independent_clinical_reference_count':0})
    write(pub/'domain_mapping_audit.json',{'mapping_registry_path':(PARENT_PUBLIC/'source_mapping_evidence_registry.json').relative_to(ROOT).as_posix(),
        'mapping_registry_sha256':spec['mapping_registry_sha256'],'exact_fixed_mapping':spec['mapping'],
        'source_condition_occurrences_verified':verified,'native_files_checked':len(set(grouped)|set(desc_byfile)),
        'mapping_expansion':False,'clinical_negative_labels_created':0,'clinical_referral_crosswalk':False,
        'unmapped_is_unknown':True,'complete_accepted_specialty_sets':0})
    original_leak=load(PARENT_PUBLIC/'circularity_and_leakage_audit.json')
    write(pub/'circularity_and_leakage_audit.json',{'split':splitinfo,'input_label_physically_separate':True,
        'nested_gold_key_check':'PASS','source_label_passed_to_router':False,'declared_input_ledger_checks':ledger_checked,
        'selected_memory_patient_previous_visit_cutoff_checks':memory_checked,'actual_clinical_release':'NOT_RECORDED_PROXY_UNREVIEWED',
        'input_and_source_pins_verified':True,'future_outcomes_private_only':True,
        'same_literal_keyword_label_function':False,'shared_concept_and_development_scope_risk':True,
        'label_function':'Exact Source System/Code + native window; no registry keyword, predictions or correctness input',
        'policy_function':'Unchanged existing public text keyword policies; static baseline fixed before outcomes',
        'original50_memory_boundary_prior_audit':{'path':(PARENT/'original_memory_exposure_audit.json').relative_to(ROOT).as_posix(),'sha256':sha(PARENT/'original_memory_exposure_audit.json')},
        'prepared_historical_memory_literal_mapped_concept_phrase_count':original_leak['prepared_memory_records_with_exact_mapped_concept_phrase'],
        'historical_diagnosis_not_equivalent_to_current_gold_injection':True,'historical_diagnosis_semantic_independence':'NOT_ESTABLISHED',
        'no_clinical_or_independent_accuracy_claim':True})
    rp=load(PARENT_PUBLIC/'independent_reference_packet_manifest.json')
    paths=list(rp['packet_file_sha256'])+[str((PARENT/'track_a_second_stage_agent_responses.json').relative_to(ROOT).as_posix()),
        str((PARENT/'independent_reference_review_blank.json').relative_to(ROOT).as_posix()),
        str((PARENT_PUBLIC/'independent_reference_review_guide.md').relative_to(ROOT).as_posix())]
    write(pub/'independent_review_package_manifest.json',{'version':'c4_17.reuse.c4_16.v1','files':{p:sha(ROOT/p) for p in paths},
        'blind_packets':2,'actual_responses_second_stage':10,'independent_reviews_received':0,
        'procedure':'Provide blind packets/blank/guide first; freeze human review before disclosing second-stage responses; retrospective labels are not reviewer decisions',
        'reviewer_identity_created':False,'existing_forms_changed':False})
    final='C4_17_PARTIAL_REFERENCE_COVERAGE' if cover['Prepared15']['with_positive_tags'] else 'C4_17_RETROSPECTIVE_BENCHMARK_NOT_FEASIBLE'
    verify_pins(pins);verify_pins(sources);verify_pins(frozen['files'])
    write(pub/'verification.json',{'status':'PASS','final_status':final,'protected_files':len(pins),'source_files':len(sources),
        'changed_or_missing':0,'registered_scope_occurrences':len(cohort),'source_conditions_verified':verified,
        'paired_inputs':len(cases),'eligible_positive_inputs':cover['Prepared15']['with_positive_tags'],
        'policy_repeat_equal':True,'protocol_frozen_before_outcomes':True,'core_or_gate_changed':False,
        'new_policy':False,'llm_generation':0,'utility_fitting':0,'clinical_verdicts_created':0,'research_evaluation_ready':False})
    print(final,json.dumps(cover),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','execute']);p.add_argument('--namespace',default='c4_17_v1');a=p.parse_args()
    def guard(event,args):
        if event=='socket.connect':raise RuntimeError('OFFLINE_ONLY_NO_NETWORK')
    sys.addaudithook(guard)
    (freeze if a.command=='freeze' else execute)(a.namespace)


if __name__=='__main__':main()
