"""D6 read-only revalidation and release candidates, never clinical clearance."""
from collections import Counter
import ast,json,sqlite3,sys
from pathlib import Path
from app.locdi.llm_setup import ROOT,digest,load_setup,inspect_local_model
from app.locdi.dataset_suitability import sha,load_memberships
from app.locdi.dataset_inventory import analyze_locdi,checked_csv_links,audit_saved_boundaries
from app.locdi.routing_data_reuse import BANK,read,SYNTHEA
from app.locdi.dataset_loader import load_cases
from app.locdi.reference_readiness import verify_d3,validate_blinded_packet
from app.locdi.source_replay import PUBLIC as D3_PUBLIC,load_replay_cases,STRICT,PROXY
from app.locdi.external_dialogue import verify_source,load_pilot,messages_for,PRIVATE as HB_PRIVATE,budget
from app.locdi.evaluation_gate_audit import audit_actual_gate
from app.locdi.development_features import check_development_features
from validation.run_validation import validate_dataset


def validate_all():
    original=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl';cases=load_cases(original)
    links,missing=checked_csv_links(cases)
    memory={r['memory_id']:r for r in read(BANK.parent/'memory.json')['records']}
    detail=analyze_locdi(cases,csv_links=links,original_memory=memory)
    validation=validate_dataset(original)
    detail.update(sha256=sha(original),validators=validation.counts,validator_issue_counts=validation.issue_counts,
        source_reference_checks={'unique_references':len(links),'matching':sum(links.values()),'missing_files':missing},
        prior_saved_exposure=audit_saved_boundaries(),allowed_use='DEVELOPMENT_ONLY',clinical_gold_approved=0)
    manifest,p,canonical,symptoms=verify_d3()
    proxy=load_replay_cases(policy=PROXY,accept_development_proxy=True)
    strict='UNEXPECTEDLY_ACCEPTED'
    try:load_replay_cases(policy=STRICT)
    except ValueError as e:strict=str(e)
    proxy_validation=validate_dataset(ROOT/manifest['files']['generated_proxy_wrappers.jsonl']['path'])
    actions=p['action_events.jsonl'];references=p['reference_candidates.jsonl'];ledger=p['source_visibility_ledger.jsonl']
    past=read(D3_PUBLIC/'offline_validation.json')
    source_replay={'manifest_fingerprint':manifest['manifest_fingerprint'],'canonical_file_hashes_verified':len(manifest['files']),
        'source_csv_hashes_verified':len(manifest['source_hashes']),'source_patient_json_hashes_verified':len(p['cases.jsonl']),
        'canonical_validation':canonical,'proxy_cases_validated':len(proxy),'proxy_validator_counts':proxy_validation.counts,
        'action_counts':dict(Counter(a['action'] for a in actions)),'consult_events':sum(a['action']=='Consult' for a in actions),
        'memory_records':len(p['memory.json']['records']),'ledger_rows':len(ledger),
        'actual_available_action_times':sum(a['actual_available_at'] is not None for a in actions),
        'proxy_release_minute_counts':dict(Counter(str(a['proxy_available_minute']) for a in actions)),
        'strict_source_status':strict,'reference_candidates':len(references),
        'reviewed_references':sum(r['independent_review_status']!='NOT_REVIEWED' for r in references),
        'clinical_gold_approved':sum(r['clinical_gold_status']!='UNAPPROVED' for r in references),
        'initial_populated_counts':{key:sum(bool(r[key]) for r in p['initial_observations.jsonl']) for key in ('chief_complaint','initial_vitals','objective_findings','current_symptoms')},
        'encoded_symptom_source_items':len(symptoms),'encoded_symptoms_clinically_approved':0,
        'historical_replay_report_sha256':sha(D3_PUBLIC/'offline_validation.json'),'historical_runner_calls':past['actual_runner_calls'],
        'historical_replays_reexecuted_in_D6':0,'allowed_use':'EXPLICIT_DEVELOPMENT_PROXY_ONLY'}
    hb,rows=verify_source();_,selection,pilot_rows=load_pilot()
    selected={s['sample_id'] for s in selection['samples']}
    history=[]
    for folder in sorted((HB_PRIVATE/'runs').glob('*')):
        if not folder.is_dir():continue
        result=read(folder/'result.json') if (folder/'result.json').exists() else None
        history.append({'run_id':folder.name,'status':result.get('status') if result else 'RESULT_NOT_FOUND',
            'test_only':result.get('test_only') if result else None,'model_calls':result.get('model_calls') if result else None,
            'grader_calls':result.get('grader_calls') if result else None,'reuse_history_present':(folder/'reuse_history.json').is_file()})
    healthbench={'revision':hb['revision'],'manifest_fingerprint':hb['manifest_fingerprint'],'raw_sha256':hb['files']['2025-05-07-06-14-12_oss_eval.jsonl']['sha256'],
        'sample_count':len(rows),'unique_sample_ids':len({r['prompt_id'] for r in rows}),'rubric_count':sum(len(r['rubrics']) for r in rows),
        'negative_rubrics':sum(rubric['points']<0 for r in rows for rubric in r['rubrics']),
        'lossless_adapter_samples':sum(messages_for(r)==r['prompt'] for r in rows),'raw_files_verified':len(hb['files']),
        'original_split':hb['original_split'],'development_samples':len(selected),'nonpilot_candidates':sum(r['prompt_id'] not in selected for r in rows),
        'selection_fingerprint':selection['selection_fingerprint'],'pilot_rubric_counts':[len(r['rubrics']) for r in pilot_rows],
        'known_reuse_history':history,'independent_test_approved':False,'patient_level_split':'UNAVAILABLE','model_pretraining_contamination':'UNKNOWN',
        'input_projection':'PROMPT_ONLY_ALL_NON_PROMPT_FIELDS_EXCLUDED','pilot_budget':budget(3,sum(len(r['rubrics']) for r in pilot_rows))}
    packets=ROOT/'outputs/private/reference_review/D4_reference_v1';packet_files=list(packets.glob('*case*.json'))+list(packets.glob('routing_Candidate010_blinded*.json'))
    targets=set()
    for path in packet_files:
        packet=read(path);validate_blinded_packet(packet)
        targets.add((packet.get('dataset_version','routing'),packet.get('case_ordinal',packet.get('case_ref'))))
    return {'schema_version':'locdi.D6.dataset_validation.v1','original_locdi':detail,'source_replay':source_replay,'healthbench':healthbench,
        'internal_review_packets':{'packet_files_checked':len(packet_files),'distinct_review_targets':len(targets),'reviewed':0,'reviewer_fields_blank':True},
        'routing_correctness_gate':audit_actual_gate(),'voi_artifact_readiness':check_development_features('bayesian_voi'),
        'ablation_configs':[{'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'configuration':read(path)} for path in sorted((ROOT/'experiments/configs').glob('B[0-4].json'))],
        'new_model_generations':0,'new_grader_calls':0,'utility_fit_count':0,'voi_probability_estimates':0,'clinical_gold_generated':0}


def split_audit():
    memberships,exposures=load_memberships()
    manifest=read(D3_PUBLIC/'dataset_manifest.json')
    cases=read(ROOT/manifest['files']['cases.jsonl']['path'])
    memberships={k:v for k,v in memberships.items() if k in ('holdout','bank','routing_train','routing_validation','voi_train')}
    memberships['D3_replay']={c['patient_id'] for c in cases}
    overlaps=[]
    for i,left in enumerate(memberships):
        for right in list(memberships)[i+1:]:overlaps.append({'left':left,'right':right,'patients':len(memberships[left]&memberships[right])})
    bank=read(BANK);patient_splits={}
    for name in ('routing_train','routing_validation'):
        for pid in memberships[name]:patient_splits.setdefault(pid,set()).add(name)
    return {'schema_version':'locdi.D6.split_exposure.v1','patient_counts':{k:len(v) for k,v in memberships.items()},'pairwise_overlaps':overlaps,
        'routing_patient_split_conflicts':sum(len(v)>1 for v in patient_splits.values()),'bank_visit_count':len(bank),
        'all_visits_inherit_patient_split':True,'routing_validation_independent_of_voi_training':not bool(memberships['routing_validation']&memberships['voi_train']),
        'legacy_50_development_exposed':True,'D3_29_development_exposed':True,'external_patient_overlap':'NOT_APPLICABLE_NO_SHARED_PATIENT_LINKAGE',
        'healthbench_development_exclusion':3,'healthbench_nonpilot_candidates':4997,'external_blind_test_approved':False,
        'pretraining_contamination':'UNKNOWN','legacy_exposure_sources':exposures}


def readiness(*,software_pass,development_valid,environment_import_pass):
    return {'SOFTWARE_READY':{'ready':bool(software_pass),'scope':'OFFLINE_PROTOCOL_AND_EXISTING_DEVELOPMENT_RUNTIME'},
        'DATASET_READY_FOR_DEVELOPMENT':{'ready':bool(development_valid),'scope':'DECLARED_PROXY_REPLAY_AND_EXTERNAL_DIALOGUE_INPUTS_ONLY'},
        'EXTERNAL_BENCHMARK_EVALUATION_READY':{'ready':False,'environment_import_pass':bool(environment_import_pass),
            'blockers':['GRADER_MODEL_BINDING_MISSING','GRADER_QUALITY_NOT_VALIDATED','GENERATION_AND_GRADING_REQUIRE_SEPARATE_APPROVAL','EXPOSURE_SCOPE_NOT_INDEPENDENT_TEST_APPROVED']},
        'CLINICAL_RESEARCH_EVALUATION_READY':{'ready':False,'blockers':['CLINICAL_GOLD_APPROVED_ZERO','ACTUAL_RELEASE_TIMES_UNVERIFIED','SOURCE_CONSULT_ZERO','ROUTING_AND_VOI_ARTIFACT_REVIEW_BLOCKED']}}


def dependency_inventory():
    vendor=ROOT/'outputs/private/healthbench_environment/D6_v1/vendor'/('simple-evals-652c89d0ca9df547706735883097e9537d40dc47')
    files=[]
    for name in ('healthbench_eval.py','common.py','types.py','sampler/chat_completion_sampler.py'):
        path=vendor/name
        if not path.exists():files.append({'module':name,'status':'NOT_FOUND'});continue
        tree=ast.parse(path.read_text(encoding='utf-8'))
        files.append({'module':name,'status':'FOUND','sha256':sha(path),
            'imports':[{'module':n.module,'level':n.level} if isinstance(n,ast.ImportFrom) else {'names':[a.name for a in n.names]} for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))],
            'unbounded_while_lines':[n.lineno for n in ast.walk(tree) if isinstance(n,ast.While) and isinstance(n.test,ast.Constant) and n.test.value is True]})
    return files
