"""C3-01 cross-workstream audit only. No conversion, fitting or clinical labels."""
from collections import Counter,defaultdict
from itertools import combinations
import json
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import load,sha,verify as verify_freeze,PRIVATE
from app.locdi.episode_exporter import _json


def field_counts(values):
    counts=Counter('null' if v is None else 'empty' if v in ('',[],{}) else 'value' for v in values)
    return {'total':len(values),**{s:counts[s] for s in ('null','empty','value')}}


def initial_matrix(original,initials,actions,references):
    def rows(values,fields):
        return {k:{**field_counts([r.get(k) for r in values]),'field_absent':sum(k not in r for r in values)} for k in fields}
    doctors=[c['agent_visible']['current_initial_state']['doctor_visible_info'] for c in original]
    events=[e for c in original for e in c['environment']['current_encounter_events']]
    original_fields=rows(doctors,['chief_complaint','initial_vitals','objective_findings','history','current_symptoms'])
    projected=sum(bool(c['agent_visible']['current_initial_state']['patient_visible_info'].get('associated_symptoms')) for c in original)
    return {'schema_version':'locdi.c3.initial_information.v1',
        'Original':{'cases':len(original),'fields':original_fields,
            'projected_associated_symptom_cases':projected,'validated_current_symptom_cases':0,
            'current_symptom_interpretation':'PROJECTION_LABEL_NOT_VERIFIED_CURRENT_SYMPTOM_OR_ANSWER',
            'arrival_context_cases':sum(bool(d.get('arrival_info')) for d in doctors),
            'arrival_context_scope':'ENCOUNTER_CLASS_GENERIC_PURPOSE_NOT_OBSERVED_CHIEF_COMPLAINT',
            'encounter_classes':dict(Counter(d.get('arrival_info',{}).get('encounter_class','UNKNOWN') for d in doctors)),
            'ask_count':sum(e['action']=='Ask' for e in events),
            'ask_template_count':len({e['target'] for e in events if e['action']=='Ask'}),
            'test_count':sum(e['action']=='Test' for e in events),
            'test_target_count':len({e['target'] for e in events if e['action']=='Test'}),
            'source_consult_count':sum(e['action']=='Consult' for e in events),
            'source_condition_count':sum(len(c['provenance']['source_case_audit']['diagnosis_sources']) for c in original),
            'clinical_gold_approved':0,'metric_ready_cases':{k:sum(c['task_annotations']['metric_readiness'].get(k) is True for c in original)
                for k in ('diagnosis_accuracy','memory_retrieval','routing_accuracy','voi_regret')}},
        'D3':{'cases':len(initials),'fields':rows(initials,['chief_complaint','initial_vitals','objective_findings','history','current_symptoms']),
            'arrival_context_cases':sum(bool(r.get('demographics')) for r in initials),
            'arrival_context_scope':'DEMOGRAPHICS_AND_CLASS_UNDER_EXPLICIT_DEVELOPMENT_ASSUMPTION',
            'encounter_classes':dict(Counter(r['encounter_class'] for r in initials)),
            'ask_count':sum(a['action']=='Ask' for a in actions),
            'ask_template_policy':'SYNTHETIC_INTERACTION_TEMPLATE_WITH_SOURCE_RECORDED_ANSWER',
            'test_count':sum(a['action']=='Test' for a in actions),
            'test_target_count':len({a['canonical_target'] for a in actions if a['action']=='Test'}),
            'source_consult_count':sum(a['action']=='Consult' for a in actions),
            'source_condition_count':len(references),'clinical_gold_approved':sum(r['clinical_gold_status']=='APPROVED' for r in references),
            'strict_source_blocked_cases':len(initials),'strict_scope':'GLOBAL_LOADER_CONTRACT_APPLIES_TO_ALL_CASES_NOT_29_REPLAYS'}}


def routing_groups(data):
    agents={a['agent_id'] for a in data['registry']['agents'] if a['role']=='specialty'}
    grouped=defaultdict(list)
    for row in data['outcomes']:grouped[row['decision_id']].append(row)
    result=[]
    for index,d in enumerate(data['decisions'],1):
        rr=grouped[d['decision_id']]
        unresolved=[r for r in rr if r['response']['certainty']=='unresolved' or r['response']['diagnosis']=='Unresolved']
        result.append({'decision_ref':'Candidate%03d'%index,'split':d['split'],'responses':len(rr),
            'collection_complete':len(rr)==len(agents)==5 and {r['agent_id'] for r in rr}==agents,
            'supported':len(rr)-len(unresolved),'unresolved':len(unresolved),
            'supported_only_complete':len(rr)==len(agents)==5 and {r['agent_id'] for r in rr}==agents and not unresolved})
    unresolved=[r for r in data['outcomes'] if r['response']['certainty']=='unresolved' or r['response']['diagnosis']=='Unresolved']
    return {'schema_version':'locdi.c3.routing_groups.v1','decision_count':len(result),'agent_count':len(agents),
        'response_count':len(data['outcomes']),'supported':len(data['outcomes'])-len(unresolved),'unresolved':len(unresolved),
        'named_diagnosis_unresolved':sum(r['response']['diagnosis']!='Unresolved' for r in unresolved),
        'diagnosis_literal_unresolved':sum(r['response']['diagnosis']=='Unresolved' for r in unresolved),
        'correctness_nonnull':sum(r.get('correct') is not None for r in data['outcomes']),
        'all_decisions_have_unresolved':bool(result) and all(r['unresolved'] for r in result),
        'collection_complete_groups':sum(r['collection_complete'] for r in result),
        'supported_only_complete_groups':sum(r['supported_only_complete'] for r in result),'groups':result,
        'clinical_verdicts_generated':0,'review_alone_unlocks_current_groups':False,
        'reason':'UNRESOLVED_DIAGNOSIS_EXCLUDED_AND_ALL_FIVE_AGENTS_REQUIRED'}


def split_matrix(members,*,observed_original,observed_d3,observed_bank,observed_voi,observed_routing,pilot_ids,frozen_pilot_ids):
    sets={k:set(v) for k,v in members.items()}
    observed={'holdout':set(observed_original),'D3_replay':set(observed_d3),'bank':set(observed_bank),
        'voi_train':set(observed_voi),**{('routing_'+k if k in ('train','validation') else k):set(v) for k,v in observed_routing.items()}}
    matches={k:observed[k]==v for k,v in sets.items()}
    return {'schema_version':'locdi.c3.split_independence.v1','membership_recheck':'PASS' if all(matches.values()) and set(pilot_ids)==set(frozen_pilot_ids) else 'FAIL',
        'counts':{k:len(v) for k,v in sets.items()},'current_matches_frozen':matches,
        'pairwise_overlaps':{a+'__'+b:len(sets[a]&sets[b]) for a,b in combinations(sorted(sets),2)},
        'routing_validation_in_voi_train':len(sets['routing_validation']&sets['voi_train']),
        'healthbench_development_samples':len(pilot_ids),'healthbench_frozen_selection_matches':set(pilot_ids)==set(frozen_pilot_ids),
        'healthbench_patient_independence':'UNKNOWN_NO_PATIENT_LINKAGE',
        'original_and_D3_development_exposed':True,'blind_test_approved':False,'split_reassigned':False,
        'required_independence':[
            {'purpose':'Routing Utility Validation','requirement':'Disjoint patients from Routing train; no test-based tuning; independent reviewed query labels'},
            {'purpose':'VoI Validation','requirement':'Disjoint from outcome/prior estimation and artifact tuning; source/time/action references'},
            {'purpose':'Combined System Validation','requirement':'Disjoint from BOTH Routing and VoI training/tuning; current routing validation fails'},
            {'purpose':'Final Paired Evaluation','requirement':'Independent approved test and fixed case/source/repeat manifests across B0-B4; exposed cases not silently blind'},
            {'purpose':'HealthBench External Evaluation','requirement':'Exclude reserved three development samples; disclose unknown contamination/patient linkage; separate approval'}]}


def review_comparison():
    from validation.review_submission_contract import REQUIRED,submission_contract
    schema=load(ROOT/'schemas/routing_review_results_v1.schema.json')
    # Candidate field correspondences, NEVER data transformation instructions.
    mapping={'review_case_ref':'case_ref','decision_ref':'decision_id','reference_source':'source_ref',
        'source_fingerprint':'source_fingerprint','reference_fingerprint':'reference_fingerprint',
        'reviewer_ref':'review.reviewer_ref','reviewer_role':'review.reviewer_kind',
        'reviewer_independent':'review.independent','review_timestamp':'review.reviewed_at',
        'review_policy_version':'review.procedure_version','reference_timing_status':'timing_status',
        'correctness_label':'axes.diagnosis_correctness.verdict','response_ref':'response_ref',
        'response_fingerprint':'response_fingerprint','clinical_evidence_rationale':'clinical_evidence_rationale OR axes.<target>.clinical_evidence_rationale',
        'not_evaluable_reason':'axes.<target>.not_evaluable_reason','decision_status':'review.status OR axes.<target>.status'}
    new_rows=[]
    for key in REQUIRED:
        new_rows.append({'source_field':key,'target_field':mapping.get(key),'type':
            'boolean|null' if key in ('correctness_label','reviewer_independent') else 'object|null' if key=='approval_proof' else 'string|null',
            'semantic_meaning':key,'cardinality':'ONE_SUBMISSION_TO_ONE_REFERENCE_OR_ONE_RESPONSE_AXIS; LEGACY_REQUIRES_JOINED_BUNDLE',
            'lossless_conversion':'NOT_PROVEN','required_review_evidence':'Source/Reference/Response identity and immutable snapshots; explicit per-axis human review',
            'authentication_required':True,'unknown_unresolved':'PRESERVE_NULL_AND_STATUS_NO_IMPUTATION',
            'conversion_blocker':'CONTEXT_AND_SEMANTIC_JOIN_NOT_IMPLEMENTED' if key in mapping else 'NO_EQUIVALENT_LEGACY_FIELD_OR_REQUIRES_EXTERNAL_CONTEXT'})
    legacy=[]
    for scope in ('reference','response','review','scope','axis'):
        definition=schema['$defs'][scope]
        for key in definition['required']:
            prop=definition['properties'][key]
            legacy.append({'source_field':scope+'.'+key,'target_field':next((k for k,v in mapping.items() if v==key or v=='review.'+key),None),
                'type':prop.get('type',prop.get('enum',prop.get('$ref'))),'semantic_meaning':scope+'.'+key,
                'cardinality':'reference=1/decision; response=1/agent-response; axis=3/response; scope=6 distinct input approvals',
                'lossless_conversion':'NOT_PROVEN','required_review_evidence':'Private contextual identity/snapshot lineage and explicit independent judgment',
                'authentication_required':True,'unknown_unresolved':'KEEP_DISTINCT_NOT_REVIEWED_NOT_EVALUABLE_RETROSPECTIVE',
                'conversion_blocker':'NEW_SUBMISSION_ALONE_CANNOT_RECONSTRUCT_LEGACY_BUNDLE'})
    return {'schema_version':'locdi.c3.review_contract_comparison.v1','legacy_schema':'locdi.routing_review_results.v1',
        'new_schema':'locdi.review_submission.v1','submission_to_legacy_fields':new_rows,'legacy_required_fields':legacy,
        'legacy_bundle_required_fields':schema['required'],'new_contract':submission_contract(),
        'conversion_status':'NOT_IMPLEMENTED_NO_AUTOMATIC_CONVERSION','evidence_authentication':'NOT_IMPLEMENTED',
        'actual_path':['actual_gate_inputs -> empty_review_results + private sources/contexts -> validate_evaluation_gate -> legacy validator only if policy passes',
            'assess_official_evaluation -> validate_submission -> review_submission_checks (no transformation into correctness_inputs)',
            'assess_official_evaluation -> assess_artifacts(correctness_inputs) -> existing validate_evaluation_gate'],
        'semantic_gaps':['PRIVATE_PATIENT_VISIT_AGENT_REFERENCE_IDENTITY_JOIN','OBSERVATION_MEMORY_CONDITION_FINGERPRINTS',
            'REFERENCE_CONTENT_AND_LABEL_CATALOG','SEPARATE_REFERENCE_THEN_RESPONSE_REVIEW_TIMESTAMPS',
            'LEGACY_FIXED_POLICY_HASH_AND_PROCEDURE_VERSION','THREE_DISTINCT_AXIS_VERDICTS',
            'MEMORY_RELEVANCE_LABELS_NOT_MEMORY_REVIEWED_BOOLEAN','ACTUAL_TIMING_NOT_VISIBILITY_REVIEWED_BOOLEAN',
            'APPROVAL_PROOF_HAS_NO_LEGACY_SLOT_OR_TRUST_VERIFIER','NEW_REVIEWED_NON_DIAGNOSIS_TARGET_HAS_NO_GENERIC_AXIS_VERDICT_FIELD',
            'NEW_RETROSPECTIVE_REVIEW_CANNOT_PASS_CURRENT_REVIEWED_SUBMISSION_RULE'],
        'review_dimensions':{k:'SEPARATE_REQUIRED_NOT_MERGED_INTO_DIAGNOSIS_BOOLEAN' for k in
            ('Reference Validity','Response Correctness','Memory Relevance','Source Timing','Routing Appropriateness')},
        'clinical_review_ready':False,'clinical_verdicts_generated':0}


def temporal_matrix(original,actions,ledger):
    events=[e for c in original for e in c['environment']['current_encounter_events']]
    return {'schema_version':'locdi.c3.temporal_visibility.v1','rows':[
        {'field':'source_recorded_at','Original':'source_event_store.timestamp; date/record precision only','D3':'action.recorded_at / ledger.source_recorded_at','clinical_time_proof':False},
        {'field':'observed_at','Original':'past_visits.records.observed_at; structural ordering checked','D3':'memory observed_at from recorded interval','clinical_time_proof':False},
        {'field':'available_at','Original':'Memory availability surrogate, not actual clinician release','D3':'actual_available_at null; proxy separated','clinical_time_proof':False},
        {'field':'actual_request_at','Original':'NOT_RECORDED_IN_CLINICAL_SOURCE','D3':'NOT_RECORDED_IN_CLINICAL_SOURCE','clinical_time_proof':False},
        {'field':'actual_receipt_at','Original':'NOT_RECORDED_IN_CLINICAL_SOURCE','D3':'NOT_RECORDED_IN_CLINICAL_SOURCE','clinical_time_proof':False},
        {'field':'proxy_available_minute','Original':'event.available_minute: Ask0/Test1','D3':'action.proxy_available_minute=1','clinical_time_proof':False},
        {'field':'result_received','Original':'Scheduler simulation event after order/start; real source availability remains unverified','D3':'same proxy replay Scheduler path','clinical_time_proof':False},
        {'field':'decision_cutoff','Original':'index encounter + runtime minute; Memory identity cutoff','D3':'canonical decision_time / source interval policy','clinical_time_proof':False}],
        'Original_event_count':len(events),'Original_proxy_minutes':dict(Counter(str(e['available_minute']) for e in events)),
        'D3_action_count':len(actions),'D3_ledger_count':len(ledger),
        'D3_actual_available_count':sum(a.get('actual_available_at') is not None for a in actions),
        'D3_proxy_minutes':dict(Counter(str(a['proxy_available_minute']) for a in actions)),
        'source_consult_count':sum(e['action']=='Consult' for e in events+actions),
        'proxy_permitted_for':['Explicit development replay','Software action/release boundary','Model-based TEST_ONLY calculations'],
        'strict_or_independent_timing_required_for':['Clinical turnaround/efficiency','Actual information availability','Current-decision reference validity','Clinical timing-dependent VoI claims'],
        'reference_timing_review_must_not_be_inferred_from_proxy':True}


def baseline_dependencies(configs):
    rows=[]
    for name,c in configs.items():
        f=c['features'];common=c['common']
        rows.append({'baseline':name,'features':f,'declared_ready':c['ready'],'research_status':'BLOCKED',
            'clinical_gold':'BLOCKED_ALL','memory_labels':'BLOCKED_FOR_MEMORY_QUALITY_METRICS' if f['memory_mode']!='none' else 'NOT_APPLICABLE',
            'routing_utility':'BLOCKED' if name in ('B3','B4') else 'NOT_APPLICABLE',
            'voi_artifact':'BLOCKED' if name=='B4' else 'NOT_APPLICABLE',
            'model_prompt_registry_binding':'BLOCKED','action_catalog':'BLOCKED_COMMON_NOT_BOUND',
            'evaluation_metric_translator':'NOT_IMPLEMENTED','paired_runner':'NOT_IMPLEMENTED',
            'patient_split':'BLOCKED_COMMON_NULL_AND_INDEPENDENCE_UNAPPROVED',
            'budget_timeout_failure_policy':'BLOCKED_FORMAL_POLICY_NOT_BOUND',
            'unconfigured_common_fields':[k for k,v in common.items() if v is None],
            'sequential_handoff':'NOT_SPECIFIED_NOT_INFERRED',
            'comparison_policy':'USER_CONFIRMED_SWITCHES_NOT_PAPER_APPROVED_NO_SINGLE_FEATURE_CAUSAL_CLAIM'})
    return {'schema_version':'locdi.c3.baseline_dependencies.v1','rows':rows,'conditions_changed':False,'experiments_executed':0,
        'memory_labels_scope':'Not required for B0 execution; required for retrieval-quality claims, not mere memory access',
        'test_only_ABCD_are_official_baselines':False}


def policy_options():
    return [
        {'option':'A_KEEP_FIVE_AGENT_COMPLETE','state':'PROPOSED_NOT_SELECTED','utility_formula_change':False,'correctness_gate_change':False,
            'clinical_review':'Independent current Reference, each supported response, abstention separately, visibility/memory/cutoff',
            'bias':'Complete-case exclusion may select easy cases; cannot remove unresolved agents',
            'recollection':'Required for usable diagnosis groups in this fixed 75-response baseline; review alone insufficient',
            'evaluation_scope':'Current supported-only complete groups 0','B3_B4':'Preserves accuracy-only semantics and same candidate Registry'},
        {'option':'B_MODEL_UNRESOLVED_SEPARATELY','state':'PROPOSED_NOT_SELECTED','utility_formula_change':'YES_IF_OPTIMIZING_ABSTENTION_UTILITY; NO_IF_DESCRIPTIVE_SIDECAR_ONLY',
            'correctness_gate_change':'NEW_TARGET_OR_VERSION_FOR_FITTING; CURRENT_DIAGNOSIS_GATE_UNCHANGED',
            'clinical_review':'Independent appropriateness/contribution targets, explicit utility of abstention and correctness',
            'bias':'Coverage-risk tradeoff; abstention not automatically wrong or right','recollection':'Existing outputs useful for review, new training sufficiency UNKNOWN',
            'evaluation_scope':'Separate response coverage and abstention appropriateness; no Boolean imputation',
            'B3_B4':'Research-target change if router utility changes; requires protocol decision'},
        {'option':'C_PARTIAL_EVALUABLE_AGENTS','state':'PROPOSED_NOT_SELECTED','utility_formula_change':'CORE_SAMPLE_VALIDATOR_AND_CANDIDATE_SUPPORT_CHANGE_REQUIRED; score algebra could remain',
            'correctness_gate_change':True,'clinical_review':'Per-agent supported responses plus missingness/abstention review',
            'bias':'Missing-not-at-random selection, unequal specialty observations and positive support, optimistic conditional accuracy',
            'recollection':'May reduce recollection but existing usable sample sufficiency UNKNOWN',
            'evaluation_scope':'Not permitted by current five-agent validator; cannot claim existing fit ready',
            'B3_B4':'Changes support/candidate access; unequal comparisons unless separately approved'},
        {'option':'D_REFERENCE_THEN_NEW_COLLECTION','state':'PROPOSED_NOT_SELECTED','utility_formula_change':False,'correctness_gate_change':False,
            'clinical_review':'Source/initial information/reference/cutoff reviewed before collection; independent response judgments after',
            'bias':'Case/candidate/Prompt redesign and selecting only successful reruns must be prevented',
            'recollection':'YES separate version and explicit later generation approval; not executed here',
            'evaluation_scope':'No guarantee responses become supported; preserve all failures and original 75',
            'B3_B4':'Compatible only with common new Registry/catalog/Prompt/split/binding fixed across comparisons'}]


def research_purposes():
    descriptions=[
        ('Software Workflow Verification','Schema, action source, clock, deterministic offline client','Existing 50/29 transport and source replay','Real-time clinical interpretation','READY_DEVELOPMENT_ONLY','NOT_APPLICABLE','Execution/event contract, no clinical correctness needed','NO_FOR_EXISTING_DEVELOPMENT_SCOPE'),
        ('Longitudinal Memory Retrieval','Same patient past records, ranked retrieval and cutoff','72590 Original and 1256 D3 source-linked records','Reviewed relevance/stale/conflict labels, real availability','READY_STRUCTURAL','BLOCKED','Independent decision-specific eligible relevant set','HUMAN_REVIEW_AND_POSSIBLE_SOURCE_ACQUISITION'),
        ('Source-backed Action Replay','Recorded answer/result and release ledger','Original Ask50/Test200; D3 Ask29/Test301','Source Consult, real request/receipt visibility','READY_EXPLICIT_PROXY_ONLY','BLOCKED_STRICT_TIMING','Source/semantic/time review per action','YES_FOR_UNSUPPORTED_ACTIONS_AND_ACTUAL_TIMING'),
        ('Diagnosis Accuracy','Sufficient current observations and valid independent reference','Unreviewed Condition candidates only','Chief complaint/current symptoms/initial findings and approved Gold','READY_RESPONSE_CONTRACT_ONLY','BLOCKED','Current-decision diagnosis plus independent response judgment','SOURCE_AND_HUMAN_REVIEW_REQUIRED_NO_GUARANTEE'),
        ('Adaptive Routing Effectiveness','Comparable candidate agents, reviewed training/validation outcomes','75 real specialty outputs; Registry5','Complete reviewed groups, appropriateness/contribution references, utility','READY_CONTRACT_ONLY','BLOCKED','Diagnosis and/or routing-success objectives kept separate','POLICY_REVIEW_AND_LIKELY_NEW_COLLECTION'),
        ('Bayesian VoI Clinical Benefit','Validated prior/likelihood, public state and clinical utility','Existing synthetic outcome model and TEST_ONLY planner','Reviewed likelihood/provenance/utility, independent validation, action-quality reference','READY_TEST_ONLY_MATH','BLOCKED','State-specific acceptable actions and clinical-benefit evaluation','YES_VALIDATED_ARTIFACT_AND_SOURCE'),
        ('Clinical Timing / Efficiency','Actual request/receipt/clinician visibility and censoring','Simulation events and proxy minutes','Observed clinical times and appropriateness reference','READY_SIMULATED_EVENT_ORDER','BLOCKED','Independent timing and clinically comparable workflow','YES_ACTUAL_TIME_SOURCE_REQUIRED')]
    return [{'purpose':p,'required_clinical_input':i,'available_fields':a,'missing_fields':m,'structural_validation':s,
        'clinical_evaluation':c,'independent_reference':r,'additional_data':d} for p,i,a,m,s,c,r,d in descriptions]


def gap_ledger():
    entries=[
        ('G01','INITIAL_INFORMATION','Observation/ER','CC/vitals/objective/history 0/50; D3 CC/vitals/objective/current symptoms 0/29','SOURCE_ACQUISITION','C3-02','Current clinical context, source pointers and lawful access'),
        ('G02','RESEARCH_SUITABILITY','Dataset protocol','Original wellness33/emergency3; limited source actions, D3 no reviewed initial context','POLICY_DECISION','C3-02','Approved purpose and inclusion/adequacy criteria'),
        ('G03','REFERENCE','Clinical reference','Source Condition is unreviewed; original22/26 and Candidate010 future date','HUMAN_REVIEW','C3-02','Response-blinded current-decision reference and qualified reviewer'),
        ('G04','ROUTING_GROUP','Routing training','All15 have unresolved; supported-only five-agent groups0 even before review','POLICY_DECISION','C3-03','Choice among complete/abstention/partial/new-collection policies'),
        ('G05','REVIEW_INTEROP','Correctness submissions','Legacy bundle and new individual submission have no lossless adapter','MIXED','C3-04','Identity/snapshot/axis/procedure links, conversion specification and consent'),
        ('G06','REVIEW_AUTHENTICATION','Research approval','No trusted signer/reviewer/authority verification','MIXED','C3-04','Authority registry, provenance and approval procedure'),
        ('G07','MEMORY_REFERENCE','Memory metrics','Ranked records exist; independent relevant/stale/conflict labels missing','HUMAN_REVIEW','C3-02','Per-query relevance and cutoff review, not memory_reviewed Boolean'),
        ('G08','TEMPORAL','Source visibility','Proxy release and simulation receipt do not prove actual availability','MIXED','C3-02','Actual recorded/request/receipt/visibility evidence and precision'),
        ('G09','INDEPENDENCE','Combined validation','Routing validation5 overlap VoI train173; test sets development-exposed','MIXED','C3-03','Cross-module patient lineage and independent approved evaluation scope'),
        ('G10','ARTIFACT','Routing/VoI','No reviewed fitted Utility; synthetic VoI and unreviewed likelihood/clinical Utility','MIXED','STAGE_3_AFTER_C3-04','Reviewed groups, support, independent validation, bindings/lineage and fitting approval'),
        ('G11','ACTIONS','Action expansion','No source Consult; Original Test4 vs D3 targets33; no actual clinical timings','SOURCE_ACQUISITION','STAGE_4','Lawful source results and reference/time review before new action allowlist'),
        ('G12','EVALUATOR','Official B0-B4','Common bindings null; official runner/metric translator unimplemented','CODE_ONLY','STAGE_5_AFTER_STAGE_3_4','Approved protocol/config/metrics/manifest and execution approval'),
        ('G13','EXTERNAL','HealthBench','5000 inputs and Python environment ready, Grader binding/quality/approval absent','MIXED','STAGE_5_EXTERNAL_SEPARATE','Independent grader validation, bounded calls, pilot exclusion and explicit approval'),
        ('G14','PAPER_PROTOCOL','Research protocol','Definitive manuscript missing; user-approved B0-B4 switches remain engineering contract','POLICY_DECISION','C3-02','Definitive paper version/equations and approved metric/Handoff design')]
    deps={'G01':['G02'],'G03':['G01','G02'],'G04':['G03'],'G05':['G03','G04'],'G06':['G03'],
        'G07':['G01','G03'],'G08':['G01'],'G09':['G02'],'G10':['G03','G04','G05','G06','G09'],
        'G11':['G01','G08'],'G12':['G02','G03','G09','G10','G14'],'G13':['G02','G06']}
    return [{'gap_id':g,'category':cat,'affected_module':module,'observed_evidence':e,'current_state':'BLOCKED',
        'research_impact':'Blocks corresponding clinical claim or evaluation; development-only audit may pass',
        'possible_resolutions':{'classification':kind,'status':'PROPOSED','description':need},
        'requires_human_review':kind!='CODE_ONLY','requires_code_change':kind in ('CODE_ONLY','MIXED'),
        'required_source':need,'upstream_dependencies':deps.get(g,[]),
        'downstream_blocked_tasks':['Clinical research readiness','Official B0-B4 evaluation'] if g!='G13' else ['External HealthBench scoring'],
        'verification_method':'Immutable source/snapshot hashes + TEST_ONLY checks; separate qualified independent review and authority proof where required',
        'priority':1 if stage in ('C3-02','C3-03') else 2 if stage=='C3-04' else 3,'recommended_stage':stage}
        for g,cat,module,e,kind,stage,need in entries]


def audit():
    from app.locdi.source_replay import PUBLIC as D3_PUBLIC,load_replay_cases,STRICT
    from app.locdi.evaluation_gate_audit import actual_gate_inputs
    from validation.routing_correctness_gate import validate_evaluation_gate
    from app.locdi.research_protocol_audit import inspect_rc
    from app.locdi.routing_data_reuse import BANK,read
    from validation.official_evaluation_readiness import assess_official_evaluation
    frozen=verify_freeze(full=False)
    private=load(ROOT/PRIVATE);dm=load(D3_PUBLIC/'dataset_manifest.json')
    if digest({k:v for k,v in dm.items() if k!='manifest_fingerprint'})!=dm['manifest_fingerprint']:raise ValueError('D3_MANIFEST_CHANGED')
    def d3(name):
        entry=dm['files'][name];p=(ROOT/entry['path']).resolve()
        if not p.is_relative_to((ROOT/'outputs/private/simulation_datasets').resolve()) or sha(p)!=entry['sha256']:raise ValueError('D3_CANONICAL_CHANGED')
        return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()] if name.endswith('.jsonl') else load(p)
    original=[json.loads(s) for s in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    initials,actions,refs,dcases,ledger=[d3(n) for n in ('initial_observations.jsonl','action_events.jsonl','reference_candidates.jsonl','cases.jsonl','source_visibility_ledger.jsonl')]
    strict='NOT_CHECKED'
    try:load_replay_cases(policy=STRICT)
    except ValueError as exc:
        if str(exc)=='STRICT_SOURCE_ACTUAL_AVAILABILITY_UNVERIFIED':strict='BLOCKED_ACTUAL_AVAILABILITY_UNVERIFIED'
        else:raise
    data,opts=actual_gate_inputs();gate=validate_evaluation_gate(data,**opts)
    routing=routing_groups(data);routing.update(correctness_gate={k:gate[k] for k in
        ('complete_train_groups','validation_group_count','clinical_review_ready','source_reference_ready','utility_fit_ready','positive_similarity_support','blocker_codes')},policy_options=policy_options())
    b=read(BANK);voi=read(BANK.parent/'voi_model.json')
    selection=load(ROOT/'outputs/private/external_evaluation/healthbench/D5_pilot_v1/selection_manifest.json')
    split=split_matrix(private['private_memberships']['patient_memberships'],
        observed_original=[c['patient_id'] for c in original],observed_d3=[c['patient_id'] for c in dcases],
        observed_bank=[r['training_patient_id'] for r in b],observed_voi=voi['provenance']['training_patients'],
        observed_routing=data['split_manifest']['patients'],pilot_ids=[r['sample_id'] for r in selection['samples']],
        frozen_pilot_ids=private['private_memberships']['healthbench_excluded_pilot_ids'])
    configs={n:load(ROOT/'experiments/configs'/(n+'.json')) for n in ('B0','B1','B2','B3','B4')}
    official=assess_official_evaluation(configs,cases=original)
    documents=sorted(p for p in (ROOT/'docs').rglob('*.md') if p.name.startswith(('D1_','D2_','D3_','D4_','D5_','D6_','D7_','022_','023_','024_','035_','036_')))
    return {'initial_information_matrix':{**initial_matrix(original,initials,actions,refs),'research_purposes':research_purposes()},
        'routing_group_analysis':routing,'review_contract_comparison':review_comparison(),
        'temporal_visibility_matrix':{**temporal_matrix(original,actions,ledger),'D3_strict_loader':strict},
        'split_independence_matrix':{**split,'bank_visits':len(b)},
        'baseline_dependency_matrix':{**baseline_dependencies(configs),'existing_official_gate_status':official['status']},
        'gap_ledger':{'schema_version':'locdi.c3.gap_ledger.v1','decisions_approved':0,'gaps':gap_ledger()},
        'verification_summary':{'schema_version':'locdi.c3.verification.v1','D7_hash_verification':frozen,'RC':inspect_rc(),
            'reused_full_D7_verification':load(ROOT/'outputs/research_protocol/phase30_d7_verify_v1/reports/d7_verification.json'),
            'source_scan_scope':'CURRENT_50_AND_D3_CANONICAL_29_AND_REAL75; NO_LARGE_RAW_CSV_OR_SQLITE_RESCAN',
            'source_documents':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in documents],
            'strict_loader':strict,'clinical_reference_readiness':'BLOCKED','research_evaluation_ready':False,
            'C3_01_AUDIT':'PENDING_TESTS_AND_FINAL_PRESERVATION',
            'stage_plan':{'1':'PASS_DEVELOPMENT_STRUCTURAL_SCOPE','2':'PASS_DEVELOPMENT_FREEZE_SCOPE','3':'CURRENT_C3-01',
                '4':'INCOMPLETE','5':'NOT_STARTED','6':'NOT_STARTED'},
            'actual_generation_calls':0,'healthbench_grading':0,'utility_fitting':0,'voi_probability_estimation':0,'clinical_verdicts_generated':0}}

