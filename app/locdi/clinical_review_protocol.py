"""C3-02 proposed scope/review protocol. No clinical judgments or conversion.

Technical missing-input classifications are conditional on an UNAPPROVED scope,
not declarations that a disease is unassessable. Actual approval is never issued.
"""
from collections import Counter
from datetime import datetime,date,timedelta
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import load,sha

STATES=('DEVELOPMENT_EVALUABLE','CLINICAL_REVIEW_REQUIRED','DATA_INSUFFICIENT',
        'SOURCE_TIMING_UNVERIFIED','REFERENCE_UNAVAILABLE','NOT_APPLICABLE')
PURPOSES=('Software Workflow Verification','Longitudinal Memory Retrieval','Source-backed Action Replay',
          'Diagnosis Accuracy','Specialty Routing Effectiveness','Multi-Step Bayesian VoI','Clinical Timing / Efficiency')
FIELDS=('chief_complaint','initial_vitals','objective_findings','history','current_symptoms')
REFERENCE_OUTCOMES=('CURRENT_DECISION_REFERENCE_VALID','RETROSPECTIVE_OUTCOME_ONLY','REFERENCE_REJECTED',
                    'INSUFFICIENT_INFORMATION','NOT_REVIEWABLE')


def presence(record,key):
    if key not in record:return 'ABSENT'
    value=record[key]
    def content(v):
        if v is None:return False
        if isinstance(v,str):return bool(v.strip())
        if isinstance(v,dict):return any(content(x) for x in v.values())
        if isinstance(v,list):return any(content(x) for x in v)
        return True
    return 'NULL' if value is None else 'EMPTY' if not content(value) else 'VALUE_PRESENT_UNREVIEWED'


def classify_case(facts):
    """Counts/booleans only; no diagnosis, clinical adequacy verdict or Gold."""
    public={k:facts[k] for k in ('dataset','ordinal','initial_presence','encounter_class','memory_count','ask_count',
        'test_count','consult_count','reference_candidate_count','actual_availability_count','development_exposed')}
    initial=any(v=='VALUE_PRESENT_UNREVIEWED' for v in facts['initial_presence'].values())
    assessments=[]
    for purpose in PURPOSES:
        dev=(purpose=='Software Workflow Verification' or purpose=='Longitudinal Memory Retrieval' and facts['memory_count']>0
            or purpose=='Source-backed Action Replay' and facts['ask_count']+facts['test_count']>0
            or purpose=='Clinical Timing / Efficiency')
        states=['DEVELOPMENT_EVALUABLE'] if dev else []
        reasons=[]
        if purpose=='Software Workflow Verification':states.append('NOT_APPLICABLE')
        else:
            states.append('CLINICAL_REVIEW_REQUIRED')
            if purpose in ('Diagnosis Accuracy','Specialty Routing Effectiveness','Multi-Step Bayesian VoI') and not initial:
                states.append('DATA_INSUFFICIENT');reasons.append('NO_DOCUMENTED_INITIAL_CLINICAL_CONTEXT_UNDER_DRAFT_SCOPE')
            if purpose=='Longitudinal Memory Retrieval' and not facts['memory_count']:
                states.append('DATA_INSUFFICIENT');reasons.append('NO_ELIGIBLE_PAST_MEMORY')
            if purpose=='Source-backed Action Replay' and not facts['ask_count']+facts['test_count']:
                states.append('DATA_INSUFFICIENT');reasons.append('NO_SOURCE_REPLAY_ACTION')
            if purpose in ('Longitudinal Memory Retrieval','Diagnosis Accuracy','Specialty Routing Effectiveness','Multi-Step Bayesian VoI'):
                states.append('REFERENCE_UNAVAILABLE');reasons.append('INDEPENDENT_REVIEWED_REFERENCE_OR_RELEVANCE_MISSING')
            if facts['actual_availability_count']<facts['ask_count']+facts['test_count'] or purpose in ('Longitudinal Memory Retrieval','Clinical Timing / Efficiency'):
                states.append('SOURCE_TIMING_UNVERIFIED')
            if purpose=='Specialty Routing Effectiveness':reasons.append('ROUTING_UTILITY_AND_TARGET_POLICY_UNPREPARED')
            if purpose=='Multi-Step Bayesian VoI':reasons.append('RESEARCH_PRIOR_LIKELIHOOD_UTILITY_UNPREPARED')
        assessments.append({'purpose':purpose,'states':sorted(set(states)),'reason_codes':reasons,
            'structural_proxy_scope':'EXPLICIT_DEVELOPMENT_ONLY' if dev else 'NO_RESEARCH_PERFORMANCE_CLAIM',
            'clinical_judgment':'NOT_REVIEWED','information_sufficiency_verdict':None,
            'expert_review_required':purpose!='Software Workflow Verification','clinical_evaluation_approved':False})
    return {**public,'initial_clinical_fields_present':initial,'initial_visibility':'NOT_REVIEWED',
        'reference_candidate_present':facts['reference_candidate_count']>0,'reviewed_reference_available':False,
        'assessments':assessments,'policy_status':'PROPOSED_POLICY_APPROVAL_REQUIRED',
        'diagnosis_or_disease_assessability_verdict':None}


def scope_matrix():
    specifications=[
        (PURPOSES[0],'Clinical initial data not mandatory for transport/event tests','Source action binding and TEST_ONLY response; no invented result','NOT_APPLICABLE','Identity/clock/cutoff for boundary tests','Execution episode/event',
         'All planned scenarios; preserve failed/incomplete/blocked separately, expected failure is not clinical success','Explicit Proxy/Fake clients only for development','Not a clinical performance endpoint'),
        (PURPOSES[1],'Defined reviewed clinical query/task; context may include past memory after scope approval','No new action needed for retrieval-only test','Independent query-specific relevance, not diagnosis Gold by default','Same patient, past visit, observed/available cutoff, stable ranked IDs','Query/decision/cutoff and each eligible record/rank',
         'P@K fixed K draft; recall requires full eligible relevance set; MRR requires nonempty reviewed relevant set. Missing labels blocked, true empty denominator NA','Structural retrieval/identity tests','Independent relevance/stale/conflict and availability review'),
        (PURPOSES[2],'Declared information-access task; empty initial state only under approved scope','Recorded answer/result, source pointer and order/release contract','Not needed for descriptive replay; needed for clinical action appropriateness','Action time/source ledger and eligible memory if used','Action/source-instance at decision',
         'All planned source actions; missing source/blocked actions separate, no manufactured outcomes','Explicit Proxy order/result release; synthetic question text labeled','Reviewed source meaning and real availability for strict clinical claims'),
        (PURPOSES[3],'Human-reviewed contemporaneous task/context and sufficient findings; not a universal requirement to have every field','Only recorded source answers/results available by decision','Independent current-decision diagnosis Reference plus distinct response review','Contemporaneous public state and memory cutoff','Decision Reference first; then each supported response',
         'Supported evaluable responses over reviewed supported denominator; unresolved separate; no false imputation; exclusions/reasons and coverage reported','Response-contract tests only','Human sufficiency/reference review, credentials/authentication and approved scope'),
        (PURPOSES[4],'Same clinically reviewed observation and memory snapshot for all candidate agents','Same action/diagnosis catalogs and fixed Registry/Prompt/model bindings','Separate diagnosis correctness, acceptable specialty and contribution references','Patient-disjoint train/validation and cross-artifact lineage; cutoff','Routing decision/candidate and contribution event',
         'Current accuracy fitting requires intact five-agent supported/reviewed group; conditional selected accuracy plus abstention coverage; policy C3-03 unresolved','Registry/TEST_ONLY router tests','Independent appropriateness/target policy, reviewed Utility and disjoint validation'),
        (PURPOSES[5],'Reviewed public state, task and model-supported diagnosis/action set','Source-bound action outcomes, validated prior/likelihood/clinical utility','Independent acceptable next action and clinical-benefit reference','Current observation only, real visibility where time claims depend on it','Decision/state/action/outcome trajectory',
         'Action-quality denominator independently reviewed executed actions; expected model IG separate from realized change/clinical benefit; unavailable artifacts blocked','TEST_ONLY Bellman/Bayes only','Reviewed non-toy artifact, independent validation, objective approval and source timing'),
        (PURPOSES[6],'Comparable task/severity/start state and terminal/censoring policy','Actual request/receipt/doctor visibility linked to real workflow','Clinical comparability and appropriate finalization review','Actual timestamps with precision/timezone; proxy separate','Action timing interval and episode terminal',
         'Resolved finalized diagnosis event minus arrival; failed/incomplete/unresolved censored/separate; simulation elapsed is not actual turnaround','Simulated event order/elapsed description','Observed actual clinical clocks and independent comparability review')]
    return {'schema_version':'locdi.c3_02.scope.v1','policy_status':'PROPOSED','policy_approved':False,
        'rows':[dict(zip(('purpose','minimum_initial_information','actions_and_source_results','current_diagnosis_reference',
            'memory_and_time','independent_review_unit','denominator_and_exclusions','development_proxy_allowed','clinical_prerequisites'),row))
            for row in specifications],
        'classification_states':list(STATES),'classification_is_clinical_verdict':False,
        'initial_minimum_rule':'Draft task-specific documented clinical context + independent sufficiency review; demographic metadata alone not sufficient under this draft',
        'unapproved_exclusion_not_applied':True}


def policy_register():
    entries=[
        ('A_DEVELOPMENT_ONLY','Use low-initial-information cases only for development',
         'Preserves all existing cases and avoids unsupported clinical claims','No diagnosis/routing clinical efficacy evidence',
         '50 Original + 29 D3 structurally available, kept as separate cohorts','Restricts claims to software/replay; paper original missing',
         'B0-B4 may be developed offline; no official clinical paired claim','No new clinical source needed for existing development','No Core change; approved purpose/labels/gates documentation'),
        ('B_REVIEWED_SUFFICIENT_SUBSET','Include only independently sufficient cases in diagnosis metrics',
         'Aligns clinical denominator with reviewed task','Selection bias, case-mix and response-coverage changes must be reported',
         'Approved clinical subset0; eventual count UNKNOWN before review','Requires preregistered inclusion/exclusion and coverage reporting',
         'Same approved cases across B0-B4, no per-condition outcome-based filtering','Qualified independent reference/sufficiency review; extra source if inadequate','Selection/metric manifest adapter later, no current filtering'),
        ('C_NEW_SOURCE_VERSION','Acquire source-grounded context in a new dataset version',
         'May improve initial context/action/temporal evidence','Availability, license, validity and independence not guaranteed',
         'New approved cases0; obtainable size UNKNOWN','New version/purpose and source lineage disclosed; old Freeze remains baseline',
         'All conditions share new source/split/reference, never selectively replace good runs','Lawful initial context/timestamps and independent reviewer evidence','New version/adapter/validators later; immutable original preserved'),
        ('D_EMPTY_STATE_INFORMATION_ACQUISITION','Explicitly study acquisition from sparse/empty initial state',
         'Can match existing replay and sequential information-access structure','Does not make diagnosis accuracy or clinical optimality valid automatically',
         'Potential79 source-replay cases; approved research cohort0','Reframes claims toward information access; manuscript/protocol approval needed',
         'Memory, specialty access, action timing and selector confounds explicitly recorded','Existing source results for development; independent action-quality/reference for clinical claims','Scope/metric/censoring and candidate policy approval; no invented symptoms')]
    return {'schema_version':'locdi.c3_02.policy_register.v1','approved_policy_count':0,
        'decisions':[{'policy_id':i,'description':d,'advantages':a,'clinical_validity_limits':l,'current_scale':n,
            'paper_claim_impact':p,'B0_B4_comparability':b,'new_source_required':s,'implementation_change':c,
            'human_approval_required':True,'status':'POLICY_APPROVAL_REQUIRED','decision':'PROPOSED_NOT_SELECTED'}
            for i,d,a,l,n,p,b,s,c in entries],
        'approval_decisions_pending':['Study purpose/minimum information','Current vs retrospective endpoint','Inclusion/denominator/exclusion rules',
            'Use of Proxy when actual timing cannot be acquired','Routing unresolved policy C3-03','Reviewer qualification/authority/independence',
            'Cross-module independent split/exposure','Memory query/cutoff/relevance policy','Disagreement/adjudication and final signing procedure']}


def rubrics():
    shared={'disagreement':'Retain independent records, assign separately qualified adjudication; unresolved disagreement stays not approved',
        'approval':'Verified qualification/independence + applicable policy + evidence hashes + trusted authority verification; all currently absent'}
    rows=[
        ('Reference Validity','Response-blinded source/encounter/decision snapshot, available memory/actions and timing evidence',
         'Agent outputs/correctness, model rationale, future results hidden from current-decision assessment; retrospective data only in labeled separate phase',
         'Before response correctness review and model result comparison',list(REFERENCE_OUTCOMES),
         'INSUFFICIENT_INFORMATION and NOT_REVIEWABLE are human verdicts; current state NOT_REVIEWED',
         'Source condition code/name/time, current supporting evidence, available_at proof and explicit basis/rationale'),
        ('Response Correctness','Authenticated independently confirmed current Reference + exact recorded Agent response + same public input snapshot',
         'Other Agent judgments and model-identity cues blinded where feasible; no hindsight facts added to current snapshot',
         'After independent Reference approval and after response recorded_at',
         ['CORRECT','INCORRECT','UNRESOLVED','NOT_EVALUABLE','NOT_REVIEWED'],
         'Unresolved is no diagnosis Boolean; separately review abstention appropriateness',
         'Response/body fingerprint, reference fingerprint, supported/diagnosis consistency, claim/evidence rationale'),
        ('Memory Relevance / Stale / Conflict','Exact query/decision/cutoff and ordered eligible same-patient past records with source times',
         'Agent answers and future records excluded; independent labels blind to system ranking when practical then join recorded ranks',
         'At defined query/cutoff before retrieval metric calculation',
         ['RELEVANT','NOT_RELEVANT','UNCERTAIN','NOT_ASSESSABLE','NOT_REVIEWED'],
         'Stale and conflict are separate dimensions with YES/NO/UNCERTAIN/NOT_ASSESSABLE, not automatically irrelevant',
         'Record IDs privately mapped, source/evidence, context, supersession/conflict pair and uncertainty; full eligible set for recall'),
        ('Source Timing / Visibility','Original timestamp/precision, actual request/receipt/available/doctor visibility and source ledger',
         'Proxy kept labeled; no future event or final condition used to infer earlier knowledge',
         'Before current-input/Reference admission and before any strict timing metric',
         ['VERIFIED_AVAILABLE_BY_CUTOFF','VERIFIED_AFTER_CUTOFF','PROXY_ONLY','UNVERIFIED','NOT_ASSESSABLE','NOT_REVIEWED'],
         'Missing evidence stays UNVERIFIED; cannot derive actual receipt from recorded_at or simulation result_received',
         'Immutable original log/source pointer, timezone/precision, cutoff, reviewer rationale and separate proxy assumption'),
        ('Routing Appropriateness','Reviewed current clinical task/public snapshot, eligible specialties and later documented contribution in separate scope',
         'Gold department excluded from Agent input; diagnosis correctness and other Agent results not substituted for contribution',
         'Target/acceptable specialty reviewed before outcome-based contribution assessment',
         ['APPROPRIATE','INAPPROPRIATE','ACCEPTABLE_SET','APPROPRIATE_ABSTENTION','NOT_EVALUABLE','NOT_REVIEWED'],
         'Separate unique target/acceptable set/abstention/contribution; no automatic correctness from label department',
         'Independent department/contribution reference and policy, selected/abstained/unavailable status, actual handoff/action linkage')]
    return {'schema_version':'locdi.c3_02.rubrics.v1','policy_status':'PROPOSED','actual_reviewers_assigned':0,
        'axes':[{**dict(zip(('axis','reviewer_visible_input','withheld_input','review_order','permitted_judgments','unresolved_rules','required_evidence'),r)),**shared}
            for r in rows],
        'memory_separate_dimensions':{'relevance':'Per query/cutoff/record','stale':'As-of currency; old date alone not stale','conflict':'Named record pairs and as-of resolution, not automatic diagnosis disagreement'},
        'actual_clinical_verdicts':0}


def reference_protocol():
    return {'schema_version':'locdi.c3_02.reference_protocol.v1','protocol_status':'PROPOSED_POLICY_APPROVAL_REQUIRED',
        'existing_packet_files':81,'existing_unique_reference_targets':80,'packets_changed':False,
        'permitted_human_outcomes':list(REFERENCE_OUTCOMES),'actual_reference_status':'NOT_REVIEWED',
        'current_reference':'Independent contemporaneous evidence sufficient for the defined decision at its exact cutoff; Source Condition alone is a candidate',
        'retrospective_outcome':'Later outcome can support an explicitly retrospective endpoint; never relabel as current gold or expose to the Agent',
        'insufficient_information':'Human cannot establish requested inference from available context; do not generate missing symptoms or label a disease impossible',
        'not_reviewable':'Source/access/lineage or reviewer competence/evidence missing; distinct from rejected reference',
        'steps':['Approve scope, policies and permitted claim','Verify qualified clinical reviewer via authoritative credential issuer and institutional authority',
            'Record role/specialty/license status/scope/expiration check and private credential evidence; no fictitious reviewer',
            'Check independence/conflicts, blinding and access controls; self-report alone insufficient',
            'Verify source/case/decision fingerprints, reference date/time and actual visibility/precision',
            'Perform response-blinded Reference and sufficiency review, retaining uncertainty',
            'Record disagreement separately and adjudicate without overwriting original judgments',
            'Authorized authority signs immutable reference record; unverified signatures remain blocked',
            'Only after authenticated current Reference, separately review exact Agent responses and axis-specific evidence'],
        'required_evidence_fields':['review_target_ref','dataset_version','private_patient_visit_decision_link','source_fingerprint',
            'case_fingerprint','decision_snapshot_fingerprint','reference_fingerprint','response_fingerprint_when_applicable',
            'observation_memory_fingerprint','reviewer_ref','reviewer_role','qualification_evidence_ref','credential_verification_source',
            'credential_verified_at','independence_evidence','policy_version_and_fingerprint','review_timestamp',
            'reference_timing_status','evidence_citations','response_blinded','uncertainty_reason','axis_specific_judgment',
            'disagreement_and_adjudication_refs','approval_authority_ref','approval_proof','immutable_record_hash'],
        'authority_verifier':'NOT_IMPLEMENTED','legacy_adapter':'NOT_IMPLEMENTED_DEFERRED_C3_04',
        'legacy_contract':'locdi.routing_review_results.v1','new_contract':'locdi.review_submission.v1',
        'interop_constraints':['Private identity and source/snapshot/Label/Judgment joins required',
            'Separate five review meanings and legacy three response axes','Legacy fixed procedures/policy fingerprint not arbitrary nonempty version',
            'New NOT_EVALUABLE may omit reason; legacy axis requires it','New non-diagnosis target has no generic axis verdict slot',
            'Reference reviewer response-blind; response judged only after independent Reference','Approval proof must be authenticated, not just self-hashed'],
        'clinical_review_ready':False,'policy_approved':False,'automatic_conversion':False}


def check_transition(current,target,evidence,*,test_only=False):
    """Procedure-state validator only. Never produces or authenticates a verdict."""
    edges={'DRAFT':{'REVIEW_REQUESTED'},'REVIEW_REQUESTED':{'SUBMITTED'},'SUBMITTED':{'DISAGREEMENT','APPROVAL_PENDING'},
           'DISAGREEMENT':{'ADJUDICATION_PENDING'},'ADJUDICATION_PENDING':{'APPROVAL_PENDING'},'APPROVAL_PENDING':{'APPROVED'}}
    codes=[]
    if target not in edges.get(current,set()):codes.append('INVALID_REVIEW_STATE_TRANSITION')
    if not isinstance(evidence,dict):evidence={};codes.append('EVIDENCE_CONTRACT_INVALID')
    def require(keys):
        for key in keys:
            if evidence.get(key) is not True:codes.append('MISSING_'+key.upper())
    if target in ('SUBMITTED','APPROVAL_PENDING','APPROVED'):
        require(('policy_binding','source_snapshot_binding','qualified_reviewer_evidence','independence_evidence'))
        if evidence.get('review_axis') not in ('Reference Validity','Response Correctness','Memory Relevance / Stale / Conflict','Source Timing / Visibility','Routing Appropriateness'):
            codes.append('REVIEW_AXIS_REQUIRED')
        if evidence.get('reviewer_kind')!='HUMAN_CLINICAL_REVIEWER':codes.append('QUALIFIED_HUMAN_REVIEWER_REQUIRED')
    if evidence.get('review_axis')=='Reference Validity' and target in ('SUBMITTED','APPROVAL_PENDING','APPROVED'):
        require(('response_blinded',))
        if evidence.get('response_visible') or evidence.get('diagnosis_correctness_label') is not None:
            codes.append('REFERENCE_REVIEW_MUST_NOT_INCLUDE_RESPONSE_JUDGMENT')
    if evidence.get('review_axis')=='Response Correctness' and target in ('SUBMITTED','APPROVAL_PENDING','APPROVED'):
        require(('reference_independently_confirmed','reference_authentication_verified','response_fingerprint_bound','response_after_reference_review'))
    if evidence.get('certainty')=='unresolved' and evidence.get('diagnosis_correctness_label') is not None:
        codes.append('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN')
    if evidence.get('source_time_basis')=='PROXY' and evidence.get('strict_clinical_timing_claim'):
        codes.append('PROXY_IS_NOT_ACTUAL_CLINICAL_TIME')
    if target in ('APPROVAL_PENDING','APPROVED') and evidence.get('disagreement_unresolved'):
        codes.append('UNRESOLVED_REVIEW_DISAGREEMENT')
    simulated=test_only is True and evidence.get('purpose')=='TEST_ONLY'
    if target in ('APPROVAL_PENDING','APPROVED'):
        if simulated:require(('test_only_authority_fixture',))
        else:codes.append('AUTHENTICATION_AND_APPROVAL_AUTHORITY_NOT_IMPLEMENTED')
    return {'transition_contract_allowed':not codes,'simulation_only':simulated,'blocker_codes':sorted(set(codes)),
        'clinical_review_ready':False,'policy_approved':False,'clinical_verdicts_generated':0,
        'authority_status':'TEST_ONLY_FIXTURE_NOT_REAL_APPROVAL' if simulated else 'NOT_IMPLEMENTED'}


def temporal_priorities(original,d3cases,d3refs,data,contexts):
    context={r['decision_id']:r for r in contexts};rows=[];private=[]
    targets=[('Original',22),('Original',26),('D3',19),('RoutingCandidate',10)]
    for dataset,ordinal in targets:
        if dataset=='Original':
            c=original[ordinal-1];a=c['agent_visible'];at=a['index_encounter_start'];pid=c['patient_id'];visit=a['index_encounter_id']
            refs=[{'source_condition_start':r['source_record']['START'],'source':r} for r in c['provenance']['source_case_audit']['diagnosis_sources']]
            decision_ref='INDEX_DECISION';snapshot_fp=digest(a['current_initial_state'])
        elif dataset=='D3':
            c=d3cases[ordinal-1];at=c['decision_time'];pid=c['patient_id'];visit=c['visit_id'];refs=[r for r in d3refs if r['case_id']==c['case_id']]
            decision_ref=c['case_id'];snapshot_fp=digest(c)
        else:
            c=data['decisions'][ordinal-1];ctx=context[c['decision_id']];pid=c['patient_id'];visit=c['visit_id']
            at=(datetime.fromisoformat(ctx['encounter_started_at'].replace('Z','+00:00'))+timedelta(minutes=c['minute'])).isoformat()
            refs=[{'source_condition_start':r['source_record']['START'],'source':r} for r in ctx['reference_candidates']['source_condition_rows']]
            decision_ref=c['decision_id'];snapshot_fp=digest(c['observation'])
        deltas=[(date.fromisoformat(r['source_condition_start'][:10])-date.fromisoformat(at[:10])).days for r in refs]
        safe_ref=dataset+'%03d'%ordinal
        rows.append({'target_ref':safe_ref,'dataset':dataset,'ordinal':ordinal,'reference_candidate_count':len(refs),
            'date_delta_days':deltas,'priority':1,'reason':'FUTURE_CONDITION_OR_DAY_PRECISION_REQUIRES_CURRENT_VS_RETROSPECTIVE_REVIEW',
            'source_availability':'NOT_VERIFIED','reference_timing_status':'REFERENCE_TIMING_NOT_REVIEWED',
            'human_verdict':None,'candidate010_is_holdout_case010':False if dataset=='RoutingCandidate' else 'NOT_APPLICABLE'})
        private.append({'target_ref':safe_ref,'patient_id':pid,'visit_id':visit,'decision_ref':decision_ref,'decision_time':at,
            'source_references':refs,'snapshot_fingerprint':snapshot_fp,'scope':'PRIVATE_SOURCE_LINK_ONLY_NO_REVIEW_VERDICT'})
    return {'schema_version':'locdi.c3_02.temporal_priorities.v1','rows':rows,'clinical_verdicts_generated':0},private


def workload(data,original,d3cases):
    routing_memory=sum(len(d['observation'].get('longitudinal_memory',{}).get('records',[])) for d in data['decisions'])
    return {'schema_version':'locdi.c3_02.workload.v1','existing_packet_files':81,'existing_unique_reference_targets':80,
        'existing_reference_composition':{'Original':50,'D3':29,'RoutingCandidate010':1},
        'routing_decisions':len(data['decisions']),'routing_response_review_units':len(data['outcomes']),
        'additional_reference_decisions_for_other_routing':len(data['decisions'])-1,
        'proposed_reference_decision_units_if_all_included':len(original)+len(d3cases)+len(data['decisions']),
        'routing_selected_memory_snapshot_refs':routing_memory,'routing_shared_snapshot_count':len(data['decisions']),
        'memory_query_cohort_and_full_relevance_review_units':'UNKNOWN_SCOPE_AND_QUERY_NOT_APPROVED',
        'source_action_instances_by_dataset':{'Original':sum(len(c['environment']['current_encounter_events']) for c in original),
            'D3':sum(len(c['action_ids']) for c in d3cases)},'actual_timing_evidence_verified':0,
        'no_case_count_155':True,'reviewers_assigned':0,'reviewer_count':'UNKNOWN','throughput':'UNKNOWN','completion_capacity':'UNKNOWN',
        'priorities':[{'priority':1,'unit':'Four temporal priority targets','prerequisite':'Source linkage, credential and scope approval'},
            {'priority':2,'unit':'Routing15 independent decision references then75 responses','prerequisite':'Reference-blind confirmation and C3-03 policy;14 reference packets not yet prepared'},
            {'priority':3,'unit':'Original50/D3_29 initial sufficiency by approved task','prerequisite':'No automatic exclusion; same decision may already be in temporal review'},
            {'priority':4,'unit':'Query/cutoff/eligible record relevance and rank labels','prerequisite':'Approved query cohort, private mapping, Recall eligible-set coverage'},
            {'priority':5,'unit':'Original250/D3_330 source action-instance visibility','prerequisite':'Actual request/receipt/doctor visibility source; proxy does not fill missing evidence'}],
        'unit_warning':'80 reference targets and75 responses are overlapping hierarchical review units, not155 independent cases;94 proposed decision targets are not94 reviewed cases',
        'schedule':[{'window':'2026-11-09..11','proposal':'Scope, policies, qualifications/authority and access decisions','status':'PROPOSED_NO_ASSIGNMENT'},
            {'window':'2026-11-12..15','proposal':'Four temporal targets and blinded Reference pilot, only if prerequisites met','status':'PROPOSED_NO_ASSIGNMENT'},
            {'window':'2026-11-16..20','proposal':'Routing references then conditional response review, C3-03 decisions','status':'PROPOSED_NO_ASSIGNMENT'},
            {'window':'2026-11-21..25','proposal':'Query relevance and actual source timing evidence; scope may remain blocked','status':'PROPOSED_NO_ASSIGNMENT'},
            {'window':'2026-11-26..29','proposal':'Disagreement/adjudication and immutable approval evidence; C3-04 authority still required','status':'PROPOSED_NO_ASSIGNMENT'}]}


def _protocol_check(outputs):
    required={'evaluation_scope_matrix','case_suitability_matrix','reference_review_protocol','review_axis_rubrics',
              'reviewer_workload_plan','policy_decision_register','temporal_review_priorities'}
    codes=[]
    if set(outputs)!=required:codes.append('REQUIRED_PROTOCOL_OUTPUT_MISSING')
    if not codes:
        cases=outputs['case_suitability_matrix'];scope=outputs['evaluation_scope_matrix'];r=outputs['reference_review_protocol'];p=outputs['policy_decision_register']
        if {x.get('purpose') for x in scope.get('rows',[])}!=set(PURPOSES) or len(outputs['review_axis_rubrics'].get('axes',[]))!=5:codes.append('PURPOSE_OR_RUBRIC_COVERAGE_INVALID')
        scope_keys={'purpose','minimum_initial_information','actions_and_source_results','current_diagnosis_reference','memory_and_time',
            'independent_review_unit','denominator_and_exclusions','development_proxy_allowed','clinical_prerequisites'}
        if any(set(x)!=scope_keys or any(not isinstance(v,str) or not v for v in x.values()) for x in scope.get('rows',[])):
            codes.append('MINIMUM_SCOPE_SPECIFICATION_INCOMPLETE')
        if Counter(c['dataset'] for c in cases.get('cases',[]))!=Counter({'Original':50,'D3':29}):codes.append('CASE_COVERAGE_INVALID')
        if (len({(c['dataset'],c['ordinal']) for c in cases.get('cases',[])})!=79 or
                any({a['purpose'] for a in c['assessments']}!=set(PURPOSES) or any(set(a['states'])-set(STATES) for a in c['assessments']) for c in cases.get('cases',[]))):
            codes.append('CASE_CLASSIFICATION_INVALID')
        if len(p.get('decisions',[]))!=4 or p.get('approved_policy_count')!=0 or any(x.get('status')!='POLICY_APPROVAL_REQUIRED' for x in p.get('decisions',[])):codes.append('UNAPPROVED_POLICY_PROMOTION')
        if (scope.get('policy_approved') is not False or r.get('clinical_review_ready') is not False or
                cases.get('clinical_reference_approved_count')!=0 or any(a.get('clinical_evaluation_approved') is not False for c in cases.get('cases',[]) for a in c['assessments'])):
            codes.append('UNREVIEWED_CLINICAL_PROMOTION')
        if (set(r.get('permitted_human_outcomes',[]))!=set(REFERENCE_OUTCOMES) or r.get('actual_reference_status')!='NOT_REVIEWED'
                or r.get('authority_verifier')!='NOT_IMPLEMENTED' or r.get('automatic_conversion') is not False):
            codes.append('REFERENCE_PROTOCOL_PROMOTION_OR_MISSING_CONTRACT')
        if len(outputs['temporal_review_priorities'].get('rows',[]))!=4:codes.append('PRIORITY_TARGETS_MISSING')
    return {'protocol_contract_status':'PASS' if not codes else 'BLOCKED','blocker_codes':codes,
        'C3_02_POLICY_APPROVED':False,'CLINICAL_REFERENCE_READY':False,'OFFICIAL_B0_B4_READY':False,
        'approval_authority':'NOT_IMPLEMENTED','clinical_verdicts_generated':0}


def protocol_check(outputs):
    try:return _protocol_check(outputs)
    except (KeyError,TypeError,ValueError,AttributeError):
        return {'protocol_contract_status':'BLOCKED','blocker_codes':['PROTOCOL_INPUT_CONTRACT_INVALID'],
            'C3_02_POLICY_APPROVED':False,'CLINICAL_REFERENCE_READY':False,'OFFICIAL_B0_B4_READY':False,
            'approval_authority':'NOT_IMPLEMENTED','clinical_verdicts_generated':0}


def build():
    import json
    from app.locdi.source_replay import PUBLIC
    from app.locdi.evaluation_gate_audit import actual_gate_inputs
    base=ROOT/'outputs/research_evaluation/c3_01_v3';manifest=load(base/'analysis_manifest.json');copy=dict(manifest);fp=copy.pop('fingerprint')
    if digest(copy)!=fp or any(sha(ROOT/n)!=h for n,h in manifest['files'].items()):raise ValueError('C3_01_MANIFEST_CHANGED')
    if load(base/'verification_summary.json')['C3_01_AUDIT']!='PASS_DEVELOPMENT_ONLY':raise ValueError('C3_01_NOT_VALID')
    dm=load(PUBLIC/'dataset_manifest.json')
    def d3(name):
        entry=dm['files'][name];path=(ROOT/entry['path']).resolve()
        if not path.is_relative_to((ROOT/'outputs/private/simulation_datasets').resolve()) or sha(path)!=entry['sha256']:raise ValueError('CANONICAL_SOURCE_CHANGED')
        return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()] if name.endswith('.jsonl') else load(path)
    original=[json.loads(s) for s in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    cases,initials,actions,refs,memory,ledger=[d3(n) for n in ('cases.jsonl','initial_observations.jsonl','action_events.jsonl','reference_candidates.jsonl','memory.json','source_visibility_ledger.jsonl')]
    public=[];private=[]
    for dataset,values in (('Original',original),('D3',cases)):
        for index,c in enumerate(values,1):
            if dataset=='Original':
                a=c['agent_visible'];doctor=a['current_initial_state']['doctor_visible_info'];ev=c['environment']['current_encounter_events']
                pid=c['patient_id'];visit=a['index_encounter_id'];at=a['index_encounter_start'];rc=c['provenance']['source_case_audit']['diagnosis_sources']
                mem=sum(len(v['records']) for v in a['past_visits']);cls=doctor.get('arrival_info',{}).get('encounter_class','UNKNOWN')
                fields={k:presence(doctor,k) for k in FIELDS};count_actual=0;source_fingerprint=digest(c)
            else:
                initial=next(r for r in initials if r['case_id']==c['case_id']);ev=[r for r in actions if r['case_id']==c['case_id']]
                pid=c['patient_id'];visit=c['visit_id'];at=c['decision_time'];rc=[r for r in refs if r['case_id']==c['case_id']]
                mem=len(c['memory_ids']);cls=initial['encounter_class'];fields={k:presence(initial,k) for k in FIELDS}
                count_actual=sum(r.get('actual_available_at') is not None for r in ev);source_fingerprint=digest(c)
            f={'dataset':dataset,'ordinal':index,'initial_presence':fields,'encounter_class':cls,'memory_count':mem,
                'ask_count':sum(e['action']=='Ask' for e in ev),'test_count':sum(e['action']=='Test' for e in ev),
                'consult_count':sum(e['action']=='Consult' for e in ev),'reference_candidate_count':len(rc),
                'actual_availability_count':count_actual,'development_exposed':True}
            public.append(classify_case(f));private.append({'target_ref':dataset+'%03d'%index,'patient_id':pid,'visit_id':visit,
                'decision_time':at,'source_fingerprint':source_fingerprint,'reference_source_links':rc,
                'ledger_refs':[r['ledger_id'] for r in ledger if r['case_id']==c['case_id']] if dataset=='D3' else 'SOURCE_EVENT_STORE_INTERNAL_LINK',
                'clinical_verdict':None,'purpose':'REVIEW_PLANNING_PRIVATE_LINKS_NOT_REVIEW_RESULTS'})
    data,options=actual_gate_inputs();priorities,private_priorities=temporal_priorities(original,cases,refs,data,options['decision_contexts'])
    outputs={'evaluation_scope_matrix':scope_matrix(),'case_suitability_matrix':{'schema_version':'locdi.c3_02.case_suitability.v1',
        'cases':public,'cohorts_merged':False,'clinical_exclusion_applied':False,'classification_only':True,
        'technical_rule_is_unapproved':True,'clinical_reference_approved_count':0},
        'reference_review_protocol':reference_protocol(),'review_axis_rubrics':rubrics(),
        'reviewer_workload_plan':workload(data,original,cases),'policy_decision_register':policy_register(),
        'temporal_review_priorities':priorities}
    source_paths=[p for p in (ROOT/'docs').rglob('*.md') if p.name.startswith(('C3_01_','035_','036_','D1_','D2_','D3_','D4_','D7_','024_'))]
    evidence={'C3_01_manifest_fingerprint':manifest['fingerprint'],'C3_01_matrices':{n:sha(base/(n+'.json')) for n in
        ('gap_ledger','initial_information_matrix','routing_group_analysis','review_contract_comparison','temporal_visibility_matrix','split_independence_matrix','baseline_dependency_matrix')},
        'D3_manifest_fingerprint':dm['manifest_fingerprint'],'source_documents':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in sorted(source_paths)],
        'privacy_scope':'Public ordinals/aggregates/status; source/identity/date links in separate new Private folder',
        'large_source_rescan':False,'new_model_or_replay_execution':False}
    return outputs,{'case_links':private,'priority_links':private_priorities},evidence
