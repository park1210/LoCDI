"""C3-05 descriptive E1 only; no causal labels, external matching or generation."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json

from jsonschema import Draft202012Validator
from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_freeze import load, sha
from app.locdi.unresolved_analysis import response_state, name_mapping, condition_link
from app.locdi.routing_policy_analysis import request_budget
from app.locdi.reference_source_survey import inventory


def fraction(numerator, denominator):
    return {'numerator':numerator,'denominator':denominator,'value':numerator/denominator if denominator else None,
            'status':'DESCRIPTIVE_ONLY' if denominator else 'NOT_APPLICABLE'}


def mapping_guard(match_kind, requested_type='SOURCE_GROUNDED_REFERENCE'):
    if match_kind not in ('exact_code','normalized_terminology','ontology_candidate','clinical_semantic_equivalence','current_decision_timing'):
        raise ValueError('UNKNOWN_MAPPING_KIND')
    blocked=requested_type=='INDEPENDENT_CLINICAL_REFERENCE' or match_kind in ('clinical_semantic_equivalence','current_decision_timing')
    return {'match_kind':match_kind,'status':'REVIEW_REQUIRED' if blocked else 'CANDIDATE_ONLY',
            'clinical_equivalence':None,'clinical_gold_approved':False,'correctness':None,
            'reason':'Code/term relations do not authenticate patient context, clinical equivalence, timing or human review'}


def exam_labels_usable(dataset, split, label_status):
    if dataset=='MedMCQA' and split=='test':
        return {'status':'BLOCKED','reason':'OFFICIAL_TEST_GROUND_TRUTH_WITHHELD','clinical_gold_approved':False}
    return {'status':'DOCUMENTED_LABEL_CANDIDATE' if label_status=='DOCUMENTED_PUBLIC' else 'NOT_CHECKED',
            'reason':'Must verify exact acquired revision/index/schema and approved separate exam task before use',
            'clinical_gold_approved':False}


def analyze_responses(bundle, references, records, sources, prompts):
    """Private records in, anonymous counts out. All causes remain hypotheses."""
    before=digest([bundle,references,records,sources,prompts])
    expected={a['agent_id']:a for a in bundle['registry']['agents'] if a['role']=='specialty'}
    if len(expected)!=5 or len(records)!=len(bundle['candidates'])*5:
        raise ValueError('COMPLETE_RECORDED_SCOPE_REQUIRED')
    source_map={s['case_id']:s for s in sources}; refs={r['case_ref']:r for r in references}
    grouped=defaultdict(list)
    for r in records:grouped[r['case_ref']].append(r)
    schema=Draft202012Validator(bundle['response_schema'])
    specialty={a['department']:Counter() for a in expected.values()}
    group_rows=[]; private=[]; all_flags=Counter(); unresolved_flags=Counter(); class_matrix={}; stratum=Counter()
    timestamps=Counter(); fingerprints=[]; condition_days=[]; bindings=Counter()
    for index,c in enumerate(bundle['candidates'],1):
        d=c['decision']; obs=d['observation']; rows=grouped[c['case_ref']]
        if len(rows)!=5 or {r['agent_id'] for r in rows}!=set(expected):
            raise ValueError('MISSING_OR_DUPLICATE_AGENT')
        if any(r.get('correct') is not None for r in rows):raise ValueError('UNREVIEWED_BASELINE_CHANGED')
        source=source_map.get(d['source_ref'])
        if (source is None or source['patient_id']!=d['patient_id'] or
                source['index_encounter']['encounter_id']!=d['visit_id'] or
                source['index_encounter']['start_time']!=c['visit_started_at']):
            raise ValueError('SOURCE_IDENTITY_MISMATCH')
        encounter_class=source['index_encounter'].get('encounter_class','UNKNOWN')
        reference=refs[c['case_ref']]; mapping=name_mapping(reference['declared_label'],bundle['catalog'])
        links=condition_link(c,reference); delta=max((r['start_day_offset'] for r in links),default=0);condition_days.append(delta)
        objective=len(obs.get('objective_findings',[])); memory=obs.get('longitudinal_memory',{}).get('records',[])
        shared=digest(c['shared_payload'])
        group={'decision_ref':'Candidate%03d'%index,'split':d['split'],'encounter_class':encounter_class,
            'encounter_class_scope':'SOURCE_METADATA_NOT_IN_AGENT_SNAPSHOT',
            'shared_input_fingerprint':shared,'memory_fingerprint':digest(obs.get('longitudinal_memory',{})),
            'chief_complaint_present':bool(obs.get('chief_complaint')),'symptom_count':len(obs.get('current_symptoms',[])),
            'history_count':len(obs.get('history',[])),'objective_count':objective,'memory_count':len(memory),
            'input_description':'SPARSE_RECORDED_CONTEXT_CLINICAL_SUFFICIENCY_NOT_REVIEWED',
            'source_label_exact_text_present':mapping['text_status']=='EXACT_TEXT_PRESENT',
            'source_condition_positive_date_offset':delta>0,'maximum_source_condition_day_offset':delta,
            'reference_timing_status':'REFERENCE_TIMING_NOT_REVIEWED',
            'excluded_selected_future_source_events':c.get('excluded_selected_future_events','UNKNOWN'),
            'supported':0,'unresolved':0,'named_unresolved':0,'literal_unresolved':0,'agent_states':[],
            'schema_invalid':0,'failed':0,'same_common_input':True,'clinical_correctness':'NOT_REVIEWED'}
        class_matrix.setdefault(encounter_class,Counter())
        for row in rows:
            a=expected[row['agent_id']]; answer=row['response']; state,_=response_state(row)
            valid=schema.is_valid(answer); unresolved=state=='UNRESOLVED'
            group['unresolved' if unresolved else 'supported']+=1
            group['schema_invalid']+=not valid;group['failed']+=row['execution_status']!='SUCCEEDED'
            named=unresolved and answer['diagnosis']!='Unresolved'
            group['named_unresolved']+=named;group['literal_unresolved']+=unresolved and not named
            specialty[a['department']]['unresolved' if unresolved else 'supported']+=1
            specialty[a['department']]['named_unresolved']+=named
            specialty[a['department']][d['split']+'_supported']+=not unresolved
            class_matrix[encounter_class]['unresolved' if unresolved else 'supported']+=1
            stratum[('same_source_stratum_' if a['department']==c['stratum'] else 'other_source_stratum_')+('unresolved' if unresolved else 'supported')]+=1
            flags=[]; text=answer['rationale'].casefold()
            if any(t in text for t in ('insufficient','lack','not provided','not specified','not mentioned','limited','additional','further')):
                flags.append('LEXICAL_INSUFFICIENCY_OR_CONFIRMATION_MENTION')
            # Lexical exploration only; not a clinical hallucination judgment.
            visible=json.dumps(c['shared_payload'],ensure_ascii=False).casefold()
            for term in ('ecg','troponin'):
                if term in text and term not in visible:flags.append('RATIONALE_'+term.upper()+'_NOT_IN_SNAPSHOT_TEXT')
            all_flags.update(flags)
            if unresolved:unresolved_flags.update(flags)
            matches=(all(row.get(k)==a[k] for k in ('model','version','department','role')) and
                row['system_fingerprint']==d['generation']['prompt_fingerprints'][a['agent_id']] and
                row['observed_model_identifier']==d['generation']['model_identifier'] and row['generation']==d['generation'])
            bindings['binding_match' if matches else 'binding_mismatch']+=1
            group['same_common_input'] &= row['shared_input_fingerprint']==shared==c['shared_input_fingerprint']
            try:
                at=datetime.fromisoformat(row['recorded_at'].replace('Z','+00:00'))
                start=datetime.fromisoformat(row['request_started_at'].replace('Z','+00:00'))
                timestamps['timezone_aware']+=at.tzinfo is not None and start.tzinfo is not None
                timestamps['recorded_after_request']+=at>=start
            except (KeyError,ValueError,TypeError,AttributeError):timestamps['unknown_or_invalid']+=1
            fingerprints.append(row['response_fingerprint'])
            if row['response_fingerprint']!=digest(answer):raise ValueError('ORIGINAL_RESPONSE_HASH_MISMATCH')
            group['agent_states'].append({'agent_ref':row['agent_ref'],'specialty':a['department'],'certainty':answer['certainty'],
                'response_form':'NAMED_UNRESOLVED' if named else 'LITERAL_UNRESOLVED' if unresolved else 'SUPPORTED_NAMED',
                'response_fingerprint':row['response_fingerprint'],'rationale_flags':flags})
            private.append({'case_ref':c['case_ref'],'response_ref':row['response_ref'],'patient_id':d['patient_id'],
                'visit_id':d['visit_id'],'decision_id':d['decision_id'],'agent_id':row['agent_id'],
                'selected_diagnosis':answer['diagnosis'],'rationale_fingerprint':digest(answer['rationale']),
                'recorded_at':row.get('recorded_at'),'request_started_at':row.get('request_started_at'),
                'source_ref':d['source_ref'],'reference_candidate_fingerprint':digest(reference),
                'correctness':None,'review_status':'NOT_REVIEWED'})
        group_rows.append(group)
    if not all(r['same_common_input'] and not r['schema_invalid'] and not r['failed'] for r in group_rows) or bindings['binding_mismatch']:
        raise ValueError('COLLECTION_CONTRACT_MISMATCH')
    total=len(records);u=sum(r['unresolved'] for r in group_rows);named=sum(r['named_unresolved'] for r in group_rows)
    literal=sum(r['literal_unresolved'] for r in group_rows)
    public={'schema_version':'locdi.c3_05.unresolved_audit.v1','run_id':'phase24b_collect_001',
        'preparation':'phase24b_candidates_v2','preparation_fingerprint':bundle['fingerprint'],
        'response_count':total,'decision_count':len(group_rows),'supported':total-u,'unresolved':u,
        'named_unresolved':named,'literal_unresolved':literal,'named_among_unresolved':fraction(named,u),
        'unresolved_rate':fraction(u,total),'supported_coverage':fraction(total-u,total),
        'independently_reviewed_correctness':0,'supported_only_complete_groups':sum(g['unresolved']==0 for g in group_rows),
        'gate_accepted_fit_groups':0,'clinical_metrics':'NOT_EVALUATED','selective_accuracy':'BLOCKED_NO_INDEPENDENT_REFERENCE',
        'groups':group_rows,'response_set_fingerprint':digest(sorted(fingerprints)),
        'schema_invalid':sum(g['schema_invalid'] for g in group_rows),'failed':sum(g['failed'] for g in group_rows),
        'timestamp_checks':dict(timestamps),'binding_checks':dict(bindings),
        'one_base_model':len({a['model'] for a in expected.values()})==1,
        'registered_model_versions':[{'specialty':a['department'],'model':a['model'],'version':a['version']} for a in expected.values()],
        'prompt_fingerprints':{d:hashlib.sha256(t.encode('utf8')).hexdigest() for d,t in prompts.items()},
        'prompt_policy_flags':{d:{'mentions_unresolved':'unresolved' in t.casefold(),
            'mentions_supported':'supported' in t.casefold(),'mentions_evidence':'evidence' in t.casefold()} for d,t in prompts.items()},
        'action_schema':'diagnose_only; supported/unresolved; fixed common diagnosis catalog',
        'rationale_flags_all_responses':dict(all_flags),'rationale_flags_unresolved':dict(unresolved_flags),
        'rationale_interpretation':'LEXICAL_FLAGS_NOT_CAUSAL_PROOF_OR_CLINICAL_ERROR_LABELS',
        'source_label_literal_coverage':sum(g['source_label_exact_text_present'] for g in group_rows),
        'future_condition_decisions':sum(v>0 for v in condition_days),
        'causal_assignment':'NOT_DETERMINED_FOR_ALL_UNRESOLVED',
        'reference_absence_interpretation':'Assessment barrier for75, not proof of an Agent-input cause; Gold must stay outside Agent input'}
    matrix={'schema_version':'locdi.c3_05.specialty_matrix.v1',
        'specialties':{d:{**dict(v),'named_among_unresolved':fraction(v['named_unresolved'],v['unresolved']),
            'unresolved_rate':fraction(v['unresolved'],v['unresolved']+v['supported'])} for d,v in specialty.items()},
        'encounter_classes':{k:dict(v) for k,v in class_matrix.items()},'source_stratum_comparison':dict(stratum),
        'stratum_warning':'Existing Source Label taxonomy is NOT independently reviewed appropriate-specialty reference',
        'input_groups':[{'decision_ref':g['decision_ref'],'shared_input_fingerprint':g['shared_input_fingerprint'],
            'memory_fingerprint':g['memory_fingerprint'],'supported':g['supported'],'unresolved':g['unresolved']} for g in group_rows],
        'full_payloads':'Common Observation/Memory/Catalog same; department and system prompt differ intentionally',
        'clinical_reference_count':0,'clinical_correctness':None}
    assert digest([bundle,references,records,sources,prompts])==before
    return public,matrix,private


def hypotheses(audit,matrix):
    groups=audit['groups'];n=len(groups);u=audit['unresolved']
    missing=sum(not g['chief_complaint_present'] or not g['symptom_count'] for g in groups)
    zero=sum(g['objective_count']==0 for g in groups)
    lexical=audit['rationale_flags_unresolved'].get('LEXICAL_INSUFFICIENCY_OR_CONFIRMATION_MENTION',0)
    specifications=[
      ('H1','Initial Clinical Context Insufficiency',{'sparse_decisions':missing,'no_current_objective_decisions':zero},
       'Supported responses also occur in the same sparse snapshots; clinical sufficiency has not been reviewed',
       'Contemporaneous clinical task, sufficiency review, source-grounded richer input; do not invent findings',False),
      ('H2','Specialty-Case Mismatch',matrix['source_stratum_comparison'],
       'Source stratum is not a clinical appropriate-specialty reference; prompt/model/context confounded',
       'Independent acceptable-specialty/contribution reference at identical snapshot',False),
      ('H3','Model/Prompt Certainty Policy',{'one_base_model':audit['one_base_model'],'prompt_fingerprints':audit['prompt_fingerprints'],'action_schema':audit['action_schema']},
       'Both certainty states are permitted and observed; no prompt-only paired contrast',
       'Approved paired prompt/certainty comparison with fixed model/source/catalog and independent reference',True),
      ('H4','Diagnosis Catalog or Label Mapping',{'exact_source_label_coverage':audit['source_label_literal_coverage'],'decisions':n,
        'literal_missing_target_claim':'CONTRADICTED_BY_OBSERVATION' if audit['source_label_literal_coverage']==n else 'NOT_DETERMINED'},
       'All declared source targets are literally present, but synonyms/clinical granularity/current diagnosis validity remain unreviewed',
       'Code-system-aware terminology candidates and independent clinical/catalog scope review',True),
      ('H5','Incomplete Evidence at Decision Time',{'no_current_objective_decisions':zero,'future_condition_decisions':audit['future_condition_decisions'],
        'excluded_future_source_events':sum(g['excluded_selected_future_source_events'] for g in groups if type(g['excluded_selected_future_source_events']) is int)},
       'Future withholding is a required information boundary, not an implementation defect; missing clinical evidence not fully known',
       'Actual release/visibility records and approved information sufficiency at cutoff',True),
      ('H6','Appropriate Clinical Abstention',{'unresolved':u,'lexical_confirmation_mentions_in_unresolved':lexical},
       'Rationale is self-report, not independent appropriateness evidence; named52/literal1 not a correctness verdict',
       'Independent reference and explicit abstention appropriateness policy/clinical review',False),
      ('H7','Model Reasoning or Capability Limitation',{'one_base_model':audit['one_base_model'],'supported':audit['supported'],'unresolved':u},
       'No alternate-model control; unresolved is not an incorrect reasoning label;22 supported are also unreviewed',
       'Approved model comparison, source/reference sufficiency, independent outcome grading',False)]
    return {'schema_version':'locdi.c3_05.hypotheses.v1','hypotheses':[
        {'hypothesis_id':h,'hypothesis':name,'observed_evidence':observed,'counter_evidence':counter,
         'affected_count':u,'affected_count_unit':'UNRESOLVED_REVIEW_SCOPE_NOT_CAUSALLY_ATTRIBUTED_NONEXCLUSIVE',
         'evidence_level':'DESCRIPTIVE_SNAPSHOT_AND_LEXICAL_ONLY','alternative_explanations':['Other H1-H7 hypotheses','Reference and timing uncertainty','Single-run confounding'],
         'additional_data_needed':need,'testable_without_generation':True,
         'without_generation_scope':'Descriptive evidence/coverage review only, not causal confirmation',
         'needs_reference':True,'requires_controlled_experiment':h in ('H1','H2','H3','H7'),
         'structural_subquestion_only':partial,'final_status':'NOT_DETERMINED'}
        for h,name,observed,counter,need,partial in specifications],
        'causal_identification':'NOT_ESTABLISHED','reference_missing_is_not_agent_gold_input_requirement':True}


def compatibility(survey):
    statuses={
      'Synthea_existing':['STRUCTURAL_ONLY','REVIEW_REQUIRED','REVIEW_REQUIRED','STRUCTURAL_ONLY','STRUCTURAL_ONLY','REVIEW_REQUIRED'],
      'MedQA':['NOT_EVALUABLE','REVIEW_REQUIRED','NOT_EVALUABLE','ADAPTER_REQUIRED','NOT_EVALUABLE','NOT_EVALUABLE'],
      'MedMCQA':['NOT_EVALUABLE','REVIEW_REQUIRED','NOT_EVALUABLE','ADAPTER_REQUIRED','NOT_EVALUABLE','NOT_EVALUABLE'],
      'MedCaseReasoning':['NOT_EVALUABLE','REVIEW_REQUIRED','NOT_EVALUABLE','REVIEW_REQUIRED','NOT_EVALUABLE','NOT_EVALUABLE'],
      'HealthBench':['NOT_EVALUABLE']*6,
      'PMC_Open_Access_case_reports':['REVIEW_REQUIRED','REVIEW_REQUIRED','NOT_EVALUABLE','REVIEW_REQUIRED','REVIEW_REQUIRED','REVIEW_REQUIRED'],
      'Independent_current_decision_reference':['UNKNOWN']*6}
    return {'schema_version':'locdi.c3_05.compatibility.v1','columns':['Memory','Routing','VoI','Diagnosis','Action','Timing'],
        'rows':[{'dataset':r['dataset'],'reference_type':r['reference_type'],
            'functions':dict(zip(('Memory','Routing','VoI','Diagnosis','Action','Timing'),statuses[r['dataset']])),
            'intended_scope':'Separate exam QA only' if r['reference_type']=='EXAM_GOLD' else 'Separate rubric response-quality task only' if r['reference_type']=='RUBRIC_REFERENCE' else 'Candidate/reference-type specific review',
            'execution_ready':False,'existing75_correctness_transfer_allowed':False} for r in survey['sources']],
        'scope_note':'Diagnosis ADAPTER_REQUIRED for exam sources means exam-answer task, not clinical patient diagnosis. HealthBench separate rubric task is not any internal clinical metric.',
        'minimum_human_review':'Exam option scoring can avoid per-response human judgments after label/index verification; case semantic grading, timing/routing/clinical-transfer claims still require separate reviewed policy/reference'}


def experiments():
    rows=[]
    specifications=[
      ('E1','Existing Response Root-Cause Audit','None; stored75 only','H1-H7 descriptive evidence, no causal identification',0,5,1,1),
      ('E2','Initial Context Sufficiency Comparison','Two source-grounded context variants','H1/H5 under identical model/prompt/catalog; richer is not automatically better',15,5,2,1),
      ('E3','Prompt/Certainty Policy Comparison','Two versioned prompt policies','H3; same context, model, catalog and action schema',15,5,2,1),
      ('E4','Specialty Appropriateness Comparison','Five agents at identical approved input; reference-defined strata','H2; source taxonomy is not Gold department',15,5,1,1),
      ('E5','Reference-Grounded Selective Accuracy','Separate approved external/reference cohort, example single-model study','Coverage plus reviewed selective accuracy; no Synthea Gold transfer',30,1,1,1)]
    for eid,name,var,hyp,n,a,v,r in specifications:
        budget=request_budget(n,a,r);budget['variant_count']=v
        budget['initial_generation_requests']*=v;budget['maximum_generation_requests']*=v;budget['identity_checks_upper_bound']*=v
        rows.append({'experiment_id':eid,'name':name,'status':'COMPLETED_READ_ONLY' if eid=='E1' else 'PROPOSED_NOT_APPROVED',
            'changed_variable':var,'fixed_variables':['Patient/decision and release cutoff within paired arms','Source and memory snapshot except E2 approved changed component',
                'Model/version/digest, decoding/seed/repeats','Agent registry, common action and diagnosis candidate sets except explicitly approved new task'],
            'hypotheses':hyp,'reference_required':'None for descriptive E1; current independent reference and axis-specific criteria for clinical claims',
            'metrics':['Supported/unresolved coverage and failure rates','Independent-reference selective accuracy only if nonempty evaluable denominator','Abstention appropriateness separately'],
            'leakage_prevention':['Gold/reference/answer explanation/article title/full text/future source excluded from input','Patient/source-article dedup and fixed split','Freeze arms before responses; preserve all failures and original75'],
            'budget':budget,'grader_requests':'0 for E1; UNKNOWN_UNTIL_APPROVED_GRADING_POLICY' if eid!='E1' else 0,
            'interpretation':'Unresolved reduction alone not improvement; selected accuracy and coverage/exclusions required',
            'prerequisites':['Policy/scope/rights approval','Verified current source/input sufficiency and independent reference','Bounded generation+grading+retry+repair plan','Independent split and exact bindings'] if eid!='E1' else ['Pinned immutable original75'],
            'actual_generation_calls':0})
    return {'schema_version':'locdi.c3_05.experiment_plan.v1','experiments':rows,
        'illustrative_E2_E5_generation_total':sum(r['budget']['maximum_generation_requests'] for r in rows[1:]),
        'total_all_requests':'UNKNOWN_UNTIL_GRADER_BUDGET_FIXED; current approved total0',
        'retry_repair_plan':'Examples use total retry0/repair0 per task. Any change requires new cap including every attempt/identity check; no infinite retry',
        'cost_currency':'UNKNOWN_LOCAL_RUNTIME_AND_REVIEWER_COST_NOT_PROVIDED','reviewer_throughput':'UNKNOWN',
        'clinical_metrics_for_actual75':'NOT_EVALUATED'}


def schedule():
    entries=[
      ('2026-11-06','Structural Development','GO_DEVELOPMENT_ONLY','Existing structural RC/live static evidence; new offline regression/preservation required',
       'Source-backed Consult, official evaluation, clinical artifacts not implied','Software workflow/replay scope'),
      ('2026-11-15','Reference Acquisition & Research Scope','NO_GO_CURRENT_CLINICAL_SCOPE','Metadata/reference candidates found; clinical approvals0',
       'Actual reviewers/authority, rights reconciliation, source timing/sufficiency, policy approval','Freeze remains development-only; approve separate exam/rubric scope if suitable'),
      ('2026-11-29','Stage3/4 Integration','NO_GO_CURRENT_CLINICAL_SCOPE','TEST_ONLY Bridge/handoff integration exists',
       'Operational trust/evidence admission, current references, auxiliary adapters and source-backed action expansion','Contract-tested software integration only'),
      ('2026-12-10','Official B0-B4 Research Evaluation','NO_GO_CURRENTLY','Fit groups0; reviewed Utility/VoI absent; official runner/metrics blocked',
       'Authenticated current Gold, complete fit/support groups, independent combined validation, approved artifacts/model/time/repeat bindings and official runner/metrics','Separately approved software/uncertainty/exam/rubric study; never rename as official ablation'),
      ('2026-12-18','Thesis/Presentation Completion','CONDITIONAL_SCOPE_APPROVAL_REQUIRED','Development evidence/reporting available; clinical results absent',
       'Supervisor/research-owner scope and claim approval, manuscript completion/reproducibility; human availability UNKNOWN','Software structural verification and limitations if research owner approves')]
    return {'schema_version':'locdi.c3_05.schedule.v1','as_of':'2026-11-01','timezone':'Asia/Seoul',
        'milestones':[{'date':day,'target':name,'decision':state,'actual_completed':done,'remaining_blocker':block,
            'required_approval':'Research owner; institutional/clinical authority and rights approval where applicable',
            'external_dependency':'Reviewer/authority/source access/throughput UNKNOWN',
            'additional_development':'Evidence enrollment/authenticator, auxiliary adapters, independent validation and official runner/metrics where applicable',
            'fallback':fallback,'new_ready_claim':False} for day,name,state,done,block,fallback in entries],
        'clinical_go_conditions':['Authenticated current reference and reviewer approval','Approved routing policy/task/scope',
            'All-five supported reviewed complete train/validation groups with query support','Non-toy clinically reviewed VoI artifact',
            'Patient-disjoint combined validation and exposure-aware final paired cases','Official runner/metrics and fixed common bindings','Bounded explicit execution approval'],
        'deadline_success_prediction':'UNKNOWN_NOT_ESTIMATED_FROM_HYPOTHETICAL_REVIEW_THROUGHPUT'}


def collection_options():
    return {'schema_version':'locdi.c3_05.collection_options.v1','options':[
      {'option':'A','source':'Original50/D329 retained','available':'Frozen replay and raw links','additional_source':'None for software tests',
       'without_reviewer':'Structural source/clock/cutoff/export checks','minimum_human':'Scope approval; clinical reference needed only for clinical claims',
       'timing':'Proxy, not clinical release','model_prompt':'Existing bindings preserved','A_D_B1':'A/B1 development-compatible; no fit groups',
       'B0_B4_direct':'NO_CLINICAL_EVALUATION','deadline_risk':'Low for software; clinical claims blocked'},
      {'option':'B','source':'Separate public case/exam/rubric dataset','available':'Metadata only for new sources; HealthBench5000 already local',
       'additional_source':'Approved pinned acquisition/version/license verification','without_reviewer':'Exam option scoring or source/schema tests with correct labels',
       'minimum_human':'Task/rights/label mapping and grading-method validation; no automatic current-patient Gold',
       'timing':'No interactive release/longitudinal ledger in reviewed benchmark schemas','model_prompt':'New adapter/task versions; answer/exp/reference hidden',
       'A_D_B1':'Separate task; not a replacement for A training labels','B0_B4_direct':'NO','deadline_risk':'Conditional on rights/grading/approval, not guaranteed'},
      {'option':'C','source':'Existing bank173/229','available':'Source/Condition candidates,15 selected,75 responses',
       'additional_source':'Current-decision reference and sufficiency/timing evidence','without_reviewer':'Source/code/duplicate/split coverage',
       'minimum_human':'15 decision references + supported22 and abstention53 separate reviews; reviewer auth',
       'timing':'Unreviewed recorded upper bounds/proxies','model_prompt':'Do not expose Condition/label to Agents',
       'A_D_B1':'Reference review only cannot create current supported-complete groups','B0_B4_direct':'BLOCKED','deadline_risk':'High/UNKNOWN'},
      {'option':'D','source':'New richer Synthea version','available':'Existing source pool candidates, no new data created',
       'additional_source':'Source-grounded initial context/time and new independent split','without_reviewer':'Generator/source consistency only',
       'minimum_human':'Clinical task/sufficiency/current reference and appropriateness approval remain necessary',
       'timing':'Synthetic timestamps are not validated real clinical information release','model_prompt':'New shared catalog/snapshot/prompt version if changed',
       'A_D_B1':'Potential D collection under unchanged A, yield UNKNOWN','B0_B4_direct':'NO_UNTIL_FULL_APPROVAL','deadline_risk':'High, no promised yield'},
      {'option':'E','source':'Approved external clinical material','available':'NOT_ACQUIRED_THIS_PHASE',
       'additional_source':'Lawful access/DUA, contemporaneous source and governance','without_reviewer':'Metadata/schema only after authorized access',
       'minimum_human':'Data owner, independent clinical reviewer and authority approval',
       'timing':'Must be evidenced, never inferred from narrative publication dates','model_prompt':'Private data transport/model-use approval required',
       'A_D_B1':'Possible separate versioned D design, not assumed','B0_B4_direct':'NO','deadline_risk':'UNKNOWN_EXTERNAL_DEPENDENCIES'}],
       'new_collection_executed':0,'new_reference_approved':0,'policy_approved':False}


def build():
    from app.locdi.routing_collector import load_preparation, PRIVATE_ROOT
    from app.locdi.routing_data_reuse import SOURCE,read
    from app.locdi.routing_policy_analysis import build as policy_build,verify_manifest
    from app.llm.prompt_loader import specialty_system_prompt
    verify_manifest(ROOT/'outputs/research_evaluation/c3_04_v1/completion_manifest.json')
    base,links,evidence=policy_build()
    bundle=load_preparation('phase24b_candidates_v2')
    references=load(PRIVATE_ROOT/'phase24b_candidates_v2/reference_candidates.json')
    records=[load(p) for p in sorted((PRIVATE_ROOT/'phase24b_collect_001').glob('Candidate*.json'))]
    prompts={a['department']:specialty_system_prompt(a['department']) for a in bundle['registry']['agents'] if a['role']=='specialty'}
    audit,matrix,private=analyze_responses(bundle,references,records,read(SOURCE),prompts)
    if (audit['response_count'],audit['unresolved'],audit['named_unresolved'],audit['source_label_literal_coverage'])!=(75,53,52,15):
        raise ValueError('ACTUAL_BASELINE_CHANGED')
    # Reuse local pinned HealthBench; hash every receipt file without copying raw.
    hb_path=next((ROOT/'datasets/external/healthbench').rglob('dataset_manifest_canonical_v2.json'))
    hb=load(hb_path)
    if digest({k:v for k,v in hb.items() if k!='manifest_fingerprint'})!=hb['manifest_fingerprint']:
        raise ValueError('HEALTHBENCH_MANIFEST_CHANGED')
    if any(sha(ROOT/r['path'])!=r['sha256'] for r in hb['files'].values()):raise ValueError('HEALTHBENCH_RAW_CHANGED')
    survey=inventory(); health=next(r for r in survey['sources'] if r['dataset']=='HealthBench')
    health['local_manifest_fingerprint']=hb['manifest_fingerprint'];health['raw_hash_recheck']='PASS'
    audit['existing_correctness_gate']=base['existing_response_groups']['gate']
    audit['review_contract']='Existing C3-04 production authority NOT_CONFIGURED; approved policy null; no admission generated'
    matrix['policy_impact']=base['policy_comparison']
    matrix['supported_candidate_upper_bounds']=base['existing_response_groups']['supported_review_candidates_per_agent_split']
    matrix['split_independence']=base['split_independence_analysis']
    options=collection_options(); options['routing_policy_approval_register']=base['policy_approval_register']
    mapping={'schema_version':'locdi.c3_05.mapping_risks.v1','levels':[mapping_guard(k) for k in
        ('exact_code','normalized_terminology','ontology_candidate','clinical_semantic_equivalence','current_decision_timing')],
        'promotion_attempt':mapping_guard('exact_code','INDEPENDENT_CLINICAL_REFERENCE'),
        'reference_types':['SOURCE_GROUNDED_REFERENCE','EXAM_GOLD','RUBRIC_REFERENCE','CLINICAL_CASE_REFERENCE_CANDIDATE','INDEPENDENT_CLINICAL_REFERENCE'],
        'automatic_gold_or_correctness_written':0,'external_to_Synthea_patient_matching_performed':False,
        'MedMCQA_test_label_contract':exam_labels_usable('MedMCQA','test','DOCUMENTED_PUBLIC')}
    evidence['healthbench_manifest_sha256']=sha(hb_path)
    evidence['source_bank_reference_sha256']=sha(SOURCE)
    documents=[p for p in (ROOT/'docs').rglob('*.md') if p.name.startswith(('C3_01_','C3_02_','C3_03_','C3_04_',
        'D1_','D2_','D3_','D4_','D6_','D7_','022_','023_','024_','035_','036_'))]
    evidence['documents']={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(documents)}
    links.update(schema_version='locdi.private.c3_05_source_links.v1',purpose='READ_ONLY_AUDIT_NOT_REVIEW_RESULTS',response_links=private)
    return {'unresolved_evidence_audit':audit,'specialty_uncertainty_matrix':matrix,'root_cause_hypotheses':hypotheses(audit,matrix),
        'reference_source_inventory':survey,'reference_compatibility_matrix':compatibility(survey),'automatic_mapping_risks':mapping,
        'collection_and_experiment_options':options,'e1_e5_experiment_plan':experiments(),'schedule_go_no_go':schedule()},links,evidence
