"""C4-16 reference-only helpers. No router, utility fitting or clinical labels.

Silver admission is deliberately narrower than a recorded onset/domain match.
All present current-encounter conditions must have mapping and temporal proof.
Unknown domains are never treated as negative specialties.
"""
from copy import deepcopy
from collections import Counter
from app.locdi.llm_setup import digest
from app.core.longitudinal_memory import timestamp
from app.locdi.abstention_routing import clean

VERSION='locdi.source_coding_domain.reference.c4_16.v1'
RULE={
    'version':VERSION,'target':'SOURCE_CODE_DOMAIN_SILVER_NOT_CLINICAL_ROUTING',
    'membership':'Exact explicit System/Code and same patient/current Encounter only',
    'time':'Explicit source available_at <= Decision AND onset <= Decision; onset alone is insufficient',
    'multi_condition':'All current conditions must have verified exact mappings and timing; union accepted specialties',
    'unknown':'Any unresolved code/mapping/time/identity => ABSTAIN, never negative',
    'negative':'Only explicit verified mappings with no eligible candidate and a complete current scope; absence is not negative',
    'mapping':'Published code identity + official specialty scope are verified separately; composition is researcher-defined',
    'scope':'No clinical appropriateness, diagnosis correctness, abstention appropriateness or utility fitting',
}


def rule_spec():return {**deepcopy(RULE),'fingerprint':digest(RULE)}


def blind_packet(case, current_conditions, past_conditions):
    """Allowlisted first-stage export, never accepts router/model output."""
    for row in current_conditions+past_conditions:
        if timestamp(row['recorded_onset'])>timestamp(case['decision_time']):raise ValueError('FUTURE_CONDITION_IN_BLIND_PACKET')
        if timestamp(row['encounter_started_at'])>timestamp(case['decision_time']):raise ValueError('FUTURE_ENCOUNTER_IN_BLIND_PACKET')
        if row.get('actual_available_at') and timestamp(row['actual_available_at'])>timestamp(case['decision_time']):raise ValueError('FUTURE_RELEASE_IN_BLIND_PACKET')
    packet={'version':'locdi.routing_clinical_blind_packet.c4_16.v1','case_ref':case['case_ref'],
        'decision_id':case['decision_id'],'decision_time':case['decision_time'],
        'doctor_observable':clean(case['observation']),'observable_input_fingerprint':case['observable_input_fingerprint'],
        'candidate_specialties':sorted(case['candidate_departments']),'registry_fingerprint':case['registry_fingerprint'],
        'source_fingerprint':case['source_fingerprint'],'current_conditions_private_source_appendix':deepcopy(current_conditions),
        'past_conditions_recorded_by_decision_private_appendix':deepcopy(past_conditions),
        'chief_complaint_missing':not bool(case['observation'].get('chief_complaint')),
        'symptom_narrative_missing':not bool(case['observation'].get('current_symptoms')),
        'recorded_time_proxy':'UNREVIEWED_FOR_CLINICAL_REFERENCE','actual_clinical_release':'NOT_RECORDED',
        'source_appendix_warning':'Source coding is not doctor input or approved current clinical Gold. No future observations included.',
        'review_status':'NOT_REVIEWED'}
    assert_blind(packet)
    return packet


def assert_blind(packet):
    forbidden={'selected_specialty','actual_response','response','diagnosis','certainty','rationale',
        'predicted_diagnosis','policy_prediction','policy_execution_trace','keyword_scores','rank'}
    def visit(o):
        if isinstance(o,dict):
            if set(o)&forbidden:raise ValueError('PREDICTION_IN_BLIND_PACKET')
            for v in o.values():visit(v)
        elif isinstance(o,list):
            for v in o:visit(v)
    visit(packet)


def second_stage(cases,responses):
    known={c['case_ref']:c for c in cases};seen=set();out=[]
    for row in responses:
        key=(row['case_ref'],row['agent_id'])
        if key in seen or row['case_ref'] not in known:raise ValueError('DUPLICATE_OR_UNKNOWN_RESPONSE')
        seen.add(key)
        value=row['actual_response']
        if digest(value)!=row['response_fingerprint']:raise ValueError('RESPONSE_FINGERPRINT_MISMATCH')
        out.append({'case_ref':row['case_ref'],'agent_id':row['agent_id'],'response_fingerprint':row['response_fingerprint'],
            'actual_response':deepcopy(value),'observable_input_fingerprint':known[row['case_ref']]['observable_input_fingerprint'],
            'review_status':'NOT_REVIEWED','agent_abstention_appropriateness':None,'diagnosis_correctness':None,
            'reviewer':None,'reviewed_at':None})
    return {'version':'locdi.agent_abstention_second_stage.c4_16.v1','release_only_after_independent_stage1_review':True,
        'stage1_review_completed':False,'rows':out}


def time_state(fact,decision):
    try:
        d=timestamp(decision);onset=timestamp(fact['recorded_at'])
    except (ValueError,TypeError,KeyError):return 'MISSING_OR_INVALID_RECORDED_TIME'
    if onset>d:return 'FUTURE_ONSET'
    if not fact.get('available_at'):return 'ONSET_BY_DECISION_RELEASE_UNKNOWN'
    try:release=timestamp(fact['available_at'])
    except (ValueError,TypeError):return 'INVALID_AVAILABLE_TIME'
    if fact.get('availability_evidence')!='EXPLICIT_SOURCE_AVAILABLE_AT':return 'UNVERIFIED_AVAILABLE_TIME'
    if release<onset:return 'REVERSED_AVAILABLE_TIME'
    if release>d:return 'FUTURE_RELEASE'
    return 'EXPLICIT_SOURCE_TIME_ELIGIBLE'


def narrow_candidate(case,facts,mapping):
    """Admit only fully evidenced silver sets; retain all excluded/unknown cases."""
    entries={(r['system'],r['code']):r for r in mapping['mappings']}
    if len(entries)!=len(mapping['mappings']):raise ValueError('DUPLICATE_MAPPING')
    reasons=[];details=[];accepted=set();evidenced=set()
    if not facts:reasons.append('NO_CURRENT_CONDITION_FACT')
    for f in facts:
        if f.get('source_resource_type')!='Condition':reasons.append('NOT_CONDITION_RESOURCE')
        identity=(f['patient_reference']==case['patient_id'] and f['visit_reference']==case['visit_id'])
        if not identity:reasons.append('PATIENT_OR_ENCOUNTER_MISMATCH')
        if f.get('source_verified') is not True:reasons.append('SOURCE_POINTER_OR_HASH_NOT_VERIFIED')
        if digest(f['source_value'])!=f['source_value_fingerprint']:raise ValueError('SOURCE_VALUE_FINGERPRINT_MISMATCH')
        from app.locdi.source_enrichment import iso_ms
        try:
            if timestamp(iso_ms(f['source_value']['start']))!=timestamp(f['recorded_at']):reasons.append('SOURCE_RECORDED_ONSET_MISMATCH')
        except (KeyError,TypeError,ValueError):reasons.append('SOURCE_RECORDED_ONSET_MISMATCH')
        ts=time_state(f,case['decision_time'])
        if ts!='EXPLICIT_SOURCE_TIME_ELIGIBLE':reasons.append(ts)
        codes=f['source_value'].get('codes',[])
        condition_mapped=False;linked=[]
        for code in codes:
            m=entries.get((code.get('system'),str(code.get('code'))))
            if m is None:continue
            if m.get('code_identity_verified') is not True or m.get('specialty_scope_verified') is not True:
                reasons.append('MAPPING_EVIDENCE_INCOMPLETE');continue
            if not set(m['specialties'])<=set(case['candidate_departments']):
                reasons.append('MAPPING_OUTSIDE_CURRENT_CANDIDATES');continue
            condition_mapped=True;linked.append(m['mapping_id']);evidenced.update(m['specialties'])
            if ts=='EXPLICIT_SOURCE_TIME_ELIGIBLE' and identity and f.get('source_verified') is True:accepted.update(m['specialties'])
        if not condition_mapped:reasons.append('UNMAPPED_CURRENT_CONDITION')
        stop=f['source_value'].get('stop')
        if stop:
            if timestamp(iso_ms(stop))<=timestamp(case['decision_time']):reasons.append('CONDITION_ENDED_BY_DECISION')
        details.append({'fact_ref':f['fact_ref'],'source_value_fingerprint':f['source_value_fingerprint'],
            'time_state':ts,'identity_linked':identity,'mapping_ids':linked,'mapping_verified':condition_mapped})
    state='ABSTAIN' if reasons else 'POSITIVE' if accepted else 'NEGATIVE'
    return {'case_ref':case['case_ref'],'dataset':case['dataset'],'reference_kind':'SOURCE_GROUNDED_SILVER',
        'target':RULE['target'],'decision_time':case['decision_time'],'state':state,
        'accepted_specialties':None if reasons else sorted(accepted),'candidate_domains_with_mapping_evidence':sorted(evidenced),
        'blocker_codes':sorted(set(reasons)),'source_time_eligible':bool(facts) and all(x['time_state']=='EXPLICIT_SOURCE_TIME_ELIGIBLE' for x in details),
        'all_current_conditions_mapped':bool(facts) and all(x['mapping_verified'] for x in details),
        'mapping_evidence_present':any(x['mapping_verified'] for x in details),'fact_count':len(facts),'fact_checks':details,
        'clinical_reference':False,'clinical_correctness':None,'acceptable_abstention':None,
        'rule_fingerprint':rule_spec()['fingerprint'],'mapping_registry_fingerprint':digest(mapping)}


def summarize(candidates):
    n=len(candidates);valid=[r for r in candidates if r['state'] in ['POSITIVE','NEGATIVE']]
    return {'registered_cases':n,'source_time_eligible':sum(r['source_time_eligible'] for r in candidates),
        'cases_with_mapping_evidence':sum(r['mapping_evidence_present'] for r in candidates),
        'all_current_conditions_mapped':sum(r['all_current_conditions_mapped'] for r in candidates),
        'positive_label':sum(r['state']=='POSITIVE' for r in candidates),'negative_label':sum(r['state']=='NEGATIVE' for r in candidates),
        'abstain_unknown':sum(r['state']=='ABSTAIN' for r in candidates),
        'multiple_accepted_specialties':sum(len(r['accepted_specialties'] or [])>1 for r in valid),
        'no_eligible_specialty_verified':sum(r['state']=='NEGATIVE' for r in valid),
        'reference_coverage':{'numerator':len(valid),'denominator':n,'value':len(valid)/n if n else None,'status':'COMPUTED' if n else 'NOT_APPLICABLE'},
        'blocker_case_counts':dict(Counter(k for r in candidates for k in r['blocker_codes'])),
        'time_state_occurrences':dict(Counter(x['time_state'] for r in candidates for x in r['fact_checks']))}


def benchmark(cases,predictions,references):
    """Dedicated narrow target; never converts silver into clinical Reference."""
    if any(r.get('target')!=RULE['target'] or r.get('reference_kind')!='SOURCE_GROUNDED_SILVER' or r.get('clinical_reference') is not False for r in references):
        raise ValueError('SILVER_TARGET_ONLY')
    if len({r['case_ref'] for r in references})!=len(references):raise ValueError('DUPLICATE_REFERENCE')
    if not {r['case_ref'] for r in references}<=set(cases):raise ValueError('REFERENCE_OUTSIDE_REGISTERED_CASES')
    if not references:return {'status':'NOT_RUN_EMPTY_ADMITTED_REFERENCE','reference_coverage':{'numerator':0,'denominator':len(cases),'value':0 if cases else None},
        'source_grounded_agreement':None,'selective_accuracy_on_narrow_labels':None,'routing_coverage':None,
        'abstention_rate':None,'incorrect_selection_count':None,'clinical_metrics':'NOT_EVALUATED','router_calls':0}
    if any(r['state'] not in ['POSITIVE','NEGATIVE'] or r['blocker_codes'] or not r['source_time_eligible']
           or not r['all_current_conditions_mapped'] or r['rule_fingerprint']!=rule_spec()['fingerprint'] for r in references):raise ValueError('UNADMITTED_REFERENCE')
    if {p['case_ref'] for p in predictions}!={r['case_ref'] for r in references} or len(predictions)!=len(references):raise ValueError('PAIRED_PREDICTIONS_REQUIRED')
    selected=abstained=invalid=correct=wrong=0
    ref={r['case_ref']:r for r in references}
    for p in predictions:
        d=p['decision'];choice=d.get('selected_specialty')
        if d.get('decision_status')=='abstained' and choice is None:abstained+=1
        elif d.get('decision_status')=='selected' and choice in d.get('candidate_departments',[]):
            selected+=1
            if choice in ref[p['case_ref']]['accepted_specialties']:correct+=1
            else:wrong+=1
        else:invalid+=1
    n=len(references)
    return {'status':'NARROW_SILVER_ONLY','reference_coverage':{'numerator':n,'denominator':len(cases),'value':n/len(cases) if cases else None},
        'source_grounded_agreement':{'numerator':correct,'denominator':n,'value':correct/n},
        'selective_accuracy_on_narrow_labels':{'numerator':correct,'denominator':selected,'value':correct/selected if selected else None},
        'routing_coverage':{'numerator':selected,'denominator':n,'value':selected/n},'abstention_rate':{'numerator':abstained,'denominator':n,'value':abstained/n},
        'incorrect_selection_count':wrong,'invalid_count':invalid,'clinical_metrics':'NOT_EVALUATED',
        'multiple_specialty_acceptance':sum(len(r['accepted_specialties'])>1 for r in references)}
