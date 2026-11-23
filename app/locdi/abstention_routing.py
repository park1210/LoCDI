"""C4-15 development routing and independent-reference metrics, outside Core.

No learned probabilities, fitting, LLM, diagnosis grading, or runtime injection.
Policy choices below are exposed-data development assumptions, not clinical rules.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re

from app.core.routing_utility import observed_tokens
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest
from app.locdi.routing_evaluation import observable, split_check

VERSION = 'locdi.abstention_routing.development.v1'
REFERENCE = 'locdi.routing_suitability_router_abstention.reference.v1'
SCOPE = 'ROUTING_SUITABILITY_AND_ROUTER_ABSTENTION'
SPEC = {
    'version': VERSION, 'target': 'routing_suitability_not_diagnosis_correctness',
    'scope': 'DEVELOPMENT_ONLY_NOT_LIVE_RUNTIME',
    'primary_support': 'Exact configured keyword token-set inclusion in current chief complaint, symptoms, objective findings',
    'context_support': 'Same keyword check in public history and non-superseded memory; context alone cannot select',
    'selection': 'Select iff exactly one candidate has positive current support; otherwise abstain',
    'multiple_current_candidates': 'ABSTAIN_AMBIGUOUS_SUPPORT_NO_ARBITRARY_TIEBREAK',
    'no_current_candidate': 'ABSTAIN_NO_CURRENT_SUPPORT',
    'empty_keywords': 'NO_CONFIGURED_KEYWORD_SUPPORT_NOT_CLINICAL_UNSUITABILITY',
    'numeric_thresholds': None, 'learned_success_probabilities': None, 'utility_weights': None,
    'tuning': 'No threshold search; defined on development-exposed data, requires separate validation',
    'clinical_validity': 'NOT_VALIDATED', 'seed_used': False,
}


def policy_spec():
    return {**deepcopy(SPEC), 'fingerprint': digest(SPEC)}


def clean(observation):
    obs = observable(observation)
    forbidden = {'acceptable_abstention', 'permitted_diagnoses', 'reviewer_rationale',
                 'reference_scope', 'source_condition', 'source_conditions', 'response_correctness'}
    def inspect(o):
        if isinstance(o, dict):
            if set(o) & forbidden: raise ValueError('REFERENCE_IN_POLICY_INPUT')
            for v in o.values(): inspect(v)
        elif isinstance(o, list):
            for v in o: inspect(v)
        elif isinstance(o, str) and o.lstrip().startswith(('{','[')):
            try: v = json.loads(o)
            except ValueError: return
            inspect(v)
    inspect(obs)
    return obs


def candidates_checked(registry, candidates):
    bound = {a['department'] for a in registry.snapshot()['agents'] if a['role']=='specialty'}
    if (not isinstance(candidates,list) or not candidates or any(not isinstance(x,str) for x in candidates)
            or len(set(candidates)) != len(candidates) or not set(candidates) <= bound):
        raise ValueError('CANDIDATE_SET_INVALID')
    return sorted(candidates)


def select(observation, registry, candidates, *, expected_registry_fingerprint):
    obs = clean(observation)
    if digest(registry.snapshot()) != expected_registry_fingerprint:
        raise ValueError('REGISTRY_FINGERPRINT_MISMATCH')
    candidates = candidates_checked(registry,candidates)
    current = {k: deepcopy(obs[k]) for k in ('chief_complaint','current_symptoms','objective_findings') if k in obs}
    context = {k: deepcopy(obs[k]) for k in ('history','longitudinal_memory') if k in obs}
    current_tokens, context_tokens = set(observed_tokens(current)), set(observed_tokens(context))
    support = []
    for department in candidates:
        keywords = registry.get('specialty',department)['keywords']
        def matched(tokens):
            return [kw for kw in keywords if (set(re.findall(r'\w+',kw.casefold())) and set(re.findall(r'\w+',kw.casefold())) <= tokens)]
        a,b = matched(current_tokens),matched(context_tokens)
        support.append({'department':department,'current_keyword_matches':a,'context_keyword_matches':b,
            'support_status':'CURRENT_EXPLICIT_KEYWORD_SUPPORT' if a else 'CONTEXT_ONLY_SUPPORT' if b else 'NO_CONFIGURED_KEYWORD_SUPPORT',
            'evidence_fingerprint':digest({'current':current,'context':context,'department':department,'matches':[a,b]})})
    supported = [s['department'] for s in support if s['current_keyword_matches']]
    choice = supported[0] if len(supported)==1 else None
    return {'mode':'abstention_aware_development','decision_status':'selected' if choice else 'abstained',
        'selected_specialty':choice,'candidate_departments':candidates,'support':support,
        'rationale':'UNIQUE_CURRENT_SUPPORT' if choice else 'AMBIGUOUS_CURRENT_SUPPORT' if supported else 'NO_CURRENT_SUPPORT',
        'input_fingerprint':digest(obs),'policy_fingerprint':policy_spec()['fingerprint'],
        'registry_fingerprint':expected_registry_fingerprint,'clinical_appropriateness':'NOT_EVALUATED'}


def validate_cases(cases, registry, training_patients=()):
    split_check(cases,training_patients)
    if not cases or len({c['decision_id'] for c in cases}) != len(cases): raise ValueError('REGISTERED_DECISIONS_REQUIRED')
    for c in cases:
        clean(c['observation']); candidates_checked(registry,c['candidate_departments'])
        for key,value in [('source_fingerprint',c['source_fingerprint']),('evidence_fingerprint',c['evidence_fingerprint'])]:
            if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value): raise ValueError('SOURCE_OR_EVIDENCE_FINGERPRINT_REQUIRED')
        if c['observable_input_fingerprint'] != digest(c['observation']): raise ValueError('OBSERVABLE_INPUT_FINGERPRINT_MISMATCH')
        if c['registry_fingerprint'] != digest(registry.snapshot()): raise ValueError('REGISTRY_FINGERPRINT_MISMATCH')
    return cases


def compare(cases,registry,*,static_department='neurology',training_patients=()):
    """Same snapshot/candidates for four actual routing paths; no outcome input."""
    validate_cases(cases,registry,training_patients)
    if registry.utility is not None: raise ValueError('VERIFIED_ADAPTIVE_UTILITY_INJECTION_NOT_IMPLEMENTED_USE_EXISTING_C4_08_GATE')
    traces=[]
    for c in cases:
        obs=clean(c['observation']); candidates=candidates_checked(registry,c['candidate_departments'])
        if static_department not in candidates: raise ValueError('STATIC_NOT_IN_CANDIDATES')
        decisions=[('A_static_development',{'mode':'static_development','decision_status':'selected','selected_specialty':static_department,'candidate_departments':candidates,'rationale':'PREDECLARED_C4_08_NEUROLOGY_BASELINE'}),
            ('B_rule_similarity',registry.route(deepcopy(obs),candidates,mode='rule',seed=42)),
            ('C_adaptive_accuracy',registry.route(deepcopy(obs),candidates,mode='adaptive',seed=42)),
            ('D_abstention_aware',select(obs,registry,candidates,expected_registry_fingerprint=c['registry_fingerprint']))]
        for condition,decision in decisions:
            traces.append({'case_ref':c['case_ref'],'decision_id':c['decision_id'],'condition':condition,
                'execution_status':'MISSING_UTILITY' if condition=='C_adaptive_accuracy' else 'EXECUTED',
                'input_fingerprint':digest(obs),'memory_fingerprint':digest(obs.get('longitudinal_memory',{})),
                'candidate_fingerprint':digest(candidates),'registry_fingerprint':c['registry_fingerprint'],
                'source_fingerprint':c['source_fingerprint'],'decision':decision})
    return traces


def reference_blank(cases):
    return {'version':REFERENCE,'target':SCOPE,'cases_fingerprint':digest(cases),'reviews':[
        {'case_ref':c['case_ref'],'decision_id':c['decision_id'],'observable_input_fingerprint':c['observable_input_fingerprint'],
         'source_fingerprint':c['source_fingerprint'],'registry_fingerprint':c['registry_fingerprint'],
         'candidate_fingerprint':digest(sorted(c['candidate_departments'])),'reference_scope':SCOPE,
         'source_timing_status':'NOT_REVIEWED','observation_sufficiency_review':None,
         'recorded_time_proxy_review':'NOT_REVIEWED','accepted_specialties':None,'acceptable_abstention':None,
         'abstention_review_status':'NOT_REVIEWED','permitted_diagnoses':None,'reviewer_rationale':None,
         'reviewer_kind':None,'reviewer_qualification':None,'reviewer_private_ref':None,'reviewer_independent':None,
         'procedure_version':REFERENCE,'reviewed_at':None,'independent_review_status':'NOT_REVIEWED',
         'evidence_fingerprint':c['evidence_fingerprint'],'not_evaluable_reason':None,
         'agent_abstention_reviews':None} for c in cases]}


def validated_reviews(cases, references, *, test_only=False):
    if not isinstance(references,dict): raise ValueError('INDEPENDENT_REFERENCE_REQUIRED')
    if (references.get('version')!=REFERENCE or references.get('target')!=SCOPE or references.get('cases_fingerprint')!=digest(cases)):
        raise ValueError('REFERENCE_TARGET_OR_COHORT_MISMATCH')
    refs=references.get('reviews',[])
    if [r.get('decision_id') for r in refs] != [c['decision_id'] for c in cases]: raise ValueError('FULL_REGISTERED_COHORT_REQUIRED')
    judged={}
    for c,r in zip(cases,refs):
        for key,want in [('case_ref',c['case_ref']),('observable_input_fingerprint',c['observable_input_fingerprint']),
                         ('source_fingerprint',c['source_fingerprint']),('registry_fingerprint',c['registry_fingerprint']),
                         ('candidate_fingerprint',digest(sorted(c['candidate_departments']))),('evidence_fingerprint',c['evidence_fingerprint']),('reference_scope',SCOPE)]:
            if r.get(key)!=want: raise ValueError('REFERENCE_BINDING_MISMATCH')
        status=r.get('independent_review_status')
        if status=='NOT_REVIEWED':
            if any(r.get(k) is not None for k in ['accepted_specialties','acceptable_abstention','observation_sufficiency_review','permitted_diagnoses','reviewed_at','reviewer_independent']):
                raise ValueError('UNREVIEWED_JUDGMENT_NOT_ALLOWED')
            if any(r.get(k)!='NOT_REVIEWED' for k in ['source_timing_status','recorded_time_proxy_review','abstention_review_status']):
                raise ValueError('UNREVIEWED_TIMING_NOT_APPROVED')
            continue
        kind='TEST_ONLY' if test_only else 'HUMAN_CLINICAL_REVIEWER'
        if test_only and not c['patient_id'].startswith('TEST_ONLY'): raise ValueError('TEST_ONLY_IDENTITY_REQUIRED')
        if (r.get('reviewer_kind')!=kind or r.get('reviewer_independent') is not True
                or not all(isinstance(r.get(k),str) and r[k].strip() for k in ['reviewer_private_ref','reviewer_qualification','reviewer_rationale'])
                or r.get('procedure_version')!=REFERENCE): raise ValueError('INDEPENDENT_REVIEW_REQUIRED')
        try:
            if not timestamp(c['decision_time']) <= timestamp(r['reviewed_at']) <= datetime.now(timezone.utc): raise ValueError()
        except (ValueError,TypeError,KeyError): raise ValueError('REVIEW_TIME_INVALID')
        if status=='NOT_EVALUABLE':
            if not r.get('not_evaluable_reason') or r.get('accepted_specialties') is not None or r.get('acceptable_abstention') is not None:
                raise ValueError('NOT_EVALUABLE_REASON_REQUIRED')
            continue
        accepted=r.get('accepted_specialties')
        if (status!='REVIEWED' or r.get('source_timing_status')!='CURRENT_DECISION_REFERENCE_VALID'
                or type(r.get('observation_sufficiency_review')) is not bool
                or r.get('recorded_time_proxy_review') not in ['ACCEPTED_FOR_DEVELOPMENT_ROUTING_ONLY','NOT_APPLICABLE_ACTUAL_RELEASE_VERIFIED']
                or not isinstance(accepted,list) or len(set(accepted))!=len(accepted) or not set(accepted)<=set(c['candidate_departments'])):
            raise ValueError('ROUTING_REFERENCE_REVIEW_REQUIRED')
        if r.get('abstention_review_status')=='REVIEWED':
            if type(r.get('acceptable_abstention')) is not bool: raise ValueError('ABSTENTION_BOOLEAN_REVIEW_REQUIRED')
        elif r.get('abstention_review_status')!='NOT_REVIEWED' or r.get('acceptable_abstention') is not None:
            raise ValueError('ABSTENTION_TARGET_REVIEW_REQUIRED')
        if not accepted and r.get('acceptable_abstention') is not True: raise ValueError('NO_ACCEPTABLE_DECISION_REFERENCE')
        judged[c['decision_id']]=r
    return judged


def fraction(n,d,*,missing=False):
    return {'numerator':n,'denominator':d,'value':None if not d or missing else n/d,
            'status':'NOT_EVALUATED' if missing else 'NOT_APPLICABLE' if not d else 'COMPUTED'}


def metrics(cases,traces,references,*,test_only=False,purpose='independent_evaluation',training_patients=()):
    """No diagnosis grading. Cohort/patient binding is checked before counting."""
    split_check(cases,training_patients)
    if purpose not in ['independent_evaluation','development_reviewed']: raise ValueError('EVALUATION_PURPOSE_REQUIRED')
    if purpose=='independent_evaluation' and any(c['split']=='train' for c in cases): raise ValueError('TRAIN_NOT_INDEPENDENT_EVALUATION')
    if test_only != all(c['patient_id'].startswith('TEST_ONLY') for c in cases): raise ValueError('TEST_ONLY_SCOPE_MISMATCH')
    judged=validated_reviews(cases,references,test_only=test_only)
    expected={c['decision_id']:c for c in cases}
    if len(traces)!=len(cases) or {t['decision_id'] for t in traces}!=set(expected): raise ValueError('ONE_TRACE_PER_REGISTERED_DECISION_REQUIRED')
    if len({t['condition'] for t in traces})>1: raise ValueError('ONE_PAIRED_CONDITION_AT_A_TIME')
    selected=abstained=invalid=correct=wrong=appropriate=inappropriate=unreviewed_abstention=0
    for t in traces:
        c=expected[t['decision_id']]
        if any(t[k]!=want for k,want in [('input_fingerprint',c['observable_input_fingerprint']),('source_fingerprint',c['source_fingerprint']),('registry_fingerprint',c['registry_fingerprint']),('candidate_fingerprint',digest(sorted(c['candidate_departments']))) ]):
            raise ValueError('TRACE_SNAPSHOT_MISMATCH')
        if t['decision_id'] not in judged: continue
        r=judged[t['decision_id']];d=t['decision'];status=d.get('decision_status');choice=d.get('selected_specialty')
        if t.get('execution_status')!='EXECUTED': invalid+=1
        elif status=='selected' and choice in c['candidate_departments'] and d.get('candidate_departments')==sorted(c['candidate_departments']):
            selected+=1
            if choice in r['accepted_specialties']: correct+=1
            else: wrong+=1
        elif status=='abstained' and choice is None and d.get('candidate_departments')==sorted(c['candidate_departments']):
            abstained+=1
            if r['abstention_review_status']!='REVIEWED': unreviewed_abstention+=1
            elif r['acceptable_abstention']: appropriate+=1
            else: inappropriate+=1
        else: invalid+=1
    n=len(judged);reviewed_abstentions=appropriate+inappropriate
    execution_blockers=sorted({t['execution_status'] for t in traces if t.get('execution_status')!='EXECUTED'})
    missing=n==0 or not any(t.get('execution_status')=='EXECUTED' for t in traces)
    return {'registered_decisions':len(cases),'reference_evaluable_decisions':n,'unique_registered_patients':len({c['patient_id'] for c in cases}),
        'counts':{'selected':selected,'abstained':abstained,'invalid':invalid,'acceptable_selection':correct,'outside_accepted_set':wrong,'appropriate_abstention':appropriate,'inappropriate_abstention':inappropriate,'abstention_appropriateness_unreviewed':unreviewed_abstention},
        'routing_coverage':fraction(selected,n,missing=missing),'selective_routing_accuracy':fraction(correct,selected,missing=missing),
        'incorrect_routing_rate':fraction(wrong,n,missing=missing),'routing_abstention_rate':fraction(abstained,n,missing=missing),
        'invalid_decision_rate':fraction(invalid,n,missing=missing),'appropriate_abstention_rate':fraction(appropriate,reviewed_abstentions,missing=reviewed_abstentions==0),
        'inappropriate_abstention_rate':fraction(inappropriate,reviewed_abstentions,missing=reviewed_abstentions==0),
        'reference_coverage':fraction(n,len(cases)),'clinical_metrics':'NOT_EVALUATED' if test_only or missing else 'REVIEWED_ROUTING_DESCRIPTIVE_ONLY',
        'execution_blockers':execution_blockers,
        'evidence_level':'TEST_ONLY' if test_only else 'DEVELOPMENT_REVIEWED' if purpose=='development_reviewed' else 'INDEPENDENT_REVIEWED',
        'diagnosis_correctness':'NOT_EVALUATED_SEPARATE_GATE','correlation_policy':'Decision-level descriptive counts; same-patient decisions are clustered, no independent-sample CI or significance claim',
        'research_evaluation_ready':False}
