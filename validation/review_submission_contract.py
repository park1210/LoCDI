"""Private review submission SHAPE only; never authenticates clinical judgment.

Independent reference review and response judgment must be separate submissions.
A hash/signature field without a trusted issuer/key registry is not approval.
"""
from datetime import datetime
import re
from app.locdi.llm_setup import digest
from validation.routing_correctness_gate import POLICY

REQUIRED=('schema_version','submission_kind','review_case_ref','decision_ref','reference_source',
    'source_fingerprint','reference_fingerprint','observed_at','available_at','decision_at',
    'reviewer_ref','reviewer_role','reviewer_independent','review_timestamp','decision_status',
    'evidence_reference','correctness_label','review_policy_version','reference_timing_status',
    'evaluation_target','response_ref','response_fingerprint','response_recorded_at','certainty',
    'clinical_evidence_rationale','not_evaluable_reason','approval_proof')


def submission_contract():
    return {'schema_version':'locdi.review_submission_contract.v1','private_only':True,
        'required_fields':list(REQUIRED),'submission_kinds':['REFERENCE_REVIEW','RESPONSE_JUDGMENT'],
        'evaluation_targets':list(POLICY['policies']),
        'decision_statuses':['NOT_REVIEWED','NOT_EVALUABLE','REVIEWED'],
        'approval_methods':['DETACHED_SIGNATURE','INSTITUTIONAL_APPROVAL'],
        'authority_registry':'NOT_CONFIGURED','authenticated_review_ready':False,
        'blank_template_policy':'Existing templates immutable; no reviewer/verdict defaults generated'}


def validate_submission(value):
    codes=[]
    def stamp(s):
        try:
            t=datetime.fromisoformat(s.replace('Z','+00:00'))
            return t if t.tzinfo is not None else None
        except (ValueError,TypeError,AttributeError):return None
    def fp(s):return isinstance(s,str) and re.fullmatch('[0-9a-f]{64}',s) is not None
    if not isinstance(value,dict) or set(value)!=set(REQUIRED):
        return {'status':'BLOCKED','structural_status':'BLOCKED','authentication':'NOT_CHECKED',
            'blocker_codes':['REVIEW_SUBMISSION_FIELDS_INVALID'],'clinical_review_ready':False}
    if value['schema_version']!='locdi.review_submission.v1':codes.append('REVIEW_SCHEMA_VERSION_INVALID')
    target=value['evaluation_target'];kind=value['submission_kind'];state=value['decision_status']
    if target not in POLICY['policies'] or kind not in ('REFERENCE_REVIEW','RESPONSE_JUDGMENT') or state not in ('NOT_REVIEWED','NOT_EVALUABLE','REVIEWED'):
        codes.append('REVIEW_TARGET_KIND_OR_STATUS_INVALID')
    for key in ('review_case_ref','decision_ref','reference_source','reviewer_ref','evidence_reference'):
        if not isinstance(value[key],str) or not value[key].strip():codes.append('PRIVATE_REVIEW_REFERENCE_MISSING')
    if not all(fp(value[k]) for k in ('source_fingerprint','reference_fingerprint')):codes.append('REVIEW_SOURCE_FINGERPRINT_INVALID')
    times={k:stamp(value[k]) for k in ('observed_at','available_at','decision_at','review_timestamp')}
    if not all(times.values()) or not (times['observed_at']<=times['available_at'] and times['review_timestamp']>=times['available_at']):
        codes.append('REVIEW_TIMESTAMP_INVALID')
    if value['reviewer_role']!='HUMAN_CLINICAL_REVIEWER' or value['reviewer_independent'] is not True:
        codes.append('REVIEWER_INDEPENDENCE_NOT_ESTABLISHED')
    if not isinstance(value['review_policy_version'],str) or not value['review_policy_version'].strip():codes.append('REVIEW_PROCEDURE_VERSION_MISSING')
    timing=value['reference_timing_status']
    if timing not in ('CURRENT_DECISION_REFERENCE_VALID','RETROSPECTIVE_OUTCOME_ONLY','REFERENCE_TIMING_NOT_REVIEWED','REFERENCE_REJECTED'):
        codes.append('REFERENCE_TIMING_STATUS_INVALID')
    if state=='REVIEWED':
        if not isinstance(value['clinical_evidence_rationale'],str) or not value['clinical_evidence_rationale'].strip():codes.append('REVIEW_RATIONALE_MISSING')
        if timing!='CURRENT_DECISION_REFERENCE_VALID' or not all(times.values()) or times['available_at']>times['decision_at']:
            codes.append('CURRENT_DECISION_REFERENCE_NOT_VERIFIED')
    if kind=='REFERENCE_REVIEW':
        if any(value[k] is not None for k in ('correctness_label','response_ref','response_fingerprint','response_recorded_at','certainty')):
            codes.append('REFERENCE_REVIEW_MUST_BE_RESPONSE_BLINDED')
    elif kind=='RESPONSE_JUDGMENT':
        if not isinstance(value['response_ref'],str) or not value['response_ref'].strip():
            codes.append('RESPONSE_REFERENCE_MISSING')
        recorded=stamp(value['response_recorded_at'])
        if not fp(value['response_fingerprint']) or not recorded or not times['review_timestamp'] or recorded>times['review_timestamp']:
            codes.append('RESPONSE_FINGERPRINT_OR_REVIEW_TIME_INVALID')
        if target!='diagnosis_correctness' and value['correctness_label'] is not None:codes.append('DIAGNOSIS_LABEL_TARGET_MIXED')
        if value['certainty']=='unresolved' and value['correctness_label'] is not None:codes.append('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN')
        if target=='diagnosis_correctness' and state=='REVIEWED' and (value['certainty']!='supported' or type(value['correctness_label']) is not bool):
            codes.append('SUPPORTED_DIAGNOSIS_JUDGMENT_REQUIRED')
    if state!='REVIEWED' and value['correctness_label'] is not None:codes.append('UNREVIEWED_LABEL_NOT_NULL')
    proof=value['approval_proof']
    if proof is not None:
        if not isinstance(proof,dict) or set(proof)!={'method','issuer_ref','approval_ref','payload_fingerprint','proof_sha256'}:
            codes.append('APPROVAL_PROOF_SHAPE_INVALID')
        elif (proof['method'] not in ('DETACHED_SIGNATURE','INSTITUTIONAL_APPROVAL')
              or any(not isinstance(proof[k],str) or not proof[k].strip() for k in ('issuer_ref','approval_ref'))
              or not fp(proof['proof_sha256']) or proof['payload_fingerprint']!=digest({k:v for k,v in value.items() if k!='approval_proof'})):
            codes.append('APPROVAL_PROOF_PAYLOAD_MISMATCH')
    shape_ok=not codes
    codes.append('INDEPENDENT_SIGNATURE_OR_APPROVAL_AUTHORITY_NOT_VERIFIED')
    return {'schema_version':'locdi.review_submission_check.v1','status':'BLOCKED',
        'structural_status':'READY' if shape_ok else 'BLOCKED','authentication':'NOT_CHECKED_NO_TRUSTED_AUTHORITY',
        'clinical_review_ready':False,'blocker_codes':sorted(set(codes)),
        'source_scope':'PRIVATE_INPUT_NOT_ECHOED_OR_PROMOTED','clinical_verdicts_generated':0}
