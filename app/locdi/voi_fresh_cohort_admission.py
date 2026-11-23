"""C4-35 proposed Gate approval and separate frozen evaluation approval."""
from copy import deepcopy
from app.locdi.llm_setup import digest
from app.locdi.new_cohort_execution import approval,execute_panel,summarize

ADMISSION_SCOPE='C4_35_REVISED_ADMISSION_CONTRACT'
EVALUATION_SCOPE='C4_35_FROZEN_POLICY_EVALUATION'


def classify_collision(*,mapping_error=False,provenance_conflict=False,full_history_match=False,module_evidence=None):
    if mapping_error:return 'SOURCE_MAPPING_ERROR'
    if provenance_conflict:return 'PROVENANCE_CONFLICT'
    if full_history_match:return 'PATIENT_REUSE_SUSPECTED'
    if module_evidence and all(module_evidence.get(k) is True for k in
            ('pinned_generator_module_verified','literal_condition_and_observation_match','fixed_value_and_unit_match')):
        return 'COMMON_PATTERN_SUPPORTED'
    return 'UNRESOLVED_CONTENT_COLLISION'


def approve_admission(proposal,contract,record):
    if contract['fingerprint']!=digest({k:v for k,v in contract.items() if k!='fingerprint'}):raise ValueError('ADMISSION_CONTRACT_HASH')
    approval(record,ADMISSION_SCOPE,contract['fingerprint'])
    if proposal['global_blockers']:raise ValueError('GLOBAL_INDEPENDENCE_BLOCKED')
    if contract['proposal_fingerprint']!=digest(proposal):raise ValueError('PROPOSED_PANEL_CHANGED')
    if record.get('bounded_export_history_review_accepted') is not True:
        raise ValueError('EXPORT_SCOPE_LIMIT_REVIEW_REQUIRED')
    panel=deepcopy(proposal['proposed'])
    return {'status':'ADMISSION_APPROVED_EVALUATION_NOT_APPROVED','admission_contract_fingerprint':contract['fingerprint'],
        'panel':panel,'panel_fingerprint':digest(panel),'patients':len({r['patient_id'] for r in panel}),
        'visits':len(panel),'clinical_reference_approved':False,'policy_evaluation_approved':False}


def run_approved(manifest,model,evaluation_protocol,record,sink,*,test_only=False):
    approval(record,EVALUATION_SCOPE,evaluation_protocol['fingerprint'])
    if manifest['status']!='ADMISSION_APPROVED_EVALUATION_NOT_APPROVED':raise ValueError('ADMISSION_NOT_APPROVED')
    if manifest['panel_fingerprint']!=digest(manifest['panel']):raise ValueError('ADMITTED_PANEL_CHANGED')
    if evaluation_protocol['admission_panel_fingerprint']!=manifest['panel_fingerprint']:
        raise ValueError('EVALUATION_PANEL_BINDING')
    if len({r['patient_id'] for r in manifest['panel']})<2:raise ValueError('INSUFFICIENT_PATIENTS')
    records,membership,resources=execute_panel(model,manifest['panel'],sink)
    return records,membership,resources,summarize(records,membership)
