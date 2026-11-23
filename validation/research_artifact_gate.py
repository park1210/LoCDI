"""Research admission prerequisites over existing validators; no training/calls.

Clinical evidence authentication is not installed in this project. Declarations
alone therefore NEVER establish clinical review or research authorization.
READY denotes the named mechanically verified prerequisite, not an artifact's
overall research readiness. Caller data/reports are never trusted as gate output.
"""
from app.locdi.llm_setup import digest
from validation.artifact_readiness import validate_routing_artifact, validate_voi_artifact
from validation.routing_correctness_gate import validate_evaluation_gate
from validation.routing_correctness_gate import to_policy_fit_rows
from app.core.routing_utility import observed_tokens


def assess_artifacts(registry_manifest, *, routing_artifact=None, voi_artifact=None,
                     correctness_inputs=None, evaluation_patient_ids=None,
                     source_actions=None, diagnosis_candidates=None, binding=None, declarations=None):
    """Read-only check; private inputs produce only statuses/counts/fingerprints.

    binding is the EXPECTED current runtime contract, not a clinical certificate.
    No arbitrary precomputed correctness report is accepted.
    """
    rows = {'routing': [], 'voi': []}
    def check(kind, code, value):
        rows[kind].append({'code':code, 'status':value})
    def ready(value):
        return 'READY' if value else 'BLOCKED'
    expected = binding if isinstance(binding, dict) else {}
    declarations = declarations if isinstance(declarations, dict) else {}
    def declaration(kind, artifact):
        value=declarations.get(kind,{})
        return value if isinstance(value,dict) and artifact is not None and value.get('artifact_fingerprint')==digest(artifact) else {}
    rd=declaration('routing',routing_artifact);vd=declaration('voi',voi_artifact)
    gate = None
    if correctness_inputs is not None:
        try:
            data, options = correctness_inputs
            # Re-run BOTH current policy and legacy validator, never caller report.
            gate = validate_evaluation_gate(data, **options)
        except Exception:
            gate = None
    r = validate_routing_artifact(registry_manifest, routing_artifact,
        evaluation_patient_ids=evaluation_patient_ids)
    v = validate_voi_artifact(voi_artifact, source_actions=source_actions,
        evaluation_patient_ids=evaluation_patient_ids, diagnosis_candidates=diagnosis_candidates)
    for kind, report in (('routing',r), ('voi',v)):
        check(kind, 'CORE_STRUCTURAL_CONTRACT', ready(report['structural_status']=='STRUCTURAL_VALID'))
        check(kind, 'DEVELOPMENT_TRANSPORT_CONTRACT', ready(report['development_status']=='DEVELOPMENT_CONTRACT_READY'))
        for c in report['checks']:
            check(kind, c['code'], 'READY' if c['status']=='READY' else 'NOT_CHECKED' if c['status']=='NOT_REVIEWED' else 'BLOCKED')
    for code, field in [('INDEPENDENT_REFERENCE','source_reference_ready'),
                        ('CORRECTNESS_GATE','utility_fit_ready'), ('INDEPENDENT_RESPONSE_REVIEW','clinical_review_ready')]:
        check('routing',code, ready(gate.get(field) is True) if gate else 'NOT_CHECKED')
    check('routing','COMPLETE_TRAIN_GROUP',ready(gate.get('complete_train_groups',0)>0) if gate else 'NOT_CHECKED')
    check('routing','VALIDATION_GROUP',ready(gate.get('validation_group_count',0)>0) if gate else 'NOT_CHECKED')
    check('routing','TRAIN_VALIDATION_SPLIT',ready(isinstance(gate.get('legacy_validator'),dict) and
        not gate['legacy_validator']['blocker_codes']) if gate else 'NOT_CHECKED')
    check('routing','POSITIVE_SIMILARITY_SUPPORT',ready(gate.get('positive_similarity_support',{}).get('status')==
        'SUPPORTED_FOR_SUPPLIED_VALIDATION_QUERIES') if gate else 'NOT_CHECKED')
    lineage=False
    if gate and gate.get('utility_fit_ready') is True and isinstance(routing_artifact,dict):
        try:
            converted=to_policy_fit_rows(correctness_inputs[0],**correctness_inputs[1])['rows']
            expected_samples=[{k:r[k] for k in ('patient_id','decision_id','agent_id','correct')} for r in converted]
            for sample,row in zip(expected_samples,converted):sample['tokens']=observed_tokens(row['observation'])
            lineage=sorted(map(digest,expected_samples))==sorted(map(digest,routing_artifact['samples']))
        except Exception:
            pass
    check('routing','FIT_ROWS_ARTIFACT_LINEAGE',ready(lineage))
    check('routing','REGISTRY_MODEL_PROMPT_BINDING',ready(bool(expected.get('prompt_fingerprint')) and
        bool(expected.get('model_identifier')) and expected.get('registry_fingerprint')==digest(registry_manifest) and
        rd.get('research_binding')==expected))
    check('routing','ARTIFACT_PROVENANCE_VERSION',ready(isinstance(rd.get('provenance'),dict) and bool(rd.get('artifact_version'))))
    check('routing','FAILURE_ABSTENTION_POLICY',ready(rd.get('evaluation_target')=='diagnosis_correctness' and
        rd.get('failure_policy')=='failed_and_unresolved_not_imputed' and
        rd.get('abstention_policy')=='insufficient_support_or_tied_best'))
    provenance = voi_artifact.get('provenance',{}) if isinstance(voi_artifact,dict) else {}
    cross_overlap=None
    if gate is not None and correctness_inputs is not None and isinstance(provenance,dict):
        try:
            validation=set(correctness_inputs[0]['split_manifest']['patients']['validation'])
            training=set(provenance['training_patients'])
            cross_overlap=len(validation & training)
        except (KeyError,TypeError):
            pass
    check('voi','ROUTING_VALIDATION_VOI_TRAIN_DISJOINT',
        'NOT_CHECKED' if cross_overlap is None else ready(cross_overlap==0))
    for code in ('DIAGNOSIS_CATALOG_INDEPENDENT_REVIEW','PRIOR_PROVENANCE_REVIEW',
                 'LIKELIHOOD_PROVENANCE_REVIEW','SOURCE_RELEASE_TIMING_REVIEW',
                 'INDEPENDENT_VALIDATION','CLINICAL_UTILITY_REVIEW'):
        check('voi',code,'NOT_CHECKED')
    check('voi','NON_TOY_PROVENANCE',ready(isinstance(provenance,dict) and provenance.get('synthetic') is False))
    check('voi','MODEL_ACTION_MEMORY_BINDING',ready(bool(expected.get('action_catalog_fingerprint')) and
        bool(expected.get('memory_fingerprint')) and bool(expected.get('model_identifier')) and
        bool(expected.get('prompt_fingerprint')) and expected.get('registry_fingerprint')==digest(registry_manifest) and
        vd.get('research_binding')==expected))
    # No trust registry/evidence verifier currently exists. A reviewer name or
    # review_status in an artifact cannot resolve this prerequisite.
    for kind in rows:
        check(kind,'INDEPENDENT_REVIEW_EVIDENCE_AUTHENTICATION','NOT_CHECKED')
    out = {'schema_version':'locdi.research_artifact_gate.v1','artifacts':{},
        'correctness':({k:gate.get(k) for k in ('source_reference_ready','clinical_review_ready','response_policy_ready',
            'complete_train_groups','validation_group_count','positive_similarity_support','blocker_codes','response_states')}
            if gate else {'status':'NOT_CHECKED'}),
        'correctness_scope':correctness_inputs[0].get('purpose') if gate else 'NOT_CHECKED',
        'research_evaluation_ready':False,'clinical_metrics':'NOT_EVALUATED',
        'routing_validation_voi_train_overlap_count':cross_overlap,
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}
    for kind, report in (('routing',r),('voi',v)):
        checks = rows[kind]
        out['artifacts'][kind]={'status':'BLOCKED','checks':checks,
            'artifact_fingerprint':report['artifact_fingerprint'],
            'blocker_codes':[c['code'] for c in checks if c['status']!='READY'],
            'clinical_review_evidence_status':'NOT_CHECKED_NO_AUTHENTICATED_REVIEW_STORE',
            'scope':'READINESS_PREREQUISITES_NOT_EXECUTION_APPROVAL'}
    return out
