"""Read-only research checklist. No runner, approval authority or model calls.

Flags and submitted review shapes cannot authenticate an independent review.
Development execution remains governed by the existing, separate live gates.
"""
from collections.abc import Mapping
from experiments.ablation_fairness import check_ablation_fairness
from experiments.run_locdi_ablation import CONDITIONS
from validation.research_artifact_gate import assess_artifacts
from validation.review_submission_contract import validate_submission


def assess_official_evaluation(configs, *, cases=None, case_sets=None,
        split_manifest=None, artifact_inputs=None, review_submissions=None,
        protocol=None, research_approval=None):
    checks=[]
    def check(code,status,scope='ALL'):
        checks.append({'code':code,'status':status,'scope':scope})
    fairness=check_ablation_fairness(configs,cases=cases,case_sets=case_sets)
    for row in fairness['checks']:
        check(row['code'],row['status'])
    check('PAIRED_INPUT_ORDER','READY' if fairness.get('paired_input_order_equal') is True else 'BLOCKED')
    check('PROMPT_AND_REQUEST_BUDGET_BINDING','BLOCKED')
    check('SEQUENTIAL_HANDOFF_PROTOCOL_UNSPECIFIED','BLOCKED')
    for name,row in fairness.get('conditions',{}).items():
        check('OFFICIAL_FEATURES_'+name,row['official_features'],name)
        check('MODEL_BINDING_'+name,row['model_bindings'],name)
    valid_split=False
    if isinstance(split_manifest,Mapping) and set(split_manifest)=={'train','validation','test'}:
        groups=[split_manifest[k] for k in ('train','validation','test')]
        valid_split=all(isinstance(g,list) and all(isinstance(p,str) and p for p in g) and len(set(g))==len(g) for g in groups)
        if valid_split:
            sets=list(map(set,groups))
            valid_split=all(not sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
            if isinstance(cases,(list,tuple)) and cases and all(isinstance(c,Mapping) for c in cases):
                valid_split=valid_split and {c.get('patient_id') for c in cases}==sets[2]
            else:valid_split=False
    check('PATIENT_SPLIT_AND_TEST_COVERAGE','READY' if valid_split else 'BLOCKED')
    # Approval declarations are not a trusted paper, review or research authority.
    check('CONFIRMED_PAPER_AND_PROTOCOL','NOT_CHECKED')
    check('INDEPENDENT_CLINICAL_REFERENCE','BLOCKED')
    check('INDEPENDENT_MEMORY_RELEVANCE','BLOCKED')
    check('INDEPENDENT_REVIEW_AUTHENTICATION','NOT_CHECKED')
    check('RESEARCH_APPROVAL_AUTHENTICATION','NOT_CHECKED')
    check('OFFICIAL_METRIC_POLICY_AND_TRANSLATOR','BLOCKED')
    check('OFFICIAL_PAIRED_RUNNER','BLOCKED')
    check('ACTUAL_PAIRED_EXECUTION_CONSISTENCY','NOT_CHECKED')
    review_checks=[validate_submission(r) for r in (review_submissions or [])]
    artifacts=None
    if artifact_inputs is not None:
        try:artifacts=assess_artifacts(**artifact_inputs)
        except (ValueError,TypeError,KeyError,AttributeError):
            check('RESEARCH_ARTIFACT_INPUT_CONTRACT','BLOCKED')
    for name in CONDITIONS:
        if name in ('B3','B4'):
            check('RESEARCH_ROUTING_ARTIFACT','BLOCKED',name)
        if name=='B4':check('RESEARCH_VOI_ARTIFACT','BLOCKED',name)
    # The existing research Artifact Gate is re-executed, never replaced by a
    # caller-supplied READY report. Its current authority/lineage gaps remain.
    return {'schema_version':'locdi.official_evaluation_readiness.v1','status':'BLOCKED',
        'research_evaluation_ready':False,'mode':'READ_ONLY_NO_EXECUTION_OR_APPROVAL',
        'checks':checks,'blocker_codes':sorted({c['code'] for c in checks if c['status']!='READY'}),
        'fairness':fairness,'artifact_gate':artifacts,'review_submission_checks':review_checks,
        'conditions':{name:{'status':'BLOCKED','routing_artifact_required':name in ('B3','B4'),
            'voi_artifact_required':name=='B4'} for name in CONDITIONS},
        'declarations_received':{'protocol':protocol is not None,'research_approval':research_approval is not None},
        'clinical_metrics':'NOT_EVALUATED','generation_requests':0,'utility_fitting':0,
        'clinical_verdicts_generated':0}
