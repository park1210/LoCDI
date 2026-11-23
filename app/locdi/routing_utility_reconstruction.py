"""Evidence-gated utility reconstruction outside unchanged Routing Core.

Source code comparisons are facts, never implicit correctness. Development
artifacts require explicit proxy judgments, complete groups and disjoint splits.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility, observed_tokens, jaccard_similarity, METHOD_V2
from app.locdi.llm_setup import digest
from app.locdi.routing_evaluation import observable, paired_route, split_check
from validation.routing_correctness_gate import to_policy_fit_rows

VERSION = 'locdi.routing_reconstruction.c4_11.v1'
PROXY = 'SOURCE_PROXY_DEVELOPMENT'


def source_code_bindings(references):
    """Exact explicit source description->coding relation; no synonyms/normalization."""
    index = defaultdict(set)
    for reference in references:
        for item in reference['source_condition_rows']:
            r = item['source_record']
            if all(r.get(k) for k in ('SYSTEM', 'CODE', 'DESCRIPTION')):
                index[r['DESCRIPTION']].add((r['SYSTEM'], r['CODE']))
    return {name: sorted(values) for name, values in index.items()}


def reference_facts(candidate, reference):
    d = candidate['decision']; day = date.fromisoformat(candidate['visit_started_at'][:10])
    rows=[]; problems=[]
    for source in reference['source_condition_rows']:
        r=source['source_record']
        valid = r.get('PATIENT')==d['patient_id'] and r.get('ENCOUNTER')==d['visit_id']
        delta=(date.fromisoformat(r['START'])-day).days
        if not valid:problems.append('SOURCE_PATIENT_ENCOUNTER_MISMATCH')
        if delta>0:problems.append('SOURCE_CONDITION_AFTER_DECISION_DAY')
        rows.append({'system':r.get('SYSTEM'),'code':r.get('CODE'),'description':r.get('DESCRIPTION'),
                     'patient_encounter_link':valid,'start_day_offset':delta,
                     'source_row_fingerprint':digest(r),'date_only':len(r['START'])==10})
    if not rows:problems.append('SOURCE_DIAGNOSIS_MISSING')
    # Source rows can establish recorded linkage, not clinical task suitability.
    problems.extend(['DECISION_REFERENCE_SUITABILITY_NOT_REVIEWED','DATE_ONLY_REFERENCE_TIME_NOT_REVIEWED',
                     'MULTIPLE_DIAGNOSIS_SCOPE_NOT_REVIEWED'])
    return {'case_ref':candidate['case_ref'],'source_fingerprint':candidate['source_fingerprint'],
            'condition_rows':rows,'declared_label':reference['declared_label'],
            'clinical_reference_approved':False,'decision_reference_suitability':'NOT_REVIEWED',
            'blocker_codes':sorted(set(problems))}


def response_evidence(row, candidate, facts, bindings):
    """Inventory original response; correct/proxy_correct remain null without actual judgments."""
    answer=row.get('response') or {}; issues=[]
    if row.get('response_fingerprint')!=digest(answer):issues.append('RESPONSE_FINGERPRINT_MISMATCH')
    if row.get('shared_input_fingerprint')!=candidate['shared_input_fingerprint']:issues.append('INPUT_FINGERPRINT_MISMATCH')
    if row.get('execution_status')!='SUCCEEDED':state='REQUEST_FAILED'
    elif row.get('response_contract_status')!='VALID':state='SCHEMA_INVALID'
    elif answer.get('certainty')=='unresolved' or answer.get('diagnosis')=='Unresolved':state='UNRESOLVED'
    elif answer.get('certainty')=='supported' and answer.get('diagnosis') in candidate['shared_payload']['diagnosis_catalog']:state='SUPPORTED'
    else:state='NOT_EVALUABLE'
    codes=bindings.get(answer.get('diagnosis'),[]) if state=='SUPPORTED' else []
    extracted='EXACT_SOURCE_DESCRIPTION_CODE_BINDING' if len(codes)==1 else 'AMBIGUOUS_OR_MISSING_CODE_BINDING' if state=='SUPPORTED' else 'NOT_EVALUABLE'
    ref_codes={(r['system'],r['code']) for r in facts['condition_rows'] if r['system'] and r['code']}
    comparison=tuple(codes[0]) in ref_codes if len(codes)==1 else None
    if state!='SUPPORTED':issues.append(state+'_NO_DIAGNOSIS_BOOLEAN')
    if state=='SUPPORTED' and len(codes)!=1:issues.append('DIAGNOSIS_CODE_MAPPING_UNRESOLVED')
    issues.extend(facts['blocker_codes'])
    return {'patient_ref':candidate['case_ref'].replace('Candidate','Patient'), 'decision_ref':candidate['case_ref'],
        'agent_id':row['agent_id'],'department':row['department'],'split':row['split'],
        'model':row['model'],'model_version':row['version'],'observed_model_identifier':row.get('observed_model_identifier'),
        'observable_input_fingerprint':row.get('observation_fingerprint'),
        'shared_input_fingerprint':row.get('shared_input_fingerprint'),'response_fingerprint':row.get('response_fingerprint'),
        'response_status':state,'answer_extraction_status':extracted,'extracted_codes':codes,
        'exact_reference_code_match':comparison,'correctness_reference_type':'UNREVIEWED_SOURCE_CONDITION',
        'correctness_status':'NOT_EVALUABLE' if state!='SUPPORTED' else 'REFERENCE_NOT_REVIEWED',
        'source_proxy_status':'UNJUDGED','correct':None,'proxy_correct':None,
        'eligibility_for_training':False,'exclusion_reason':sorted(set(issues)),
        'source_fingerprint':candidate['source_fingerprint'],'evidence_type':'SOURCE_FACTS_NOT_CLINICAL_JUDGMENT'}


def fit_verified(data, options):
    """Only the existing two gates may supply actual correctness rows to Core fit."""
    result=to_policy_fit_rows(data,**options)
    if result['rows'] is None:
        return None,result['report']
    if data.get('purpose')!='routing_training' or not result['report']['utility_fit_ready']:
        raise ValueError('INDEPENDENT_REVIEWED_FIT_REQUIRED')
    from app.core.routing_utility import fit_utility
    registry=AgentRegistry(options['registry_manifest'])
    artifact=fit_utility(result['rows'],registry,
        min_samples=result['report']['positive_similarity_support']['min_samples'])
    artifact.update(reference_type='INDEPENDENT_REVIEWED',provenance={
        'data_fingerprint':digest(data),'review_fingerprint':digest(options['review_results']),
        'registry_fingerprint':digest(registry.snapshot()),'gate_report_fingerprint':digest(result['report'])})
    RoutingUtility(artifact,registry)
    return artifact,result['report']


def build_development(rows, manifest, *, evaluation_patients=(), min_samples=2, test_only=False):
    """Algorithmic sample projection, not clinical Core fitting or missing-label imputation.

    A proxy row must carry an explicit supplied judgment and reviewed source-scope
    link. Real unreviewed inventory rows cannot enter this path.
    """
    if type(min_samples)is not int or min_samples<1:raise ValueError('INVALID_MIN_SAMPLES')
    registry=AgentRegistry(manifest); agents={a['agent_id']:a for a in manifest['agents'] if a['role']=='specialty'}
    if not rows:raise ValueError('NO_ELIGIBLE_PROXY_ROWS')
    required_type='TEST_ONLY' if test_only else PROXY
    groups=defaultdict(list);seen=set();samples=[]
    for row in rows:
        key=(row['patient_id'],row['decision_id'],row['agent_id'])
        if key in seen:raise ValueError('DUPLICATE_OUTCOME_NO_SELECTIVE_REPLACEMENT')
        seen.add(key)
        if row['patient_id'] in set(evaluation_patients) or row.get('split')!='train':raise ValueError('TRAIN_EVALUATION_PATIENT_OVERLAP')
        if (row.get('reference_type')!=required_type or (row['patient_id'].startswith('TEST_ONLY')!=test_only)
            or row.get('response_status')!='SUPPORTED' or type(row.get('proxy_correct')) is not bool):
            raise ValueError('SUPPORTED_EXPLICIT_PROXY_JUDGMENT_REQUIRED')
        if (row.get('source_scope_status')!='REVIEWED_FOR_SOURCE_PROXY_ONLY'
            or not row.get('scope_review_ref') or row.get('reference_timing_status')!='CURRENT_DECISION_SOURCE_PROXY_VALID'
            or row.get('source_start_after_decision') is not False
            or row.get('response_fingerprint')!=digest(row.get('response'))
            or row.get('source_fingerprint')!=digest(row.get('source_evidence'))):
            raise ValueError('PROXY_SOURCE_PROVENANCE_REQUIRED')
        agent=agents.get(row['agent_id'])
        if agent is None or any(row.get(k)!=agent[k] for k in ('model','version')):raise ValueError('REGISTRY_MODEL_VERSION_MISMATCH')
        obs=observable(row['observation']); tokens=observed_tokens(obs)
        groups[key[:2]].append((row,tokens))
        samples.append({'patient_id':row['patient_id'],'decision_id':row['decision_id'],
            'agent_id':row['agent_id'],'correct':row['proxy_correct'],'tokens':tokens})
    for group in groups.values():
        if ({r['agent_id'] for r,_ in group}!=set(agents) or len(group)!=len(agents)
            or any(tokens!=group[0][1] or digest(r['observation'])!=digest(group[0][0]['observation']) for r,tokens in group)):
            raise ValueError('FIVE_AGENT_SAME_INPUT_COMPLETE_GROUP_REQUIRED')
    if len(groups)<min_samples:raise ValueError('MIN_SAMPLES_NOT_MET')
    artifact={'schema_version':2,'method':METHOD_V2,'registry':registry.snapshot(),
        'min_samples':min_samples,'samples':samples,'reference_type':required_type,
        'clinical_fit_approved':False,'training_patients':sorted({r['patient_id'] for r in rows}),
        'provenance':{'row_fingerprint':digest(rows),'registry_fingerprint':digest(manifest),
            'target':'DIAGNOSIS_SOURCE_PROXY_NOT_SPECIALTY_SUITABILITY','core_fit_utility_called':False}}
    RoutingUtility(artifact,registry)
    return artifact


def require_clinical_artifact(artifact):
    if artifact.get('reference_type')!='INDEPENDENT_REVIEWED':
        raise ValueError('DEVELOPMENT_OR_TEST_ONLY_NOT_CLINICAL')
    # This marker is necessary, never sufficient: the existing evaluator must
    # additionally call validate_utility_provenance and the existing gates.
    return True


def execute_conditions(cases, manifest, artifact=None, *, static_department='neurology'):
    registry=AgentRegistry(manifest,artifact)
    split_check(cases,registry.utility.training_patients if registry.utility else ())
    paired=paired_route(cases,registry,static_department=static_department)
    out=[]
    for i,case in enumerate(cases):
        static,adaptive=paired[2*i:2*i+2]
        rule=registry.route(observable(case['observation']),case['candidate_departments'],mode='rule',seed=42)
        rule_row={**{k:deepcopy(v) for k,v in static.items() if k not in ('decision','condition')},
            'condition':'rule_similarity','decision':rule}
        out.extend([static,rule_row,adaptive])
    return out


def execution_summary(traces):
    out={}
    for condition in ('static_development','rule_similarity','adaptive_native_abstention'):
        rows=[r for r in traces if r['condition']==condition]; ds=[r['decision'] for r in rows]
        scores=[e['score'] for d in ds for e in d.get('utility_estimates',{}).values() if e is not None]
        out[condition]={'decisions':len(rows),'selected':sum(d['decision_status']=='selected' for d in ds),
            'abstained':sum(d['decision_status']=='abstained' for d in ds),
            'missing_utility':sum(d.get('reason_code')=='MISSING_UTILITY' for d in ds),
            'insufficient_support':sum(d.get('reason_code')=='INSUFFICIENT_SUPPORT' for d in ds),
            'tie':sum(bool(d.get('tied_departments')) for d in ds),
            'score_distribution':{'count':len(scores),'min':min(scores) if scores else None,'max':max(scores) if scores else None},
            'clinical_accuracy':'NOT_EVALUATED','specialty_reference_coverage':{'numerator':0,'denominator':len(rows),'value':0 if rows else None}}
    return out
