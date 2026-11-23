"""C4-17 retrospective coded-domain measurement; no clinical gate or router."""
from collections import Counter
from copy import deepcopy
import json

from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest
from app.locdi.abstention_routing import clean

VERSION = 'locdi.retrospective_source_domain.c4_17.v1'
TARGET = 'RETROSPECTIVE_SOURCE_CODE_DOMAIN_CLASSIFICATION'
FIXED = {'44054006':'endocrinology','431857002':'nephrology',
         '26929004':'neurology','414545008':'cardiology','195967001':'pulmonology'}


def protocol():
    return {'version':VERSION,'target':TARGET,
        'window':'Native completed Encounter start <= Condition recorded onset <= native stop (inclusive)',
        'condition_scope':'All native Conditions linked to that same patient and Encounter; no active-prevalence inference',
        'future_outcome':'Onset after router input but within native completed window is permitted ONLY as private retrospective outcome',
        'missing_end':'UNKNOWN_NO_WINDOW; no invented observation horizon',
        'inputs':'All 15 preexisting Prepared15 minute1 shared observations, unchanged; other scopes INPUT_UNAVAILABLE',
        'cohorts':['Prepared15','Original50','TrainingBank'],
        'selection':'Full registered cohorts; no outcome-driven sampling; all 15 inputs compared even if outcome UNKNOWN',
        'mapping':FIXED,'system':'SNOMED-CT','mapping_expansion':False,
        'unknown':'Unmapped/malformed/missing-time Conditions preserve unknown domains; never negative',
        'negative_labels':'None: no validated clinical negatives or complete accepted specialty reference',
        'full_code_scope':'At least one in-window Condition and every in-window Condition mapped, no missing-time Condition; not clinical completeness',
        'metrics':'Confirmed-positive tag hits only; full accepted-set agreement/disagreement NOT_EVALUATED',
        'policies':{'A':'A_static_development','B':'B_rule_similarity','C':'D_abstention_aware','D':'C_adaptive_accuracy'},
        'static_specialty':'neurology','adaptive':'MISSING_UTILITY',
        'clinical_reference':False,'research_evaluation_ready':False,'llm_generation':0,'utility_fitting':0}


def validate_mapping(mapping):
    ms = mapping['mappings']
    if len(ms)!=5 or {(m['system'],m['code'],tuple(m['specialties'])) for m in ms} != {
            ('SNOMED-CT',k,(v,)) for k,v in FIXED.items()}:
        raise ValueError('FIXED_C4_16_MAPPING_REQUIRED')
    if any(m.get('code_identity_verified') is not True or m.get('specialty_scope_verified') is not True for m in ms):
        raise ValueError('MAPPING_EVIDENCE_REQUIRED')


def public_input(observation):
    """Allow existing snapshots only; reject labels also inside encoded JSON."""
    forbidden={'confirmed_positive_domains','unknown_domains','retrospective_label',
        'retrospective_source_labels','mapping_id','source_value','source_codes',
        'recorded_onset','correctness','correct_diagnosis','gold_department'}
    def walk(x):
        if isinstance(x,dict):
            if {str(k).casefold() for k in x}&forbidden:raise ValueError('RETROSPECTIVE_LABEL_IN_INPUT')
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,str) and x.lstrip().startswith(('{','[')):
            try:y=json.loads(x)
            except ValueError:return
            walk(y)
    walk(observation)
    return clean(observation)


def label(case, facts, encounter, mapping, extracted_at):
    """Facts must already pass native value/pointer/provenance validation."""
    validate_mapping(mapping)
    timestamp(extracted_at);timestamp(case['decision_time'])
    out={'version':VERSION,'target':TARGET,'case_ref':case['case_ref'],'dataset':case['dataset'],
        'router_input_time':case['decision_time'],'extracted_at':extracted_at,
        'encounter_start':encounter.get('start'),'encounter_stop':encounter.get('stop'),
        'confirmed_positive_domains':[],'negative_domains':[],
        'unknown_domains':sorted(FIXED.values()),'in_window_conditions':0,'unmapped_conditions':0,
        'missing_time_conditions':0,'outside_window_conditions':0,'future_outcome_conditions':0,
        'source_condition_count':len(facts),'source_facts':[],'window_verified':False,
        'coverage':'UNKNOWN','clinical_reference':False}
    try:
        start,stop=timestamp(encounter['start']),timestamp(encounter['stop'])
        if stop<start or encounter.get('completed') is not True:raise ValueError()
    except (ValueError,TypeError,KeyError):
        out['reason']='COMPLETED_NATIVE_WINDOW_UNAVAILABLE';return out
    if encounter.get('patient_id')!=case['patient_id'] or encounter.get('visit_id')!=case['visit_id']:
        raise ValueError('ENCOUNTER_IDENTITY_MISMATCH')
    if timestamp(extracted_at)<stop:raise ValueError('EXTRACTION_BEFORE_WINDOW_END')
    out['window_verified']=True;positive=set()
    for f in facts:
        if (f.get('source_verified') is not True or f.get('source_resource_type')!='Condition'
                or f['patient_reference']!=case['patient_id'] or f['visit_reference']!=case['visit_id']):
            raise ValueError('SOURCE_IDENTITY_OR_VERIFICATION_MISMATCH')
        if digest(f['source_value'])!=f['source_value_fingerprint']:
            raise ValueError('SOURCE_VALUE_FINGERPRINT_MISMATCH')
        row={'fact_ref':f['fact_ref'],'recorded_onset':f['recorded_at'],
             'source_value_fingerprint':f['source_value_fingerprint'],'mapping_ids':[],
             'source_file_sha256':f['source_file_sha256'],'source_field':f['source_field']}
        try:t=timestamp(f['recorded_at'])
        except (ValueError,TypeError):
            out['missing_time_conditions']+=1;row['status']='UNKNOWN_TIME';out['source_facts'].append(row);continue
        # Recorded onset must be native source start, not an inferred clinical release.
        from app.locdi.source_enrichment import iso_ms
        if iso_ms(f['source_value'].get('start'))!=f['recorded_at']:
            raise ValueError('SOURCE_RECORDED_ONSET_MISMATCH')
        if not start<=t<=stop:
            out['outside_window_conditions']+=1;row['status']='OUTSIDE_DECLARED_WINDOW'
            out['source_facts'].append(row);continue
        out['in_window_conditions']+=1
        row['after_router_input']=t>timestamp(case['decision_time'])
        out['future_outcome_conditions']+=int(row['after_router_input'])
        matches=[m for m in mapping['mappings'] if any(
            code.get('system')==m['system'] and code.get('code')==m['code']
            for code in f['source_value'].get('codes',[]))]
        for m in matches:positive.update(m['specialties'])
        row['mapping_ids']=[m['mapping_id'] for m in matches]
        row['status']='EXACT_MAPPED' if matches else 'UNMAPPED_UNKNOWN'
        out['unmapped_conditions']+=int(not matches);out['source_facts'].append(row)
    out['confirmed_positive_domains']=sorted(positive)
    out['unknown_domains']=sorted(set(FIXED.values())-positive)
    if positive:
        out['coverage']='FULL_CODE_SCOPE' if not out['unmapped_conditions'] and not out['missing_time_conditions'] else 'PARTIAL_POSITIVE'
    out['reason']='FIXED_CODE_SCOPE_ONLY_NOT_CLINICAL_ACCEPTED_SET'
    return out


def coverage(labels):
    return {'registered_decisions':len(labels),'window_verified':sum(x['window_verified'] for x in labels),
        'with_positive_tags':sum(bool(x['confirmed_positive_domains']) for x in labels),
        'full_code_scope':sum(x['coverage']=='FULL_CODE_SCOPE' for x in labels),
        'partial_positive':sum(x['coverage']=='PARTIAL_POSITIVE' for x in labels),
        'unknown_no_positive':sum(not x['confirmed_positive_domains'] for x in labels),
        'multiple_domain_cases':sum(len(x['confirmed_positive_domains'])>1 for x in labels),
        'positive_domain_tags':sum(len(x['confirmed_positive_domains']) for x in labels),'negative_domain_tags':0,
        'unknown_domain_slots':sum(len(x['unknown_domains']) for x in labels),
        'unmapped_conditions':sum(x['unmapped_conditions'] for x in labels),
        'missing_time_conditions':sum(x['missing_time_conditions'] for x in labels),
        'outside_window_conditions':sum(x['outside_window_conditions'] for x in labels),
        'future_outcome_conditions':sum(x['future_outcome_conditions'] for x in labels)}


def measurements(labels, traces):
    """One paired policy, all registered inputs. Unknown selections are not errors."""
    byref={x['case_ref']:x for x in labels}
    if len(byref)!=len(labels) or len(traces)!=len(labels) or {t['case_ref'] for t in traces}!=set(byref):
        raise ValueError('FULL_PAIRED_COHORT_REQUIRED')
    if len({t['condition'] for t in traces})!=1:raise ValueError('ONE_POLICY_REQUIRED')
    n=len(labels);executed=selected=abstained=hits=unknown_selections=invalid=positive_selected=0
    for t in traces:
        if t['execution_status']!='EXECUTED':continue
        executed+=1;d=t['decision'];choice=d.get('selected_specialty');status=d.get('decision_status')
        if status=='selected' and choice in FIXED.values():
            selected+=1;l=byref[t['case_ref']]
            positive_selected+=int(bool(l['confirmed_positive_domains']))
            if choice in l['confirmed_positive_domains']:hits+=1
            else:unknown_selections+=1
        elif status=='abstained' and choice is None:abstained+=1
        else:invalid+=1
    def rate(a,b):return {'numerator':a,'denominator':b,'value':a/b if b else None,'status':'COMPUTED' if b else 'NOT_APPLICABLE'}
    unavailable={'status':'NOT_EVALUATED','value':None,'reason':'NO_COMPLETE_ACCEPTED_SPECIALTY_REFERENCE; UNKNOWN_IS_NOT_NEGATIVE'}
    return {'registered_inputs':n,'executed':executed,'missing_utility':n-executed,
        'selected':selected,'abstained':abstained,'invalid':invalid,
        'selection_coverage':rate(selected,executed),'abstention_rate':rate(abstained,executed),
        'confirmed_positive_tag_match_count':hits,'selected_with_positive_reference':positive_selected,
        'unconfirmed_selection_count':unknown_selections,
        'unknown_no_positive_reference_count':sum(not x['confirmed_positive_domains'] for x in labels),
        'source_label_coverage':rate(sum(bool(x['confirmed_positive_domains']) for x in labels),n),
        'source_domain_agreement':deepcopy(unavailable),'selective_domain_agreement':deepcopy(unavailable),
        'domain_disagreement_count':deepcopy(unavailable),'clinical_metrics':'NOT_EVALUATED'}
