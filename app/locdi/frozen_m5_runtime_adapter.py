"""C4-37 exact Source bindings and an uninstalled released-category view.

No Gate bypass, no probability changes, no policy selection or clinical claim.
"""
from copy import deepcopy
import json
import math
from app.core.information_outcomes import InformationOutcomes
from app.core.clinical_observation import prepare_encounter
from app.locdi.voi_information_improvement import bucket
from app.locdi.llm_setup import digest
from validation.artifact_readiness import validate_voi_artifact

SYSTEM = 'CSV_OBSERVATIONS_SYSTEM_UNSPECIFIED'
M5_SHA = '7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c'


def bind_value(model, visit, key):
    """Return a category only for an exact bound existing numeric Source row."""
    if key not in model['provenance']['action_key_mapping']:
        raise ValueError('NOT_SUPPORTED')
    if key not in visit['values']:
        raise ValueError('SOURCE_MISSING')
    value=visit['values'][key]; raw=value['source_record']
    kind, system, code, unit, resource=json.loads(key)
    if raw.get('UNITS') != unit:
        raise ValueError('UNIT_MISMATCH')
    if (system != SYSTEM or raw.get('SYSTEM',SYSTEM) not in ('',SYSTEM)
            or raw.get('CODE') != code or resource != 'Observation'):
        raise ValueError('SEMANTIC_MISMATCH')
    if raw.get('PATIENT') != visit['patient_id'] or raw.get('ENCOUNTER') != visit['visit_id']:
        raise ValueError('SOURCE_IDENTITY_MISMATCH')
    if digest(raw) != value['source_record_fingerprint']:
        raise ValueError('SOURCE_FINGERPRINT_MISMATCH')
    if raw.get('TYPE','').lower() != 'numeric' or value.get('data_type') != 'numeric':
        raise ValueError('SEMANTIC_MISMATCH')
    number=float(raw['VALUE'])
    if (not math.isfinite(number) or type(value['value']) not in (int,float)
            or not math.isfinite(value['value']) or number!=value['value']):
        raise ValueError('SOURCE_VALUE_MISMATCH')
    if value['availability_basis']!='RECORDED_TIME_PROXY' or type(value['available_minute']) is not int or not 0<=value['available_minute']<=45:
        raise ValueError('SOURCE_TIME_CONTRACT')
    if value['recorded_at']!=raw['DATE'] or value.get('actual_available_at') is not None:
        raise ValueError('SOURCE_TIME_CONTRACT')
    aid=model['provenance']['action_key_mapping'][key]
    action=next(a for a in model['actions'] if a['id']==aid)
    if action['type'] != {'Ask':'ask_patient','Test':'request_test'}.get(kind):
        raise ValueError('SEMANTIC_MISMATCH')
    outcome_id=bucket(number,model['provenance']['binning_thresholds'][key])
    outcome=next(o for o in action['outcomes'] if o['id']==outcome_id)
    return {'status':'ADAPTER_COMPATIBLE','source_identity':'EXACT_BINDING','action_id':aid,
        'type':action['type'],'target':action['target'],'source_key':key,
        'outcome_id':outcome_id,'observation':deepcopy(outcome['observation']),
        'raw_value':raw['VALUE'],'numeric_value':number,'available_minute':value['available_minute'],
        'availability_basis':value['availability_basis'],'source_record_fingerprint':value['source_record_fingerprint']}


class ReleasedCategoryView:
    """Standalone contract utility, NOT installed as Hospital InformationPolicy.

    An explicit selected action and release time are mandatory. The categorical
    text view retains exactly the stored category semantics, not raw text as a
    disease-specific confirmation. Full runtime execution still requires Gate.
    """
    def __init__(self,model,visit):
        self.model=deepcopy(model);self.visit=deepcopy(visit);self.used=set();self.released=[]
        InformationOutcomes(model)
    def release(self,action_id,*,selected,minute):
        if selected!=action_id:raise ValueError('OUTCOME_BEFORE_SELECTION')
        if action_id in self.used:raise ValueError('REPEATED_ACTION')
        key=next((k for k,a in self.model['provenance']['action_key_mapping'].items() if a==action_id),None)
        if key is None:raise ValueError('NOT_SUPPORTED')
        row=bind_value(self.model,self.visit,key)
        if type(minute) not in (int,float) or not math.isfinite(minute) or minute<row['available_minute']:
            raise ValueError('FUTURE_SOURCE_OBSERVATION')
        self.used.add(action_id);self.released.append(row)
        return deepcopy(row)
    def observation(self):
        result={'history':[],'objective_findings':[]}
        for row in self.released:
            if row['type']=='ask_patient':
                result['history'].append({'question':row['target'],'answer':row['observation']['text'],'answer_status':'known'})
            else:result['objective_findings'].append(row['observation']['text'])
        return result


def candidate_case(model,visit,past_visits=()):
    """Engine-private candidate with empty unknown initial fields, not a validated LoCDI wrapper."""
    events=[]
    for key in model['provenance']['action_key_mapping']:
        row=bind_value(model,visit,key)
        # Raw source values remain explicit. No aliasing a category to a
        # supported clinical diagnosis, no Reference in the engine candidate.
        value=visit['values'][key]
        evidence=json.dumps({'code':json.loads(key)[2],'system':SYSTEM,'unit':json.loads(key)[3],
            'value':value['source_record']['VALUE'],'type':value['source_record']['TYPE']},sort_keys=True)
        events.append({'event_id':value['event_id'],'action':'Ask' if row['type']=='ask_patient' else 'Test',
            'target':row['target'],'available_minute':row['available_minute'],'evidence':evidence})
    candidate={'case_id':visit['case_id'],'index_encounter_id':visit['visit_id'],
        'index_encounter_start':visit['decision_time'],
        'longitudinal':{'patient_id':visit['patient_id'],'visit_id':visit['visit_id'],'started_at':visit['decision_time']},
        'current_initial_state':{'doctor_visible_info':{},'patient_visible_info':{}},
        'past_visits':deepcopy(list(past_visits)),'current_encounter_events':events}
    prepare_encounter(candidate)
    return candidate


def runtime_readiness(model):
    gate=validate_voi_artifact(model)
    return {'status':'RUNTIME_CONTRACT_BLOCKED','source_action_adapter':'ADAPTER_COMPATIBLE_IF_EXACT_SOURCE_CHECKS_PASS',
        'information_outcomes_schema':'PASS','artifact_gate':gate,
        'hospital_execute_allowed':False,'category_view_installed':False,
        'source_schema':'engine-private prepare_encounter candidate, NOT validated LoCDI full wrapper',
        'blocker_codes':['FROZEN_ENCODING_OUTSIDE_EXISTING_RUNTIME_GATE',
            'RAW_EVENT_TEXT_NOT_CANONICAL_OUTCOME_OBSERVATION','NO_MODEL_OUTCOME_FINALIZATION_SUPPORT',
            'NO_SOURCE_TEST_DURATION_OR_RESOURCE_MEASUREMENTS','NEW_SOURCE_RUNTIME_APPROVAL_NOT_PROVIDED']}
