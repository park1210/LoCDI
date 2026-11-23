"""Opt-in synthetic sequential source admission. No v1 grant or Core changes."""
from collections import defaultdict
from copy import deepcopy
import json
import random

from app.locdi.llm_setup import digest
from app.locdi.synthea_diagnostic_snapshot import source_check,source_view
from app.locdi.synthea_development_proxy import row_checks
from app.core.longitudinal_memory import timestamp

POLICY='SYNTHEA_SEQUENTIAL_RECORDED_TIME_PROXY.v1'
B012_POLICY='SYNTHEA_SEQUENTIAL_RECORDED_TIME_PROXY_B012.v1'
TAT=5


def stratified_panel(candidates, seed=42):
    """Selection accepts only ordinal/stratum metadata, never D4 outcomes."""
    if any(set(x)!={'patient_ordinal','stratum'} for x in candidates):raise ValueError('SELECTION_METADATA_ONLY_NO_OUTCOMES')
    groups=defaultdict(list)
    for x in candidates:groups[x['stratum']].append(x['patient_ordinal'])
    if set(groups)!={'endocrinology','neurology'} or any(len(x)!=10 for x in groups.values()):
        raise ValueError('EXPECTED_FIXED20_TWO_STRATA10_EACH')
    if len({x['patient_ordinal'] for x in candidates})!=20:raise ValueError('DUPLICATE_PATIENT')
    rng=random.Random(seed);chosen=[]
    for name in ('endocrinology','neurology'):chosen.extend(rng.sample(sorted(groups[name]),5))
    return sorted(chosen)


def policy_contract(rows, *, test_only=False, allow_full_history=False):
    spec={'policy':B012_POLICY if allow_full_history else POLICY,'source_rows_fingerprint':digest(rows),'source_rows':len(rows),
        'source_record_cutoff_minute':45,'turnaround_minutes':TAT,'resource_occupancy_minutes':TAT,
        'synthetic_available_minute':'ceil(recorded_timestamp_upper-index); source gate only, not clinical TAT',
        'result_rule':'ready=max(test_started_minute+5,source_available_minute)',
        'scope':'B0_B1_B2_ER_ONLY_VOI_OFF' if allow_full_history else 'B0_B2_ER_ONLY_VOI_OFF','clinical_release_approved':False,'v1_grants_created':0,
        'candidate_wire_projection':'action,target only; omit redundant action_id/type and null cost/outcome_model; no clinical data truncation',
        'test_only':test_only}
    spec['fingerprint']=digest(spec)
    return spec


def validate_permission(record,spec,plan_fingerprint,*,test_only):
    if (not isinstance(record,dict) or record.get('scope')!='SYNTHETIC_SEQUENTIAL_ER_PILOT_ONLY'
        or record.get('plan_fingerprint')!=plan_fingerprint or record.get('policy_fingerprint')!=spec['fingerprint']
        or record.get('evidence_type')!=('TEST_ONLY' if test_only else 'USER_EXPLICIT')
        or not isinstance(record.get('user_statement'),str) or not record['user_statement'].strip()
        or record.get('accept_source_record_proxy_cutoff') is not True
        or record.get('accept_fixed_five_minute_synthetic_turnaround') is not True
        or record.get('authorize_sequential_source_and_pilot') is not True
        or record.get('accept_reference_timing_limits') is not True
        or record.get('clinical_release_approved') is not False or record.get('v1_grants_created')!=0):
        raise ValueError('NEW_SEQUENTIAL_POLICY_AND_PILOT_AUTHORITY_REQUIRED')


class SequentialSyntheticSourceView:
    """Validated post-adapter overlay, strict default OFF, no arbitrary callback."""
    def __init__(self,rows,*,spec,authorization,plan_fingerprint,catalog):
        self._allow_full_history=spec.get('policy')==B012_POLICY
        expected=policy_contract(rows,test_only=spec.get('test_only') is True,allow_full_history=self._allow_full_history)
        if spec!=expected:raise ValueError('SEQUENTIAL_SOURCE_POLICY_CHANGED')
        validate_permission(authorization,spec,plan_fingerprint,test_only=spec['test_only'])
        if not spec['test_only'] and len(rows)!=345:raise ValueError('EXACT_FROZEN345_REQUIRED')
        self._rows=deepcopy(rows);self._spec=deepcopy(spec);self._catalog=deepcopy(catalog)

    def metadata(self):
        return {'policy':self._spec['policy'],'fingerprint':self._spec['fingerprint'],'test_only':self._spec['test_only'],
            'turnaround_minutes':TAT,'clinical_release_approved':False,'v1_grants_created':0,
            'candidate_wire_projection':'lossless clinical data; action,target pairs unchanged'}

    @staticmethod
    def compact_payload(payload):
        result=deepcopy(payload)
        for c in result.get('candidate_actions',[]):
            if c.get('cost') is not None or c.get('outcome_model') is not None:
                raise ValueError('NON_NULL_VOI_MODEL_CANNOT_BE_COMPACTED')
        result['candidate_actions']=[{'action':c['action'],'target':c['target']} for c in result.get('candidate_actions',[])]
        return result

    def apply(self, adapted, *, features,purpose):
        if self._spec['test_only'] and purpose!='development_smoke':raise ValueError('TEST_ONLY_NEVER_LIVE')
        modes=('none','selective','full_history') if self._allow_full_history else ('none','selective')
        if (features['memory_mode'] not in modes or features['top_k'] not in (None,5)
            or features['memory_mode']=='full_history' and features['top_k'] is not None
            or features['memory_mode']=='selective' and features['top_k']!=5
            or features['routing_mode']!='er_only' or features['voi_enabled'] is not False
            or features['action_selector']!='specialist_heuristic'):
            raise ValueError('SEQUENTIAL_VIEW_ONLY_B0_B2_ER_ONLY_VOI_OFF')
        original=adapted.runtime_case
        # Reuse the old helper's copy-only OFF path; its strict grant-enabled
        # branch is neither altered nor made to pass by synthetic permission.
        runtime=source_view(original,[],{},enabled=False)
        identity=runtime['longitudinal'];index_ms=int(timestamp(identity['started_at']).timestamp()*1000)
        sources={e['event_id']:e for e in adapted.environment['source_event_store']}
        names=set(runtime['available_tests']);added=[]
        for row in self._rows:
            e=row['source_event']
            if e['patient_id']!=identity['patient_id'] or e['encounter_id']!=identity['visit_id']:continue
            if e['event_id'] not in sources or digest(e)!=digest(sources[e['event_id']]):raise ValueError('SOURCE_EVENT_NOT_EXACT_ORIGINAL')
            facts=row_checks(e,patient=identity['patient_id'],visit=identity['visit_id'],index_ms=index_ms,
                cutoff=45,source_row_matches=True,catalog=self._catalog)
            if not facts['structural_candidate']:raise ValueError('SEQUENTIAL_SOURCE_STRUCTURAL_REJECTED')
            strict=source_check(e,patient=identity['patient_id'],visit=identity['visit_id'],index_ms=index_ms,cutoff=45)
            if strict!='RELEASE_NOT_REVIEWED':raise ValueError('UNEXPECTED_V1_SOURCE_STATE')
            # Full source description preserved. Ordinal disambiguates units/
            # repeated records without inserting a diagnosis or patient ID.
            label='%s [%s] (source %02d)'%(e['target']['description'],e['target']['code'],len(added)+1)
            if len(label)>240 or label in names:raise ValueError('INVALID_UNIQUE_SOURCE_TEST_NAME')
            names.add(label)
            available=row['proxy_available_minute']
            if available!=max(0,__import__('math').ceil((e['timestamp_upper_epoch_ms']-index_ms)/60000)):
                raise ValueError('SOURCE_AVAILABLE_OFFSET_MISMATCH')
            event={'event_id':'sequential-source:'+digest(e),'action':'Test','target':label,
                'available_minute':available,'evidence':json.dumps({
                    'value':e['evidence']['value'],'units':e['evidence'].get('units'),
                    'recorded_at':e['timestamp']},ensure_ascii=False,sort_keys=True,separators=(',',':'))}
            runtime['current_encounter_events'].append(event);runtime['available_tests'].append(label)
            profile=runtime.setdefault('resource_profile',{})
            profile.setdefault('test_turnaround',{})[label]=TAT
            profile.setdefault('test_occupancy',{})[label]=TAT
            profile.setdefault('test_resources',{})[label]='lab_slot'
            added.append({'test_name':label,'source_fingerprint':digest(e),'available_minute':available,
                'source_cutoff_minute':45,'synthetic_turnaround_minute':TAT,'strict_v1_status':strict})
        return runtime,added
