"""Separate PROPOSED synthetic snapshot admission, never a v1 release grant.

No clinician/reviewer, clinical release, or source_check PASS is invented.
Exact operator authorization of this distinct policy is required at execution.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import math
import re

from app.locdi.llm_setup import digest
from app.locdi.synthea_diagnostic_snapshot import source_check
from app.locdi.synthea_diagnosis_audit import normalize, proxy_minute

POLICY = 'SYNTHEA_STATIC_RECORDED_TIME_PROXY.v2'


def source_key(event):
    return (event['patient_id'], event['encounter_id'], str(event['target']['code']),
            event['target']['description'], event['timestamp'], event['evidence'].get('units'))


def row_checks(event, *, patient, visit, index_ms, cutoff, source_row_matches, catalog):
    record = event.get('source_metadata', {}).get('source_record', {})
    ev = event.get('evidence', {}); target = event.get('target', {})
    try:
        value_ok = not isinstance(ev.get('value'), bool) and math.isfinite(float(ev['value']))
    except (TypeError, ValueError, KeyError):
        value_ok = False
    # These candidates have second precision. Verify, rather than trusting a
    # caller-controlled epoch field which could relabel future observations.
    bounds = False
    try:
        text = event['timestamp']
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', text):
            low = int(datetime.fromisoformat(text.replace('Z','+00:00')).timestamp()*1000)
            bounds = (event.get('timing_precision') == 'second'
                and event.get('timestamp_lower_epoch_ms') == low
                and event.get('timestamp_upper_epoch_ms') == low+999)
    except (KeyError, ValueError, TypeError):
        pass
    description = normalize(target.get('description')) or ''
    phrases = [normalize(x.removesuffix(' (disorder)')) for x in catalog]
    literal = any(x and re.search(r'(?<!\w)'+re.escape(x)+r'(?!\w)', description) for x in phrases)
    result = {
        'patient_visit_match': event.get('patient_id') == patient == record.get('PATIENT')
            and event.get('encounter_id') == visit == record.get('ENCOUNTER'),
        'raw_csv_row_exact': source_row_matches is True,
        'numeric_objective_storage': event.get('event_type') == 'observations'
            and ev.get('kind') == 'recorded_observation'
            and ev.get('category') in ('laboratory','vital-signs')
            and record.get('TYPE') == 'numeric' and value_ok,
        'pain_not_objective': str(target.get('code')) != '72514-3',
        'source_fields_match': record.get('CODE') == str(target.get('code'))
            and record.get('DESCRIPTION') == target.get('description')
            and record.get('VALUE') == ev.get('value')
            and record.get('UNITS') == ev.get('units')
            and record.get('CATEGORY') == ev.get('category')
            and record.get('DATE') == event.get('timestamp'),
        'verified_timestamp_bounds': bounds,
        'recorded_proxy_within_cutoff': bounds and proxy_minute(event,index_ms) <= cutoff,
        'no_catalog_diagnosis_literal': not literal,
    }
    return {'checks': result, 'structural_candidate': all(result.values()),
        'strict_v1_status': source_check(event,patient=patient,visit=visit,index_ms=index_ms,cutoff=cutoff),
        'clinical_release_status':'NOT_REVIEWED', 'post_diagnosis_intraday_status':'NOT_VERIFIED',
        'clinical_semantic_leakage_status':'NOT_REVIEWED',
        'source_fingerprint':digest(event)}


def resolve_duplicates(rows):
    """Common source-only rule: remove exact copies; exclude ALL value conflicts.

    Input rows must already be structurally qualified. Reference labels are
    neither accepted nor used. Original rows and provenance are never deleted.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[source_key(row['source_event'])].append(row)
    retained=[]; audit=[]
    for key, group in groups.items():
        values={r['source_event']['evidence']['value'] for r in group}
        ordered=sorted(group,key=lambda r:(r['source_event']['source_metadata']['source_file'],
            r['source_event']['source_metadata']['source_row'],r['source_event_id']))
        for i,row in enumerate(ordered):
            status = 'CONFLICTING_SAME_KEY_EXCLUDED' if len(values)>1 else 'PROPOSED_KEEP' if i==0 else 'IDENTICAL_DUPLICATE_NOT_REPEATED'
            audit.append({'patient_ordinal':row['patient_ordinal'],'source_event_id':row['source_event_id'],
                          'status':status,'source_fingerprint':digest(row['source_event'])})
            if status=='PROPOSED_KEEP':retained.append(deepcopy(row))
    retained.sort(key=lambda r:(r['patient_ordinal'],str(r['source_event']['target']['code']),
        r['source_event']['timestamp'],str(r['source_event']['evidence'].get('units')),
        r['source_event']['source_metadata']['source_row']))
    return retained,audit


def draft_observation(base, rows):
    """OFFLINE DRAFT only; no grant, Source View, model, or admission assertion."""
    result=deepcopy(base)
    for row in rows:
        e=row['source_event']
        result['objective_findings'].append(
            '%s [%s]: %s %s (recorded_at=%s)' % (e['target']['description'],e['target']['code'],
                e['evidence']['value'],e['evidence'].get('units') or '',e['timestamp']))
    return result


def verify_authorization(record, plan):
    """Only distinct synthetic development permission, never v1 clinical PASS."""
    if not isinstance(record,dict):raise ValueError('NEW_DEVELOPMENT_POLICY_AUTHORIZATION_REQUIRED')
    required={'accept_recorded_time_proxy','accept_intraday_diagnosis_order_unknown',
        'accept_reference_ordinal12_16_next_day_limit','authorize_new_snapshot_policy',
        'authorize_fixed_D2_D4_generation'}
    if (record.get('evidence_type')!='USER_EXPLICIT'
            or not isinstance(record.get('user_statement'),str) or not record['user_statement'].strip()
            or record.get('plan_fingerprint')!=plan['fingerprint']
            or record.get('policy')!=POLICY or record.get('scope')!='SYNTHETIC_STATIC_D2_D4_ONLY'
            or not all(record.get(k) is True for k in required)
            or record.get('clinical_release_approved') is not False
            or record.get('clinical_reviewer_created') is not False
            or record.get('strict_v1_grants_created') != 0):
        raise ValueError('EXACT_SYNTHETIC_POLICY_AND_EXECUTION_AUTHORIZATION_REQUIRED')
    try:
        t=datetime.fromisoformat(record['recorded_at'])
        if t.tzinfo is None or t>datetime.now(timezone.utc):raise ValueError()
    except (KeyError,ValueError,TypeError):
        raise ValueError('ACTUAL_AUTHORIZATION_RECORD_TIME_REQUIRED') from None
    return True
