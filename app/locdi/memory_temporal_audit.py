"""Analysis-only temporal facts; never selects records or judges clinical relevance."""
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest
from app.locdi.memory_concept_evaluation import concept_key


def measurement(record, reference, identity):
    if reference['original_record_fingerprint'] != digest(record):
        raise ValueError('SOURCE_RECORD_FINGERPRINT')
    cutoff = timestamp(identity['started_at'])
    if (record['patient_id'] != identity['patient_id'] or record['visit_id'] == identity['visit_id']
            or timestamp(record['visit_started_at']) >= cutoff
            or timestamp(record['available_at']) > cutoff
            or timestamp(record['observed_at']) > timestamp(record['available_at'])):
        raise ValueError('MEMORY_BOUNDARY')
    key, status = concept_key(reference['source_alignment'], reference['source_type'])
    native = reference['source_alignment'].get('native') or {}
    start = native.get('start') if key else None
    if start is not None and (type(start) not in (int, float) or not math.isfinite(start)):
        raise ValueError('INVALID_SOURCE_TIMESTAMP')
    when = datetime.fromtimestamp(start / 1000, timezone.utc) if start is not None else None
    if when is not None and when > cutoff:
        raise ValueError('FUTURE_SOURCE_MEASUREMENT')
    value = native.get('native_value') if key else None
    # JSON type and exact source value retained; no text parsing or unit conversion.
    value_key = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) if value is not None else None
    return {'memory_id': record['memory_id'], 'concept_key': list(key) if key else None,
        'mapping_status': status, 'measurement_epoch_ms': start,
        'measurement_time': when.isoformat() if when else None,
        'available_at_proxy': record['available_at'], 'actual_available_at': None,
        'observed_at_proxy': record['observed_at'], 'value': value, 'value_key': value_key,
        'unit': key[2] if key else None, 'encounter': record['visit_id'],
        'source_pointer': native.get('pointer'), 'source_record_fingerprint': digest(record)}


def group(rows):
    out = defaultdict(list)
    for r in rows:
        if r['concept_key'] is not None: out[tuple(r['concept_key'])].append(r)
    return out


def facts(rows):
    times = {r['measurement_epoch_ms'] for r in rows if r['measurement_epoch_ms'] is not None}
    values = {r['value_key'] for r in rows if r['value_key'] is not None}
    timed = [r for r in rows if r['measurement_epoch_ms'] is not None]
    # Equal timestamps with conflicting values do not define a trend.
    ordered = sorted(timed, key=lambda r: r['measurement_epoch_ms'])
    bytime = defaultdict(set)
    for r in timed: bytime[r['measurement_epoch_ms']].add(r['value_key'])
    numeric = all(type(r['value']) in (int, float) and math.isfinite(r['value']) for r in timed)
    changes = None
    if len(times) > 1 and numeric and all(len(v) == 1 for v in bytime.values()):
        distinct = {r['measurement_epoch_ms']: r['value'] for r in ordered}
        vs = list(distinct.values())
        changes = {'increases': sum(b > a for a,b in zip(vs,vs[1:])),
                   'decreases': sum(b < a for a,b in zip(vs,vs[1:])),
                   'equal': sum(b == a for a,b in zip(vs,vs[1:]))}
    return {'records': len(rows), 'distinct_times': len(times), 'distinct_values': len(values),
        'missing_timestamp': len(rows)-len(timed), 'missing_value': sum(r['value_key'] is None for r in rows),
        'multi_timepoint': len(times)>1, 'span_seconds': (max(times)-min(times))/1000 if times else None,
        'repeated_value_records': sum(r['value_key'] is not None for r in rows)-len(values),
        'duplicate_timestamp_records': len(timed)-len(times),
        'time_ordered_numeric_changes': changes}


def compare(old, new, eligible):
    a,b,e = group(old),group(new),group(eligible)
    details=[]
    for key in sorted(set(a)|set(b)):
        ar,br,er=a.get(key,[]),b.get(key,[]),e.get(key,[])
        at={r['measurement_epoch_ms'] for r in ar if r['measurement_epoch_ms'] is not None}
        bt={r['measurement_epoch_ms'] for r in br if r['measurement_epoch_ms'] is not None}
        av={r['value_key'] for r in ar if r['value_key'] is not None}
        bv={r['value_key'] for r in br if r['value_key'] is not None}
        et={r['measurement_epoch_ms'] for r in er if r['measurement_epoch_ms'] is not None}
        details.append({'concept_key':list(key), 'baseline':facts(ar),'diverse':facts(br),
            'lost_times':len(at-bt),'added_times':len(bt-at),
            'lost_values':len(av-bv),'added_values':len(bv-av),
            'new_concept':not ar,'dropped_concept':not br,
            'earliest_baseline_retained': min(at) in bt if at else None,
            'latest_baseline_retained': max(at) in bt if at else None,
            'eligible_earliest_baseline': min(et) in at if et and ar else None,
            'eligible_earliest_diverse': min(et) in bt if et and br else None,
            'eligible_latest_baseline': max(et) in at if et and ar else None,
            'eligible_latest_diverse': max(et) in bt if et and br else None})
    modes={}
    for name,rs,gs in [('baseline',old,a),('diverse',new,b)]:
        fs=[facts(g) for g in gs.values()]
        modes[name]={'records':len(rs),'mapped_records':sum(len(g) for g in gs.values()),
            'distinct_concepts':len(gs), 'records_per_concept':[len(g) for g in gs.values()],
            **{k:sum(f[k] for f in fs) for k in ('distinct_times','distinct_values','missing_timestamp',
                'missing_value','repeated_value_records','duplicate_timestamp_records','multi_timepoint')},
            'unmapped_records':sum(r['concept_key'] is None for r in rs),
            'max_span_seconds':max([f['span_seconds'] for f in fs if f['span_seconds'] is not None],default=None)}
    shared=[d for d in details if not d['new_concept'] and not d['dropped_concept']]
    summary={'modes':modes, **{k:sum(d[k] for d in details) for k in ('lost_times','added_times','lost_values','added_values')},
        'new_concepts':sum(d['new_concept'] for d in details),'dropped_concepts':sum(d['dropped_concept'] for d in details),
        'shared_concepts':len(shared),'shared_concepts_losing_times':sum(d['lost_times']>0 for d in shared),
        'shared_concepts_losing_values':sum(d['lost_values']>0 for d in shared)}
    for field in ('earliest_baseline_retained','latest_baseline_retained','eligible_earliest_baseline',
                  'eligible_earliest_diverse','eligible_latest_baseline','eligible_latest_diverse'):
        vals=[d[field] for d in details if d[field] is not None]
        summary[field]={'numerator':sum(vals),'denominator':len(vals),'value':sum(vals)/len(vals) if vals else None}
    return summary,details


def runtime_fragment(records, descriptors, ids):
    """Exact row shape and role_call JSON options; fragment only, not full wire."""
    if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_TRACE_ID')
    byid={r['memory_id']:r for r in records}; cs={r['memory_id']:r for r in descriptors}
    result=[]
    for mid in ids:
        row=deepcopy(byid[mid]);row.update(status_as_of_visit='historical',retrieval_score=cs[mid]['score'])
        result.append(row)
    return result,json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False)
