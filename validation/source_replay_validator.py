"""D3 canonical/projection boundary, independent of clinical correctness."""
import json
from app.core.longitudinal_memory import timestamp


def validate_source_bundle(bundle):
    p=bundle['private'];cases=p['cases.jsonl'];initial={r['case_id']:r for r in p['initial_observations.jsonl']}
    memories={r['memory_id']:r for r in p['memory.json']['records']}
    actions={(r['case_id'],r['action_id']):r for r in p['action_events.jsonl']}
    references=p['reference_candidates.jsonl'];wrappers={w['case_id']:w for w in p['generated_proxy_wrappers.jsonl']}
    if len(memories)!=len(p['memory.json']['records']):raise ValueError('DUPLICATE_MEMORY')
    if len({c['patient_id'] for c in cases})!=len(cases):raise ValueError('PATIENT_INDEX_DUPLICATED')
    counts={'cases':len(cases),'memory_case_references':0,'actions':len(actions),'violations':0,
            'reference_candidates':len(references),'clinically_approved':0}
    for c in cases:
        current=initial[c['case_id']];cutoff=timestamp(c['decision_time']);w=wrappers[c['case_id']]
        if (current['chief_complaint'] is not None or current['initial_vitals'] or current['objective_findings'] or current['current_symptoms']):
            raise ValueError('UNKNOWN_INITIAL_AVAILABILITY_EXPORTED')
        if current['patient_id']!=c['patient_id'] or w['patient_id']!=c['patient_id']:raise ValueError('PATIENT_MISMATCH')
        if w['agent_visible']['current_initial_state']['patient_visible_info']['associated_symptoms']:
            raise ValueError('ENCODED_SYMPTOMS_PROMOTED')
        if w['environment']['data_replay']['observed_release_time'] is not False:raise ValueError('PROXY_PROMOTED')
        for mid in c['memory_ids']:
            m=memories[mid];s=m['source']['source_record'];counts['memory_case_references']+=1
            if (m['patient_id']!=c['patient_id'] or s['PATIENT']!=c['patient_id'] or s['ENCOUNTER']!=m['visit_id']
                    or m['visit_id']==c['visit_id']):raise ValueError('MEMORY_ISOLATION')
            if not timestamp(m['visit_started_at'])<=timestamp(m['observed_at'])<=timestamp(m['proxy_available_at'])<=cutoff:
                raise ValueError('MEMORY_CUTOFF')
            if timestamp(m['visit_started_at'])>=cutoff:raise ValueError('CURRENT_OR_FUTURE_VISIT_MEMORY')
            if m['available_at'] is not None or m['availability_unknown'] is not True:raise ValueError('MEMORY_AVAILABILITY_PROMOTED')
            if json.loads(m['text'])!={k:s[k] for k in ('CATEGORY','CODE','DESCRIPTION','VALUE','UNITS','TYPE')}:
                raise ValueError('MEMORY_CONTENT_CHANGED')
        for aid in c['action_ids']:
            a=actions[c['case_id'],aid];r=a['source']['source_record']
            if a['patient_id']!=c['patient_id'] or r['PATIENT']!=c['patient_id'] or r['ENCOUNTER']!=c['visit_id']:
                raise ValueError('ACTION_SOURCE_IDENTITY')
            if a['action'] not in {'Ask','Test'}:raise ValueError('CONSULT_UNSUPPORTED')
            if (a['value']!=r['VALUE'] or a['units']!=r['UNITS'] or a['code']!=r['CODE']
                    or a['recorded_at']!=r['DATE']):raise ValueError('ACTION_SOURCE_VALUE_CHANGED')
            available=a['proxy_available_minute']
            if type(available) is not int or available<1 or (timestamp(a['recorded_at'])-cutoff).total_seconds()+0.999>available*60:
                raise ValueError('FUTURE_RESULT_RELEASE')
            if a['actual_available_at'] is not None or a['release_assumption_id'] is None:
                raise ValueError('ACTION_AVAILABILITY_PROMOTED')
        if any(v is not None for v in w['task_annotations']['gold'].values()) or any(w['task_annotations']['metric_readiness'].values()):
            raise ValueError('GOLD_PROMOTION')
        if any(w['evaluation'].values()) or w['provenance']['source_case_audit']['diagnosis_sources']:
            raise ValueError('REFERENCE_IN_WRAPPER')
    for ref in references:
        if ref['independent_review_status']!='NOT_REVIEWED' or ref['clinical_gold_status']!='UNAPPROVED' or ref['correct'] is not None:
            raise ValueError('UNREVIEWED_REFERENCE_PROMOTION')
    for policy in ('generated_proxy_wrappers.jsonl','generated_strict_wrappers.jsonl'):
        for w in p[policy]:
            if any(e['action']=='Consult' for e in w['environment']['current_encounter_events']):raise ValueError('CONSULT_UNSUPPORTED')
            if policy=='generated_strict_wrappers.jsonl' and (w['agent_visible']['past_visits'] or w['environment']['current_encounter_events']):
                raise ValueError('STRICT_UNKNOWN_AVAILABILITY')
    manifest=p['patient_split_manifest.json']
    if any(r['assigned_split'] is not None for r in manifest['patients']):raise ValueError('UNAPPROVED_SPLIT_ASSIGNMENT')
    old=set().union(*(set(v) for v in manifest['legacy_patient_sets'].values()))
    if {c['patient_id'] for c in cases}&old:raise ValueError('LEGACY_PATIENT_OVERLAP')
    return counts
