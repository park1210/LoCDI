"""Private lexical scoring and explicit denominators; real C3-12 scores forbidden."""
from collections import Counter
import re
import statistics
import unicodedata
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.llm_setup import digest

SCORING_VERSION = 'agentclinic.lexical_scoring.v1'


def load_private_keys(path, sha256):
    from app.locdi.agentclinic_scenario_schema import ScoringKey
    value = pinned_private_json(path, sha256)
    return tuple(ScoringKey(r['scenario_ref'], r['record_fingerprint'], r['correct_diagnosis'])
                 for r in value['keys'])


def normalize(text):
    if not isinstance(text, str):
        raise ValueError('PREDICTION_NOT_TEXT')
    text = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text).strip()).casefold()
    return text[:-1].rstrip() if text.endswith('.') else text


class ApprovedAliases:
    def __init__(self, path, sha256, *, scope):
        record = pinned_private_json(path, sha256)
        if (record.get('schema_version') != 'agentclinic.alias_policy.v1'
                or record.get('status') != 'APPROVED' or record.get('scope') != scope
                or not all(record.get(k) for k in ('reviewed_by','reviewed_at','procedure_version'))):
            raise ValueError('APPROVED_ALIAS_EVIDENCE_REQUIRED')
        self.mapping, self.sha256, self.scope = {}, sha256, scope
        rows = record.get('mapping')
        if not isinstance(rows,list) or not rows:
            raise ValueError('ALIAS_MAPPING_REQUIRED')
        for row in rows:
            reference = normalize(row['reference'])
            if not reference or not isinstance(row['aliases'],list):
                raise ValueError('ALIAS_MAPPING_SHAPE')
            for alias in row['aliases']:
                value = normalize(alias)
                if not value or value in self.mapping:
                    raise ValueError('ALIAS_DUPLICATE_OR_EMPTY')
                self.mapping[value] = reference


def _score(episode, key, aliases):
    if (episode['scenario_ref'] != key.scenario_id
            or episode['source']['record_fingerprint'] != key.source_record_fingerprint):
        raise ValueError('SCORING_SOURCE_BINDING')
    r = episode['result']; status = r['status']; prediction = r.get('predicted_diagnosis')
    if status not in {'completed','incomplete','failed'}:
        raise ValueError('SCORING_TERMINAL_STATUS')
    state = status
    if status == 'completed':
        if not r.get('completed') or not isinstance(prediction,str) or len(prediction)>4096:
            state = 'invalid_prediction'
        elif not prediction.strip():
            state = 'diagnosis_unresolved'
        elif r.get('diagnosis_unresolved'):
            state = 'invalid_prediction'
    exact = state == 'completed' and normalize(prediction) == normalize(key.correct_diagnosis)
    alias = bool(state == 'completed' and aliases and
                 aliases.mapping.get(normalize(prediction)) == normalize(key.correct_diagnosis))
    return {'scenario_ref':key.scenario_id,'state':state,'exact_match':exact,
            'alias_match':alias,'review_required':state=='completed' and not (exact or alias),
            'clinical_equivalence':'CLINICAL_EQUIVALENCE_NOT_VERIFIED'}


def evaluate_test_only(registered_scenarios, episodes, scoring_keys, *, aliases=None):
    """No live flag: benchmark rows are NOT_EVALUATED even when doctor is Fake."""
    if aliases is not None and not isinstance(aliases, ApprovedAliases):
        raise ValueError('UNREVIEWED_ALIAS_FORBIDDEN')
    registered = list(registered_scenarios)
    if not registered or len({s.scenario_id for s in registered}) != len(registered):
        raise ValueError('PREREGISTERED_COHORT_REQUIRED')
    if any(not s.source.get('test_only') for s in registered):
        return {'status':'NOT_EVALUATED','reason':'REAL_BENCHMARK_SCORING_NOT_AUTHORIZED_C3_12',
                'scoring_version':SCORING_VERSION,'registered_scenarios':len(registered),
                'clinical_metrics':'NOT_EVALUATED','clinical_equivalence':'CLINICAL_EQUIVALENCE_NOT_VERIFIED'}
    scope = digest([s.scenario_id for s in registered])
    if aliases and aliases.scope != scope:
        raise ValueError('ALIAS_COHORT_SCOPE_MISMATCH')
    ids = {s.scenario_id for s in registered}; keys = {k.scenario_id:k for k in scoring_keys}
    records = {e['scenario_ref']:e for e in episodes}
    if len(records)!=len(episodes) or len(keys)!=len(scoring_keys) or set(keys)!=ids or set(records)-ids:
        raise ValueError('SCORING_COHORT_OR_KEY_COMPLETENESS')
    rows = []
    for s in registered:
        if keys[s.scenario_id].source_record_fingerprint != s.source['record_fingerprint']:
            raise ValueError('SCORING_REFERENCE_SOURCE')
        if s.scenario_id not in records:
            rows.append({'scenario_ref':s.scenario_id,'state':'failed','reason':'NOT_EXECUTED',
                         'exact_match':False,'alias_match':False,'review_required':False})
        else:
            rows.append(_score(records[s.scenario_id],keys[s.scenario_id],aliases))
    n = len(rows); states = Counter(r['state'] for r in rows)
    def rate(num, den):
        return {'numerator':num,'denominator':den,'value':num/den if den else None,
                'status':'TEST_ONLY' if den else 'NOT_APPLICABLE'}
    trace = [t for e in episodes for t in e['trace']]
    actions = Counter(t['action'] for t in trace)
    exact = sum(r['exact_match'] for r in rows); completion = states['completed']
    def distribution(field):
        values = [e['result'][field] for e in episodes if type(e['result'].get(field)) is int]
        return {'sum':sum(values),'denominator':len(values),'mean':statistics.mean(values) if values else None,
                'median':statistics.median(values) if values else None,'missing_planned':n-len(values),
                'scope':'All recorded episodes including incomplete/failed; missing episodes not imputed'}
    metrics = {'exact_match_accuracy':rate(exact,n),
               'alias_adjusted_accuracy':rate(sum(r['exact_match'] or r['alias_match'] for r in rows),n)
                  if aliases else {'status':'NOT_EVALUATED_NO_APPROVED_ALIASES','value':None},
               'valid_prediction_exact_accuracy':rate(exact,completion),
               'diagnosis_completion_rate':rate(completion,n),
               'unresolved_rate':rate(states['diagnosis_unresolved'],n),
               'incomplete_rate':rate(states['incomplete'],n),'failed_rate':rate(states['failed'],n),
               'invalid_prediction_rate':rate(states['invalid_prediction'],n),
               'turns':distribution('turn_count'),'action_requests':distribution('action_count'),
               'ask_count':actions['ASK'],'examine_count':actions['EXAMINE'],'test_count':actions['TEST'],
               'invalid_unavailable_action_rate':rate(sum(t['status'] in {'INVALID','UNAVAILABLE','UNSUPPORTED'} for t in trace),len(trace)),
               'repeated_action_rate':rate(sum(bool(t['repeat_of']) for t in trace),
                                          sum(t['action'] in {'ASK','EXAMINE','TEST'} for t in trace)),
               'review_required_count':sum(r['review_required'] for r in rows)}
    return {'status':'TEST_ONLY','scoring_version':SCORING_VERSION,'normalization':
            'NFC, casefold, outer trim, whitespace collapse, remove exactly one trailing period; no medical synonyms',
            'registered_scenarios':n,'cohort_fingerprint':scope,'state_counts':dict(states),
            'metrics':metrics,'rows':rows,'alias_policy_sha256':aliases.sha256 if aliases else None,
            'clinical_metrics':'NOT_EVALUATED','clinical_equivalence':'CLINICAL_EQUIVALENCE_NOT_VERIFIED'}
