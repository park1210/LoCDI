"""C4-09 literature-grounded operational weak labels; never human Gold or clinical fitting."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import validate_candidate, judged_metrics
from app.locdi.memory_reference_evaluation import ratio
from app.core.longitudinal_memory import PatientMemoryStore, timestamp
from scripts.enrich_full_cohort_sources import DATA, write
from scripts.audit_memory_review_progress import check_hashes
from scripts.classify_pre_review_evidence import jsonl

VERSION = 'locdi.memory_silver.c4_09.v1'
A = ('TASK_A_STANDARDIZED_MEASUREMENT', 'same_measurement')
AX = ('TASK_A_LITERAL_DESCRIPTION_UNIT', 'literal_field_identity')
B = ('TASK_B_CLINICAL_DECISION', 'clinical_relevance')
STALE = ('TASK_B_CLINICAL_DECISION', 'stale_current_assertion')
CONFLICT = ('TASK_B_CLINICAL_DECISION', 'same_event_conflict')
TARGETS = (A, AX, B, STALE, CONFLICT)
SEED = 20261109

SOURCES = {
    'S1': {'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5951191/', 'title': 'Ratner et al., Snorkel (2017)',
           'section': 'Abstract; 2.1; 3.1; 5',
           'claim': 'Labeling functions encode heuristics with unknown accuracy/dependence and may abstain; a label model can model noisy sources.',
           'not_implied': 'No LoCDI relevance rule or clinical correctness is supplied.'},
    'S2': {'url': 'https://trec.nist.gov/data/reljudge_eng.html', 'title': 'NIST TREC English Relevance Judgements',
           'section': 'Working relevance definition; collection/qrels matching',
           'claim': 'Relevance is defined for a topic and collection; assessment files must match the evaluated collection.',
           'not_implied': 'TREC report-writing relevance is not a clinical decision reference.'},
    'S3': {'url': 'https://ohdsi.github.io/TheBookOfOhdsi/DataQuality.html', 'title': 'The Book of OHDSI, Chapter 15',
           'section': '15.2, 15.2.2; study-specific data quality',
           'claim': 'Data quality checks include conformance, completeness and plausibility; suitability depends on intended use.',
           'not_implied': 'A passed data-quality check does not establish clinical relevance.'},
    'S4': {'url': 'https://hl7.org/fhir/R4/provenance.html', 'title': 'HL7 FHIR R4 Provenance',
           'section': '6.3.1; occurred[x], recorded, target, entity',
           'claim': 'Provenance links resources, agents and activities; activity occurrence and recording are distinct elements.',
           'not_implied': 'Native Synthea is not automatically FHIR, and recorded time is not verified clinical receipt time.'},
    'S5': {'url': 'https://hl7.org/fhir/R4/medicationstatement.html', 'title': 'HL7 FHIR R4 MedicationStatement',
           'section': '11.4.1-2; status; effective[x]; dateAsserted',
           'claim': 'A medication statement may describe past, current or future use; status, effective time and assertion time differ.',
           'not_implied': 'A prescription, age or absent stop field does not prove a current-state assertion is stale or valid.'},
    'S6': {'url': 'https://www.nist.gov/publications/bias-and-limits-pooling-large-collections',
           'title': 'Buckley et al., Bias and the Limits of Pooling for Large Collections (2007)', 'section': 'Abstract',
           'claim': 'Small pools in large collections can be biased toward documents containing topic-title words.',
           'not_implied': 'No quantitative bias correction or complete judgments for this LoCDI pool are supplied.'},
}


def registry():
    specifications = [
        ('A01_NONMEASUREMENT_SCOPE', A, ['source.type', 'structural_pass'], ['NEGATIVE', 'ABSTAIN'], ['S2', 'S3'],
         'Only native observations are in the declared identical-measurement retrieval target. Other explicit resource types are out of scope.',
         'A negative is only for this narrow task, never clinical irrelevance.'),
        ('A02_EXPLICIT_CODING_MATCH', A, ['source.code/system/description/units/type/value', 'released_facts', 'cutoff'], ['POSITIVE', 'ABSTAIN'], ['S2', 'S3', 'S4'],
         'Exact nonempty namespace+code+description+unit and observation type define operational measurement identity; no crosswalk or conversion.',
         'No terminology validation, version mapping, specimen or clinical equivalence guarantee; missing namespace forces abstention.'),
        ('X01_LITERAL_MATCH', AX, ['source.description/units/type/value', 'released_facts'], ['POSITIVE', 'ABSTAIN'], ['S2', 'S6'],
         'Auxiliary task is exact full description string and exact nonempty unit equality within observations, without asserting concept equivalence.',
         'Strong circularity risk with query text, lexical retrieval and pool construction; not standardized or clinical relevance.'),
        ('X02_LITERAL_NONMATCH', AX, ['source.description/units/type', 'complete_released_fact_set'], ['NEGATIVE', 'ABSTAIN'], ['S2', 'S3'],
         'Only for the fixed literal target: explicit non-observation resources or an observed description outside the complete literal target set are negative.',
         'Missing fields and same description with different units abstain. Synonyms can be clinically related despite a literal negative.'),
        ('B01_CLINICAL_EVIDENCE_GUARD', B, ['chief_complaint', 'clinical_task_reference'], ['ABSTAIN'], ['S1', 'S2', 'S3'],
         'Source identity and field matching are insufficient to establish clinical decision usefulness.',
         'Does not infer clinical usefulness even if complaint text is present.'),
        ('B02_EXPLICIT_CURRENT_ASSERTION_REPLACEMENT', STALE, ['assertion_scope', 'cutoff_eligible_explicit_superseder'], ['POSITIVE', 'ABSTAIN'], ['S4', 'S5'],
         'Only an explicitly identified current-state assertion with an available explicit replacement can receive a weak stale label.',
         'Historical measurements are not stale just because later measurements exist; no absent-end negative.'),
        ('B03_SAME_EVENT_CONFLICT_GUARD', CONFLICT, ['same_patient', 'same_concept/time/unit', 'same_event_or_correction_proof'], ['ABSTAIN'], ['S3', 'S4', 'S5'],
         'No clinical conflict label without comparable same-event evidence and explicit correction/context information.',
         'Different visits, numeric changes and unit ambiguity do not prove contradiction; this version only abstains.'),
    ]
    rules = []
    for i, (rid, target, fields, outputs, refs, assumption, limitation) in enumerate(specifications):
        rules.append({'rule_id': rid, 'rule_version': VERSION, 'target_task': target[0], 'target_axis': target[1],
                      'required_fields': fields, 'permitted_output': outputs, 'source_reference': refs,
                      'exact_source_claim': [{'source_id': ref, 'claim_field': 'sources.' + ref + '.claim',
                                              'representation': 'PARAPHRASE_NOT_VERBATIM'} for ref in refs],
                      'researcher_operational_assumption': assumption, 'limitations': limitation,
                      'code_location': 'scripts/evaluate_memory_silver.py:label_functions', 'priority': i,
                      'conflict_policy': 'NO_MAJORITY; opposite or multiple applicable votes => ABSTAIN; never combine tasks/axes'})
    return {'schema_version': VERSION, 'access_date_kst': '2026-11-07',
            'source_verification': 'Six URLs directly opened with web tool; no whole-page-byte hash claimed',
            'sources': SOURCES, 'rules': rules,
            'source_claim_scope': 'Literature motivates methodology; all LoCDI label conditions are researcher-defined operational assumptions'}


def label_functions(context):
    s = context.get('source') or {}; targets = context.get('released_facts', [])
    valid = context.get('structural_pass') is True
    observation = s.get('type') == 'observations'
    out = []
    def vote(rule, label='ABSTAIN', reason='REQUIRED_EVIDENCE_MISSING'):
        out.append({'rule_id': rule, 'output': label, 'reason': reason})
    vote('A01_NONMEASUREMENT_SCOPE', 'NEGATIVE' if valid and s.get('type') and not observation else 'ABSTAIN',
         'EXPLICIT_RESOURCE_OUTSIDE_FIXED_MEASUREMENT_TASK' if valid and s.get('type') and not observation else 'OBSERVATION_OR_MISSING_TYPE')
    coding = valid and observation and all(s.get(k) not in (None, '') for k in ('system', 'code', 'description', 'units', 'value'))
    same = coding and any(all(f.get(k) not in (None, '') and f[k] == s[k] for k in ('system', 'code', 'description', 'units'))
                          and f.get('value') not in (None, '') and f.get('source_type') == 'observations' for f in targets)
    vote('A02_EXPLICIT_CODING_MATCH', 'POSITIVE' if same else 'ABSTAIN',
         'EXPLICIT_CODING_DESCRIPTION_UNIT_IDENTITY' if same else 'EXPLICIT_STANDARDIZED_IDENTITY_NOT_ESTABLISHED')
    complete = bool(targets) and all(f.get('description') not in (None, '') and f.get('units') not in (None, '') and f.get('value') not in (None, '') for f in targets)
    literal_ready = valid and observation and s.get('description') not in (None, '') and s.get('units') not in (None, '') and s.get('value') not in (None, '') and complete
    literal = literal_ready and any(s['description'] == f['description'] and s['units'] == f['units'] for f in targets)
    vote('X01_LITERAL_MATCH', 'POSITIVE' if literal else 'ABSTAIN', 'EXACT_FULL_DESCRIPTION_AND_UNIT' if literal else 'NO_PERMITTED_LITERAL_MATCH')
    negative = valid and bool(s.get('type')) and not observation
    if literal_ready and s['description'] not in {f['description'] for f in targets}: negative = True
    vote('X02_LITERAL_NONMATCH', 'NEGATIVE' if negative else 'ABSTAIN',
         'OUTSIDE_LITERAL_ONLY_RETRIEVAL_TARGET' if negative else 'MISSING_FIELDS_OR_MATCH_OR_UNRESOLVED_UNIT_DIFFERENCE')
    vote('B01_CLINICAL_EVIDENCE_GUARD', reason='NO_INDEPENDENT_CLINICAL_USEFULNESS_REFERENCE')
    stale = valid and context.get('assertion_scope') == 'CURRENT_STATE_ASSERTION' and context.get('cutoff_eligible_explicit_superseder') is True
    vote('B02_EXPLICIT_CURRENT_ASSERTION_REPLACEMENT', 'POSITIVE' if stale else 'ABSTAIN',
         'EXPLICIT_REPLACEMENT_OF_CURRENT_ASSERTION' if stale else 'NO_EXPLICIT_CURRENT_ASSERTION_REPLACEMENT_PROOF')
    vote('B03_SAME_EVENT_CONFLICT_GUARD', reason='SAME_EVENT_CONTEXT_AND_CORRECTION_PROOF_NOT_ESTABLISHED')
    if not valid:
        for row in out: row.update(output='ABSTAIN', reason='STRUCTURAL_BOUNDARY_NOT_PASSED')
    return out


def combine(votes):
    active = [v for v in votes if v['output'] != 'ABSTAIN']
    if any(v['output'] not in ('POSITIVE', 'NEGATIVE', 'ABSTAIN') for v in votes): raise ValueError('INVALID_LF_OUTPUT')
    conflict = len({v['output'] for v in active}) > 1
    if len(active) == 1:
        return active[0]['output'], None, conflict
    return 'ABSTAIN', ('RULE_CONFLICT' if conflict else 'MULTIPLE_APPLICABLE_RULES' if active else 'NO_APPLICABLE_DECISIVE_RULE'), conflict


def sampling(packets):
    """Freeze before LF/retrieval: five ordinal strata, two cases/stratum, one record/kind stratum."""
    rng = random.Random(SEED); selected = []
    for start in range(0, 50, 10):
        for index in sorted(rng.sample(list(range(start, start + 10)), 2)):
            p = packets[index]
            for objective in (True, False):
                records = sorted([r for r in p['candidate_memories'] if (r['kind'] == 'objective_finding') == objective], key=lambda r: r['memory_id'])
                if records:
                    r = records[rng.randrange(len(records))]
                    selected.append({'snapshot_id': p['snapshot_id'], 'memory_id': r['memory_id'],
                                     'ordinal_stratum': start // 10, 'source_kind_stratum': 'objective' if objective else 'other'})
    return {'seed': SEED, 'selection_rule': '5 ordinal strata x 2 cases; <=1 per source-kind group; no label/rank/result input',
            'selected': selected, 'clinical_review_status': 'NOT_PERFORMED'}


def validate_query_evidence(rows, packets):
    if [r['snapshot_id'] for r in rows] != [p['snapshot_id'] for p in packets]: raise ValueError('QUERY_EVIDENCE_ORDER')
    for r,p in zip(rows,packets):
        if digest({k:v for k,v in r.items() if k!='evidence_fingerprint'}) != r['evidence_fingerprint']:
            raise ValueError('QUERY_EVIDENCE_FINGERPRINT')
        if r['query_fingerprint'] != p['query_candidate']['query_fingerprint']:
            raise ValueError('QUERY_EVIDENCE_BINDING')


def classify_all(contexts, rule_registry, disabled=None):
    rules = {r['rule_id']: r for r in rule_registry['rules']}; lf_rows = []; silver = []
    for c in contexts:
        votes = label_functions(c)
        for v in votes:
            if v['output'] not in rules[v['rule_id']]['permitted_output']: raise ValueError('RULE_OUTPUT_CONTRACT')
            lf_rows.append({'snapshot_id': c['snapshot_id'], 'memory_id': c['memory_id'], **v,
                            'target_task':rules[v['rule_id']]['target_task'],'target_axis':rules[v['rule_id']]['target_axis'],
                            'rule_version': VERSION, 'evidence_fingerprint': c['evidence_fingerprint']})
        for task, axis in TARGETS:
            relevant = [v for v in votes if v['rule_id'] != disabled and
                        (rules[v['rule_id']]['target_task'], rules[v['rule_id']]['target_axis']) == (task, axis)]
            label, reason, conflict = combine(relevant)
            used = [v['rule_id'] for v in relevant if v['output'] != 'ABSTAIN']
            citations = sorted({ref for rid in used for ref in rules[rid]['source_reference']})
            silver.append({'snapshot_id': c['snapshot_id'], 'memory_id': c['memory_id'], 'task_id': task,
                           'label_axis': axis, 'silver_label': label, 'abstain_reason': reason,
                           'individual_abstain_reasons': {v['rule_id']: v['reason'] for v in relevant if v['output'] == 'ABSTAIN'},
                           'applied_rule_ids': used, 'supporting_source_fingerprints': [c['evidence_fingerprint'], c['query_fingerprint']],
                           'citations': [SOURCES[ref]['url'] for ref in citations],
                           'assumptions': [rules[rid]['researcher_operational_assumption'] for rid in used],
                           'label_conflict': conflict, 'rule_version': VERSION, 'reference_type': 'SILVER_AUTOMATIC_NOT_HUMAN_GOLD'})
    return lf_rows, silver


def coverage(silver):
    groups = defaultdict(Counter)
    for r in silver:
        key = r['task_id'] + '/' + r['label_axis']; groups[key][r['silver_label']] += 1
        groups[key]['conflicts'] += r['label_conflict']
    return {k: {'counts': {l: v[l] for l in ('POSITIVE', 'NEGATIVE', 'ABSTAIN')},
                'silver_label_coverage': ratio(v['POSITIVE'] + v['NEGATIVE'], v['POSITIVE']+v['NEGATIVE']+v['ABSTAIN']),
                'abstention_rate': ratio(v['ABSTAIN'], v['POSITIVE']+v['NEGATIVE']+v['ABSTAIN']),
                'rule_conflict_rate': ratio(v['conflicts'], v['POSITIVE']+v['NEGATIVE']+v['ABSTAIN'])} for k, v in groups.items()}


def evaluate_labels(silver, selections, pool_ids):
    grouped = defaultdict(list)
    for row in silver: grouped[(row['task_id'], row['label_axis'], row['snapshot_id'])].append(row)
    results = []
    for task, axis in TARGETS:
        for mode in ('none', 'full_history', 'selective'):
            case_rows = []
            for snapshot, pool in pool_ids.items():
                labels = grouped[(task, axis, snapshot)]
                judgments = [{'memory_id': r['memory_id'], 'relevant': None if r['silver_label'] == 'ABSTAIN' else r['silver_label'] == 'POSITIVE', 'stale': None} for r in labels]
                selected = selections[(snapshot, mode)]
                stats = judged_metrics(selected, pool, judgments)
                case_rows.append({'snapshot_id': snapshot, 'returned_count': len(selected),
                                  'silver_judged_precision': stats['judged_set_precision'],
                                  'silver_judged_coverage': stats['judged_coverage'],
                                  'known_silver_positive_retrieval_count': stats['known_relevant_retrieval_count'],
                                  'unjudged_count': stats['unjudged_returned_count']})
            returned = sum(r['returned_count'] for r in case_rows)
            judged = sum(r['silver_judged_precision']['denominator'] for r in case_rows)
            positive = sum(r['known_silver_positive_retrieval_count'] for r in case_rows)
            results.append({'task_id': task, 'axis': axis, 'mode': mode, 'registered_cases': len(pool_ids),
                            'returned_count': returned, 'silver_judged_precision': ratio(positive, judged),
                            'silver_judged_coverage': ratio(judged, returned),
                            'known_silver_positive_retrieval_count': positive, 'unjudged_count': returned - judged,
                            'cases_with_judged_returns': sum(r['silver_judged_precision']['denominator'] > 0 for r in case_rows),
                            'full_pool_recall': 'NOT_IDENTIFIABLE_NOT_COMPUTED', 'case_rows': case_rows})
    return results


def protect():
    previous = ROOT / 'outputs/private/research_evaluation/c4_08_v1'
    pins = load(previous / 'protected_hashes.json')
    mp = ROOT / 'outputs/research_evaluation/c4_08_v1/reports/output_manifest.json'; m = load(mp)
    if digest({k:v for k,v in m.items() if k != 'fingerprint'}) != m['fingerprint']: raise ValueError('PARENT_MANIFEST')
    ip = previous / 'private_index.json'
    if sha(ip) != m['private_index_sha256']: raise ValueError('PARENT_PRIVATE_INDEX')
    pins.update(m['files']); pins.update(load(ip)['files'])
    pins[mp.relative_to(ROOT).as_posix()] = sha(mp); pins[ip.relative_to(ROOT).as_posix()] = sha(ip)
    for p in (ROOT / 'outputs/private/research_evaluation/c4_06_reviews').glob('**/*'):
        if p.is_file(): pins[p.relative_to(ROOT).as_posix()] = sha(p)
    check_hashes(pins)
    return pins


def run(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace): raise ValueError('INVALID_NAMESPACE')
    public = ROOT / 'outputs/research_evaluation' / namespace
    private = ROOT / 'outputs/private/research_evaluation' / namespace
    if public.exists() or private.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    pins = protect()
    source_pins = load(ROOT / 'outputs/private/research_evaluation/c4_07_v1/source_hashes.json')
    check_hashes(source_pins)
    original = load_cases(DATA)
    bundle = ROOT / 'outputs/private/research_evaluation/c4_05_v1/judged_pool_v1'; m = load(bundle / 'manifest.json')
    if digest({k:v for k,v in m.items() if k != 'fingerprint'}) != m['fingerprint']: raise ValueError('POOL_MANIFEST')
    packets = []
    for e in m['packets']:
        if Path(e['file']).name != e['file'] or sha(bundle/e['file']) != e['sha256']: raise ValueError('PACKET_SHA')
        p = load(bundle/e['file'])
        if digest(p) != e['packet_fingerprint']: raise ValueError('PACKET_FINGERPRINT')
        packets.append(p)
    if len(original) != 50 or [p['snapshot_id'] for p in packets] != ['OriginalCase%03d'%i for i in range(1,51)]:
        raise ValueError('ALL50_REQUIRED')
    expected = [(p['snapshot_id'], r['memory_id']) for p in packets for r in p['candidate_memories']]
    evidence = [json.loads(l) for l in (ROOT/'outputs/private/research_evaluation/c4_07_v1/pre_review_evidence.jsonl').open(encoding='utf8')]
    if len(expected) != 989 or expected != [(r['snapshot_id'],r['memory_id']) for r in evidence]: raise ValueError('ALL989_ORDER_REQUIRED')
    for r in evidence:
        if digest({k:v for k,v in r.items() if k != 'evidence_fingerprint'}) != r['evidence_fingerprint']: raise ValueError('EVIDENCE_FINGERPRINT')
    evidence_map = {(r['snapshot_id'], r['memory_id']):r for r in evidence}
    query_evidence = [json.loads(l) for l in (ROOT/'outputs/private/research_evaluation/c4_07_v1/query_evidence.jsonl').open(encoding='utf8')]
    validate_query_evidence(query_evidence,packets)
    rule_registry = registry()
    protocol = {'version': VERSION, 'tasks': [list(t) for t in TARGETS], 'query_mode': 'UNCHANGED_C4_05_FINAL_RELEASED_QUERY',
                'original50':50, 'fixed_memory':72590, 'pool':989, 'top_k':5, 'combination':'SINGLE_DECISIVE_RULE_ELSE_ABSTAIN',
                'clinical_gold_status':'NOT_CREATED', 'registry_fingerprint':digest(rule_registry),
                'sampling_seed':SEED, 'full_pool_recall':'NOT_COMPUTED'}
    # Persist the operational targets and human-audit selection before any LF or retrieval.
    selection = sampling(packets)
    write(public/'literature_rule_registry.json',rule_registry); write(public/'protocol.json',protocol)
    write(private/'audit_selection_manifest.json',selection)
    contexts = []; selections = {}; pool_ids = {}; trace_rows = []; structural = Counter()
    audit_keys = {(r['snapshot_id'], r['memory_id']) for r in selection['selected']}; audit_packets = []
    for p, case in zip(packets, original):
        identity = case['agent_visible']['longitudinal']; q = p['query_candidate']
        if identity != p['private_identity']: raise ValueError('IDENTITY_BINDING')
        validate_candidate(q, case['environment']['current_encounter_events'], identity)
        records = [r for v in case['agent_visible']['past_visits'] for r in v['records']]
        by_id = {r['memory_id']:r for r in records}
        for r in p['candidate_memories']:
            if by_id.get(r['memory_id']) != r: raise ValueError('ORIGINAL_MEMORY_CHANGED')
        events = {e['event_id']:e for e in case['environment']['current_encounter_events']}
        facts = []
        for link in q['source_links']:
            event = events[link['event_id']]; raw = json.loads(event['evidence'])
            for f in (raw['facts'] if event['action'] == 'Ask' else raw['records']):
                cleaned = next(x for x in q['source_facts'] if x['code'] == f['code'] and x['recorded_at'] == f['recorded_at'] and x['description'] == f['description'])
                facts.append(dict(cleaned, system=f.get('system'), source_type='observations'))
        pool_ids[p['snapshot_id']] = [r['memory_id'] for r in p['candidate_memories']]
        for r in p['candidate_memories']:
            e = evidence_map[(p['snapshot_id'],r['memory_id'])]
            contexts.append({'snapshot_id':p['snapshot_id'],'memory_id':r['memory_id'], 'source':e['source_value'],
                             'released_facts':facts,'structural_pass':'STRUCTURAL_PASS' in e['classifications'],
                             'assertion_scope':'HISTORICAL_MEASUREMENT' if (e['source_value'] or {}).get('type') == 'observations' else 'UNKNOWN',
                             'cutoff_eligible_explicit_superseder':False,
                             'evidence_fingerprint':e['evidence_fingerprint'],'query_fingerprint':q['query_fingerprint']})
            if (p['snapshot_id'],r['memory_id']) in audit_keys:
                audit_packets.append({'snapshot_id':p['snapshot_id'],'memory_id':r['memory_id'],
                                      'query_projection':q['projection'],'source_evidence':e['source_value'],
                                      'memory_record':r,'query_fingerprint':q['query_fingerprint'],
                                      'independent_audit_results':None,'silver_labels_and_search_ranks_hidden':True})
        store = PatientMemoryStore(records)
        for mode in ('none','full_history','selective'):
            selected = store.retrieve(identity,q['query'],mode,5)['records']
            for r in selected:
                structural['patient_visit_violations'] += r['patient_id'] != identity['patient_id'] or r['visit_id'] == identity['visit_id']
                structural['cutoff_violations'] += timestamp(r['visit_started_at']) >= timestamp(identity['started_at']) or timestamp(r['available_at']) > timestamp(identity['started_at'])
            ids = [r['memory_id'] for r in selected]; selections[(p['snapshot_id'],mode)] = ids
            trace_rows.append({'snapshot_id':p['snapshot_id'],'mode':mode,'query_fingerprint':q['query_fingerprint'],
                               'memory_ids':ids,'cutoff':identity['started_at']})
    if sum(len(v['records']) for c in original for v in c['agent_visible']['past_visits']) != 72590: raise ValueError('MEMORY_COUNT')
    lf_rows,silver = classify_all(contexts,rule_registry)
    if (lf_rows,silver) != classify_all(contexts,rule_registry): raise ValueError('NONDETERMINISTIC_LABELS')
    dataset_fp = digest({'source_pins':source_pins,'manifest':m['fingerprint'],'protocol':protocol,'selection':selection})
    for r in silver:
        r['parent_dataset_fingerprint'] = dataset_fp
        r['dataset_fingerprint'] = digest([dataset_fp,r['task_id'],r['label_axis']])
    evaluations = evaluate_labels(silver,selections,pool_ids)
    sensitivities = []
    for rule in rule_registry['rules']:
        _,variant = classify_all(contexts,rule_registry,disabled=rule['rule_id'])
        changed = sum(a['silver_label'] != b['silver_label'] for a,b in zip(silver,variant))
        sensitivities.append({'removed_rule':rule['rule_id'],'changed_labels_across_4945_task_axis_rows':changed,
                              'coverage':coverage(variant), 'evaluation':evaluate_labels(variant,selections,pool_ids)})
    counts = {r['rule_id']:dict(Counter(v['output'] for v in lf_rows if v['rule_id'] == r['rule_id'])) for r in rule_registry['rules']}
    check_hashes(pins); check_hashes(source_pins)
    jsonl(private/'labeling_function_results.jsonl',lf_rows); jsonl(private/'silver_reference.jsonl',silver)
    jsonl(private/'task_a_standardized_reference.jsonl',[r for r in silver if r['task_id']==A[0]])
    jsonl(private/'task_a_literal_reference.jsonl',[r for r in silver if r['task_id']==AX[0]])
    jsonl(private/'task_b_reference.jsonl',[r for r in silver if r['task_id']==B[0]])
    jsonl(private/'objective_checks.jsonl',[{'snapshot_id':r['snapshot_id'],'memory_id':r['memory_id'],
                                           'checks':r['checks'],'classifications':r['classifications'],
                                           'source_evidence_fingerprint':r['evidence_fingerprint']} for r in evidence])
    write(private/'retrieval_trace.json',trace_rows);write(private/'human_audit_packet.json',{'selection_fingerprint':digest(selection),'items':audit_packets})
    write(private/'protected_hashes.json',pins); write(private/'source_hashes.json',source_pins)
    write(public/'rule_coverage.json',{'rule_counts':counts,'tasks':coverage(silver),'human_gold':False})
    write(public/'rule_conflicts.json',{'conflicts':sum(r['label_conflict'] for r in silver),'task_axis_rows':len(silver),
                                      'multiple_decisive_rule_rows':sum(r['abstain_reason']=='MULTIPLE_APPLICABLE_RULES' for r in silver)})
    write(public/'silver_evaluation.json',{'status':'SILVER_AUTOMATIC_ONLY','clinical_metrics':'NOT_EVALUATED',
                                          'registered_cases':50,'pool_records':989,'full_original_memory':72590,
                                          'structural_leakage':dict(structural),'rows':evaluations})
    write(public/'sensitivity_analysis.json',{'method':'LEAVE_ONE_LABELING_FUNCTION_OUT_NO_RETRIEVAL_CHANGE','variants':sensitivities})
    write(public/'circularity_audit.json',{'status':'CIRCULAR_EVALUATION_RISK','task_A_literal_risk':'HIGH',
        'reasons':['Full description and unit occur both in structured LF operands and lexical retrieval text.',
                   'Existing pool includes lexical and source-text matching candidates.',
                   'Exact field identity is not identical to token-overlap ranking but shares textual signal.'],
        'single_token_labeling_rule':False,'retrieval_scores_used_in_labeling':False,
        'independent_clinical_quality_improvement_claim':False,'full_pool_recall_identifiable':False,
        'snorkel_label_model':'NOT_FIT_NO_PACKAGE_INSTALLED',
        'label_model_limitations':['989 clustered records / 50 snapshots, not independent rows',
                                   'Literal positive/negative functions share field dependencies',
                                   'No independent development judgments for calibration',
                                   'No need to learn probabilities for deterministic literal task']})
    result = {'status':'C4_09_SILVER_EVALUATION_COMPLETED','scope':'AUTOMATIC_OPERATIONAL_TASKS_NOT_CLINICAL_GOLD',
              'dataset_fingerprint':dataset_fp,'patients':50,'memory_records':72590,'pool_records':989,
              'label_function_rows':len(lf_rows),'silver_task_axis_rows':len(silver),'task_coverage':coverage(silver),
              'protected_files':len(pins),'changed_or_missing':0,'source_assets_checked':len(source_pins),
              'audit_records':len(audit_packets),'audit_cases':len({r['snapshot_id'] for r in audit_packets}),
              'retrieval_calls':len(trace_rows),'labels_deterministic':True,'actual_llm_generations':0,
              'model_fitting':0,'human_review_completed':0,'clinical_verdicts_generated':0}
    write(public/'result.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--namespace',default='c4_09_v1')
    a=parser.parse_args()
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('OFFLINE_ONLY')
    sys.addaudithook(guard)
    print(json.dumps(run(a.namespace),ensure_ascii=True))
