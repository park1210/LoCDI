"""Offline descriptive audit. No clinical equivalence or reviewer inference."""
from collections import Counter
from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import unicodedata

from app.locdi.llm_setup import ROOT, digest
from app.locdi.routing_collector import PRIVATE_ROOT, PUBLIC_ROOT, _directory, atomic_json
from app.locdi.specialty_collection_analysis import _safe_json, analyze_records
from app.locdi.routing_collection_preparation import CATALOG, TAXONOMY
from app.llm.prompt_loader import specialty_system_prompt


def name_mapping(label, catalog):
    """Text relations only; even exact spelling is not clinical Gold."""
    normalize = lambda s: ' '.join(unicodedata.normalize('NFKC', s).casefold().split())
    exact = [i + 1 for i, x in enumerate(catalog) if x == label]
    normalized = [i + 1 for i, x in enumerate(catalog) if normalize(x) == normalize(label)]
    return {'text_status': 'EXACT_TEXT_PRESENT' if exact else 'REVIEW_REQUIRED',
            'exact_catalog_ordinals': exact, 'normalized_candidate_ordinals': normalized,
            'code_mapping': 'REVIEW_REQUIRED_NO_CATALOG_CODES',
            'clinical_equivalence': None, 'clinical_gold_approved': False}


def response_state(row):
    if row.get('execution_status') != 'SUCCEEDED':
        return 'REQUEST_FAILED', 'NOT_EVALUABLE'
    if row.get('response_contract_status') != 'VALID':
        return 'SCHEMA_INVALID', 'NOT_EVALUABLE'
    answer = row['response']
    if answer['certainty'] == 'unresolved' or answer['diagnosis'] == 'Unresolved':
        return 'UNRESOLVED', 'NOT_EVALUABLE'
    return 'RESPONSE_RECORDED', 'NOT_REVIEWED'


def condition_link(candidate, reference):
    d = candidate['decision']
    encounter_day = datetime.fromisoformat(candidate['visit_started_at'].replace('Z', '+00:00')).date()
    rows = []
    for source in reference['source_condition_rows']:
        record = source['source_record']
        delta = (date.fromisoformat(record['START']) - encounter_day).days
        rows.append({'patient_link': record['PATIENT'] == d['patient_id'],
                     'encounter_link': record['ENCOUNTER'] == d['visit_id'],
                     'start_day_offset': delta,
                     'timing_status': 'REVIEW_REQUIRED_START_AFTER_ENCOUNTER' if delta > 0 else 'REVIEW_REQUIRED_DATE_ONLY',
                     'clinical_reference_valid': None})
    return rows


def analyze(bundle, references, records, taxonomy, prompts):
    """Pure aggregate analysis plus PRIVATE source mappings and blank reviews."""
    before = digest([bundle, references, records])
    refs = {r['case_ref']: r for r in references}
    groups, private_cases, template, private_responses = [], [], [], []
    matrix, matched = {}, {'same_source_stratum': Counter(), 'other_source_stratum': Counter()}
    certainty = Counter(); tags = Counter(); rationale_flags = Counter()
    for c in bundle['candidates']:
        case, d = c['case_ref'], c['decision']; ref = refs[case]
        mapping = name_mapping(ref['declared_label'], bundle['catalog'])
        source_maps = [{'system': s['source_record']['SYSTEM'], 'code': s['source_record']['CODE'],
                        'description': s['source_record']['DESCRIPTION'],
                        'mapping': name_mapping(s['source_record']['DESCRIPTION'], bundle['catalog'])}
                       for s in ref['source_condition_rows']]
        links = condition_link(c, ref)
        obs = d['observation']; memories = obs['longitudinal_memory']['records']
        case_rows = [r for r in records if r['case_ref'] == case]
        if len(case_rows) != 5 or len({r['agent_ref'] for r in case_rows}) != 5:
            raise ValueError('COMPLETE_BASELINE_REQUIRED_NO_MERGE')
        stats = {'case_ref': case, 'split': d['split'], 'source_stratum': c['stratum'],
                 'explicit_label_present': bool(mapping['exact_catalog_ordinals']),
                 'objective_finding_count': len(obs['objective_findings']), 'history_count': len(obs['history']),
                 'chief_complaint_present': bool(obs['chief_complaint']), 'symptom_count': len(obs['current_symptoms']),
                 'memory_count': len(memories), 'unique_memory_text_count': len({m['text'] for m in memories}),
                 'unresolved': 0, 'response_recorded': 0,
                 'source_timing_review_required': any(x['start_day_offset'] > 0 for x in links)}
        for row in case_rows:
            if row.get('correct') is not None:
                raise ValueError('UNREVIEWED_CORRECTNESS_MUST_REMAIN_NULL')
            state, evaluation = response_state(row); answer = row['response']
            certainty[answer['certainty']] += 1
            stats['unresolved' if state == 'UNRESOLVED' else 'response_recorded'] += 1
            matrix.setdefault(c['stratum'], {}).setdefault(row['department'], Counter())[state] += 1
            matched['same_source_stratum' if row['department'] == c['stratum'] else 'other_source_stratum'][state] += 1
            rationale = answer['rationale'].casefold()
            visible = json.dumps(c['shared_payload'], ensure_ascii=False).casefold()
            hints = []
            if any(t in rationale for t in ('insufficient', 'lack', 'not provided', 'not specified', 'not mentioned', 'limited', 'additional', 'further')):
                hints.append('LEXICAL_INSUFFICIENCY_OR_CONFIRMATION_MENTION')
            for term in ('ecg', 'troponin'):
                if term in rationale and term not in visible:
                    hints.append('RATIONALE_' + term.upper() + '_NOT_IN_SNAPSHOT_TEXT')
            rationale_flags.update(hints)
            hypotheses = []
            if state == 'UNRESOLVED':
                hypotheses = ['PROMPT_EFFECT_POSSIBLE', 'MODEL_LIMITATION_POSSIBLE',
                              'LEGITIMATE_ABSTENTION_POSSIBLE', 'CAUSE_UNDETERMINED']
                if not obs['chief_complaint'] or not obs['current_symptoms']:
                    hypotheses.append('OBSERVATION_INSUFFICIENT_POSSIBLE')
                if not mapping['exact_catalog_ordinals']:
                    hypotheses.append('CATALOG_GAP')  # Literal gap only; clinical relation still unreviewed.
                tags.update(hypotheses)
            private_responses.append({'case_ref': case, 'agent_ref': row['agent_ref'], 'response_ref': row['response_ref'],
                'response_fingerprint': row['response_fingerprint'], 'classification': state,
                'evaluation_status': evaluation, 'certainty': answer['certainty'],
                'named_diagnosis_with_unresolved_certainty': answer['certainty'] == 'unresolved' and answer['diagnosis'] != 'Unresolved',
                'hypotheses': hypotheses, 'rationale_flags': hints,
                'rationale_interpretation': 'EXPLORATORY_NOT_CAUSAL_OR_CLINICALLY_VERIFIED', 'correct': None})
            template.append({'case_ref': case, 'agent_ref': row['agent_ref'], 'response_ref': row['response_ref'],
                'response_fingerprint': row['response_fingerprint'], 'status': 'NOT_REVIEWED',
                'reviewer_ref': None, 'reviewer_qualification': None, 'conflict_of_interest': None,
                'procedure_version': None, 'reviewed_at': None,
                'source_encounter_link_valid': None, 'visibility_valid': None, 'proxy_acceptable': None,
                'observation_sufficient': None, 'memory_sufficient': None, 'reference_valid': None,
                'reference_source_ref': None, 'reference_fingerprint': None, 'reference_time_scope': None,
                'catalog_mapping_valid': None, 'clinical_equivalence': None,
                'response_reference_correspondence': None, 'unresolved_evaluation_policy': None,
                'abstention_appropriate': None, 'routing_success': None, 'correct': None,
                'verdict_rationale': None, 'unresolved_review_questions': None})
        groups.append(stats)
        private_cases.append({'case_ref': case, 'reference_candidate': deepcopy(ref),
            'declared_label_mapping': mapping, 'source_condition_mappings': source_maps,
            'condition_encounter_checks': links, 'clinical_review_status': 'NOT_REVIEWED',
            'observation_sufficiency': None, 'memory_sufficiency': None})
    public = {'schema_version': 'locdi.unresolved_audit.v1', 'run_id': 'phase24b_collect_001',
        'preparation': 'phase24b_candidates_v2', 'preparation_fingerprint': bundle['fingerprint'],
        'decisions': groups, 'response_count': len(records), 'certainty_counts': dict(certainty),
        'named_diagnosis_unresolved_count': sum(r['named_diagnosis_with_unresolved_certainty'] for r in private_responses),
        'source_stratum_agent_matrix': matrix, 'source_stratum_comparison': matched,
        'literal_coverage': {'present_decisions': sum(g['explicit_label_present'] for g in groups),
                             'absent_decisions': sum(not g['explicit_label_present'] for g in groups),
                             'clinical_equivalence': 'REVIEW_REQUIRED', 'uncovered_comparison': 'NOT_AVAILABLE_IF_EMPTY'},
        'catalog_entries_by_existing_taxonomy': dict(Counter(taxonomy.get(x, 'UNMAPPED') for x in bundle['catalog'])),
        'taxonomy_clinical_status': 'NOT_REVIEWED_EXISTING_STRATIFICATION_ONLY',
        'hypothesis_counts_nonexclusive': dict(tags), 'rationale_flag_counts_all_responses': dict(rationale_flags),
        'prompt_fingerprints': {dept: hashlib.sha256(text.encode('utf-8')).hexdigest() for dept, text in prompts.items()},
        'causal_identification': 'NOT_ESTABLISHED', 'raw_wire_payload': 'NOT_CHECKED_NOT_SAVED',
        'review_status': 'NOT_REVIEWED', 'training_readiness': 'BLOCKED', 'clinical_metrics': 'NOT_EVALUATED',
        'research_readiness': False, 'new_llm_calls': 0, 'utility_fit_count': 0, 'clinical_gold_generated': 0}
    private = {'schema_version': 'locdi.private_unresolved_audit.v1', 'cases': private_cases, 'responses': private_responses}
    review = {'schema_version': 'locdi.blank_independent_review.v1', 'status': 'NOT_REVIEWED',
        'run_id': public['run_id'], 'source_review_packet': 'phase24c_review_v2',
        'preparation_fingerprint': bundle['fingerprint'], 'items': template,
        'instructions': 'Blank clinician input, not reviewed evidence or validator-ready judgments. Review source/time/reference before responses. Preserve null if uncertain; no string-only correctness or false imputation.'}
    assert digest([bundle, references, records]) == before
    return public, private, review


def load_analysis():
    prep = PRIVATE_ROOT / 'phase24b_candidates_v2'; run = PRIVATE_ROOT / 'phase24b_collect_001'
    bundle = _safe_json(prep/'preparation.json'); refs = _safe_json(prep/'reference_candidates.json')
    original_review = ROOT/'outputs/private/routing_collection_review/phase24c_review_v2'
    packet = _safe_json(original_review/'review_packet.json'); manifest = _safe_json(original_review/'manifest.json')
    if manifest.get('complete') is not True or manifest['packet_fingerprint'] != digest(packet):
        raise ValueError('REVIEW_PACKET_CHANGED')
    if packet['preparation_fingerprint'] != bundle['fingerprint'] or packet['run_id'] != run.name:
        raise ValueError('REVIEW_RUN_LINK_MISMATCH')
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.json')}
    if files != packet['source_file_sha256']:
        raise ValueError('ORIGINAL_RUN_CHANGED_SINCE_PHASE24C')
    records = [(p.name, _safe_json(p)) for p in sorted(run.glob('Candidate*.json'))]
    prompts = {a['department']: specialty_system_prompt(a['department']) for a in bundle['registry']['agents'] if a['role'] == 'specialty'}
    from app.locdi.routing_training_setup import REGISTRY
    integrity, _ = analyze_records(bundle, refs, records, header=_safe_json(run/'run_header.json'),
        completion=_safe_json(run/'run_complete.json'), audit=_safe_json(PUBLIC_ROOT/(run.name+'_routing_collection.json')),
        manifest=_safe_json(prep/'manifest.json'), current_registry=_safe_json(REGISTRY),
        current_prompts={d: hashlib.sha256(t.encode('utf-8')).hexdigest() for d,t in prompts.items()})
    if integrity['integrity_issue_codes'] or integrity['state'] != 'COLLECTION_COMPLETE_NOT_CLINICALLY_EVALUATED':
        raise ValueError('BASELINE_INTEGRITY_BLOCKED')
    if _safe_json(CATALOG) != bundle['catalog']:
        raise ValueError('CATALOG_CHANGED')
    public, private, review = analyze(bundle, refs, [r for _,r in records], _safe_json(TAXONOMY), prompts)
    public['baseline_integrity'] = 'PASS_RECORDED_HASHES_AND_PHASE24C_FILE_HASHES'
    public['training_validation'] = integrity['training_validation']
    private['phase24c_packet_fingerprint'] = digest(packet)
    private['baseline_file_hashes'] = files
    return public, private, review


def save_analysis(public, private, review, name):
    root = ROOT/'outputs/private/unresolved_analysis'
    if ROOT.resolve() not in root.resolve().parents:
        raise ValueError('PRIVATE_ROOT_REDIRECTED')
    public_path = PUBLIC_ROOT/(name + '_summary.json')
    path = _directory(root, name)
    if public_path.exists():
        raise FileExistsError('PUBLIC_RESULT_EXISTS')
    path = _directory(root, name, create=True)
    atomic_json(path/'coverage_and_response_analysis.json', private)
    atomic_json(path/'independent_review_template.json', review)
    atomic_json(path/'manifest.json', {'complete': True, 'analysis_fingerprint': digest(private), 'template_fingerprint': digest(review)})
    atomic_json(public_path, public)
    return path
