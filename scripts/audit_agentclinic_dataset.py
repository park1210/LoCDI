"""Read-only AgentClinic inventory. No adapter, clinical grading, network or admission.

All original strings and diagnostic labels remain in private inputs/results.
Only aggregate counts, anonymous ordinals and lexical flags escape publicly.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.source_replay import atomic_rows

FIELDS = (
    'Objective_for_Doctor', 'Patient_Actor', 'Patient_Actor.Demographics',
    'Patient_Actor.History', 'Patient_Actor.Symptoms',
    'Patient_Actor.Symptoms.Primary_Symptom',
    'Patient_Actor.Symptoms.Secondary_Symptoms',
    'Patient_Actor.Past_Medical_History', 'Patient_Actor.Social_History',
    'Patient_Actor.Review_of_Systems', 'Physical_Examination_Findings',
    'Physical_Examination_Findings.Vital_Signs', 'Test_Results', 'Correct_Diagnosis')
PLACEHOLDERS = {'information not provided', 'not provided', 'not specified',
                'unknown', 'n/a', 'not available', 'not assessed', 'none provided',
                'no additional symptoms provided', 'no information provided'}
MISSING = object()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf8')


def parse(data):
    """Reject malformed records without exposing their text in errors."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('NONFINITE_NUMBER')
    try:
        content = data.decode('utf8', errors='strict')
    except UnicodeError:
        return [], [{'line': None, 'code': 'INVALID_UTF8'}]
    rows, errors = [], []
    for ordinal, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            errors.append({'line': ordinal, 'code': 'BLANK_RECORD'})
            continue
        try:
            row = json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
            if not isinstance(row, dict):
                raise ValueError('OBJECT_REQUIRED')
            canonical(row)
            rows.append(row)
        except (ValueError, TypeError, OverflowError):
            errors.append({'line': ordinal, 'code': 'INVALID_JSON_OBJECT_RECORD'})
    return rows, errors


def compare(local, official):
    a, ae = parse(local)
    b, be = parse(official)
    def eol(data):
        value = data.replace(b'\r\n', b'\n')
        return value[:-1] if value.endswith(b'\n') else value
    parsed_equal = not ae and not be and bool(a) and a == b
    return {'local_bytes': len(local), 'official_bytes': len(official),
            'local_sha256': sha(local), 'official_sha256': sha(official),
            'byte_identical': local == official,
            'ordered_parsed_records_equal': parsed_equal,
            'line_endings_and_one_terminal_newline_only': parsed_equal and eol(local) == eol(official),
            'local_CRLF': local.count(b'\r\n'), 'official_CRLF': official.count(b'\r\n'),
            'local_final_newline': local.endswith(b'\n'),
            'official_final_newline': official.endswith(b'\n'),
            'local_records': len(a), 'official_records': len(b),
            'local_errors': ae, 'official_errors': be,
            'different_ordinals': [i for i, (x, y) in enumerate(zip(a, b), 1) if x != y],
            'canonical_content_sha256': sha(canonical(a)) if parsed_equal else None,
            'copy_or_edit_history': 'NOT_VERIFIED; content differences do not prove their cause'}


def leaves(value, path=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + '.' + str(key) if path else str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, path + '[' + str(i) + ']')
    else:
        yield path, value


def state(value):
    if value is MISSING:
        return 'MISSING'
    if value is None or value == '' or value == [] or value == {}:
        return 'EMPTY_OR_NULL'
    if isinstance(value, str):
        if not value.strip():
            return 'EMPTY_OR_NULL'
        if value.strip().casefold().rstrip('. ') in PLACEHOLDERS:
            return 'EXPLICIT_UNPROVIDED'
    return 'PRESENT_NOT_CLINICALLY_REVIEWED'


def get(value, path):
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def lexical_hit(label, value):
    """Literal case-insensitive whole phrase, not synonym/clinical equivalence."""
    if not isinstance(label, str) or not label.strip() or not isinstance(value, str):
        return False
    return re.search(r'(?<!\w)' + re.escape(label.strip()) + r'(?!\w)', value,
                     flags=re.IGNORECASE) is not None


def label_variants(label):
    """Exploratory parenthesis variants only, never approved diagnoses/aliases."""
    if not isinstance(label, str):
        return []
    variants = [re.sub(r'\s*\([^)]*\)', '', label).strip()]
    variants += re.findall(r'\(([^)]+)\)', label)
    return [x for x in dict.fromkeys(variants) if len(x) >= 3 and x != label]


def audit(data):
    rows, errors = parse(data)
    coverage = {key: Counter() for key in FIELDS}
    types = {key: Counter() for key in FIELDS}
    hashes, narratives, diagnoses = defaultdict(list), defaultdict(list), set()
    public_cases, private_cases = [], []
    schema_errors, empty_core, test_branches, exam_branches, symptom_values = [], [], 0, 0, 0
    leaf_states = Counter()
    root_keys, actor_keys = Counter(), Counter()
    for ordinal, row in enumerate(rows, 1):
        hashes[sha(canonical(row))].append(ordinal)
        osce = row.get('OSCE_Examination')
        if set(row) != {'OSCE_Examination'} or not isinstance(osce, dict):
            schema_errors.append(ordinal)
            continue
        root_keys.update(osce.keys())
        expected = {'Objective_for_Doctor': str, 'Patient_Actor': dict,
                    'Physical_Examination_Findings': dict, 'Test_Results': dict,
                    'Correct_Diagnosis': str}
        if not all(isinstance(osce.get(field), kind) for field, kind in expected.items()):
            schema_errors.append(ordinal)
        if any(state(get(osce, field)) != 'PRESENT_NOT_CLINICALLY_REVIEWED' for field in expected):
            empty_core.append(ordinal)
        for key in FIELDS:
            value = get(osce, key)
            coverage[key][state(value)] += 1
            types[key]['missing' if value is MISSING else type(value).__name__] += 1
        actor = osce.get('Patient_Actor', {})
        if isinstance(actor, dict):
            actor_keys.update(actor.keys())
        narrative = ' '.join(str(v).casefold() for _, v in leaves(actor))
        narratives[' '.join(narrative.split())].append(ordinal)
        diagnosis = osce.get('Correct_Diagnosis')
        if isinstance(diagnosis, str):
            diagnoses.add(diagnosis.casefold().strip())
        matches, variant_matches = [], []
        for section in ('Objective_for_Doctor', 'Patient_Actor',
                        'Physical_Examination_Findings', 'Test_Results'):
            for path, value in leaves(osce.get(section), section):
                leaf_states[state(value)] += 1
                # Test keys can themselves contain disease labels, so inspect both.
                if lexical_hit(diagnosis, value) or lexical_hit(diagnosis, path.replace('_', ' ')):
                    matches.append({'section': section, 'path': path})
                for variant in label_variants(diagnosis):
                    if lexical_hit(variant, value) or lexical_hit(variant, path.replace('_', ' ')):
                        variant_matches.append({'section': section, 'path': path,
                                                'variant': variant, 'status': 'REVIEW_REQUIRED'})
        tests = osce.get('Test_Results', {})
        exams = osce.get('Physical_Examination_Findings', {})
        nt = len(tests) if isinstance(tests, dict) else 0
        ne = len(exams) if isinstance(exams, dict) else 0
        test_branches += nt
        exam_branches += ne
        ns = len(actor.get('Symptoms', {}).get('Secondary_Symptoms', [])) if isinstance(actor, dict) and isinstance(actor.get('Symptoms'), dict) else 0
        symptom_values += ns
        pub = {'ordinal': ordinal, 'test_top_level_branches': nt,
               'physical_top_level_branches': ne,
               'literal_reference_in_sections': sorted({m['section'] for m in matches}),
               'exploratory_variant_in_sections': sorted({m['section'] for m in variant_matches}),
               'unprovided_leaf_count': sum(state(v) == 'EXPLICIT_UNPROVIDED' for _, v in leaves(osce)),
               'source_time_or_release_policy_verified': False}
        public_cases.append(pub)
        private_cases.append({'ordinal': ordinal, 'source_record_sha256': sha(canonical(row)),
                              'literal_matches': matches, 'exploratory_matches': variant_matches,
                              'reference_for_private_review_only': diagnosis})
    sections = ('Objective_for_Doctor', 'Patient_Actor', 'Physical_Examination_Findings', 'Test_Results')
    summary = {'schema_version': 'locdi.c3_10.agentclinic_inventory.v1',
               'file_bytes': len(data), 'file_sha256': sha(data), 'records': len(rows),
               'parse_errors': errors, 'schema_error_ordinals': schema_errors,
               'cases_with_empty_or_unprovided_core_values': empty_core,
               'root_field_counts': dict(root_keys), 'actor_field_counts': dict(actor_keys),
               'field_coverage': {k: dict(v) for k, v in coverage.items()},
               'field_types': {k: dict(v) for k, v in types.items()},
               'empty_or_unprovided_leaf_states': dict(leaf_states),
               'duplicate_exact_record_groups': [v for v in hashes.values() if len(v) > 1],
               'duplicate_normalized_patient_narrative_groups': [v for v in narratives.values() if len(v) > 1],
               'semantic_duplicate_review': 'NOT_REVIEWED',
               'unique_diagnostic_label_strings': len(diagnoses),
               'test_top_level_branch_count': test_branches,
               'physical_top_level_branch_count': exam_branches,
               'secondary_symptom_strings': symptom_values,
               'literal_reference_case_counts': {s: sum(s in c['literal_reference_in_sections'] for c in public_cases) for s in sections},
               'exploratory_reference_case_counts_REVIEW_REQUIRED': {s: sum(s in c['exploratory_variant_in_sections'] for c in public_cases) for s in sections},
               'leak_check_scope': 'Literal labels in keys/values plus separate exploratory parenthesis variants; not semantic/clinical equivalence or a completed exposure policy',
               'case_flags': public_cases,
               'clinical_reference_approved': False, 'actual_llm_calls': 0}
    return summary, {'cases': private_cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', required=True)
    parser.add_argument('--official', required=True)
    parser.add_argument('--public-dir', required=True)
    parser.add_argument('--private-dir', required=True)
    args = parser.parse_args()
    public, private = Path(args.public_dir).resolve(), Path(args.private_dir).resolve()
    if not public.is_relative_to(ROOT/'outputs/research_evaluation') or not private.is_relative_to(ROOT/'outputs/private'):
        parser.error('OUTPUT_SCOPE')
    local_data, official_data = Path(args.local).read_bytes(), Path(args.official).read_bytes()
    comparison = compare(local_data, official_data)
    summary, details = audit(official_data)
    atomic_rows(public, 'source_comparison.json', comparison, jsonl=False)
    atomic_rows(public, 'dataset_inventory.json', summary, jsonl=False)
    atomic_rows(private, 'lexical_review_details.json', details, jsonl=False)
    print(json.dumps({'records': summary['records'], 'ordered_records_equal': comparison['ordered_parsed_records_equal'],
                      'parse_errors': len(summary['parse_errors']), 'actual_llm_calls': 0}))


if __name__ == '__main__':
    main()
