"""C4-06 read-only review audit. Never creates clinical judgments or scores."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.llm_setup import digest
from app.locdi.source_enrichment import validate_candidate, validate_partial_review
from app.locdi.dataset_loader import load_cases
from scripts.enrich_full_cohort_sources import DATA, partial_evaluate, write


def blank_progress(manifest, packets, review):
    """Inventory the existing blank, not a validator for submitted reviews."""
    ids = ['OriginalCase%03d' % i for i in range(1, 51)]
    if ([e['snapshot_id'] for e in manifest['packets']] != ids
            or [p['snapshot_id'] for p in packets] != ids
            or [r['snapshot_id'] for r in review['snapshots']] != ids):
        raise ValueError('ALL50_ORDER_REQUIRED')
    if review['manifest_fingerprint'] != manifest['fingerprint']:
        raise ValueError('REFERENCE_MANIFEST_BINDING')
    inventory = []
    for packet, row in zip(packets, review['snapshots']):
        if (row['packet_fingerprint'] != digest(packet)
                or row['query_fingerprint'] != packet['query_candidate']['query_fingerprint']):
            raise ValueError('REFERENCE_PACKET_BINDING')
        expected = [r['memory_id'] for r in packet['candidate_memories']]
        if [r['memory_id'] for r in row['records']] != expected or len(set(expected)) != len(expected):
            raise ValueError('POOL_RECORD_ORDER')
        if (row['review_status'] != 'UNREVIEWED' or row['query_suitable'] is not None
                or row['reviewer_private_ref'] is not None or row['reviewed_at'] is not None
                or any(r[k] is not None for r in row['records'] for k in ('relevant', 'stale', 'conflict'))):
            raise ValueError('NOT_BLANK_REQUIRES_REAL_REFERENCE_VALIDATION')
        inventory.append({'case_ref': packet['snapshot_id'], 'query_status': 'NOT_REVIEWED',
                          'pool_records': len(expected), 'memory_judged_count': 0,
                          'evaluable': False, 'blocker': 'INDEPENDENT_REFERENCE_REQUIRED'})
    if sum(r['pool_records'] for r in inventory) != 989:
        raise ValueError('ORIGINAL989_REQUIRED')
    return inventory


def check_hashes(pins):
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected:
            raise ValueError('PROTECTED_OR_SOURCE_HASH_MISMATCH')
    return len(pins)


def audit(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'outputs/research_evaluation'):
        raise ValueError('PUBLIC_OUTPUT_REQUIRED')
    if output.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    private = ROOT / 'outputs/private/research_evaluation/c4_05_v1'
    old = load(ROOT / 'outputs/research_evaluation/c4_05_v1/reports/output_manifest.json')
    if digest({k: v for k, v in old.items() if k != 'fingerprint'}) != old['fingerprint']:
        raise ValueError('C4_05_MANIFEST_FINGERPRINT')
    index_path = private / 'private_index.json'
    if sha(index_path) != old['private_index_sha256']:
        raise ValueError('PRIVATE_INDEX_HASH')
    counts = {'protected_baseline': check_hashes(load(private / 'baseline_hashes.json')),
              'c4_05_public_code_report': check_hashes(old['files']),
              'c4_05_private': check_hashes(load(index_path)['files']),
              'source_assets': check_hashes(load(private / 'all_source_hashes.json')),
              'additional_source_assets': check_hashes(load(private / 'additional_source_asset_hashes.json'))}
    root = private / 'judged_pool_v1'
    manifest = load(root / 'manifest.json')
    if digest({k: v for k, v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('POOL_MANIFEST_FINGERPRINT')
    packets = []
    original = load_cases(DATA)
    for entry, case in zip(manifest['packets'], original):
        if Path(entry['file']).name != entry['file'] or sha(root / entry['file']) != entry['sha256']:
            raise ValueError('PACKET_HASH')
        packet = load(root / entry['file'])
        if digest(packet) != entry['packet_fingerprint']:
            raise ValueError('PACKET_FINGERPRINT')
        if packet['private_identity'] != case['agent_visible']['longitudinal']:
            raise ValueError('IDENTITY_BINDING')
        validate_candidate(packet['query_candidate'], case['environment']['current_encounter_events'],
                           case['agent_visible']['longitudinal'])
        packets.append(packet)
    template = load(root / 'review_results_blank.json')
    inventory = blank_progress(manifest, packets, template)
    rejected = 0
    for packet, review in zip(packets, template['snapshots']):
        try:
            validate_partial_review(packet, review)
        except ValueError as exc:
            if str(exc) != 'INDEPENDENT_PARTIAL_REVIEW_REQUIRED':
                raise
            rejected += 1
        else:
            raise ValueError('UNREVIEWED_GATE_BYPASS')
    with patch('app.core.longitudinal_memory.PatientMemoryStore.retrieve') as retrieve:
        try:
            partial_evaluate(root, None, None, output / 'UNUSED_QUALITY_OUTPUT')
        except ValueError as exc:
            missing_reason = str(exc)
        else:
            raise ValueError('MISSING_REFERENCE_GATE_BYPASS')
        if retrieve.call_count:
            raise ValueError('UNAUTHORIZED_QUALITY_RETRIEVAL')
    memory_count = sum(len(v['records']) for c in original for v in c['agent_visible']['past_visits'])
    candidate_count = sum(1 for _ in (private / 'query_candidates.jsonl').open(encoding='utf8'))
    if len(original) != 50 or memory_count != 72590 or candidate_count != 250:
        raise ValueError('COHORT_OR_MEMORY_CHANGED')
    # A path inventory is not approval of any discovered file.
    review_dirs = [ROOT / 'outputs/private/research_evaluation' / name
                   for name in ('c4_05_reviews', 'c4_06_reviews')]
    found = [p for d in review_dirs if d.exists() for p in d.rglob('*.json')]
    summary = {'status': 'C4_06_REVIEW_PREPARED', 'registered_cases': 50,
               'original_memory_records': memory_count, 'source_snapshots': candidate_count,
               'pool_candidates': 989, 'query_reviewed_count': 0, 'query_suitable_count': 0,
               'memory_judged_count': 0, 'remaining_pool_unjudged': 989,
               'remaining_full_unjudged': 72590, 'evaluable_case_count': 0, 'blocked_case_count': 50,
               'quality_status': 'NOT_RUN', 'quality_metrics': 'NOT_EVALUATED',
               'full_pool_recall': 'NOT_IDENTIFIABLE_NOT_COMPUTED',
               'expected_review_directory_json_count': len(found),
               'reference_scope': 'EXISTING_BLANK_ONLY_NO_REAL_REFERENCE_SUBMITTED',
               'template_sha256': sha(root / 'review_results_blank.json'),
               'dataset_manifest_sha256': sha(private / 'dataset_manifest.json'),
               'pool_manifest_fingerprint': manifest['fingerprint'],
               'hash_checks': counts, 'protected_changed_or_missing': 0,
               'blank_queries_rejected': rejected, 'missing_reference_blocker': missing_reason,
               'quality_retrieval_calls': 0, 'llm_generations': 0, 'utility_fitting': 0}
    if found:
        raise ValueError('REVIEW_FILES_FOUND_REQUIRE_EXPLICIT_SELECTION_NO_MERGE')
    write(output / 'reports/review_progress.json', summary)
    write(output / 'reports/case_review_inventory.json', inventory)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-namespace', default='c4_06_v1')
    args = parser.parse_args()
    import re
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', args.output_namespace):
        raise ValueError('INVALID_NAMESPACE')
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError('OFFLINE_ONLY'))
                    if event in ('socket.connect', 'socket.connect_ex', 'socket.bind') else None)
    print(json.dumps(audit(ROOT / 'outputs/research_evaluation' / args.output_namespace)))
