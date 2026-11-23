"""Offline C3-07 source inventory and optional explicitly pinned private raw audit."""
import argparse
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load, sha, verify as verify_freeze
from app.locdi.source_replay import atomic_rows
from app.locdi.reference_source_resolution import source_inventory, final_state
from app.locdi.reference_raw_validation import validate_manifest, safe_public, inventory_local_candidate
from app.locdi.public_reference_preparation import acquisition_plans
from app.locdi.external_reference_readiness import evaluate_gate
from app.locdi.structural_release import preservation
from app.locdi.research_protocol_audit import inspect_rc
from scripts.prepare_c3_06 import verify_prior


def verify_c3_06():
    from app.locdi.routing_policy_analysis import verify_manifest
    manifest = load(ROOT/'outputs/research_evaluation/c3_06_v2/completion_manifest.json')
    checked = verify_manifest(ROOT/'outputs/research_evaluation/c3_06_v2/completion_manifest.json')
    p = ROOT/'outputs/private/research_evaluation/c3_06_v2/completion_index.json'
    if sha(p) != manifest['private_completion_index_sha256']:
        raise ValueError('C3_06_PRIVATE_INDEX_HASH')
    index = load(p)
    if digest({k:v for k,v in index.items() if k != 'fingerprint'}) != manifest['private_completion_index_fingerprint']:
        raise ValueError('C3_06_PRIVATE_INDEX_FINGERPRINT')
    for mapping in ('files', 'prior_attempt_files'):
        for name, expected in index.get(mapping, {}).items():
            target = (ROOT/name).resolve()
            if not target.is_relative_to(ROOT.resolve()) or sha(target) != expected:
                raise ValueError('C3_06_PRIVATE_FILE_HASH')
    return checked


def tests(folder):
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'), pattern='test_reference_raw_validation.py')
    with (folder/'test_only.txt').open('x', encoding='utf8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests': result.testsRun,
            'pass': result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
            'fail': len(result.failures), 'errors': len(result.errors), 'skip': len(result.skipped),
            'fixtures': 'TEST_ONLY nonclinical symbols/bytes; not acquired reference records',
            'source_sha256': sha(ROOT/'tests/locdi/test_reference_raw_validation.py')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--namespace', default='c3_07_v1')
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--raw-manifest')
    parser.add_argument('--raw-manifest-sha256')
    parser.add_argument('--local-candidate', help='Exact named file for quarantined shape inventory only; never adapter admission')
    args = parser.parse_args(argv)
    if bool(args.raw_manifest) != bool(args.raw_manifest_sha256):
        parser.error('Explicit private manifest path AND out-of-band SHA256 required')
    if args.regression and not args.save:
        parser.error('--regression requires --save')
    def offline(event, arguments):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'):
            raise RuntimeError('C3_07_OFFLINE_NO_NETWORK')
    sys.addaudithook(offline)
    try:
        public = safe_public({'schema_version': 'locdi.c3_07.raw_validation.v1', 'status': 'BLOCKED',
            'row_count': 0, 'split_counts': {}, 'raw_sha256': None,
            'blocker_codes': ['EXPLICIT_LAWFUL_RAW_AND_PINNED_EVIDENCE_REQUIRED'],
            'external_evaluation_ready': False, 'clinical_metrics': 'NOT_EVALUATED'})
        private_raw = None
        candidate = candidate_private = None
        if args.local_candidate:
            candidate, candidate_private = inventory_local_candidate(args.local_candidate)
        if args.raw_manifest:
            public, private_raw = validate_manifest(args.raw_manifest, args.raw_manifest_sha256)
        flags = final_state(private_raw is not None, private_raw is not None)
        if not args.save:
            print(json.dumps({'mode': 'DRY_RUN', **flags, 'raw': public,
                              'local_candidate': candidate, 'sources': source_inventory()}))
            return 0
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}', args.namespace):
            raise ValueError('INVALID_NAMESPACE')
        root = ROOT/'outputs/research_evaluation'/args.namespace
        private = ROOT/'outputs/private/research_evaluation'/args.namespace
        if not root.resolve().is_relative_to((ROOT/'outputs/research_evaluation').resolve()) or not private.resolve().is_relative_to((ROOT/'outputs/private').resolve()):
            raise ValueError('OUTPUT_PATH_ESCAPE')
        if root.exists() or private.exists():
            raise ValueError('NAMESPACE_ALREADY_EXISTS')
        root.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
        prior = verify_prior(); prior['c3_06_v2'] = verify_c3_06()
        inventory = {'schema_version': 'locdi.c3_07.source_inventory.v1', 'sources': source_inventory(),
            'actual_new_raw_downloads': 0, 'documentary_note_hash_is_raw_hash': False,
            'metadata_failures': ['Official Drive preview unavailable', 'Pinned HF file tree/API/blob unavailable; no data byte request'],
            'rights_approval_created': False}
        atomic_rows(root, 'source_resolution.json', inventory, jsonl=False)
        atomic_rows(root, 'raw_validation.json', public, jsonl=False)
        if candidate:
            atomic_rows(root, 'local_candidate_inventory.json', candidate, jsonl=False)
            atomic_rows(private, 'local_candidate_reference.json', candidate_private, jsonl=False)
        # Private input tasks and scoring keys have separate files, not a public bundle.
        if private_raw:
            atomic_rows(private, 'model_tasks.json', private_raw.pop('tasks_by_original_split'), jsonl=False)
            atomic_rows(private, 'scoring_answer_keys.json', private_raw.pop('answer_keys_by_original_split'), jsonl=False)
            atomic_rows(private, 'raw_lineage.json', private_raw, jsonl=False)
        notes = {'schema_version': 'locdi.c3_07.documentary_notes.v1', 'inventory_fingerprint': digest(inventory),
            'sources': inventory['sources'], 'rights_evidence': [], 'actual_approvals': [],
            'note': 'Own documentary observations; hashes do not authenticate remote file bytes or grant rights'}
        atomic_rows(private, 'source_notes.json', notes, jsonl=False)
        gates = [evaluate_gate(plan) for plan in acquisition_plans()]
        atomic_rows(root, 'existing_gate_recheck.json', {'datasets': gates}, jsonl=False)
        atomic_rows(root, 'raw_input_contract.json', {
            'schema_version': 'locdi.c3_07.raw_manifest_spec.v1',
            'input': 'Operator-supplied private path plus SHA256 argument; no filesystem discovery',
            'fields': ['source_identity', 'expected_source_identity', 'splits'],
            'identity_fields': ['source_id', 'revision'],
            'split_entry_fields': ['plan', 'raw_evidence', 'index_evidence', 'option_count'],
            'evidence_contract': 'Exact C3-06 acquisition plan, pinned acquisition receipt, PinnedApproval and pinned index contract',
            'missing_receipt': 'BLOCKED; never fabricate a download receipt for an arbitrary local file',
            'supported': 'Single original JSONL per split <=64MiB; explicit4/5 options; exact original split',
            'unsupported': ['Drive authentication/redirect', 'Parquet transformation', 'ZIP-to-JSONL admission without lineage'],
            'zip_inspection': 'Bounded read-only inspection; no extraction or adapter promotion',
            'answer_key_policy': 'Private scoring file only; model_tasks never contains scoring keys',
            'clinical_routing_voi_transfer': 'FORBIDDEN', 'model_execution': 'FORBIDDEN'}, jsonl=False)
        folder = root/'verification'; folder.mkdir()
        test_result = tests(folder)
        atomic_rows(root, 'test_only_verification.json', test_result, jsonl=False)
        print(json.dumps({'stage': 'OFFLINE_PREPARED', **flags, 'test_only': test_result}), flush=True)
        regression = {'status': 'NOT_CHECKED'}
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            folder = root/'regression'; folder.mkdir()
            regression = run_regression(folder)
            atomic_rows(folder, 'result.json', regression, jsonl=False)
        kept = preservation(load(ROOT/'outputs/private/c3_07_baseline_hashes_v1.json'))
        frozen = verify_freeze(full=False); rc = inspect_rc()
        passed = all(r['status'] == 'PASS' for r in (kept, frozen, rc, test_result))
        passed = passed and (not args.regression or regression['status'] == 'PASS')
        summary = {'schema_version': 'locdi.c3_07.readiness.v1', **flags,
            'development_verification': 'PASS' if passed and args.regression else 'INCOMPLETE',
            'test_only': test_result, 'regression': regression, 'preservation': kept, 'D7': frozen, 'RC': rc,
            'prior_manifests': prior, 'new_raw_downloads': 0, 'actual_generation_calls': 0,
            'actual_grader_calls': 0, 'utility_fitting': 0, 'voi_probability_estimation': 0,
            'clinical_verdicts_generated': 0, 'completion_marker': 'Separate last manifest binds final report and verification'}
        atomic_rows(root, 'readiness_summary.json', summary, jsonl=False)
        print(json.dumps({'development_verification': summary['development_verification'],
                          'regression': regression, 'preservation': kept}), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        # Messages can contain source text/paths; only fixed sanitized type escapes.
        print(json.dumps({'status': 'BLOCKED', 'error_type': type(exc).__name__,
                          'blocker_codes': ['RAW_OR_PREPARATION_CONTRACT_REJECTED'],
                          'actual_generation_calls': 0, 'actual_downloads': 0}), flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
