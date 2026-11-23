"""C3-04 offline bridge audit, new namespaces only; no operational review import."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.clinical_review_bridge import PrivateContextResolver, EvidenceStore, admit_reviews, bridge_contract
from app.locdi.routing_policy_analysis import build as build_policy, verify_manifest
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load, sha, verify as verify_d7
from app.locdi.source_replay import atomic_rows
from app.locdi.structural_release import preservation
from app.locdi.research_protocol_audit import inspect_rc
from scripts.audit_c3_01 import privacy_check


def analyze():
    resolver = PrivateContextResolver.from_actual()
    policy_outputs, private_links, evidence = build_policy()
    schema = load(ROOT/'outputs/research_evaluation/c3_03_v1/c3_04_handoff.json')['schema_comparison']
    production = admit_reviews(resolver, EvidenceStore({}), [], production=True)
    sys.path.insert(0, str(ROOT/'tests/locdi'))
    from test_clinical_review_bridge import counterexamples
    examples = counterexamples()
    if examples['status'] != 'PASS':
        raise ValueError('TEST_ONLY_BRIDGE_CONTRACT_FAILED')
    actual = policy_outputs['existing_response_groups']
    if (actual['response_count'], actual['supported'], actual['unresolved'], actual['collected_complete_groups'],
        actual['supported_only_complete_groups'], actual['independently_reviewed_complete_groups'], actual['gate_accepted_fit_groups']) != (75, 22, 53, 15, 0, 0, 0):
        raise ValueError('REAL_BASELINE_CHANGED')
    evidence['new_dependency_code'] = {n: sha(ROOT/n) for n in (
        'validation/review_submission_contract.py', 'validation/research_artifact_gate.py',
        'app/locdi/clinical_review_protocol.py', 'app/locdi/routing_policy_analysis.py', 'app/locdi/integrated_approval.py')}
    source = resolver.public_audit()
    source.update(groups=actual, split_independence=policy_outputs['split_independence_analysis'],
        pinned_input_scope='Preparation/Run/review files, current Registry/Prompt, raw50 and D7 manifests; canonical snapshot seal; no raw wire reconstruction')
    sequence = {'schema_version': 'locdi.c3_04.sequence.v1', 'required': [
        'Independent authenticated response-blinded Reference before response review',
        'Immutable original response fingerprint and recorded_at', 'Trusted submission receipt sequence and timestamp',
        'Reference receipt before response reviewer timestamp', 'Same explicit reference version and policy',
        'Blinding/access evidence verified by external authority, not inferred from timestamps'],
        'production_blinding': 'NOT_VERIFIED',
        'test_only_results': [r for r in examples['cases'] if r['case'] in ('response_before_reference', 'blind_not_verified', 'reference_receipt_after_response')],
        'live_access_ledger_connected': False}
    verification = {'schema_version': 'locdi.c3_04.evidence_contract.v1',
        'integrity': 'Byte SHA for pinned files; canonical JSON fingerprint for immutable submitted documents and sidecars',
        'authenticity': 'Independent authority receipt required; self-hash is not authentication',
        'identity': 'Dataset/patient/visit/decision/case/agent/model/prompt/registry/observation/memory/source/ledger/catalog/split pinned',
        'missing_evidence': 'BLOCKED; no inferred fields/default label/reviewer/verdict',
        'five_axes': bridge_contract()['axes'],
        'auxiliary_axes': 'Separate evidence retained; non-diagnosis REVIEWED requires separate reference adapter, currently BLOCKED',
        'legacy_gate_semantics': 'Delegated to unchanged validators, never duplicated or relaxed',
        'original_submission_result': 'validate_submission remains authentication BLOCKED; independent envelope receipt does not rewrite it',
        'source_timing': 'Condition date, actual source availability, proxy and review timestamps remain separate'}
    authority = {'schema_version': 'locdi.c3_04.authority_contract.v1',
        'interface': 'validation.review_authority_bridge.AuthorityProvider.verify(subject_ref, fingerprint, requirements)',
        'production_status': 'AUTHENTICATION_NOT_CONFIGURED', 'production_provider_implemented': False,
        'test_provider': 'TestOnlyAuthority: explicit symbolic ledger, no real signature/credential verification',
        'required_checks': ['Identity', 'Qualification', 'Specialty/task suitability', 'Independence and conflict',
            'Policy and scope', 'Institutional authority', 'Issuer signature/approval source', 'Evidence lineage',
            'Validity at review and admission, revocation', 'Blinding evidence', 'Submission receipt', 'Adjudicator authority'],
        'rejected_as_approval': ['Name', 'HUMAN/DOCTOR', 'REVIEWED', 'Self SHA', 'Arbitrary signature/institution', 'TEST_ONLY success', 'LLM judgment'],
        'custom_provider_self_authorization': 'BLOCKED_NO_ENROLLMENT_OR_TRUST_REGISTRY', 'clinical_review_ready': False,
        'future_integration': 'Authoritative credential/issuer/key/institution registry, current revocation/time source, approval/access records; separate implementation/review required'}
    good = next(r['result'] for r in examples['cases'] if r['case'] == 'complete_symbolic_evidence')
    integration = {'schema_version': 'locdi.c3_04.legacy_integration.v1', 'production_admission': production,
        'test_only_success': good,
        'real75_baseline_gate_audit': actual['gate'],
        'baseline_scope': 'Existing read-only original Gate audit is separate from bridge forwarding; production bridge forwards zero unreviewed records',
        'call_chain': ['Verified envelope and explicit legacy sidecar', 'validate_evaluation_gate', 'validate_training_data',
            'to_fit_rows eligibility (TEST_ONLY transient)', 'Research Artifact Gate'],
        'all_original_constraints_preserved': True, 'real_fitting_calls': 0,
        'policy_A_D_B1': {'approval': None, 'A': 'All-five supported current reviewed reference/labels required',
            'D': 'New source/reference and responses separately versioned; existing75 immutable',
            'B1': 'Descriptive sidecar; no uncertainty-to-correctness conversion', 'B2_C': 'Separate policy revision approval required'}}
    blockers = {'schema_version': 'locdi.c3_04.blockers.v1', 'production_status': 'BLOCKED',
        'blocker_codes': ['AUTHORITY_UNVERIFIED', 'REVIEW_POLICY_NOT_APPROVED', 'MISSING_REQUIRED_EVIDENCE',
            'REFERENCE_NOT_APPROVED', 'NO_REVIEWED_COMPLETE_TRAIN_GROUP', 'ROUTING_VALIDATION_VOI_TRAIN_OVERLAP',
            'PRODUCTION_EVIDENCE_STORE_ENROLLMENT_NOT_IMPLEMENTED', 'AUXILIARY_REFERENCE_ADAPTER_NOT_IMPLEMENTED'],
        'policy_approved': False, 'clinical_reference_ready': False,
        'reference_timing_candidate010': 'REFERENCE_TIMING_NOT_REVIEWED; source begins four days later',
        'next_C3_05': ['Obtain real independent reviewer/source/reference and task-policy approval',
            'Enroll trusted credential, institutional authority, policy registry and blinding/access evidence source',
            'Implement audited private document admission and typed auxiliary reference adapters',
            'Review contemporaneous reference then response; preserve unresolved=null and all-five complete contract',
            'Define patient-independent combined validation; no fitting until all gates and separate approval']}
    outputs = {'schema_bridge_contract': {'schema_version': 'locdi.c3_04.schema_bridge.v1', 'bridge': bridge_contract(),
        'legacy_new_comparison': schema, 'conversion_scope': 'Lossless joined subdomain only; original submissions/legacy records copied unchanged',
        'sidecar_semantics': 'Explicit label/source/review/scope/judgment rows and lineage; nothing synthesized'},
        'source_identity_audit': source, 'evidence_verification_contract': verification, 'authority_provider_contract': authority,
        'review_sequence_checks': sequence, 'conversion_test_only_results': examples,
        'legacy_gate_integration': integration, 'unresolved_blockers': blockers}
    private_links['purpose'] = 'PRIVATE_IDENTITY_LINKS_ONLY_NO_CLINICAL_REVIEW_RESULTS'
    private_links['schema_version'] = 'locdi.private.c3_04_identity_links.v1'
    private_links['resolver_context_fingerprint'] = resolver.public_audit()['context_fingerprint']
    return outputs, private_links, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', action='store_true'); parser.add_argument('--regression', action='store_true')
    parser.add_argument('--namespace', default='c3_04_v1'); args = parser.parse_args(argv)
    if args.regression and not args.save: parser.error('--regression requires new --save namespace')
    def guard(event, arguments):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'): raise RuntimeError('C3_04_OFFLINE')
    sys.addaudithook(guard)
    try:
        root = private = None
        if args.save:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}', args.namespace): raise ValueError('INVALID_NAMESPACE')
            root = ROOT/'outputs/research_evaluation'/args.namespace
            private = ROOT/'outputs/private/research_evaluation'/args.namespace
            if not all(p.resolve().is_relative_to(ROOT.resolve()) for p in (root, private)):
                raise ValueError('OUTPUT_REDIRECTED')
            if root.exists() or private.exists(): raise FileExistsError('NAMESPACE_ALREADY_USED')
            root.mkdir(parents=True, exist_ok=False); private.mkdir(parents=True, exist_ok=False)
        outputs, links, evidence = analyze()
        if root:
            for name, value in outputs.items(): atomic_rows(root, name+'.json', value, jsonl=False)
            atomic_rows(private, 'source_links.json', links, jsonl=False)
        print(json.dumps({'stage': 'ANALYSIS_SAVED' if root else 'READ_ONLY', 'test_only_cases': outputs['conversion_test_only_results']['case_count'],
            'production_admission': outputs['legacy_gate_integration']['production_admission']['status'],
            'actual_generation_calls': 0}), flush=True)
        regression = {'status': 'NOT_CHECKED'}
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            folder = root/'regression'; folder.mkdir(exist_ok=False)
            regression = run_regression(folder); atomic_rows(folder, 'result.json', regression, jsonl=False)
        d7, rc = verify_d7(full=False), inspect_rc()
        kept = preservation(load(ROOT/'outputs/private/c3_04_baseline_hashes_v1.json'))
        doc = next((ROOT/'docs').rglob('C3_04_CLINICAL_REVIEW_BRIDGE_AND_AUTHENTICATION_REPORT.md'), None)
        privacy = privacy_check(root, doc) if root else {'status': 'NOT_CHECKED_NOT_SAVED'}
        passed = outputs['conversion_test_only_results']['status'] == 'PASS' and all(r['status'] == 'PASS' for r in (d7, rc, kept))
        if root: passed = passed and privacy['status'] == 'PASS'
        if args.regression: passed = passed and regression['status'] == 'PASS'
        ready = passed and root is not None and args.regression
        result = {'schema_version': 'locdi.c3_04.readiness.v1',
            'C3_04_BRIDGE_CONTRACT_READY': 'PASS_DEVELOPMENT_ONLY' if ready else 'INCOMPLETE' if passed else 'FAIL',
            'PRODUCTION_AUTHENTICATION_READY': False, 'CLINICAL_REFERENCE_READY': False,
            'ROUTING_ARTIFACT_READY': False, 'OFFICIAL_B0_B4_READY': False, 'ROUTING_POLICY_APPROVED': False,
            'APPROVED_POLICY': None, 'regression': regression, 'D7': d7, 'RC': rc, 'preservation': kept, 'privacy': privacy,
            'input_evidence': evidence, 'private_index_sha256': sha(private/'source_links.json') if private else None,
            'actual_generation_calls': 0, 'healthbench_grading': 0, 'utility_fitting': 0,
            'voi_probability_estimation': 0, 'clinical_verdicts_generated': 0,
            'test_only_scope': 'Symbolic declarations; existing regression may exercise old in-memory fitting fixtures; no research data fitting'}
        if root: atomic_rows(root, 'readiness_summary.json', result, jsonl=False)
        print(json.dumps({k:result[k] for k in ('C3_04_BRIDGE_CONTRACT_READY','regression','preservation','privacy')}, ensure_ascii=True), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        print(json.dumps({'C3_04_BRIDGE_CONTRACT_READY': 'FAIL', 'error_type': type(exc).__name__,
            'error_code': 'BRIDGE_AUDIT_OR_NEW_NAMESPACE_UNAVAILABLE', 'actual_generation_calls': 0}), flush=True)
        return 2


if __name__ == '__main__': sys.exit(main())
