"""Private, fail-closed C3-04 evidence join. Never infers clinical judgments.

Legacy rows, label contents, judgments and input scopes must be EXPLICIT supplied
evidence. This bridge copies them after independent binding checks; it does not
fill missing fields or change existing schemas. Production is blocked until a
trusted authority/policy registry is actually installed.
"""
from copy import deepcopy
from datetime import timedelta

from jsonschema import Draft202012Validator
from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_freeze import load, sha
from validation.review_submission_contract import validate_submission
from validation.review_authority_bridge import timestamp, verify_authority
from validation.routing_correctness_gate import POLICY, REVIEW_SCHEMA, empty_review_results, validate_evaluation_gate
from validation.routing_training_data import to_fit_rows
from validation.research_artifact_gate import assess_artifacts

AXES = ('REFERENCE_VALIDITY', 'RESPONSE_CORRECTNESS', 'MEMORY_RELEVANCE', 'SOURCE_TIMING', 'ROUTING_APPROPRIATENESS')
VALIDITY = ('CURRENT_DECISION_REFERENCE_VALID', 'RETROSPECTIVE_OUTCOME_ONLY', 'REFERENCE_REJECTED',
            'INSUFFICIENT_INFORMATION', 'NOT_REVIEWABLE', 'NOT_REVIEWED')
ENVELOPE_FIELDS = {'schema_version', 'purpose', 'submission_ref', 'submission_fingerprint', 'identity',
    'context_fingerprint', 'axis', 'reference_validity', 'reference_timing', 'reference_envelope_ref',
    'reference_label_ref', 'legacy_record_ref', 'source_record_ref', 'judgment_ref', 'scope_evidence_ref',
    'procedure_version', 'policy_fingerprint', 'review_timestamp', 'uncertainty_status', 'adjudication_status',
    'blinding_status', 'axis_verdict_ref', 'evidence_lineage'}


class EvidenceStore:
    """Private immutable document store; integrity is NOT authenticity.

    No remote URL resolution, arbitrary file reads, or path from a submission.
    Production registration/import remains blocked pending trust enrollment.
    """
    def __init__(self, documents):
        self._documents = deepcopy(documents)
        self._hashes = {k: digest(v) for k, v in self._documents.items()}

    def get(self, reference):
        if reference not in self._documents:
            raise ValueError('MISSING_REQUIRED_EVIDENCE')
        value = self._documents[reference]
        if digest(value) != self._hashes[reference]:
            raise ValueError('IMMUTABLE_EVIDENCE_CHANGED')
        return deepcopy(value)

    def fingerprint(self, reference):
        return digest(self.get(reference))


class PrivateContextResolver:
    """Pinned original context, not identity strings submitted by a reviewer."""
    def __init__(self, data, options, *, dataset_version, test_only, anchors=None):
        if test_only is not True:
            raise ValueError('USE_PINNED_ACTUAL_LOADER')
        if data.get('purpose') != 'TEST_ONLY' or options.get('allow_test_only') is not True:
            raise ValueError('TEST_ONLY_CONTEXT_REQUIRED')
        self._initialize(data, options, dataset_version, True, anchors or {})

    def _initialize(self, data, options, version, test_only, anchors):
        self._data, self._options = deepcopy(data), deepcopy(options)
        self.dataset_version, self.test_only = version, test_only
        self._anchors = deepcopy(anchors)
        self._seal = digest([self._data, self._options, version, test_only, self._anchors])

    @classmethod
    def from_actual(cls):
        from app.locdi.evaluation_gate_audit import actual_gate_inputs
        from app.locdi.specialty_collection_analysis import analyze_run
        from app.locdi.routing_policy_analysis import verify_manifest
        for namespace, marker in (('c3_01_v3', 'analysis_manifest.json'), ('c3_02_v1', 'completion_manifest.json'),
                                  ('c3_03_v1', 'completion_manifest.json')):
            verify_manifest(ROOT/'outputs/research_evaluation'/namespace/marker)
        data, options = actual_gate_inputs()
        summary, _ = analyze_run('phase24b_collect_001', 'phase24b_candidates_v2')
        if summary['run_id'] != 'phase24b_collect_001' or summary['integrity_issue_codes']:
            raise ValueError('SOURCE_CONTEXT_MISMATCH')
        prefixes = ('outputs/private/routing_collection/phase24b_candidates_v2',
                    'outputs/private/routing_collection/phase24b_collect_001',
                    'outputs/private/routing_collection_review/phase24c_review_v2',
                    'outputs/private/unresolved_analysis/phase24d_analysis_v1')
        anchors = {p.relative_to(ROOT).as_posix(): sha(p) for folder in prefixes for p in (ROOT/folder).rglob('*') if p.is_file()}
        from app.locdi.routing_training_setup import REGISTRY
        anchors[REGISTRY.relative_to(ROOT).as_posix()] = sha(REGISTRY)
        for p in (ROOT/'prompts').rglob('*'):
            if p.is_file(): anchors[p.relative_to(ROOT).as_posix()] = sha(p)
        for name in ('datasets/raw/longitudinal_benchmark_50.jsonl',
                     'datasets/releases/locdi_dataset_v1.0/freeze_manifest.json',
                     'outputs/private/dataset_release/D7_locdi_dataset_v1.0/manifest.json'):
            anchors[name] = sha(ROOT/name)
        obj = cls.__new__(cls)
        obj._initialize(data, options, 'phase24b_candidates_v2', False, anchors)
        return obj

    def verify(self):
        if digest([self._data, self._options, self.dataset_version, self.test_only, self._anchors]) != self._seal:
            raise ValueError('IMMUTABLE_CONTEXT_CHANGED')
        if any(not (ROOT/n).is_file() or sha(ROOT/n) != h for n, h in self._anchors.items()):
            raise ValueError('PINNED_SOURCE_CHANGED')

    def identity(self, decision_id, agent_id=None):
        self.verify()
        decisions = [d for d in self._data['decisions'] if d['decision_id'] == decision_id]
        if len(decisions) != 1:
            raise ValueError('SOURCE_CONTEXT_MISMATCH')
        d = decisions[0]
        ctx = next(c for c in self._options['decision_contexts'] if c['decision_id'] == decision_id)
        if any(ctx[k] != d[k] for k in ('patient_id', 'visit_id', 'decision_id')):
            raise ValueError('SOURCE_CONTEXT_MISMATCH')
        agent = None
        if agent_id is not None:
            agent = next((a for a in self._data['registry']['agents'] if a['agent_id'] == agent_id and a['role'] == 'specialty'), None)
            if agent is None or len([r for r in self._data['outcomes'] if r['decision_id'] == decision_id and r['agent_id'] == agent_id]) != 1:
                raise ValueError('SOURCE_CONTEXT_MISMATCH')
        identity = {'dataset_version': self.dataset_version, 'case_ref': ctx['case_ref'],
                    **{k: d[k] for k in ('patient_id', 'visit_id', 'decision_id')}, 'agent_id': agent_id}
        snapshot = {'identity': identity, 'source_fingerprint': ctx['source_fingerprint'],
            'observation_fingerprint': digest(d['observation']),
            'memory_fingerprint': digest(d['observation'].get('longitudinal_memory', {})),
            'condition_fingerprint': digest(ctx['reference_candidates']), 'visibility_fingerprint': digest(ctx['visibility']),
            'catalog_fingerprint': digest(ctx['diagnosis_catalog']), 'split_fingerprint': digest(self._data['split_manifest']),
            'registry_fingerprint': digest(self._data['registry']), 'generation_fingerprint': digest(d['generation']),
            'agent_binding': agent}
        return identity, snapshot, deepcopy(d), deepcopy(ctx)

    def resolve(self, envelope):
        supplied = envelope['identity']
        identity, snapshot, decision, ctx = self.identity(supplied['decision_id'], supplied['agent_id'])
        if supplied != identity or envelope['context_fingerprint'] != digest(snapshot):
            raise ValueError('SOURCE_CONTEXT_MISMATCH')
        return identity, snapshot, decision, ctx

    def public_audit(self):
        self.verify()
        return {'status': 'PASS', 'dataset_version': self.dataset_version, 'decision_count': len(self._data['decisions']),
            'response_count': len(self._data['outcomes']), 'anchor_file_count': len(self._anchors),
            'context_fingerprint': self._seal, 'scope': 'PINNED_PRIVATE_LINKS_NOT_CLINICAL_REFERENCE_APPROVAL',
            'clinical_reference_ready': False}


def _legacy_valid(record, definition):
    schema = {'$ref': '#/$defs/' + definition, '$defs': REVIEW_SCHEMA['$defs']}
    return Draft202012Validator(schema).is_valid(record)


def _check_envelope(ref, envelope, store, resolver, provider, policy_ref, *, production):
    """Return explicit evidence and blockers; no value inferred from diagnosis text."""
    codes = []
    result = None
    try:
        if set(envelope) != ENVELOPE_FIELDS or envelope['schema_version'] != 'locdi.review_envelope.v1':
            raise ValueError('TYPED_ENVELOPE_FIELDS_INVALID')
        if envelope['axis'] not in AXES or envelope['reference_validity'] not in VALIDITY:
            raise ValueError('AXIS_SEMANTIC_MISMATCH')
        if envelope['purpose'] != ('TEST_ONLY' if resolver.test_only else 'private_review'):
            raise ValueError('EVIDENCE_PURPOSE_MISMATCH')
        identity, snapshot, decision, ctx = resolver.resolve(envelope)
        lineage = envelope['evidence_lineage']
        pointers = ('submission_ref', 'reference_label_ref', 'legacy_record_ref', 'source_record_ref',
                    'judgment_ref', 'scope_evidence_ref', 'axis_verdict_ref', 'reference_envelope_ref')
        required_refs = {envelope[k] for k in pointers if envelope[k] is not None}
        if not isinstance(lineage, dict) or set(lineage) != required_refs or any(store.fingerprint(k) != h for k, h in lineage.items()):
            raise ValueError('EVIDENCE_LINEAGE_MISMATCH')
        submission = store.get(envelope['submission_ref'])
        if digest(submission) != envelope['submission_fingerprint']:
            codes.append('ORIGINAL_SUBMISSION_FINGERPRINT_MISMATCH')
        if envelope['axis'] not in ('REFERENCE_VALIDITY', 'RESPONSE_CORRECTNESS'):
            # Auxiliary five-axis evidence cannot be squeezed into old diagnosis fields.
            codes.extend(['AXIS_SEMANTIC_MISMATCH', 'LOSSLESS_CONVERSION_NOT_PROVEN'])
            verdict = store.get(envelope['axis_verdict_ref']) if envelope['axis_verdict_ref'] else None
            if not isinstance(verdict, dict) or verdict.get('verdict') is None:
                codes.append('NON_DIAGNOSIS_AXIS_VERDICT_MISSING')
            return None, sorted(set(codes))
        shape = validate_submission(submission)
        if shape['structural_status'] != 'READY':
            codes.append('ORIGINAL_SUBMISSION_SCHEMA_INVALID')
        # Its unconditional authentication block is not removed: our independent
        # provider receipt is a separate prerequisite, never a modified v1 result.
        kind = 'REFERENCE_REVIEW' if envelope['axis'] == 'REFERENCE_VALIDITY' else 'RESPONSE_JUDGMENT'
        if (submission['submission_kind'] != kind or submission['evaluation_target'] != 'diagnosis_correctness' or
                submission['review_case_ref'] != identity['case_ref'] or submission['decision_ref'] != identity['decision_id'] or
                submission['reference_source'] != decision['source_ref'] or submission['source_fingerprint'] != snapshot['source_fingerprint']):
            codes.append('SOURCE_CONTEXT_MISMATCH')
        if envelope['policy_fingerprint'] != digest(POLICY):
            codes.append('REVIEW_POLICY_NOT_APPROVED')
        if submission['review_policy_version'] != envelope['procedure_version'] or submission['review_timestamp'] != envelope['review_timestamp']:
            codes.append('PROCEDURE_OR_REVIEW_TIMESTAMP_MISMATCH')
        if submission['reference_timing_status'] != envelope['reference_timing']:
            codes.append('REFERENCE_TIMING_MISMATCH')
        if envelope['reference_validity'] != 'CURRENT_DECISION_REFERENCE_VALID':
            codes.extend(['REFERENCE_NOT_APPROVED', 'LOSSLESS_CONVERSION_NOT_PROVEN'])
        if envelope['uncertainty_status'] != 'NONE':
            codes.append('UNCERTAINTY_UNRESOLVED')
        if envelope['adjudication_status'] not in ('NO_DISAGREEMENT', 'RESOLVED_WITH_AUTHORITY'):
            codes.append('UNRESOLVED_DISAGREEMENT')
        scope = store.get(envelope['scope_evidence_ref'])
        if (type(scope.get('submitted_sequence')) is not int or scope['submitted_sequence'] < 1 or
                timestamp(scope.get('submitted_at')) < timestamp(envelope['review_timestamp'])):
            codes.append('SUBMISSION_RECEIPT_REQUIRED')
        requirements = {'reviewer_ref': submission['reviewer_ref'], 'axis': envelope['axis'],
            'policy_fingerprint': envelope['policy_fingerprint'], 'at': envelope['review_timestamp'],
            'submission_sequence': scope.get('submitted_sequence'),
            'requires_blinding': kind == 'REFERENCE_REVIEW',
            'requires_adjudicator': envelope['adjudication_status'] == 'RESOLVED_WITH_AUTHORITY'}
        auth = verify_authority(provider, ref, digest(envelope), requirements, production=production)
        codes.extend(auth['blocker_codes'])
        legacy = store.get(envelope['legacy_record_ref'])
        label = store.get(envelope['reference_label_ref'])
        if not _legacy_valid(legacy, 'reference' if kind == 'REFERENCE_REVIEW' else 'response'):
            codes.append('LEGACY_SCHEMA_INVALID_LOSSLESS_CONVERSION_NOT_PROVEN')
            return None, sorted(set(codes))
        if (any(legacy.get(k) != identity[k] for k in ('case_ref', 'patient_id', 'decision_id')) or
                legacy['reference_ref'] != label['label_ref'] or legacy['reference_fingerprint'] != label['label_fingerprint'] or
                submission['reference_fingerprint'] != label['label_fingerprint'] or digest(label['reference_content']) != label['label_fingerprint']):
            codes.append('REFERENCE_FINGERPRINT_OR_IDENTITY_MISMATCH')
        review = legacy['review']
        if any(submission[s] != review[t] for s, t in (('reviewer_ref', 'reviewer_ref'), ('reviewer_role', 'reviewer_kind'),
                ('reviewer_independent', 'independent'), ('review_policy_version', 'procedure_version'), ('review_timestamp', 'reviewed_at'))):
            codes.append('LOSSLESS_CONVERSION_NOT_PROVEN')
        decision_at = timestamp(ctx['encounter_started_at']) + timedelta(minutes=decision['minute'])
        if timestamp(submission['decision_at']) != decision_at:
            codes.append('SOURCE_CUTOFF_MISMATCH')
        if scope.get('context_fingerprint') != digest(snapshot) or scope.get('review_status') != 'REVIEWED':
            codes.append('INPUT_SCOPE_EVIDENCE_MISSING')
        if kind == 'REFERENCE_REVIEW':
            source = store.get(envelope['source_record_ref'])
            original_source = next(s for s in resolver._options['source_catalog'] if s['source_ref'] == decision['source_ref'])
            if {k:v for k,v in source.items() if k != 'review'} != {k:v for k,v in original_source.items() if k != 'review'}:
                codes.append('SOURCE_CONTEXT_MISMATCH')
            if (legacy['visit_id'] != identity['visit_id'] or legacy['source_ref'] != decision['source_ref'] or
                legacy['source_fingerprint'] != snapshot['source_fingerprint'] or
                legacy['observation_fingerprint'] != snapshot['observation_fingerprint'] or
                legacy['memory_fingerprint'] != snapshot['memory_fingerprint'] or
                legacy['condition_candidates_fingerprint'] != snapshot['condition_fingerprint'] or
                legacy['input_scope'] != scope.get('input_scope') or legacy['timing_status'] != envelope['reference_timing']):
                codes.append('SOURCE_CONTEXT_MISMATCH')
            if envelope['blinding_status'] != 'VERIFIED_BY_AUTHORITY' or legacy['response_blinded'] is not True:
                codes.append('BLINDING_NOT_VERIFIED')
            if (submission['clinical_evidence_rationale'] != legacy['clinical_evidence_rationale'] or
                    source['review'] != review or label['review'] != review):
                codes.append('LOSSLESS_CONVERSION_NOT_PROVEN')
            if submission['evidence_reference'] != label['reference_content']['evidence_ref']:
                codes.append('LOSSLESS_CONVERSION_NOT_PROVEN')
            result = {'kind': kind, 'legacy': legacy, 'label': label, 'source': source, 'submission': submission,
                      'envelope': envelope, 'auth': auth, 'reference': ref}
        else:
            outcome = next(r for r in resolver._data['outcomes'] if r['decision_id'] == identity['decision_id'] and r['agent_id'] == identity['agent_id'])
            if (legacy['agent_id'] != identity['agent_id'] or legacy['response_ref'] != outcome['response_ref'] or
                    submission['response_ref'] != outcome['response_ref'] or
                    legacy['response_fingerprint'] != outcome['response_fingerprint'] or
                    submission['response_fingerprint'] != outcome['response_fingerprint'] or
                    digest(outcome['response']) != outcome['response_fingerprint'] or
                    ctx['original_response_fingerprints'].get(outcome['response_ref']) != outcome['response_fingerprint']):
                codes.append('ORIGINAL_RESPONSE_FINGERPRINT_MISMATCH')
            if submission['certainty'] != outcome['response']['certainty'] or submission['response_recorded_at'] != outcome['recorded_at']:
                codes.append('RESPONSE_CERTAINTY_OR_TIMESTAMP_MISMATCH')
            axis = legacy['axes']['diagnosis_correctness']
            if (submission['correctness_label'] != axis['verdict'] or submission['decision_status'] != axis['status'] or
                submission['clinical_evidence_rationale'] != axis['clinical_evidence_rationale'] or
                submission['not_evaluable_reason'] != axis['not_evaluable_reason']):
                codes.append('AXIS_SEMANTIC_MISMATCH')
            for name, value in legacy['axes'].items():
                if value['status'] == 'NOT_EVALUABLE' and not value['not_evaluable_reason']:
                    codes.append('NOT_EVALUABLE_REASON_REQUIRED')
                if value['status'] == 'REVIEWED' and value['verdict'] is None:
                    codes.append('NON_DIAGNOSIS_AXIS_VERDICT_MISSING' if name != 'diagnosis_correctness' else 'AXIS_SEMANTIC_MISMATCH')
                if name != 'diagnosis_correctness' and value['status'] == 'REVIEWED':
                    codes.append('NON_DIAGNOSIS_REVIEW_REQUIRES_SEPARATE_REFERENCE_ADAPTER')
            if (submission['certainty'] == 'unresolved' or outcome['response']['diagnosis'] == 'Unresolved') and axis['verdict'] is not None:
                codes.append('UNRESOLVED_DIAGNOSIS_BOOLEAN_FORBIDDEN')
            judgment = store.get(envelope['judgment_ref']) if envelope['judgment_ref'] is not None else None
            if judgment is not None and (judgment.get('response_ref') != outcome['response_ref'] or
                    judgment.get('response_fingerprint') != outcome['response_fingerprint'] or
                    judgment.get('correct') != axis['verdict'] or judgment.get('review') != review):
                codes.append('JUDGMENT_EVIDENCE_MISMATCH')
            if axis['status'] == 'REVIEWED' and judgment is None:
                codes.append('MISSING_REQUIRED_EVIDENCE')
            result = {'kind': kind, 'legacy': legacy, 'label': label, 'judgment': judgment,
                      'submission': submission, 'envelope': envelope, 'auth': auth, 'reference': ref, 'scope': scope}
        if result is not None: result['scope'] = scope
    except (KeyError, TypeError, ValueError, AttributeError, StopIteration) as exc:
        # Never echo submitted identifiers or evidence text.
        safe = str(exc) if type(exc) is ValueError and str(exc) in (
            'MISSING_REQUIRED_EVIDENCE', 'IMMUTABLE_EVIDENCE_CHANGED', 'IMMUTABLE_CONTEXT_CHANGED',
            'PINNED_SOURCE_CHANGED', 'TYPED_ENVELOPE_FIELDS_INVALID', 'AXIS_SEMANTIC_MISMATCH',
            'EVIDENCE_PURPOSE_MISMATCH', 'SOURCE_CONTEXT_MISMATCH', 'EVIDENCE_LINEAGE_MISMATCH') else 'MISSING_REQUIRED_EVIDENCE'
        codes.append(safe)
    return result, sorted(set(codes))


def admit_reviews(resolver, store, envelope_refs, *, provider=None, policy_ref=None, production=True):
    """No production forwarding of unreviewed inputs. Returns only safe counts.

    Private conversion is kept internally and passed to unchanged validators;
    caller never receives fake training rows or clinical readiness.
    """
    report = {'schema_version': 'locdi.review_admission.v1', 'status': 'BLOCKED', 'blocker_codes': [],
        'production_authentication_ready': False, 'clinical_reference_ready': False,
        'routing_artifact_ready': False, 'official_b0_b4_ready': False, 'legacy_gate_called': False,
        'legacy_validator_called': False, 'test_only_conversion_ready': False, 'reference_envelopes': 0,
        'response_envelopes': 0, 'complete_train_groups': 0, 'validation_group_count': 0,
        'fit_rows_eligibility': 'NOT_CHECKED', 'research_artifact_gate': 'NOT_CHECKED',
        'actual_generation_calls': 0, 'utility_fitting': 0, 'clinical_verdicts_generated': 0}
    codes = report['blocker_codes']
    try:
        resolver.verify()
        if type(production) is not bool:
            raise ValueError('MODE_INVALID')
        if production:
            receipt = verify_authority(provider, '', '', {}, production=True)
            report['authentication'] = receipt['status']; codes.extend(receipt['blocker_codes'])
        elif not resolver.test_only:
            codes.append('REAL_CONTEXT_CANNOT_USE_TEST_ONLY_AUTHORITY')
        if policy_ref is None:
            codes.append('REVIEW_POLICY_NOT_APPROVED')
        else:
            policy = store.get(policy_ref)
            if (policy.get('schema_version') != 'locdi.test_policy_approval.v1' or policy.get('purpose') != 'TEST_ONLY' or
                    policy.get('policy_fingerprint') != digest(POLICY) or policy.get('approved') is not True):
                codes.append('REVIEW_POLICY_NOT_APPROVED')
            auth = verify_authority(provider, policy_ref, digest(policy), {'reviewer_ref': policy.get('authority_ref'),
                'axis': 'POLICY', 'policy_fingerprint': digest(POLICY), 'at': policy.get('approved_at')}, production=production)
            codes.extend(auth['blocker_codes'])
        if not envelope_refs:
            codes.extend(['MISSING_REQUIRED_EVIDENCE', 'REFERENCE_NOT_APPROVED'])
        if codes:
            report['blocker_codes'] = sorted(set(codes)); return report
        if len(set(envelope_refs)) != len(envelope_refs):
            codes.append('DUPLICATE_ENVELOPE')
        accepted = []
        for ref in envelope_refs:
            result, blockers = _check_envelope(ref, store.get(ref), store, resolver, provider, policy_ref, production=production)
            codes.extend(blockers)
            if result is not None:
                accepted.append(result)
        refs = {r['reference']: r for r in accepted if r['kind'] == 'REFERENCE_REVIEW'}
        report['reference_envelopes'] = len(refs)
        responses = [r for r in accepted if r['kind'] == 'RESPONSE_JUDGMENT']
        report['response_envelopes'] = len(responses)
        for row in responses:
            linked = refs.get(row['envelope']['reference_envelope_ref'])
            if not linked:
                codes.append('REFERENCE_NOT_APPROVED'); continue
            if (row['legacy']['reference_fingerprint'] != linked['legacy']['reference_fingerprint'] or
                    row['legacy']['decision_id'] != linked['legacy']['decision_id'] or
                    row['legacy']['reference_ref'] != linked['legacy']['reference_ref'] or row['label'] != linked['label']):
                codes.append('REFERENCE_FINGERPRINT_OR_IDENTITY_MISMATCH')
            # Compare actual submitted ordering plus reference authority receipt,
            # not just wall-clock strings; real blinding is an authority concern.
            if (envelope_refs.index(linked['reference']) >= envelope_refs.index(row['reference']) or
                timestamp(linked['envelope']['review_timestamp']) > timestamp(row['envelope']['review_timestamp']) or
                linked['scope']['submitted_sequence'] >= row['scope']['submitted_sequence'] or
                timestamp(linked['scope']['submitted_at']) > timestamp(row['scope']['submitted_at']) or
                timestamp(linked['scope']['submitted_at']) > timestamp(row['envelope']['review_timestamp']) or
                not linked['auth']['verified']):
                codes.append('INVALID_REVIEW_SEQUENCE')
        if codes:
            report['blocker_codes'] = sorted(set(codes)); return report
        data, options = deepcopy(resolver._data), deepcopy(resolver._options)
        data['judgments'] = [r['judgment'] for r in responses if r['judgment'] is not None]
        options['review_results'] = empty_review_results(data['purpose'])
        options['review_results']['reference_reviews'] = [r['legacy'] for r in refs.values()]
        options['review_results']['response_reviews'] = [r['legacy'] for r in responses]
        options['label_catalog'] = [r['label'] for r in refs.values()]
        # No implied source review: only the explicit authenticated source rows.
        source_map = {r['source']['source_ref']: r['source'] for r in refs.values()}
        options['source_catalog'] = [source_map.get(s['source_ref'], s) for s in options['source_catalog']]
        if not Draft202012Validator(REVIEW_SCHEMA).is_valid(options['review_results']):
            codes.append('LEGACY_SCHEMA_INVALID_LOSSLESS_CONVERSION_NOT_PROVEN')
            report['blocker_codes'] = sorted(set(codes)); return report
        gate = validate_evaluation_gate(data, **options)
        report['legacy_gate_called'] = True
        report['legacy_validator_called'] = isinstance(gate['legacy_validator'], dict)
        report['legacy_gate'] = {k:gate[k] for k in ('blocker_codes', 'development_contract_ready',
            'utility_fit_ready', 'complete_train_groups', 'validation_group_count', 'positive_similarity_support')}
        report['complete_train_groups'], report['validation_group_count'] = gate['complete_train_groups'], gate['validation_group_count']
        codes.extend(gate['blocker_codes'])
        if not codes:
            # Read-only legacy eligibility check; NEVER fit_utility. These rows
            # exist only transiently for explicit TEST_ONLY contract validation.
            legacy_options = {k: options[k] for k in ('registry_manifest', 'source_catalog', 'label_catalog',
                'holdout_patient_ids', 'dataset_fingerprint', 'allow_test_only')}
            eligibility = to_fit_rows(data, **legacy_options)
            report['fit_rows_eligibility'] = 'TEST_ONLY_PRIVATE_ROWS_NOT_EXPORTED' if eligibility.get('rows') is not None else 'BLOCKED'
            artifacts = assess_artifacts(options['registry_manifest'], correctness_inputs=(data, options),
                evaluation_patient_ids=options['holdout_patient_ids'])
            report['research_artifact_gate'] = {k: artifacts[k] for k in ('research_evaluation_ready', 'clinical_metrics')}
            report['test_only_conversion_ready'] = resolver.test_only and gate['development_contract_ready']
            report['status'] = 'PASS_TEST_ONLY_CONTRACT' if report['test_only_conversion_ready'] else 'BLOCKED'
    except (ValueError, KeyError, TypeError, AttributeError):
        codes.append('MISSING_REQUIRED_EVIDENCE_OR_CONTEXT_INVALID')
    report['blocker_codes'] = sorted(set(codes))
    if codes: report['status'] = 'BLOCKED'
    return report


def bridge_contract():
    return {'schema_version': 'locdi.c3_04.bridge_contract.v1', 'envelope_fields': sorted(ENVELOPE_FIELDS),
        'axes': list(AXES), 'reference_validity_types': list(VALIDITY),
        'legacy_timing_types': sorted(REVIEW_SCHEMA['$defs']['reference']['properties']['timing_status']['enum']),
        'conversion': 'EXPLICIT_AUTHENTICATED_SIDECAR_JOIN_NOT_RENAME_OR_DEFAULT_GENERATION',
        'production_authentication': 'AUTHENTICATION_NOT_CONFIGURED',
        'production_conversion': 'BLOCKED_NO_TRUSTED_AUTHORITY_OR_APPROVED_POLICY',
        'auxiliary_axes': 'Preserved in envelope/evidence; no lossless automatic conversion into legacy diagnosis bool',
        'legacy_schema_unchanged': True, 'clinical_reference_ready': False,
        'policy_approved': False, 'test_only_provider_promotable': False}
