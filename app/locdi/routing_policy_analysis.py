"""C3-03 read-only policy analysis. No fitting, labels, split assignment or client.

TEST_ONLY probes require an explicitly supplied existing symbolic fixture.
They return contract states, never clinical training rows.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_freeze import load, sha, PRIVATE
from app.locdi.clinical_gap_analysis import routing_groups, split_matrix, review_comparison
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility, observed_tokens, _validate_samples
from validation.routing_correctness_gate import validate_evaluation_gate


def verify_manifest(path):
    """Local file integrity, never approval/authentication."""
    manifest = load(path)
    unsigned = dict(manifest)
    fingerprint = unsigned.pop('fingerprint')
    if digest(unsigned) != fingerprint or digest(manifest['files']) != manifest['files_fingerprint']:
        raise ValueError('PRIOR_MANIFEST_FINGERPRINT_MISMATCH')
    for name, expected in manifest['files'].items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not target.is_file() or sha(target) != expected:
            raise ValueError('PRIOR_MANIFEST_FILE_CHANGED')
    return {'status': 'PASS', 'files_checked': len(manifest['files']), 'fingerprint': fingerprint}


def independent_split_check(members):
    """Patient sets, so another encounter cannot escape the exclusion."""
    rt, rv, vt = (set(members[k]) for k in ('routing_train', 'routing_validation', 'voi_train'))
    direct, combined = len(rt & rv), len(rv & vt)
    return {'routing_train_validation_overlap': direct,
            'routing_validation_voi_train_overlap': combined,
            'routing_patient_split': 'BLOCKED' if direct else 'STRUCTURALLY_DISJOINT',
            'combined_validation': 'BLOCKED' if direct or combined else 'PATIENT_DISJOINT_ONLY',
            'clinical_approval': False, 'split_reassigned': False}


def request_budget(cases, agents=5, repeats=1, retry_limit=0, repair_limit=0):
    """Limits are total additional requests per initial task, not nested retries."""
    values = (cases, agents, repeats, retry_limit, repair_limit)
    if any(type(n) is not int or n < 0 for n in values) or not agents or not repeats:
        raise ValueError('INVALID_PLANNING_BUDGET')
    base = cases * agents * repeats
    return {'status': 'PROPOSED_NOT_AUTHORIZATION', 'cases': cases, 'agents': agents,
            'repeats': repeats, 'retry_limit_per_task': retry_limit,
            'repair_limit_per_task': repair_limit, 'initial_generation_requests': base,
            'maximum_generation_requests': base * (1 + retry_limit + repair_limit),
            'identity_checks_upper_bound': base * (1 + retry_limit + repair_limit),
            'grading_requests_authorized': 0, 'execution_approved': False}


def policy_comparison():
    # Each row describes a proposal, not a selection or a relaxed admission rule.
    rows = [
        dict(policy='A', goal='Independent diagnosis correctness; all five at identical information',
             math_change=False, core_change=False, gate_change=False,
             current_reuse='75 baseline analyses; 22 supported review candidates; 0 complete fit groups',
             reference='15 decision references, snapshot/timing review; supported response judgments; abstention separate',
             generation='New versioned data/response collection needed to obtain any eligible group under fixed current answers',
             selection_bias='All-five-supported selection may enrich easy decisions; report all exclusions',
             support_bias='Complete groups equalize per-agent token support, not population coverage',
             fairness='Common patients, snapshots, candidates and bindings; independent held-out queries',
             implementation='Low algorithm change; high data/review dependency', review_cost='15 decision scopes + up to 75 axis reviews; staffing UNKNOWN',
             time_risk='Reference access and eligible group yield UNKNOWN', b3_b4='Preserves accuracy-only objective; official execution still blocked',
             states=['CURRENT_CONTRACT_COMPATIBLE', 'REQUIRES_NEW_DATA', 'BLOCKED_BY_REFERENCE']),
        dict(policy='B1', goal='A plus descriptive uncertainty/abstention sidecar', math_change=False, core_change=False, gate_change=False,
             current_reuse='All 75 certainty states; no clinical appropriateness labels; 0 fit groups',
             reference='Not needed for certainty counts; independent reference needed for appropriateness claims',
             generation='None for current descriptive report; does not solve A data shortage',
             selection_bias='Report all failures and denominators, including unresolved',
             support_bias='Utility retains A limitations; sidecar cannot supply missing Boolean labels',
             fairness='Keep diagnosis/abstention/routing-success axes separate', implementation='Low sidecar complexity',
             review_cost='0 for counts; up to 53 separate abstention reviews plus A for clinical claims',
             time_risk='Descriptive output available; clinical review throughput UNKNOWN',
             b3_b4='Compatible only as sidecar, no action/utility influence',
             states=['CURRENT_CONTRACT_COMPATIBLE', 'BLOCKED_BY_REFERENCE']),
        dict(policy='B2', goal='New utility incorporating explicit abstention outcome', math_change=True, core_change=True, gate_change=True,
             current_reuse='75 response states structurally inspectable; 0 independently labeled utility outcomes',
             reference='Separate appropriateness/contribution reference, explicit outcome utility and weighting approval',
             generation='Need NOT_CHECKED until new task/reference defined',
             selection_bias='Rewarding abstention without coverage/safety objective may favor always abstaining',
             support_bias='Each outcome/category needs independent support and calibration study',
             fairness='Same labels and costs across agents; accuracy score no longer same estimand',
             implementation='High: objective/schema/gate/runtime/artifact version', review_cost='Distinct multi-axis judgments; capacity UNKNOWN',
             time_risk='Policy and reference development unresolved', b3_b4='Separate approved experiment; not existing accuracy-only B3/B4',
             states=['REQUIRES_POLICY_REVISION', 'REQUIRES_CODE_REVISION', 'BLOCKED_BY_REFERENCE']),
        dict(policy='C', goal='Agent-conditional correctness on partially observed groups', math_change='Same algebra possible, different conditioning/estimand',
             core_change=True, gate_change=True,
             current_reuse='At most 22 supported rows before review; train/validation per-agent counts in group audit; accepted now 0',
             reference='Independent supported judgments, missingness reasons and selection protocol',
             generation='NOT_CHECKED: per-query all-candidate positive support may remain insufficient',
             selection_bias='Missing not at random: agent confidence/decision difficulty changes target population',
             support_bias='Different agents evaluated on different patients and positive-neighbor support',
             fairness='Shared validation queries; missing responses retained; do not remove hard agents from candidate menu',
             implementation='Core _validate_samples AND validators/gate; all-candidate abstention policy review',
             review_cost='22 supported judgments + 15 references/snapshots/timing; missingness review; capacity UNKNOWN',
             time_risk='Unknown yield after review; pulmonary supported rows 0 in current collection',
             b3_b4='Revised policy and artifact version; cannot pool with original complete-group results',
             states=['REQUIRES_POLICY_REVISION', 'REQUIRES_CODE_REVISION', 'BLOCKED_BY_REFERENCE']),
        dict(policy='D', goal='Resolve source/reference sufficiency before new all-agent collection under A',
             math_change=False, core_change=False, gate_change=False,
             current_reuse='Keep 75 baseline; no selective replacement; existing bank source review candidates only',
             reference='Task, sufficient current snapshot, contemporaneous reference and blinded response judgments',
             generation='Separate authorization after source/split/binding approval; yield NOT_CHECKED',
             selection_bias='Prespecify inclusion before responses; preserve failures, unresolved and excluded cases',
             support_bias='Stratify source coverage before collection; min_samples is query-specific, not total N',
             fairness='Identical observation/memory/catalog, locked prompts/models/options; patient-separated validation',
             implementation='New data version, not algorithm change; no automatic resume/replacement',
             review_cost='N decision/snapshot/timing units + 5*N*R response units; capacity UNKNOWN',
             time_risk='Eligible-group yield and review time UNKNOWN; no deadline guarantee',
             b3_b4='Compatible target only after approvals; new task/input/prompt changes must be versioned',
             states=['CURRENT_CONTRACT_COMPATIBLE', 'REQUIRES_NEW_DATA', 'BLOCKED_BY_REFERENCE'])]
    return {'schema_version': 'locdi.c3_03.policy_comparison.v1', 'status': 'PROPOSED', 'policies': rows,
            'clinical_metrics': 'NOT_EVALUATED', 'outcome_imputation_allowed': False}


def approval_register():
    return {'schema_version': 'locdi.c3_03.approval.v1',
            'RECOMMENDED_POLICY': {'default_contract': 'A', 'data_resolution_candidate': 'D', 'descriptive_sidecar': 'B1',
                                   'separate_policy_approval': ['B2', 'C']},
            'APPROVED_POLICY': None, 'POLICY_APPROVAL_REQUIRED': True,
            'UNRESOLVED_CONFLICTS': ['CURRENT75_HAVE_NO_ALL_SUPPORTED_GROUP', 'CURRENT_DECISION_REFERENCE_NOT_REVIEWED',
                'REVIEWER_AUTHORITY_NOT_CONFIGURED', 'ROUTING_VALIDATION_IN_VOI_TRAIN',
                'DEVELOPMENT_EXPOSURE_NOT_BLIND', 'TASK_INFORMATION_SUFFICIENCY_NOT_APPROVED',
                'OFFICIAL_RUNNER_METRICS_AND_LIVE_INTEGRATED_EXECUTOR_NOT_IMPLEMENTED'],
            'approval_evidence': None, 'clinical_approval': False}


def test_only_cases(reviewed_fixture):
    """Actual Gate/Router probes using an existing explicitly TEST_ONLY fixture.

    No fit_utility invocation; manually declared symbolic samples only in memory.
    The asymmetric-score probe is explicitly outside valid artifact admission.
    """
    data, opts = reviewed_fixture()
    if data.get('purpose') != 'TEST_ONLY' or opts.get('allow_test_only') is not True:
        raise ValueError('EXPLICIT_TEST_ONLY_FIXTURE_REQUIRED')
    before = digest([data, opts])
    result = []
    def gate_case(name, d, o):
        gate = validate_evaluation_gate(d, **o)
        result.append({'case': name, 'function': 'validate_evaluation_gate',
                       'blocker_codes': gate['blocker_codes'], 'development_contract_ready': gate['development_contract_ready'],
                       'complete_train_groups': gate['complete_train_groups'],
                       'utility_fit_ready': gate['utility_fit_ready'], 'clinical_approval': False})
    gate_case('all_five_supported_independent_test_labels', data, opts)
    for count, name in ((1, 'one_unresolved'), (3, 'multiple_unresolved'), (5, 'all_unresolved')):
        d, o = deepcopy(data), deepcopy(opts)
        for row in d['outcomes'][:count]:
            row['response']['certainty'] = 'unresolved'; row['response_fingerprint'] = digest(row['response'])
            ctx = next(c for c in o['decision_contexts'] if c['decision_id'] == row['decision_id'])
            ctx['original_response_fingerprints'][row['response_ref']] = row['response_fingerprint']
            rr = next(r for r in o['review_results']['response_reviews'] if r['response_ref'] == row['response_ref'])
            rr['response_fingerprint'] = row['response_fingerprint']
            rr['axes']['diagnosis_correctness'].update(status='NOT_EVALUABLE', verdict=None,
                clinical_evidence_rationale=None, not_evaluable_reason='TEST_ONLY_UNRESOLVED')
            rr['axes']['abstention_appropriateness'].update(status='REVIEWED', verdict=True,
                clinical_evidence_rationale='TEST_ONLY_SEPARATE_AXIS')
            d['judgments'] = [j for j in d['judgments'] if j['response_ref'] != row['response_ref']]
        gate_case(name, d, o)
    d, o = deepcopy(data), deepcopy(opts)
    o['review_results']['reference_reviews'][0]['timing_status'] = 'REFERENCE_TIMING_NOT_REVIEWED'
    gate_case('reference_timing_not_reviewed', d, o)
    d, o = deepcopy(data), deepcopy(opts)
    # Same clinical fields, different actual memory, while original snapshot hashes remain pinned.
    d['decisions'][0]['observation']['longitudinal_memory']['records'].append({'text': 'TEST_ONLY_DIFFERENT_MEMORY'})
    gate_case('same_current_observation_different_memory', d, o)
    d, o = deepcopy(data), deepcopy(opts)
    d['split_manifest']['patients']['validation'].append(d['split_manifest']['patients']['train'][0])
    gate_case('same_patient_train_validation', d, o)
    registry = AgentRegistry(data['registry'])
    agents = [a for a in registry.snapshot()['agents'] if a['role'] == 'specialty']
    departments = [a['department'] for a in agents]
    samples = [{'patient_id': 'TEST_ONLY_P%d' % i, 'decision_id': 'TEST_ONLY_D%d' % i,
                'agent_id': a['agent_id'], 'tokens': ['symbol'], 'correct': j == 0}
               for i in range(2) for j, a in enumerate(agents)]
    artifact = {'schema_version': 2, 'method': 'jaccard_beta_accuracy_v2',
                'registry': registry.snapshot(), 'min_samples': 2, 'samples': samples}
    active = AgentRegistry(registry.snapshot(), artifact)
    def route_case(name, router, query, **extra):
        routed = router.route(query, departments, mode='adaptive')
        result.append({'case': name, 'function': 'AgentRegistry.route', 'decision_status': routed['decision_status'],
                       'selected_count': int(routed['selected_specialty'] is not None),
                       'allowed_count': len(routed['allowed_departments']), 'tied_count': len(routed['tied_departments']),
                       'support_per_candidate': {a['department']: (routed['utility_estimates'][a['department']] or {}).get('matched_samples') for a in agents},
                       'clinical_approval': False, **extra})
    route_case('unique_best_symbolic', active, {'chief_complaint': 'symbol'})
    route_case('all_similarity_zero', active, {'chief_complaint': 'unseen'})
    tied = deepcopy(artifact)
    for row in tied['samples']: row['correct'] = True
    route_case('tied_best', AgentRegistry(registry.snapshot(), tied), {'chief_complaint': 'symbol'})
    partial = deepcopy(samples[:-1])
    try:
        _validate_samples(partial, registry)
        rejected = False
    except ValueError:
        rejected = True
    result.append({'case': 'candidate_support_imbalance', 'function': '_validate_samples',
                   'partial_artifact_rejected': rejected, 'clinical_approval': False})
    # Unreachable through a valid complete-group artifact; interface stress only.
    class TestOnlyAsymmetricUtility:
        fingerprint = 'TEST_ONLY_INTERFACE_NOT_ARTIFACT'
        def score(self, observation, agent):
            return None if agent['agent_id'] == agents[-1]['agent_id'] else active.utility.score(observation, agent)
    stress = AgentRegistry(registry.snapshot()); stress.utility = TestOnlyAsymmetricUtility()
    route_case('only_some_candidates_meet_min_samples', stress, {'chief_complaint': 'symbol'},
               scope='TEST_ONLY_INTERFACE_STRESS_INVALID_AS_PRODUCTION_ARTIFACT')
    split = independent_split_check({'routing_train': ['TEST_ONLY_P'], 'routing_validation': ['TEST_ONLY_V'],
                                     'voi_train': ['TEST_ONLY_V']})
    result.append({'case': 'routing_validation_in_voi_train', 'function': 'independent_split_check', **split})
    if digest([data, opts]) != before:
        raise ValueError('TEST_ONLY_FIXTURE_MUTATED')
    return {'schema_version': 'locdi.c3_03.test_only.v1', 'purpose': 'TEST_ONLY',
            'cases': result, 'case_count': len(result), 'fitting_calls': 0, 'clinical_verdicts_generated': 0,
            'scope': 'Symbolic declarations, not medical labels; no training rows exported'}


def build():
    """Re-read real data and existing immutable manifests; no new verdicts."""
    from app.locdi.evaluation_gate_audit import actual_gate_inputs
    from app.locdi.specialty_collection_analysis import analyze_run
    from app.locdi.routing_collector import load_preparation
    from app.locdi.routing_data_reuse import BANK, read
    from app.locdi.source_replay import PUBLIC
    prior = {}
    for stage, namespace, marker in (('C3_01', 'c3_01_v3', 'analysis_manifest.json'),
                                      ('C3_02', 'c3_02_v1', 'completion_manifest.json')):
        prior[stage] = verify_manifest(ROOT / 'outputs/research_evaluation' / namespace / marker)
    data, opts = actual_gate_inputs()
    gate = validate_evaluation_gate(data, **opts)
    bundle = load_preparation('phase24b_candidates_v2')
    collection, _ = analyze_run(run_id='phase24b_collect_001', preparation='phase24b_candidates_v2')
    if collection['run_id'] != 'phase24b_collect_001' or collection['integrity_issue_codes']:
        raise ValueError('COLLECTION_INTEGRITY_NOT_VERIFIED')
    template = load(ROOT/'outputs/private/unresolved_analysis/phase24d_analysis_v1/independent_review_template.json')
    if (len(template['items']) != 75 or any(r.get('correct') is not None or r.get('status') != 'NOT_REVIEWED'
            for r in template['items']) or opts['review_results']['reference_reviews'] or opts['review_results']['response_reviews']):
        raise ValueError('EXPECTED_UNREVIEWED_BASELINE_CHANGED')
    groups = routing_groups(data)
    agents = [a for a in data['registry']['agents'] if a['role'] == 'specialty']
    groups.update(collected_complete_groups=groups['collection_complete_groups'],
                  independently_reviewed_complete_groups=0, gate_accepted_fit_groups=gate['complete_train_groups'],
                  gate={k: gate[k] for k in ('source_reference_ready', 'clinical_review_ready', 'response_policy_ready',
                      'complete_train_groups', 'validation_group_count', 'positive_similarity_support', 'utility_fit_ready', 'blocker_codes')},
                  collection_integrity={k: collection[k] for k in ('state', 'checks', 'integrity_issue_codes', 'missing_tasks',
                      'orphan_record_count', 'splits', 'raw_wire_payload', 'historical_overwrite')},
                  per_specialty=collection['specialties'],
                  review_template_status={'items': len(template['items']), 'correctness_nonnull': 0,
                      'reviewed_items': 0, 'authority_authenticated_items': 0},
                  supported_review_candidates_per_agent_split={a['department']: {s: sum(r['agent_id'] == a['agent_id'] and
                      r['split'] == s and r['response']['certainty'] == 'supported' for r in data['outcomes'])
                      for s in ('train', 'validation')} for a in agents},
                  per_policy_structural_reuse={'A': {'fit_groups': 0}, 'B1': {'descriptive_responses': len(data['outcomes']), 'fit_groups': 0},
                      'B2': {'outcome_state_candidates': len(data['outcomes']), 'independent_utility_labels': 0},
                      'C': {'supported_review_candidates': groups['supported'], 'accepted_fit_rows': 0},
                      'D': {'baseline_preserved': len(data['outcomes']), 'new_collection_executed': 0}})
    groups['split_counts'] = {s: {'decisions': sum(d['split'] == s for d in data['decisions']),
        'supported_responses': sum(r['split'] == s and r['response']['certainty'] == 'supported' for r in data['outcomes']),
        'collected_complete': sum(g['split'] == s and g['collection_complete'] for g in groups['groups']),
        'supported_only_complete': sum(g['split'] == s and g['supported_only_complete'] for g in groups['groups']),
        'independently_reviewed_complete': 0, 'gate_accepted_fit': 0} for s in ('train', 'validation')}
    for g, reference in zip(groups['groups'], gate['groups']):
        g['reference_timing_status'] = reference['reference_timing_status']
        g['condition_start_after_decision_days'] = reference['source_condition_start_after_decision_days']
    original = [json.loads(s) for s in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    dm = load(PUBLIC/'dataset_manifest.json'); entry = dm['files']['cases.jsonl']; path = (ROOT/entry['path']).resolve()
    if not path.is_relative_to((ROOT/'outputs/private/simulation_datasets').resolve()) or sha(path) != entry['sha256']:
        raise ValueError('D3_CASES_CHANGED')
    dcases = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
    frozen = load(ROOT/PRIVATE)['private_memberships']; members = frozen['patient_memberships']
    bank, voi = read(BANK), read(BANK.parent/'voi_model.json')
    selection = load(ROOT/'outputs/private/external_evaluation/healthbench/D5_pilot_v1/selection_manifest.json')
    splits = split_matrix(members, observed_original=[r['patient_id'] for r in original],
        observed_d3=[r['patient_id'] for r in dcases], observed_bank=[r['training_patient_id'] for r in bank],
        observed_voi=voi['provenance']['training_patients'], observed_routing=data['split_manifest']['patients'],
        pilot_ids=[r['sample_id'] for r in selection['samples']], frozen_pilot_ids=frozen['healthbench_excluded_pilot_ids'])
    splits.update(independent_split_check(members), bank_visits=len(bank),
        raw_sqlite_scope='D1/D2/D7 anchored inventories reused; no new full raw/SQLite scan',
        raw_sqlite_holdout_overlap='KNOWN_INCLUSION_OF_HOLDOUT50; NOT_AN_APPROVED_TRAINING_POOL',
        proposed_design={'status': 'PROPOSED', 'assignments_made': 0,
            'routing_train': 'Patient-disjoint from routing validation and final test; complete reviewed groups only',
            'routing_validation': 'No overlap with routing train; no fitting or test-based tuning',
            'voi_fitting': 'Independent of VoI validation/final test; clinically reviewed outcome data',
            'voi_validation': 'Not currently established; separate patient set from VoI fitting/tuning',
            'combined_validation': 'Outside union of BOTH routing and VoI fitting/tuning; current routing validation ineligible',
            'final_paired': 'Approved exposure-aware test; identical ordered patient/encounter/source/cutoff/repeat manifests B0-B4'},
        paired_linkage={'required': ['patient_split_fingerprint', 'ordered_case_fingerprint', 'source_snapshot_fingerprint',
            'memory_policy_and_cutoff_fingerprint', 'registry_model_prompt_decoding_fingerprints',
            'action_diagnosis_catalog_fingerprints', 'seed_time_budget_repeat_fingerprints', 'reference_policy_version'],
            'status': 'PROPOSED_NOT_IMPLEMENTED_FOR_OFFICIAL_EVALUATION',
            'memory_values': 'Expected differences B0/B1/B2; same underlying source/cutoff, identical B2/B3/B4 memory policy',
            'development_exposure': 'Original50/D329 and HealthBench3 not silently blind; pretraining contamination UNKNOWN'})
    contracts = {'schema_version': 'locdi.c3_03.routing_contract.v1', 'prior_manifests': prior,
        'token_fields': ['chief_complaint', 'current_symptoms', 'objective_findings', 'known history.answer', 'non-superseded memory.text'],
        'token_algorithm': 'Unicode word regex, casefold, unique sorted tokens; excluded metadata is not a model input',
        'similarity': 'J(x,xi)=intersection/union; empty union=0; only J>0 contributes',
        'score': '(1 + sum(J*correct)) / (2 + sum(J))', 'score_semantics': 'UNCALIBRATED_SIMILARITY_WEIGHTED_CORRECTNESS',
        'core_samples': 'Nonempty; unique patient/decision/agent; Boolean correct; tokens list; all registered specialties and identical tokens per decision',
        'gate_extra_contract': 'Reviewed current reference, identical observation AND memory fingerprints, independent response judgments, bindings and patient split; not merely identical token sets',
        'unresolved': 'Not diagnosis Boolean; separate appropriateness review; never true/false imputation',
        'min_samples_default': 2, 'support': 'Positive J count per candidate, for EACH query; all candidates required',
        'tie': {'absolute_tolerance': 1e-12, 'relative_tolerance': 0, 'status': 'abstained', 'allowed_menu_retained': True},
        'missing_support': 'abstained; selected_specialty=null; allowed candidate menu retained',
        'missing_utility': 'abstained', 'empty_candidates': 'unavailable', 'er_only': 'disabled',
        'registry': 'Exactly one ER, unique role/department and IDs, exact snapshot model/version binding',
        'gold_department': 'Not used by observed_tokens or adaptive score; no label-conditioned candidate narrowing',
        'live_integrated': 'BLOCKED: independent artifacts plus executor not implemented',
        'clinical_policy_order': 'Encounter ER proposal -> InformationPolicy -> validated action -> Scheduler; belief-conditioned ER hook OFF',
        'memory_equivalence': 'Equal token sets alone do not prove equal source/memory snapshot; Gate fingerprints remain necessary'}
    approval = approval_register()
    handoff = {'schema_version': 'locdi.c3_03.c3_04_handoff.v1', 'status': 'DESIGN_ONLY_POLICY_NOT_APPROVED',
        'schema_comparison': review_comparison(), 'policy_approval_required': True,
        'required_sequence': ['Approve task/policy and independent splits', 'Authenticate qualified independent reviewers',
            'Blind current-decision Reference/source/snapshot/timing review', 'Separate supported correctness and abstention review',
            'Implement explicit typed schema bridge with loss/unknown handling tests', 'Recheck original hashes and legacy Gate',
            'Independent validation and separate fitting approval'],
        'required_bindings': ['agent', 'decision', 'source', 'reference', 'response', 'observation', 'memory', 'catalog', 'policy_version'],
        'timing': 'Current/retrospective/not-reviewed/rejected remain distinct; Candidate010 +4 days NOT_REVIEWED',
        'authority': 'NOT_IMPLEMENTED; REVIEWED string/hash is not authentication',
        'unresolved': 'Separate status and abstention axis, correctness=null; never fill missing agent labels',
        'adapter_implemented': False, 'clinical_admission_allowed': False}
    budgets = {'schema_version': 'locdi.c3_03.workload.v1', 'status': 'PROPOSED',
        'existing_units': {'bank_patients': len({r['training_patient_id'] for r in bank}), 'bank_visits': len(bank),
            'decision_reference_candidates': 15, 'collected_responses': 75, 'supported_review_candidates': 22,
            'abstention_review_candidates': 53, 'shared_snapshot_review_units': 15, 'selected_memory_references': 75,
            'reviewed_approved_decisions': 0, 'reviewed_approved_responses': 0, 'reviewed_complete_train_groups': 0},
        'human_reviewer_count': 'UNKNOWN', 'human_review_throughput': 'UNKNOWN', 'completion_date_estimate': 'UNKNOWN',
        'per_policy': {p: {'reference_decision_review': 15, 'snapshot_memory_review': 15, 'source_timing_review': 15,
            'supported_response_review_candidates': 22, 'separate_abstention_review_candidates': 53,
            'reviewer_authority_approval': 'REQUIRED_NOT_IMPLEMENTED', 'artifact_fitting_executed': 0,
            'independent_validation': 'Routing 5 structural candidates; combined/VoI independent set not established',
            'new_collection_target': 'N_NEW_PROPOSED' if p == 'D' else 'NOT_SET',
            'clinical_review_claim': 'No one-to-one or guaranteed approval implied; shared units not multiplied by agents'}
            for p in ('A', 'B1', 'B2', 'C', 'D')},
        'new_collection_formula': 'N_new * 5 * repeats * (1 + total_retry_limit + total_repair_limit)',
        'illustrative_plan': request_budget(15), 'new_case_target_approved': None,
        'collection_preconditions': ['Task and input sufficiency', 'Verified source and current reference', 'Patient-level independent splits',
            'Identical snapshot/memory/catalog', 'Locked Registry/model/prompt/decoding', 'Preserve all outcomes and failures',
            'Explicit budget and execution approval', 'Independent response review after collection'],
        'yield': 'UNKNOWN; no predicted success rate; 75 illustrative requests do not guarantee a fit group'}
    docs = [p for p in (ROOT/'docs').rglob('*.md') if p.name.startswith(('C3_01_', 'C3_02_', '022_', '023_', '024_', 'D1_', 'D2_', 'D4_', 'D7_', '035_', '036_'))]
    code = ['app/core/routing_utility.py', 'app/core/agent_registry.py', 'validation/routing_training_data.py',
        'validation/routing_correctness_gate.py', 'app/locdi/routing_collector.py', 'app/locdi/routing_collection_preparation.py',
        'app/locdi/evaluation_gate_audit.py', 'app/locdi/specialty_collection_analysis.py', 'app/locdi/integrated_approval.py',
        'app/core/hospital_llm_policy.py']
    evidence = {'documents': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(docs)},
                'code': {p: sha(ROOT/p) for p in code}, 'voi_source_sha256': sha(BANK.parent/'voi_model.json')}
    links = {'schema_version': 'locdi.private.c3_03_links.v1', 'purpose': 'ANALYSIS_LINKS_NOT_REVIEW_RESULTS',
        'memberships': members, 'decision_links': [{'ordinal': i, **{k: d[k] for k in ('patient_id', 'visit_id', 'decision_id', 'source_ref')},
            'source_fingerprint': c['source_fingerprint'], 'observation_fingerprint': digest(d['observation']),
            'memory_fingerprint': digest(d['observation']['longitudinal_memory']),
            'response_fingerprints': c['original_response_fingerprints']} for i, (d, c) in enumerate(zip(data['decisions'], opts['decision_contexts']), 1)],
        'review_verdicts_created': 0}
    return {'routing_contract_audit': contracts, 'existing_response_groups': groups,
            'policy_comparison': policy_comparison(), 'split_independence_analysis': splits,
            'collection_budget_plan': budgets, 'policy_approval_register': approval, 'c3_04_handoff': handoff}, links, evidence
