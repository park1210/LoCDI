"""C4-33 frozen input preparation and offline replay, no model/network calls."""
from copy import deepcopy
import random

from app.locdi.llm_setup import digest
from app.locdi.voi_information_improvement import select_actions, estimate
from app.locdi.dynamic_horizon import policy_spec
from app.locdi.dynamic_horizon_source_evaluation import run_episode
from app.locdi.voi_sequential_policy_evaluation import episode, canonical_order, score_episode

POLICIES = ['prior', 'random', 'fixed', 'global_eig', 'information_gain', 'voi_h1', 'voi_h2', 'voi_h3', 'dynamic']


def training_protocol(source_sha, keys_sha, template_sha):
    return {'version': 'locdi.final_m5.development.c4_33.v1',
        'training': 'C4-30 base_visits only: 173 patients / 229 visits; no augmentation, no fresh holdout',
        'source_sha256': source_sha, 'action_keys_sha256': keys_sha, 'template_sha256': template_sha,
        'selection': 'reuse select_actions: base5 + <=3 extras; support>=20p, coverage>=10 diagnoses, missing<=.5, distinct>=3; binary EIG*observed fraction; Pearson redundancy .95/20 pairs',
        'encoding': 'reuse estimate/thresholds ternary train quantiles 1/3,2/3; exact stored float boundaries; alpha=1',
        'prior': 'one count per visit with >=1 selected action, Dirichlet alpha1',
        'maximum_estimations': 1, 'diagnoses': 15, 'maximum_actions': 8,
        'holdout_access_before_freeze': False, 'clinical_ready': False}


def fit_final(template, train, keys, excluded_patients):
    if len(template['diagnoses']) != 15:
        raise ValueError('DIAGNOSIS15_REQUIRED')
    if set(excluded_patients) & {v['patient_id'] for v in train}:
        raise ValueError('TRAIN_EVALUATION_OVERLAP')
    prior = dev_selection_prior(template, train, keys['base'])
    selected, stats, ledger = select_actions(train, keys['base'], keys['candidates'], prior)
    # select_actions EIG prior must be computed from development rows, never a fold
    # or test reference. Caller supplies the existing alpha1 prior on base5 rows.
    model, rows = estimate(template, train, selected, 'ternary', excluded_patients)
    model['model_id'] = 'C4_33_FINAL_M5_DEVELOPMENT_ONLY'
    model['provenance'].update(training_scope='ALL_EXISTING_C4_30_BASE229_ONLY',
        clinical_support='NOT_APPROVED', actual_value_source='EXISTING_RECORDED_SOURCE')
    return model, rows, {'selected_keys': selected, 'statistics': stats, 'ledger': ledger}


def dev_selection_prior(template, train, base_keys):
    seen = {}
    for v in train:
        if set(base_keys) & set(v['values']):
            key = (v['patient_id'], v['visit_id'])
            if key in seen:
                raise ValueError('DUPLICATE_TRAIN_VISIT')
            seen[key] = v['diagnosis']
    if not seen or any(d not in template['diagnoses'] for d in seen.values()):
        raise ValueError('INVALID_DEVELOPMENT_REFERENCE')
    return {d: (1 + sum(y == d for y in seen.values())) / (len(template['diagnoses']) + len(seen)) for d in template['diagnoses']}


def evaluation_protocol(model_sha, selection_sha):
    return {'protocol': 'locdi.voi_new_synthetic_holdout.c4_33.v1',
        'artifact_sha256': model_sha, 'action_selection_sha256': selection_sha,
        'policies': POLICIES, 'dynamic_spec': policy_spec(), 'budgets': list(range(6)),
        'primary': {'metric': 'patient-macro Source Diagnosis Top1 AUBC budgets1..5', 'comparison': ['voi_h3', 'global_eig']},
        'secondary': [['voi_h1', 'global_eig']],
        'report': ['visit_and_patient_macro', 'budget5', 'top1', 'top3', 'nll', 'brier', 'actions', 'planner_states', 'failures'],
        'bootstrap': {'seed': 42, 'resamples': 2000, 'cluster': 'patient, preserve visits and budgets', 'CI': 'percentile95'},
        'random': {'repeats': 32, 'seed_rule': 'C4-31 digest(seed42,scenario_ordinal,repeat)[:16]', 'MC_error': 'separate'},
        'reference': 'UNREVIEWED_SOURCE_DIAGNOSIS, never clinical Gold',
        'eligibility': 'all complete exact final menu visits, no sampling by scores; >=2 patients for interval feasibility only, not power',
        'planner': {'horizon_max': 3, 'max_states': 20000, 'epsilon': 1e-12, 'utility': 'unchanged indicator 1/0'},
        'evaluation_approval': 'SEPARATE_REQUIRED', 'generation_approval': 'SEPARATE_REQUIRED',
        'no_seed_search': True, 'no_holdout_fitting': True}


def policy_episode(model, public_values, policy, budget, planner, eig_order, *, ordinal, repeat=0):
    """Scorer/reference never enters this interface; real C4-31/Core paths only."""
    if policy not in POLICIES:
        raise ValueError('POLICY_NOT_FROZEN')
    if policy in ('dynamic', 'voi_h1', 'voi_h2', 'voi_h3'):
        return run_episode(model, public_values, policy, budget, planner)
    order = None
    if policy == 'random':
        if not 0 <= repeat < 32:
            raise ValueError('RANDOM_REPEAT_RANGE')
        order = sorted(a['id'] for a in model['actions'])
        random.Random(int(digest({'seed': 42, 'scenario_ordinal': ordinal, 'repeat': repeat})[:16], 16)).shuffle(order)
    return episode(model, public_values, policy, budget, planner, eig_order, forced_order=order)


def score_after_execution(result, private_reference):
    return score_episode(deepcopy(result), private_reference)
