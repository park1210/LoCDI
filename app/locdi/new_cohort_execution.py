"""C4-34 execution gates, resource watchdog and frozen replay. No fitting."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import math
import time

from app.locdi.llm_setup import digest
from app.locdi.voi_new_cohort_source import load_csv, overlap, adapt
from app.locdi.voi_new_cohort_evaluation import POLICIES, policy_episode, score_after_execution
from app.locdi.dynamic_horizon_source_evaluation import PlannerMeter, ComputeBudget
from app.locdi.voi_sequential_policy_evaluation import SCORE_KEYS, global_eig_order, mean_metrics
from scripts.analyze_voi_sequential_policies import aubc_rows, paired_comparison

VERSION = 'locdi.new_cohort_execution.c4_34.v1'
GIB = 1024**3


def approval(record, scope, plan_fingerprint):
    if not isinstance(record, dict) or record.get('approved') is not True:
        raise ValueError('EXPLICIT_APPROVAL_REQUIRED')
    if record.get('scope') != scope or record.get('plan_fingerprint') != plan_fingerprint:
        raise ValueError('APPROVAL_SCOPE_OR_PLAN')
    if record.get('evidence_type') != 'USER_EXPLICIT' or not record.get('user_statement'):
        raise ValueError('APPROVAL_EVIDENCE_REQUIRED')
    if not record.get('recorded_at'):
        raise ValueError('APPROVAL_TIME_REQUIRED')


def resource_reason(output_bytes, wall_seconds, *, limit_bytes=10*GIB, limit_seconds=3600):
    if output_bytes >= limit_bytes:
        return 'OUTPUT_LIMIT_REACHED'
    if wall_seconds >= limit_seconds:
        return 'WALL_TIME_LIMIT_REACHED'
    return None


def tree_size(root):
    total = 0
    for p in Path(root).rglob('*'):
        if p.is_symlink():
            raise ValueError('OUTPUT_SYMLINK_NOT_ALLOWED')
        if p.is_file():
            total += p.stat().st_size
    return total


def verify_receipt(receipt, plan, check_hashes):
    if receipt.get('status') != 'GENERATION_COMPLETED' or receipt.get('exit_code') != 0:
        raise ValueError('GENERATION_NOT_COMPLETED')
    if receipt.get('plan_fingerprint') != plan['fingerprint']:
        raise ValueError('GENERATION_PLAN_MISMATCH')
    if receipt.get('image_id') != plan['image_id'] or receipt.get('jar_sha256') != plan['jar_sha256']:
        raise ValueError('GENERATOR_IDENTITY_MISMATCH')
    if receipt.get('seed') != 2026111601 or receipt.get('attempts') != 1:
        raise ValueError('GENERATION_SEED_OR_ATTEMPTS')
    pins = receipt.get('raw_files')
    if not isinstance(pins, dict) or not pins or check_hashes(pins)['changed_or_missing']:
        raise ValueError('RAW_FILE_HASH_MISMATCH')
    if not receipt.get('approval_sha256') or receipt.get('stop_reason') is not None:
        raise ValueError('GENERATION_RECEIPT_INCOMPLETE')
    required = {'csv/patients.csv','csv/encounters.csv','csv/conditions.csv','csv/observations.csv'}
    actual = {Path(p).as_posix() for p in pins}
    if not all(any(p.endswith('/'+name) for p in actual) for name in required):
        raise ValueError('GENERATION_REQUIRED_EXPORTS')


def admit(root, model, exclusions):
    tables = load_csv(root)
    o = overlap(tables, exclusions['patients'], exclusions['encounters'], exclusions['encounter_fingerprints'].values())
    visits, ledger, summary = adapt(tables, root, model)
    complete = {(r['patient_id'],r['visit_id']) for r in ledger if r['complete']}
    selected = [v for v in visits if (v['patient_id'],v['visit_id']) in complete]
    model_overlap = len({v['patient_id'] for v in selected} & set(model['provenance']['training_patients']))
    for v in selected:
        if len(v['values']) != 8 or set(v['values']) != set(model['provenance']['action_key_mapping']):
            raise ValueError('COMPLETE_MENU_REQUIRED')
        for e in v['values'].values():
            if e['availability_basis'] != 'RECORDED_TIME_PROXY' or not 0<=e['available_minute']<=45:
                raise ValueError('SOURCE_VISIBILITY_CONTRACT')
            if digest(e['source_record']) != e['source_record_fingerprint']:
                raise ValueError('SOURCE_RECORD_FINGERPRINT')
    count = len({v['patient_id'] for v in selected})
    blocked = o['status']!='PASS' or model_overlap>0
    report = {'overlap':o,'model_train_patient_overlap':model_overlap,'source':summary,
        'source_complete_patients_before_overlap_gate':count,'source_complete_visits_before_overlap_gate':len(selected),
        'complete_patients':0 if blocked else count,'complete_visits':0 if blocked else len(selected),
        'status':'SOURCE_OVERLAP_BLOCKED' if blocked else 'TECHNICAL_PANEL_AVAILABLE' if count>=2 else 'INSUFFICIENT_PANEL',
        'clinical_reference_approved':False,'power_established':False}
    return ([] if blocked else selected), ledger, report


def execute_panel(model, visits, trace_sink, *, random_repeats=32, test_only=False):
    if random_repeats != 32 and not test_only:
        raise ValueError('FROZEN_RANDOM32_REQUIRED')
    if len({v['patient_id'] for v in visits}) < 2:
        raise ValueError('INSUFFICIENT_PANEL')
    mapping = model['provenance']['action_key_mapping']
    for v in visits:
        if set(v['values']) != set(mapping):
            raise ValueError('MISSING_ACTION_PANEL')
    meter = PlannerMeter(cached=True,budget=ComputeBudget(max_calls=20000,max_states=50000000,max_seconds=3600))
    eig = global_eig_order(model,meter)
    records, membership = [], {}
    # Patient and encounter ordering is fixed before policy scores are available.
    for ordinal, v in enumerate(sorted(visits,key=lambda r:(r['patient_id'],r['visit_id'])),1):
        membership[ordinal] = {'patient_id':v['patient_id'],'visit_id':v['visit_id']}
        for policy in POLICIES:
            for budget in range(6):
                paths = []
                for repeat in range(random_repeats if policy=='random' else 1):
                    before = deepcopy(meter.stats)
                    raw = policy_episode(model,v['values'],policy,budget,meter,eig,ordinal=ordinal,repeat=repeat)
                    if any(e['released_at_minute']<e['available_minute'] for e in raw['releases']):
                        raise ValueError('FUTURE_OBSERVATION_RELEASE')
                    ids = [e['action_id'] for e in raw['releases']]
                    if len(ids)!=len(set(ids)) or len(ids)>budget:
                        raise ValueError('ACTION_REPEATED_OR_BUDGET')
                    # Label is supplied strictly after the real episode terminates.
                    scored = score_after_execution(raw,v['diagnosis'])
                    scored.update(scenario_ordinal=ordinal,random_repeat=repeat if policy=='random' else None,
                                  planner_state_delta=meter.stats['logical_states']-before['logical_states'])
                    trace_sink(scored)
                    paths.append(scored)
                if policy=='random':
                    keys = SCORE_KEYS
                    result = {'status':'completed','policy':policy,'budget':budget,'scenario_ordinal':ordinal,
                        'score':{k:sum(p['score'][k] for p in paths)/len(paths) for k in keys},
                        'actual_actions':sum(p['actual_actions'] for p in paths)/len(paths),
                        'stop':0,'budget_exhausted':1,'random_repeats':len(paths),
                        'planner_state_delta':0,
                        'random_top1_mean_variance':sum((p['score']['top1']-sum(q['score']['top1'] for q in paths)/len(paths))**2 for p in paths)/max(len(paths)-1,1)/len(paths)}
                else:
                    result = paths[0]
                records.append(result)
    return records, membership, deepcopy(meter.stats)


def summarize(records,membership,*,bootstrap_repetitions=2000,test_only=False):
    if bootstrap_repetitions!=2000 and not test_only:
        raise ValueError('FROZEN_BOOTSTRAP2000_REQUIRED')
    expected = {(i,p,k) for i in membership for p in POLICIES for k in range(6)}
    actual = {(r['scenario_ordinal'],r['policy'],r['budget']) for r in records}
    if actual != expected or len(records)!=len(expected) or any(r['status']!='completed' for r in records):
        raise ValueError('INCOMPLETE_PAIRED_DENOMINATOR')
    metrics = {p:{str(k):mean_metrics([r for r in records if r['policy']==p and r['budget']==k],membership)
                  for k in range(6)} for p in POLICIES}
    ab = {p:aubc_rows(records,p) for p in POLICIES}
    for p,rs in ab.items():
        for r in rs:
            matched=[x for x in records if x['policy']==p and x['scenario_ordinal']==r['scenario_ordinal'] and 1<=x['budget']<=5]
            r.update(stop=sum(x['stop'] for x in matched)/5,
                     budget_exhausted=sum(x['budget_exhausted'] for x in matched)/5)
    comparisons = {}
    for a,b in [('voi_h3','global_eig'),('voi_h1','global_eig'),('dynamic','voi_h3'),('voi_h3','random'),('voi_h3','fixed')]:
        comparisons[a+'__'+b] = paired_comparison(ab[a],ab[b],membership,repetitions=bootstrap_repetitions,seed=42)
    primary = comparisons['voi_h3__global_eig']['patient_macro']['top1']
    decision = 'FRESH_SYNTHETIC_VOI_GAIN' if primary['ci95'][0]>0 else (
        'WEAK_BASELINE_ONLY_GAIN' if comparisons['voi_h3__random']['patient_macro']['top1']['difference']>0 or comparisons['voi_h3__fixed']['patient_macro']['top1']['difference']>0 else 'NO_GENERALIZATION_GAIN')
    auc={p:mean_metrics(ab[p],membership) for p in POLICIES}
    for table in list(metrics.values())+[auc]:
        for entry in table.values():entry['accuracy_denominator']='all admitted visits, patient macro preserves encounter clusters; failures block complete inference'
    monte_carlo={}
    for k in range(6):
        rs=[r for r in records if r['policy']=='random' and r['budget']==k]
        groups=defaultdict(list)
        for r in rs:groups[membership[r['scenario_ordinal']]['patient_id']].append(r)
        monte_carlo[str(k)]={'visit_mean_top1_MC_SE':math.sqrt(sum(r['random_top1_mean_variance'] for r in rs))/len(rs),
            'patient_macro_top1_MC_SE':math.sqrt(sum(sum(r['random_top1_mean_variance'] for r in grp)/len(grp)**2 for grp in groups.values()))/len(groups),
            'scope':'conditional fixed-source finite-permutation MC, patient draws independent; not patient bootstrap or cross-budget AUBC error'}
    return {'metrics':metrics,'aubc':auc,
        'comparison':comparisons,'decision':decision,'primary_interval':primary,
        'random_monte_carlo':monte_carlo,
        'clinical_validity':'NOT_APPROVED','random_MC':'32 finite paths; per-row random_top1_mean_variance retained privately',
        'accuracy_denominator':'all admitted visits; patient macro averages visits per patient, then patients',
        'test_only':test_only}
