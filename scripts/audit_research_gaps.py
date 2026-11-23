"""C4-20 read-only evidence audit. Never executes a policy/model or changes inputs.

Python 3.9 compatible; all writes are exclusive and confined to a fresh namespace.
freeze pins the completed C4-19 chains; analyze reads saved results; seal rehashes.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.llm_setup import digest


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def check(pins):
    bad = []
    for name, expected in pins.items():
        p = ROOT / name
        if not p.is_file() or sha(p) != expected:
            bad.append(name)
    return {'checked': len(pins), 'changed_or_missing': len(bad),
            'status': 'PASS' if not bad else 'FAIL', 'failures': bad}


def locations(namespace):
    if not namespace or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in namespace):
        raise ValueError('INVALID_NAMESPACE')
    return (ROOT / 'outputs/research_evaluation' / namespace,
            ROOT / 'outputs/private/research_evaluation' / namespace)


def freeze(namespace):
    pub, private = locations(namespace)
    if pub.exists() or private.exists():
        raise ValueError('NAMESPACE_ALREADY_USED_NO_RESUME')
    base = ROOT / 'outputs/private/research_evaluation/c4_19_v1'
    pins = read(base / 'protected_hashes.json')
    sources = read(base / 'source_hashes.json')
    parents = []
    for ns in ['c4_19_replay_002', 'c4_19_v1']:
        mp = ROOT / 'outputs/research_evaluation' / ns / 'output_manifest.json'
        m = read(mp)
        assert m['fingerprint'] == digest({k:v for k,v in m.items() if k != 'fingerprint'})
        ip = ROOT / m['private_index_path']
        assert sha(ip) == m['private_index_sha256']
        for name, value in {**m['files'], **read(ip)['files']}.items():
            if name in pins:
                assert pins[name] == value, 'CONFLICTING_PARENT_PIN'
            pins[name] = value
        pins[mp.relative_to(ROOT).as_posix()] = sha(mp)
        pins[ip.relative_to(ROOT).as_posix()] = sha(ip)
        parents.append({'path':mp.relative_to(ROOT).as_posix(), 'sha256':sha(mp)})
    a, b = check(pins), check(sources)
    assert a['status'] == b['status'] == 'PASS'
    write(private / 'protected_hashes.json', pins)
    write(private / 'source_hashes.json', sources)
    write(private / 'initial_preservation.json', {'protected':a, 'source':b, 'parents':parents})
    write(pub / 'analysis_protocol.json', {
        'version':'c4_20.audit.v1', 'scope':'READ_ONLY_GAP_ANALYSIS',
        'new_labels':False, 'new_policy':False, 'threshold_changes':False,
        'generation':0, 'fitting':0, 'official_ablation_execution':False,
        'parents':parents, 'manuscript_alignment':'NOT_CHECKED',
        'follow_up_selection_criteria':['existing source reference', 'no clinical label fabrication',
                                       'patient preserving', 'non-generative', 'core contribution'],
        'no_performance_improvement_hypothesis_confirmed':True})
    print(json.dumps({'freeze':'PASS','protected':len(pins),'source':len(sources)}))


def summary(values):
    values = sorted(values)
    def quantile(q):
        i = (len(values)-1)*q
        lo = int(i)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(i-lo)
    return {'count':len(values), 'min':values[0], 'p25':quantile(.25),
            'median':statistics.median(values), 'p75':quantile(.75),
            'max':values[-1], 'mean':statistics.mean(values),
            'quantile_method':'linear (n-1)*q; descriptive only'}


def signature_inventory(path):
    text = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({'name':node.name,'line':node.lineno,'signature':ast.unparse(node.args)})
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    out.append({'name':node.name+'.'+method.name,'line':method.lineno,
                                'signature':ast.unparse(method.args)})
    return {'path':path.relative_to(ROOT).as_posix(), 'sha256':sha(path), 'functions':out,
            'line_count':len(text.splitlines()), 'status':'AST_READ'}


def analyze(namespace):
    pub, private = locations(namespace)
    assert (pub / 'analysis_protocol.json').exists()
    assert not (pub / 'dynamic_horizon_saturation_analysis.json').exists()
    reports = []
    for p in sorted((ROOT / 'docs').rglob('*.md')):
        stage = re.match(r'C4_(\d+)', p.name)
        if (stage and 7 <= int(stage.group(1)) <= 19) or p.name.startswith('C3_'):
            text = p.read_text(encoding='utf-8-sig')
            reports.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),
                            'lines':len(text.splitlines()), 'headings':[s for s in text.splitlines() if s.startswith('#')]})
    write(private / 'document_inventory.json', reports)
    paths = [
        'app/core/longitudinal_memory.py','app/core/memory_retrieval.py','app/core/clinical_observation.py',
        'app/core/routing_utility.py','app/core/agent_registry.py','app/locdi/abstention_routing.py',
        'app/core/diagnostic_belief.py','app/core/information_outcomes.py','app/core/information_value.py',
        'app/core/voi_planning.py','app/core/voi_evaluation.py','app/locdi/dynamic_horizon.py',
        'app/core/hospital_llm_policy.py','app/core/longitudinal_runner.py',
        'app/locdi/integrated_runtime.py','app/locdi/handoff_integration.py',
        'app/locdi/evaluation_gate_audit.py','validation/routing_training_data.py',
        'validation/routing_correctness_gate.py',
        'experiments/run_locdi_ablation.py','scripts/evaluate_measurement_identity.py',
        'app/locdi/routing_evaluation.py']
    # Resolve the real correctness gate module rather than inventing a path.
    for p in (ROOT/'app/locdi').glob('*.py'):
        if 'def correctness_gate' in p.read_text(encoding='utf-8-sig'):
            paths.append(p.relative_to(ROOT).as_posix())
    inv = [signature_inventory(ROOT/p) for p in sorted(set(paths))]
    write(pub / 'code_contract_inventory.json', inv)
    write(pub / 'document_review_inventory.json', reports)
    trace_path = ROOT/'outputs/private/research_evaluation/c4_19_replay_002/dynamic_execution_trace.jsonl'
    trace = rows(trace_path)
    feats = {'normalized_entropy':[], 'margin':[], 'max_posterior':[]}
    horizons, request, remaining = Counter(), Counter(), Counter()
    changes = Counter()
    states = Counter()
    errors = []
    policy_checks = []
    for r in trace:
        b = r['horizon_input_belief']
        vals = sorted(b.values(), reverse=True)
        en = -sum(p*math.log(p) for p in b.values() if p)/math.log(len(b))
        margin = vals[0]-vals[1]
        n = len(r['information_action_interface'])
        hE = 1 if en<=1/3 else 2 if en<=2/3 else 3
        hM = 1 if margin>=2/3 else 2 if margin>=1/3 else 3
        expected = min(max(hE,hM), n,3) if n else None
        actual = r['results']['dynamic']['horizon_policy']
        # Recalculation tolerates only summation-order feature rounding; decisions exact.
        if expected != actual['selected_horizon']: errors.append('HORIZON_MISMATCH')
        observed = actual['observable_feature_summary']
        if not math.isclose(en,observed['normalized_entropy'],rel_tol=0,abs_tol=1e-14):errors.append('ENTROPY_MISMATCH')
        if margin != observed['top1_top2_margin']:errors.append('MARGIN_MISMATCH')
        policy_checks.append({'case_ref':r['case_ref'],'decision_ordinal':r['decision_ordinal'],
                              'entropy_recomputed':en,'margin_recomputed':margin,
                              'expected_horizon':expected,'saved_horizon':actual['selected_horizon']})
        feats['normalized_entropy'].append(en);feats['margin'].append(margin);feats['max_posterior'].append(vals[0])
        horizons[str(expected)]+=1;remaining[str(n)]+=1
        if n:request[str(max(hE,hM))]+=1
        for condition in ['H1','H2','H3','dynamic']:
            states[condition] += r['results'][condition]['planner_evaluated_states']
        for c in ['H1','H2','H3']:
            changes[c] += r['derived']['dynamic']['binding'] != r['derived'][c]['binding']
    assert len(trace)==263 and not errors
    saturation = {'scope':'DESCRIPTIVE_RECALCULATION_OF_SAVED_REPLAY_NO_NEW_POLICY',
        'decisions':len(trace),'patients_with_decisions':len({r['case_ref'] for r in trace}),
        'registered_patients':50,'distributions':{k:summary(v) for k,v in feats.items()},
        'selected_horizon':dict(horizons),'requested_horizon_nonempty':dict(request),
        'remaining_actions':dict(remaining),'changed_vs_fixed':dict(changes),'planner_states':dict(states),
        'sufficient_saturation_condition':'En > 2/3 OR margin < 1/3 => max(hE,hM)=3',
        'observed_stronger_condition':'Both hold on every saved belief; only 232 nonempty states request a horizon',
        'root_cause_confirmed':'Operational equal-third thresholds never cross observed belief features; action-count cap alone creates H1/H2',
        'posterior_diffuseness_cause':'POSSIBLE: sparse smoothed low-information likelihood; not isolated by counterfactual model experiments',
        'wall_time_speedup':'NOT_ESTABLISHED_CONCURRENT_REGRESSION_AND_INSTRUMENTATION',
        'new_policy_or_threshold_evaluated':False,'source_trace_sha256':sha(trace_path)}
    write(pub/'dynamic_horizon_saturation_analysis.json',saturation)
    write(private/'feature_recalculation.json',policy_checks)
    refp=ROOT/'outputs/private/research_evaluation/c4_10_v1/csv_export_aligned/verified_identity_reference.jsonl'
    labels,unknown,unknown_snapshots=Counter(),Counter(),set()
    for r in rows(refp):
        labels[r['label']]+=1
        if r['label']=='UNKNOWN':unknown[r['reason']]+=1;unknown_snapshots.add(r['snapshot_id'])
    retrieval=read(ROOT/'outputs/research_evaluation/c4_10_v1/csv_export_aligned/retrieval_evaluation.json')
    complete=[c for c in retrieval['cases'] if c['modes']['selective']['recall_status']=='COMPUTED_COMPLETE_POOL']
    numerator=sum(c['modes']['selective']['positive_returned'] for c in complete)
    denominator=sum(c['modes']['selective']['full_pool_positive'] for c in complete)
    positives=sum(c['modes']['selective']['positive_returned'] for c in retrieval['cases'])
    judged=sum(c['modes']['selective']['judged'] for c in retrieval['cases'])
    zero=sum(c['modes']['selective']['full_pool_positive']==0 for c in retrieval['cases'])
    cap=sum(min(5,c['modes']['selective']['full_pool_positive']) for c in complete)
    memory={'records':sum(labels.values()),'labels':dict(labels),'unknown_reasons':dict(unknown),
        'query_count':len(retrieval['cases']),'unknown_queries':len(unknown_snapshots),
        'zero_positive_queries':zero,'computable_recall_queries':len(complete),
        'precision':{'numerator':positives,'denominator':judged,'value':positives/judged},
        'recall':{'numerator':numerator,'denominator':denominator,'value':numerator/denominator},
        'top5_record_recall_capacity_bound':{'maximum_numerator':cap,'denominator':denominator,'value':cap/denominator,
                                           'meaning':'Arithmetic capacity bound, not new retrieval performance'},
        'clinical_relevance':'NOT_REVIEWED','labels_generated':0,'references_changed':0,
        'reference_sha256':sha(refp)}
    assert memory['records']==72590 and numerator==198 and denominator==2242
    write(pub/'memory_reference_diagnostics.json',memory)
    configs={f'B{i}':read(ROOT/('experiments/configs/B%d.json'%i)) for i in range(5)}
    diffs={f'B{i-1}->B{i}':{k:{'before':configs[f'B{i-1}']['features'][k],'after':v}
             for k,v in configs[f'B{i}']['features'].items() if v!=configs[f'B{i-1}']['features'][k]}
             for i in range(1,5)}
    runner=ast.parse((ROOT/'experiments/run_locdi_ablation.py').read_text(encoding='utf-8-sig'))
    func=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='run_ablation')
    raises=any(isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call) and isinstance(n.exc.func,ast.Name)
               and n.exc.func.id=='NotImplementedError' for n in ast.walk(func))
    write(pub/'ablation_feasibility_assessment.json',{
        'official_features':{k:v['features'] for k,v in configs.items()},'adjacent_differences':diffs,
        'ready':{k:v['ready'] for k,v in configs.items()},'unresolved_common':{k:[f for f,x in v['common'].items() if x is None] for k,v in configs.items()},
        'runner_not_implemented_raise_verified_ast':raises,'official_runner_executed':False,
        'execution_readiness':'BLOCKED','independent_clinical_validation':'BLOCKED',
        'historical_results':'C4-18 historical B4: llm/memory_aware_llm router, topK15, context16384; not current official B4',
        'minimal_implementation_order':['Reviewed/pinned execution inputs and unified adapter',
             'Patient x repeat x condition coordinator with no overwrite and per-case failures',
             'Shared source release/action validation and scheduler; OFF/full/top5 translation',
             'Reviewed live Routing/VoI/handoff injection, not promotion of TEST_ONLY executor',
             'Separate reference/scoring loader, paired completeness and denominator checks'],
        'interpretation':'B2->B3 includes specialty access; B3->B4 includes action/timing change; no single-effect claim',
        'consult_source_events_original50':0,'source_consult_and_model_specialty_opinion_distinct':True})
    write(pub/'recalculation_checks.json',{'horizon_decisions_checked':len(trace),'horizon_mismatches':len(errors),
        'memory_totals_match':dict(labels)=={'POSITIVE':3884,'NEGATIVE':68602,'UNKNOWN':104},
        'recall_numerator':numerator,'recall_denominator':denominator,'unknown_queries':len(unknown_snapshots),
        'zero_positive_queries':zero,'clinical_labels_generated':0,'new_policy_runs':0})
    print(json.dumps({'analyze':'PASS','decisions':len(trace),'records':sum(labels.values()),
                      'reports_read':len(reports),'code_modules':len(inv),'memory':memory},ensure_ascii=True))


def routing(namespace):
    """Reapply existing Gate; never converts labels or invokes fit_utility."""
    from app.locdi.evaluation_gate_audit import actual_gate_inputs
    from validation.routing_correctness_gate import validate_evaluation_gate
    from app.locdi.routing_data_reuse import BANK
    pub, private = locations(namespace)
    data, options = actual_gate_inputs()
    gate = validate_evaluation_gate(data, **options)
    old = data['outcomes']
    newroot = ROOT/'outputs/private/research_evaluation/c4_14_v1/collection/c4_14_ten_001'
    new = [read(p) for p in sorted(newroot.glob('Request*_after.json'))]
    def counts(rs):
        certainty = Counter(r['response']['certainty'] for r in rs)
        return {'responses':len(rs),'certainty':dict(certainty),
                'correct_nonnull':sum(r.get('correct') is not None for r in rs)}
    bank = rows(BANK)
    artifact = read(BANK.parent/'voi_model.json')
    patients={s:{d['patient_id'] for d in data['decisions'] if d['split']==s} for s in ['train','validation']}
    holdout=set(options['holdout_patient_ids'])
    vp=set(artifact['provenance']['training_patients'])
    split={'train':len(patients['train']),'validation':len(patients['validation']),'holdout':len(holdout),
           'train_validation_overlap':len(patients['train']&patients['validation']),
           'train_holdout_overlap':len(patients['train']&holdout),
           'validation_holdout_overlap':len(patients['validation']&holdout),
           'routing_train_voi_train_overlap':len(patients['train']&vp),
           'routing_validation_voi_train_overlap':len(patients['validation']&vp),
           'holdout_voi_train_overlap':len(holdout&vp),
           'bank_visits':len(bank),'bank_patients':len({r['training_patient_id'] for r in bank})}
    assert counts(old)['certainty']=={'supported':22,'unresolved':53}
    assert counts(new)['certainty']=={'supported':4,'unresolved':6}
    assert gate['complete_train_groups']==0 and gate['utility_fit_ready'] is False
    write(private/'existing_gate_recheck.json', gate)
    write(pub/'routing_reference_recheck.json',{
        'original75':counts(old),'separate_new10':counts(new),'merged':False,
        'new10_schema_valid':sum(r['schema_valid'] is True for r in new),
        'new10_complete_collection_groups':sum(len([r for r in new if r['case_ref']==c])==5 for c in {r['case_ref'] for r in new}),
        'complete_train_groups':gate['complete_train_groups'], 'validation_group_count':gate['validation_group_count'],
        'utility_fit_ready':gate['utility_fit_ready'],'blocker_codes':gate['blocker_codes'],
        'split':split,'new_reference_labels':0,'fitting':0})
    print(json.dumps({'routing_recheck':'PASS','split':split,'fit_ready':False}))


def seal(namespace):
    pub, private = locations(namespace)
    before=read(private/'initial_preservation.json')
    protected=check(read(private/'protected_hashes.json'));sources=check(read(private/'source_hashes.json'))
    assert protected['status']==sources['status']=='PASS'
    verification={'status':'PASS','initial':before,'final':{'protected':protected,'sources':sources},
        'core_runtime_gate_changes':0,'prior_artifact_changes':0,'source_changes':0,
        'new_generation':0,'fitting':0,'clinical_verdicts_generated':0,'official_ablation_run':False,
        'manuscript_modified':False,'git_commit_push':False}
    write(pub/'verification.json',verification)
    report=ROOT/'docs/연구 평가/C4_20_RESEARCH_GAPS_AND_REMEDIATION_PLAN.md'
    newfiles=[p for p in pub.iterdir() if p.is_file()]+[report,Path(__file__).resolve()]
    index={'files':{p.relative_to(ROOT).as_posix():sha(p) for p in private.iterdir() if p.is_file()}}
    write(private/'private_index.json',index)
    manifest={'status':'C4_20_REMEDIATION_PLAN_COMPLETE',
        'files':{p.relative_to(ROOT).as_posix():sha(p) for p in newfiles},
        'private_index_path':(private/'private_index.json').relative_to(ROOT).as_posix(),
        'private_index_sha256':sha(private/'private_index.json'),'generation':0,'fitting':0,
        'clinical_validation':'NOT_EVALUATED','manuscript_alignment':'NOT_CHECKED'}
    manifest['fingerprint']=digest(manifest)
    write(pub/'output_manifest.json',manifest)
    print(json.dumps({'seal':'PASS','protected':protected['checked'],'sources':sources['checked']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','analyze','routing','seal'])
    p.add_argument('--namespace',default='c4_20_v1')
    a=p.parse_args()
    globals()[a.action](a.namespace)
