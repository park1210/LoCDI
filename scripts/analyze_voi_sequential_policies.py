"""C4-31 analysis-only clarification: matched-budget AUBC discordance.

Frozen policy execution code is untouched. No model/policy/refit/selection change.
Also summarizes actual stored random paths rather than theoretical marginals.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.evaluate_voi_sequential_policies as evaluation
from scripts.audit_research_gaps import read,rows,sha,write,check
from app.locdi.llm_setup import digest

ORIGINAL_AUBC=evaluation.aubc_rows
ORIGINAL_PAIRED=evaluation.paired_comparison
ADDITIONAL=['scripts/analyze_voi_sequential_policies.py','tests/locdi/test_voi_sequential_analysis.py']


def aubc_rows(records,policy):
    result=ORIGINAL_AUBC(records,policy)
    for r in result:
        r['top1_by_budget']={str(x['budget']):x['score']['top1'] for x in records if x['policy']==policy and x['scenario_ordinal']==r['scenario_ordinal'] and x['budget'] in range(1,6)}
    return result


def paired_comparison(left,right,membership,**kw):
    result=ORIGINAL_PAIRED(left,right,membership,**kw)
    if left and 'top1_by_budget' in left[0]:
        other={r['scenario_ordinal']:r for r in right};joint=Counter()
        for r in left:
            a=r['top1_by_budget'];b=other[r['scenario_ordinal']]['top1_by_budget']
            if set(a)!=set(b) or set(a)!={str(k) for k in range(1,6)}:raise ValueError('AUBC_MATCHED_BUDGETS')
            for k,x in a.items():
                y=b[k];joint['both']+=x*y/5;joint['left_only']+=x*(1-y)/5;joint['right_only']+=(1-x)*y/5;joint['neither']+=(1-x)*(1-y)/5
        result['correctness_joint_counts']=dict(joint)
        result['correctness_joint_scope']='mean of SAME-budget joint counts over budgets1..5; not product of AUBC marginal accuracies'
    return result


def register(namespace):
    pub,pr=evaluation.places(namespace);evaluation.verify_protocol(pub)
    clarification={'type':'ANALYSIS_CALCULATION_CLARIFICATION_ONLY','registered_at':datetime.now(timezone.utc).isoformat(),
        'reason':'AUBC fractional marginals cannot be multiplied to infer matched-budget joint correctness; use mean of paired within-budget joint indicators',
        'code_hashes':{n:sha(ROOT/n) for n in ADDITIONAL},'policy_code_changed':False,'policy_results_read_for_registration':False,
        'primary_metric_changed':False,'bootstrap_changed':False,'new_comparisons':False,
        'execution_protocol_fingerprint':read(pub/'sequential_policy_protocol.json')['fingerprint']}
    clarification['fingerprint']=digest(clarification);write(pub/'analysis_calculation_clarification.json',clarification)


def validate(namespace):
    pub,pr=evaluation.places(namespace);evaluation.verify_protocol(pub)
    c=read(pub/'analysis_calculation_clarification.json')
    if digest({k:v for k,v in c.items() if k!='fingerprint'})!=c['fingerprint'] or check(c['code_hashes'])['changed_or_missing']:raise ValueError('ANALYSIS_CODE_CHANGED')
    return pub,pr


def analyze(namespace):
    pub,pr=validate(namespace)
    evaluation.aubc_rows=aubc_rows;evaluation.paired_comparison=paired_comparison
    evaluation.analyze(namespace)
    # Actual random path counts averaged over32 registered repetitions.
    keys=read(evaluation.PARENT/'action_keys.json');base=set(keys['base'])
    inputs={r['scenario_ordinal']:r for r in read(evaluation.PREP/'policy_inputs.json')}
    models={f:read(evaluation.PARENT/('fold%02d_M5_model.json'%f)) for f in range(1,6)}
    counts=Counter();pos=Counter();kind=Counter();bybudget=defaultdict(Counter);paths=0;leak=0
    with (pr/'random_paths.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);paths+=1;m=models[inputs[r['scenario_ordinal']]['fold']];reverse={a:k for k,a in m['provenance']['action_key_mapping'].items()}
            for j,e in enumerate(r['releases'],1):
                a=e['action_id'];counts[a]+=1;pos[str(j)+':'+a]+=1;kind['base5' if reverse[a] in base else 'new3']+=1;bybudget[r['budget']][a]+=1
                leak+=e['released_at_minute']<e['available_minute']
    write(pub/'actual_random_action_distribution.json',{'paths':paths,'actual_raw_path_acquisitions':sum(counts.values()),
        'per32_mean_action_counts':{a:n/32 for a,n in counts.items()},'per32_mean_position_action_counts':{a:n/32 for a,n in pos.items()},
        'per32_mean_base5_new3':{a:n/32 for a,n in kind.items()},'per_budget_mean_counts':{k:{a:n/32 for a,n in c.items()} for k,c in bybudget.items()},
        'future_release_violations':leak,'scope':'actual finite32 random paths, independent replay per budget; averaged counts are fractional'})


def regression(namespace):
    import unittest,time
    from unittest.mock import patch
    import socket
    pub,pr=validate(namespace)
    old=read(pub/'test_results.json');suite=unittest.TestSuite();counts={}
    for pattern in old['patterns']:
        if pattern.endswith('.py'):
            ss=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern=pattern);counts[pattern]=ss.countTestCases();suite.addTests(ss)
    ss=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern='test_voi_sequential_analysis.py');counts['test_voi_sequential_analysis.py']=ss.countTestCases();suite.addTests(ss)
    ss=unittest.defaultTestLoader.loadTestsFromName('test_phase211_alignment.DecisionMathAlignmentTests');counts['test_phase211_alignment.DecisionMathAlignmentTests']=ss.countTestCases();suite.addTests(ss)
    start=time.perf_counter()
    with (pr/'final_regression.log').open('x',encoding='utf-8') as f:
        with patch.object(socket.socket,'connect',side_effect=RuntimeError('NETWORK_DISABLED_C4_31')),patch.object(socket.socket,'bind',side_effect=RuntimeError('NETWORK_DISABLED_C4_31')):
            result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    report={'total':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.perf_counter()-start,
        'patterns':counts,'scope':'RELATED_REGRESSION including analysis clarification; full suite NOT_RUN','network':'DISABLED','generation':0,'source_fits':0}
    write(pub/'final_test_results.json',report);print(json.dumps(report),flush=True)
    if not result.wasSuccessful():raise ValueError('FINAL_REGRESSION_FAILED')


def seal(namespace):
    pub,pr=validate(namespace);protected=check(read(pr/'protected_hashes.json'));sources=check(read(pr/'source_hashes.json'))
    if protected['changed_or_missing'] or sources['changed_or_missing']:raise ValueError('PRESERVATION_FAILED')
    text='\n'.join(p.read_text(encoding='utf-8') for p in pub.glob('*.json'))
    ids=read(evaluation.PREP/'identity_membership.json');refs=read(evaluation.PREP/'scoring_keys.json')
    sensitive={r[k] for r in ids for k in ('patient_id','visit_id')}|{r['source_diagnosis_reference'] for r in refs}
    if any(s in text for s in sensitive):raise ValueError('PUBLIC_ID_OR_REFERENCE')
    if read(pub/'final_test_results.json')['failures'] or read(pub/'final_test_results.json')['errors']:raise ValueError('TEST_FAILURE')
    write(pub/'verification.json',{'status':'PASS','protected':protected,'sources':sources,
        'code':check(read(pub/'sequential_policy_protocol.json')['code_hashes']), 'analysis_code':check(read(pub/'analysis_calculation_clarification.json')['code_hashes']),
        'Core_changes':0,'source_changes':0,'old_artifact_changes':0,'old_report_changes':0,'LLM_generation':0,'new_probability_fit':0,'routing_fit':0,'new_gold':0,
        'source_sampling':0,'manuscript_changes':0,'git_commit_push':False,'public_identity_and_reference_scan':'PASS','clinical_evaluation_ready':False,
        'official_B0_B4':'NOT_RUN','D7':'Existing code-version mismatch remains; original artifact not rewritten/resealed'})
    commands=[('evaluate_voi_sequential_policies.py','prepare'),('evaluate_voi_sequential_policies.py','run'),('evaluate_voi_sequential_policies.py','regression'),
              ('analyze_voi_sequential_policies.py','register'),('analyze_voi_sequential_policies.py','analyze'),('analyze_voi_sequential_policies.py','regression'),('analyze_voi_sequential_policies.py','seal')]
    with (pub/'actual_execution_command.md').open('x',encoding='utf-8') as f:
        f.write('# Actual execution stages\n\nThe first related regression ran concurrently with frozen policy execution. Analysis clarification was registered before scores were inspected.\n\n```powershell\n'+
            '\n'.join('python -B scripts/'+script+' '+stage+' --namespace '+namespace for script,stage in commands)+'\n```\n\nNo resume or overwrite. Reproduction requires a fresh namespace with identical inputs/code; final seal follows report creation.\n')
    index={'files':{p.relative_to(ROOT).as_posix():sha(p) for p in pr.rglob('*') if p.is_file()}};index['fingerprint']=digest(index['files']);write(pr/'private_index.json',index)
    report=next((ROOT/'docs').rglob('C4_31_VOI_SEQUENTIAL_POLICY_EVALUATION_REPORT.md'))
    paths=[p for p in pub.rglob('*') if p.is_file()]+[pr/'private_index.json',report]+[ROOT/n for n in evaluation.CODE+ADDITIONAL]
    manifest={'status':read(pub/'execution_status.json')['status'],'protocol_fingerprint':read(pub/'sequential_policy_protocol.json')['fingerprint'],
        'analysis_clarification_fingerprint':read(pub/'analysis_calculation_clarification.json')['fingerprint'],'files':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    manifest['fingerprint']=digest(manifest['files']);write(pub/'completion_manifest.json',manifest)
    print(json.dumps({'seal':'PASS','protected':protected['checked'],'sources':sources['checked']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['register','analyze','regression','seal']);ap.add_argument('--namespace',default='c4_31_v1');a=ap.parse_args();globals()[a.stage](a.namespace)
