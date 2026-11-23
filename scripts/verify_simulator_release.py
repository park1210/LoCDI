"""Read-only inputs; optional new structural RC namespace, NEVER live execution."""
import argparse,json,sys,unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/locdi'))
from app.locdi.dataset_freeze import load,sha,verify as verify_freeze
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.episode_exporter import _write,_json
from app.locdi.structural_release import (RUNS,inspect_case,audit_handoff,preservation,evaluate_release,
    verify_boundary,validate_episode_links)
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from app.locdi.development_features import read_artifact
from app.locdi.integrated_approval import preflight_integrated
from validation.research_artifact_gate import assess_artifacts
from scripts.verify_saved_er_pilot import verify as verify_er
import app.locdi.live_handoff as live
import app.locdi.live_runtime as live_er


def run_regression(root):
    """Phase 2.11 exact discover/isolation method, new immutable log paths."""
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern='test_*.py')
    with TemporaryDirectory(dir=str(ROOT/'outputs')) as directory,(root/'regression.txt').open('x',encoding='utf-8') as stream:
        with patch.object(live,'HANDOFF_ROOT',Path(directory)/'HANDOFF_TEST_ONLY'):
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    return {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
        'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'fail':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),
        'failures':[str(t) for t,_ in result.failures+result.errors],
        'skips':[(str(t),reason) for t,reason in result.skipped],
        'isolation':'PHASE_211_IDENTICAL_DISCOVER_AND_TEMPORARY_HANDOFF_ROOT_NO_TEST_SUPPRESSED',
        'source_suite_fingerprint':digest({p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'tests/locdi').glob('*.py'))})}


def verify_manifest(root,manifest):
    copy=dict(manifest);fingerprint=copy.pop('fingerprint',None)
    if digest(copy)!=fingerprint:return False
    expected=set()
    for row in manifest['files']:
        rel=Path(row['path']);path=(root/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or root.resolve() not in path.parents or row['path'] in expected:return False
        expected.add(row['path'])
        if not path.is_file() or sha(path)!=row['sha256']:return False
        try:
            if path.suffix=='.json':_json(load(path))
            elif path.suffix=='.jsonl':
                for line in path.read_text(encoding='utf-8').splitlines():
                    if line.strip():_json(json.loads(line))
        except (ValueError,TypeError):return False
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and 'private' not in p.relative_to(root).parts and p.relative_to(root).as_posix()!='manifests/release_manifest.json'}
    private=root/'private/selection.json'
    private_valid=sha(private)==manifest['private_selection_sha256'] if manifest.get('private_selection_sha256') else not private.exists()
    report_valid=digest(load(root/'reports/release_decision.json'))==manifest['report_fingerprint'] if manifest.get('report_fingerprint') else True
    return actual==expected and private_valid and report_valid


def stored_offline():
    root=ROOT/'outputs/phase211';report=load(root/'reports/phase211_audit_v1.json')
    checks=[]
    for row in report['records']:
        name='TEST_ONLY_'+row['case_ref']+'_'+row['scenario']
        checks.append(validate_episode_links(load(root/'episodes'/(name+'.json')),load(root/'reports'/(name+'_handoff_v2.json')),
            load(root/'metrics'/(name+'.jsonl'))))
    snapshot=load(root/'reports/phase211_current_snapshot_v2.json')
    linked=sha(root/'reports/phase211_audit_v1.json')==snapshot['prior_snapshot_sha256'] and sha(root/'private/selection_v1.json')==snapshot['selection_manifest_sha256']
    return {'status':'PASS' if linked and len(checks)==40 and all(c['status']=='PASS' for c in checks) else 'FAIL',
        'preserved_scenarios':len(checks),'preserved_ordinals':[r['ordinal'] for r in report['selected_cases']],
        'scope':'STORED_EXPORT_AND_SNAPSHOT_INTEGRITY_NOT_NEW_PAYLOAD_OBSERVATION'}


def expanded_offline(cases,root=None):
    from phase211_support import execute_case,features
    previous=stored_offline()['preserved_ordinals']
    chosen=[i for i in range(1,len(cases)+1) if i not in previous][:5]
    if root is not None:
        selection={'schema_version':'locdi.rc.offline_selection.v1','selection_rule':'FIRST_FIVE_INPUT_ORDINALS_NOT_PHASE211',
            'chosen_before_outputs':True,'dataset_sha256':sha(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'),
            'cases':[{'ordinal':i,'patient_id':cases[i-1]['patient_id'],'visit_id':cases[i-1]['agent_visible']['index_encounter_id'],
                'source_fingerprint':digest(dict(cases[i-1]))} for i in chosen]}
        _write(selection,'private/selection.json',root,'private')
    rows=[]
    for ordinal in chosen:
        case=cases[ordinal-1]
        for scenario in ('A','B','C','D','E','F_TIMEOUT','G_BUDGET','H_INVALID_EVIDENCE'):
            result,client,row=execute_case(case,features(case,ordinal),scenario,output_root=root)
            row['boundary']=verify_boundary(case,client,result)
            if root is not None:
                name='TEST_ONLY_'+row['case_ref']+'_'+scenario
                row['export_integrity']=validate_episode_links(load(root/'episodes'/(name+'.json')),
                    load(root/'reports'/(name+'_handoff_v2.json')),load(root/'metrics'/(name+'.jsonl')))
            else:row['export_integrity']={'status':'NOT_CHECKED_NOT_SAVED'}
            rows.append(row)
    expected={'A':'completed','B':'completed','C':'completed','D':'completed','E':'completed',
        'F_TIMEOUT':'failed','G_BUDGET':'failed','H_INVALID_EVIDENCE':'failed'}
    good=len(rows)==40 and all(r['status']==expected[r['scenario']] and r['boundary']['status']=='PASS' and
        (root is None or r['export_status']=='EXPORTED' and r['export_integrity']['status']=='PASS') for r in rows)
    return {'status':'PASS' if good else 'FAIL','ordinals':chosen,'scenario_count':len(rows),
        'terminal_counts':dict(Counter(r['status'] for r in rows)),'records':rows,'actual_generation_calls':0}


def artifact_check(cases):
    config=load(ROOT/'configs/development_specialty_llm.json');registry=load(ROOT/config['llm']['registry_path'])
    artifact=read_artifact(config['voi_model_path']) if config['voi_model_path'] else None
    utility=read_artifact(config['routing_utility_path']) if config['routing_utility_path'] else None
    inputs=actual_gate_inputs()
    args=dict(routing_artifact=utility,voi_artifact=artifact,correctness_inputs=inputs,
        evaluation_patient_ids=[c['patient_id'] for c in cases],
        source_actions=[e for c in cases for e in c['environment']['current_encounter_events']],diagnosis_candidates=None)
    readiness=assess_artifacts(registry,**args)
    plans=[preflight_integrated(run_id='202610300000000000000000000212%02d'%i,mode=mode,**args)
        for i,mode in enumerate(('adaptive_routing','bayesian_voi','adaptive_and_voi'),1)]
    enforced=all(p['plan'] is not None and 'LIVE_INTEGRATED_EXECUTOR_NOT_IMPLEMENTED' in p['blocker_codes'] and
        any(c.endswith('_RESEARCH_ARTIFACT_BLOCKED') for c in p['blocker_codes']) for p in plans)
    return {'status':'PASS' if enforced and not readiness['research_evaluation_ready'] else 'FAIL',
        'research_readiness':'BLOCKED','gate':readiness,'live_plans':plans}


def verify_release(*,root=None,full_regression=False,expand=False):
    stages={}
    def stage(name,fn):
        try:value=fn()
        except Exception as exc:value={'status':'BLOCKED','error_codes':['CHECK_INPUT_OR_CONTRACT_UNAVAILABLE'],'error_type':type(exc).__name__}
        stages[name]=value
        if root is not None:_write(value,'reports/'+name+'.json',root,'reports')
        print(name+': '+value['status'],flush=True)
        return value
    baseline=load(ROOT/'outputs/private/phase212_baseline_hashes_v1.json')
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    stage('preservation_before',lambda:preservation(baseline))
    def er():
        r=verify_er();return {'status':'PASS' if r['classification']=='DEVELOPMENT_PILOT_COMPLETED' else 'FAIL','audit':r}
    stage('live_er',er)
    for i,run in enumerate(RUNS,1):stage('live_handoff_%d'%i,lambda run=run:audit_handoff(run))
    def static():
        registry=load(ROOT/'configs/agent_registry.specialty_development.json')
        rows=[inspect_case(c,i+1,registry) for i,c in enumerate(cases)]
        return {'status':'PASS' if len(rows)==50 and all(r['status']=='PASS' for r in rows) else 'FAIL',
            'case_count':len(rows),'rows':rows,'clinical_reference':'NOT_CHECKED','actual_generation_calls':0}
    stage('all_case_static',static)
    stage('dataset_freeze',verify_freeze)
    stage('artifact_gate',lambda:artifact_check(cases))
    stage('prior_offline',stored_offline)
    stage('additional_offline',lambda:expanded_offline(cases,root) if expand else {'status':'NOT_CHECKED','reason':'EXPAND_OFFLINE_NOT_REQUESTED'})
    def modes():
        a=live_er.preflight(run_id='d'*32);b=live.preflight(run_id='e'*32)
        return {'status':'PASS' if not a['blocker_codes'] and not b['blocker_codes'] else 'BLOCKED',
            'ER_ONLY_LIVE':'EXPLICIT_APPROVAL_REQUIRED_SERVER_NOT_CHECKED',
            'STATIC_HANDOFF_LIVE':'EXPLICIT_APPROVAL_REQUIRED_SERVER_NOT_CHECKED',
            'HANDOFF_ROUTING_TEST_ONLY':'OFFLINE_ONLY','HANDOFF_VOI_TEST_ONLY':'OFFLINE_ONLY',
            'HANDOFF_ROUTING_VOI_TEST_ONLY':'OFFLINE_ONLY','LIVE_ADAPTIVE_VOI':'BLOCKED',
            'local_preflight_blockers':{'er':a['blocker_codes'],'handoff':b['blocker_codes']},'actual_generation_calls':0}
    stage('mode_availability',modes)
    stage('regression',lambda:run_regression(root) if full_regression and root is not None else {'status':'NOT_CHECKED','reason':'CURRENT_FULL_REGRESSION_NOT_REQUESTED'})
    stage('preservation_after',lambda:preservation(baseline))
    def aggregate(*names):
        statuses=[stages[n]['status'] for n in names]
        return 'PASS' if all(s=='PASS' for s in statuses) else 'FAIL' if 'FAIL' in statuses else 'BLOCKED' if 'BLOCKED' in statuses else 'NOT_CHECKED'
    criteria={'structural_contract_ready':aggregate('dataset_freeze','mode_availability'),
        'offline_integration_ready':aggregate('prior_offline','additional_offline'),
        'live_er_verified':stages['live_er']['status'],'live_sequential_handoff_verified':stages['live_handoff_3']['status'],
        'all_case_static_validated':stages['all_case_static']['status'],'full_regression_passed':stages['regression']['status'],
        'source_boundary_verified':aggregate('additional_offline','all_case_static'),
        'artifact_gate_enforced':stages['artifact_gate']['status'],
        'clinical_reference_ready':'BLOCKED','live_adaptive_voi_ready':'BLOCKED','research_evaluation_ready':'BLOCKED',
        'preservation_verified':aggregate('preservation_before','preservation_after','live_handoff_1','live_handoff_2'),
        'output_integrity_verified':aggregate('prior_offline','additional_offline') if root is not None else 'NOT_CHECKED'}
    report={'schema_version':'locdi.structural_release.v1','release_id':root.name if root else None,
        'decision':evaluate_release(criteria),'stage_statuses':{k:v['status'] for k,v in stages.items()},
        'source_boundary_scope':'STATIC_ALL_50_AND_OBSERVED_TEST_ONLY_ADDITIONAL_5_NOT_HISTORICAL_RAW_WIRE_OR_CLINICAL_RELEASE',
        'paper_equivalence':'NOT_CHECKED_NO_WORKSPACE_ORIGINAL','pre_decision_hook':False,
        'current_order':'ER_PROPOSAL_THEN_INFORMATION_POLICY_BELIEF_AND_VOI_THEN_VALIDATED_ACTION',
        'clinical_metrics':'NOT_EVALUATED','actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0}
    if root is not None:
        _write(report,'reports/release_decision.json',root,'reports')
        files=[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file() and 'private' not in p.relative_to(root).parts]
        manifest={'schema_version':'locdi.structural_release_manifest.v1','files':files,'report_fingerprint':digest(report),
            'private_selection_sha256':sha(root/'private/selection.json') if (root/'private/selection.json').exists() else None,
            'source_code_fingerprint':digest({p.relative_to(ROOT).as_posix():sha(p) for folder in ('app','validation','scripts','tests/locdi') for p in sorted((ROOT/folder).rglob('*.py')) if '__pycache__' not in p.parts}),
            'commit_rule':'NO_VALID_RC_WITHOUT_THIS_MANIFEST_AND_SUCCESSFUL_HASH_RECHECK'}
        manifest['fingerprint']=digest(manifest)
        if not verify_manifest(root,manifest):raise ValueError('RC_OUTPUT_INTEGRITY_FAILURE')
        _write(manifest,'manifests/release_manifest.json',root,'manifests')
        if not verify_manifest(root,load(root/'manifests/release_manifest.json')):raise ValueError('RC_COMMIT_VERIFICATION_FAILURE')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true');parser.add_argument('--release-id',default='phase212_rc_v1')
    parser.add_argument('--full-regression',action='store_true');parser.add_argument('--expand-offline',action='store_true')
    args=parser.parse_args()
    if args.full_regression and not args.save:parser.error('--full-regression requires --save in a NEW namespace')
    root=None
    if args.save:
        import re
        if not re.fullmatch('[a-zA-Z0-9_-]{1,64}',args.release_id):parser.error('invalid release ID')
        base=(ROOT/'outputs/structural_rc').resolve();root=base/args.release_id
        if not base.is_relative_to((ROOT/'outputs').resolve()):parser.error('output redirected')
        try:root.mkdir(parents=True,exist_ok=False)
        except FileExistsError:parser.error('release ID already used; existing results preserved')
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind'}:raise RuntimeError('RC_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    try:
        report=verify_release(root=root,full_regression=args.full_regression,expand=args.expand_offline)
        print(json.dumps(report,ensure_ascii=True));sys.exit(0 if report['decision']['STRUCTURAL_RELEASE_CANDIDATE']=='READY_DEVELOPMENT_ONLY' else 2)
    except Exception:
        print(json.dumps({'status':'FAIL','error_codes':['RC_VERIFICATION_OR_NEW_OUTPUT_STORAGE_FAILED'],
            'commit_status':'NOT_VERIFIED_DO_NOT_APPROVE','actual_generation_calls':0}));sys.exit(2)
