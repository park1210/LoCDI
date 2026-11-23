"""C3-01 offline audit. New namespace only; no generation/grading/fitting."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.clinical_gap_analysis import audit
from app.locdi.dataset_freeze import load,sha
from app.locdi.source_replay import atomic_rows
from app.locdi.structural_release import preservation
from app.locdi.llm_setup import digest


def verify_findings(result):
    a=result['initial_information_matrix'];r=result['routing_group_analysis'];s=result['split_independence_matrix']
    tests={
        'Original50':a['Original']['cases']==50 and all(a['Original']['fields'][k]['value']==0 for k in ('chief_complaint','initial_vitals','objective_findings','history')),
        'Original_class_actions':a['Original']['encounter_classes']=={'wellness':33,'outpatient':8,'urgentcare':6,'emergency':3} and
            a['Original']['ask_template_count']==1 and a['Original']['test_target_count']==4 and a['Original']['source_consult_count']==0,
        'D3_29':a['D3']['cases']==29 and all(a['D3']['fields'][k]['value']==0 for k in ('chief_complaint','initial_vitals','objective_findings','current_symptoms')) and a['D3']['test_target_count']==33,
        'Strict_source':result['temporal_visibility_matrix']['D3_strict_loader']=='BLOCKED_ACTUAL_AVAILABILITY_UNVERIFIED',
        'Actual75':(r['decision_count'],r['response_count'],r['supported'],r['unresolved'],r['named_diagnosis_unresolved'],r['correctness_nonnull'])==(15,75,22,53,52,0),
        'Complete_group_constraint':r['collection_complete_groups']==15 and r['all_decisions_have_unresolved'] and r['supported_only_complete_groups']==0 and r['correctness_gate']['complete_train_groups']==0,
        'Split':s['membership_recheck']=='PASS' and s['routing_validation_in_voi_train']==5 and s['counts']=={'holdout':50,'D3_replay':29,'bank':173,'voi_train':173,'routing_train':10,'routing_validation':5},
        'Review_no_conversion':result['review_contract_comparison']['conversion_status']=='NOT_IMPLEMENTED_NO_AUTOMATIC_CONVERSION',
        'D7':result['verification_summary']['D7_hash_verification']['status']=='PASS',
        'RC':result['verification_summary']['RC']['status']=='PASS',
        'Clinical_blocked':result['verification_summary']['clinical_reference_readiness']=='BLOCKED',
        'B0_B4_blocked':all(r['research_status']=='BLOCKED' for r in result['baseline_dependency_matrix']['rows'])}
    return {'status':'PASS' if all(tests.values()) else 'FAIL','checks':tests}


def privacy_check(root,document=None):
    identifiers=set();text_values=set()
    def collect(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in ('patient_id','visit_id','case_id','encounter_id','index_encounter_id','training_patient_id') and isinstance(x,str) and x:identifiers.add(x)
                collect(x)
        elif isinstance(v,list):
            for x in v:collect(x)
    from app.locdi.source_replay import PUBLIC
    dm=load(PUBLIC/'dataset_manifest.json')
    for name in ('cases.jsonl','action_events.jsonl'):
        for line in (ROOT/dm['files'][name]['path']).read_text(encoding='utf8').splitlines():collect(json.loads(line))
    for line in (ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl').read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        c=json.loads(line);collect(c)
        text_values.update(e['evidence'] for e in c['environment']['current_encounter_events'] if len(e['evidence'])>=24)
    for p in (ROOT/'outputs/private/routing_collection/phase24b_collect_001').glob('Candidate*.json'):collect(load(p))
    forbidden={'patient_id','visit_id','case_id','gold','evaluation','provenance','source_event_store','raw_response'}
    def private(v):
        if isinstance(v,dict):return bool(set(v)&forbidden) or any(private(x) for x in v.values())
        return isinstance(v,list) and any(private(x) for x in v)
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
    if document and document.is_file():files.append(document)
    codes=[]
    for p in files:
        raw=p.read_text(encoding='utf8')
        if any(v in raw for v in identifiers|text_values):codes.append('KNOWN_PRIVATE_ID_OR_SOURCE_TEXT')
        if p.suffix=='.json' and private(load(p)):codes.append('PRIVATE_JSON_KEY')
    return {'status':'PASS' if not codes else 'FAIL','public_files_checked':len(files),'leak_codes':sorted(set(codes)),
        'scope':'Known Original/D3/collection IDs, Original evidence>=24 chars and forbidden JSON keys; not independent clinical privacy or raw-wire audit'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--save',action='store_true')
    parser.add_argument('--namespace',default='c3_01');parser.add_argument('--regression',action='store_true')
    parser.add_argument('--reuse-regression-from',help='Existing new-branch namespace; suite fingerprint and PASS log required')
    args=parser.parse_args()
    if args.regression and not args.save:parser.error('--regression requires new --save namespace')
    if args.reuse_regression_from and (not args.save or args.regression):parser.error('reuse requires --save and no --regression')
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C3_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    try:
        root=None
        if args.save:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.namespace):raise ValueError('INVALID_NAMESPACE')
            base=(ROOT/'outputs/research_evaluation').resolve();root=base/args.namespace
            if not base.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('OUTPUT_REDIRECTED')
            root.mkdir(parents=True,exist_ok=False)
        result=audit();checks=verify_findings(result)
        if root:
            for name,value in result.items():
                if name!='verification_summary':atomic_rows(root,name+'.json',value,jsonl=False)
        regression={'status':'NOT_CHECKED'}
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            directory=root/'regression';directory.mkdir(exist_ok=False)
            regression=run_regression(directory);atomic_rows(directory,'result.json',regression,jsonl=False)
        elif args.reuse_regression_from:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.reuse_regression_from):raise ValueError('INVALID_REGRESSION_NAMESPACE')
            prior=ROOT/'outputs/research_evaluation'/args.reuse_regression_from/'regression'
            regression=load(prior/'result.json')
            suite_fp=digest({p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'tests/locdi').glob('*.py'))})
            log=(prior/'regression.txt').read_text(encoding='utf8')
            if (regression.get('status')!='PASS' or regression.get('fail')!=0 or regression.get('errors')!=0
                    or regression.get('skip')!=1 or regression.get('source_suite_fingerprint')!=suite_fp
                    or 'Ran %d tests'%regression['tests'] not in log or 'OK (skipped=1)' not in log):
                raise ValueError('REGRESSION_EVIDENCE_INVALID')
            regression={**regression,'reused':True,'evidence_namespace':args.reuse_regression_from,
                'result_sha256':sha(prior/'result.json'),'log_sha256':sha(prior/'regression.txt'),
                'reuse_scope':'Same test sources; metadata key rename only, supplemented by targeted tests. Not a new full regression.'}
        kept=preservation(load(ROOT/'outputs/private/c3_01_baseline_hashes_v1.json'))
        document=next((p for p in (ROOT/'docs').rglob('C3_01_CLINICAL_RESEARCH_GAP_CONSOLIDATION_REPORT.md')),None)
        privacy=privacy_check(root,document) if root else {'status':'NOT_CHECKED_NOT_SAVED'}
        verification=result['verification_summary'];verification.update(findings=checks,regression=regression,preservation=kept,privacy=privacy)
        requested_regression=args.regression or bool(args.reuse_regression_from)
        ok=checks['status']==kept['status']=='PASS' and (not root or privacy['status']=='PASS') and (not requested_regression or regression['status']=='PASS')
        verification['C3_01_AUDIT']='PASS_DEVELOPMENT_ONLY' if ok and root and requested_regression else 'INCOMPLETE' if ok else 'FAIL'
        if root:atomic_rows(root,'verification_summary.json',verification,jsonl=False)
        print(json.dumps({'C3_01_AUDIT':verification['C3_01_AUDIT'],'findings':checks,'regression':regression,'preservation':kept['status'],
            'actual_generation_calls':0,'grading':0,'fitting':0,'clinical_verdicts_generated':0},ensure_ascii=True))
        sys.exit(0 if ok else 2)
    except Exception as exc:
        print(json.dumps({'C3_01_AUDIT':'FAIL','error_type':type(exc).__name__,'error_code':'AUDIT_OR_NEW_NAMESPACE_UNAVAILABLE','actual_generation_calls':0}));sys.exit(2)
