"""C3-06 external task preparation; dry-run or new offline audit namespace only."""
import argparse
import json
from pathlib import Path
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha,verify as verify_freeze
from app.locdi.source_replay import atomic_rows
from app.locdi.routing_policy_analysis import verify_manifest
from app.locdi.public_reference_preparation import contract_outputs,metadata_notes,acquisition_plans
from app.locdi.external_reference_readiness import evaluate_gate
from app.locdi.structural_release import preservation
from app.locdi.research_protocol_audit import inspect_rc
from scripts.audit_c3_01 import privacy_check


def verify_prior():
    result={}
    for stage,name in [('c3_01_v3','analysis_manifest.json'),('c3_02_v1','completion_manifest.json'),
        ('c3_03_v1','completion_manifest.json'),('c3_04_v1','completion_manifest.json'),('c3_05_v1','completion_manifest.json')]:
        result[stage]=verify_manifest(ROOT/'outputs/research_evaluation'/stage/name)
    return result


def contract_tests(folder):
    """TEST_ONLY suite, no raw fixture output or successful-case clinical metrics."""
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/locdi'),pattern='test_external_reference_contract.py')
    with (folder/'test_only.txt').open('x',encoding='utf8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    return {'schema_version':'locdi.c3_06.test_only.v1','status':'PASS' if result.wasSuccessful() else 'FAIL',
       'tests':result.testsRun,'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
       'fail':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),
       'source_sha256':sha(ROOT/'tests/locdi/test_external_reference_contract.py'),
       'fixtures':'TEST_ONLY symbolic questions/options/byte streams; no acquired raw or actual clinical verdicts',
       'actual_network_requests':0,'actual_model_requests':0,'actual_grader_requests':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true');parser.add_argument('--namespace',default='c3_06_v1')
    parser.add_argument('--regression',action='store_true')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-download',action='store_true')
    args=parser.parse_args(argv)
    if args.execute or args.allow_download:
        print(json.dumps({'status':'ACQUISITION_BLOCKED','blocker_codes':['C3_06_REAL_DOWNLOAD_FORBIDDEN_TEST_ONLY_STAGE'],'actual_downloads':0}))
        return 2
    if args.regression and not args.save:parser.error('--regression requires new --save namespace')
    def guard(event,arguments):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C3_06_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    try:
        root=private=None
        if args.save:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.namespace):raise ValueError('INVALID_NAMESPACE')
            root=ROOT/'outputs/research_evaluation'/args.namespace;private=ROOT/'outputs/private/research_evaluation'/args.namespace
            if not root.resolve().is_relative_to((ROOT/'outputs/research_evaluation').resolve()) or not private.resolve().is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('PATH_ESCAPE')
            if root.exists() or private.exists():raise FileExistsError('NAMESPACE_ALREADY_USED')
            root.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
        prior=verify_prior()
        old=load(ROOT/'outputs/research_evaluation/c3_05_v1/unresolved_evidence_audit.json')
        assert (old['response_count'],old['supported'],old['unresolved'],old['named_unresolved'],old['literal_unresolved'],old['gate_accepted_fit_groups'])==(75,22,53,52,1,0)
        gates=[];private_receipts=[]
        for note,plan in zip(metadata_notes(),acquisition_plans()):
            evidence=None
            if private:
                name=note['dataset_id'].lower()+'_metadata.json';atomic_rows(private,name,note,jsonl=False)
                evidence={'path':str(private/name),'sha256':sha(private/name)}
                private_receipts.append({'dataset_id':note['dataset_id'],'document':name,'sha256':evidence['sha256']})
            gates.append(evaluate_gate(plan,metadata=evidence))
        outputs=contract_outputs(gates)
        from app.locdi.external_dialogue import MANIFEST,EXPECTED_MANIFEST
        hb=load(MANIFEST)
        assert hb['manifest_fingerprint']==EXPECTED_MANIFEST==digest({k:v for k,v in hb.items() if k!='manifest_fingerprint'})
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in hb['files'].values())
        source_index={'schema_version':'locdi.private.c3_06_source_index.v1','metadata_receipts':private_receipts,
            'source_rows_acquired':0,'actual_approval_records':[],'clinical_review_records':[],
            'prior_manifests':prior,'healthbench_manifest_sha256':sha(MANIFEST),
            'existing_routing_audit_sha256':sha(ROOT/'outputs/research_evaluation/c3_05_v1/unresolved_evidence_audit.json')}
        if private:atomic_rows(private,'source_index.json',source_index,jsonl=False)
        for value in gates:
            assert not value['states']['RAW_ACQUIRED'] and not value['states']['EXTERNAL_EVALUATION_READY']
        check={'status':'PASS','current_raw_acquired':False,'external_evaluation_ready':False,'acquisition_plans_blocked':3,
            'routing_response_count':75,'routing_fit_group_count':0,'healthbench_hash':'PASS','prior_manifests':prior}
        test_result={'status':'NOT_CHECKED'};regression={'status':'NOT_CHECKED'}
        if root:
            for name,value in outputs.items():atomic_rows(root,name+'.json',value,jsonl=False)
            folder=root/'verification';folder.mkdir(exist_ok=False)
            test_result=contract_tests(folder);atomic_rows(root,'test_only_verification.json',test_result,jsonl=False)
        print(json.dumps({'stage':'OFFLINE_PREPARED' if root else 'DRY_RUN','checks':check,'test_only':test_result}),flush=True)
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            folder=root/'regression';folder.mkdir(exist_ok=False)
            regression=run_regression(folder);atomic_rows(folder,'result.json',regression,jsonl=False)
        d7,rc=verify_freeze(full=False),inspect_rc()
        kept=preservation(load(ROOT/'outputs/private/c3_06_baseline_hashes_v1.json'))
        doc=next((ROOT/'docs').rglob('C3_06_REFERENCE_ACQUISITION_AND_EVALUATION_CONTRACT_REPORT.md'),None)
        privacy=privacy_check(root,doc) if root else {'status':'NOT_CHECKED_NOT_SAVED'}
        passed=all(v['status']=='PASS' for v in (d7,rc,kept))
        if root:passed=passed and test_result['status']==privacy['status']=='PASS'
        if args.regression:passed=passed and regression['status']=='PASS'
        ready=passed and bool(root) and args.regression
        summary={'schema_version':'locdi.c3_06.readiness.v1',
          'C3_06_REFERENCE_CONTRACT_READY':'PASS_DEVELOPMENT_ONLY' if ready else 'INCOMPLETE' if passed else 'FAIL',
          'RAW_ACQUIRED':False,'EXTERNAL_EVALUATION_READY':False,'CLINICAL_REFERENCE_READY':False,
          'ROUTING_ARTIFACT_READY':False,'ROUTING_POLICY_APPROVED':False,'PRODUCTION_AUTHENTICATION_READY':False,'OFFICIAL_B0_B4_READY':False,
          'checks':check,'test_only':test_result,'regression':regression,'D7':d7,'RC':rc,'preservation':kept,'privacy':privacy,
          'actual_generation_calls':0,'actual_grader_calls':0,'utility_fitting':0,'voi_probability_estimation':0,
          'new_raw_downloads':0,'clinical_verdicts_generated':0,
          'private_index_sha256':sha(private/'source_index.json') if private else None,
          'claim_scope':'Software contracts; current new raw0; prior actual75 analyzed through hash-preserved C3-05 evidence, not new model execution',
          'test_scope':'Existing regression includes in-memory TEST_ONLY fitting; no real artifact fitting',
          'completion_marker':'Separate final hash manifest binds final report, sources and verification logs'}
        if root:atomic_rows(root,'readiness_summary.json',summary,jsonl=False)
        print(json.dumps({k:summary[k] for k in ('C3_06_REFERENCE_CONTRACT_READY','regression','preservation','privacy')}),flush=True)
        return 0 if passed else 2
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error_code':'PREPARATION_OR_NEW_NAMESPACE_UNAVAILABLE',
            'actual_downloads':0,'actual_model_calls':0}),flush=True)
        return 2


if __name__=='__main__':sys.exit(main())
