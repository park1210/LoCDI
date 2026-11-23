"""C3-02 offline protocol preparation; exclusive new Public/Private namespaces."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.clinical_review_protocol import build,protocol_check
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha,verify as verify_freeze
from app.locdi.source_replay import atomic_rows
from app.locdi.research_protocol_audit import inspect_rc
from app.locdi.structural_release import preservation
from scripts.audit_c3_01 import privacy_check


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--save',action='store_true')
    parser.add_argument('--namespace',default='c3_02_v1');parser.add_argument('--regression',action='store_true')
    args=parser.parse_args()
    if args.regression and not args.save:parser.error('--regression requires a new saved namespace')
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C3_02_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    try:
        root=private=None
        if args.save:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.namespace):raise ValueError('INVALID_NAMESPACE')
            base=(ROOT/'outputs/research_evaluation').resolve();pbase=(ROOT/'outputs/private/research_evaluation').resolve()
            if not base.is_relative_to((ROOT/'outputs').resolve()) or not pbase.is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('OUTPUT_REDIRECTED')
            root=base/args.namespace;private=pbase/args.namespace
            if root.exists() or private.exists():raise FileExistsError('NAMESPACE_ALREADY_USED')
            root.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
        outputs,links,evidence=build();check=protocol_check(outputs)
        d7=verify_freeze(full=False);rc=inspect_rc()
        if root:
            for name,value in outputs.items():atomic_rows(root,name+'.json',value,jsonl=False)
            atomic_rows(private,'source_links.json',links,jsonl=False)
        regression={'status':'NOT_CHECKED'}
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            folder=root/'regression';folder.mkdir(exist_ok=False)
            regression=run_regression(folder);atomic_rows(folder,'result.json',regression,jsonl=False)
        kept=preservation(load(ROOT/'outputs/private/c3_02_baseline_hashes_v1.json'))
        document=next((ROOT/'docs').rglob('C3_02_CLINICAL_EVALUATION_SCOPE_AND_REFERENCE_REVIEW_PROTOCOL.md'),None)
        privacy=privacy_check(root,document) if root else {'status':'NOT_CHECKED_NOT_SAVED'}
        verified=(check['protocol_contract_status']=='PASS' and d7['status']==rc['status']==kept['status']=='PASS'
            and (not root or privacy['status']=='PASS') and (not args.regression or regression['status']=='PASS'))
        result={'schema_version':'locdi.c3_02.readiness.v1','C3_02_PROTOCOL_READY':'PASS_DEVELOPMENT_ONLY' if verified and root and args.regression else 'INCOMPLETE' if verified else 'FAIL',
            'C3_02_POLICY_APPROVED':False,'CLINICAL_REFERENCE_READY':False,'OFFICIAL_B0_B4_READY':False,
            'protocol_contract':check,'D7':d7,'RC':rc,'preservation':kept,'privacy':privacy,'regression':regression,
            'input_evidence':evidence,'private_links_sha256':sha(private/'source_links.json') if private else None,
            'clinical_verdicts_generated':0,'actual_generation_calls':0,'grading':0,'fitting':0,'voi_probability_estimation':0,
            'source_scan_scope':'Small frozen Original/D3 canonical and existing75 response links; no new large CSV/SQLite scan in preparer'}
        if root:
            atomic_rows(root,'readiness_summary.json',result,jsonl=False)
            files={p.relative_to(ROOT).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
            manifest={'schema_version':'locdi.c3_02.output_manifest.v1','files':files,'files_fingerprint':digest(files),
                'private_links_sha256':result['private_links_sha256'],'scope':'LOCAL_INTEGRITY_NOT_CLINICAL_APPROVAL'}
            manifest['fingerprint']=digest(manifest);atomic_rows(root,'output_manifest.json',manifest,jsonl=False)
        print(json.dumps({k:result[k] for k in ('C3_02_PROTOCOL_READY','C3_02_POLICY_APPROVED','CLINICAL_REFERENCE_READY','OFFICIAL_B0_B4_READY','regression','privacy')},ensure_ascii=True))
        sys.exit(0 if verified else 2)
    except Exception as exc:
        print(json.dumps({'C3_02_PROTOCOL_READY':'FAIL','error_type':type(exc).__name__,'error_code':'PROTOCOL_OR_NEW_NAMESPACE_UNAVAILABLE','actual_generation_calls':0}));sys.exit(2)
