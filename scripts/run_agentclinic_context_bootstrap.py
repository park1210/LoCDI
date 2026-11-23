"""One explicitly pinned nonclinical context probe; defaults to no execution."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.agentclinic_context_bootstrap import PinnedBootstrapApproval,run_bootstrap

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',required=True)
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--approval-sha256')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-llm',action='store_true')
    a=parser.parse_args()
    if not a.execute:
        print(json.dumps({'mode':'DRY_RUN','generation_calls':0,'status':'EXPLICIT_PINNED_BOOTSTRAP_AUTHORITY_REQUIRED'}));return 0
    try:
        authority=PinnedBootstrapApproval(a.approval,a.approval_sha256) if a.approval and a.approval_sha256 else None
        r=run_bootstrap(a.namespace,authority=authority,execute=a.execute,allow_llm=a.allow_llm)
        print(json.dumps(r,ensure_ascii=True));return 0 if r['status']=='BOOTSTRAP_ENVIRONMENT_PASS' else 2
    except Exception as exc:
        print(json.dumps({'status':'BOOTSTRAP_NOT_AUTHORIZED_OR_FAILED','error_type':type(exc).__name__,'automatic_retry':False}));return 2

if __name__=='__main__':sys.exit(main())
