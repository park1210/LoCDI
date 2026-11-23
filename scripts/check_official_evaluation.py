"""Read-only Phase 3.0 audit; optional exclusive new output, no execute mode."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.research_protocol_audit import audit
from app.locdi.episode_exporter import _write
from validation.review_submission_contract import submission_contract


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true')
    parser.add_argument('--audit-id',default='phase30_v1')
    args=parser.parse_args()
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('READ_ONLY_NO_NETWORK')
    sys.addaudithook(guard)
    root=None
    if args.save:
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.audit_id):parser.error('invalid audit ID')
        base=(ROOT/'outputs/research_protocol').resolve();root=base/args.audit_id
        if not base.is_relative_to((ROOT/'outputs').resolve()):parser.error('redirected output')
        root.mkdir(parents=True,exist_ok=False)
    result=audit()
    if root is not None:
        _write(result,'reports/readiness.json',root,'reports')
        _write(submission_contract(),'reports/review_submission_contract.json',root,'reports')
    print(json.dumps({'status':'BLOCKED','rc_integrity':result['rc']['status'],
        'blocker_count':len(result['blocker_plan']),'actual_generation_calls':0}))
