"""Controlled sequential live Pilot; default static preflight, no model calls."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.live_handoff import run_live_handoff,load_config,DEFAULT_CONFIG


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-llm',action='store_true');p.add_argument('--accept-release-assumption',action='store_true')
    p.add_argument('--expected-plan-fingerprint');a=p.parse_args(argv)
    if a.execute and not (a.allow_llm and a.accept_release_assumption and a.expected_plan_fingerprint):p.error('Execution requires all approval flags and a preflight fingerprint')
    try:r=run_live_handoff(run_id=a.run_id,config=load_config(a.config),execute=a.execute,allow_llm=a.allow_llm,
        accept_release_assumption=a.accept_release_assumption,expected_plan_fingerprint=a.expected_plan_fingerprint)
    except Exception:r={'status':'blocked','blocker_codes':['HANDOFF_INPUT_ERROR'],'request_count':0}
    print(json.dumps(r,ensure_ascii=True,allow_nan=False));return 0 if r['status'] in {'dry_run','completed'} and not r.get('blocker_codes') else 2


if __name__=='__main__':sys.exit(main())
