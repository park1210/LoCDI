"""Default DRY_RUN; one existing train Decision, at most five real generations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.routing_collector import load_preparation
from app.locdi.routing_utility_pilot import collect_pilot

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace',default='c4_11_v1');p.add_argument('--run-id',default='c4_11_collect_001')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true')
    p.add_argument('--accept-proxy',action='store_true');p.add_argument('--max-requests',type=int,default=5)
    a=p.parse_args(argv)
    result=collect_pilot(load_preparation('phase24b_candidates_v2'),namespace=a.namespace,run_id=a.run_id,
        execute=a.execute,allow_llm=a.allow_llm,accept_proxy=a.accept_proxy,max_requests=a.max_requests)
    print(json.dumps(result,ensure_ascii=True));return 0

if __name__=='__main__':raise SystemExit(main())
