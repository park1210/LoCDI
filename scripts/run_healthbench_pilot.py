"""D5 dialogue-only pilot; dry-run by default, grading is not enabled."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.external_dialogue import prepare_pilot,preflight,run_pilot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true',help='Record three private IDs before any model responses')
    parser.add_argument('--run-id');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--allow-llm',action='store_true');parser.add_argument('--expected-plan-fingerprint')
    args=parser.parse_args(argv)
    if args.prepare and (args.execute or args.allow_llm):parser.error('Preparation and model execution must be separate')
    if args.execute and not (args.allow_llm and args.expected_plan_fingerprint):parser.error('Execution requires explicit model approval and plan fingerprint')
    if not args.prepare and not args.run_id:parser.error('--run-id required for preflight/execution')
    try:
        if args.prepare:
            selection=prepare_pilot()
            result={'status':'PREPARED','selection_fingerprint':selection['selection_fingerprint'],'development_samples':3,'model_calls':0,'grader_calls':0}
        else:result=run_pilot(args.run_id,execute=args.execute,allow_llm=args.allow_llm,expected_plan_fingerprint=args.expected_plan_fingerprint)
    except Exception:result={'status':'BLOCKED','blocker_codes':['PREPARATION_OR_SOURCE_ERROR'],'model_calls':0,'grader_calls':0}
    print(json.dumps(result,ensure_ascii=True))
    return 0 if result['status'] in {'PREPARED','DRY_RUN','COMPLETED'} and not result.get('blocker_codes') else 2


if __name__=='__main__':sys.exit(main())
