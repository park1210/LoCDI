"""Dry-run by default. Real generation requires all three explicit approvals."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.routing_collector import load_preparation, collect


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', required=True, help='Private preparation ID, not arbitrary path')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-llm', action='store_true')
    parser.add_argument('--accept-development-assumptions', action='store_true',
                        help='Accept pinned visibility/Memory proxies, development catalog, deferred reference review for collection only')
    parser.add_argument('--run-id')
    parser.add_argument('--max-requests', type=int, default=75)
    args = parser.parse_args(argv)
    try:
        if args.execute and (not args.allow_llm or not args.accept_development_assumptions or not args.run_id):
            raise ValueError('MISSING_EXECUTION_APPROVAL')
        report = collect(load_preparation(args.preparation), execute=args.execute, allow_llm=args.allow_llm,
            accept_development_assumptions=args.accept_development_assumptions, run_id=args.run_id, max_requests=args.max_requests)
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except Exception:
        print(json.dumps({'status': 'BLOCKED', 'blocker_codes': ['COLLECTION_BLOCKED_OR_STOPPED'],
                          'generation_count': 'NOT_RECORDED_BY_CLI_ON_EXCEPTION', 'utility_fit_count': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
