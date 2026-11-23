"""C4-13 frozen two-Decision collection. Default DRY_RUN; no implicit approval."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.routing_two_decision_collection import collect, assess_run
from app.locdi.dataset_freeze import load, sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='c4_13_v1')
    parser.add_argument('--run-id', default='c4_13_collect_001')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-llm', action='store_true')
    parser.add_argument('--accept-proxy-for-generation', action='store_true')
    parser.add_argument('--acknowledge-timeout-observations', action='store_true')
    parser.add_argument('--expected-plan-fingerprint')
    parser.add_argument('--max-requests', type=int, default=10)
    parser.add_argument('--check-fit-readiness', action='store_true')
    parser.add_argument('--review-package')
    parser.add_argument('--review-package-sha256')
    args = parser.parse_args(argv)
    try:
        if args.check_fit_readiness:
            if args.execute or args.allow_llm: raise ValueError('READ_ONLY_REVIEW_COMMAND')
            package = None
            if args.review_package:
                if sha(args.review_package) != args.review_package_sha256: raise ValueError('REVIEW_PACKAGE_HASH_MISMATCH')
                package = load(args.review_package)
            result = assess_run(args.namespace, args.run_id, package)
        else:
            if args.review_package or args.review_package_sha256: raise ValueError('REVIEW_OPTIONS_REQUIRE_READINESS_MODE')
            result = collect(args.namespace, args.run_id, execute=args.execute, allow_llm=args.allow_llm,
                accept_proxy=args.accept_proxy_for_generation, acknowledge_timeout=args.acknowledge_timeout_observations,
                expected_plan_fingerprint=args.expected_plan_fingerprint, max_requests=args.max_requests)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
        return 0
    except Exception as exc:
        # No path, patient ID, SDK body or clinical evidence in public errors.
        print(json.dumps({'status': 'BLOCKED_OR_FAILED', 'error_type': type(exc).__name__,
            'details': 'See private checkpoints if execution actually started', 'fitting_calls': 0}))
        return 2


if __name__ == '__main__': raise SystemExit(main())
