"""Single Case001 ER pilot; default read-only preflight, no server probes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.live_runtime import run_pilot
from app.locdi.llm_setup import DEFAULT_CONFIG, load_setup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True, help='Explicit new 32-character lowercase hex ID')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--mode', choices=('er_only', 'specialty', 'adaptive', 'adaptive_voi', 'research'), default='er_only')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--allow-llm', action='store_true')
    parser.add_argument('--accept-release-assumption', action='store_true')
    parser.add_argument('--expected-plan-fingerprint')
    args = parser.parse_args(argv)
    if args.execute and not (args.allow_llm and args.accept_release_assumption and args.expected_plan_fingerprint):
        parser.error('Live execution requires all approval flags and the dry-run plan fingerprint')
    try:
        report = run_pilot(run_id=args.run_id, config=load_setup(args.config), mode=args.mode,
            execute=args.execute, allow_llm=args.allow_llm,
            accept_release_assumption=args.accept_release_assumption,
            expected_plan_fingerprint=args.expected_plan_fingerprint)
    except Exception:
        report = {'status': 'blocked', 'blocker_codes': ['PILOT_INPUT_ERROR'], 'request_count': 0}
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0 if report['status'] in {'completed', 'dry_run'} and not report.get('blocker_codes') else 2


if __name__ == '__main__':
    sys.exit(main())
