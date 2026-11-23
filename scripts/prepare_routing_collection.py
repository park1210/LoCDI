"""Prepare candidates only. Default stdout audit; explicit save is private."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.routing_collection_preparation import build_preparation, preparation_audit
from app.locdi.routing_collector import save_preparation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-preparation', metavar='ID', help='Exclusive private preparation directory ID')
    args = parser.parse_args(argv)
    try:
        bundle, references = build_preparation()
        if args.save_preparation:
            save_preparation(bundle, references, args.save_preparation)
        report = preparation_audit(bundle)
        report['private_preparation_saved'] = args.save_preparation is not None
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except Exception:
        print(json.dumps({'status': 'BLOCKED', 'blocker_codes': ['PREPARATION_FAILED'], 'generation_requests': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
