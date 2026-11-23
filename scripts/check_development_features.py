"""Read-only feature checker. No execution or network permission flags."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.development_features import MODES, check_development_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='er_only')
    args = parser.parse_args(argv)
    try:
        report = check_development_features(args.mode)
    except Exception:
        print(json.dumps({'status': 'BLOCKED', 'error_type': 'STATIC_CHECK_ERROR', 'generation_requests': 0}))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if report['development_readiness'] == 'STATIC_CONTRACT_READY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
