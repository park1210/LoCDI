"""Offline fixed-baseline analysis. Optional new private blank review template."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.unresolved_analysis import load_analysis, save_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-analysis', metavar='NEW_ID')
    args = parser.parse_args(argv)
    try:
        public, private, review = load_analysis()
        if args.save_analysis:
            save_analysis(public, private, review, args.save_analysis)
        print(json.dumps(public, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except Exception:
        print(json.dumps({'status': 'ANALYSIS_BLOCKED', 'reason': 'BASELINE_MISSING_CHANGED_OR_INVALID', 'new_llm_calls': 0, 'utility_fit_count': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
