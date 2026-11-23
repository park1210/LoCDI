"""Read-only single-run analysis. Optional NEW review packet, never resume."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.specialty_collection_analysis import analyze_run, save_review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='phase24b_collect_001')
    parser.add_argument('--preparation', default='phase24b_candidates_v2')
    parser.add_argument('--save-review', metavar='NEW_ID')
    args = parser.parse_args(argv)
    try:
        summary, private = analyze_run(args.run_id, args.preparation)
        if args.save_review:
            save_review(summary, private, args.save_review)
        print(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False))
        return 0
    except Exception:
        print(json.dumps({'status': 'ANALYSIS_BLOCKED', 'reason': 'SOURCE_MISSING_AMBIGUOUS_OR_INVALID',
                          'new_llm_calls': 0, 'utility_fit_count': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
