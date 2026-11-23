"""Print aggregate read-only reuse audit. No model, fitting or file writes."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.routing_data_reuse import audit_existing_data


def main():
    try:
        report = audit_existing_data()
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0 if report.get('status') == 'READ_ONLY_AUDIT' else 2
    except Exception:
        print(json.dumps({'status': 'BLOCKED', 'blocker_codes': ['SOURCE_AUDIT_INPUT_INVALID'],
                          'approved_training_group_count': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
