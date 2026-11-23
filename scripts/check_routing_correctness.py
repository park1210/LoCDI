"""Offline real-baseline gate. No model, fitting or review generation options."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.evaluation_gate_audit import audit_actual_gate
from app.locdi.llm_setup import ROOT
from app.locdi.routing_collector import _id, atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-report', metavar='NEW_ID')
    args = parser.parse_args(argv)
    try:
        report = audit_actual_gate()
        if args.save_report:
            root = ROOT/'outputs/reports'
            if ROOT.resolve() not in root.resolve().parents:
                raise ValueError('OUTPUT_REDIRECTED')
            atomic_json(root/(_id(args.save_report)+'_gate.json'), report)
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0 if report['utility_fit_ready'] else 2
    except Exception:
        print(json.dumps({'status':'AUDIT_INPUT_BLOCKED','reason':'BASELINE_MISSING_CHANGED_OR_INVALID','new_llm_calls':0,'utility_fit_count':0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
