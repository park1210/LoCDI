"""Inspect routing data readiness only. No collection, LLM, fitting or writes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import project_file
from app.locdi.routing_training_setup import training_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-data', help='Private project-relative collection JSON')
    parser.add_argument('--split-manifest', help='Explicit private project-relative split JSON')
    parser.add_argument('--source-catalog', help='Separate reviewed private source catalog JSON array')
    parser.add_argument('--label-catalog', help='Separate reviewed private label catalog JSON array')
    args = parser.parse_args(argv)
    def read(value):
        return json.loads(project_file(value).read_text(encoding='utf-8-sig')) if value else None
    try:
        report = training_preflight(collection=read(args.collection_data), split_manifest=read(args.split_manifest),
                                    source_catalog=read(args.source_catalog), label_catalog=read(args.label_catalog))
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        return 0 if report['training_readiness'] == 'FIT_INPUT_CONTRACT_READY' else 2
    except Exception:
        print(json.dumps({'status': 'BLOCKED', 'training_readiness': 'BLOCKED', 'blocker_codes': ['INPUT_OR_INVENTORY_INVALID'],
                          'generation_requests': 0, 'utility_fit_count': 0, 'research_readiness': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
