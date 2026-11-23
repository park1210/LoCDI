"""D6 read-only audits; --write adds new reports, never changes source files."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.dataset_release_readiness import validate_all,split_audit,dependency_inventory
from app.locdi.llm_setup import ROOT
from app.locdi.source_replay import atomic_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    validation=validate_all();split=split_audit();dependencies=dependency_inventory()
    if args.write:
        for name,value in [('D6_dataset_validation_v1.json',validation),('D6_split_and_exposure_v1.json',split),('D6_official_dependencies_v1.json',dependencies)]:
            atomic_rows(ROOT/'outputs/reports',name,value,jsonl=False)
    print(json.dumps({'original_cases':validation['original_locdi']['case_count'],'d3':validation['source_replay'],
        'healthbench':validation['healthbench'],'splits':split,'routing_gate':validation['routing_correctness_gate']},ensure_ascii=True))


if __name__=='__main__':main()
