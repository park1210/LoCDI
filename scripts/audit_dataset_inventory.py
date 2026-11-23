"""D1 offline inventory; optional new aggregate JSON/CSV, never source writes."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.dataset_inventory import audit_inventory, audit_saved_boundaries
from app.locdi.llm_setup import ROOT
from app.locdi.routing_collector import atomic_json, _id


def save_inventory(report, run_id):
    name=_id(run_id)
    root=ROOT/'outputs/reports'
    if ROOT.resolve() not in root.resolve().parents:raise ValueError('OUTPUT_REDIRECTED')
    path=root/(name+'.json');csv_path=root/(name+'_coverage.csv')
    if path.exists() or csv_path.exists():raise FileExistsError('AUDIT_EXISTS')
    rows=report['locdi']['coverage'];buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]) if rows else ['field'])
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in row.items()})
    # Exclusive creation: aggregate CSV only; original datasets never written.
    with csv_path.open('x',encoding='utf-8',newline='') as stream:stream.write(buffer.getvalue())
    atomic_json(path,report)
    return path,csv_path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--save',metavar='NEW_ID')
    parser.add_argument('--boundaries-only',action='store_true');args=parser.parse_args(argv)
    try:
        if args.boundaries_only:
            report=audit_saved_boundaries()
            if args.save:
                root=ROOT/'outputs/reports'
                if ROOT.resolve() not in root.resolve().parents:raise ValueError('OUTPUT_REDIRECTED')
                atomic_json(root/(_id(args.save)+'.json'),report)
            print(json.dumps(report,ensure_ascii=True));return 0
        report=audit_inventory()
        if args.save:save_inventory(report,args.save)
        print(json.dumps({'status':'READ_ONLY_AUDIT_COMPLETE','case_count':report['locdi']['case_count'],
            'inventory_entries':len(report['inventory']),'training_readiness':report['actual_routing_gate']['training_readiness'],
            'new_llm_generations':0,'utility_fit_count':0,'clinical_gold_generated':0}))
        return 0
    except Exception:
        print(json.dumps({'status':'AUDIT_BLOCKED','reason':'SOURCE_MISSING_OR_INVALID','new_llm_generations':0,'utility_fit_count':0}));return 2


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('External execution disabled')
    sys.addaudithook(guard)
    raise SystemExit(main())
