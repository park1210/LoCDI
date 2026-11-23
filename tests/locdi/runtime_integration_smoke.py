"""TEST_ONLY executable A/B/C/D harness, real first LoCDI case, no real LLM."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT
from app.locdi.integrated_runtime import run_integrated
from app.locdi.episode_exporter import _write
from runtime_integration_support import options_for


def execute(output_root=None):
    case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
    rows=[]
    for name,routing,voi in [('A',False,False),('B',True,False),('C',False,True),('D',True,True)]:
        r=run_integrated(case,options=options_for(case,routing=routing,voi=voi),test_only=True,
            output_root=output_root,export_id='TEST_ONLY_'+name)
        e=r['episode']
        rows.append({'scenario':name,'status':r['status'],'readiness':r['readiness'],'blocker_codes':r['blocker_codes'],
            'episode_id':e['run_id'] if e else None,'result':e['result'] if e else None,
            'configuration_fingerprint':e['configuration_fingerprint'] if e else None})
    report={'schema_version':'locdi.runtime_smoke.v1','test_only':True,'cases':rows,
        'clinical_metrics':'NOT_EVALUATED','new_llm_generation_count':0,'utility_fit_count':0}
    if output_root is not None:
        _write(report,'reports/TEST_ONLY_smoke.json',output_root,'reports')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root')
    args=parser.parse_args()
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:
            raise RuntimeError('External execution disabled')
    sys.addaudithook(guard)
    report=execute(args.output_root)
    print(json.dumps(report,ensure_ascii=True))
    sys.exit(0 if all(r['readiness']['runtime_integration_ready'] for r in report['cases']) else 2)
