"""Persist TEST_ONLY A-F evidence, using the actual Core and existing source case."""
import json
import argparse
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.locdi.llm_setup import ROOT
from app.locdi.dataset_loader import load_cases
from app.locdi.sequential_handoff import run_sequential_handoff
from app.locdi.episode_exporter import _write
from test_simulation import smoke_options
from test_sequential_handoff import HandoffClient,registry_fixture
from test_routing_voi_readiness import test_only_routing


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',default='phase28_test_only_v1');args=parser.parse_args()
    if not re.fullmatch(r'phase28_test_only_v[1-9][0-9]*',args.run_id):raise ValueError('INVALID_TEST_ONLY_RUN_ID')
    root=ROOT/'outputs/sequential_handoff'/args.run_id
    if root.exists():raise FileExistsError('NO_OUTPUT_REUSE')
    root.mkdir(parents=True,exist_ok=False)
    case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
    records=[]
    scenarios=[('A_normal',None,10,'static_development'),('B_timeout','timeout',10,'static_development'),
        ('B_schema_invalid','schema_invalid',10,'static_development'),('C_incomplete_response','unfinished',10,'static_development'),
        ('D_budget',None,2,'static_development'),('E_abstention',None,10,'adaptive'),
        ('F_repair','repair',11,'static_development'),('A_repeat',None,10,'static_development')]
    for name,mode,budget,routing in scenarios:
        registry=registry_fixture();client=HandoffClient(mode);utility=None
        if routing=='adaptive':utility=test_only_routing(registry);utility['min_samples']=999
        result=run_sequential_handoff(case,options=smoke_options(client,registry_manifest=registry),test_only=True,
            budget=budget,routing=routing,static_department='neurology' if routing=='static_development' else None,
            routing_utility=utility,output_root=root,export_id=name)
        records.append({'scenario':name,'status':result['status'],'blocker_codes':result['blocker_codes'],
            'request_count':result['audit']['request_count'],'result':result['episode']['result'],
            'trace_fingerprint':result['episode']['normalized_trace_fingerprint'],
            'handoff_status':result['audit']['handoffs'][0]['handoff_status'],
            'export_status':result['audit']['export_status']})
    report={'schema_version':'locdi.phase28_smoke.v1','test_only':True,'cases':records,
        'normal_repeat_trace_equal':records[0]['trace_fingerprint']==records[-1]['trace_fingerprint'],
        'new_llm_generations':0,'utility_fit_count':0,'clinical_verdict_count':0,
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    _write(report,'reports/smoke.json',root,'reports');print(json.dumps(report,ensure_ascii=True))
    assert report['normal_repeat_trace_equal'] and all(r['export_status']=='EXPORTED' for r in records)


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('NO_EXTERNAL_CALLS')
    sys.addaudithook(guard);main()
