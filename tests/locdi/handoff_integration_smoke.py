"""TEST_ONLY real first case; A-H runner diagnostics, I/J in contract tests."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT
from app.locdi.handoff_integration import run_handoff_integrated
from app.locdi.episode_exporter import _write
from app.core.longitudinal_runner import run_with_recovery
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.voi_planning import plan_voi
from app.core.diagnostic_belief import posterior
from handoff_integration_support import integrated_options


def execute(output_root=None):
    case=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0];rows=[]
    definitions=[('A',False,False,None,False,10),('B',True,False,None,False,10),
        ('C',False,True,None,False,10),('D',True,True,None,False,10),('E_ABSTAIN',True,False,None,True,10),
        ('F_MISSING_VOI',False,True,None,False,10),('G_TIMEOUT',False,False,'timeout',False,10),
        ('G_REPAIR',False,False,'semantic_repair',False,10),('G_REPAIR_FAILED',False,False,'semantic_failure',False,10),
        ('H_BUDGET',False,False,'semantic_repair',False,2)]
    for name,routing,voi,mode,abstain,budget in definitions:
        options=integrated_options(case,routing=routing,voi=voi,abstain=abstain,mode=mode)
        if name=='F_MISSING_VOI':options=replace(options,voi_model=None)
        with patch('app.locdi.run_simulation.run_with_recovery',wraps=run_with_recovery) as runner,patch(
                'app.core.longitudinal_runner.HospitalParallelScheduler',wraps=HospitalParallelScheduler) as scheduler,patch(
                'app.core.information_value.plan_voi',wraps=plan_voi) as planner,patch(
                'app.core.information_outcomes.posterior',wraps=posterior) as bayes:
            r=run_handoff_integrated(case,options=options,test_only=True,routing='adaptive_test_only' if routing else 'static_development',
                budget=budget,output_root=output_root,export_id='TEST_ONLY_'+name)
        rows.append({'scenario':name,'status':r['status'],'blocker_codes':r['blocker_codes'],'export_status':r['export_status'],
            'integration_contract_ready':r['integration_contract_ready'],'handoff_status':r['audit']['handoff_status'] if r['audit'] else None,
            'offline_dispatch_count':r['audit']['offline_dispatch_count'] if r['audit'] else 0,
            'core_action_decision_count':r['audit']['core_action_decision_count'] if r['audit'] else 0,
            'repair_count':r['audit']['specialty_semantic_repair_count'] if r['audit'] else 0,
            'result':r['episode']['result'] if r['episode'] else None,
            'voi_trace':r['audit']['voi'] if r['audit'] else [],
            'actual_function_calls':{'runner':runner.call_count,'scheduler':scheduler.call_count,'plan_voi':planner.call_count,'posterior':bayes.call_count}})
    report={'schema_version':'locdi.phase210.offline_smoke.v1','test_only':True,'scenarios':rows,
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False,'new_model_generations':0,'utility_fitting':0}
    if output_root is not None:_write(report,'reports/phase210_smoke_v1.json',output_root,'reports')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root');args=p.parse_args()
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('TEST_ONLY_NO_EXTERNAL_CALLS')
    sys.addaudithook(guard)
    r=execute(args.output_root);print(json.dumps(r,ensure_ascii=True))
    expected={'A':'completed','B':'completed','C':'completed','D':'completed','E_ABSTAIN':'completed',
              'F_MISSING_VOI':'blocked','G_TIMEOUT':'failed','G_REPAIR':'completed','G_REPAIR_FAILED':'failed','H_BUDGET':'failed'}
    sys.exit(0 if all(row['status']==expected[row['scenario']] for row in r['scenarios']) else 2)
