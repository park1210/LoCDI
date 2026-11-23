"""Offline Phase 2.11 actual readiness + optional TEST_ONLY multi-case evidence."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests/locdi'))
from app.locdi.llm_setup import digest
from app.locdi.dataset_loader import load_cases
from app.locdi.dataset_freeze import load,sha
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from app.locdi.development_features import read_artifact
from app.locdi.episode_exporter import _write
from app.locdi.integrated_approval import preflight_integrated
from validation.research_artifact_gate import assess_artifacts
from phase211_support import select_cases,execute_case,boundary_check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true');parser.add_argument('--multi-case',action='store_true')
    args=parser.parse_args()
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    definition=load(ROOT/'configs/development_specialty_llm.json')
    registry=load(ROOT/definition['llm']['registry_path'])
    path=definition['voi_model_path'];artifact=read_artifact(path) if path is not None else None
    actual=actual_gate_inputs()
    readiness=assess_artifacts(registry,routing_artifact=None,voi_artifact=artifact,correctness_inputs=actual,
        evaluation_patient_ids=[c['patient_id'] for c in cases],
        source_actions=[e for c in cases for e in c['environment']['current_encounter_events']],
        diagnosis_candidates=None,binding=definition['voi_binding'])
    selected=select_cases(cases)
    report={'schema_version':'locdi.phase211.audit.v1','artifact_readiness':readiness,
        'source_artifact_sha256':sha((ROOT/path).resolve()) if path else None,
        'configured_routing_utility':'MISSING_NULL_CONFIGURATION',
        'selection_rule':'BEFORE_OUTCOMES_FIRST_MEMORY_MIN_MAX_ENCOUNTER_MIN_MAX_DEDUP_FILL_MEMORY_DESC',
        'selected_cases':selected,'records':[], 'actual_generation_calls':0,'utility_fitting':0,
        'clinical_verdicts_generated':0,'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    plans=[preflight_integrated(run_id='202610300000000000000000000211%02d'%i,mode=mode,
        voi_artifact=artifact,correctness_inputs=actual) for i,mode in enumerate(('adaptive_routing','bayesian_voi','adaptive_and_voi'),1)]
    report['live_plans']=plans
    if args.save:
        private={'schema_version':'locdi.phase211.selection.v1','selection_rule':report['selection_rule'],
            'dataset_sha256':sha(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'),
            'selected':[{'case_ref':r['case_ref'],'patient_id':cases[r['ordinal']-1]['patient_id'],
                'visit_id':cases[r['ordinal']-1]['agent_visible']['index_encounter_id'],'source_fingerprint':r['source_fingerprint']} for r in selected],
            'model_outputs_seen_before_selection':False}
        _write(private,'private/selection_v1.json',ROOT/'outputs/phase211','private')
    if args.multi_case:
        for row in selected:
            case=cases[row['ordinal']-1]
            for scenario in ('A','B','C','D','E','F_TIMEOUT','G_BUDGET','H_INVALID_EVIDENCE'):
                result,client,record=execute_case(case,row,scenario,output_root=ROOT/'outputs/phase211' if args.save else None)
                record['source_boundary_pass']=boundary_check(case,client,result)
                report['records'].append(record)
    if args.save:_write(report,'reports/phase211_audit_v1.json',ROOT/'outputs/phase211','reports')
    print(json.dumps({'selected':[r['case_ref'] for r in selected],'scenario_count':len(report['records']),
        'readiness':{k:v['status'] for k,v in readiness['artifacts'].items()},'saved':args.save,'actual_generation_calls':0},ensure_ascii=True))


if __name__=='__main__':
    def guard(event,args):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:raise RuntimeError('OFFLINE_ONLY')
    sys.addaudithook(guard);main()
