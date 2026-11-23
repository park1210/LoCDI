"""Read-only reference feasibility audit. No generation/download/fitting options."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.reference_feasibility import build
from app.locdi.dataset_freeze import load,sha,verify as verify_freeze
from app.locdi.llm_setup import digest
from app.locdi.source_replay import atomic_rows
from app.locdi.research_protocol_audit import inspect_rc
from app.locdi.structural_release import preservation
from scripts.audit_c3_01 import privacy_check


def findings(outputs):
    a=outputs['unresolved_evidence_audit'];m=outputs['specialty_uncertainty_matrix']
    checks={
      'pinned_75':(a['response_count'],a['supported'],a['unresolved'],a['named_unresolved'],a['literal_unresolved'])==(75,22,53,52,1),
      'bindings_schema':a['binding_checks'].get('binding_match')==75 and a['schema_invalid']==a['failed']==0,
      'same_snapshot':all(g['same_common_input'] for g in a['groups']),
      'all_causes_undetermined':all(h['final_status']=='NOT_DETERMINED' for h in outputs['root_cause_hypotheses']['hypotheses']),
      'literal_coverage_not_gold':a['source_label_literal_coverage']==15 and a['independently_reviewed_correctness']==0,
      'existing_fit_blocked':a['gate_accepted_fit_groups']==0 and not a['existing_correctness_gate']['utility_fit_ready'],
      'combined_validation_blocked':m['split_independence']['routing_validation_voi_train_overlap']==5,
      'no_reference_approval':all(not r['project_use_approved'] and not r['downloaded_this_phase'] for r in outputs['reference_source_inventory']['sources']),
      'mcqa_test_blocked':outputs['automatic_mapping_risks']['MedMCQA_test_label_contract']['status']=='BLOCKED',
      'budget_design_only':outputs['e1_e5_experiment_plan']['illustrative_E2_E5_generation_total']==405,
      'no_policy_approval':outputs['collection_and_experiment_options']['routing_policy_approval_register']['APPROVED_POLICY'] is None}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save',action='store_true');parser.add_argument('--namespace',default='c3_05_v1')
    parser.add_argument('--regression',action='store_true');args=parser.parse_args(argv)
    if args.regression and not args.save:parser.error('--regression requires --save')
    def guard(event,arguments):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C3_05_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    try:
        root=private=None
        if args.save:
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.namespace):raise ValueError('INVALID_NAMESPACE')
            root=ROOT/'outputs/research_evaluation'/args.namespace
            private=ROOT/'outputs/private/research_evaluation'/args.namespace
            if (not root.resolve().is_relative_to((ROOT/'outputs/research_evaluation').resolve()) or
                not private.resolve().is_relative_to((ROOT/'outputs/private').resolve())):raise ValueError('OUTPUT_REDIRECTED')
            if root.exists() or private.exists():raise FileExistsError('NAMESPACE_ALREADY_USED')
            root.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
        outputs,links,evidence=build();check=findings(outputs)
        if root:
            for name,value in outputs.items():atomic_rows(root,name+'.json',value,jsonl=False)
            atomic_rows(private,'source_links.json',links,jsonl=False)
        print(json.dumps({'analysis_contract':check,'stage':'ANALYSIS_SAVED' if root else 'READ_ONLY'}),flush=True)
        regression={'status':'NOT_CHECKED'}
        if args.regression:
            from scripts.verify_simulator_release import run_regression
            folder=root/'regression';folder.mkdir(exist_ok=False)
            regression=run_regression(folder);atomic_rows(folder,'result.json',regression,jsonl=False)
        d7,rc=verify_freeze(full=False),inspect_rc()
        kept=preservation(load(ROOT/'outputs/private/c3_05_baseline_hashes_v1.json'))
        document=next((ROOT/'docs').rglob('C3_05_REFERENCE_ACQUISITION_AND_UNRESOLVED_AUDIT_REPORT.md'),None)
        privacy=privacy_check(root,document) if root else {'status':'NOT_CHECKED_NOT_SAVED'}
        passed=all(r['status']=='PASS' for r in (check,d7,rc,kept))
        if root:passed=passed and privacy['status']=='PASS'
        if args.regression:passed=passed and regression['status']=='PASS'
        ready=passed and root is not None and args.regression
        summary={'schema_version':'locdi.c3_05.readiness.v1',
          'C3_05_REFERENCE_AUDIT':'PASS_DEVELOPMENT_ONLY' if ready else 'INCOMPLETE' if passed else 'FAIL',
          'ROUTING_POLICY_APPROVED':False,'ROUTING_ARTIFACT_READY':False,'CLINICAL_REFERENCE_READY':False,
          'PRODUCTION_AUTHENTICATION_READY':False,'OFFICIAL_B0_B4_READY':False,
          'findings':check,'regression':regression,'D7':d7,'RC':rc,'preservation':kept,'privacy':privacy,
          'input_evidence':evidence,'private_index_sha256':sha(private/'source_links.json') if private else None,
          'actual_generation_calls':0,'healthbench_grading':0,'utility_fitting':0,'voi_probability_estimation':0,
          'clinical_verdicts_generated':0,'new_dataset_downloads':0,
          'test_only_scope':'Regression may exercise existing in-memory fixture fitting; no real artifact fitting or new clinical labels',
          'source_scan_scope':'Pinned bank229/source links/actual75/prior manifests and inventories; no large raw/SQLite rescan',
          'completion_scope':'Final completion manifest separately binds report, new source, private index and verification log'}
        if root:atomic_rows(root,'readiness_summary.json',summary,jsonl=False)
        print(json.dumps({k:summary[k] for k in ('C3_05_REFERENCE_AUDIT','regression','privacy','preservation')}),flush=True)
        return 0 if passed else 2
    except Exception as exc:
        print(json.dumps({'C3_05_REFERENCE_AUDIT':'FAIL','error_type':type(exc).__name__,
          'error_code':'AUDIT_OR_NEW_NAMESPACE_UNAVAILABLE','actual_generation_calls':0}),flush=True)
        return 2


if __name__=='__main__':sys.exit(main())
