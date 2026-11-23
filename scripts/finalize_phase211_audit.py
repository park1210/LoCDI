"""Current-code readiness/approval snapshots; never replay or modify results."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_loader import load_cases
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.development_features import read_artifact
from app.locdi.evaluation_gate_audit import actual_gate_inputs
from app.locdi.integrated_approval import preflight_integrated
from app.locdi.episode_exporter import _write
from validation.research_artifact_gate import assess_artifacts


if __name__=='__main__':
    source=ROOT/'outputs/phase211/reports/phase211_audit_v1.json'
    old=load(source);cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    config=load(ROOT/'configs/development_specialty_llm.json')
    registry=load(ROOT/config['llm']['registry_path']);a=read_artifact(config['voi_model_path']);inputs=actual_gate_inputs()
    options={'voi_artifact':a,'correctness_inputs':inputs,
        'evaluation_patient_ids':[c['patient_id'] for c in cases],
        'source_actions':[e for c in cases for e in c['environment']['current_encounter_events']],
        'diagnosis_candidates':None}
    gate=assess_artifacts(registry,**options)
    plans=[preflight_integrated(run_id='202610300000000000000000000211%02d'%i,mode=mode,**options)
        for i,mode in enumerate(('adaptive_routing','bayesian_voi','adaptive_and_voi'),1)]
    report={'schema_version':'locdi.phase211.current_snapshot.v2','prior_snapshot_sha256':sha(source),
        'selection_manifest_sha256':sha(ROOT/'outputs/phase211/private/selection_v1.json'),
        'artifact_readiness':gate,'live_plans':plans,
        'scenario_results_reused_read_only':True,'scenario_count':len(old['records']),
        'scenario_status_counts':{s:sum(r['status']==s for r in old['records']) for s in ('completed','incomplete','failed','blocked')},
        'all_source_boundary_checks_pass':all(r['source_boundary_pass'] for r in old['records']),
        'all_exports':all(r['export_status']=='EXPORTED' for r in old['records']),
        'source_artifact_sha256':sha((ROOT/config['voi_model_path']).resolve()),
        'binding_declaration_format':{'storage':'EXTERNAL_SIDECAR_NOT_CORE_ARTIFACT_MUTATION',
            'required':['artifact_fingerprint','research_binding','artifact_version','provenance'],
            'research_binding':['registry_fingerprint','model_identifier','prompt_fingerprint','action_catalog_fingerprint','memory_fingerprint'],
            'clinical_evidence_authentication':'NOT_IMPLEMENTED_FAIL_CLOSED'},
        'alignment':{'paper_source_status':'NOT_FOUND_IN_WORKSPACE',
            'current_encounter_order':'RELEASED_SOURCE_TO_ER_PROPOSAL_TO_BELIEF_RECOMPUTATION_TO_VOI_TO_VALIDATION_TO_SCHEDULER',
            'belief_exposed_to_er_llm_before_proposal':False,'pre_decision_hook_enabled':False,
            'hook_requirement_from_paper':'NOT_CHECKED_PAPER_NOT_SUPPLIED',
            'bayes_update_semantics':'FROM_FIXED_PRIOR_REPLAY_ONCE_PER_MATCHED_ACTION_NOT_RECURSIVE_MULTIPLICATION'},
        'actual_generation_calls':0,'utility_fitting':0,'clinical_verdicts_generated':0,'research_evaluation_ready':False}
    _write(report,'reports/phase211_current_snapshot_v2.json',ROOT/'outputs/phase211','reports')
    print(json.dumps({k:report[k] for k in ('scenario_count','scenario_status_counts','all_source_boundary_checks_pass','all_exports')},ensure_ascii=True))
