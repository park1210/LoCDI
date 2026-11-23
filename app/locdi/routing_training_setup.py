"""Read-only source inventory and routing training preflight; no execution."""
import hashlib
import json
import os

from app.core.agent_registry import AgentRegistry
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import ROOT
from validation.routing_training_data import validate_training_data, validate_split_manifest

ARCHIVE_EXPERIMENTS = ROOT.parent / '001_synthea-docker/experiments'
HOLDOUT = ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl'
REGISTRY = ROOT / 'configs/agent_registry.specialty_development.json'
HOSPITAL = ROOT.parents[1] / '003_SoftwareCapstonDesign/006_ClinicaTextSim_HospitalWorkflow'


def _read(path):
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    return json.loads(path.read_text(encoding='utf-8-sig'))


def inspect_sources():
    """Bounded inventory, not exhaustive clinical corpus/evaluation validation.

    Labels are counted privately, never promoted to Gold or model correctness.
    Archived probes/banks have incompatible purposes; no auto-import is done.
    """
    rows = load_cases(HOLDOUT)
    result = {'status': 'PARTIALLY_AVAILABLE', 'holdout_case_count': len(rows),
              'holdout_patient_count': len({r['patient_id'] for r in rows}),
              'holdout_metric_flags_ready_count': sum(any(r['task_annotations']['metric_readiness'].values()) for r in rows),
              'holdout_non_null_gold_field_count': sum(v is not None for r in rows for v in r['task_annotations']['gold'].values()),
              'eligible_reviewed_specialty_group_status': 'MISSING', 'inspected_archives': []}
    patterns = ('29_*/output/*/routing_probes.json', '30_*/output/*/routing_probes.json',
                '34_*/output/*/training_bank.jsonl', '34_*/output/*/evaluation_labels.json')
    for pattern in patterns:
        for path in sorted(ARCHIVE_EXPERIMENTS.glob(pattern)):
            try:
                data = _read(path)
                item = {'source': path.relative_to(ARCHIVE_EXPERIMENTS).as_posix(), 'record_count': len(data),
                        'imported_for_training': False}
                if isinstance(data, list):
                    item.update(agent_id_record_count=sum(isinstance(x, dict) and 'agent_id' in x for x in data),
                                boolean_correct_record_count=sum(isinstance(x, dict) and type(x.get('correct')) is bool for x in data))
                result['inspected_archives'].append(item)
            except (OSError, ValueError, TypeError):
                result['inspected_archives'].append({'source': path.relative_to(ARCHIVE_EXPERIMENTS).as_posix(), 'status': 'UNREADABLE_NOT_IMPORTED'})
    # Follow the concrete historical run referenced by Synthea audit_failure.py.
    old_path = HOSPITAL / 'outputs/action_replay_pipeline_20261024_004817_4f9741/routing_train.json'
    result['historical_collection'] = {'status': 'MISSING', 'imported_for_training': False}
    if old_path.is_file():
        old = _read(old_path)
        current = _read(REGISTRY)
        bindings = {a['agent_id']: (a['model'], a['version']) for a in current['agents'] if a['role'] == 'specialty'}
        patients = {r['patient_id'] for r in old}
        groups = {(r['patient_id'], r['decision_id']) for r in old}
        result['historical_collection'].update(status='PRESENT_NOT_ELIGIBLE', record_count=len(old),
            patient_count=len(patients), decision_group_count=len(groups),
            boolean_correct_record_count=sum(type(r.get('correct')) is bool for r in old),
            review_metadata_record_count=sum(any('review' in k for k in r) for r in old),
            holdout_overlap_count=len(patients & {r['patient_id'] for r in rows}),
            current_registry_binding_match_count=sum(bindings.get(r['agent_id']) == (r.get('model'), r.get('version')) for r in old))
    result['historical_utilities'] = []
    for path in sorted((HOSPITAL / 'app/data').glob('*/routing_utility*.json')):
        artifact = _read(path)
        result['historical_utilities'].append({'source': path.relative_to(HOSPITAL).as_posix(),
            'schema_version': artifact.get('schema_version'), 'sample_count': len(artifact.get('samples', [])),
            'current_registry_snapshot_matches': artifact.get('registry') == _read(REGISTRY), 'imported_for_training': False})
    count = 0
    for root in (ROOT, ARCHIVE_EXPERIMENTS, HOSPITAL):
        for directory, folders, names in os.walk(root):
            folders[:] = [f for f in folders if f not in {'.venv', 'venv', '.git', '__pycache__', 'node_modules'}]
            count += sum('medqa' in n.casefold() for n in names)
            count += sum('medqa' in n.casefold() for n in folders)
    result['medqa_named_file_or_directory_count'] = count
    result['medqa_content_review'] = 'NOT_REVIEWED' if count else 'NOT_FOUND_IN_SCANNED_ROOTS'
    medqa = HOSPITAL / 'data/raw/_medqa.jsonl'
    result['medqa_candidate'] = {'status': 'MISSING', 'imported_for_training': False}
    if medqa.is_file():
        records = _read(medqa)
        result['medqa_candidate'].update(status='PRESENT_NOT_ELIGIBLE', record_count=len(records),
            declared_reference_label_count=sum(isinstance(r.get('OSCE_Examination'), dict)
                and bool(r['OSCE_Examination'].get('Correct_Diagnosis')) for r in records),
            patient_identifier_record_count=sum('patient_id' in r for r in records),
            agent_id_record_count=sum('agent_id' in r for r in records),
            boolean_correct_record_count=sum(type(r.get('correct')) is bool for r in records),
            clinical_reference_validity='NOT_REVIEWED')
    result['scan_scope'] = 'current_project_adjacent_synthea_experiments_and_referenced_hospital_named_sources_not_all_disk'
    return result


def training_preflight(*, collection=None, split_manifest=None, source_catalog=None, label_catalog=None):
    """Sanitized report only. Even valid data is never converted/fitted by CLI."""
    rows = load_cases(HOLDOUT)
    holdout = sorted({r['patient_id'] for r in rows})
    fingerprint = hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
    manifest = _read(REGISTRY)
    # Keep independently supplied split authoritative; never infer assignments.
    if collection is not None and split_manifest is not None and collection.get('split_manifest') != split_manifest:
        raise ValueError('CONFLICTING_SPLIT_MANIFEST')
    report = validate_training_data(collection, registry_manifest=manifest, source_catalog=source_catalog,
        label_catalog=label_catalog, holdout_patient_ids=holdout, dataset_fingerprint=fingerprint)
    report.update(schema_version='locdi.routing_training_preflight.v1', status='dry_run',
                  clinical_metrics='NOT_EVALUATED', inventory=inspect_sources())
    if collection is None and split_manifest is not None:
        split = validate_split_manifest(split_manifest, holdout_patient_ids=holdout, dataset_fingerprint=fingerprint)
        report['split_status'] = split['status']
        report.update({k: split[k] for k in ('train_patient_count', 'evaluation_patient_count')})
        report['blocker_codes'] = sorted(set([c for c in report['blocker_codes'] if c != 'SPLIT_MANIFEST_MISSING'] + split['blocker_codes']))
    registry = AgentRegistry(manifest)
    report['specialty_agent_count'] = sum(a['role'] == 'specialty' for a in registry.snapshot()['agents'])
    return report
