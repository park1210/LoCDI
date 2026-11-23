"""C4-33 prepare/freeze once or inspect an approved generated export. No generation."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, rows, sha, write, check
from scripts.evaluate_dynamic_horizon_source_policy import merge_pins
from app.locdi.llm_setup import digest
from app.locdi.reference_readiness import COHORT
from app.locdi.routing_data_reuse import BANK, SOURCE, DATABASE
from app.locdi.source_replay import PRIVATE as D3
from app.locdi.voi_source_action_inventory import database_sources
from app.locdi.voi_new_cohort_source import load_csv, identities, encounter_fingerprints, overlap, adapt
from app.locdi.voi_new_cohort_evaluation import training_protocol, fit_final, evaluation_protocol

NS = 'c4_33_v1'
PUB = ROOT / 'outputs/research_evaluation' / NS
PR = ROOT / 'outputs/private/research_evaluation' / NS
PARENT = ROOT / 'outputs/private/research_evaluation/c4_32_v1'
PPUB = ROOT / 'outputs/research_evaluation/c4_32_v1'
DEV = ROOT / 'outputs/private/research_evaluation/c4_30_v2'
SYN = ROOT.parent / '001_synthea-docker'
AUDIT = SYN / 'experiments/11_docker_symptom_audit/output/details'
IMAGE = 'sha256:8fedabd7421b470291f3decbdda24b8242d8764cc500ea7a30ace96099c3482c'
JAR = '074c2ed01f7ad58fe6792926a1101d25b1c11a0c8c85c4fb642db04dda0ff978'
COMMIT = '29a7d63ce80e9ed2ff83a6060696035538612ccf'
CODE = ['app/locdi/voi_new_cohort_source.py', 'app/locdi/voi_new_cohort_evaluation.py',
        'scripts/prepare_new_voi_cohort.py', 'tests/locdi/test_voi_new_cohort.py']


def observe(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20, errors='replace')
        return {'command': argv, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command': argv, 'error_type': type(e).__name__}


def prepare():
    if PUB.exists() or PR.exists():
        raise ValueError('NO_OVERWRITE_NO_RESUME')
    pins = read(PARENT / 'protected_hashes.json')
    for f in (PPUB / 'completion_manifest.json', PARENT / 'private_index.json'):
        m = read(f)
        if digest(m['files']) != m['fingerprint']:
            raise ValueError('PARENT_MANIFEST')
        merge_pins(pins, m['files'])
        merge_pins(pins, {f.relative_to(ROOT).as_posix(): sha(f)})
    if check(pins)['changed_or_missing']:
        raise ValueError('PROTECTED_CHANGED')
    sources = dict(read(PARENT / 'source_hashes.json'))
    # Supplementary C4-32 raw-only inspection is already development exposure.
    rawpins = PARENT / 'raw_only_source_hashes.json'
    if rawpins.exists():
        merge_pins(sources, read(rawpins))
    if check(sources)['changed_or_missing']:
        raise ValueError('SOURCE_CHANGED')
    PUB.mkdir(parents=True)
    PR.mkdir(parents=True)
    write(PR / 'protected_hashes.json', pins)
    runtime = {'docker': observe(['docker', 'version', '--format', '{{json .}}']),
               'images': observe(['docker', 'images', '--no-trunc']), 'java': observe(['java', '-version'])}
    write(PR / 'runtime_inventory.json', runtime)
    original_metadata = list((COHORT / 'metadata').glob('*.json'))
    metadata = [read(f) for f in original_metadata]
    oldjar = (COHORT / 'jar_sha256.txt').read_text().strip()
    historical_image = read(AUDIT / 'docker_image_inspect.json')
    if isinstance(historical_image, list):
        historical_image = historical_image[0]
    archived_commit = (AUDIT / 'inspection/git_commit.txt').read_text().strip()
    if historical_image['Id'] != IMAGE or archived_commit != COMMIT:
        raise ValueError('ARCHIVED_GENERATOR_PIN')
    evidence = original_metadata + [COHORT / 'jar_sha256.txt', AUDIT / 'docker_image_inspect.json',
        AUDIT / 'inspection/git_commit.txt', AUDIT / 'inspection/git_describe.txt',
        AUDIT / 'inspection/jar_sha256.txt', SYN / 'Dockerfile',
        SYN / 'experiments/12_symptom_cohort_100/run_synthea_symptom_cohort.sh']
    resources = {'cpu_limit': 4, 'generator_threads': 2, 'memory_gib': 6, 'java_heap_gib': 3,
        'tmpfs_gib': 1, 'wall_seconds': 3600, 'output_budget_gib': 10,
        'disk_enforcement': 'operator filesystem quota or watchdog required before generation; Docker bind mount has no size quota',
        'preflight_free_disk_min_gib': 20, 'free_disk_bytes_observed': shutil.disk_usage(ROOT).free,
        'population_requested': 1000, 'generation_attempts': 1, 'retry': 0,
        'estimate': 'planning ceilings, not measured new cohort costs; historical metadata 1175 exported/235sec on20 threads is not a runtime guarantee'}
    inv = {'status': 'GENERATOR_RUNTIME_NOT_VERIFIED', 'historical_cohort_metadata': metadata,
        'historical_cohort_jar_sha256': oldjar, 'archived_audit_image_id': IMAGE,
        'archived_commit': archived_commit, 'archived_describe': (AUDIT/'inspection/git_describe.txt').read_text().strip(),
        'archived_jar_sha256': JAR, 'cohort_jar_matches_archived_audit': oldjar == JAR,
        'exact_original_generator_reproducible': False, 'docker_daemon_available': runtime['docker'].get('returncode') == 0,
        'current_image_exists': 'NOT_CHECKED' if runtime['images'].get('returncode') else IMAGE in runtime['images']['stdout'],
        'host_java': runtime['java'], 'proposed_environment': 'archived immutable local Docker image; availability/JAR/resources must be rechecked, never rebuild moving latest',
        'original_conditions': 'all modules *, Massachusetts; no target-diagnosis prevalence intervention',
        'original_full_command_and_properties': 'NOT_FOUND for cohort1000; metadata is partial',
        'source_db': 'existing annotation_store.sqlite contexts/events/sources assembled by experiments21; fresh CSV adapter avoids cloning clinical review DB',
        'evidence_sha256': {str(f): sha(f) for f in evidence}}
    write(PUB / 'synthea_generator_inventory.json', inv)
    write(PUB / 'generation_resource_budget.json', resources)
    sourcefiles = [DEV / 'base_visits.json', DEV / 'action_keys.json', BANK.parent / 'voi_model.json']
    sources.update({str(f): sha(f) for f in sourcefiles + evidence})
    # Freeze plan before the single allowed estimate and before any new export exists.
    tp = training_protocol(*(sha(f) for f in sourcefiles))
    tp['selection_prior'] = 'alpha1 development visit counts on existing base5; same C4-30 estimator convention'
    tp['fingerprint'] = digest(tp)
    write(PUB / 'final_m5_training_protocol.json', tp)
    base, keys, template = (read(f) for f in sourcefiles)
    excluded = set(read(DEV / 'excluded_evaluation_patients.json'))
    if len(base) != 229 or len({v['patient_id'] for v in base}) != 173:
        raise ValueError('DEVELOPMENT_BASE_SCOPE')
    write(PR / 'Estimation001_before.json', {'protocol_fingerprint': tp['fingerprint'], 'maximum_estimations': 1,
          'new_source_accessed': False, 'at': datetime.now(timezone.utc).isoformat()})
    model, fitted_rows, selection = fit_final(template, base, keys, excluded)
    write(PR / 'final_m5_model.json', model)
    write(PR / 'training_rows.json', fitted_rows)
    write(PR / 'final_action_selection.json', selection)
    write(PR / 'Estimation001_after.json', {'status': 'DEVELOPMENT_ONLY', 'artifact_sha256': sha(PR / 'final_m5_model.json'),
        'training_patients': len(model['provenance']['training_patients']), 'training_visits': model['provenance']['training_visits'],
        'training_rows': len(fitted_rows), 'clinical_approval': 'NOT_APPROVED'})
    am = {'status': 'FROZEN_DEVELOPMENT_ONLY', 'estimations': 1, 'protocol_fingerprint': tp['fingerprint'],
        'artifact_sha256': sha(PR / 'final_m5_model.json'), 'selection_sha256': sha(PR / 'final_action_selection.json'),
        'action_count': len(model['actions']), 'diagnosis_count': len(model['diagnoses']),
        'model_has_observed_new_patients': False, 'clinical_reference_approved': False,
        'prior': model['prior'], 'selected_keys': selection['selected_keys'], 'training_patients': 173, 'training_visits': 229,
        'fit_training_rows': len(fitted_rows), 'core_fit_function': 'app.core.voi_evaluation.fit_outcomes(alpha=1)',
        'frozen_before_generation': True}
    write(PUB / 'final_m5_artifact_manifest.json', am)
    finish_preparation()


def finish_preparation():
    """Complete source inventory after a recorded failure, never repeat estimation."""
    if (PUB/'new_cohort_generation_protocol.json').exists():
        raise ValueError('PREPARATION_ALREADY_FROZEN')
    if sha(PR/'final_m5_model.json') != read(PUB/'final_m5_artifact_manifest.json')['artifact_sha256']:
        raise ValueError('FROZEN_FINAL_MODEL_CHANGED')
    am = read(PUB/'final_m5_artifact_manifest.json')
    model = read(PR/'final_m5_model.json')
    resources = read(PUB/'generation_resource_budget.json')
    sources = dict(read(PARENT/'source_hashes.json'))
    merge_pins(sources, read(PARENT/'raw_only_source_hashes.json'))
    inv = read(PUB/'synthea_generator_inventory.json')
    sources.update(inv['evidence_sha256'])
    for f in (DEV/'base_visits.json', DEV/'action_keys.json', BANK.parent/'voi_model.json'):
        sources[str(f)] = sha(f)
    base = read(DEV/'base_visits.json')
    # Known exposure: every raw cohort patient has already been inventoried in C4-32.
    raw = load_csv(COHORT)
    pids, encounters = identities(raw)
    fp = encounter_fingerprints(raw)
    development = read(DEV / 'base_visits.json') + read(DEV / 'additional_visits.json')
    original = rows(ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl')
    d3 = rows(D3 / 'cases.jsonl')
    db = list(database_sources())
    groups = {'raw_previously_inventoried': pids, 'all_C430_fit_selection': {v['patient_id'] for v in development},
        'C429_train_cv': {v['patient_id'] for v in base}, 'Original50': {v['patient_id'] for v in original},
        'D3': {v['patient_id'] for v in d3}, 'indexed_source_exposure': {v['patient_id'] for v in db}}
    allp = set().union(*groups.values())
    alle = set(encounters) | {v['index_encounter']['encounter_id'] for v in db}
    alle |= {v['visit_id'] for v in development}
    # Include earlier adjacent synthetic batches, not only the latest1175.
    # A content match is a conservative review blocker, not proof of copied patients.
    prior_raw_inventory = []
    for patient_file in sorted((SYN/'experiments').rglob('patients.csv')):
        prior_root = patient_file.parent.parent
        if prior_root.resolve() == COHORT.resolve():
            continue
        needed = [prior_root/'csv'/(name+'.csv') for name in ('patients','encounters','conditions','observations')]
        if not all(f.exists() for f in needed):
            prior_raw_inventory.append({'root':str(prior_root),'status':'MISSING_REQUIRED_CSV'})
            continue
        old_tables = load_csv(prior_root)
        oldp, olde = identities(old_tables)
        oldfp = encounter_fingerprints(old_tables)
        allp.update(oldp); alle.update(olde)
        fp.update({str(prior_root)+':'+eid: h for eid,h in oldfp.items()})
        prior_raw_inventory.append({'root':str(prior_root),'status':'STRUCTURAL_INVENTORY',
            'patients':len(oldp),'encounters':len(olde),'fingerprints':len(oldfp)})
        for f in needed:sources[str(f)] = sha(f)
    groups['all_prior_raw_batches'] = allp - set().union(*groups.values())
    write(PR/'prior_raw_batch_inventory.json',prior_raw_inventory)
    write(PR / 'exclusion_inventory.json', {'patients': sorted(allp), 'encounters': sorted(alle),
        'encounter_fingerprints': fp, 'groups': {k: sorted(v) for k, v in groups.items()}})
    for name in ('patients', 'encounters', 'conditions', 'observations'):
        f = COHORT / 'csv' / (name + '.csv')
        sources[str(f)] = sha(f)
    sources.update({str(DATABASE): sha(DATABASE), str(D3/'cases.jsonl'):sha(D3/'cases.jsonl')})
    write(PR / 'source_hashes.json', sources)
    write(PUB / 'cohort_overlap_audit.json', {'new_source_status': 'NOT_RUN', 'known_exclusion_patients': len(allp),
        'known_exclusion_encounters': len(alle), 'group_patient_counts': {k: len(v) for k, v in groups.items()},
        'fingerprint_encounters': len(fp), 'unknown_external_exposure': 'NOT_CHECKED', 'new_overlap_result': 'NOT_RUN'})
    ep = evaluation_protocol(am['artifact_sha256'], am['selection_sha256'])
    ep['fingerprint'] = digest(ep)
    write(PUB / 'new_cohort_evaluation_protocol.json', ep)
    gp = {'protocol': 'locdi.voi_new_synthetic_holdout.c4_33.v1', 'namespace': NS,
        'seed': 2026111601, 'clinician_seed': 2026111601, 'reference_date': '20261116', 'end_date': '20261116',
        'population_requested': 1000, 'attempts': 1, 'state': 'Massachusetts', 'modules': '*',
        'diagnosis_targeting': False, 'seed_changes_after_results': False,
        'generator_image_id': IMAGE, 'generator_commit': COMMIT, 'generator_jar_sha256': JAR,
        'generator_family_shared_with_development': True,
        'raw_relative_path': 'outputs/private/research_evaluation/c4_33_v1/raw',
        'export': {'csv': True, 'native_json': True, 'fhir': False, 'history_years': 0, 'symptoms_mode': 1},
        'parser_version': 'locdi.voi_new_cohort_source.c4_33.v1', 'parser_hash': sha(ROOT / CODE[0]),
        'source_eligibility': 'single explicit existing-universe Condition DESCRIPTION on same encounter, numeric exact system/code/unit, recorded0..45min; Ask72514-3 at/before arrival only',
        'reference_timing': 'Source Condition START not verified diagnosis confirmation; retrospective development reference only',
        'resource_budget': resources, 'final_artifact_sha256': am['artifact_sha256'],
        'evaluation_protocol_fingerprint': ep['fingerprint'],
        'generation_approval': 'AWAITING_GENERATION_APPROVAL', 'evaluation_approval': 'NOT_GRANTED',
        'stop_conditions': ['generator pin mismatch', 'source hash mismatch', 'patient/encounter/content duplication',
            'resource cap', 'invalid model or Core failure', 'no complete panel: no evaluation or seed reroll'],
        'code_hashes': {f: sha(ROOT / f) for f in CODE}}
    gp['fingerprint'] = digest(gp)
    write(PUB / 'new_cohort_generation_protocol.json', gp)
    for name in ('diagnosis_eligibility', 'action_panel_eligibility', 'source_availability_audit'):
        write(PUB / (name + '.json'), {'status': 'NOT_RUN', 'reason': 'new generation not authorized or executed',
             'actual_new_patients': 0, 'actual_new_visits': 0, 'complete_panels': 0,
             'clinical_availability': 'NOT_VERIFIED', 'no_missing_outcome_imputation': True})
    readiness = {'status': 'C4_33_GENERATOR_OR_SOURCE_BLOCKED', 'generation_state': 'AWAITING_GENERATION_APPROVAL',
        'evaluation_state': 'NOT_RUN', 'new_generated_patients': 0, 'complete_panels': 0,
        'single_final_m5': 'FROZEN_DEVELOPMENT_ONLY', 'blocker_codes': ['GENERATION_APPROVAL_REQUIRED',
            'DOCKER_ENGINE_UNAVAILABLE', 'PINNED_IMAGE_RUNTIME_NOT_VERIFIED', 'ORIGINAL_COHORT_JAR_DIFFERS_FROM_ARCHIVED_IMAGE',
            'STORAGE_QUOTA_OR_WATCHDOG_NOT_CONFIGURED', 'NO_NEW_SOURCE_PANEL', 'EVALUATION_APPROVAL_REQUIRED'],
        'LLM_generation': 0, 'Synthea_generation': 0, 'new_policy_evaluation': 0, 'probability_estimations': 1,
        'research_evaluation_ready': False}
    write(PUB / 'new_cohort_evaluation_readiness.json', readiness)
    command_documents(gp)
    print(json.dumps({'status': readiness['status'], 'protocol_fingerprint': gp['fingerprint'], 'final_actions':len(model['actions']), 'fitting': 1, 'generation': 0}))


def command_documents(gp):
    # A command is provided for review only. This program never calls docker run.
    command = '''# Proposed generation command -- NOT EXECUTED / NOT YET APPROVED

First start Docker Desktop manually, verify the pinned image exists, inspect
its /app commit/JAR/properties without generation, and configure a hard 10GiB
output quota or a watchdog that terminates this container at 10GiB. Do not
rebuild a moving latest image. Obtain approval bound to this protocol hash.
The historical1000 JAR differs; this is the archived audit generator proposal,
not a claim to reproduce the 1000 cohort exactly. Approval must accept that.

```powershell
$raw = Join-Path (Get-Location) 'outputs/private/research_evaluation/c4_33_v1/raw'
if (Test-Path -LiteralPath $raw) { throw 'NO_OVERWRITE_NO_RESUME' }
# Only after the checks/approval above:
New-Item -ItemType Directory -Path $raw -ErrorAction Stop
docker run --rm --name locdi-c4-33-generation --network none --cpus 4 --memory 6g --memory-swap 6g --read-only --tmpfs /tmp:rw,size=1g --mount "type=bind,source=$raw,target=/new" --entrypoint /bin/sh IMAGE_PIN -ec 'test "$(sha256sum /app/build/libs/synthea-with-dependencies.jar | cut -d " " -f 1)" = JAR_PIN; test "$(git -C /app rev-parse HEAD)" = COMMIT_PIN; cd /app; timeout 3600 java -Xmx3g -jar build/libs/synthea-with-dependencies.jar -p 1000 -o false -s 2026111601 -cs 2026111601 -r 20261116 -E 20261116 --generate.thread_pool_size=2 --generate.max_attempts_to_keep_patient=1 --generate.only_alive_patients=false --generate.only_dead_patients=false --exporter.baseDirectory=/new --exporter.years_of_history=0 --exporter.csv.export=true --exporter.csv.append_mode=false --exporter.csv.folder_per_run=false --exporter.csv.included_files= --exporter.csv.excluded_files=patient_expenses.csv --exporter.fhir.export=false --exporter.fhir_stu3.export=false --exporter.fhir_dstu2.export=false --exporter.ccda.export=false --exporter.text.export=false --exporter.json.export=true --exporter.json.include_module_history=false --exporter.symptoms.csv.export=true --exporter.symptoms.text.export=true --exporter.symptoms.csv.append_mode=false --exporter.symptoms.csv.folder_per_run=false --exporter.symptoms.mode=1 Massachusetts > /new/generation.log 2>&1'
```

Requested population1000 is not asserted to equal actual exported patients.
One attempt only. Save actual inspect/JAR/properties/command/exit status and
hash every produced file after shutdown in an actual_generation_manifest.
No subsequent seed change, retry or automatic resume is authorized.
'''.replace('IMAGE_PIN', IMAGE).replace('JAR_PIN', JAR).replace('COMMIT_PIN', COMMIT)
    command += '\nGeneration protocol fingerprint: `' + gp['fingerprint'] + '`\n'
    (PUB / 'generator_execution_command.md').write_text(command, encoding='utf-8')
    (PUB / 'actual_evaluation_command.md').write_text('''# Source eligibility command, after approved generation

```powershell
python -B scripts/prepare_new_voi_cohort.py inspect-source --source-root outputs/private/research_evaluation/c4_33_v1/raw
```

This reads and hashes source; it does not execute evaluation. Existing C4-31
CLI is hard-bound to the old92/fivefold panel and must not be pointed at new
data. New single-model replay entry point: app.locdi.voi_new_cohort_evaluation.policy_episode,
using actual C4-31 episode and C4-32 Dynamic adapter. Separate evaluation
approval and a sealed actual generation receipt are required before a future
batch CLI invokes it. End-to-end approved evaluation CLI: NOT_READY; no
misleading runnable performance command is provided before Source admission.
All nine policy TEST_ONLY paths are tested independently of generation.
''', encoding='utf-8')
    (PUB / 'next_research_decision.md').write_text('''# Next decision

Restore/inspect the archived immutable generator first. The current Docker
engine is unavailable and cohort1000 JAR does not match the archived audit.
Do not grant generation approval against a guessed image. After image/JAR
and output quota verification, approve the frozen seed2026111601, population1000,
single attempt, 4CPU/6GiB/3GiB heap/10GiB output/3600sec and new raw namespace.
No new cohort is present. Single final M5 is frozen before generation. If no
complete panel is obtained, retain every exclusion and stop without seed search.
Separate Source receipt/admission and evaluation execution approval follow.
Independent synthetic replay remains different from independent clinical validation.
''', encoding='utf-8')


def verify():
    gp = read(PUB / 'new_cohort_generation_protocol.json')
    if gp['fingerprint'] != digest({k:v for k,v in gp.items() if k != 'fingerprint'}):
        raise ValueError('PROTOCOL_HASH')
    for f, h in gp['code_hashes'].items():
        if sha(ROOT / f) != h:
            raise ValueError('CODE_CHANGED')
    m = read(PUB / 'final_m5_artifact_manifest.json')
    if sha(PR / 'final_m5_model.json') != m['artifact_sha256'] or sha(PR / 'final_action_selection.json') != m['selection_sha256']:
        raise ValueError('FINAL_MODEL_CHANGED')
    a, b = check(read(PR / 'protected_hashes.json')), check(read(PR / 'source_hashes.json'))
    if a['changed_or_missing'] or b['changed_or_missing']:
        raise ValueError('PROTECTED_OR_SOURCE_CHANGED')
    return {'status': 'PASS', 'protected': a, 'sources': b, 'Core_changes': 0, 'dynamic_threshold_changes': 0,
            'LLM_generation': 0, 'Synthea_generation': 0, 'new_source_performance': 'NOT_RUN'}


def inspect_source(root):
    verify()
    root = Path(root).resolve()
    if root != (PR / 'raw').resolve():
        raise ValueError('SOURCE_NAMESPACE')
    if not (PR / 'Estimation001_after.json').exists():
        raise ValueError('MODEL_MUST_BE_FROZEN_BEFORE_SOURCE')
    # This is a read-only eligibility inspection, never admission or performance.
    pins = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
    tables = load_csv(root)
    old = read(PR / 'exclusion_inventory.json')
    audit = overlap(tables, old['patients'], old['encounters'], old['encounter_fingerprints'].values())
    if audit['status'] != 'PASS':
        raise ValueError('NEW_COHORT_OVERLAP')
    model = read(PR / 'final_m5_model.json')
    visits, ledger, summary = adapt(tables, root, model)
    if check(pins)['changed_or_missing']:
        raise ValueError('SOURCE_CHANGED_DURING_INSPECTION')
    # Exclusive files: never edit completed preparation records.
    write(PR / 'new_source_inspected_hashes.json', pins)
    write(PR / 'new_source_visits.json', visits)
    write(PR / 'new_source_eligibility_ledger.json', ledger)
    write(PUB / 'new_source_structural_inspection.json', {'overlap': audit, 'eligibility': summary,
          'actual_generation_receipt_verified': False, 'evaluation_approved': False, 'evaluation_ready': False})
    print(json.dumps({'status': 'STRUCTURAL_INSPECTION_ONLY', 'patients':summary['generated_patients'], 'complete':summary['complete_panel_visits']}))


def regression():
    verify()
    parent = read(PPUB / 'test_results.json')
    patterns = list(parent['patterns'])
    if 'test_voi_new_cohort.py' not in patterns:
        patterns.append('test_voi_new_cohort.py')
    suite = unittest.TestSuite()
    counts = {}
    for pattern in patterns:
        ss = unittest.defaultTestLoader.discover(str(ROOT / 'tests/locdi'), pattern=pattern) if pattern.endswith('.py') else unittest.defaultTestLoader.loadTestsFromName(pattern)
        counts[pattern] = ss.countTestCases()
        suite.addTests(ss)
    with (PR/'test_log.txt').open('x', encoding='utf-8') as f:
        with patch.object(socket.socket, 'connect', side_effect=RuntimeError('NETWORK_DISABLED')), patch.object(socket.socket, 'bind', side_effect=RuntimeError('NETWORK_DISABLED')):
            result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
    write(PUB / 'test_results.json', {'tests':result.testsRun, 'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'fail':len(result.failures), 'error':len(result.errors), 'skip':len(result.skipped), 'patterns':patterns, 'counts':counts,
        'scope':'related inherited C4-32 regression plus C4-33; full repository NOT_RUN', 'network':'DISABLED'})
    write(PUB / 'verification.json', verify())
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({'tests':result.testsRun,'success':True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'verify', 'regression', 'inspect-source', 'finish-preparation'])
    parser.add_argument('--source-root')
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'finish-preparation': finish_preparation()
    elif args.stage == 'regression': regression()
    elif args.stage == 'inspect-source': inspect_source(args.source_root)
    else: print(json.dumps(verify()))
