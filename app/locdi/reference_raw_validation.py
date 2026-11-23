"""Bounded read-only raw validation. Reuses C3-06 adapter and approval gate.

Pure byte parsers are NOT source/license approvals. Actual input is an operator-
pinned private manifest and existing acquisition receipt/approval/index evidence.
ZIP inspection is defensive only; no extraction or invented acquisition receipt.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from app.locdi.llm_setup import ROOT, digest
from app.locdi.reference_acquisition import strict_json, validate_plan, PinnedApproval
from app.locdi.external_reference_readiness import pinned_private_json, evaluate_gate
from app.locdi.external_reference_task import adapt_exam, model_projection, public_projection
from app.locdi.reference_source_resolution import require_identity

MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100000
MAX_LINE_BYTES = 1024 * 1024


def bounded_integer(value, maximum, code):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(code)
    return value


def inventory_local_candidate(path):
    """Explicit path only, quarantine shape inventory, never approved raw/adaptation.

    User-supplied legacy files may be misnamed/derived. No directory enumeration,
    copying, index guessing, source attribution or clinical interpretation.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('EXPLICIT_LOCAL_FILE_NOT_FOUND')
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('RAW_SIZE_LIMIT')
    with path.open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    actual = hashlib.sha256(data).hexdigest()
    rows = parse_jsonl(data, name=path.name, expected_sha256=actual, expected_bytes=len(data))
    shape = sum(isinstance(r.get('question'), str) and isinstance(r.get('options'), dict)
                and isinstance(r.get('answer_idx'), str) for r in rows)
    public = {'schema_version': 'locdi.c3_07.quarantined_local_inventory.v1',
        'status': 'QUARANTINED_NOT_SOURCE_VALIDATED', 'file_exists': True,
        'file_bytes': len(data), 'observed_local_file_sha256': actual, 'record_count': len(rows),
        'direct_medqa_minimum_shape_count': shape,
        'osce_envelope_count': sum(set(r) == {'OSCE_Examination'} for r in rows),
        'official_original_sha256': None, 'source_revision': None,
        'source_status': 'UNKNOWN', 'license_status': 'UNKNOWN',
        'user_believes_research_use_possible': True, 'license_evidence_accepted': False,
        'adapter_executed': False, 'clinical_metrics': 'NOT_EVALUATED',
        'blocker_codes': ['SOURCE_AND_RIGHTS_UNVERIFIED'] +
                         (['LOCAL_SCHEMA_NOT_DIRECT_MEDQA'] if shape != len(rows) else []),
        'MEDQA_RAW_ACQUIRED': False, 'EXTERNAL_EVALUATION_READY': False}
    private = {'explicit_path': str(path), 'observed_local_file_sha256': actual,
               'scope': 'Only quarantined file shape, not source/license/schema admission',
               'operator_statement': 'Origin and rights unknown; user believes research use possible'}
    return public, private


def parse_jsonl(data, *, name, expected_sha256, expected_bytes, max_bytes=MAX_BYTES,
                max_rows=MAX_ROWS):
    bounded_integer(max_bytes, MAX_BYTES, 'FINITE_BYTE_BUDGET_REQUIRED')
    bounded_integer(max_rows, MAX_ROWS, 'FINITE_ROW_BUDGET_REQUIRED')
    if len(data) > max_bytes:
        raise ValueError('RAW_SIZE_LIMIT')
    if len(data) != expected_bytes:
        raise ValueError('RAW_BYTES_MISMATCH')
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('RAW_HASH_MISMATCH')
    if Path(name).suffix.lower() != '.jsonl' or data[:4] in (b'PK\x03\x04', b'PAR1'):
        raise ValueError('EXTENSION_CONTENT_MISMATCH')
    try:
        content = data.decode('utf8', errors='strict')
    except UnicodeDecodeError:
        raise ValueError('INVALID_UTF8') from None
    if content.lstrip().startswith(('<', '[')) or content.startswith('\ufeff'):
        raise ValueError('EXTENSION_CONTENT_MISMATCH')
    rows = []
    for line in content.splitlines():
        if not line.strip():
            raise ValueError('BLANK_JSONL_RECORD')
        if len(line.encode('utf8')) > MAX_LINE_BYTES:
            raise ValueError('JSONL_LINE_SIZE_LIMIT')
        if len(rows) >= max_rows:
            raise ValueError('RAW_ROW_LIMIT')
        try:
            row = strict_json(line)
        except ValueError as exc:
            if 'DUPLICATE_JSON_KEY' in str(exc):
                raise ValueError('DUPLICATE_JSON_KEY') from None
            raise ValueError('INVALID_JSON_RECORD') from None
        if not isinstance(row, dict):
            raise ValueError('JSONL_OBJECT_REQUIRED')
        try:
            json.dumps(row, allow_nan=False)
        except ValueError:
            raise ValueError('NONFINITE_JSON_NUMBER') from None
        rows.append(row)
    if not rows:
        raise ValueError('EMPTY_RAW')
    return rows


def inspect_zip(data, *, max_expanded_bytes=MAX_BYTES, max_members=1000):
    """Inspect without writing. Never returns decoded QA or approval."""
    bounded_integer(max_expanded_bytes, MAX_BYTES, 'FINITE_ARCHIVE_BUDGET_REQUIRED')
    bounded_integer(max_members, 1000, 'FINITE_ARCHIVE_MEMBER_BUDGET_REQUIRED')
    if len(data) > MAX_BYTES:
        raise ValueError('ARCHIVE_INPUT_SIZE_LIMIT')
    if not data.startswith(b'PK\x03\x04'):
        raise ValueError('ARCHIVE_FORMAT_MISMATCH')
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError('ARCHIVE_FORMAT_MISMATCH') from None
    with archive:
        infos = archive.infolist()
        if len(infos) > max_members:
            raise ValueError('ARCHIVE_MEMBER_LIMIT')
        seen = set(); total = 0
        for entry in infos:
            # On Windows ZipInfo normalizes backslashes in filename; inspect the
            # original archive header too so traversal is not silently normalized.
            name = entry.orig_filename
            path = PurePosixPath(name)
            if (not name or '\\' in name or ':' in name or '\x00' in entry.orig_filename or
                    path.is_absolute() or '..' in path.parts or any(p in ('', '.') for p in name.split('/'))):
                raise ValueError('ARCHIVE_PATH_TRAVERSAL')
            key = name.casefold()
            if key in seen:
                raise ValueError('ARCHIVE_DUPLICATE_PATH')
            seen.add(key)
            if stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                raise ValueError('ARCHIVE_UNSUPPORTED_MEMBER')
            total += entry.file_size
            if total > max_expanded_bytes or entry.file_size > max(1, entry.compress_size) * 200:
                raise ValueError('ARCHIVE_EXPANSION_LIMIT')
        # Read in bounded chunks to verify actual expansion/CRC too; never extract.
        actual = 0
        try:
            for entry in infos:
                with archive.open(entry) as stream:
                    count = 0
                    while True:
                        chunk = stream.read(min(65536, max_expanded_bytes - actual + 1))
                        if not chunk:
                            break
                        count += len(chunk); actual += len(chunk)
                        if actual > max_expanded_bytes:
                            raise ValueError('ARCHIVE_EXPANSION_LIMIT')
                    if count != entry.file_size:
                        raise ValueError('ARCHIVE_BYTES_MISMATCH')
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError):
            raise ValueError('ARCHIVE_MEMBER_INTEGRITY') from None
    return {'member_count': len(infos), 'expanded_bytes': actual,
            'status': 'INSPECTED_NOT_EXTRACTED', 'raw_adapter_ready': False,
            'blocker_codes': ['ARCHIVE_SOURCE_TO_JSONL_LINEAGE_REQUIRED']}


def adapt_verified(rows, contract, *, contract_fingerprint, identity, expected_identity,
                   expected_option_count):
    require_identity(identity, expected_identity)
    if (contract.get('dataset_id') != 'MedQA' or contract.get('version') != identity['revision']):
        raise ValueError('INDEX_SOURCE_REVISION_MISMATCH')
    if type(expected_option_count) is not int or expected_option_count not in (4, 5):
        raise ValueError('OPTION_VARIANT_REQUIRED')
    if any(not isinstance(r.get('options'), dict) or len(r['options']) != expected_option_count for r in rows):
        raise ValueError('SOURCE_OPTION_VARIANT_MISMATCH')
    tasks, keys = adapt_exam(rows, contract, verified_contract_fingerprint=contract_fingerprint)
    for task in tasks:
        model_projection(task)
    return tasks, keys


def cross_split_check(split_rows):
    seen_ids = {}; seen_questions = {}
    for split, rows in split_rows.items():
        if split not in ('train', 'dev', 'validation', 'test'):
            raise ValueError('ORIGINAL_SPLIT_REQUIRED')
        for r in rows:
            if r.get('id'):
                if r['id'] in seen_ids and seen_ids[r['id']] != split:
                    raise ValueError('CROSS_SPLIT_DUPLICATE_ID')
                seen_ids[r['id']] = split
            q = ' '.join(r['question'].casefold().split())
            if q in seen_questions and seen_questions[q] != split:
                raise ValueError('CROSS_SPLIT_DUPLICATE_QUESTION')
            seen_questions[q] = split
    return {'status': 'PASS_EXACT_ID_AND_NORMALIZED_TEXT_ONLY',
            'semantic_duplicates': 'NOT_CHECKED', 'patient_independence': 'NOT_APPLICABLE_EXAM_TASK'}


def safe_public(value):
    """Strict public aggregate schema: reject private extras, do not redact silently."""
    allowed = {'schema_version', 'status', 'row_count', 'split_counts', 'raw_sha256',
               'source_id', 'revision', 'blocker_codes', 'external_evaluation_ready',
               'clinical_metrics', 'adapter_source_sha256', 'input_projection_fingerprint'}
    if set(value) - allowed:
        raise ValueError('PRIVATE_FIELD_IN_PUBLIC_OUTPUT')
    if value.get('external_evaluation_ready') is not False:
        raise ValueError('EXECUTION_APPROVAL_REQUIRED')
    if value.get('status') not in ('PASS_STRUCTURAL_ONLY', 'BLOCKED'):
        raise ValueError('PUBLIC_STATUS_INVALID')
    if value.get('status') == 'PASS_STRUCTURAL_ONLY' and not value.get('raw_sha256'):
        raise ValueError('RAW_REQUIRED_FOR_VALIDATION_PASS')
    if value.get('source_id') is not None and value['source_id'] not in (
            'medqa_official_drive', 'medqa_bigbio_en4_source', 'medqa_awinml_questions'):
        raise ValueError('PUBLIC_SOURCE_ID_INVALID')
    if value.get('revision') is not None and not re.fullmatch('[0-9a-f]{40,64}', value['revision']):
        raise ValueError('PUBLIC_REVISION_INVALID')
    if value.get('clinical_metrics') not in (None, 'NOT_EVALUATED'):
        raise ValueError('CLINICAL_METRICS_FORBIDDEN')
    if not isinstance(value.get('blocker_codes', []), list) or any(
            not isinstance(c, str) or not re.fullmatch('[A-Z][A-Z0-9_]{0,95}', c)
            for c in value.get('blocker_codes', [])):
        raise ValueError('PUBLIC_BLOCKER_INVALID')
    if 'row_count' in value and (type(value['row_count']) is not int or value['row_count'] < 0):
        raise ValueError('PUBLIC_ROW_COUNT_INVALID')
    if value.get('schema_version') != 'locdi.c3_07.raw_validation.v1':
        raise ValueError('PUBLIC_SCHEMA_INVALID')
    if not isinstance(value.get('split_counts', {}), dict) or any(
            k not in ('train', 'dev', 'validation', 'test') or type(v) is not int or v < 0
            for k, v in value.get('split_counts', {}).items()):
        raise ValueError('PUBLIC_SPLIT_COUNTS_INVALID')
    for field in ('raw_sha256', 'adapter_source_sha256', 'input_projection_fingerprint'):
        if value.get(field) is not None and not re.fullmatch('[0-9a-f]{64}', value[field]):
            raise ValueError('PUBLIC_FINGERPRINT_INVALID')
    return value


def validate_manifest(path, expected_sha256):
    """Private input contract, pinned OUTSIDE the file by operator CLI argument.

    manifest: source_identity, expected_source_identity, plans[] with plan and
    raw_evidence/index_evidence as C3-06, plus option_count per split. Existing
    PinnedApproval authenticates plan/license scope; no new boolean approval.
    Returns public aggregate and private projected tasks/keys separately.
    """
    manifest = pinned_private_json(path, expected_sha256)
    require_identity(manifest['source_identity'], manifest['expected_source_identity'])
    entries = manifest.get('splits')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 3:
        raise ValueError('FINITE_SPLIT_MANIFEST_REQUIRED')
    tasks_by_split = {}; keys_by_split = {}; source_rows = {}; lineage = []; gates = []
    for entry in entries:
        plan = entry['plan']; raw = entry['raw_evidence']; index = entry['index_evidence']
        if validate_plan(plan)['blocker_codes']:
            raise ValueError('ACQUISITION_PLAN_NOT_READY')
        if len(plan['expected_files']) != 1:
            raise ValueError('SINGLE_JSONL_REQUIRED')
        spec = plan['expected_files'][0]
        if spec['expected_bytes'] > MAX_BYTES:
            raise ValueError('RAW_SIZE_LIMIT')
        if Path(spec['name']).suffix.lower() != '.jsonl':
            raise ValueError('FORMAT_NOT_SUPPORTED_FOR_ADMISSION')
        source_id = manifest['source_identity']['source_id']
        if source_id == 'medqa_bigbio_en4_source':
            raise ValueError('PARQUET_SOURCE_TRANSFORMATION_NOT_VERIFIED')
        prefix = {'medqa_awinml_questions': 'https://huggingface.co/datasets/awinml/medqa/',
                  'medqa_official_drive': 'https://drive.google.com/'}
        if source_id not in prefix or not spec['url'].startswith(prefix[source_id]):
            raise ValueError('SOURCE_DISTRIBUTION_URL_MISMATCH')
        PinnedApproval(raw['approval_path'], raw['approval_sha256']).verify(plan)
        if plan['dataset_id'] != 'MedQA' or plan['pinned_revision'] != manifest['source_identity']['revision']:
            raise ValueError('SOURCE_REVISION_MISMATCH')
        split = plan['expected_split']
        if split in source_rows:
            raise ValueError('DUPLICATE_SPLIT')
        source = (Path(raw['receipt_path']).resolve().parent / spec['name']).resolve()
        if not source.is_relative_to((ROOT / 'outputs/private').resolve()):
            raise ValueError('RAW_PRIVATE_PATH_REQUIRED')
        if source.stat().st_size > MAX_BYTES:
            raise ValueError('RAW_SIZE_LIMIT')
        # Gate checks actual acquisition receipt/URL/file bytes and index proof.
        gate = evaluate_gate(plan, raw_evidence=raw, index_evidence=index)
        if not gate['states']['RAW_ACQUIRED']:
            raise ValueError('ACQUISITION_RECEIPT_NOT_VERIFIED')
        # Read bounded even if supplied receipt plan allows up to 2GB.
        with source.open('rb') as stream:
            data = stream.read(MAX_BYTES + 1)
        rows = parse_jsonl(data, name=spec['name'], expected_sha256=spec['sha256'], expected_bytes=spec['expected_bytes'])
        c = pinned_private_json(index['path'], index['sha256'])
        if c.get('split') != split:
            raise ValueError('INDEX_SPLIT_MISMATCH')
        tasks, keys = adapt_verified(rows, c, contract_fingerprint=index['fingerprint'],
            identity=manifest['source_identity'], expected_identity=manifest['expected_source_identity'],
            expected_option_count=entry['option_count'])
        if not all(gate['states'][k] for k in ('RAW_SCHEMA_VERIFIED', 'ANSWER_INDEX_VERIFIED', 'INPUT_LEAKAGE_CHECKED')):
            raise ValueError('EXISTING_GATE_CONTRACT_NOT_VERIFIED')
        tasks_by_split[split] = tasks; keys_by_split[split] = keys; source_rows[split] = rows
        lineage.append({'split': split, 'original_sha256': hashlib.sha256(data).hexdigest(),
            'original_bytes': len(data), 'conversion': 'NONE_ORIGINAL_JSONL_BYTES_PRESERVED',
            'index_contract_sha256': index['sha256'], 'task_projection_fingerprint': digest(tasks)})
        gates.append(gate)
    overlap = cross_split_check(source_rows)
    all_tasks = [t for tasks in tasks_by_split.values() for t in tasks]
    public = safe_public({'schema_version': 'locdi.c3_07.raw_validation.v1', 'status': 'PASS_STRUCTURAL_ONLY',
        'row_count': len(all_tasks), 'split_counts': {k: len(v) for k, v in source_rows.items()},
        'raw_sha256': digest(lineage), 'source_id': manifest['source_identity']['source_id'],
        'revision': manifest['source_identity']['revision'], 'blocker_codes': ['MODEL_EXECUTION_NOT_APPROVED'],
        'external_evaluation_ready': False, 'clinical_metrics': 'NOT_EVALUATED',
        'adapter_source_sha256': hashlib.sha256((ROOT/'app/locdi/external_reference_task.py').read_bytes()).hexdigest(),
        'input_projection_fingerprint': digest([model_projection(t) for t in all_tasks])})
    private = {'tasks_by_original_split': tasks_by_split, 'answer_keys_by_original_split': keys_by_split,
               'source_to_task_lineage': lineage, 'split_check': overlap, 'existing_gate_results': gates,
               'public_adapter_projection': public_projection(all_tasks), 'clinical_reference_approved': False}
    return public, private
