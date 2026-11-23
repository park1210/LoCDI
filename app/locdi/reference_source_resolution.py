"""C3-07 documentary source distinctions, never a license/approval authority."""
import re
from app.locdi.llm_setup import digest

STATES = ('SOURCE_FOUND', 'SOURCE_VERSION_PINNED', 'SOURCE_FILE_LIST_VERIFIED',
          'SOURCE_BYTES_VERIFIED', 'SOURCE_HASH_VERIFIED', 'LICENSE_REVIEW_REQUIRED',
          'LICENSE_SCOPE_VERIFIED', 'ACQUISITION_APPROVAL_REQUIRED',
          'ACQUISITION_ALLOWED', 'ACQUISITION_BLOCKED')


def source_inventory():
    """Observed metadata only. Git commits are not downloaded-data SHA256s."""
    rows = [
        {'source_id': 'medqa_official_drive', 'operator': 'jind11/MedQA',
         'url': 'https://github.com/jind11/MedQA',
         'metadata_url': 'https://raw.githubusercontent.com/jind11/MedQA/27b02f66aac217933c9648a06f82e9f720377925/README.md',
         'code_revision': '27b02f66aac217933c9648a06f82e9f720377925',
         'data_revision': None, 'format': 'DRIVE_ARCHIVE_FORMAT_UNVERIFIED',
         'distribution_url': 'https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view',
         'documented_files': ['XX_qbank.jsonl', 'train/dev/test JSONL; exact member paths UNKNOWN'],
         'documented_options': [4, 5], 'index_convention': 'medqa_letter',
         'split_names': ['train', 'dev', 'test'],
         'license_observation': 'Repository MIT; data archive and underlying materials scope unverified',
         'qa_only': 'QA folders separate from textbooks in description; archive member list unavailable'},
        {'source_id': 'medqa_bigbio_en4_source', 'operator': 'bigbio/med_qa',
         'url': 'https://huggingface.co/datasets/bigbio/med_qa',
         'metadata_url': 'https://huggingface.co/datasets/bigbio/med_qa/blob/main/README.md',
         'revision_evidence_url': 'https://huggingface.co/datasets/bigbio/med_qa/commit/484a6c066fe8e75c83edea0c88b5169316714fcd',
         'code_revision': None, 'data_revision': '484a6c066fe8e75c83edea0c88b5169316714fcd',
         'format': 'PARQUET', 'documented_options': [4], 'index_convention': 'UNVERIFIED_TRANSFORMED_SCHEMA',
         'documented_files': ['med_qa_en_4options_source/train-*', 'med_qa_en_4options_source/validation-*',
                              'med_qa_en_4options_source/test-*'],
         'split_names': ['train', 'validation', 'test'],
         'license_observation': 'Dataset card license unknown; public access is not a usage grant',
         'qa_only': 'Explicit en_4options_source configuration; default is en_source, not the same variant'},
        {'source_id': 'medqa_awinml_questions', 'operator': 'awinml/medqa',
         'url': 'https://huggingface.co/datasets/awinml/medqa',
         'metadata_url': 'https://huggingface.co/datasets/awinml/medqa/blob/main/README.md',
         'revision_evidence_url': 'https://huggingface.co/datasets/awinml/medqa/commit/afdb980b627374a7f10a0a11ed68455596c52742',
         'code_revision': None, 'data_revision': 'afdb980b627374a7f10a0a11ed68455596c52742',
         'format': 'JSONL_DOCUMENTED', 'documented_options': [4], 'index_convention': 'medqa_letter',
         'documented_files': ['questions/train.jsonl', 'questions/validation.jsonl', 'questions/test.jsonl'],
         'split_names': ['train', 'validation', 'test'],
         'documented_counts': {'train': 10178, 'validation': 1272, 'test': 1273},
         'license_observation': 'Redistributor claims MIT based on original repository; underlying rights still require review',
         'qa_only': 'questions config only; exclude entire corpus config and corpus/train.jsonl'}]
    for r in rows:
        r.update(reviewed_on='2026-11-02', actual_file_inventory_verified=False,
                 actual_bytes_verified=False, actual_hash_verified=False,
                 actual_file_sha256=None, actual_file_bytes=None, accepted_license_proof=None,
                 acquisition_approval=None, byte_equivalence_to_original='UNKNOWN',
                 raw_rows_acquired=0, clinical_reference_approved=False)
        # This hashes OUR note, not a remote document or actual dataset bytes.
        r['observation_note_fingerprint'] = digest(r)
        r['states'] = resolve(r)
    return rows


def resolve(note):
    # Documentary caller claims cannot grant acquisition or promote data.
    return dict(zip(STATES, (bool(note.get('url')),
        bool(re.fullmatch('[0-9a-f]{40,64}', note.get('data_revision') or '')),
        False, False, False, True, False, True, False, True)))


def require_identity(actual, expected):
    if actual.get('source_id') != expected.get('source_id'):
        raise ValueError('SOURCE_DISTRIBUTION_MISMATCH')
    if actual.get('revision') != expected.get('revision'):
        raise ValueError('SOURCE_REVISION_MISMATCH')
    if not re.fullmatch('[0-9a-f]{40,64}', actual.get('revision') or ''):
        raise ValueError('SOURCE_REVISION_REQUIRED')


def final_state(raw_pass=False, acquired=False):
    if raw_pass and not acquired:
        raise ValueError('RAW_REQUIRED_FOR_VALIDATION_PASS')
    return {'C3_07_SOURCE_RESOLUTION': 'PASS_DEVELOPMENT_ONLY',
            'C3_07_RAW_VALIDATION': 'PASS' if raw_pass else 'BLOCKED',
            'MEDQA_RAW_ACQUIRED': bool(acquired), 'EXTERNAL_EVALUATION_READY': False,
            'clinical_reference_ready': False, 'clinical_metrics': 'NOT_EVALUATED'}
