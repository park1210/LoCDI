"""C3-06 evidence-bound external task checks. Does not admit clinical references."""
import hashlib
import json
from pathlib import Path
from app.locdi.llm_setup import ROOT,digest
from app.locdi.reference_acquisition import validate_plan,strict_json
from app.locdi.external_reference_task import adapt_exam,model_projection


def pinned_private_json(path,expected_sha256):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('EVIDENCE_PRIVATE_PATH')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('EVIDENCE_HASH')
    return strict_json(raw.decode('utf8'))


def evaluate_gate(plan,*,metadata=None,raw_evidence=None,index_evidence=None):
    """Paths + pinned hashes, not booleans/status claims. No executable model branch.

    Metadata receipt proves recorded documentary scope only. Raw/schema/index/leakage
    transitions require actual bytes and a matching acquisition receipt. C3-06 keeps
    execution/clinical approval blocked even with valid symbolic test fixtures.
    """
    check=validate_plan(plan);blockers=set(check['blocker_codes'])
    flags={k:False for k in ('SOURCE_METADATA_VERIFIED','RAW_ACQUIRED','RAW_SCHEMA_VERIFIED','ANSWER_INDEX_VERIFIED',
        'INPUT_LEAKAGE_CHECKED','EVALUATION_CONTRACT_READY','EXTERNAL_EVALUATION_READY')}
    flags.update(LICENSE_REVIEW_REQUIRED=True,ACQUISITION_APPROVAL_REQUIRED=True,EXECUTION_APPROVAL_REQUIRED=True)
    proof={};tasks=[]
    try:
        if metadata:
            note=pinned_private_json(metadata['path'],metadata['sha256'])
            if note['dataset_id']!=plan['dataset_id'] or not note.get('observations') or not note.get('official_urls'):raise ValueError('METADATA_SCOPE')
            flags['SOURCE_METADATA_VERIFIED']=note.get('current_documentary_review')=='CONFIRMED'
            proof['metadata_sha256']=metadata['sha256']
        if raw_evidence:
            receipt=pinned_private_json(raw_evidence['receipt_path'],raw_evidence['receipt_sha256'])
            if check['blocker_codes']:raise ValueError('PLAN_NOT_ACQUISITION_READY')
            if (receipt.get('status')!='RAW_ACQUIRED' or receipt.get('download_complete') is not True or receipt['plan_fingerprint']!=plan['plan_fingerprint'] or
                receipt['revision']!=plan['pinned_revision'] or receipt['dataset_id']!=plan['dataset_id'] or receipt['split']!=plan['expected_split']):raise ValueError('RAW_RECEIPT_SCOPE')
            from app.locdi.reference_acquisition import PinnedApproval
            PinnedApproval(raw_evidence['approval_path'],raw_evidence['approval_sha256']).verify(plan)
            files=[]
            for entry in plan['expected_files']:
                path=(Path(raw_evidence['receipt_path']).resolve().parent/entry['name']).resolve()
                if not path.is_relative_to((ROOT/'outputs/private').resolve()):raise ValueError('RAW_PATH_ESCAPE')
                data=path.read_bytes()
                expected={'name':entry['name'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':entry['url']}
                if len(data)!=entry['expected_bytes'] or expected['sha256']!=entry['sha256'] or not any(all(r.get(k)==v for k,v in expected.items()) for r in receipt['files']):raise ValueError('RAW_HASH_OR_RECEIPT_FILE')
                files.append(data)
            flags['RAW_ACQUIRED']=True;flags['LICENSE_REVIEW_REQUIRED']=False;flags['ACQUISITION_APPROVAL_REQUIRED']=False
            proof['raw_receipt_sha256']=raw_evidence['receipt_sha256']
            if len(files)!=1:raise ValueError('SINGLE_EXAM_JSONL_REQUIRED_NO_ARCHIVE_EXTRACTION')
            rows=[strict_json(line) for line in files[0].decode('utf8').splitlines() if line.strip()]
            if not rows:raise ValueError('EMPTY_RAW')
            if index_evidence:
                index=pinned_private_json(index_evidence['path'],index_evidence['sha256'])
                if (index['dataset_id'],index['version'],index['split'])!=(plan['dataset_id'],plan['pinned_revision'],plan['expected_split']):raise ValueError('INDEX_REVISION_OR_SPLIT_MISMATCH')
                tasks,_=adapt_exam(rows,index,verified_contract_fingerprint=index_evidence['fingerprint'])
                flags.update(RAW_SCHEMA_VERIFIED=True,ANSWER_INDEX_VERIFIED=True)
                for task in tasks:model_projection(task)
                flags['INPUT_LEAKAGE_CHECKED']=True
                proof['index_contract_sha256']=index_evidence['sha256']
    except (ValueError,KeyError,TypeError,OSError,UnicodeError) as exc:
        # No paths, question text or raw exception message in public result.
        blockers.add('EVIDENCE_INVALID_OR_INCOMPLETE_'+type(exc).__name__.upper())
    flags['EVALUATION_CONTRACT_READY']=all(flags[k] for k in ('SOURCE_METADATA_VERIFIED','RAW_ACQUIRED','RAW_SCHEMA_VERIFIED','ANSWER_INDEX_VERIFIED','INPUT_LEAKAGE_CHECKED')) and not blockers
    for key in ('RAW_ACQUIRED','RAW_SCHEMA_VERIFIED','ANSWER_INDEX_VERIFIED','INPUT_LEAKAGE_CHECKED'):
        if not flags[key]:blockers.add(key+'_REQUIRED')
    blockers.add('C3_06_EXECUTION_FORBIDDEN')
    blockers.add('NO_CLINICAL_REFERENCE_ADMISSION')
    return {'schema_version':'locdi.external_reference_readiness.v1','dataset_id':plan['dataset_id'],'states':flags,
      'blocker_codes':sorted(blockers),'evidence_fingerprints':proof,'task_count':len(tasks),
      'actual_generation_calls':0,'actual_grader_calls':0,'clinical_reference_approved':False,
      'scope':'Only separate external task; no Core, routing, VoI or independent patient-test admission'}
