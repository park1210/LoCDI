"""Resolve archived Synthea-relative paths without rewriting Source records.

Only the exact sibling project/suffix is considered. No cohort search, aliasing
or data replacement. Original cohort CSV bytes must match its archived manifest.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SYNTHEA=ROOT.parent/'001_synthea-docker'
COHORT='experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1/csv/'
MANIFEST=SYNTHEA/'experiments/15_symptom_cohort_1000/output/derived/raw_manifest.json'


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=128)
def _checked_hash(path,size,mtime_ns,manifest_path,manifest_size,manifest_mtime):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    key='output/cohort_1000_json_v1/csv/'+Path(path).name
    expected=manifest.get(key)
    if not expected or sha_file(path)!=expected:raise ValueError('ARCHIVED_SYNTHEA_CSV_HASH_MISMATCH')
    return expected


def resolve_source_file(name):
    original=Path(name)
    if original.is_absolute():
        if not original.is_file():raise FileNotFoundError(name)
        return original.resolve()
    if original.drive or '..' in original.parts:raise ValueError('UNSAFE_SOURCE_RELATIVE_PATH')
    relative=original.as_posix()
    candidates=[ROOT/original]
    if relative.startswith('experiments/'):
        candidates.append(SYNTHEA/original)
    present=[p.resolve() for p in candidates if p.is_file()]
    if not present:raise FileNotFoundError('EXACT_SOURCE_NOT_FOUND: '+relative)
    if relative.startswith(COHORT):
        ms=MANIFEST.stat()
        for path in present:
            st=path.stat()
            _checked_hash(str(path),st.st_size,st.st_mtime_ns,str(MANIFEST),ms.st_size,ms.st_mtime_ns)
    elif len(present)>1 and len({sha_file(p) for p in present})!=1:
        raise ValueError('AMBIGUOUS_SOURCE_PATH_DIFFERENT_BYTES')
    return present[0]
