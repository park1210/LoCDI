"""Read-only Original50 source integrity. No retrieval ranking or clinical labels."""
from collections import defaultdict
from pathlib import Path
import csv,json,sqlite3
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_loader import load_cases
from app.locdi.source_paths import resolve_source_file,sha_file,MANIFEST
from app.locdi.routing_data_reuse import DATABASE
from app.core.longitudinal_memory import timestamp

DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
ENRICHMENT=ROOT/'outputs/private/research_evaluation/c4_05_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def original_cases():
    manifest=read(ENRICHMENT/'dataset_manifest.json')
    if sha_file(DATA)!=manifest['original_dataset_sha256']:raise ValueError('ORIGINAL50_HASH_CHANGED')
    cases=load_cases(DATA)
    if len(cases)!=50 or len({c['patient_id'] for c in cases})!=50:raise ValueError('ORIGINAL50_SCOPE')
    return cases

def native_index():
    manifest=read(ENRICHMENT/'dataset_manifest.json');p=ENRICHMENT/'case_source_index.jsonl'
    if sha_file(p)!=manifest['files'][p.name]['sha256']:raise ValueError('SOURCE_INDEX_HASH')
    return {r['ordinal']:r for r in (json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()) if r['dataset']=='Original50'}

def validate_csv_claims(claims):
    """Original archived manifest + exact row equality, grouped into one CSV scan."""
    grouped=defaultdict(dict);paths={}
    for name,n,row in claims:
        path=resolve_source_file(name);paths[name]=path
        if n in grouped[path] and grouped[path][n]!=row:raise ValueError('CONFLICTING_ROW_CLAIMS')
        grouped[path][n]=row
    checked=0
    for path,wanted in grouped.items():
        found={};last=max(wanted)
        with path.open(encoding='utf-8-sig',newline='') as f:
            for n,row in enumerate(csv.DictReader(f),2):
                if n in wanted:found[n]=row
                if n>=last:break
        if found!=wanted:raise ValueError('ORIGINAL_RAW_CSV_ROW_MISMATCH')
        checked+=len(found)
    return {'unique_rows':checked,'files':{str(p):sha_file(p) for p in grouped},'manifest_sha256':sha_file(MANIFEST)}

def memory_source_map(cases):
    """Source verification only. Future clinical availability is never inferred."""
    records=[r for c in cases for v in c['agent_visible']['past_visits'] for r in v['records']]
    ids=list({r['source_id'] for r in records});origins={}
    con=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        con.execute('PRAGMA query_only=ON')
        for start in range(0,len(ids),400):
            part=ids[start:start+400];q=','.join('?' for _ in part)
            origins.update((key,json.loads(payload)) for key,payload in con.execute('SELECT rid,payload FROM memory WHERE rid IN ('+q+')',part))
    finally:con.close()
    claims=[];result={};time_fields={'conditions':'START','observations':'DATE','procedures':'START','medications':'START','allergies':'START','careplans':'START','imaging_studies':'DATE','immunizations':'DATE','devices':'START','supplies':'DATE'}
    for r in records:
        o=origins.get(r['source_id'])
        if o is None:
            result[r['memory_id']]={'status':'MISSING_SOURCE','record_fingerprint':digest(r)};continue
        meta=o['_source'];raw=meta['source_record'];field=time_fields.get(o['type']);description=raw.get('DESCRIPTION') or raw.get('BODYSITE_DESCRIPTION') or raw.get('MODALITY_DESCRIPTION','')
        normalized=' | '.join(str(x) for x in (raw.get(field,''),o['type'],description,raw.get('VALUE',''),raw.get('UNITS','')))
        valid=(o['patient_id']==raw.get('PATIENT')==r['patient_id'] and o['encounter_id']==raw.get('ENCOUNTER')==r['visit_id']
          and o['normalized_text']==r['text']==normalized and o['record_id']==r['source_id']
          and o['observed_at_upper_bound']==r['observed_at'] and o['available_at']==r['available_at'])
        if not valid:raise ValueError('MEMORY_SOURCE_VALUE_OR_IDENTITY_MISMATCH')
        claims.append((meta['source_file'],meta['source_row'],raw))
        result[r['memory_id']]={'status':'VERIFIED_EXACT_CSV_SOURCE','record_fingerprint':digest(r),'source_id':r['source_id'],
            'source_file':meta['source_file'],'source_row':meta['source_row'],'source_row_fingerprint':digest(raw),
            'source_type':o['type'],'code':o['code'],'description':o['description'],'value':o['value'],'unit':o['units'],
            'recorded_at':o['timestamp'],'available_at':o['available_at'],'availability_basis':o['availability_basis'],
            'clinical_release_review':'NOT_REVIEWED'}
    csv_report=validate_csv_claims(claims)
    return result,{'records':len(records),'verified':sum(v['status']=='VERIFIED_EXACT_CSV_SOURCE' for v in result.values()),
        'missing':sum(v['status']=='MISSING_SOURCE' for v in result.values()),'csv':csv_report,
        'database_path':str(DATABASE),'database_sha256':sha_file(DATABASE),'dataset_sha256':sha_file(DATA)}

def eligible(record,identity):
    cutoff=timestamp(identity['started_at'])
    return (record['patient_id']==identity['patient_id'] and record['visit_id']!=identity['visit_id']
        and timestamp(record['visit_started_at'])<cutoff and timestamp(record['observed_at'])<=timestamp(record['available_at'])<=cutoff)
