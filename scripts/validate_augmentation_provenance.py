"""Validate new augmentation provenance without overwriting initial artifacts."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
NS='ORIGINAL50_LITERATURE_AUGMENTATION_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
def pointer(obj,path):
    for key in path.lstrip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~');obj=obj[int(key)] if isinstance(obj,list) else obj[key]
    return obj
def main():
    cases={r['case_ordinal']:r for r in (json.loads(s) for s in (PR/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    with (PR/'AUGMENTATION_PROVENANCE.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    offsets={};source_cache={};repaired=[]
    for row in rows:
        n=int(row['case_ordinal']);case=cases[n]
        if row['item_type']=='SOURCE_RECOVERED':
            file=row['source_file']
            if file not in source_cache:source_cache[file]=read(file)
            event=pointer(source_cache[file],row['source_pointer'])
            fact=next(f for f in case['source_recovered_findings'] if f['finding_id']==row['item_id'])
            if digest(event)!=fact['source']['event_fingerprint']:raise ValueError('RAW_SOURCE_EVENT_POINTER_MISMATCH')
        else:
            offset=offsets.get(n,0);s=case['literature_informed_synthetic']
            path='/literature_informed_synthetic/chief_complaint' if offset==0 else '/literature_informed_synthetic/symptoms/'+str(offset-1) if offset<=len(s['symptoms']) else '/literature_informed_synthetic/candidate_tests/'+str(offset-1-len(s['symptoms']))
            if digest(pointer(case,path))!=row['source_sha256']:raise ValueError('AUTHORED_ITEM_FINGERPRINT')
            if row['source_pointer']!=path:repaired.append({'case_ordinal':n,'item_id':row['item_id'],'old_pointer':row['source_pointer'],'validated_pointer':path,
                'reason':'INITIAL_PROVENANCE_POINTER_USED_FLAT_POSITION_NOT_AN_OBJECT_FIELD','old_file_preserved':True})
            row['source_pointer']=path;offsets[n]=offset+1
    with (PR/'AUGMENTATION_PROVENANCE_VALIDATED.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (PR/'REJECTED_PROVENANCE_DRAFT_ROWS.jsonl').open('x',encoding='utf-8') as f:
        for row in repaired:f.write(json.dumps(row)+'\n')
    report={'status':'PASS','validated_provenance_rows':len(rows),'source_event_pointers_resolved':sum(r['item_type']=='SOURCE_RECOVERED' for r in rows),
        'authored_item_pointers_resolved':sum(r['item_type']=='LITERATURE_INFORMED_SYNTHETIC' for r in rows),
        'initial_pointer_fields_corrected_in_new_file':len(repaired),'initial_file_preserved':True,
        'use_file':'AUGMENTATION_PROVENANCE_VALIDATED.csv','no_patient_finding_or_reference_changed':True,'LLM_generation':0,'utility_fitting':0}
    atomic_json(PUB/'provenance_validation.json',report);print(json.dumps(report))
if __name__=='__main__':main()
