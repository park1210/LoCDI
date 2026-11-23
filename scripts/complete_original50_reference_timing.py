# -*- coding: utf-8 -*-
"""Private evaluation-only Condition lookup, never an agent input or grant."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import csv,json,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,native_index,read,validate_csv_claims
from app.locdi.source_paths import resolve_source_file,sha_file,COHORT
from app.locdi.synthea_diagnosis_audit import reference_time
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
NS='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS

def main():
    cases=original_cases();idx=native_index();wanted={};found=defaultdict(list)
    for n,c in enumerate(cases,1):
        for x in c['evaluation']['observed_index_diagnoses']:wanted[c['patient_id'],c['agent_visible']['index_encounter_id'],str(x['code'])]=n
    relative=COHORT+'conditions.csv';path=resolve_source_file(relative);claims=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for line,r in enumerate(csv.DictReader(f),2):
            n=wanted.get((r['PATIENT'],r['ENCOUNTER'],r['CODE']))
            if n is not None:found[n].append((line,r));claims.append((relative,line,r))
    if set(found)!=set(range(1,51)):raise ValueError('SOURCE_REFERENCE_ROW_MISSING')
    out=[];counts=Counter()
    for n,c in enumerate(cases,1):
        raw=read(idx[n]['raw_path'])
        if sha_file(idx[n]['raw_path'])!=idx[n]['declared_raw_sha256']:raise ValueError('NATIVE_REFERENCE_SOURCE_HASH')
        enc=next(e for e in raw['record']['encounters'] if e.get('uuid')==c['agent_visible']['index_encounter_id'])
        entries=[]
        for line,r in found[n]:
            native=[d for d in enc.get('conditions',[]) if any(str(x['code'])==r['CODE'] for x in d.get('codes',[]))]
            native_times=[datetime.fromtimestamp(d['start']/1000,timezone.utc).isoformat() for d in native]
            if not native_times or not all(t[:10]==r['START'][:10] for t in native_times):raise ValueError('NATIVE_CSV_REFERENCE_DATE_MISMATCH')
            timing=reference_time(r['START'],c['agent_visible']['index_encounter_start']);counts[timing['status']]+=1
            entries.append({'code':r['CODE'],'description':r['DESCRIPTION'],'condition_started_at':r['START'],
                'source_file':relative,'source_row':line,'source_row_fingerprint':digest(r),'native_json_starts':native_times,
                'clinical_gold_approved':False,'actual_clinical_available_at':'NOT_RECORDED',**timing})
        out.append({'ordinal':n,'timing':entries})
    validation=validate_csv_claims(claims);reference_document={'rows':out,'validation':validation,
        'note':'Case source_event_store intentionally omits Conditions; timing must come from private original Condition CSV/JSON, not from the public event store.'}
    target=PR/'reference_timing_complete.json'
    if target.exists():
        if read(target)!=reference_document:raise ValueError('REFERENCE_SOURCE_CHANGED_NO_OVERWRITE')
    else:atomic_json(target,reference_document)
    # Complete only this new audit's originally blank timing cells.
    lookup={r['ordinal']:r['timing'] for r in out}
    p=PR/'ORIGINAL50_PATIENT_MATRIX.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:r['reference_timing']=json.dumps(lookup[int(r['patient_ordinal'])],ensure_ascii=False)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    main=PR/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md';text=main.read_text(encoding='utf-8');parts=re.split(r'(?=^## Patient \d+$)',text,flags=re.M)
    for j in range(1,len(parts)):
        n=int(re.search(r'^## Patient (\d+)',parts[j]).group(1));value=json.dumps(lookup[n],ensure_ascii=False)
        parts[j]=re.sub(r'Reference 시점: `.*?`\.',lambda _:'Reference 시점: `'+value+'`.',parts[j],count=1)
    main.write_text(''.join(parts),encoding='utf-8')
    full=read(PR/'analysis_full.json');full['reference_timing']=out
    for r in full['patient_rows']:r['reference_timing']=json.dumps(lookup[r['patient_ordinal']],ensure_ascii=False)
    atomic_json(PR/'analysis_with_reference_timing.json',full)
    coverage=read(PUB/'final_coverage.json');coverage['reference_timing_record_counts']=dict(counts);atomic_json(PUB/'final_coverage_with_reference_timing.json',coverage)
    report={'patients_with_source_reference_timing':len(out),'reference_rows':sum(counts.values()),'timing_record_counts':dict(counts),
        'current_public_event_store_contains_reference_conditions':False,'CSV_native_reference_date_consistency':'PASS',
        'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED','LLM_generation':0}
    atomic_json(PUB/'reference_timing_summary.json',report)
    doc=ROOT/'docs/B0_B4_실험/17_Original50_환자별_진단증상검사_분석/SUMMARY_AND_COVERAGE.md'
    doc.write_text(doc.read_text(encoding='utf-8')+'\n## 평가 전용 Reference 시점 보완\n\nCase의 공개용 source_event_store에는 Condition이 의도적으로 포함되지 않아, 별도 원본 conditions.csv와 동일 방문의 Native JSON Condition을 직접 교차 확인했다. 모델 입력에 추가하지 않았다.\n\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n\n위의 최초 timing 빈 집계는 이 실제 원천 조회 결과로 완성했다. CSV의 날짜와 JSON 시작일이 같아도 intraday 공개 가능성 또는 임상 Reference의 적격성을 자동 승인하지 않는다.\n',encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
