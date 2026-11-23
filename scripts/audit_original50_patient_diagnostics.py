"""Independent Original50 source/trace inventory. No new retriever, model or fitting."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,native_index,memory_source_map,validate_csv_claims,eligible,read,DATA
from app.locdi.source_paths import sha_file
from app.locdi.llm_setup import digest
from app.locdi.common_prediction_contract import extract_core_prediction
from app.locdi.synthea_diagnosis_audit import reference_time,proxy_minute
from app.core.longitudinal_memory import timestamp

NS='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1'
PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
DOC=ROOT/'docs'/'B0_B4_\uc2e4\ud5d8'/'17_Original50_\ud658\uc790\ubcc4_\uc9c4\ub2e8\uc99d\uc0c1\uac80\uc0ac_\ubd84\uc11d'
def put(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def table(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def md(value):return str(value).replace('|','\\|').replace('\n',' ')

def baseline(name):
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1'/name
    p=read(ROOT/'outputs/research_evaluation/B34_CONNECTION_v1'/name/'plan.json');index=read(base/'case_index.json');run=base/p['run_id']
    out={};refs=read(base/'panel.json')
    for row in read(run/'collection.json')['rows']:
        folder=run/'requests'/('Episode%04d'%row['episode_ordinal']);before=sorted(folder.glob('Request*_before.json'))
        ds=read(folder/'actual_core_decisions.json') if (folder/'actual_core_decisions.json').exists() else []
        pred=extract_core_prediction(ds,row['status'].lower());obs={};payload={}
        if before:payload=json.loads(read(before[-1])['request']['messages'][1]['content']);obs=payload['observation']
        usage=Counter()
        for ap in folder.glob('Request*_after.json'):
            e=read(ap).get('response_envelope',{});usage.update(prompt=e.get('prompt_eval_count',0),output=e.get('eval_count',0))
        original=index[row['case_ref']];reference=refs[row['patient_ordinal']-1]['reference']
        out[original,row['condition']]={'plan_id':name,'plan_fingerprint':p['fingerprint'],'panel_ordinal':row['patient_ordinal'],
            'status':row['status'],'prediction':pred,'reference':reference,'observation':obs,'last_payload':payload,
            'memory':obs.get('longitudinal_memory',{}).get('records',[]),'decisions':ds,'tokens':dict(usage),
            'source_exact':pred['diagnosis'] in reference['accepted_diagnoses'] if pred['diagnosis'] else None,
            'source_binding_fingerprint':p.get('source_binding_fingerprint'),'model':p['model_identifier'],
            'code_pins':p['code_hashes'],'runtime_contract':p.get('terminal_contract'),
            'output_schema_fingerprint':digest(read(before[-1])['request']['format']) if before else None,
            'request_pointer':str(before[-1].relative_to(ROOT)) if before else None}
    return out

def run():
    global DOC
    if DOC.exists():
        i=2
        while DOC.with_name(DOC.name+'_v'+str(i)).exists():i+=1
        DOC=DOC.with_name(DOC.name+'_v'+str(i))
    DOC.mkdir(parents=True);cases=original_cases();idx=native_index();memorymap,source_report=memory_source_map(cases)
    put(PR/'memory_source_map.json',memorymap);put(PR/'memory_source_verification.json',source_report)
    old=baseline('recovery_llm_all_001');new=baseline('runtime_recovery_all_001');put(PR/'baseline_trace_details.json',{'old':[{'original_ordinal':k[0],'condition':k[1],**v} for k,v in old.items()],
        'latest':[{'original_ordinal':k[0],'condition':k[1],**v} for k,v in new.items()]})
    m5=read(ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json')
    test_keys={(json.loads(k)[2],json.loads(k)[3]) for k in m5['provenance']['action_key_mapping'] if json.loads(k)[0]=='Test'}
    alltests=[];symptoms=[];matrix=[];regression=[];diseases=Counter();departments=Counter();classification=Counter();claims=[];privatepatients=[]
    main=PR/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md'
    with main.open('x',encoding='utf-8') as f:
        f.write('# Original50 환자별 진단·증상·검사·과거 Memory 통합 문서 — PRIVATE\n\n원본50명만 포함한다. 합성 Routing15명은 포함하지 않는다. Source 진단은 임상 Gold 승인이 아니다. 공개시각은 별도 검토되지 않은 Proxy와 실제 저장 Trace를 구분한다. 아래 과거 기록은 모두 현재 Evidence가 아니다.\n\n')
        for ordinal,c in enumerate(cases,1):
            av=c['agent_visible'];identity=av['longitudinal'];patient=c['patient_id'];visit=av['index_encounter_id'];start=av['index_encounter_start'];indexms=int(timestamp(start).timestamp()*1000)
            ni=idx[ordinal];path=Path(ni['raw_path'])
            if sha_file(path)!=ni['declared_raw_sha256']:raise ValueError('NATIVE_JSON_HASH')
            raw=read(path)
            if raw['attributes'].get('id')!=patient or ni['patient_id']!=patient or ni['visit_id']!=visit:raise ValueError('NATIVE_PATIENT_BINDING')
            encs=[(j,e) for j,e in enumerate(raw['record']['encounters']) if e.get('uuid')==visit]
            if len(encs)!=1:raise ValueError('NATIVE_ENCOUNTER_BINDING')
            native_i,native=encs[0];records=[r for v in av['past_visits'] for r in v['records']];good=[r for r in records if eligible(r,identity)]
            if len(good)!=len(records):raise ValueError('ORIGINAL_MEMORY_BOUNDARY')
            evs=c['environment']['source_event_store'];labs=[];conditions=[]
            for e in evs:
                if e['patient_id']!=patient or e['encounter_id']!=visit:raise ValueError('CURRENT_SOURCE_IDENTITY')
                meta=e['source_metadata'];claims.append((meta['source_file'],meta['source_row'],meta['source_record']))
                if e['event_type']=='conditions':conditions.append(e)
                if e['event_type']!='observations':continue
                code=str(e['target']['code']);value=e['evidence'].get('value');unit=e['evidence'].get('units');minute=proxy_minute(e,indexms)
                labs.append({'patient_ordinal':ordinal,'scope':'CURRENT_ENCOUNTER','code':code,'description':e['target']['description'],'value':value,'unit':unit,
                    'source_event':e['event_id'],'source_file':meta['source_file'],'source_row':meta['source_row'],'recorded_at':e['timestamp'],
                    'recorded_proxy_available_minute':minute,'actual_clinical_available_at':'NOT_RECORDED','M5_Test_mappable':(code,unit) in test_keys,
                    'kind':'PATIENT_REPORT_ASK_ONLY' if code=='72514-3' else 'RECORDED_OBSERVATION',
                    'B0_B4_receipts':';'.join(co for co in ('B0','B1','B2','B3','B4') if (ordinal,co) in new and any(
                        m.get('code')==code and m.get('unit')==unit and str(m.get('value'))==str(value) for m in new[ordinal,co]['last_payload'].get('released_measurements',[]))),
                    'exposure':'NOT_AVAILABLE' if value in (None,'',[]) else 'NOT_RELEASED','cutoff_status':'UNKNOWN' if minute is None else 'AFTER_45_MIN_PROXY_CUTOFF' if minute>45 else 'PROXY_CUTOFF_ELIGIBLE_NOT_CLINICAL_APPROVAL'})
                if labs[-1]['B0_B4_receipts']:labs[-1]['exposure']='RECEIVED_IN_SAVED_LLM_PAYLOAD'
            for r in good:
                o=memorymap[r['memory_id']]
                if o['status']=='VERIFIED_EXACT_CSV_SOURCE' and o['source_type']=='observations':
                    labs.append({'patient_ordinal':ordinal,'scope':'HISTORICAL_MEMORY','code':o['code'],'description':o['description'],'value':o['value'],'unit':o['unit'],
                        'source_event':r['source_id'],'source_file':o['source_file'],'source_row':o['source_row'],'recorded_at':r['observed_at'],
                        'recorded_proxy_available_minute':None,'actual_clinical_available_at':'NOT_RECORDED','M5_Test_mappable':(str(o['code']),o['unit']) in test_keys,
                        'kind':'PATIENT_REPORT_ASK_ONLY' if str(o['code'])=='72514-3' else 'RECORDED_OBSERVATION','B0_B4_receipts':'',
                        'exposure':'HISTORICAL_NOT_CURRENT_EVIDENCE','cutoff_status':'ELIGIBLE_PAST_MEMORY'})
            labs.sort(key=lambda x:(x['scope'],x['recorded_at'],x['code'],str(x['source_row'])))
            alltests.extend(labs);labels=c['evaluation']['diagnosis_labels'];depts=c['evaluation']['accepted_departments_source_policy'];diseases.update(labels);departments.update(depts)
            codes={str(x['code']) for x in c['evaluation']['observed_index_diagnoses']};refs=[e for e in conditions if str(e['target']['code']) in codes]
            timing=[{'source_event':e['event_id'],'recorded_at':e['timestamp'],**reference_time(e['timestamp'],start)} for e in refs]
            doctor=av['current_initial_state']['doctor_visible_info'];info=av['current_initial_state']['patient_visible_info'];b2=new.get((ordinal,'B2'));b1=new.get((ordinal,'B1'));oldb2=old.get((ordinal,'B2'))
            selected={r['memory_id'] for r in b2['memory']} if b2 else set();visible_codes={(m['code'],m['unit']) for m in b2['last_payload'].get('released_measurements',[])} if b2 else set()
            candidates=[r for r in good if r['memory_id'] not in selected and memorymap[r['memory_id']]['status']=='VERIFIED_EXACT_CSV_SOURCE' and
                (str(memorymap[r['memory_id']].get('code')),memorymap[r['memory_id']].get('unit')) in visible_codes]
            tags=['SOURCE_INFORMATION_MISSING'] if not doctor.get('chief_complaint') else []
            if any(x['scope']=='CURRENT_ENCOUNTER' and x['exposure']=='NOT_RELEASED' for x in labs):tags.append('INFORMATION_EXISTS_BUT_NOT_RELEASED')
            if candidates:tags.append('MEMORY_RETRIEVAL_MISS_CANDIDATE')
            if any((ordinal,co) in new and new[ordinal,co]['source_exact'] is False and new[ordinal,co]['observation'].get('objective_findings') for co in ('B0','B1','B2','B3','B4')):tags.append('INFORMATION_PROVIDED_BUT_DIAGNOSIS_MISMATCH')
            if any(t['status'] not in ('EARLIER_DAY',) for t in timing) or not timing:tags.append('REFERENCE_TIME_AMBIGUOUS')
            tags.append('INSUFFICIENT_EVIDENCE_TO_CLASSIFY');classification.update(tags)
            pain=sum(str(memorymap[r['memory_id']].get('code'))=='72514-3' for r in b2['memory']) if b2 else 0
            symptoms.append({'patient_ordinal':ordinal,'source_chief_complaint':doctor.get('chief_complaint') or 'NOT_RECORDED','symptom_narrative':'NOT_RECORDED',
                'initial_associated_symptoms':json.dumps(info.get('associated_symptoms',[]),ensure_ascii=False),'pain_instrument_is_not_complaint':True,
                'current_pain_source_rows':sum(x['scope']=='CURRENT_ENCOUNTER' and x['kind']=='PATIENT_REPORT_ASK_ONLY' for x in labs),
                'past_pain_source_rows':sum(str(memorymap[r['memory_id']].get('code'))=='72514-3' for r in good),
                'raw_export_symptom_state_terms':len(raw.get('symptoms',{})),'raw_symptom_state_visibility':'NOT_POINT_IN_TIME_REVIEWED_NOT_AGENT_INPUT'})
            row={'patient_ordinal':ordinal,'age':info.get('age','UNKNOWN'),'sex':info.get('sex','UNKNOWN'),'index_encounter_start':start,'evaluation10':bool((ordinal,'B0') in new),
                'source_reference_diagnoses':';'.join(labels),'source_reference_departments':';'.join(depts),'reference_timing':json.dumps(timing),
                'past_visits':len(av['past_visits']),'eligible_memory':len(good),'current_observation_rows':sum(x['scope']=='CURRENT_ENCOUNTER' for x in labs),
                'historical_observation_rows':sum(x['scope']=='HISTORICAL_MEMORY' for x in labs),'latest_B2_memory_count':len(selected),'latest_B2_pain_count':pain,
                'unselected_shared_measurement_candidates':len(candidates),'candidate_clinical_relevance':'POTENTIALLY_RELEVANT_UNREVIEWED','quality_tags':';'.join(tags)}
            for co in ('B0','B1','B2','B3','B4'):
                t=new.get((ordinal,co));row[co+'_status']=t['status'] if t else 'NOT_RUN';row[co+'_diagnosis']=t['prediction']['diagnosis'] if t else None;row[co+'_source_exact']=t['source_exact'] if t else None
            matrix.append(row)
            if b2:
                regression.append({'original_ordinal':ordinal,'panel_ordinal':b2['panel_ordinal'],'old_plan':oldb2['plan_id'],'new_plan':b2['plan_id'],
                  'same_contract':False,'old_source_exact':oldb2['source_exact'],'latest_source_exact':b2['source_exact'],
                  'old_diagnosis':oldb2['prediction']['diagnosis'],'latest_diagnosis':b2['prediction']['diagnosis'],
                  'CORRECT_TO_INCORRECT':oldb2['source_exact'] is True and b2['source_exact'] is False,
                  'old_memory_ids':json.dumps([r['memory_id'] for r in oldb2['memory']]),'latest_memory_ids':json.dumps(list(selected)),
                  'old_source_results_equal_latest':set(oldb2['observation'].get('objective_findings',[]))==set(b2['observation'].get('objective_findings',[])),
                  'latest_omitted_matching_measurement_records':len(candidates),'cause':'OBSERVED_DIFFERENCES_NOT_CAUSAL_PROOF'})
            privatepatients.append({'ordinal':ordinal,'native_file':str(path),'native_sha256':ni['declared_raw_sha256'],
                'native_current_encounter_pointer':'/record/encounters/'+str(native_i),'reference_timing':timing,'raw_export_symptom_state':raw.get('symptoms'),
                'omitted_candidate_ids':[r['memory_id'] for r in candidates],'quality_tags':tags})
            f.write('\n## Patient %02d\n\n'%ordinal);f.write('**Source 진단명:** '+md('; '.join(labels))+'\n\n**Source 진료과:** '+md('; '.join(depts))+'\n\n')
            f.write('나이/성별: %s / %s; index: %s; 평가10명 포함: %s.\n\n'%(row['age'],row['sex'],start,row['evaluation10']))
            f.write('### 증상\n\n- 실제 Chief Complaint: NOT_RECORDED\n- 실제 초기 증상 narrative: NOT_RECORDED\n- 초기 associated_symptoms는 통증 척도 항목명이며 통증 호소·위치·원인으로 해석하지 않음.\n- 현재 통증 Source 행: %d; 과거 통증 행: %d.\n'%(symptoms[-1]['current_pain_source_rows'],symptoms[-1]['past_pain_source_rows']))
            f.write('- Raw export symptom state: %d개 term. index 시점 공개 여부가 확인되지 않아 현재/과거 증상으로 자동 승격하지 않음. 자세한 원문은 patient_source_details.json의 해당 ordinal.\n\n'%symptoms[-1]['raw_export_symptom_state_terms'])
            f.write('Reference 시점: `'+md(json.dumps(timing,ensure_ascii=False))+'`. 임상 공개 시각은 NOT_RECORDED.\n\n')
            f.write('### 현재 Encounter 검사·응답 Source\n\n|검사/관측|수치|단위|기록 시각|Proxy minute|M5 Test|실제 수신 조건|Source 위치|\n|---|---|---|---|---|---|---|---|\n')
            for x in labs:
                if x['scope']=='CURRENT_ENCOUNTER':f.write('|'+ '|'.join(md(x[k]) for k in ('description','value','unit','recorded_at','recorded_proxy_available_minute','M5_Test_mappable','B0_B4_receipts'))+'|'+md(x['source_file'])+':'+str(x['source_row'])+'|\n')
            f.write('\n현재 검사값이 Source에 있어도 실제 LLM 수신 여부는 별도다. 수신 조건이 빈 경우 NOT_RELEASED(미실행 환자는 NOT_RUN), cutoff 이후값은 모델 공개 후보가 아니다.\n\n')
            f.write('### 과거 병력·검사·처치 및 Memory 전체\n\n적격 기록 %d개 / 과거 방문 %d개. 아래는 원본 기록 전체이며 현재 Evidence가 아니다. 단위 변환이나 정상/비정상 판정은 하지 않았다.\n\n'%(len(good),len(av['past_visits'])))
            for r in sorted(good,key=lambda x:(timestamp(x['observed_at']),x['memory_id'])):
                o=memorymap[r['memory_id']];f.write('- '+md(r['text'])+'; observed='+r['observed_at']+'; available='+r['available_at']+'; Memory='+r['memory_id']+'; Source='+md(o.get('source_file','MISSING'))+':'+str(o.get('source_row','?'))+'\n')
            f.write('\n### 실제 B1/B2 제공 Memory\n\n')
            for name,t in [('최신 B1',b1),('최신 B2',b2),('과거 B2',oldb2)]:
                f.write('**'+name+'**: '+(t['status'] if t else 'NOT_RUN')+'\n\n')
                if t:
                    for r in t['memory']:f.write('- '+md(r['text'])+'; '+r['observed_at']+'; '+r['memory_id']+'\n')
            f.write('\nTop5 밖의 공개 측정과 동일 Code/Unit인 적격 과거 기록 후보: %d개. POTENTIALLY_RELEVANT_UNREVIEWED이며 임상 관련성 정답이 아니다.\n\n'%len(candidates))
            f.write('### 최신 B0~B4\n\n|조건|실행 상태|최종 진단|certainty|Source Exact|최종 판단 minute|\n|---|---|---|---|---|---|\n')
            for co in ('B0','B1','B2','B3','B4'):
                t=new.get((ordinal,co));f.write('|'+co+'|'+(t['status'] if t else 'NOT_RUN')+'|'+md(t['prediction']['diagnosis'] if t else 'NOT_RUN')+'|'+md(t['prediction']['certainty'] if t else 'NOT_RUN')+'|'+str(t['source_exact'] if t else 'NOT_EVALUATED')+'|'+str(t['decisions'][-1]['minute'] if t and t['decisions'] else 'NOT_RUN')+'|\n')
            f.write('\n품질 상태: '+', '.join(tags)+'. 진단 불일치의 임상적 원인이나 정보 충분성을 확정하지 않는다.\n')
    csv_current=validate_csv_claims(claims);put(PR/'current_source_verification.json',csv_current);put(PR/'patient_source_details.json',privatepatients)
    table(PR/'ORIGINAL50_PATIENT_MATRIX.csv',matrix);table(PR/'ORIGINAL50_TEST_RESULTS.csv',alltests);table(PR/'ORIGINAL50_SYMPTOM_AUDIT.csv',symptoms)
    distribution=[{'diagnosis':k,'patients':v,'denominator':50} for k,v in diseases.items()];table(PR/'ORIGINAL50_DISEASE_DISTRIBUTION.csv',distribution);table(PR/'B2_REGRESSION_CASE_COMPARISON.csv',regression)
    report={'patients_processed':len(matrix),'fixed_evaluation_patients':sum(r['evaluation10'] for r in matrix),'synthetic_pilot_patients_included':0,
        'memory_records':sum(r['eligible_memory'] for r in matrix),'source_verified_memory_records':source_report['verified'],
        'current_observation_rows':sum(r['scope']=='CURRENT_ENCOUNTER' for r in alltests),'historical_observation_rows':sum(r['scope']=='HISTORICAL_MEMORY' for r in alltests),
        'chief_complaint_recorded_patients':sum(r['source_chief_complaint']!='NOT_RECORDED' for r in symptoms),'narrative_symptoms_recorded_patients':0,
        'diagnosis_distribution':dict(diseases),'department_distribution':dict(departments),'quality_tags_patient_counts':dict(classification),
        'patients_with_unselected_matching_measurement_candidates':sum(r['unselected_shared_measurement_candidates']>0 for r in matrix),
        'unselected_matching_measurement_candidates':sum(r['unselected_shared_measurement_candidates'] for r in matrix),
        'old_B2_correct_panel_ordinals':[r['panel_ordinal'] for r in regression if r['old_source_exact'] is True],
        'B2_correct_to_incorrect_original_ordinals':[r['original_ordinal'] for r in regression if r['CORRECT_TO_INCORRECT']],
        'latest_B2_final_pain_records':sum(r['latest_B2_pain_count'] for r in matrix),'latest_B2_final_records':sum(r['latest_B2_memory_count'] for r in matrix),
        'actual_LLM_generation':0,'fitting':0,'memory_algorithm_changed':False,'clinical_accuracy':'NOT_EVALUATED','private_main_document':str(main.relative_to(ROOT))}
    put(PUB/'summary.json',report);put(PR/'analysis_full.json',{'summary':report,'patient_rows':matrix,'reference_timing':[{'ordinal':r['ordinal'],'timing':r['reference_timing']} for r in privatepatients]})
    (PR/'B2_MEMORY_RETRIEVAL_AUDIT.md').write_text('# Private Memory 检查\n\n原始记录全部保留，完整内容见主文档。'+json.dumps(regression,ensure_ascii=False,indent=2),encoding='utf-8')
    (PR/'DATA_QUALITY_FINDINGS.md').write_text('# Private data quality\n\n'+json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (PR/'README.md').write_text('# PRIVATE Original50 audit\n\nMain: ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md. All patient Source/Reference details stay here; no model/fitting/retrieval changes.',encoding='utf-8')
    (DOC/'README.md').write_text('# Original50 환자별 Source 통합 감사\n\nPublic은 집계만 포함한다. 실제 환자별 Reference·검사 수치·전체 적격 과거 기록을 포함한 메인 문서는 `'+str(main.relative_to(ROOT))+'`이다.\n\nGeneration0/fitting0. Synthetic Pilot15명은 포함하지 않았다.\n\n기계 집계: `'+str((PUB/'summary.json').relative_to(ROOT))+'`.\n',encoding='utf-8')
    (DOC/'DATA_QUALITY_FINDINGS.md').write_text('# 비식별 데이터 품질 집계\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\n태그는 기계적 사실/검토 후보이며 독립 임상적 원인 판정이 아니다. 모든 환자의 정보 충분성은 미검토다. 원본과 실제 모델의 입력·판단 차이는 Private baseline_trace_details.json 및 회귀 CSV에 보존했다.\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))
if __name__=='__main__':run()
