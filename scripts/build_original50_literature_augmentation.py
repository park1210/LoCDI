# -*- coding: utf-8 -*-
"""Build isolated authored development candidates. No model client or fitting."""
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,native_index,memory_source_map,validate_csv_claims,read,sha_file,DATA
from app.locdi.source_paths import SYNTHEA,COHORT,MANIFEST
from app.locdi.original50_augmentation import VERSION,SCOPE,recover_symptom_events,scenario,validate_augmentation,initial_preview
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.synthea_diagnosis_audit import proxy_minute
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json

NS='ORIGINAL50_LITERATURE_AUGMENTATION_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
DOC=ROOT/'docs/B0_B4_실험/19_Original50_문헌기반_데이터보강'
M5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
POLICY={'version':VERSION,'scope':SCOPE,'seed':20261122,'population':'ALL_ORIGINAL50_IN_ORIGINAL_ORDER',
    'label_conditioning':'SOURCE_REFERENCE_USED_ONLY_FOR_PRIVATE_LITERATURE_PROFILE_ASSIGNMENT',
    'outcome_or_error_conditioning':False,'clinical_reviewed':False,'clinical_diagnosability':'UNREVIEWED',
    'generation_method':'DETERMINISTIC_AUTHORED_FEATURE_COMPOSITION_NO_OLLAMA',
    'symptom_count':'0 to 2 selected verified feature candidates; sparse/no-current-report permitted',
    'numeric_synthetic_tests':0,'past_history_genetics_fractures_onset_age_generated':False,
    'source_recovery':'positive timed native symptom events with unique recorded encounter-interval linkage; no final-state carry-forward',
    'source_cutoff_minutes':45,'clinical_source_grants_created':0,'native_runtime_registered':False,
    'LLM_generation':0,'utility_fitting':0,'official_B0_B4_unchanged':True}

def put_once_or_equal(path,document):
    # Non-generating build stages may be rechecked after a technical failure.
    # An existing different artifact is never replaced or merged.
    if path.exists():
        if read(path)!=document:raise ValueError('EXISTING_BUILD_ARTIFACT_DIFFERS_NO_OVERWRITE: '+path.name)
    else:atomic_json(path,document)

def table(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsonl(p,rows):
    with p.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
def md(v):return str(v).replace('|','\\|').replace('\n',' ')

def candidate_test(t,events,model,indexms):
    matching=[e for e in events if e['event_type']=='observations' and str(e['target']['code']) in t['matching_source_codes']]
    eligible=[e for e in matching if proxy_minute(e,indexms) is not None and 0<=proxy_minute(e,indexms)<=45 and e['evidence'].get('value') not in ('',None)]
    keys={(json.loads(k)[2],json.loads(k)[3]) for k in model['provenance']['action_key_mapping'] if json.loads(k)[0]=='Test'}
    mapped=[e for e in eligible if (str(e['target']['code']),e['evidence'].get('units')) in keys]
    status='CURRENT_M5_COMPATIBLE' if mapped else 'SOURCE_EXISTS_BUT_ACTION_UNAVAILABLE' if matching else 'LITERATURE_RECOMMENDED_NOT_OBSERVED'
    return {**deepcopy(t),'origin':'LITERATURE_INFORMED_SYNTHETIC','item_kind':'RECOMMENDATION_NOT_A_TEST_RESULT',
        'action_compatibility':status,'observed_source_event_ids':[e['event_id'] for e in matching],
        'eligible_source_event_ids':[e['event_id'] for e in eligible],'M5_compatible_source_event_ids':[e['event_id'] for e in mapped],
        'result_generated':False,'clinical_reviewed':False,'execution_release_approved':False,
        'new_action_or_likelihood_created':False,'recommendation_time_basis':'AUTHORED_LOCAL_MINUTE_0_NOT_CLINICAL_ORDER'}

def build():
    global DOC
    if DOC.exists():
        n=2
        while DOC.with_name(DOC.name+'_v'+str(n)).exists():n+=1
        DOC=DOC.with_name(DOC.name+'_v'+str(n))
    # Freeze the authoring policy before using references or creating any patient augmentation.
    put_once_or_equal(PUB/'authoring_policy.json',{**POLICY,'fingerprint':digest(POLICY)})
    catalog=evidence_catalog();put_once_or_equal(PUB/'disease_evidence_catalog.json',catalog)
    cases=original_cases();idx=native_index();model=read(M5);archive=read(MANIFEST)
    if {c['disease'] for c in catalog['cards']}!=set(model['diagnoses']):raise ValueError('SAME_FIFTEEN_CATALOG_REQUIRED')
    for source in catalog['sources'].values():
        if urlparse(source['url']).scheme!='https' or source['verification']!='PRIMARY_PAGE_CONTENT_OBSERVED':raise ValueError('UNVERIFIED_EXTERNAL_SOURCE')
    source_map,source_report=memory_source_map(cases);put_once_or_equal(PR/'memory_source_verification.json',source_report)
    put_once_or_equal(PR/'original_memory_source_map.json',source_map)
    prior=ROOT/'outputs/private/research_evaluation/ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1'
    timing=read(prior/'reference_timing_complete.json');reference={r['ordinal']:r['timing'] for r in timing['rows']}
    # Recheck exact reference CSV rows; do not promote a Source label into clinical Gold.
    condition_path=SYNTHEA/COHORT/'conditions.csv';wanted={t['source_row']:t['source_row_fingerprint'] for r in timing['rows'] for t in r['timing']};condition_claims=[]
    with condition_path.open(encoding='utf-8-sig',newline='') as f:
        for line,r in enumerate(csv.DictReader(f),2):
            if line in wanted:
                if digest(r)!=wanted[line]:raise ValueError('SOURCE_REFERENCE_ROW_CHANGED')
                condition_claims.append((COHORT+'conditions.csv',line,r))
    if len(condition_claims)!=len(wanted):raise ValueError('REFERENCE_ROWS_MISSING')
    condition_check=validate_csv_claims(condition_claims);put_once_or_equal(PR/'reference_source_validation.json',condition_check)
    fhir=[]
    for p in ((SYNTHEA/COHORT).parent/'fhir').glob('*.json'):
        key='output/cohort_1000_json_v1/fhir/'+p.name
        if archive.get(key)!=sha_file(p):raise ValueError('FHIR_ARCHIVE_HASH')
        bundle=read(p);fhir.append({'path':str(p),'sha256':sha_file(p),'types':dict(Counter(e['resource']['resourceType'] for e in bundle.get('entry',[])))})
    put_once_or_equal(PR/'fhir_inventory.json',{'files':fhir,'patient_clinical_bundles_found':0,'no_unrelated_cohort_searched':True})
    source_claims=[];augmented=[];rejected=[];sourceaudit=[];provenance=[];missing=[];originalmemorycount=0;counts=Counter();used_chiefs=set();bylabel={c['disease']:c for c in catalog['cards']};nativepins={}
    for n,c in enumerate(cases,1):
        native=idx[n];path=Path(native['raw_path']);h=sha_file(path)
        if h!=native['declared_raw_sha256'] or archive.get('output/cohort_1000_json_v1/json/'+path.name)!=h:raise ValueError('NATIVE_JSON_ARCHIVED_HASH')
        nativepins[str(path)]=h;raw=read(path);av=c['agent_visible'];indexms=timestamp(av['index_encounter_start']).timestamp()*1000
        current=[e for e in raw['record']['encounters'] if e.get('uuid')==av['index_encounter_id']]
        if len(current)!=1 or raw['attributes']['id']!=c['patient_id']:raise ValueError('EXACT_CURRENT_ENCOUNTER_BINDING')
        recovered,bad=recover_symptom_events(c,raw,path,h)
        for b in bad:b['case_ordinal']=n
        rejected.extend(bad);diagnoses=c['evaluation']['diagnosis_labels']
        if len(diagnoses)!=1:raise ValueError('AMBIGUOUS_SOURCE_PROFILE_NOT_AUTOMATICALLY_ASSIGNED')
        card=bylabel[diagnoses[0]];synthetic=scenario(n,card,catalog,POLICY,used_chiefs=used_chiefs)
        events=c['environment']['source_event_store'];originals=[e for e in events if e['event_type']=='observations']
        for e in events:
            meta=e['source_metadata'];source_claims.append((meta['source_file'],meta['source_row'],meta['source_record']))
        synthetic['candidate_tests']=[candidate_test(catalog['tests'][key],events,model,indexms) for key in card['candidate_test_ids']]
        row={'version':VERSION,'case_ordinal':n,'original_case_fingerprint':digest(c),'original_dataset_sha256':sha_file(DATA),
            'source_reference':{'scope':'PRIVATE_ONLY_SOURCE_NOT_CLINICAL_GOLD','diagnoses':diagnoses,
                'departments':c['evaluation']['accepted_departments_source_policy'],'timing':reference[n],'clinical_reviewed':False},
            'private_original_identity':av['longitudinal'],'native_source':{'file':str(path),'file_sha256':h},
            'original_initial_state':deepcopy(av['current_initial_state']),'original_findings':deepcopy(originals),
            'original_history':{'past_visits':len(av['past_visits']),'fixed_memory_records':sum(len(v['records']) for v in av['past_visits']),
                'source_verified_memory_ids':[r['memory_id'] for v in av['past_visits'] for r in v['records']],
                'full_history_audit_pointer':str(prior/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md')+'#patient-%02d'%n,
                'full_history_audit_sha256':sha_file(prior/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md')},
            'source_recovered_findings':recovered,'literature_informed_synthetic':synthetic,
            'literature_profile_id':card['concept_id'],'literature_fingerprint':digest(catalog),
            'authoring_policy_fingerprint':digest(POLICY),'evaluation_scope':SCOPE,'clinical_reviewed':False,
            'clinical_diagnosability':'UNREVIEWED','clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False,
            'source_release_grants_created':0,'actual_LLM_generation':0,'utility_fitting':0,
            'confirmation_still_missing':card['confirmation_needed'],
            'provenance':{'original':'EXACT_ORIGINAL50_JSONL_AND_ARCHIVED_CSV_NATIVE_JSON',
                'recovered':'POSITIVE_TIMED_ENGINE_EVENT_WITH_DERIVED_UNIQUE_ENCOUNTER_INTERVAL',
                'synthetic':'AUTHORED_SYMPTOM_CANDIDATE_NOT_ORIGINAL_PATIENT_OBSERVATION',
                'literature_sources':card['sources'],'clinical_reviewed':False}}
        row['row_fingerprint']=digest(row);validate_augmentation(c,row,catalog,verify_source_files=True)
        # Only a preview; never registers Source events or admits an experiment.
        preview={'case_ordinal':n,'D0':initial_preview(c,row,catalog,view='D0'),
            'D1_PREVIEW_ONLY':initial_preview(c,row,catalog,view='D1',accept_development_proxy=True),
            'D2_PREVIEW_ONLY':initial_preview(c,row,catalog,view='D2',accept_development_proxy=True),
            'scope':SCOPE,'runtime_admitted':False,'clinical_reviewed':False}
        put_once_or_equal(PR/('Patient%02d_preview.json'%n),preview)
        current_recovered=[f for f in recovered if f['scope']=='CURRENT'];history_recovered=[f for f in recovered if f['scope']=='HISTORICAL']
        sourceaudit.append({'case_ordinal':n,'source_diagnosis':diagnoses[0],'source_department':';'.join(row['source_reference']['departments']),
            'source_reference_timing':json.dumps(reference[n]),'original_chief_complaint':av['current_initial_state']['doctor_visible_info'].get('chief_complaint') or 'NOT_RECORDED',
            'original_symptom_narrative':'NOT_RECORDED','current_original_observations':len(originals),'past_visits':len(av['past_visits']),
            'original_memory_records':row['original_history']['fixed_memory_records'],'current_recovered_symptom_events':len(current_recovered),
            'historical_recovered_symptom_events':len(history_recovered),'synthetic_symptom_items':len(synthetic['symptoms']),
            'synthetic_numeric_tests':0,'clinical_diagnosability':'UNREVIEWED','native_source_sha256':h})
        originalmemorycount+=row['original_history']['fixed_memory_records'];counts.update(original_patients=1,original_observation_rows=len(originals),
            recovered_facts=len(recovered),recovered_current=len(current_recovered),recovered_historical=len(history_recovered),
            synthetic_chief_complaints=1,synthetic_symptoms=len(synthetic['symptoms']),literature_test_recommendations=len(synthetic['candidate_tests']))
        counts['patients_with_recovered_facts']+=bool(recovered);counts['patients_with_current_recovery']+=bool(current_recovered)
        counts['synthetic_no_current_symptom_report_cases']+=not bool(synthetic['symptoms'])
        for f in recovered:provenance.append({'case_ordinal':n,'item_type':'SOURCE_RECOVERED','item_id':f['finding_id'],'scope':f['scope'],
            'source_file':f['source']['file'],'source_pointer':f['source']['json_pointer'],'source_sha256':f['source']['file_sha256'],
            'literature_urls':'','recorded_at':f['recorded_at'],'actual_clinical_available_at':'NOT_RECORDED','clinical_reviewed':False})
        authored_items=[('/chief_complaint',synthetic['chief_complaint'])]
        authored_items += [('/symptoms/'+str(i),item) for i,item in enumerate(synthetic['symptoms'])]
        authored_items += [('/candidate_tests/'+str(i),item) for i,item in enumerate(synthetic['candidate_tests'])]
        for item_pointer,item in authored_items:
            provenance.append({'case_ordinal':n,'item_type':'LITERATURE_INFORMED_SYNTHETIC','item_id':item.get('finding_id',item.get('test_id','chief_complaint')),
                'scope':'AUTHORED_NOT_ORIGINAL','source_file':'NOT_AN_ORIGINAL_SOURCE_RECORD','source_pointer':'/literature_informed_synthetic'+item_pointer,
                'source_sha256':digest(item),'literature_urls':';'.join(catalog['sources'][s]['url'] for s in item['sources']),
                'recorded_at':'AUTHORED_LOCAL_MINUTE_0','actual_clinical_available_at':'NOT_APPLICABLE_SYNTHETIC','clinical_reviewed':False})
        for t in synthetic['candidate_tests']:missing.append({'case_ordinal':n,'test_name':t['name'],'status':t['action_compatibility'],
            'source_exists':bool(t['observed_source_event_ids']),'native_action_added':False,'synthetic_result_generated':False,
            'diagnostic_requirement':card['confirmation_needed'],'clinical_diagnosability':'UNREVIEWED'})
        augmented.append(row)
    if originalmemorycount!=72590:raise ValueError('FIXED_MEMORY_PRESERVATION')
    csvcheck=validate_csv_claims(source_claims);atomic_json(PR/'current_source_verification.json',csvcheck);atomic_json(PR/'native_source_pins.json',nativepins)
    jsonl(PR/'AUGMENTED_CASES.jsonl',augmented);jsonl(PR/'REJECTED_AUGMENTATIONS.jsonl',rejected)
    table(PR/'ORIGINAL50_SOURCE_AUDIT.csv',sourceaudit);table(PR/'AUGMENTATION_PROVENANCE.csv',provenance);table(PR/'MISSING_INFORMATION_MATRIX.csv',missing)
    rejected_counts=Counter(r['reason'] for r in rejected);compat=Counter(r['status'] for r in missing)
    atomic_json(PR/'authoring_index.json',{'policy_fingerprint':digest(POLICY),'case_row_fingerprints':[r['row_fingerprint'] for r in augmented],
        'case_ordinal_scope':list(range(1,51)),'selection_used_evaluation_outcomes':False,'source_release_grants':0})
    summary={'status':'CANDIDATE_DATASET_BUILT_CLINICAL_AND_RUNTIME_NOT_ADMITTED','counts':dict(counts),
        'fixed_memory_preserved':originalmemorycount,'synthetic_numeric_test_values':0,'rejected_candidates':len(rejected),
        'rejected_reasons':dict(rejected_counts),'candidate_test_compatibility':dict(compat),'medical_primary_pages':len(catalog['sources']),
        'failed_medical_page_accesses_not_used':len(catalog['failed_access_not_used']),
        'recovered_vs_authored_symptom_counts':{'source_events':counts['recovered_facts'],'authored_symptoms':counts['synthetic_symptoms'],
            'denominator_note':'chief complaints and recommended tests are reported separately, not counted as observed tests'},
        'native_fhir_patient_bundle_count':0,'Source_Gold_Scorer_M5_or_completed_runs_changed':False,
        'evaluation_scope':SCOPE,'clinical_reviewed':False,'clinical_diagnosability':'UNREVIEWED',
        'actual_LLM_generation':0,'utility_fitting':0,'B0_B4_execution':0,'policy_fingerprint':digest(POLICY),
        'literature_fingerprint':digest(catalog),'original_dataset_sha256':sha_file(DATA)}
    atomic_json(PUB/'augmentation_summary.json',summary)
    DOC.mkdir(parents=True);documents(augmented,catalog,summary,DOC)
    atomic_json(PUB/'output_locations.json',{'public_docs':str(DOC.relative_to(ROOT)),'private_namespace':str(PR.relative_to(ROOT)),
        'private_main':'ORIGINAL50_AUGMENTATION_SUMMARY.md'})
    print(json.dumps(summary))

def documents(rows,catalog,summary,doc):
    with (PR/'ORIGINAL50_AUGMENTATION_SUMMARY.md').open('x',encoding='utf-8') as f:
        f.write('# Original50 원본·Source 복구·문헌 기반 합성 정보 — PRIVATE\n\n모든 환자 정보는 원본/복구/합성을 분리한다. SOURCE_RECOVERED는 실제 engine 이벤트의 복구이며 임상 공개 승인이 아니다. 합성은 label-conditioned 개발 후보로, 독립 일반화 성능 평가에 쓰지 않는다. 임상 충분성 전부UNREVIEWED.\n')
        for row in rows:
            n=row['case_ordinal'];syn=row['literature_informed_synthetic'];f.write('\n## Patient %02d\n\n'%n)
            f.write('Source 진단/전문과(Private 평가용): '+json.dumps(row['source_reference'],ensure_ascii=False)+'\n\n')
            f.write('### 원본 초기 정보\n\n```json\n'+json.dumps(row['original_initial_state'],ensure_ascii=False,indent=2)+'\n```\n')
            f.write('### 실제 현재 검사·관측\n\n|검사/관측명|Code|Value|Unit|기록시각|Source 행|\n|---|---|---|---|---|---|\n')
            for e in row['original_findings']:
                m=e['source_metadata'];f.write('|'+ '|'.join(md(v) for v in (e['target']['description'],e['target']['code'],e['evidence']['value'],e['evidence'].get('units'),e['timestamp'],str(m['source_file'])+':'+str(m['source_row'])))+'|\n')
            f.write('\n과거 방문/Memory: '+str(row['original_history']['past_visits'])+' / '+str(row['original_history']['fixed_memory_records'])+'. 원문 전체는 '+row['original_history']['full_history_audit_pointer']+' (SHA '+row['original_history']['full_history_audit_sha256']+').\n\n')
            f.write('### 원본에서 복구한 시각별 증상 이벤트\n\n')
            if not row['source_recovered_findings']:f.write('NO_ELIGIBLE_RECOVERY. Chief complaint를 원본에서 만들어내지 않는다.\n')
            for e in row['source_recovered_findings']:f.write('- '+e['scope']+' '+e['symptom_term']+'; engine value='+str(e['engine_value'])+' (임상 척도 아님); '+e['recorded_at']+'; visit link='+e['visit_link_basis']+'; Source='+e['source']['file']+'#'+e['source']['json_pointer']+'; SHA='+e['source']['file_sha256']+'\n')
            f.write('\n### 문헌 기반으로 새로 작성한 합성 정보\n\nChief complaint: '+syn['chief_complaint']['text']+'\n\nMode: '+syn['symptom_history_mode']+'; 이는 원본 환자가 실제 호소했다는 뜻이 아니다.\n\n')
            for s in syn['symptoms']:f.write('- '+s['text']+' — '+', '.join(catalog['sources'][i]['url'] for i in s['sources'])+'\n')
            f.write('\n### 검사 보강 후보\n\n수치형 합성 검사 결과0개. 아래는 요청/결과가 아닌 문헌 권장 검사 목록이다. 원본 값은 변경하지 않았다.\n')
            for t in syn['candidate_tests']:f.write('- '+t['name']+' — '+t['action_compatibility']+'; Source rows='+str(len(t['observed_source_event_ids']))+'; '+', '.join(catalog['sources'][i]['url'] for i in t['sources'])+'\n')
            f.write('\n확증에 여전히 필요한 근거: '+row['confirmation_still_missing']+'. 임상 충분성 UNREVIEWED, clinical_reviewed=false, 임상/Runtime admission 없음.\n')
    evidence=['# 공식 의학 근거 Catalog\n\n직접 읽은1차기관26개 페이지의 원래 요약과 링크다. Publication year가 확인되지 않으면UNKNOWN으로 두며 마지막 검토/수정일과 혼동하지 않는다. 이 문헌의 검토 상태가 개별 합성 환자의 임상 검토를 뜻하지 않는다.\n']
    for c in catalog['cards']:
        evidence.append('\n## '+c['disease']+'\n')
        for key in ('early_or_atypical','interpretation','differential','confirmation_needed','time_course','asymptomatic_or_uncertain'):evidence.append('- '+key+': '+c[key]+'\n')
        evidence.append('- 대표 증상 후보: '+', '.join(catalog['features'][i]['text'] for i in c['symptom_feature_ids'])+'\n')
        evidence.append('- 검사: '+', '.join(catalog['tests'][i]['name'] for i in c['candidate_test_ids'])+'\n')
        for i in c['sources']:
            s=catalog['sources'][i];evidence.append('- ['+s['institution']+' — '+s['title']+']('+s['url']+'); publication year=UNKNOWN; review/update='+s['revision_or_review_date']+'; checked='+s['checked_on']+'; '+s['observed_support_summary']+'\n')
    evidence.append('\n접근 실패해 사용하지 않은 자료: '+json.dumps(catalog['failed_access_not_used'])+'\n')
    (doc/'DISEASE_EVIDENCE_CATALOG.md').write_text(''.join(evidence),encoding='utf-8')
    (doc/'README.md').write_text('# Original50 문헌 기반 데이터 보강\n\n이 작업은 Original50을 수정하지 않고 D0 원본/D1 Source 복구 후보/D2 문헌 기반 합성 후보를 분리했다. 실제 환자별 Source·Reference·보강은 `outputs/private/research_evaluation/'+NS+'/ORIGINAL50_AUGMENTATION_SUMMARY.md`에 있다.\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\n숫자형 합성 확진값을 삽입하지 않았다. 권장 검사는 관측된 결과가 아니다. 신규 LLM Generation/Utility fitting/B0~B4 실행0회이며, Source Recovery는 새로운 임상 공개 grant가 아니다.\n',encoding='utf-8')
    (doc/'ACTION_COMPATIBILITY.md').write_text('# Action 호환성과 실행 경계\n\nFrozen M5는7개 객관 Test와 통증 Ask의8개 기존 Source Action을 유지한다. 새 증상 이력은 M5 Observation/likelihood나 objective Evidence로 변환하지 않는다. 통증도 객관 검사로 승격하지 않았다.\n\n```json\n'+json.dumps(summary['candidate_test_compatibility'],indent=2)+'\n```\n\nCURRENT_M5_COMPATIBLE은 동일 code/unit·기존 Source 및 cutoff45분 Proxy와 연결 가능한 경우다. 초기 모델 공개나 임상 release를 승인했다는 뜻이 아니다. SOURCE_EXISTS_BUT_ACTION_UNAVAILABLE은 Source에 있으나 모델 Action/시점이 맞지 않음, LITERATURE_RECOMMENDED_NOT_OBSERVED는 실제 결과가 없음이다. 수치/정성 합성 검사 결과는0개라 SYNTHETIC_ACTION_REQUIRES_NEW_CONTRACT로 실행할 신규 검사도0개다. 향후 생성할 경우 별도 계약이 필수이며 likelihood를 임의로 만들지 않는다.\n\n새 Adapter는 initial-state preview만 만든다. 기존 case_adapter/SourceView/메뉴/Scorer/Core/Gate를 수정하거나 Runtime에 등록하지 않았다.\n',encoding='utf-8')
    (doc/'NEXT_STEPS.md').write_text('# 사용 가능한 범위와 남은 결정\n\nD0는 원본 그대로다. D1은 provenance/시간을 검증한 Source engine 이벤트의 분석용 복구 후보이며, 방문 연결은 명시 ID가 아닌 유일한 Native encounter 시간 구간에서 도출했다. Source 실제 임상 available_at은 없고 기존 Source 공개 Guard는 유지된다. 과거 이벤트를 현재 증상으로 간주하거나 최종 resolved 상태를 과거로 소급하지 않는다.\n\nD2는 Source Reference에 따라 문헌 profile을 고른 LABEL_CONDITIONED_SYNTHETIC_DEVELOPMENT다. 원래10명 오류/모델 결과는 선택에 사용하지 않았다. 이 데이터가 진단 일치를 높여도 독립적 일반화나 원본 진단 능력 향상으로 해석할 수 없다. 문장 조합 다양성은 임상적 다양성의 검증이 아니다.\n\n추가 실행 전에 연구자가 (1) Source engine event의 활용 의미/방문 연결과 공개시각 Proxy, (2) D2의 label-conditioned 개발 목적 및 문헌·증상 조합, (3) 별도 Holdout/새 실행 계약·모델·예산을 검토해야 한다. 임상 충분성·Reference 적격성은 독립 검토 전까지UNREVIEWED다. Native B0~B4 실행 명령은 이번 범위에서 제공·실행하지 않는다. 데이터가 자동으로 기존 실험에 합쳐지지 않는다.\n\n원본 환자별 과거 병력/검사 전체는 새 Private main 문서의 Hash 연결을 통해 이전 통합 감사로 확인한다. FHIR export2개는 기관/의료인 Bundle뿐이며 환자별 FHIR 증상·검사는 없었다.\n',encoding='utf-8')

if __name__=='__main__':build()
