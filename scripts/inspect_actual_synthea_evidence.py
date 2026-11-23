"""Read-only inspection of actual source content and D1/D3 saved responses.

No models, gates, clinical judgments, previous reports, or runtime are modified.
Detailed per-patient diagnosis/evidence/rationale remains in Private CSV.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.routing_collection_preparation import TAXONOMY
from app.locdi.llm_setup import digest
from app.locdi.synthea_diagnostic_snapshot import source_check
from app.core.longitudinal_memory import timestamp

NS='FINAL_SYNTHEA_ACTUAL_INSPECTION_v1'
PUB=ROOT/'outputs/research_evaluation'/NS
PR=ROOT/'outputs/private/research_evaluation'/NS
DOC=ROOT/'docs/연구 평가'/NS
BC=ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'
BCPUB=ROOT/'outputs/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'
COM=ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_COMPLETION_v1'
COMPUB=ROOT/'outputs/research_evaluation/FINAL_SYNTHEA_COMPLETION_v1'
OLD=ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY'


def md(name,text):
    for base in (PUB,DOC):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def frequency_table(counts):
    return '|실제 출력 문자열|응답 수|\n|---|---:|\n'+''.join('|%s|%d|\n'%(k,v) for k,v in counts.most_common())


def main():
    if PUB.exists() or PR.exists() or DOC.exists():raise ValueError('FRESH_INSPECTION_NAMESPACE_REQUIRED')
    protected=read(COM/'preservation_before_live.json')
    for base in (BC,BCPUB,COM,COMPUB,ROOT/'docs/연구 평가/FINAL_SYNTHEA_COMPLETION_v1'):
        protected.update({str(f):sha(f) for f in base.rglob('*') if f.is_file()})
    write(PR/'preservation_before.json',protected)
    if check(protected)['status']!='PASS':raise ValueError('PRIOR_SOURCE_CHANGED')
    selected={'observation_inventory.json':BC/'observation_inventory.json',
        'final_observation_inventory_summary.json':BCPUB/'final_observation_inventory_summary.json',
        'post_execution_source_guard_addendum.json':BCPUB/'post_execution_source_guard_addendum.json',
        'reference_timing.json':BC/'reference_timing.json','snapshot_results.json':BCPUB/'snapshot_results.json',
        'paired_summary.json':BCPUB/'paired_summary.json'}
    located=[];objects={}
    for name,path in selected.items():
        if path.is_file():
            objects[name]=read(path);located.append({'name':name,'path':str(path.relative_to(ROOT)),
                'status':'FOUND_AND_PARSED','bytes':path.stat().st_size,'SHA256':sha(path)})
        else:located.append({'name':name,'path':str(path.relative_to(ROOT)),'status':'NOT_FOUND',
            'reason':'NOT_FOUND_AT_REQUIRED_PHASE_BC_NAMESPACE; no unrelated run substituted'})
    write(PUB/'actual_file_inventory.json',located)
    if len(objects)!=6:raise ValueError('REQUIRED_SOURCE_FILE_MISSING')
    inventory=objects['observation_inventory.json'];all378=[r for r in inventory if r['additional378_candidate']]
    pain=[r for r in all378 if str(r['source_event']['target']['code'])=='72514-3']
    candidates=[r for r in all378 if str(r['source_event']['target']['code'])!='72514-3']
    assert len(all378)==378 and len(pain)==20 and len(candidates)==358
    proposed=read(COM/'proposed_source_rows_NOT_AUTHORIZED.json');proposed_ids={r['source_event_id'] for r in proposed}
    checked={r['source_event_id']:r for r in read(COM/'source_row_validation.json')}
    duplicates={r['source_event_id']:r['status'] for r in read(COM/'source_duplicate_audit.json')}
    groups=defaultdict(list);rows=[];grants=[]
    for r in candidates:
        e=r['source_event'];ordinal=r['patient_ordinal'];identity=read(OLD/('Episode%03d.json'%(2*ordinal-1)))['private_identity']
        strict=source_check(e,patient=identity['patient_id'],visit=identity['visit_id'],
            index_ms=int(timestamp(identity['started_at']).timestamp()*1000),cutoff=r['cutoff_minute'])
        groups[(e['target']['code'],e['target']['description'],e['evidence']['category'],e['evidence'].get('units'))].append(r)
        a=checked[e['event_id']]
        rows.append({'patient_ordinal':ordinal,'code':e['target']['code'],'description':e['target']['description'],
            'category':e['evidence']['category'],'value':e['evidence']['value'],'unit':e['evidence'].get('units'),
            'recorded_at':e['timestamp'],'proxy_minute':r['proxy_available_minute'],'cutoff_minute':r['cutoff_minute'],
            'source_fingerprint':digest(e),'source_file':e['source_metadata']['source_file'],
            'source_row':e['source_metadata']['source_row'],'patient_visit_match':a['checks']['patient_visit_match'],
            'CSV_match':a['checks']['raw_csv_row_exact'],'strict_v1_status':strict,
            'projection_rule_status':duplicates[e['event_id']],'already_used_in_approved_static_run':e['event_id'] in proposed_ids,
            'actual_clinical_available_at':'NOT_VERIFIED','intraday_diagnosis_order':a['reference_order_status']})
        grants.append({'review_status':'UNREVIEWED_TEMPLATE_NOT_A_GRANT','source_event_private_ref':e['event_id'],
            'machine_source_facts':rows[-1], 'grant_fields_to_complete_by_actual_researcher':{
                'source_fingerprint':digest(e),'patient_id':e['patient_id'],'visit_id':e['encounter_id'],
                'mode':None,'release_reviewed':None,'diagnosis_leakage_reviewed':None,'contains_diagnosis':None,
                'recorded_time_proxy_accepted':None,'reviewer_private_ref':None,'evidence_private_ref':None,
                'reviewed_at':None,'procedure_version':None}})
    csv_file(PR/'ACTUAL_ADDITIONAL_TEST_VALUES.csv',rows,list(rows[0]))
    write(PR/'SOURCE_RELEASE_REVIEW_TEMPLATE.json',{'status':'BLANK_REVIEW_TEMPLATE_ONLY',
        'not_an_approval_or_grant':True,'rows':grants,'existing_v2_operator_permission_is_not_backfilled_to_v1':True})
    write(PR/'ASK_ONLY_PAIN20.json',pain)
    # Aggregate by code+description, preserving all observed category/unit variants.
    bytype=defaultdict(list)
    for r in candidates:
        e=r['source_event'];bytype[e['target']['code'],e['target']['description']].append(r)
    testrows=[]
    for (code,name),rr in sorted(bytype.items()):
        testrows.append({'test_name':name,'code':code,
            'category':'; '.join(sorted({x['source_event']['evidence']['category'] for x in rr})),
            'units':'; '.join(sorted({str(x['source_event']['evidence'].get('units')) for x in rr})),
            'patient_count':len({x['patient_ordinal'] for x in rr}),'record_count':len(rr),
            'duplicates_existing_four':code in ('8462-4','8480-6','8867-4','9279-1'),
            'proxy_time_eligible_count':sum(x['proxy_available_minute']<=x['cutoff_minute'] for x in rr),
            'recorded_proxy_minute':'; '.join(str(x) for x in sorted({x['proxy_available_minute'] for x in rr})),
            'cutoff_minute':'; '.join(str(x) for x in sorted({x['cutoff_minute'] for x in rr})),
            'strict_v1_release':'NOT_REVIEWED',
            'approved_v2_static_used_records':sum(x['source_event_id'] in proposed_ids for x in rr),
            'clinical_disclosure':'NOT_VERIFIED','potential':'NUMERIC_SOURCE_INFORMATION_NOT_PROOF_OF_DIAGNOSIS',
            'review_reason':'Actual available_at absent; intraday diagnosis order unknown; semantic review not performed; v1 grant absent'})
    assert len(testrows)==50 and sum(x['record_count'] for x in testrows)==358
    csv_file(PUB/'ACTUAL_ADDITIONAL_TEST_TYPES.csv',testrows,list(testrows[0]))
    perpatient=[]
    for i in range(1,21):
        rr=[r for r in candidates if r['patient_ordinal']==i]
        perpatient.append({'patient_ordinal':i,'candidate_records':len(rr),
            'candidate_codes':len({r['source_event']['target']['code'] for r in rr}),
            'approved_static_used_records':sum(r['source_event_id'] in proposed_ids for r in rr),
            'pain_Ask_only_records':sum(r['patient_ordinal']==i for r in pain)})
    csv_file(PUB/'ACTUAL_PATIENT_TEST_DISTRIBUTION.csv',perpatient,list(perpatient[0]))
    taxonomy=read(TAXONOMY);predictions=[];public_predictions=[];stats={}
    baseline_root=BC/'synthea_snapshot_001'
    for condition in ('D1','D3'):
        diag=Counter();dept=Counter();citation=Counter();citednames=Counter();phrases=Counter();mapstate=Counter()
        reference_set=set();chief_missing=0
        for ordinal in range(1,21):
            folder=baseline_root/('Patient%03d_%s'%(ordinal,condition))
            before=read(folder/'Request0001_before.json');after=read(folder/'Request0001_after.json')
            scoring=read(folder/'private_scoring.json');raw=json.loads(after['response_envelope']['message']['content'])
            payload=json.loads(before['request']['messages'][1]['content']);v=scoring['validated']
            assert scoring['original_model_output']==raw
            assert raw['target']==v['target'] and raw['department']==v['department']
            refs=scoring['reference'];reference_set.update(refs);chief_missing+=not bool(payload['observation']['chief_complaint'])
            diag[v['target']]+=1;dept[v['department']]+=1
            records=payload['evidence_records'];indices=raw['evidence_indices']
            assert all(type(i) is int and 0<=i<len(records) for i in indices)
            cited=[records[i] for i in indices]
            for e in cited:
                citation[e['kind']]+=1
                citednames[e['text'].split(':',1)[0] if e['kind']=='objective_finding' else e['kind']]+=1
            for term in ('chest pain','shortness of breath','dyspnea','fever','tachycardia','respiratory','blood pressure','heart rate','pain'):
                if term in raw['rationale'].lower():phrases[term]+=1
            expected=taxonomy.get(v['target'])
            consistency='NOT_MAPPED_IN_EXISTING_TAXONOMY' if expected is None else 'CONSISTENT' if expected==v['department'] else 'DECLARED_TAXONOMY_MISMATCH'
            mapstate[consistency]+=1
            number=len(predictions)+1
            predictions.append({'prediction_ref':'Prediction%03d'%number,'patient_ordinal':ordinal,'condition':condition,
                'Source_diagnosis':json.dumps(refs,ensure_ascii=False),'predicted_diagnosis':v['target'],
                'predicted_department':v['department'],'certainty':v['certainty'],
                'source_exact':v['target'] in refs,'existing_catalog_taxonomy_department':expected,
                'taxonomy_consistency':consistency,'provided_evidence':json.dumps(records,ensure_ascii=False),
                'evidence_indices':json.dumps(indices),'eligible_indices':json.dumps(payload['supported_diagnosis_eligible_indices']),
                'actually_cited_evidence':json.dumps(cited,ensure_ascii=False),'rationale':raw['rationale'],
                'assessment':raw['assessment'],'provided_observation':json.dumps(payload['observation'],ensure_ascii=False),
                'before_private_path':str(folder/'Request0001_before.json'),'after_private_path':str(folder/'Request0001_after.json')})
            public_predictions.append({'prediction_ref':'Prediction%03d'%number,'patient_ordinal':ordinal,'condition':condition,
                'source_reference_private_ref':'SourceReference%03d'%ordinal,'source_exact':v['target'] in refs,
                'certainty':v['certainty'],'provided_evidence_count':len(records),'citation_count':len(indices),
                'objective_citation_count':sum(e['kind']=='objective_finding' for e in cited),
                'patient_history_citation_count':sum(e['kind']=='patient_history' for e in cited),
                'taxonomy_consistency':consistency,'diagnosis_and_rationale':'SEE_SAME_NAMED_PRIVATE_CSV'})
        stats[condition]={'predicted_diagnosis_counts':dict(diag),'predicted_department_counts':dict(dept),
            'cited_kind_occurrences':dict(citation),'cited_entry_counts':dict(citednames),
            'rationale_phrase_counts':dict(phrases),'declared_taxonomy_consistency':dict(mapstate),
            'chief_complaint_missing':chief_missing,'predictions_outside_cohort_Source_label_set':sum(n for x,n in diag.items() if x not in reference_set)}
    csv_file(PR/'ACTUAL_DIAGNOSIS_PREDICTIONS.csv',predictions,list(predictions[0]))
    csv_file(PUB/'ACTUAL_DIAGNOSIS_PREDICTIONS.csv',public_predictions,list(public_predictions[0]))
    write(PUB/'actual_response_pattern_statistics.json',stats)
    latest=read(COMPUB/'actual_results_v1/final_closeout_manifest.json')
    approval=read(COM/'development_proxy_execution_authorization.json')
    assert latest['actual_generations']==40
    assert approval['plan_fingerprint']==read(COMPUB/'d2_d4_protocol_final.json')['fingerprint']
    assert approval['plan_fingerprint']==read(COMPUB/'actual_results_v1/independent_execution_verification.json')['approved_plan']
    raw_hashes=read(COM/'raw_file_hashes.json')
    raw_unchanged=all(Path(name).is_file() and sha(Path(name))==expected for name,expected in raw_hashes.items())
    assert raw_unchanged
    # Verify the actual result data against the sealed final manifest, without
    # rerunning the prior experiment or recreating previous reports.
    actual_path=COMPUB/'actual_results_v1/actual_results.json'
    assert sha(actual_path)==latest['actual_result_file_hashes']['actual_results.json']
    write(PUB/'current_execution_state.json',{'old_BC_snapshot_state':'D0_D1_D3_COMPLETE_D2_D4_NOT_RUN_AT_THAT_TIME',
        'actual_run_id':'synthea_d2_d4_001','actual_D2_D4_generation':40,
        'D2_Source_exact':0,'D4_Source_exact':4,'registered_per_condition':20,
        'specific_static_v2_user_approval':'ALREADY_OBTAINED_AND_CONSUMED_ONCE',
        'v1_grants':0,'clinical_release_review':'NOT_REVIEWED','new_generation_this_inspection':0,
        'actual_report_directory':'docs/연구 평가/FINAL_SYNTHEA_COMPLETION_v1/actual_results_v1',
        'actual_result_manifest_sha':sha(COMPUB/'actual_results_v1/final_closeout_manifest.json'),
        'old_D2_D4_NOT_RUN_is_not_current_state':True})
    files='\n'.join('- `%s`: `%s`，%s，SHA256 `%s`'%(x['name'],x['path'],x['status'],x['SHA256']) for x in located)
    table='|검사명|Code|Category|단위|환자수|기록수|기존검사 중복|시간적격|공개상태|\n|---|---|---|---|---:|---:|---|---|---|\n'
    for x in testrows:
        table+='|%s|%s|%s|%s|%s|%s|아니오|Proxy %s/%s|v1 미검토 / v2 정적사용 %s행|\n'%(x['test_name'],x['code'],x['category'],x['units'],x['patient_count'],x['record_count'],x['proxy_time_eligible_count'],x['record_count'],x['approved_v2_static_used_records'])
    pt='\n|환자 ordinal|추가 후보code|원천행|실제 v2 정적사용행|통증 Ask 전용|\n|---|---:|---:|---:|---:|\n'
    for r in perpatient:pt+='|%s|%s|%s|%s|%s|\n'%(r['patient_ordinal'],r['candidate_codes'],r['candidate_records'],r['approved_static_used_records'],r['pain_Ask_only_records'])
    md('ACTUAL_ADDITIONAL_TEST_SUMMARY.md','# 실제 추가 검사 내용 확인\n\n'
        '요청한6개 파일은 모두 실제로 존재하며 JSON 전체를 읽었다. NOT_FOUND0건이다. 기존8개 보고서를 재생성하지 않았다.\n\n'+files+'\n\n'+table+pt+
        '\n검사358행/50종, 20명 전원에게 기존4종 외의 Source 관측이 있다. laboratory298행, 신체계측 vital-signs60행이다. '
        '기존4종(DBP8462-4/SBP8480-6/HR8867-4/RR9279-1)과 추가code 중복0행이다. 통증20행은별도 Ask 전용으로 보존했다. '
        '동일 Source키 상이값6행 제외와 완전중복7행 제거 후345행이 이미 승인된 v2 정적 평가에 사용됐다. 원본358행은 삭제하지 않았다. '
        '환자별 값·단위·시각·Source 위치는 Private ACTUAL_ADDITIONAL_TEST_VALUES.csv에 있다. '
        '기록 Proxy는1분, cutoff45분이며 실제 clinical available_at은미확인이다. 알려진 날짜 관계와 진단/검사 intraday 순서는 구분한다. '
        '혈당/HbA1c·신장·전해질·지질·혈구·소변·간기능 등은 기존 활력과 다른 관측 정보를 제공하지만, 개별 진단의 충분한 근거라는 임상 판정은 하지 않는다.\n\n'
        '**현재 상태 정정:** 위 BC snapshot_results/paired_summary의 D2/D4 NOT_RUN은 이전 시점 기록이다. '
        '별도 승인 후 synthea_d2_d4_001에서D2/D4각20회가 완료됐고 Source Exact는각0/20,4/20이다. 실제 결과는 FINAL_SYNTHEA_COMPLETION_v1/actual_results_v1에 있다. '
        'v1 grant0과 v2 정적 개발 실행 완료는 서로 다른 상태라 동시에 성립한다. 이번 검사 작업의 Generation은0회다.\n')
    text='# 실제 D1/D3 예측 및 실패 패턴\n\n'
    for c in ('D1','D3'):
        z=stats[c];text+='## '+c+'\n\n'+frequency_table(Counter(z['predicted_diagnosis_counts']))+'\n'+frequency_table(Counter(z['predicted_department_counts']))
        text+='\n인용 entry별 횟수:\n\n'+frequency_table(Counter(z['cited_entry_counts']))+'\n'
    text+=('원문 문자열은 그대로 집계했다. Acute Coronary Syndrome과 Acute Coronary Syndrome (ACS)는 자동 동의어 병합하지 않았다. '
        '40개 원본 응답·validated target/department가 모두 일치했다. Source label/예측·Evidence 전문·인용·rationale·assessment는 '
        'Private ACTUAL_DIAGNOSIS_PREDICTIONS.csv40행에 있고, Public의 같은 이름 파일은 ordinal/내부참조/검증값40행만 제공한다.\n\n'
        'D1은Pulmonary Embolism9/20, D3는Chronic obstructive bronchitis11/20을 반복했다. '
        'D3 진료과는nephrology20/20이다. 양쪽 모두Respiratory rate를20/20회 인용했다. '
        'D1 rationale의respiratory문구는17/20, D3는19/20이며 chest pain문구는D1에서1건이다. '
        '문구 빈도는 모델의 실제 설명 패턴이며 원인의 증명이나 임상적 오류 판정이 아니다. '
        '초기 chief complaint는두조건각20/20 누락이며 실제 문진은통증점수이다. 원문 전체를 Private에서 함께 검토해야 한다.\n\n'
        'D3 예측19/20은이20명의 Source label7종에 없는 진단이었다. 그렇다고 공통15 Catalog를 환자별Gold7로 축소하지 않는다. '
        '기존 연구taxonomy와 D3 출력 진료과의 일관성은5/20이고15/20은불일치했다. '
        '이는 기존 명시적 taxonomy와의 문자열 계약 비교이며 임상적 적정 진료과 판정이 아니다. '
        '해당 실행은diagnosis_departments를 바인딩하지 않아 진단/진료과를 공동 강제하지 않았다. '
        'D1의20개 진단표기는 기존 taxonomy에 없어 해당 일관성은NOT_EVALUATED다.\n\n'
        '모델은제한된 활력/통증 정보로 호흡기·급성 진단 또는nephrology를 반복했고 Source Exact는각0/20이었다. '
        '과도한 추론·모델 prior·입력 부족·Prompt 영향은가능한 설명이지만 단독 원인은확정하지 않는다. '
        '기존 Evidence Gate는인용범위/객관entry존재를 검사하며 해당관측이 특정질환을 임상적으로 확정하는지 판정하지 않는다. '
        '환자/방문 연결·파싱·scorer 오류로 이 결과를 바꾸지 않는다. Verified Concept Match는NOT_EVALUATED다.\n\n'
        '이미 완료된확대Source 비교에서는D2가0/20, D4가4/20이었다. 추가정보+Catalog의 제한적 Source 일치 가능성이지 임상 검증이나 Memory 효과가 아니다. '
        '현재검사에서는추가모델호출·기존Run재실행·코드수정0이다.\n')
    md('ACTUAL_DIAGNOSIS_FAILURE_ANALYSIS.md',text)
    md('SOURCE_GRANT_REQUIREMENTS.md','# source_check 실제 요구사항\n\n'
        '구현: app/locdi/synthea_diagnostic_snapshot.py::source_check. 파일과 함수 원문을 읽었으며 수정하지 않았다.\n\n'
        '|필드|실제 검사 조건|현재 상태/검증 주체|\n|---|---|---|\n'
        '|source_fingerprint|digest(event)와 동일|자동 계산 가능, 원문불변 검증 필요|\n'
        '|mode|요청 mode와 동일; snapshot/sequential만 허용|실제 공개목적별 선택; 기존정적 승인으로sequential을 대체하지 않음|\n'
        '|patient_id / visit_id|각Source환자/방문과 동일|자동대조 가능, Private 보관|\n'
        '|release_reviewed|정확히 boolean true|v1 실제공개검토 없음; 자동 true 금지|\n'
        '|diagnosis_leakage_reviewed|정확히 boolean true|실제 의미·시간 검토 필요; lexical 부재와 다름|\n'
        '|contains_diagnosis|정확히 boolean false|실제검토 전null 유지; 진단시사 검사값과 정답직접문구를 구분|\n'
        '|recorded_time_proxy_accepted|정확히 boolean true|v1 범위/목적별 실제수용 필요; 기존v2승인을 소급입력하지 않음|\n'
        '|reviewer_private_ref|비어있지 않은문자열|실제검토자 참조, 가명가능; 생성금지|\n'
        '|evidence_private_ref|비어있지 않은문자열|실제검토근거 참조, 가상문서 생성금지|\n'
        '|reviewed_at|비어있지 않은문자열|실제검토시각만 작성|\n'
        '|procedure_version|비어있지 않은문자열|실제검토절차 버전|\n\n'
        'grant이전에Source환자/방문, 통증code제외, numeric category/type, value/unit/code/date일치, timestamp와cutoff, recorded_observation 종류를 검사한다. '
        '현재358행은관련 구조검증을통과하지만grant=None이므로v1은RELEASE_NOT_REVIEWED다. '
        '검사 code/description/단위/시각/Source 파일 실재·fingerprint·cutoff는자동 사실검증이 가능하다. '
        '실제 공개시각, 진단전 관측 여부, 의미기반 누출, 실제 검토 수행/신원/근거는그 사실만으로 자동승인할 수 없다.\n\n'
        '**구현의 한계:** source_check는reviewed_at의날짜 파싱·근거파일 존재/Hash·reviewer독립성/직업자격·실제승인권한을 검증하지 않고4개문자열의 비어있지않음만 확인한다. '
        '필드가문법적으로 유효하다는이유로 실제 검토완료를 주장하지 않는다. 함수가정의하지않은 기관규정이나 임상자격요건도 임의로추가하지 않는다. '
        '이번에는Guard 재구현·완화·우회·grant생성을 하지 않았다. 빈 Private 검토양식은358행이며 판정칸은null이다.\n')
    md('SOURCE_RELEASE_REVIEW_REQUEST.md','# 연구자가 결정할 사항 — 현재 상태 기준\n\n'
        '**기존 D2/D4에 대한 재승인 요청은 아니다.** 사용자는e62d6641…7edc42의 합성 Recorded-Time Proxy 정책과최대40회 실행을 이미승인했고 '
        '해당Run은완료됐다. 초기BC보고서의NOT_RUN과 현재를 혼동해재실행하지 않는다. 이번 Generation0, 신규grant0이다.\n\n'
        '연구자는우선최신실제결과(D2 0/20,D4 4/20)를종료자료로채택할지 결정하면된다. 이범위의추가승인은필요하지않다. '
        '일반임상공개나순차NativeER평가로범위를확장하려는경우에만, 다음을실제검토해야한다.\n\n'
        '1. ACTUAL_ADDITIONAL_TEST_TYPES.csv의50종과 Private358개값/단위/시각을검토한다. 통증20행은Ask전용이다.\n'
        '2. 상이값6행을보류하고완전중복7행을입력반복하지않는공통규칙 및나머지345행의허용범위를 Source fingerprint로고정한다.\n'
        '3. 실제available_at근거가있는지, 없다면어떤합성개발목적에서만Recorded-Time Proxy를수용할지명시한다. 실제clinical release로표현하지않는다.\n'
        '4. 같은날18명의intraday진단/검사순서미확정과ordinal12·16의다음날Source Condition을기록한다. 원래20명/Reference를변경·제외하지않는다.\n'
        '5. 실제검토자의판정·근거·시각·절차를 Private SOURCE_RELEASE_REVIEW_TEMPLATE.json의null칸에작성한다. 문자열만채워승인을가장하지않는다.\n'
        '6. 목적이snapshot인지sequential인지분리하고, 이후새로운Generation/재실행이필요하면별도새Protocol·Run·예산승인을받는다. 기존1회승인을재사용하지않는다.\n\n'
        '답변: 추가358행은기존활력4종반복이아니며50종의다른Source정보다. 20/20환자에새code가있다. '
        '고정합성정적D2/D4의기술조건은이미충족되어실행까지완료됐다. v1임상공개검토/순차실행은여전히별개다. '
        '임상Reference/의미검토없음을임상PASS로승격하지않고, 현재결과를근거로480회B0/B2를자동실행하지않는다.\n')
    # Verify all outputs against privacy and immutability requirements.
    identities=[]
    for i in range(1,21):
        d=read(OLD/('Episode%03d.json'%(2*i-1)))
        identities.extend(d['private_identity'][k] for k in ('patient_id','visit_id'))
    leaks=[]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.csv','.md'):
                t=f.read_text(encoding='utf-8-sig')
                if any(v in t for v in identities):leaks.append(str(f))
                if f.suffix=='.md' and ('\ufffd' in t or '??' in t):raise ValueError('ENCODING_FAILURE')
    assert not leaks
    after=check(protected)
    assert after['status']=='PASS'
    write(PUB/'inspection_verification.json',{'status':'PASS','source_files_found_and_read':6,'source_files_missing':0,
        'additional_rows':358,'additional_types':50,'patients_with_new_codes':20,'Ask_only_pain_rows':20,
        'model_responses_read_and_checked':40,'public_prediction_rows':40,'private_full_prediction_rows':40,
        'private_value_rows':358,'blank_v1_review_template_rows':358,'new_grants':0,'new_generations':0,
        'physical_raw_source_hash_unchanged':raw_unchanged,
        'new_fitting':0,'runtime_or_gate_changes':0,'previous_reports_regenerated':0,
        'current_D2_D4_actual_generation_from_previous_run':40,'current_D2_exact':0,'current_D4_exact':4,
        'protected':after,'public_patient_visit_ID_leaks':leaks,
        'code_sha':sha(Path(__file__)),'taxonomy_path':str(TAXONOMY),'taxonomy_sha':sha(TAXONOMY),
        'prior_test_results_reused_as_historical_only':'No whole-suite test count claimed for this inspection',
        'output_files':{str(f.relative_to(ROOT)):sha(f) for base in (PUB,DOC) for f in base.glob('*') if f.is_file()}})
    print(json.dumps({'status':'PASS','types':50,'rows':358,'D1_D3_responses':40,'new_generation':0,
        'current_D2_D4':'ALREADY_COMPLETED_IN_PRIOR_APPROVED_RUN','protected':after,'prediction_patterns':stats}))


if __name__=='__main__':main()
