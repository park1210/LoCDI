"""Publish final saved results without new generation or changing prior reports."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_locdi_pilot_improvement import PUB as PREP,PR,DOC as PREPDOC,RUN
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.memory_paired_recovery import aggregate
from app.locdi.llm_setup import digest

PUB=PREP/'final_results_v1';DOC=PREPDOC/'final_results_v1'


def verify():
    m=read(PUB/'FINAL_DELIVERY_MANIFEST.json')
    protection=check(read(PR/'preservation_before.json'))
    pins=read(PREP/'SYNTHEA_PILOT_PROTOCOL.json')
    actual=check({str(PR/RUN/name):s for name,s in m['actual_Run_hashes'].items()})
    reports=check({str(DOC/name):s for name,s in m['report_SHA256'].items()})
    collection=read(PR/RUN/'collection.json');a=aggregate(read(PR/'panel.json'),collection['rows'])
    exact=a==read(PR/RUN/'aggregate.json')
    assert protection['status']==actual['status']==reports['status']=='PASS' and exact
    print(json.dumps({'status':'PASS','protected':protection,'actual_Run':actual,'reports':reports,
        'stored_trace_reaggregation':'EXACT_MATCH','generation':0}))


def main():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_FINAL_RESULTS_REQUIRED')
    p=read(PREP/'SYNTHEA_PILOT_PROTOCOL.json')
    result=read(PREP/'actual_results_v1/actual_pilot_metrics.json');s=result['summary']
    failure=read(PR/'final_request_contract_failure.json')
    assert s['generation']==48 and s['complete_pairs']==1
    assert failure['schema_errors']==[{'validator':'uniqueItems','path':['memory_ids'],
        'schema_path':['properties','memory_ids','uniqueItems']}]
    protected=check(read(PR/'preservation_before.json'));assert protected['status']=='PASS'
    pincheck=check({k:v for g in ('input_hashes','new_code_hashes','binding_hashes') for k,v in p[g].items()})
    assert pincheck['status']=='PASS'
    PUB.mkdir(parents=True);DOC.mkdir(parents=True)
    for name in ('FINAL_EPISODE_RESULTS.csv','FINAL_PAIRED_RESULTS.csv','actual_pilot_metrics.json'):
        with (PUB/name).open('xb') as f:f.write((PREP/'actual_results_v1'/name).read_bytes())
    header='''|조건|등록|완료|실패|미실행|Supported|Exact / Supported|Normalized / Supported|Source 진료과 / 완료|Generation|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B0|10|2|0|8|2|0/2|0/2|0/2|24|
|B2|10|1|1|8|1|0/1|0/1|0/1|24|
'''
    footer=f'''

공통 실행 계약: Run `{RUN}`, Model `qwen2.5:3b` / `{p['model_identifier']}`.
Protocol `{p['fingerprint']}`. Source rows `{p['source_rows_fingerprint']}`.
Data SHA `{s['Data_SHA256']}`, Code Pin 집합 `{s['Code_pins_fingerprint']}`.
기존 15종 Catalog SHA `{p['catalog']['SHA256']}`.
Context32768, temperature0, seed42, output1024, timeout120초, Retry/Repair/Warmup0.
동일10명, 환자별B0→B2, 각12/총240회 이하, wall45분·Python4GiB·출력1GiB·no-resume.
Memory B0 none / B2 기존token-intersection Top5, Routing OFF, VoI OFF.
실제 요청48회 후 첫 기술 오류에서 중단했다. 분석 중 추가 Generation0회.
임상 정확도·Verified Concept Match는 NOT_EVALUATED다. 환자별 예측 진단·Source·모델 원문은 Private
`outputs/private/research_evaluation/{p['namespace']}/actual_diagnostic_analysis.json` 및 `{RUN}/requests/`에 보존했다.
'''
    documents={
        'PILOT_DATA_USAGE_COMPARISON.md':'''# 기존 Pilot과 현재 데이터 사용 방식

|항목|기존 실제 코드|현재 Native ER|
|---|---|---|
|초기 정보|experiment24의build_action_replay.py::main은 측정항목명을associated_symptoms, 수치를Ask 응답으로 분리|같은Source를Adapter가 전달한다. 없는chief complaint/증상을 생성하지 않는다.|
|실행용 검사|experiment30 build_study.py::main 내부transform은 모든진단cell에 관측 지원이 있는 행동만 남김|VoI OFF에서 별도 승인된345행 중 선정군175행을 추가메뉴로 연결|
|추가 관측|30의cases_source_enriched_review.json에 확률 미추정 검사를 보존|관측은 동일환자/방문·code/value/unit/date/fingerprint를 검증하고 요청 후만 공개|
|과거 기록|과거방문·고정Memory를 환자와 연결|B0 OFF/B2 기존Top5·index cutoff 유지|
|검사 공개|available_minute는재생가정|Scheduler의주문→시작→max(start+5,source proxy)수신→모델공개|
|Catalog|30의공통15종,34 prepare_package.py::main이 cases/registry/model을복사|같은공통15종을모든환자·조건에동일하게전달|
|확률/행동|30은학습cell·median bin·smoothing,34패키지는별도MemoryAwareLLMRouter 계약도명시|과거확률/정책을복원하지않고ER-only·VoI OFF로진단상호작용만실행|

실제 upstream 경로·SHA는 상위upstream_source_code_inventory.json에 있다. 당시 B0~B4 보고값의 원래Run을 확인하지 못한 한계를 유지한다.
이번 비교는 데이터 사용 경로 비교이며 과거1.5B 실험 재현이 아니다.
''',
        'CURRENT_SYNTHEA_INPUT_IMPROVEMENT.md':'''# 입력 전달 검증 결과

선정10명은 명시적인chief complaint가0/10이며, 초기증상 목록은 환자보고 통증의 측정항목명이다.
통증이 있다는 호소나 수치를 초기상태에 만들어 넣지 않았다. 실제값은 기존Ask 이후 공개했다.
현재Catalog는 선정10명의Source label을 모두 포함한다. 환자별Gold로 후보를 축소·정렬하지 않았다.

추가Source 메뉴175개를 B0/B2에 동일하게 연결했다. 실제48요청에서 추가검사24건을 주문·시작·수신했고,
수신 후 모델입력의 값·단위를96 occurrence에서 원천과 대조했다. 추가Evidence를 인용한 결정은10건이었다.
검사 조기 공개와Memory 환자/방문/cutoff 위반은 저장기록에서0건이었다.
이것은 검사정보가 전달됐다는 구조적 근거이며, 진단 충분성이나 임상 타당성 검증은 아니다.

이번 작업에서는 임상View·Prompt·Memory검색·공통Catalog·Source공개정책을 변경하지 않았다.
확인된Exporter 오류를 먼저 해결한 설정을 실행해 변수를 분리했다. 진단개선은 확인되지 않았다.
''',
        'FINAL_EXPORTER_VERIFICATION.md':'''# Exporter와 전체 후처리 검증

기존1.3.2 수정이 실제 적용되어 첫3개Episode는 Core 완료와Export를 모두 통과했다.
configuration.sequential_source_view의 엄격한7필드/정책/타입 검증 및 전체fingerprint 검증은 유지했다.
이전synthea_sequential_001의실패1/미실행19를 성공으로 소급하지 않았다.

실행 전13개 관련 모듈227/227 PASS, 같은10명×B0/B2 Fake Client의 실제Runner/Scheduler/Exporter20/20 PASS,
기존저장raw→Exporter→Backend→기존aggregate Scorer/별도Normalized검증PASS다. 전체저장소회귀라고확대하지않는다.
실행 후61개Code/Input Pin과보호1,976개Hash는모두일치했다.
새실패는Exporter가아닌모델memory_ids 중복의Schema위반이다. 실패Episode의raw/checkpoint는보존됐으며
완료Export로바꾸지않았다.
''',
        'FINAL_SYNTHEA_ACCURACY_IMPROVEMENT.md':'''# 진단 개선 판정

'''+header+'''
완료된3건은Source Exact/Normalized 모두불일치했다. 등록10명당정답수율은각0/10이지만,
이는각10명모두를완료한진단정확도0%라는뜻이아니다. 미실행8명씩과B2실패1건을별도로보존한다.
실패B2의수신응답을유효한최종진단으로승격해채점하지않았다.

정적D4 4/20은순차B0/B2나Memory효과로대체할수없다. 확대Evidence는전달됐지만이번순차실행에서
진단개선신호는확인되지않았다. 원인단정은금지한다. 현재증상부족·개발패널의정보한계·모델추론가능성과
응답계약위반을구분한다. 결과에맞춘Prompt/검사선택/정답매핑을하지않았고추가최적화없이종료한다.
''',
        'FINAL_SEQUENTIAL_PILOT_RESULTS.md':'''# 실제 최종 순차 Pilot

'''+header+f'''
전체20Episode: 완료3, CONTRACT_FAILED1, NOT_ATTEMPTED16. 실제48Generation, wall{s['wall_seconds']:.3f}초.
최대240회 승인은 상한이며 실패 후 남은요청을소진하지않았다. HTTP200 48건,실제Context/Digest48건일치,
Schema·인용계약동시통과47건이다. Timeout0,Retry0,Repair0,재개0이다.

실패위치: 선정두번째환자의B2, Episode0004/Request0012_after.json.
HTTP200·done=true·done_reason=stop,JSON파싱PASS,output249 tokens,wall7.718초였다.
`memory_ids`가4개지만고유ID는3개여서`properties.memory_ids.uniqueItems`에위반됐다.
NativeClient가LLMResponseError로거부한뒤Core failed/Collector CONTRACT_FAILED가됐다.
이는관측된Memory검색의환자격리실패나Evidence인덱스오류가아니다. 자동중복제거·응답수정·재시도하지않았다.

파일:FINAL_EPISODE_RESULTS.csv(20행),FINAL_PAIRED_RESULTS.csv(10행),actual_pilot_metrics.json.
실제임상공개시각·TAT와ordinal12다음날Reference한계는여전히남는다.
''',
        'FINAL_B0_B2_MEMORY_ANALYSIS.md':'''# Memory paired 결과

'''+header+'''
완료쌍1/10, Source scorable쌍1/10이다. 이한쌍의B0/B2는모두Source불일치여서조건부차이0pp다.
이를10명전체의Memory효과0 또는효과없음의증명으로일반화하지않는다. 등록분모·실패·미실행을함께보고한다.

전체prompt token은B0 85,217, B2 127,134이며각24요청이다. output token은2,653/3,618이다.
B2의24개요청에서Memory record120 occurrence(반복제공포함)를확인했고환자/이전방문/cutoff위반은0이다.
이는고유Memory120개라는뜻이아니다. 별도Memory token 분리계측과초기retrieval minute은없다.
실패한마지막모델출력의중복memory_ids를Memory검색원본의중복으로해석하지않는다.
''',
        'FINAL_THESIS_RESULTS_COMPARISON.md':'''# 논문에 반영할 실제 결과

|자료|실제분모/결과|기술할범위|
|---|---|---|
|역사적논문B0~B4|38/52/44/48/90% 보고값|과거Run미확인;현재재현값아님|
|기존정적D4|같은개발20명,Source Exact4/20|확대Evidence+15Catalog의정적개발신호|
|이전순차001|완료0,실패1,미실행19|Exporter연결실패보존|
|새순차002|완료3/20,실패1/20,미실행16/20,48요청|Exporter해결/Source공개경로기술검증과응답Schema한계|

현재순차완료응답Source일치는B0 0/2,B2 0/1,완료paired1쌍이다. 전체10명진단평가완료로쓰지않는다.
공식matched B0~B4,Routing/VoI 효과나전체Framework의임상우월성을확보했다고쓰지않는다.
동일개발패널반복노출,원천chief complaint/증상narrative부족,합성공개시각과Reference시점한계를명시한다.
기존D4·C4-35·C4-39·FINAL_THESIS와과거결과는변경하지않는다.
''',
        'FINAL_PROFESSOR_BRIEFING.md':'''# 교수님 최종 보고

기존Synthea와3B모델·연구수식을유지하고,과거Pilot의데이터사용경로를검토해추가검사의순차전달을실행했습니다.
현재자료의진단성능을검증하는작업이며과거90%를재현하거나맞추는실험이아닙니다.

확보한결과: Exporter오류를수정한설정에서실제3Episode가완료·저장됐고추가검사24건의요청/수신이실행됐습니다.
기록상조기공개/Memory경계위반은0입니다. 그러나완료진단3건모두Source와불일치했고,
네번째Episode의모델memory_ids중복으로Schema검증에실패해전체실행을중단했습니다.

최종등록20Episode는완료3/실패1/미실행16,완료쌍1/10,48Generation입니다.
진단개선과안정적인Memory효과는확인하지못했습니다. 관측된정보전달성공과성능미확인을구분해논문에기록해야합니다.
증거검증을완화하거나응답을자동수정하지않았고추가실험·모델학습·Prompt최적화없이종료합니다.
'''}
    for name,text in documents.items():
        for base in (PUB,DOC):
            with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+footer)
    # Source identifiers and clinical verbatim stay in the original Private tree.
    from app.locdi.dataset_loader import load_cases
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')
    forbidden=[c['patient_id'] for c in cases]+[c['agent_visible']['index_encounter_id'] for c in cases]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file():assert not any(x in f.read_text(encoding='utf-8-sig') for x in forbidden)
    write(PUB/'FINAL_DELIVERY_MANIFEST.json',{'status':'PARTIAL_PILOT_NO_DIAGNOSTIC_IMPROVEMENT_ESTABLISHED',
        'actual_generation':48,'generation_after_stop':0,'completed':3,'failed':1,'not_attempted':16,
        'complete_pairs':1,'schema_error':failure,'conditions':s['conditions'],'protected':protected,'pins':pincheck,
        'Source_policy_unchanged':True,'Prompt_changes':0,'algorithm_changes':0,'new_clinical_grants':0,
        'actual_Run_hashes':{str(f.relative_to(PR/RUN)):sha(f) for f in (PR/RUN).rglob('*') if f.is_file()},
        'report_SHA256':{f.name:sha(f) for f in DOC.glob('*.md')},
        'files_SHA256':{f.name:sha(f) for f in PUB.glob('*') if f.is_file()},
        'finalizer_SHA256':sha(Path(__file__)),'no_automatic_followup':True,
        'verify_command':'python -B scripts/finalize_locdi_pilot_improvement.py verify'})
    pointer='# 최종 실제 결과 안내\n\n준비 문서의NOT_RUN은실행전상태다. 실제새Run synthea_sequential_002는48요청후Schema오류로중단됐다. 완료3/실패1/미실행16,완료쌍1/10이다. 최신8개보고서는final_results_v1/에있다. 추가Generation0,자동재실행없음.\n'
    for base in (PREP,PREPDOC):
        with (base/'FINAL_ACTUAL_RESULT_POINTER.md').open('x',encoding='utf-8') as f:f.write(pointer)
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['publish','verify']);a=parser.parse_args()
    main() if a.mode=='publish' else verify()
