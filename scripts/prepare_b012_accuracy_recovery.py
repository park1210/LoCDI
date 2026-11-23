"""Generation0 preparation; one newly approved, bounded B0/B1/B2 run only."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.prepare_synthea_sequential_pilot import test_authority
from scripts.prepare_locdi_pilot_improvement import MODULES
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.synthea_sequential_proxy import policy_contract,SequentialSyntheticSourceView,validate_permission
from app.locdi.memory_citation_wire import VERSION as WIRE,encode,decode
from app.locdi.current_ablation_llm_backend import input_guard,InputContextBudgetError,make_er_backend
from app.locdi.current_ablation_runner import features_for,_process_private_bytes
from app.locdi.episode_exporter import build_episode
from app.locdi.b012_pilot import schedule,collect,aggregate
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.routing_data_reuse import BANK
from app.locdi.memory_paired_recovery import SharedRecoveryStop

NS='FINAL_LOCDI_B0_B4_EVALUATION_AND_ACCURACY_RECOVERY_v1'
BASE_PUB=ROOT/'outputs/research_evaluation'/NS;BASE_PR=ROOT/'outputs/private/research_evaluation'/NS
BASE_DOC=ROOT/'docs/연구 평가'/NS
PUB=BASE_PUB/'revision_02';PR=BASE_PR/'revision_02';DOC=BASE_DOC/'revision_02'
OLD=ROOT/'outputs/private/research_evaluation/FINAL_LOCDI_PILOT_BASED_ACCURACY_IMPROVEMENT_v2'
OLDPUB=ROOT/'outputs/research_evaluation/FINAL_LOCDI_PILOT_BASED_ACCURACY_IMPROVEMENT_v2'
DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
RUN='synthea_b012_citation_set_001'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def routing_voi(cases,panel,index,source):
    from validation.routing_training_data import validate_training_data
    holdout={c['patient_id'] for c in cases}
    bank=[json.loads(x) for x in BANK.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    bankpatients={x['training_patient_id'] for x in bank}
    selected={cases[index[p['case_ref']]-1]['patient_id'] for p in panel}
    registry_path=ROOT/'outputs/research_evaluation/final_exp_v1/routing_3b_agent_registry.json'
    gate=validate_training_data(None,registry_manifest=read(registry_path),source_catalog=None,label_catalog=None,
        holdout_patient_ids=sorted(holdout),dataset_fingerprint=sha(DATA))
    oldutility=ROOT/'outputs/research_evaluation/final_exp_v1/routing_utility_eligibility.json'
    routing={'status':'BLOCKED','code':'ROUTING_UTILITY_BLOCKED','min_samples':2,
        'bank_rows':len(bank),'bank_unique_patients':len(bankpatients),'selected10_overlap_with_bank':len(selected&bankpatients),
        'Original50_overlap_with_bank':len(holdout&bankpatients),'current_validator_without_reviewed_input':gate,
        'latest_3B_utility_evidence':read(oldutility),'latest_3B_evidence_SHA256':sha(oldutility),
        'trained_utility_created':False,'new_agent_calls':0,'fit_calls':0,
        'blockers':['NO_CURRENT_BINDING_COMPLETE_SCORABLE_TRAIN_GROUP','INDEPENDENT_CORRECTNESS_NOT_REVIEWED',
            'NO_VALIDATED_SPECIALTY_SOURCE_HANDOFF_FOR_THIS_PANEL'],
        'threshold_does_not_admit_artifact':True}
    write(PUB/'routing_B3_readiness.json',routing)
    modelpath=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json';model=read(modelpath)
    keys=[json.loads(k) for k in model['provenance']['action_key_mapping']]
    represented={(x[2],x[3]) for x in keys if x[0]=='Test'}
    menu={(r['source_event']['target']['code'],r['source_event']['evidence'].get('units')) for r in source}
    voi={'status':'BLOCKED','fixed_horizon':model['horizon'],'model_actions':len(model['actions']),
        'Frozen_M5_SHA256':sha(modelpath),'source_extra_test_code_unit_types':len(menu),
        'M5_test_code_unit_types':len(represented),'extra_types_without_M5_test_mapping':len(menu-represented),
        'new_likelihoods':0,'posterior_Bellman_STOP_H2_changes':0,
        'blockers':['B3_ROUTING_UTILITY_BLOCKED','EXPANDED_NATIVE_MENU_NOT_COMPLETE_M5_ACTION_SPACE',
            'SOURCE_ENCODING_REFERENCE_AND_NATIVE_TERMINAL_DIFFER','NO_ADMITTED_LIVE_COMBINED_B4_ARTIFACT'],
        'existing_C4_35_C4_39_results':'preserved; separate synthetic probability/policy/technical evidence, not B4 accuracy'}
    write(PUB/'voi_B4_readiness.json',voi)
    return routing,voi


def prepare():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_PREPARATION_REQUIRED')
    started=time.monotonic()
    protected=read(PR/'preservation_before.json');assert check(protected)['status']=='PASS'
    for name in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json'):
        write(PR/name,read(OLD/name))
    panel=read(PR/'panel.json');index=read(PR/'case_index.json');config=read(PR/'llm_config.json')
    source=read(PR/'source_rows.json');cases=load_cases(DATA);catalog=read(CATALOG)
    assert validate_setup(config)['static_readiness']=='READY'
    prior=read(OLDPUB/'SYNTHEA_PILOT_PROTOCOL.json')
    assert digest(source)==prior['source_rows_fingerprint'] and len(source)==345
    before=read(OLD/'synthea_sequential_002/requests/Episode0004/Request0012_before.json')
    after=read(OLD/'synthea_sequential_002/requests/Episode0004/Request0012_after.json')
    payload=json.loads(before['request']['messages'][1]['content']);response=json.loads(after['response_envelope']['message']['content'])
    memory=payload['observation']['longitudinal_memory']['records']
    from jsonschema import Draft202012Validator
    errs=list(Draft202012Validator(before['request']['format']).iter_errors(response))
    failure={'status':'REPRODUCED_NO_GENERATION','old_Run':'synthea_sequential_002','request':'Episode0004/Request0012',
        'response_array_count':len(response['memory_ids']),'response_unique_count':len(set(response['memory_ids'])),
        'retrieved_count':len(memory),'retrieved_unique_count':len({x['memory_id'] for x in memory}),
        'validator_errors':[{'validator':e.validator,'path':list(e.path)} for e in errs],
        'old_response_rewritten':False,'new_contract':WIRE,'retrieval_algorithm_changed':False,
        'old_request_SHA256':sha(OLD/'synthea_sequential_002/requests/Episode0004/Request0012_before.json'),
        'old_response_SHA256':sha(OLD/'synthea_sequential_002/requests/Episode0004/Request0012_after.json')}
    write(PUB/'memory_failure_reproduction.json',failure)
    spec=policy_contract(source,allow_full_history=True);write(PUB/'sequential_B012_policy.json',spec)
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from test_simulation import SourceActionClient,smoke_options
    from app.locdi.run_simulation import run_case
    testspec=policy_contract(source,test_only=True,allow_full_history=True)
    view=SequentialSyntheticSourceView(source,spec=testspec,authorization=test_authority(testspec),plan_fingerprint='TEST_ONLY_PROBE',catalog=catalog)
    class Client(SourceActionClient):
        def __init__(self):
            super().__init__();self.bounds=[];self.blocked=False;self.first=None
        def chat_json(self,system,user,model=None,*,schema=None):
            original=view.compact_payload(json.loads(user))
            wire,wire_schema,mapping=encode(original,schema)
            if self.first is None:self.first={'payload':original,'wire_payload':wire,'schema':schema,'wire_schema':wire_schema}
            text=json.dumps(wire,ensure_ascii=False,sort_keys=True)
            try:self.bounds.append(input_guard(system,text,wire_schema))
            except InputContextBudgetError:self.blocked=True;raise
            answer=json.loads(super().chat_json(system,json.dumps(original,ensure_ascii=False,sort_keys=True),model=model,schema=schema))
            inverse={v:k for k,v in mapping.items()}
            answer['memory_ids']={inverse[x]:True for x in answer.get('memory_ids',[])}
            Draft202012Validator(wire_schema).validate(answer)
            return json.dumps(decode(answer,mapping,schema),ensure_ascii=False)
    probes=[]
    for t in schedule(panel):
        c=cases[index[t['case_ref']]-1];client=Client()
        raw=run_case(c,options=smoke_options(client,runner_options=features_for(t['condition']),max_memory_chars=None,
            diagnosis_catalog=catalog,sequential_source_view=view))
        first=client.first;records=first['payload']['observation'].get('longitudinal_memory',{}).get('records',[]) if first else []
        allcount=first['payload']['observation'].get('longitudinal_memory',{}).get('eligible_count') if first else None
        if t['condition']=='B1':
            assert len(records)==allcount
            original={r['memory_id']:r for v in c['agent_visible']['past_visits'] for r in v['records']}
            assert all(all(record.get(k)==v for k,v in original[record['memory_id']].items()) for record in records)
        elif t['condition']=='B0':assert not records
        elif t['condition']=='B2':assert len(records)<=5
        if client.blocked and not client.calls:status='CONTEXT_BLOCKED'
        elif client.blocked:status='LATER_CONTEXT_BLOCKED'
        elif raw['status']=='completed':status='READY'
        else:status='CORE_CONTRACT_BLOCKED'
        export_status='NOT_RUN_CONTEXT_BLOCKED'
        if status=='READY':
            episode=build_episode(raw,context=adapt_case(c));export_status='PASS'
        else:episode=None
        write(PR/('TEST_ONLY_Episode%04d.json'%t['episode_ordinal']),{'raw':raw,'episode':episode,'first_request':first,
            'model_payloads':client.calls,'bounds':client.bounds,'test_only_no_model':True})
        probe={**t,'status':status,'memory_records':len(records),'eligible_records':allcount,
            'full_history_untruncated':t['condition']=='B1' and len(records)==allcount,
            'Core_status':raw['status'],'Exporter':export_status,'Mock_decisions':len(client.calls),
            'max_input_bound':max((x['conservative_input_token_bound'] for x in client.bounds),default=None)}
        if first:
            probe['initial_wire_payload_bytes']=len(json.dumps(first['wire_payload'],ensure_ascii=False,sort_keys=True).encode('utf-8'))
        probes.append(probe)
    write(PUB/'B012_offline_preflight.json',probes)
    modules=MODULES+['test_memory_citation_wire','test_b012_pilot']
    with (PR/'related_regression.log').open('x',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    tests={'tests_run':result.testsRun,'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),'scope':'15 relevant modules, not entire repository','generation':0}
    write(PUB/'test_results.json',tests)
    finish_preparation(started,protected,panel,index,config,source,cases,prior,failure,probes,tests,spec)


def seal_prepared():
    """Finish saved offline evidence after a metadata-only reporting failure."""
    if (PUB/'B012_EXECUTION_PROTOCOL.json').exists():raise ValueError('ALREADY_SEALED_NO_OVERWRITE')
    finish_preparation(time.monotonic(),read(PR/'preservation_before.json'),read(PR/'panel.json'),
        read(PR/'case_index.json'),read(PR/'llm_config.json'),read(PR/'source_rows.json'),load_cases(DATA),
        read(OLDPUB/'SYNTHEA_PILOT_PROTOCOL.json'),read(PUB/'memory_failure_reproduction.json'),
        read(PUB/'B012_offline_preflight.json'),read(PUB/'test_results.json'),read(PUB/'sequential_B012_policy.json'))


def finish_preparation(started,protected,panel,index,config,source,cases,prior,failure,probes,tests,spec):
    routing,voi=routing_voi(cases,panel,index,source)
    allowed=read(PR/'allowed_interface_changes_before.json')
    write(PUB/'interface_changes.json',{f:{'before':v,'after':sha(ROOT/f)} for f,v in allowed.items()})
    code=list(allowed)+['app/locdi/memory_citation_wire.py','app/locdi/b012_pilot.py',
        'scripts/prepare_b012_accuracy_recovery.py','tests/locdi/test_b012_pilot.py','tests/locdi/test_memory_citation_wire.py']
    code += [str(f.relative_to(ROOT)).replace('\\','/') for base in ('app/core','app/llm') for f in (ROOT/base).glob('*.py')]
    policy={'registered_denominator':10,'minimum_successes':2,'metric':'Source diagnosis OR Source department',
        'per_condition_report':True,'count_threshold_in_partial_run_separately':True,'stop_further_prompt_search_when_reached':True,
        'B3_B4_admission_still_required':True,'user_statement':'2/10 or more (20%) then freeze'}
    write(PUB/'freeze_criterion.json',policy)
    binding={str(DATA):sha(DATA),str(CATALOG):sha(CATALOG)}
    binding.update(prior['binding_hashes'])
    p={'version':'synthea.B012.memory_citation_set.v1','namespace':NS,'run_id':RUN,'registered_patients':10,
        'registered_episodes':30,'conditions':['B0','B1','B2'],'schedule':schedule(panel),
        'original20_ordinals':prior['stratified_selection']['selected_original20_ordinals'],
        'model':config['model'],'model_identifier':config['model_identifier'],'response_contract':WIRE,
        'source_rows_fingerprint':digest(source),'policy_fingerprint':spec['fingerprint'],'catalog_SHA256':sha(CATALOG),
        'context':32768,'temperature':0,'seed':42,'max_output':1024,'timeout_seconds':120,'Retry':0,'Repair':0,'Warmup':0,
        'total_generation':360,'per_episode':12,'wall_seconds':2700,'process_bytes':4*1024**3,'output_bytes':1024**3,
        'Memory':{'B0':'none','B1':'full_history max_chars=None, no truncation','B2':'selective Top5 unchanged'},
        'Routing':'ER_ONLY','VoI':False,'clinical_accuracy':'NOT_EVALUATED','no_resume':True,
        'preflight_context_blocked':'Preserve registered row as PREFLIGHT_CONTEXT_BLOCKED; no generation for that row. Continue preflight-READY rows.',
        'technical_error':'First unexpected technical/Identity/Source/global error stops all later READY rows.',
        'freeze_criterion':policy,'preflight_counts':dict(Counter(r['status'] for r in probes)),
        'additional_clinical_prompt_versions':0,'clinical_prompt_file_changes':0,'output_instruction_versions':1,
        'input_hashes':{str(PR/n):sha(PR/n) for n in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json')},
        'binding_hashes':binding,'code_hashes':{f:sha(ROOT/f) for f in sorted(set(code))},
        'preflight_SHA256':sha(PUB/'B012_offline_preflight.json'),'tests_SHA256':sha(PUB/'test_results.json'),
        'status':'NEW_AUTHORIZATION_REQUIRED' if not tests['failures'] and not tests['errors'] and all(x['status'] in ('READY','CONTEXT_BLOCKED') for x in probes) else 'OFFLINE_BLOCKED',
        'created_at':datetime.now(timezone.utc).isoformat()}
    p['fingerprint']=digest(p);write(PUB/'B012_EXECUTION_PROTOCOL.json',p)
    reports(p,probes,tests,failure,routing,voi)
    preserved=check(protected);assert preserved['status']=='PASS'
    write(PUB/'preparation_manifest.json',{'status':p['status'],'protocol':p['fingerprint'],'generation':0,'fit_calls':0,
        'tests':tests,'protected':preserved,'preflight_counts':p['preflight_counts'],'seconds':time.monotonic()-started,
        'reports':{f.name:sha(f) for f in DOC.glob('*.md')}})
    print(json.dumps({'status':p['status'],'protocol':p['fingerprint'],'preflight':p['preflight_counts'],'tests':tests,'protected':preserved}))


def reports(p,probes,tests,failure,routing,voi):
    common=f'\n\n신규 계획 `{p["fingerprint"]}`, Model `{p["model_identifier"]}`, Source `{p["source_rows_fingerprint"]}`. '
    common+='새 실제Generation0, Source Exact/Normalized/department·paired 효과 NOT_RUN. 기존Run/Source/Scorer/임상Gate는 보존한다.\n'
    table='|선정환자|B1 기록수|초기Wire bytes|B1 상태|\n|---|---:|---:|---|\n'
    for r in probes:
        if r['condition']=='B1':table+=f'|{r["patient_ordinal"]}|{r["memory_records"]}|{r["initial_wire_payload_bytes"]}|{r["status"]}|\n'
    md('MEMORY_RESPONSE_CONTRACT_RECOVERY.md','# Memory 응답 계약 수정\n\n'+json.dumps(failure,ensure_ascii=False,indent=2)+
        '\n\n검색원본5개와모델출력4개/고유3개를분리했다. 기존실패를수정하지않는다. '
        '새 memory_citation_index_set.v1은 memory_ids를기록위치→true 객체로요청하며유효한고유키만내부ID목록으로변환한다. '
        'JSON중복키·잘못된위치·superseded참조·false값·기존잘못된배열은거부한다. 기존내부Schema(최대4개인용)와Core EvidenceValidator를다시검증한다. '
        '중복제거/repair/retry가아니고사전에명시된새wire형식이다. Raw wire응답과변환된internal_response/fingerprint를Private checkpoint에구분저장한다. '
        'Model/systemPrompt/clinicalinstruction은유지하고공통wire설명1개버전만추가했다. 원래None옵션은기존계약그대로다.'+common)
    md('FULL_HISTORY_B1_IMPLEMENTATION.md','# B1 Full History 연결\n\n'
        'PatientMemoryStore.retrieve의기존full_history/max_chars=None을재사용했다. 동일환자·다른이전방문·index cutoff이전의기록을시간순그대로보존한다. '
        'superseded는원래full_history정의대로표시하며,새wire인용은non-superseded만선택가능하다. '
        '전체이력을Top5로바꾸거나절단하지않았다. B1범위를명시한새SourcePolicy B012.v1과Exporter1.3.3의엄격한메타데이터검증으로연결했다. '
        '검사결과/TAT/cutoff규칙은기존순차Proxy와같고임상공개승인이아니다.\n\n'+table+
        '\nContext초과는사전차단으로30등록분모에유지한다. Byte guard통과를LLM토크나이저동등성으로표현하지않고실행시actual Context/Digest/Token을계속검사한다.'+common)
    md('B0_B1_B2_REGRESSION_REPORT.md','# 관련 회귀 및 비생성 Core 검증\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\n30개동일환자/조건의SourceCase를실제Runner/Scheduler와TEST_ONLY Client로검사했다. '
        'B1 Context초과는의도한차단이지테스트실패숨김이아니다. 입력/임상값/Memory/새wire매핑·원래Schema·Exporter를검사했다. '
        'Source/Scorer/Prompt파일·Core수식변경0,보호SHA는preparation_manifest.json참조. 전체저장소테스트로확대하지않는다.'+common)
    md('SYNTHEA_DIAGNOSIS_ERROR_ANALYSIS.md','# 진단 오류와 정보 한계\n\n'
        '기존정적D4는4/20, D1/D2/D3는0/20이며기존순차002의완료3건Source진단0/3이다. '
        '15종Catalog가실제payload/diagnose enum에있고Source질환이포함돼있어최근불일치는단순표기차이로확정할수없다. '
        '추가검사24건이수신후모델입력에전달됐지만선택된추가검사는전체메뉴의일부이며임상추론오류/정보부족의기여를분리할근거는부족하다. '
        '초기chief complaint와서술형증상이없는값을만들지않고통증측정항목명을실제증상호소로승격하지않는다. '
        '확인된기술오류인모델Memory출력중복만수정했다. 임상진단Prompt개선은추가하지않고한변경요소로검증한다. '
        '새성능을미리개선으로주장하지않는다.'+common)
    md('FINAL_B0_B1_B2_RESULTS.md','# B0/B1/B2 실행 상태\n\nNOT_RUN — 새승인전실제모델호출없음.\n\n'
        +json.dumps(p['preflight_counts'],ensure_ascii=False,indent=2)+
        '\n\n등록30/최대360회,각12회.사전차단B1은미실행으로따로보존하고다른READY조건과환자를제외하지않는다. '
        '시작후예기치않은첫기술오류에전체중단/no-resume,부분결과보존. '
        '실행: `python -B scripts/prepare_b012_accuracy_recovery.py execute --execute --allow-llm --authorization <새승인JSON>`.'+common)
    md('FINAL_MEMORY_COMPARISON.md','# Memory 비교 및20% 고정 기준\n\n'
        '동일10명의B1−B0/B2−B1/B2−B0를각각계산한다. 등록10명·완료쌍·supported scorable쌍을구분하고unresolved/실패/미실행을임상오답라벨로변환하지않는다. '
        '사용자지정고정기준은조건별진단Source또는진료과Source일치2/10이상이다. 완료소수의비율을20%달성으로바꾸지않는다. '
        '부분실행에서2건을얻더라도완료범위를별도로표시한다. 기준달성후추가Prompt탐색을중단하며B3적격성검토로넘어갈수있지만Utility/Gate를자동통과하지않는다.'+common)
    md('ROUTING_B3_READINESS.md','# B3 Adaptive Routing 적격성\n\n'+json.dumps(routing,ensure_ascii=False,indent=2)+
        '\n\n기존Jaccard/Beta smoothing/min_samples/abstention공식과선택된Agent Registry binding을유지한다. '
        '환자분리0중복은Correctness의독립검토나현재Prompt/Digest에맞는5-Agent Complete Group을대신하지못한다. '
        '현재ER-onlyRegistry를전문과실행검증으로승격하지않고,Source-backedConsult가없는LoCDI50과별도LLMHandoff기술검증도구분한다. '
        '새Utility fitting/라벨생성/Agent호출없음. B3 BLOCKED.'+common)
    md('VOI_B4_READINESS.md','# B4 VoI 적격성\n\n'+json.dumps(voi,ensure_ascii=False,indent=2)+
        '\n\nFrozen M5의8개Action/H2및기존Bayes/Bellman/STOP을보존한다. '
        '확대345행의여러Test code/unit가모델Action과1:1지원되지않고원래categorical Source encoding과rawNativeEvidence가다르다. '
        '어떤메뉴를썼는지무시하고임의likelihood를추가할수없다. B3차단상태에서통합B4를실행하지않으며Source-time기술Probe를임상B4진단성능으로대체하지않는다.'+common)
    md('FINAL_B0_B4_EVALUATION_MATRIX.md','# 단계별 준비 상태\n\n'
        '|조건|현재상태|판정범위|\n|---|---|---|\n'
        '|B0|새승인대기|Memory none,ER-only,확대Source/동일Catalog|\n'
        '|B1|사전Context검사결과참조|untruncated full_history,초과시차단|\n'
        '|B2|새승인대기|기존selectiveTop5,다른입력B0와동일|\n'
        '|B3|BLOCKED|적격Utility/현재Binding Complete Group/독립Reference부족|\n'
        '|B4|BLOCKED|B3차단+확대Action/likelihood/Source encoding불일치|\n'
        '\n20%기준은B0~B2설정고정의개발기준이다. B3/B4를자동실행하거나과거90%를현재근거로사용하지않는다.'+common)
    md('FINAL_THESIS_RESULTS.md','# 논문 반영 범위\n\n'
        '현재새B0/B1/B2진단결과는없고기존정적/순차Run결과는그대로보존한다. '
        '예전논문38/52/44/48/90%와현재3B개발Source평가의모델·입력·분모·Proxy·검증수준차이를명시한다. '
        'B1이Context차단이면30개공통조건평가완료라고기술하지않고B0/B2가능범위와분리한다. '
        'C4-35신규합성VoI/C4-39기술일치검증은별도실험이며이번B4진단정확도가아니다.'+common)
    md('FINAL_PROFESSOR_BRIEFING.md','# 교수님 보고\n\n'
        '현재목표는같은3B/Synthea에서가능한기본진단결과를측정하고진단또는진료과2/10이상을얻으면추가최적화를중단하는것이다. '
        '실제Memory검색은고유5개였고중복은모델출력에서발생했다. 새wire계약으로중복선택을구조화했으며기존실패를수정하지않았다. '
        'B1전체이력은이미있던검색함수를절단없이연결해Context적격성을검사했다. '
        '현재성능개선은미검증이며새30등록Run은운영자승인이필요하다. '
        'B3/B4는정확도목표달성만으로열리지않고독립학습자료와실제Action/확률Source계약이여전히차단한다.'+common)


def execute(path):
    p=read(PUB/'B012_EXECUTION_PROTOCOL.json');a=read(path);spec=read(PUB/'sequential_B012_policy.json')
    if digest({k:v for k,v in p.items() if k!='fingerprint'})!=p['fingerprint']:raise ValueError('PROTOCOL_CHANGED')
    if p['status']!='NEW_AUTHORIZATION_REQUIRED':raise ValueError('OFFLINE_BLOCKED')
    validate_permission(a,spec,p['fingerprint'],test_only=False)
    if a.get('run_id')!=RUN or a.get('max_generation')!=360 or a.get('conditions')!=['B0','B1','B2']:
        raise ValueError('NEW30_EPISODE_SCOPE_AUTHORITY_REQUIRED')
    def verify_pins():
        for group in ('input_hashes','binding_hashes','code_hashes'):
            if check(p[group])['status']!='PASS':raise SharedRecoveryStop('INPUT_CODE_BINDING_CHANGED')
        if sha(PUB/'B012_offline_preflight.json')!=p['preflight_SHA256'] or sha(PUB/'test_results.json')!=p['tests_SHA256']:
            raise SharedRecoveryStop('PREFLIGHT_OR_TESTS_CHANGED')
    verify_pins();assert check(read(PR/'preservation_before.json'))['status']=='PASS'
    run=PR/RUN;run.mkdir(parents=True,exist_ok=False);write(run/'authorization.json',a)
    cases=load_cases(DATA);index=read(PR/'case_index.json');source=read(PR/'source_rows.json')
    config=read(PR/'llm_config.json');catalog=read(CATALOG)
    view=SequentialSyntheticSourceView(source,spec=spec,authorization=a,plan_fingerprint=p['fingerprint'],catalog=catalog)
    for raw,expected in read(PR/'raw_file_hashes.json').items():
        if sha(Path(raw))!=expected:raise SharedRecoveryStop('RAW_SOURCE_CHANGED')
    rawstats={k:(Path(k).stat().st_size,Path(k).stat().st_mtime_ns) for k in read(PR/'raw_file_hashes.json')}
    start=time.monotonic()
    def guard():
        verify_pins()
        if time.monotonic()-start>p['wall_seconds']:raise SharedRecoveryStop('WALL_LIMIT')
        if _process_private_bytes()>p['process_bytes']:raise SharedRecoveryStop('MEMORY_LIMIT')
        if sum(f.stat().st_size for base in (PUB,PR) for f in base.rglob('*') if f.is_file())>p['output_bytes']:
            raise SharedRecoveryStop('OUTPUT_LIMIT')
        if any((Path(k).stat().st_size,Path(k).stat().st_mtime_ns)!=s for k,s in rawstats.items()):raise SharedRecoveryStop('RAW_SOURCE_CHANGED')
    backend=make_er_backend({k:cases[v-1] for k,v in index.items()},config=config,contract_fingerprint=p['fingerprint'],
        checkpoint_root=run/'requests',authorized=True,diagnosis_catalog=catalog,sequential_source_view=view,response_contract=WIRE)
    def audit(t,error):
        return {'error_type':error,'backend_episode_ordinal':t.get('backend_episode_ordinal'),
            'raw_checkpoints_preserved':True,'response_rewritten':False}
    result=collect(read(PR/'panel.json'),backend,trace_path=run/'episodes.jsonl',preflight=read(PUB/'B012_offline_preflight.json'),
        guard=guard,total_cap=360,per_episode_cap=12,audit_failure=audit)
    write(run/'collection.json',result);write(run/'aggregate.json',aggregate(read(PR/'panel.json'),result['rows']))
    write(PUB/'actual_execution_summary.json',{'generation':result['generation_count'],'stop':result['global_stop_reason'],
        'wall_seconds':time.monotonic()-start,'clinical_metrics':'NOT_EVALUATED'})
    print(json.dumps({'generation':result['generation_count'],'stop':result['global_stop_reason']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','seal-prepared','execute'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true');parser.add_argument('--authorization',type=Path)
    a=parser.parse_args()
    if a.mode=='execute':
        if not(a.execute and a.allow_llm and a.authorization):parser.error('NEW_EXPLICIT_EXECUTION_AUTHORITY_REQUIRED')
        execute(a.authorization)
    elif a.mode=='seal-prepared':seal_prepared()
    else:prepare()
