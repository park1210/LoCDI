"""Offline stable B1/subset-wire preparation and separately authorized pilot."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
import scripts.prepare_b012_accuracy_recovery as old
from scripts.prepare_synthea_sequential_pilot import test_authority
from scripts.prepare_locdi_pilot_improvement import MODULES,saved_backend_replay
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.llm_setup import digest
from app.locdi.current_ablation_runner import features_for,_process_private_bytes
from app.locdi.current_ablation_llm_backend import input_guard,make_er_backend
from app.locdi.memory_subset_wire import encode,decode,FIELD,VERSION as WIRE
from app.locdi.context_budgeted_history import (build_plan,ContextBudgetedHistory,MAX_JOINT_BYTES,FUTURE_RESERVE_BYTES,VERSION as HISTORY)
from app.core.longitudinal_memory import PatientMemoryStore
from app.locdi.synthea_sequential_proxy import policy_contract,SequentialSyntheticSourceView,validate_permission
from app.locdi.episode_exporter import build_episode
from app.locdi.run_simulation import run_case
from app.locdi.b012_pilot import schedule,collect,aggregate
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.memory_paired_recovery import SharedRecoveryStop,verify_delivered_memory

NS='FINAL_B012_MEMORY_STABILITY_RECOVERY_v1'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
DOC=ROOT/'docs/연구 평가'/NS
RUN='synthea_b012_subset_stable_001'
DATA=old.DATA


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def prepare():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_PREPARATION_REQUIRED')
    begin=time.monotonic();protected=read(PR/'preservation_before.json');assert check(protected)['status']=='PASS'
    for name in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json'):
        write(PR/name,read(old.PR/name))
    cases=load_cases(DATA);panel=read(PR/'panel.json');index=read(PR/'case_index.json');source=read(PR/'source_rows.json');catalog=read(CATALOG)
    config=read(PR/'llm_config.json');prior=read(old.PUB/'B012_EXECUTION_PROTOCOL.json')
    baseline={r['patient_ordinal']:r['max_input_bound'] for r in read(old.PUB/'B012_offline_preflight.json') if r['condition']=='B0'}
    plans=[];plan_rows=[]
    for i,p in enumerate(panel,1):
        adapted=adapt_case(cases[index[p['case_ref']]-1]);identity=adapted.runtime_case['longitudinal']
        store=PatientMemoryStore([r for v in adapted.runtime_case['past_visits'] for r in v['records']])
        full=store.retrieve(identity,'','full_history',max_chars=None)
        budget=max(0,min(MAX_JOINT_BYTES,32768-1024-512-baseline[i]-FUTURE_RESERVE_BYTES))
        plan=build_plan(identity,full,budget);plans.append(plan)
        plan_rows.append({'patient_ordinal':i,'candidate_records':plan['candidate_count'],'selected_records':plan['selected_count'],
            'excluded_records':plan['excluded_count'],'exclusion_reason':'COMPLETE_UNIT_JOINT_CONTEXT_BUDGET',
            'joint_budget_bytes':budget,'joint_cost_bytes':plan['joint_cost_bytes'],'selected_json_bytes':plan['selected_json_bytes'],
            'baseline_no_memory_Mock_max_bound':baseline[i],'future_extra_reserve_bytes':FUTURE_RESERVE_BYTES,
            'time_range':'PRIVATE_HISTORY_SELECTION_PLAN','selection_fingerprint':plan['fingerprint']})
    write(PR/'history_selection_plans.json',plans);write(PUB/'B1_history_selection_summary.json',plan_rows)
    history=ContextBudgetedHistory(plans)
    spec=policy_contract(source,test_only=True,allow_full_history=True)
    view=SequentialSyntheticSourceView(source,spec=spec,authorization=test_authority(spec),plan_fingerprint='TEST_ONLY_PROBE',catalog=catalog)
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from test_simulation import SourceActionClient,smoke_options
    from jsonschema import Draft202012Validator
    class Client(SourceActionClient):
        def __init__(self):super().__init__();self.bounds=[];self.first=None
        def chat_json(self,system,user,model=None,*,schema=None):
            payload=view.compact_payload(json.loads(user));wire,wire_schema,mapping=encode(payload,schema)
            bound=input_guard(system,json.dumps(wire,ensure_ascii=False,sort_keys=True),wire_schema);self.bounds.append(bound)
            if self.first is None:self.first={'payload':payload,'wire_payload':wire,'schema':schema,'wire_schema':wire_schema}
            value=json.loads(super().chat_json(system,json.dumps(payload,ensure_ascii=False,sort_keys=True),model=model,schema=schema))
            # TEST_ONLY emits no diagnosis/citations. It uses actual Source
            # actions and real Core validators, never a fabricated outcome.
            value.pop('memory_ids',None);value[FIELD]='0'*max(1,len(mapping['records']))
            Draft202012Validator(wire_schema).validate(value)
            return json.dumps(decode(value,mapping,schema),ensure_ascii=False)
    probes=[];boundary=[]
    for task in schedule(panel):
        c=cases[index[task['case_ref']]-1];client=Client();is_b1=task['condition']=='B1'
        raw=run_case(c,options=smoke_options(client,runner_options=features_for(task['condition']),max_memory_chars=None,
            diagnosis_catalog=catalog,sequential_source_view=view,history_policy=history if is_b1 else None))
        status='READY' if raw['status']=='completed' and len(client.calls)>0 else 'CORE_OR_CONTEXT_BLOCKED'
        episode=build_episode(raw,context=adapt_case(c)) if status=='READY' else None
        names={r['test_name'] for r in raw.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])}
        events=raw['execution']['visits'][0]['final_state']['event_log'] if raw.get('execution') and raw['execution'].get('visits') else []
        received={e['payload']['requested_test']:e['minute'] for e in events if e['event_type']=='result_received' and e['payload'].get('requested_test') in names}
        ids=[];memory_bytes=[]
        for call in client.calls:
            o=call['payload']['observation'];memory=o.get('longitudinal_memory',{});records=memory.get('records',[])
            ids.append([r['memory_id'] for r in records]);memory_bytes.append(len(json.dumps(records,ensure_ascii=False).encode('utf-8')))
            if is_b1:
                assert ids[-1]==plans[task['patient_ordinal']-1]['selected_ids']
                assert memory['budget_excluded_count']==plans[task['patient_ordinal']-1]['excluded_count']
            elif task['condition']=='B0':assert not records
            elif records:assert verify_delivered_memory(c,memory)['status']=='PASS'
            for finding in o['objective_findings']:
                for name in names:
                    if finding.startswith(name+':') and (name not in received or o['elapsed_minutes']<received[name]):
                        boundary.append({'episode':task['episode_ordinal'],'code':'PRE_RECEIPT_EXPOSURE'})
            assert not {'gold','evaluation','task_annotations','provenance','source_event_store'}&set(call['payload'])
        if is_b1 and ids:assert all(x==ids[0] for x in ids)
        write(PR/('TEST_ONLY_Episode%04d.json'%task['episode_ordinal']),{'raw':raw,'episode':episode,'first_request':client.first,
            'model_payloads':client.calls,'input_bounds':client.bounds,'TEST_ONLY':True})
        probes.append({**task,'condition_policy':HISTORY if is_b1 else task['condition'],'status':status,'Core':raw['status'],
            'Exporter':'PASS' if episode else 'NOT_EXPORTED','memory_records':len(ids[0]) if ids else None,
            'stable_memory_set':all(x==ids[0] for x in ids) if is_b1 and ids else None,
            'memory_bytes_min':min(memory_bytes,default=None),'memory_bytes_max':max(memory_bytes,default=None),
            'Mock_decisions':len(client.calls),'max_input_bound':max((x['conservative_input_token_bound'] for x in client.bounds),default=None)})
    write(PUB/'B012_offline_preflight.json',probes);write(PUB/'source_time_boundary_checks.json',{'TEST_ONLY':True,'generation':0,'violations':boundary})
    modules=MODULES+['test_memory_citation_wire','test_b012_pilot','test_memory_subset_wire','test_context_budgeted_history']
    with (PR/'related_regression.log').open('x',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    tests={'tests_run':result.testsRun,'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),'scope':'17 relevant modules; not entire repository','generation':0}
    write(PUB/'test_results.json',tests)
    import scripts.prepare_locdi_pilot_improvement as replay_module
    with patch.object(replay_module,'PR',PR):
        replay=saved_backend_replay(cases[index[panel[0]['case_ref']]-1],config,old.PR/old.RUN/'requests/Episode0001')
    write(PUB/'TEST_ONLY_saved_backend_scoring.json',replay)
    oldfailure=read(old.PR/'actual_failure_matching_branch.json');write(PUB/'old_failure_reproduction.json',oldfailure)
    write(PUB/'sequential_policy.json',policy_contract(source,allow_full_history=True))
    code=['app/locdi/current_ablation_llm_backend.py','app/locdi/run_simulation.py','app/locdi/episode_exporter.py',
        'app/locdi/context_budgeted_history.py','app/locdi/memory_subset_wire.py','app/locdi/memory_citation_wire.py',
        'app/locdi/synthea_sequential_proxy.py','app/locdi/b012_pilot.py','scripts/prepare_b012_memory_stability.py',
        'tests/locdi/test_memory_subset_wire.py','tests/locdi/test_context_budgeted_history.py']
    code += [str(f.relative_to(ROOT)).replace('\\','/') for d in ('app/core','app/llm') for f in (ROOT/d).glob('*.py')]
    p={k:deepcopy(prior[k]) for k in ('model','model_identifier','original20_ordinals','context','temperature','seed','max_output',
        'timeout_seconds','total_generation','per_episode','wall_seconds','process_bytes','output_bytes','Routing','VoI','freeze_criterion')}
    p.update(version='synthea.B012.stable_history_subset.v1',namespace=NS,run_id=RUN,response_contract=WIRE,
        registered_patients=10,registered_episodes=30,conditions=['B0','B1','B2'],schedule=schedule(panel),
        condition_names={'B0':'Memory OFF','B1':'B1_CONTEXT_BUDGETED_HISTORY','B2':'Selective Top5'},
        Memory={'B0':'none','B1':HISTORY,'B2':'selective Top5 unchanged'},history_selection_fingerprint=history.fingerprint,
        source_rows_fingerprint=digest(source),policy_fingerprint=read(PUB/'sequential_policy.json')['fingerprint'],catalog_SHA256=sha(CATALOG),
        input_hashes={str(PR/n):sha(PR/n) for n in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json','history_selection_plans.json')},
        binding_hashes=prior['binding_hashes'],code_hashes={f:sha(ROOT/f) for f in sorted(set(code))},
        preflight_SHA256=sha(PUB/'B012_offline_preflight.json'),tests_SHA256=sha(PUB/'test_results.json'),
        history_metadata=history.metadata(),Retry=0,Repair=0,Warmup=0,no_resume=True,
        actual_enum_support='NOT_CHECKED_WITH_REAL_MODEL; first registered requests verify Native enum/context/digest before continuing',
        clinical_accuracy='NOT_EVALUATED',additional_clinical_prompt_versions=0,
        status='NEW_AUTHORIZATION_REQUIRED' if result.wasSuccessful() and not boundary and all(x['status']=='READY' for x in probes) else 'OFFLINE_BLOCKED',
        stop_policy='first unexpected technical/Source/Identity/context/global limit error stops all remaining requests',
        max_READY_input_bound=max(x['max_input_bound'] or 0 for x in probes),created_at=datetime.now(timezone.utc).isoformat())
    p['fingerprint']=digest(p);write(PUB/'B012_EXECUTION_PROTOCOL.json',p)
    reports(p,plan_rows,probes,tests,replay,oldfailure)
    preserved=check(protected);assert preserved['status']=='PASS'
    write(PUB/'preparation_manifest.json',{'status':p['status'],'protocol':p['fingerprint'],'generation':0,'fit_calls':0,
        'tests':tests,'protected':preserved,'Mock_Episodes':len(probes),'boundary':boundary,
        'elapsed_seconds':time.monotonic()-begin,'reports':{f.name:sha(f) for f in DOC.glob('*.md')}})
    print(json.dumps({'status':p['status'],'protocol':p['fingerprint'],'tests':tests,'states':dict(Counter(x['status'] for x in probes)),
        'B1_selected':[x['selected_records'] for x in plan_rows],'max_bound':p['max_READY_input_bound'],'protected':preserved}))


def reports(p,plans,probes,tests,replay,failure):
    common=f'\n\n계획 `{p["fingerprint"]}`, Model `{p["model_identifier"]}`, Source `{p["source_rows_fingerprint"]}`, Catalog `{p["catalog_SHA256"]}`. '
    common+='실제 신규 Generation0, 임상 정확도/Verified Concept Match는 NOT_EVALUATED다. 기존 결과·Source·Scorer·Core 수식·Evidence Validator는 보존한다.\n'
    table='|선정환자|후보|선택|제외|공동 예산 bytes|예상 공동 비용 bytes|\n|---|---:|---:|---:|---:|---:|\n'
    for r in plans:table+=f'|{r["patient_ordinal"]}|{r["candidate_records"]}|{r["selected_records"]}|{r["excluded_records"]}|{r["joint_budget_bytes"]}|{r["joint_cost_bytes"]}|\n'
    md('B1_CONTEXT_BUDGETED_HISTORY_DESIGN.md','# B1 Context-Budgeted History 설계\n\n'
        '원래full_history를변경하지않고명시적opt-in 어댑터로B1_CONTEXT_BUDGETED_HISTORY를정의했다. '
        '모든적격과거기록을후보로사용하고available_at 최신순/ID타이순으로완전한기록또는명시된방문블록을검토한다. '
        '큰단위가안들어가면통째로제외하고더작은다른기록을계속검토하며최종기록은시간순으로제공한다. '
        '이자료의Memory 스키마는환자/방문/시각/원문을기록마다가지는자기완결단위이며별도필수방문블록은선언돼있지않다. '
        '필요한경우atomic_groups 계약으로방문블록전체선택/제외를지원하고TEST_ONLY로검증했다. '
        '값·단위·시각·JSON필드를자르거나요약하지않고superseded 의미를보존한다.\n\n'
        'Context32768에서출력1024·기존margin512를먼저제외한다. 기존B0 Source Fake 최대프레임에추가미래정보2048 bytes를예약한뒤 '
        'Memory와인용enum Schema공동예산을최대4096 bytes로고정한다. Schema분기복제6회·wire512·기록별주석32 bytes도비용에포함한다. '
        '선택은환자별실행전에고정하고어떤후속turn에서도조용히다시줄이지않는다. 기존runtime byte/schema/context guard와실제서버Token/Context검사를유지하며초과시실패로남긴다. '
        '실측토큰으로예산을맞춘것이아니라현재실행계약의보수적Runtime계산이며,Mock와다른향후선택은여전히차단될수있다.\n\n'+table+
        '\n이조건은원래B1 Full History와다르며부분이력선택을숨기지않는다.'+common)
    md('B1_HISTORY_SELECTION_VALIDATION.md','# B1 선택 무결성\n\n'+table+
        '\n기록별원천필드와fingerprint,같은환자/이전방문/index cutoff를검사한다. '
        '실제모델에제공할기록ID집합을모든후속Mock요청에서동일하게유지했다. '
        'Source값/단위/시각은원래완전한record 그대로이고원래missing값은생성하지않았다. '
        '시점범위·선택/제외ID·제외사유는Private history_selection_plans.json에보존한다. '
        'B2는기존관련성Top5로매turn재검색하므로B1과정보량이같다고주장하지않는다. 실제Memory bytes와총prompt token은별도지표로기록한다.'+common)
    md('B2_MEMORY_SUBSET_WIRE_CONTRACT.md','# 최대4개 부분집합 enum 계약\n\n'
        'memory_citation_subset_code.v1은 memory_subset_code 한문자열을반환한다. '
        '현재제공순서의위치마다0/1을사용하고원래최대4개이하·미제공/superseded 제외조건으로유한enum을생성한다. '
        '5개기록에서00000은없음,10100은0/2,11110은0~3이며11111은enum에존재하지않는다. '
        '0개기록의코드는0이고다른위치선택을허용하지않는다. '
        '모든Core anyOf Action분기의닫힌객체에새필드를적용하고원래내부Schema를변환후다시검증한다. '
        'Raw wire와internal_response를Private checkpoint에각각보존한다. '
        '기존5개를사후4개로자르거나검색Top5/인용상한4/EvidenceValidator를변경하지않는다. '
        '기존실패응답은새유효응답으로승격하지않았다. Native Fake Transport에서format enum전송·원문보존·내부변환을확인했지만 '
        '실제Ollama enum준수는새승인후등록Task에서검증해야하며현재NOT_CHECKED다.'+common)
    md('B2_MEMORY_CITATION_REGRESSION.md','# 인용 회귀\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\n0~4개선택,5개금지,미제공/중복원천ID/superseded/잘못된폭/코드/기존배열거부를검증했다. '
        'Source검사와원래Schema의Evidence citation 검증을유지하며Raw→Internal 단계를각각검사한다. '
        '기존실제오류재현: '+json.dumps(failure,ensure_ascii=False)+common)
    md('B012_FULL_PIPELINE_REGRESSION.md','# 전체 실행 경로의 오프라인 검증\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\n동일10명×3조건은실제Runner/Scheduler/Adapter/Exporter와TEST_ONLY Source Client로실행했다. '
        'Native HTTP형식은Fake Transport 유닛테스트로구분하고,저장된Core결과의Backend/Scorer 후처리는다음비생성검증으로분리했다. '
        '완료된Runner결과를가짜로만들지않으며Mock완료를진단성능으로쓰지않는다. 전체저장소회귀로확대하지않는다.\n\n'
        +json.dumps(replay,ensure_ascii=False)+common)
    md('B012_NEXT_EXECUTION_PROTOCOL.md','# 새 실행 계획\n\n'
        f'Run `{RUN}`. 같은original20 ordinal `{p["original20_ordinals"]}`, 환자별B0→B1_CONTEXT_BUDGETED_HISTORY→B2 총30개다. '
        '기존3B/Digest·임상Prompt·공통15Catalog·Source345행중선정군175행·같은순차Proxy·Scorer/EvidenceValidator를유지한다. '
        '각12/총360 Generation 이하,Context32768/temperature0/seed42/출력1024/요청120초/Retry·Repair·Warmup0, '
        'wall45분/Python4GiB/output1GiB/no-resume이다. 첫기술/Source/Identity/Context/전역한도오류시전체중단한다. '
        '첫등록요청으로모델이로드되면actual Context/Digest/enum을확인하고후속요청을진행한다. 별도Warmup은없다. '
        'Source Recorded-Time Proxy와합성TAT5분은실제임상공개시간아니며ordinal12다음날Reference한계도유지한다. '
        '신규승인없이모델을호출하지않고기존실패Run을재사용하지않는다. '
        '명령: `python -B scripts/prepare_b012_memory_stability.py execute --execute --allow-llm --authorization <새승인JSON>`.'+common)
    md('B012_EXECUTION_APPROVAL_REQUEST.md','# 새 운영자 승인 요청 범위\n\n'
        f'고정계획 `{p["fingerprint"]}`과B1선택정책·부분집합출력계약에대한새실행승인이필요하다. '
        '등록30개/최대360회와위자원한도,다른Ollama작업없음,실제Context/Digest/Schema매회검사,첫오류중단/부분보존을명시한다. '
        '서버변경·다운로드·모델/임상Prompt최적화·다른환자·B3/B4·학습/fitting은포함하지않는다. '
        '단일호출/쓰기중일시자원초과가능성이있으며,Mock최대입력과다른실제후속입력은기존guard가절단없이차단할수있다. '
        '유효한출력만원래Core계약으로변환하며실패응답보정은없다. '
        '같은10명에서진단또는진료과Source일치2/10이상은개발최적화중단기준일뿐임상승인이나전체실행성공이아니다.'+common)
    md('FINAL_B0_B4_READINESS.md','# 최종 준비 범위\n\n'
        'B0/B1(Context-Budgeted)/B2는새프로토콜의오프라인결과를기준으로준비상태를표시하며실제모델안정성/정확도는아직NOT_RUN이다. '
        'B1을원래완전Full History와동일시하지않고선택/제외/bytes/시점범위와B2정보량차이를공개한다. '
        'B3는적격RoutingUtility/독립Reference/현재Binding자료가없어BLOCKED, B4도B3차단및확대메뉴의likelihood/encoding불일치로BLOCKED다. '
        '기존Routing/VoI수식·Artifact·Scorer·Source/진단Reference를변경하지않았다. '
        '기준2/10을달성해도기술실패·미완료·미실행을함께보고하며B3/B4는자동으로진행하지않는다.'+common)


def execute(path):
    p=read(PUB/'B012_EXECUTION_PROTOCOL.json');a=read(path);spec=read(PUB/'sequential_policy.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}) or p['status']!='NEW_AUTHORIZATION_REQUIRED':raise ValueError('PROTOCOL_NOT_READY')
    validate_permission(a,spec,p['fingerprint'],test_only=False)
    if (a.get('run_id')!=RUN or a.get('max_generation')!=360 or a.get('history_selection_fingerprint')!=p['history_selection_fingerprint']
            or a.get('response_contract')!=WIRE):raise ValueError('NEW_HISTORY_SUBSET_EXECUTION_AUTHORITY_REQUIRED')
    def pins():
        for group in ('input_hashes','binding_hashes','code_hashes'):
            if check(p[group])['status']!='PASS':raise SharedRecoveryStop('PIN_CHANGED')
        if sha(PUB/'B012_offline_preflight.json')!=p['preflight_SHA256'] or sha(PUB/'test_results.json')!=p['tests_SHA256']:raise SharedRecoveryStop('PREFLIGHT_CHANGED')
    pins();assert check(read(PR/'preservation_before.json'))['status']=='PASS'
    run=PR/RUN;run.mkdir(parents=True,exist_ok=False);write(run/'authorization.json',a)
    cases=load_cases(DATA);index=read(PR/'case_index.json');config=read(PR/'llm_config.json');source=read(PR/'source_rows.json')
    history=ContextBudgetedHistory(read(PR/'history_selection_plans.json'))
    if history.fingerprint!=p['history_selection_fingerprint']:raise ValueError('HISTORY_SELECTION_CHANGED')
    view=SequentialSyntheticSourceView(source,spec=spec,authorization=a,plan_fingerprint=p['fingerprint'],catalog=read(CATALOG))
    for file,expected in read(PR/'raw_file_hashes.json').items():
        if sha(Path(file))!=expected:raise SharedRecoveryStop('RAW_SOURCE_CHANGED')
    stats={k:(Path(k).stat().st_size,Path(k).stat().st_mtime_ns) for k in read(PR/'raw_file_hashes.json')}
    start=time.monotonic()
    def guard():
        pins()
        if time.monotonic()-start>p['wall_seconds']:raise SharedRecoveryStop('WALL_LIMIT')
        if _process_private_bytes()>p['process_bytes']:raise SharedRecoveryStop('MEMORY_LIMIT')
        if sum(f.stat().st_size for base in (PUB,PR) for f in base.rglob('*') if f.is_file())>p['output_bytes']:raise SharedRecoveryStop('OUTPUT_LIMIT')
        if any((Path(k).stat().st_size,Path(k).stat().st_mtime_ns)!=s for k,s in stats.items()):raise SharedRecoveryStop('RAW_SOURCE_CHANGED')
    backend=make_er_backend({k:cases[v-1] for k,v in index.items()},config=config,authorized=True,
        contract_fingerprint=p['fingerprint'],checkpoint_root=run/'requests',diagnosis_catalog=read(CATALOG),
        sequential_source_view=view,response_contract=WIRE,history_policy=history)
    result=collect(read(PR/'panel.json'),backend,trace_path=run/'episodes.jsonl',preflight=read(PUB/'B012_offline_preflight.json'),
        guard=guard,total_cap=360,per_episode_cap=12,audit_failure=lambda t,e:{'error_type':e,'backend_episode_ordinal':t.get('backend_episode_ordinal'),'raw_preserved':True})
    write(run/'collection.json',result);write(run/'aggregate.json',aggregate(read(PR/'panel.json'),result['rows']))
    write(PUB/'actual_execution_summary.json',{'generation':result['generation_count'],'stop':result['global_stop_reason'],'wall_seconds':time.monotonic()-start})
    print(json.dumps({'generation':result['generation_count'],'stop':result['global_stop_reason']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','execute'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true');parser.add_argument('--authorization',type=Path)
    a=parser.parse_args()
    if a.mode=='execute':
        if not(a.execute and a.allow_llm and a.authorization):parser.error('NEW_EXPLICIT_EXECUTION_AUTHORITY_REQUIRED')
        execute(a.authorization)
    else:prepare()
