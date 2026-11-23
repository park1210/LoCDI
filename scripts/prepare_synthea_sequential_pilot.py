"""Bounded sequential preparation/approved pilot. Default generation0.

Original studies/reports stay immutable. No new algorithms or clinical grants.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from scripts.run_memory_paired_recovery import inspect_failed_folder
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.current_ablation_llm_backend import make_er_backend,input_guard
from app.locdi.current_ablation_runner import _process_private_bytes
from app.locdi.memory_paired_recovery import collect,aggregate,schedule,SharedRecoveryStop
from app.locdi.synthea_sequential_proxy import (stratified_panel,policy_contract,
    SequentialSyntheticSourceView,validate_permission)

NS='FINAL_SYNTHEA_SEQUENTIAL_VALIDATION_v1'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
DOC=ROOT/'docs/연구 평가'/NS
COM=ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_COMPLETION_v1'
BC=ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'
OLD=ROOT/'outputs/private/research_evaluation/final_exp_recovery_v1'
DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
RUN='synthea_sequential_001'
ALLOWED_CHANGES={'app/locdi/current_ablation_llm_backend.py','app/locdi/run_simulation.py'}
CODE=['app/locdi/synthea_sequential_proxy.py','app/locdi/current_ablation_llm_backend.py',
      'app/locdi/run_simulation.py','scripts/prepare_synthea_sequential_pilot.py','tests/locdi/test_synthea_sequential_proxy.py']


def md(name,text):
    for root in (PUB,DOC):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        data=text.rstrip()+'\n'
        if p.exists():
            if p.read_text(encoding='utf-8')==data:continue
            raise ValueError('REPORT_ALREADY_EXISTS_NO_OVERWRITE')
        with p.open('x',encoding='utf-8') as f:f.write(data)


def load_protocol():
    return read(PUB/'SYNTHEA_PILOT_PROTOCOL.json')


def load_policy():
    final=PUB/'sequential_source_policy_final.json'
    return read(final if final.exists() else PUB/'sequential_source_policy.json')


def test_authority(spec):
    """Nonclinical offline scope only; never accepted by the live client."""
    return {'scope':'SYNTHETIC_SEQUENTIAL_ER_PILOT_ONLY','plan_fingerprint':'TEST_ONLY_PROBE',
        'policy_fingerprint':spec['fingerprint'],'evidence_type':'TEST_ONLY','user_statement':'TEST_ONLY_OFFLINE_NO_LLM',
        'accept_source_record_proxy_cutoff':True,'accept_fixed_five_minute_synthetic_turnaround':True,
        'authorize_sequential_source_and_pilot':True,'accept_reference_timing_limits':True,
        'clinical_release_approved':False,'v1_grants_created':0}


def prepare():
    if PUB.exists() or PR.exists() or DOC.exists():raise ValueError('FRESH_SEQUENTIAL_NAMESPACE_REQUIRED')
    begin=time.monotonic()
    previous=read(COM/'preservation_before_live.json');protected={};changes=[]
    for name,expected in previous.items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        rel=str(p.relative_to(ROOT)).replace('\\','/') if ROOT in p.parents else str(p)
        if rel in ALLOWED_CHANGES:
            changes.append({'file':rel,'historical_SHA256':expected,'new_SHA256':sha(p),'reason':'authorized opt-in interface extension only'})
        else:protected[name]=expected
    for base in ('outputs/research_evaluation/FINAL_SYNTHEA_COMPLETION_v1',
        'outputs/private/research_evaluation/FINAL_SYNTHEA_COMPLETION_v1',
        'docs/연구 평가/FINAL_SYNTHEA_COMPLETION_v1',
        'outputs/research_evaluation/FINAL_SYNTHEA_ACTUAL_INSPECTION_v1',
        'outputs/private/research_evaluation/FINAL_SYNTHEA_ACTUAL_INSPECTION_v1'):
        protected.update({str(p):sha(p) for p in (ROOT/base).rglob('*') if p.is_file()})
    if check(protected)['status']!='PASS':raise ValueError('PROTECTED_ASSET_CHANGED')
    write(PR/'preservation_before.json',protected);write(PUB/'authorized_interface_changes.json',changes)
    cases=load_cases(DATA);oldpanel=read(OLD/'panel.json');index=read(OLD/'case_index.json')
    strata=[]
    for ordinal,row in enumerate(oldpanel,1):
        case=cases[index[row['case_ref']]-1];dept=case['evaluation']['accepted_departments_source_policy']
        if len(dept)!=1:raise ValueError('UNIQUE_SOURCE_STRATUM_REQUIRED')
        strata.append({'patient_ordinal':ordinal,'stratum':dept[0]})
    selected=stratified_panel(strata,42)
    panel=[deepcopy(oldpanel[i-1]) for i in selected];case_index={p['case_ref']:index[p['case_ref']] for p in panel}
    source=read(COM/'proposed_source_rows_NOT_AUTHORIZED.json');catalog=read(CATALOG)
    spec=policy_contract(source)
    write(PR/'source_rows.json',source);write(PR/'panel.json',panel);write(PR/'case_index.json',case_index)
    write(PR/'llm_config.json',read(BC/'llm_config.json'));write(PR/'raw_file_hashes.json',read(COM/'raw_file_hashes.json'))
    write(PUB/'sequential_source_policy.json',spec)
    selection={'seed':42,'rule':'sorted ordinal within Source department; rng.sample5 each, endocrine then neurology; final ordinal sort',
        'selection_inputs':'ordinal+Source stratum ONLY; no D4 outcome file read by selection',
        'selected_original20_ordinals':selected,'selected_original50_ordinals':[case_index[p['case_ref']] for p in panel],
        'strata':dict(Counter(strata[i-1]['stratum'] for i in selected)),
        'remaining_original20_ordinals':[i for i in range(1,21) if i not in selected]}
    write(PUB/'stratified_selection.json',selection)
    # Real selected Source cases through the existing Runner/Scheduler, but only
    # TEST_ONLY deterministic responses. Source outcomes are never fabricated.
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from test_simulation import SourceActionClient,smoke_options
    from app.locdi.run_simulation import run_case
    from dataclasses import replace
    from app.locdi.current_ablation_runner import features_for
    all_probes=[]
    test_spec=policy_contract(source,test_only=True)
    view=SequentialSyntheticSourceView(source,spec=test_spec,authorization=test_authority(test_spec),
        plan_fingerprint='TEST_ONLY_PROBE',catalog=catalog)
    class GuardClient(SourceActionClient):
        def __init__(self):super().__init__();self.bounds=[];self.context_errors=[]
        def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
            try:self.bounds.append(input_guard(system_prompt,user_prompt,schema))
            except Exception as exc:self.context_errors.append(type(exc).__name__)
            return super().chat_json(system_prompt,user_prompt,model=model,schema=schema)
    for n,p in enumerate(panel,1):
        case=cases[case_index[p['case_ref']]-1]
        for condition in ('B0','B2'):
            if time.monotonic()-begin>2400:raise ValueError('IMPLEMENTATION_REVIEW_40MIN_LIMIT')
            client=GuardClient();opts=smoke_options(client,runner_options=features_for(condition),
                diagnosis_catalog=catalog,max_memory_chars=None,sequential_source_view=view)
            result=run_case(case,options=opts)
            write(PR/('TEST_ONLY_Patient%02d_%s.json'%(n,condition)),{'raw':result,'model_payloads':client.calls,
                'input_bounds':client.bounds,'context_errors':client.context_errors})
            added=result.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])
            names={x['test_name'] for x in added}
            visits=(result.get('execution') or {}).get('visits',[])
            events=visits[0]['final_state']['event_log'] if visits else []
            received={e['payload']['requested_test']:e['minute'] for e in events if e['event_type']=='result_received'
                      and e['payload'].get('requested_test') in names}
            violations=[]
            for c in client.calls:
                o=c['payload']['observation']
                for finding in o.get('objective_findings',[]):
                    for name in names:
                        if finding.startswith(name+':') and (name not in received or o['elapsed_minutes']<received[name]):
                            violations.append('PRE_RECEIPT_EXPOSURE')
            all_probes.append({'selected_patient_ordinal':n,'condition':condition,'TEST_ONLY':True,
                'core_status':result['status'],'additional_menu_count':len(names),'additional_received':len(received),
                'decision_count':len(client.calls),'input_context_error_count':len(client.context_errors),
                'max_input_bound':max((x['conservative_input_token_bound'] for x in client.bounds),default=None),
                'temporal_violations':violations})
    write(PUB/'TEST_ONLY_selected10_core_probe.json',all_probes)
    # Context guards are exact on each offline requested payload, but model
    # future choices/rationales differ; the same guard remains active live.
    blocked=[x for x in all_probes if x['input_context_error_count'] or x['core_status']=='failed' or x['temporal_violations']]
    protocol={'version':'synthea.sequential.pilot.v1','namespace':NS,'run_id':RUN,
        'policy_fingerprint':spec['fingerprint'],'source_rows_fingerprint':digest(source),
        'purpose':'SYNTHETIC_SEQUENTIAL_DEVELOPMENT_NOT_OFFICIAL_ABLATION','stratified_selection':selection,
        'schedule':schedule(panel),'registered_patients':10,'registered_episodes':20,'episode_order':'each selected patient B0 then B2',
        'catalog':{'count':15,'path':str(CATALOG),'SHA256':sha(CATALOG),'same_order_all_cases':True},
        'model':'qwen2.5:3b','model_identifier':read(BC/'llm_config.json')['model_identifier'],
        'context':32768,'temperature':0,'seed':42,'max_output':1024,'timeout_seconds':120,
        'max_generations':240,'per_episode':12,'wall_seconds':2700,'process_bytes':4*1024**3,'output_bytes':1024**3,
        'Memory':{'B0':'none','B2':'selective Top5 unchanged fixed source/index cutoff'},'Routing':'ER_ONLY','VoI':False,
        'Retry':0,'Repair':0,'Warmup':0,'minute_limit':120,'no_resume':True,
        'synthetic_new_test_turnaround':5,'source_cutoff_minute':45,
        'technical_failure':'first contract/timeout/context/Source/Identity/global cap failure => stop entire batch; remaining NOT_ATTEMPTED; normal incomplete/unresolved retained',
        'resource_guard':'before/after episode and before every generation; single bounded call/write can overshoot',
        'second_half_combination_rule':'remaining10 only separately approved, unchanged source/catalog/model/prompt/time/scorer/policy; no outcome selection/tuning; register all20, retain failures, report run strata; development exposure remains',
        'pilot_no_improvement':'no repeated code/Prompt optimization; stop automatic expansion',
        'clinical_accuracy':'NOT_EVALUATED','concept_match':'NOT_EVALUATED','source_v1_grants':0,
        'TEST_ONLY_probe_blocked_slots':blocked,
        'input_hashes':{str(PR/n):sha(PR/n) for n in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json')},
        'protected_manifest_sha':sha(PR/'preservation_before.json'),'new_code_hashes':{f:sha(ROOT/f) for f in CODE},
        'created_at':datetime.now(timezone.utc).isoformat()}
    protocol['fingerprint']=digest(protocol);write(PUB/'SYNTHEA_PILOT_PROTOCOL.json',protocol)
    write(PUB/'preparation_timing.json',{'seconds':time.monotonic()-begin,'limit_seconds':2400,'generation':0})
    print(json.dumps({'status':'PREPARED_AUTHORIZATION_PENDING' if not blocked else 'CONTEXT_OR_OFFLINE_BLOCKED',
        'selected':selected,'strata':selection['strata'],'blocked_slots':len(blocked),'fingerprint':protocol['fingerprint']}))


def execute(authorization_path):
    p=load_protocol();spec=load_policy();authorization=read(authorization_path)
    validate_permission(authorization,spec,p['fingerprint'],test_only=False)
    if p['TEST_ONLY_probe_blocked_slots']:raise ValueError('OFFLINE_CONTEXT_OR_CONTRACT_BLOCKED')
    body=deepcopy(p);fp=body.pop('fingerprint')
    if digest(body)!=fp:raise ValueError('PROTOCOL_CHANGED')
    for group in ('input_hashes','new_code_hashes','binding_hashes'):
        for name,expected in p[group].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(path)!=expected:raise ValueError('PIN_CHANGED')
    tests=read(PUB/'test_results_final.json')
    if tests['failures'] or tests['errors']:raise ValueError('REGRESSION_FAILED')
    protected=read(PR/'preservation_before.json')
    if check(protected)['status']!='PASS':raise ValueError('PROTECTED_SOURCE_CHANGED')
    run=PR/RUN;run.mkdir(parents=True,exist_ok=False);write(run/'authorization.json',authorization)
    rows=read(PR/'source_rows.json');catalog=read(CATALOG);config=read(PR/'llm_config.json')
    view=SequentialSyntheticSourceView(rows,spec=spec,authorization=authorization,plan_fingerprint=fp,catalog=catalog)
    cases=load_cases(DATA);index=read(PR/'case_index.json');case_map={k:cases[v-1] for k,v in index.items()}
    for file,expected in read(PR/'raw_file_hashes.json').items():
        if sha(Path(file))!=expected:raise ValueError('RAW_SOURCE_CHANGED')
    raw_stat={k:(Path(k).stat().st_size,Path(k).stat().st_mtime_ns) for k in read(PR/'raw_file_hashes.json')}
    begin=time.monotonic();first_technical_error=[None]
    def guard():
        if first_technical_error[0] is not None:
            raise SharedRecoveryStop('PILOT_FIRST_TECHNICAL_FAILURE:'+first_technical_error[0])
        if time.monotonic()-begin>2700:raise SharedRecoveryStop('GLOBAL_WALL_LIMIT')
        if _process_private_bytes()>p['process_bytes']:raise SharedRecoveryStop('GLOBAL_MEMORY_LIMIT')
        if sum(x.stat().st_size for root in (PUB,PR) for x in root.rglob('*') if x.is_file())>p['output_bytes']:raise SharedRecoveryStop('GLOBAL_OUTPUT_LIMIT')
        if any((Path(k).stat().st_size,Path(k).stat().st_mtime_ns)!=s for k,s in raw_stat.items()):raise SharedRecoveryStop('RAW_SOURCE_CHANGED')
        for group in ('input_hashes','new_code_hashes','binding_hashes'):
            for name,expected in p[group].items():
                path=Path(name) if Path(name).is_absolute() else ROOT/name
                if sha(path)!=expected:raise SharedRecoveryStop('CODE_OR_SOURCE_PIN_CHANGED')
    backend=make_er_backend(case_map,config=config,contract_fingerprint=fp,checkpoint_root=run/'requests',
        authorized=True,diagnosis_catalog=catalog,sequential_source_view=view)
    def run_one(case,features):
        try:return backend(case,features)
        except Exception as exc:
            first_technical_error[0]=type(exc).__name__
            raise
    def failure(task,error):
        return inspect_failed_folder(run/'requests'/('Episode%04d'%task['episode_ordinal']),error)
    panel=read(PR/'panel.json')
    result=collect(panel,run_one,trace_path=run/'episodes.jsonl',total_cap=240,per_episode_cap=12,guard=guard,audit_failure=failure)
    write(run/'collection.json',result);metrics=aggregate(panel,result['rows']);write(run/'aggregate.json',metrics)
    write(PUB/'pilot_execution_summary.json',{'registered_episodes':20,'generation':result['generation_count'],
        'global_stop_reason':result['global_stop_reason'],'wall_seconds':time.monotonic()-begin,'clinical_accuracy':'NOT_EVALUATED'})
    print(json.dumps({'generation':result['generation_count'],'global_stop':result['global_stop_reason']}))


def reports():
    p=load_protocol();tests=read(PUB/'test_results_final.json');selection=p['stratified_selection']
    md('SYNTHEA_SEQUENTIAL_PROXY_CONTRACT.md','# 순차 합성 Source 공개 계약\n\n'
        '정책 SYNTHEA_SEQUENTIAL_RECORDED_TIME_PROXY.v1은 기존 정적v2 및 원래source_check/v1과 별도다. 정적운영자승인을재사용하지 않는다. '
        '원래v1함수/그랜트는변경하지않고, 새순차정책/실행의실제승인이없으면Live생성은차단한다. 임상 reviewer/clinical release/v1 grant를만들지않는다.\n\n'
        '같은345행의 code/value/unit/date, 원래SourceEvent와fingerprint 및 같은환자/방문을검증한다. recorded second upper bound가고정 Source cutoff45분내인것만허용한다. '
        'available_minute는이 upper bound의visit상대분이며 실제임상공개시간아니다. 추가Test의turnaround/occupancy는모두합성5분, lab_slot이며기록시각에서추정하지않는다. '
        'Scheduler ready=max(test_started+5,available_minute); 최초요청이없으면test_started와result_received가없고모델관측에도결과가없다. '
        '원천에없는결과·통증Ask·다른환자/방문·미래기록·Gold/Condition/Department정보를추가하지않는다. Memory의index cutoff/Top5는변경하지않는다.\n\n'
        'Core encounter_evidence는시간조건만확인하는Engine-private함수다. 요청권한검사를그함수단독으로수행한다고주장하지않는다. '
        '실제LLM관측의요청/수신경계는기존Scheduler의order_test→test_started→result_received→objective_findings경로로보장/검증한다. '
        'Case Adapter 원본wrapper는변경하지않으며추가메뉴는검증후Engine복사본에만들어간다. 미존재Source는실패/미공개로남기고합성결과를만들지않는다. '
        'VoI OFF/ER-only/B0·B2에서만허용하며 기존VoI Action Space·likelihood와분리한다. 실제available_at 및18명의intraday진단순서미확정과2명의다음날Reference한계는유지한다.\n')
    md('SYNTHEA_SEQUENTIAL_IMPLEMENTATION_REPORT.md','# 순차 Native ER 연결 구현\n\n'
        '신규 app/locdi/synthea_sequential_proxy.py의SequentialSyntheticSourceView와stratified_panel을구현했다. '
        '기존make_er_backend의Catalog옵션을재사용하고sequential_source_view옵션만추가했다. SimulationOptions/run_cases는기존Validator/Adapter후검증된타입의overlay를적용한다. '
        '기본None은원래free-text/기존메뉴와같다. source_view의OFF복사경로만재사용하며그엄격한grant branch를통과시켰다고주장하지않는다. '
        '기존Core/Memory/VoI/Belief/Scheduler/Evidence Validator/scorer/Prompt/데이터는변경0이다. '
        '수정허용된app/locdi/current_ablation_llm_backend.py와run_simulation.py의이전/새SHA는authorized_interface_changes.json에명시하고과거완료Manifest는갱신하지않았다.\n\n'
        '초기20개TEST_ONLY Probe 중7개슬롯의일부후속입력이기존byte guard를넘었다. 초기차단기록은보존했다. '
        '새VoI-OFF opt-in에서모델출력에쓰이지않는중복action_id/type 및null cost/outcome_model만전송에서제거하고, '
        '결과의code/전체description은test label에유지하면서중복JSONdescription/kind는생략했다. 임상값/unit/date, Memory, 모든action-target선택지는보존했다. '
        'System/지시문/Schema/Evidence Gate/Context guard를완화하지않았으며모델응답을보며Prompt를최적화한것이아니다. '
        '최종동일20개Mock Episode의guard차단0, 공개시각위반0을확인했다.\n\n'
        '새SourceCase20Episode offline실행은TEST_ONLY deterministic client와실제Runner/Scheduler를사용한기술검증이다. '
        '실제LLM결과와혼합하지않는다. 초기및각mock turn의Native byte/schema/context guard를검사했으며실제LLM의다른행동/후속Memory크기는live에서도동일guard로차단한다.\n')
    md('SYNTHEA_SEQUENTIAL_REGRESSION_REPORT.md','# 관련 회귀\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\n실제선정10환자×B0/B2 최종TEST_ONLY CoreProbe는TEST_ONLY_compact_selected10_core_probe.json에분리했다. 초기차단Probe도그대로보존했다. '
        '이것은실제진단성능결과가아니며신규LLM Generation0이다. 전체저장소테스트통과라고확대하지않는다.\n')
    md('SYNTHEA_PILOT_EXECUTION_APPROVAL.md','# 신규 순차 공개·Pilot 실행 승인 요청\n\n'
        f'계획 `{p["fingerprint"]}`. 정책 `{p["policy_fingerprint"]}`. '
        f'선정 original20 ordinal `{selection["selected_original20_ordinals"]}`, strata `{selection["strata"]}`. '
        'Seed42, sorted ordinal에서내분비5/신경5를고정추출했으며D4정오파일을선정에사용하지않았다. 환자별B0→B2총20Episode, sameCatalog15·같은추가메뉴·합성TAT5분이다. '
        '새 synthea_sequential_001, 기존3B/Digest·Context32768·temperature0·seed42·output1024·timeout120초, 각12/총240 Generation이하, wall45분·Python4GiB·output1GiB·Retry/Repair/Warmup0. '
        'Memory B0 OFF/B2 Top5, ER-only·VoI OFF, 기존Evidence Validator/Prompt/scorer 유지. '
        '새순차합성정책과Pilot에대한실제운영자승인을받기전Generation0이다. 정적승인·TEST_ONLY인증은Live에사용할수없다. '
        '서버변경·다운로드·자동재개·나머지10명확장없음. 첫기술오류에서전체중단하고실패/미완료/미실행을분모20에보존한다. '
        'Source/ModelIdentity/Context/전역자원guard는매요청검증하며단일bounded호출/쓰기중일시초과가능하다. '
        '실제임상공개/임상Gold/전문reviewer를승인한것으로표현하지않는다.\n')
    if (PR/RUN/'collection.json').exists():
        result=read(PR/RUN/'collection.json');metrics=read(PR/RUN/'aggregate.json')
        result_text=json.dumps(metrics,ensure_ascii=False,indent=2)
        pairs=metrics['pairs']
    else:
        result_text='NOT_RUN — 신규 순차 합성 정책/실행 승인 미확보. Generation0. TEST_ONLY 결과는 실제Pilot 결과가 아니다.'
        pairs=[{'patient_ordinal':i,'B0_status':'NOT_ATTEMPTED','B2_status':'NOT_ATTEMPTED','complete_pair':False,
            'source_exact_difference':None,'verified_concept_match':'NOT_EVALUATED'} for i in range(1,11)]
    md('SYNTHEA_PILOT_RESULTS.md','# 실제 Pilot 상태\n\n'+result_text+
        '\n\n실제실행시기존수집기는등록/완료/실패/미완료/미실행·Supported/Unresolved·SourceExact·진료과·Generation을저장한다. '
        'Private Request/RawCore/telemetry에서Normalized·추가Test주문/시작/수신·인용·공개시각·Memory검증·token을읽기전용으로집계해야한다. '
        'Source 일치는독립임상정확도가아니며Concept/Clinical accuracy는NOT_EVALUATED다. 기존D4정적결과와합치지않는다.\n')
    csv_file(PUB/'SYNTHEA_PILOT_PAIRED_RESULTS.csv',pairs,list(pairs[0]))
    md('FINAL_B0_B2_DECISION.md','# 최종 B0/B2 판단\n\n'
        '현재는준비/회귀와TEST_ONLY기술검증범위이며실제Pilot은새승인이필요하다. 결과를성공으로추정하지않는다. '
        '실행후개선이없으면반복코드/Prompt최적화를하지않고연구종료자료로남긴다. '
        '나머지10명은별도결정/승인대상이며자동호출하지않는다. 전체20명결합은미리보존한complement를그대로사용하고 '
        '같은모델/Digest·Prompt/Catalog·Source/time정책·scorer·cutoff·행동상한을유지할때만고려한다. '
        '두Run을구분하고전체등록분모·실패·미완료·미실행을모두보존하며완료쌍조건부결과도함께보고한다. '
        'D4정오나Pilot결과로남은환자를선택·교체하거나이전에실패한환자를제외하지않는다. '
        '두패널모두기존개발노출환자라독립임상holdout이라고부르지않는다.\n\n'
        '실행명령: `python -B scripts/prepare_synthea_sequential_pilot.py execute --execute --allow-llm --authorization <실제새순차승인JSON>`; '
        '기존정적승인파일로는실행할수없으며미사용Run에1회만허용한다.\n')
    write(PUB/'preservation_verification.json',check(read(PR/'preservation_before.json')))
    write(PUB/'delivery_manifest.json',{'status':'PREPARED_NEW_SEQUENTIAL_AUTHORITY_PENDING',
        'LLM_generation':0,'Core_changes':0,'clinical_release_created':False,'v1_grants_created':0,
        'tests':tests,'protocol_fingerprint':p['fingerprint'],'reports':{f.name:sha(f) for f in DOC.glob('*.md')},
        'preservation':check(read(PR/'preservation_before.json'))})
    print(json.dumps({'status':'PREPARED_NOT_RUN','protocol':p['fingerprint'],'selection':selection}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','execute','reports'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true');parser.add_argument('--authorization')
    a=parser.parse_args()
    if a.mode=='execute':
        if not (a.execute and a.allow_llm and a.authorization):parser.error('EXECUTE_ALLOW_LLM_AND_NEW_SEQUENTIAL_AUTHORIZATION_REQUIRED')
        execute(Path(a.authorization))
    elif a.mode=='prepare':prepare()
    else:reports()
