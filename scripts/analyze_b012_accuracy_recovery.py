"""Read-only actual B012 result analysis. No model/fit/probability calls."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_b012_accuracy_recovery import PUB as PREP,PR,DOC as PREPDOC,RUN,DATA
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.b012_pilot import aggregate
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.current_ablation_llm_backend import _strict_object
from app.locdi.memory_paired_recovery import citation_audit,verify_delivered_memory
from jsonschema import Draft202012Validator

PUB=PREP/'actual_results_v1';DOC=PREPDOC/'actual_results_v1'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def analyze():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_ACTUAL_ANALYSIS_REQUIRED')
    p=read(PREP/'B012_EXECUTION_PROTOCOL.json');run=PR/RUN;collection=read(run/'collection.json')
    panel=read(PR/'panel.json');index=read(PR/'case_index.json');cases=load_cases(DATA)
    agg=aggregate(panel,collection['rows']);assert agg==read(run/'aggregate.json')
    allsource={digest(r['source_event']):r['source_event'] for r in read(PR/'source_rows.json')}
    rows=[];private=[];checks=Counter();errors=[];boundary=[]
    for task,scored in zip(collection['rows'],agg['episodes']):
        row={k:task[k] for k in ('episode_ordinal','patient_ordinal','condition','status','request_count','error_type','backend_episode_ordinal')}
        row.update(original20_ordinal=p['original20_ordinals'][task['patient_ordinal']-1],
            Core_status=None,export_status='NOT_ATTEMPTED',certainty=scored['certainty'],
            Source_exact=scored['source_exact'],Normalized=scored['normalized'],Source_department=scored['source_department'],
            simulation_minutes=None,selected_actions={},executed_actions=(task.get('result') or {}).get('action_counts'),
            prompt_tokens=0,completion_tokens=0,HTTP_wall_seconds=0,identity_checks=0,recorded_memory_retrievals=None,
            initial_retrieval_minute='NOT_RECORDED',memory_record_occurrences=0,memory_citation_decisions=0,
            additional_test_proposals=0,additional_test_orders=0,additional_test_starts=0,additional_results_received=0,
            additional_citation_decisions=0,source_value_unit_occurrences=0,wire_contract_valid=0)
        folder=run/'requests'/('Episode%04d'%task['backend_episode_ordinal']) if task['backend_episode_ordinal'] else None
        raw=read(folder/'raw_runner_result.json') if folder and (folder/'raw_runner_result.json').exists() else None
        case=cases[index[task['case_ref']]-1];names={};received={};started={};ordered={};details=[]
        if raw:
            row['Core_status']=raw['status'];row['export_status']='EXPORTED' if (folder/'episode_and_raw.json').exists() else 'NOT_EXPORTED'
            row['identity_checks']=raw.get('llm_execution',{}).get('identity_check_count',0)
            row['recorded_memory_retrievals']=len(raw.get('telemetry',{}).get('retrievals',[]))
            names={r['test_name']:allsource[r['source_fingerprint']] for r in raw.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])}
            visits=(raw.get('execution') or {}).get('visits',[])
            if visits:
                state=visits[0]['final_state'];events=state['event_log']
                row['simulation_minutes']=max((e['minute'] for e in events),default=0)
                row['observed_event_counts']=dict(Counter(e['event_type'] for e in events))
                if visits[0].get('failure_detail'):
                    errors.append({'episode':task['episode_ordinal'],'stage':'Core_semantic_or_runtime','detail':visits[0]['failure_detail']})
                for e in events:
                    name=e.get('payload',{}).get('requested_test')
                    if name not in names:continue
                    if e['event_type']=='test_ordered':ordered.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='test_started':started.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='result_received':received.setdefault(name,[]).append(e['minute'])
                for name,minutes in received.items():
                    if not ordered.get(name) or not started.get(name) or min(started[name])<min(ordered[name]) or min(minutes)<min(started[name])+5:
                        boundary.append({'episode':task['episode_ordinal'],'code':'SOURCE_ORDER_START_RECEIPT'})
            decisions=read(folder/'validated_ER_decisions.json');row['selected_actions']=dict(Counter(d['action'] for d in decisions))
            terminal=next((d for d in reversed(decisions) if d['action']=='diagnose'),None)
            if terminal:private.append({'episode_ordinal':task['episode_ordinal'],'condition':task['condition'],
                'original20_ordinal':row['original20_ordinal'],'terminal':terminal,'reference':case['evaluation'],
                'actual_status':task['status'],'actual_export_status':row['export_status']})
            if task['status']=='COMPLETED':
                assert terminal is not None and row['export_status']=='EXPORTED'
                prediction=task['result']['prediction']
                assert prediction['diagnosis']==(terminal['target'] if terminal['certainty']=='supported' else None)
                assert prediction['department']==terminal['department']
                checks['terminal_backend_export_consistent']+=1
        for bpath in sorted(folder.glob('Request*_before.json')) if folder else []:
            apath=bpath.with_name(bpath.name.replace('_before','_after'))
            b=read(bpath);req=b['request'];payload=json.loads(req['messages'][1]['content'])
            if not apath.exists():
                errors.append({'episode':task['episode_ordinal'],'stage':'checkpoint','code':'AFTER_MISSING'});continue
            a=read(apath);checks['before_after_pairs']+=1
            assert b['request_fingerprint']==digest(req)==a['request_fingerprint']
            assert b['response_wire_contract']=='memory_citation_index_set.v1'
            checks['wire_checkpoint_fingerprint']+=1
            env=a.get('response_envelope',{});ident=a.get('identity_after',{})
            row['prompt_tokens']+=env.get('prompt_eval_count',0);row['completion_tokens']+=env.get('eval_count',0)
            row['HTTP_wall_seconds']+=a.get('wall_seconds',0);checks['HTTP200']+=a.get('HTTP_status')==200
            if ident.get('context_length')==32768 and 'sha256:'+str(ident.get('digest','')).removeprefix('sha256:')==p['model_identifier']:
                checks['actual_context_digest_match']+=1
            if a.get('status')!='VALID':errors.append({'episode':task['episode_ordinal'],'request':bpath.name,'stage':'Native_client','error_type':a.get('error_type')})
            response=None
            if env.get('message'):
                try:
                    response=_strict_object(env['message']['content'])
                    Draft202012Validator(req['format']).validate(response)
                    ca=citation_audit(req,env)
                    if ca['error_code'] is not None:raise ValueError(ca['error_code'])
                except Exception as exc:
                    errors.append({'episode':task['episode_ordinal'],'request':bpath.name,'stage':'wire_schema_or_evidence',
                        'error_type':type(exc).__name__,'validator':getattr(exc,'validator',None),
                        'path':list(getattr(exc,'path',[]))})
                else:
                    row['wire_contract_valid']+=1;checks['wire_schema_and_evidence_valid']+=1
                    records=payload['observation'].get('longitudinal_memory',{}).get('records',[])
                    keys=response['memory_ids']
                    if not isinstance(keys,dict):raise ValueError('INVALID_APPROVED_WIRE_OBJECT')
                    expected=[records[int(k)]['memory_id'] for k in keys]
                    internal=a.get('internal_response')
                    if internal is not None:
                        assert digest(internal)==a['internal_response_fingerprint']
                        assert internal.get('memory_ids',[])==expected and len(expected)==len(set(expected))
                        checks['wire_to_internal_mapping_match']+=1
                    row['memory_citation_decisions']+=bool(keys)
                    evidence=payload['evidence_records']
                    row['additional_citation_decisions']+=any(any(evidence[i]['text'].startswith(n+':') for n in names)
                        for i in response.get('evidence_indices',[]) if type(i) is int and 0<=i<len(evidence))
                    if response.get('action')=='request_test' and response.get('target') in names:row['additional_test_proposals']+=1
                    details.append({'request':bpath.name,'wire_response':response,'internal_response':internal})
            observation=payload['observation'];memory=observation.get('longitudinal_memory',{})
            row['memory_record_occurrences']+=len(memory.get('records',[]))
            if task['condition']=='B0' and memory.get('records'):boundary.append({'episode':task['episode_ordinal'],'code':'B0_MEMORY_PRESENT'})
            if task['condition']=='B2' and memory.get('records'):
                if verify_delivered_memory(case,memory)['status']!='PASS':boundary.append({'episode':task['episode_ordinal'],'code':'MEMORY_SOURCE_CUTOFF'})
                checks['B2_memory_source_cutoff_checked']+=1
            assert not {'evaluation','gold','provenance','task_annotations','source_event_store'}&set(payload)
            for finding in observation.get('objective_findings',[]):
                for name,event in names.items():
                    if finding.startswith(name+': '):
                        if not received.get(name) or observation['elapsed_minutes']<min(received[name]):
                            boundary.append({'episode':task['episode_ordinal'],'code':'RESULT_PRE_RECEIPT_EXPOSURE'})
                        value=json.loads(finding[len(name)+2:])
                        assert value['value']==event['evidence']['value'] and value['units']==event['evidence'].get('units')
                        row['source_value_unit_occurrences']+=1
        row.update(additional_test_orders=sum(map(len,ordered.values())),additional_test_starts=sum(map(len,started.values())),
            additional_results_received=sum(map(len,received.values())))
        if details:private.append({'episode_ordinal':task['episode_ordinal'],'wire_and_internal_decisions':details})
        rows.append(row)
    assert len(rows)==30
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    write(PR/'actual_diagnostic_and_wire_analysis.json',private)
    PUB.mkdir(parents=True);DOC.mkdir(parents=True)
    csv_file(PUB/'ACTUAL_B012_EPISODES.csv',rows,fields)
    for name,pair in agg['paired'].items():csv_file(PUB/('PAIRED_'+name+'.csv'),pair['rows'],list(pair['rows'][0]))
    summary={'run_id':RUN,'registered_patients':10,'registered_Episodes':30,'generation':collection['generation_count'],
        'states':dict(Counter(r['status'] for r in rows)),'conditions':agg['conditions'],'paired':agg['paired'],
        'global_stop':collection['global_stop_reason'],'freeze_criterion':agg['freeze_threshold'],
        'checks':dict(checks),'execution_errors':errors,'boundary_violations':boundary,
        'wall_seconds':read(PREP/'actual_execution_summary.json')['wall_seconds'],
        'clinical_accuracy':'NOT_EVALUATED','Verified_Concept_Match':'NOT_EVALUATED','analysis_generation':0}
    for key in ('prompt_tokens','completion_tokens','HTTP_wall_seconds','identity_checks','additional_test_proposals',
        'additional_test_orders','additional_test_starts','additional_results_received','additional_citation_decisions',
        'memory_record_occurrences','memory_citation_decisions','source_value_unit_occurrences'):
        summary[key]=sum(r[key] for r in rows)
    write(PUB/'ACTUAL_B012_METRICS.json',{'summary':summary,'rows':rows,'aggregate':agg})
    reports(p,summary,rows)
    preservation=check(read(PR/'preservation_before.json'));assert preservation['status']=='PASS'
    pins=check({k:v for g in ('input_hashes','binding_hashes','code_hashes') for k,v in p[g].items()});assert pins['status']=='PASS'
    ids=[c['patient_id'] for c in cases]+[c['agent_visible']['index_encounter_id'] for c in cases]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file():assert not any(i in f.read_text(encoding='utf-8-sig') for i in ids)
    write(PUB/'FINAL_ACTUAL_MANIFEST.json',{'status':'ACTUAL_EXECUTION_RECORDED_WITH_LIMITATIONS','summary':summary,
        'protocol':p['fingerprint'],'protected':preservation,'pins':pins,'trace_reaggregation':'EXACT_MATCH',
        'actual_Run_hashes':{str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file()},
        'reports':{f.name:sha(f) for f in DOC.glob('*.md')},'analyzer_SHA256':sha(Path(__file__)),
        'at':datetime.now(timezone.utc).isoformat(),'additional_generation_after_stop':0})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paired','execution_errors')},ensure_ascii=True))


def reports(p,s,rows):
    failure=read(PR/'actual_failure_matching_branch.json')
    first_failure=read(PR/'actual_first_failure_analysis.json')
    tests=read(PREP/'test_results.json')
    routing=read(PREP/'routing_B3_readiness.json');voi=read(PREP/'voi_B4_readiness.json')
    table='|조건|등록|완료|계약 실패|사전 Context 차단|미실행|진단 Source 일치/등록|진료과 Source 일치/등록|Generation|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for c,t in s['conditions'].items():
        st=t['states'];exact=str(t['Source_exact_registered_success_count'])+'/10' if t['supported_scorable'] else 'NOT_EVALUATED'
        dept=str(t['Source_department_registered_success_count'])+'/10' if st.get('COMPLETED',0) else 'NOT_EVALUATED'
        table+=f'|{c}|10|{st.get("COMPLETED",0)}|{st.get("CONTRACT_FAILED",0)}|{st.get("PREFLIGHT_CONTEXT_BLOCKED",0)}|{st.get("NOT_ATTEMPTED",0)}|{exact}|{dept}|{t["generation"]}|\n'
    # Keep prose and all denominators explicit, without labeling failed or
    # unattempted patients as clinical false answers.
    ratio='\n등록 분모의 값은 실제 확보한 Source 일치 수율이다. 실패·미실행을 임상 오답으로 변환하지 않는다. '
    ratio+='B0 완료 응답 중 진단 일치는0/1, 진료과 일치는1/1이다. 전체10명을완료한정확도라고쓰지않는다. B1/B2는완료진단이없어조건부정확도는NOT_EVALUATED다.\n'
    config={k:p[k] for k in ('model','model_identifier','response_contract','context','temperature','seed','max_output',
        'timeout_seconds','total_generation','per_episode','wall_seconds','process_bytes','output_bytes','Memory','Routing','VoI')}
    footer=f'''\n\n## 공통 근거와 재현성

실제 Run `{RUN}`, Protocol `{p['fingerprint']}`.
Model Digest `{p['model_identifier']}`, Source fingerprint `{p['source_rows_fingerprint']}`,
Catalog SHA `{p['catalog_SHA256']}`, Code Pin 집합 `{digest(p['code_hashes'])}`.
같은10명·공통15Catalog·같은Source 공개 규칙·같은임상Prompt/Scorer를 사용했다.
동일 조건에서 추가 생성·재시도·자동Resume·실패응답 수정은 하지 않았다.
기존 242개 관련 회귀와 30개 사전 Source 슬롯 검증을 사용하며, 전체 discovery는 NOT_COMPLETED로 분리한다.
Source 일치는 합성 Source Reference 일치이며 독립 임상 정확도 또는 Verified Concept Match가 아니다.
전체 설정과 행별 분모는 `ACTUAL_B012_METRICS.json`, 30개 등록행은 `ACTUAL_B012_EPISODES.csv`,
환자별 세 비교는 `PAIRED_B1-B0.csv`, `PAIRED_B2-B1.csv`, `PAIRED_B2-B0.csv`에 있다.
환자별 실제 진단·Source·모델 원문·Memory는 Private `actual_diagnostic_and_wire_analysis.json` 및 원래 Run 요청 Checkpoint에 보존했다.
'''
    md('MEMORY_RESPONSE_CONTRACT_RECOVERY.md','# Memory 출력 계약의 실제 검증 결과\n\n'
        '기존 배열 중복 실패를 보존하고 새 객체형 인용 계약을 1회 실제 실행했다. '
        'B0의12회는새wire계약과내부Core 연결을통과했다. B2 첫응답은고유한객체키5개를선택해 기존최대4개제한을위반했다. '
        '객체형이중복방지에도움이될수있어도개수상한까지실제모델이준수한다고검증된것은아니다. Live 계약안정화성공으로표시하지않는다.\n\n'
        +json.dumps(failure,ensure_ascii=False,indent=2)+
        '\n\nRaw 응답은그대로보존했다. 5개중1개를삭제하거나최대4개를5개로늘리지않았고,변환된유효내부응답도만들지않았다. '
        'Memory 검색원본은고유5개이며검색의환자격리/Top5오류로분류하지않는다. '
        '현재원인은출력Schema의기존인용상한미준수다. 모델이왜모든기록을선택했는지단일원인으로확정하지않는다.'+footer)
    b1=[r for r in read(PREP/'B012_offline_preflight.json') if r['condition']=='B1']
    b1table='|선정환자|원본 전체기록 수|최초 wire payload bytes|실제 상태|\n|---|---:|---:|---|\n'
    for r in b1:b1table+=f'|{r["patient_ordinal"]}|{r["memory_records"]}|{r["initial_wire_payload_bytes"]}|PREFLIGHT_CONTEXT_BLOCKED|\n'
    md('FULL_HISTORY_B1_IMPLEMENTATION.md','# B1 Full History 실제 처리\n\n'
        '기존full_history/max_chars=None을연결했으나10명모두기존보수적Context guard를넘어실제Generation0이다. '
        '원천기록을자르거나요약해서Full History라고표시하지않았다. 실제서버토큰측정을한것으로표현하지않는다.\n\n'+b1table+
        '\n새실행의등록30개에는B1 10개를모두사전차단으로보존했다. B1−B0/B2−B1 완료쌍은0이며그효과는NOT_EVALUATED다. '
        '다른환자/미래정보/현재방문이력을섞지않으며원래검색/시간/superseded 규칙은변경하지않았다.'+footer)
    md('B0_B1_B2_REGRESSION_REPORT.md','# 최신 코드 회귀와 실제 실행 한계\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\nB0/B2 20개 Source Fake Core/Exporter 경로와B1 10개초기guard차단을사전확인했다. '
        'Fake 응답의계약통과가실제LLM의상한준수를보장하지않으며이번B2의개수위반이그한계다. '
        '실제검사결과는아래저장Checkpoint 검증과분리한다. 전체discovery는수정전코드실행을중단해NOT_COMPLETED다.\n\n'
        +json.dumps({'actual_checks':s['checks'],'boundary_violations':s['boundary_violations']},ensure_ascii=False,indent=2)+footer)
    md('SYNTHEA_DIAGNOSIS_ERROR_ANALYSIS.md','# 실제 진단과 실패 원인\n\n'
        '새Run의완료진단은B0 1건이며Source Exact/Normalized 불일치,Source 진료과는일치했다. '
        'B2는첫질문행동에서출력계약에실패했으므로유효한최종진단이없다. '
        '같은15종Catalog를모델에제공했고기존정적D4의4/20과다른상호작용·분모를갖는다. '
        '단일완료응답으로Catalog/검사정보/모델추론중한원인을확정하지않는다. '
        '원천chief complaint와서술형증상부족은생성으로채우지않았다. '
        '이번출력오류와진단Source불일치를혼동하지않고정답문자열이나Scorer를바꾸지않는다.\n\n'+table+ratio+footer)
    md('FINAL_B0_B1_B2_RESULTS.md','# 실제 B0/B1/B2 실행 결과\n\n'+table+ratio+
        f'\n실제Generation {s["generation"]}, 전체wall {s["wall_seconds"]:.3f}초. 중단이유 `{s["global_stop"]}`. '
        'B1은미리알려진사전차단이고, B2의첫실제오류뒤다른READY18개를NOT_ATTEMPTED로남겼다. '
        '등록30개와기존Run을그대로보존했다. Timeout/Context/Digest/Source 불일치로추정하지않고저장된실패종류만보고한다.\n\n'
        +json.dumps({'actual_configuration':config,'first_failure':first_failure,'matching_branch_failure':failure},ensure_ascii=False,indent=2)+footer)
    md('FINAL_MEMORY_COMPARISON.md','# Memory 비교 및20% 고정 기준 판정\n\n'+table+ratio+
        '\nB1−B0/B2−B1/B2−B0 모두완료쌍0이다. Memory 효과·paired 차이는NOT_EVALUATED다. '
        'B0 진료과 일치1/10은사용자기준2/10에미달한다. 완료된한건의1/1을기준달성으로바꾸지않는다. '
        '현재새설정은성과기준으로고정됐다고판정하지않는다. 새자동Run이나추가Prompt탐색을시작하지않는다.\n\n'
        +json.dumps(s['freeze_criterion'],ensure_ascii=False,indent=2)+footer)
    md('ROUTING_B3_READINESS.md','# B3 적격성 — BLOCKED\n\n'
        '이번실행은ER-only이며Source 진료과일치한건을전문과별Correctness/Utility 학습자료로바꾸지않는다. '
        '기존3B자료에는적격Complete Scorable Group이없고독립Reference/현재Binding의학습근거가부족하다. '
        '진단/진료과2/10기준달성과Routing Artifact 적격성은별도다. 이번에는기준도미달했다. '
        'Jaccard/smoothing/min_samples/abstention공식·기존Gate를보존하며fitting/Agent호출0이다.\n\n'
        +json.dumps(routing,ensure_ascii=False,indent=2)+footer)
    md('VOI_B4_READINESS.md','# B4 적격성 — BLOCKED\n\n'
        'B3 차단에더해확대된Native 검사메뉴와기존Frozen M5의8개Action/likelihood/관측표현이일치하지않는다. '
        '새확률·Utility를만들거나원래H2/Bayes/Bellman/STOP을변경하지않았다. '
        'C4-35의합성Source정책평가와C4-39의Runtime수치일치검증은별도증거이며현재통합B4진단정확도가아니다.\n\n'
        +json.dumps(voi,ensure_ascii=False,indent=2)+footer)
    md('FINAL_B0_B4_EVALUATION_MATRIX.md','# 실제 최종 조건표\n\n'
        '|조건|실제상태|해석|\n|---|---|---|\n'
        '|B0|완료1/등록10,미실행9|진단0/1,진료과1/1;등록수율0/10·1/10|\n'
        '|B1|사전차단10/10,Generation0|절단없는전체이력의기존Context guard초과|\n'
        '|B2|계약실패1/등록10,미실행9|최초응답에서최대4개인용상한을넘겨전체중단|\n'
        '|B3|BLOCKED/NOT_RUN|적격Routing Utility/Reference/전문Agent 연결계약부족|\n'
        '|B4|BLOCKED/NOT_RUN|B3차단및확대Action/likelihood/encoding불일치|\n'
        '\n공식matched B0~B4나현재3B 전체성능평가완료가아니다. 역사적38/52/44/48/90%는현재재현값으로사용하지않는다.'+footer)
    md('FINAL_THESIS_RESULTS.md','# 논문에 반영할 현재 증거\n\n'+table+ratio+
        '\n구현성과는기존Memory/Routing/VoI수식을보존한Native ER 연결과Source공개/출력검증이다. '
        '이번실험은Memory효과나진단개선을입증하지못했다. B0 진료과Source한건일치는탐색적사례일뿐이다. '
        '기존정적D4 4/20·이전실패/완료Run·Final M5/C4-35/C4-39를서로다른분모의실험으로유지한다. '
        '개발노출패널,초기서술형증상부족,Recorded-Time Proxy/합성5분,ordinal12다음날Reference,독립임상검토없음을명시한다. '
        '원래논문파일과FINAL_THESIS완료자료는변경하지않았다.'+footer)
    md('FINAL_PROFESSOR_BRIEFING.md','# 교수님 실제 실행 보고\n\n'+table+
        f'\n사용자승인된새Run을1회실행해{s["generation"]}회Generation을수행했다. '
        'B0 1건은완료·Export됐고Source진료과가일치했으나진단명은일치하지않았다. '
        'B1 10건은전체이력을절단하지않고사전차단으로남겼다. '
        'B2는새객체형Memory인용에서5개를선택해기존상한4개를위반했고전체실행을중단했다. '
        '기존실패를수정하거나응답을삭제·보정하지않았다. '
        '등록10명중2명일치라는고정기준은아직미달이며B3/B4도별도자료계약으로차단됐다. '
        '추가생성·모델학습·Prompt최적화·Source 변경없이현재한계와실패를보존했다.'+footer)


if __name__=='__main__':analyze()
