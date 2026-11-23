"""Read-only fresh retry analysis with B1 selection and subset citation audits."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.rerun_b012_memory_stability import PUB as PREP,PR,DOC as PREPDOC,RUN
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.b012_pilot import aggregate
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.current_ablation_llm_backend import _strict_object
from app.locdi.memory_subset_wire import FIELD,allowed_codes
from app.locdi.memory_paired_recovery import verify_delivered_memory,citation_audit
from jsonschema import Draft202012Validator

PUB=PREP/'actual_results_v1';DOC=PREPDOC/'actual_results_v1'
DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_ACTUAL_RESULTS_REQUIRED')
    p=read(PREP/'B012_EXECUTION_PROTOCOL.json');run=PR/RUN;collection=read(run/'collection.json')
    trace=[json.loads(x) for x in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert trace==collection['rows']
    panel=read(PR/'panel.json');agg=aggregate(panel,trace);assert agg==read(run/'aggregate.json')
    cases=load_cases(DATA);index=read(PR/'case_index.json');plans=read(PR/'history_selection_plans.json')
    source={digest(x['source_event']):x['source_event'] for x in read(PR/'source_rows.json')}
    scored={x['episode_ordinal']:x for x in agg['episodes']}
    counters=Counter();errors=[];violations=[];rows=[];reqrows=[];private=[]
    for task in trace:
        score=scored[task['episode_ordinal']];case=cases[index[task['case_ref']]-1]
        original={r['memory_id']:r for v in case['agent_visible']['past_visits'] for r in v['records']}
        plan=plans[task['patient_ordinal']-1]
        row={k:task[k] for k in ('episode_ordinal','patient_ordinal','condition','status','request_count','error_type','backend_episode_ordinal')}
        row.update(original20_ordinal=p['original20_ordinals'][task['patient_ordinal']-1],
            condition_policy=p['condition_names'][task['condition']],certainty=score['certainty'],
            Source_exact=score['source_exact'],Normalized=score['normalized'],Source_department=score['source_department'],
            Core_status=None,export_status='NOT_ATTEMPTED',prompt_tokens_reported=0,output_tokens_reported=0,
            requests_with_token_usage=0,HTTP_wall_seconds=0,identity_checks=0,
            memory_records_min=None,memory_records_max=None,memory_bytes_min=None,memory_bytes_max=None,
            memory_record_occurrences=0,memory_citation_decisions=0,memory_cited_id_occurrences=0,
            B1_candidate_count=plan['candidate_count'] if task['condition']=='B1' else None,
            B1_selected_count=plan['selected_count'] if task['condition']=='B1' else None,
            B1_excluded_count=plan['excluded_count'] if task['condition']=='B1' else None,
            B1_time_range_private_ref='history_selection_plans.json' if task['condition']=='B1' else None,
            initial_retrieval_minute='NOT_RECORDED',recorded_retrievals=None,
            additional_test_proposals=0,additional_test_orders=0,additional_result_receipts=0,additional_evidence_citation_decisions=0)
        folder=run/'requests'/('Episode%04d'%task['backend_episode_ordinal']) if task['backend_episode_ordinal'] else None
        raw=read(folder/'raw_runner_result.json') if folder and (folder/'raw_runner_result.json').exists() else None
        names={};received={};starts={};orders={};memcounts=[];membytes=[];sets=[];memory_private=[]
        if raw:
            row['Core_status']=raw['status'];row['export_status']='EXPORTED' if (folder/'episode_and_raw.json').exists() else 'RAW_ONLY'
            row['identity_checks']=raw.get('llm_execution',{}).get('identity_check_count',0)
            row['recorded_retrievals']=len(raw.get('telemetry',{}).get('retrievals',[]))
            visits=(raw.get('execution') or {}).get('visits',[])
            names={x['test_name']:source[x['source_fingerprint']] for x in raw.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])}
            if visits:
                events=visits[0]['final_state']['event_log'];row['event_counts']=dict(Counter(e['event_type'] for e in events))
                row['simulation_minutes_last_event']=max((e['minute'] for e in events),default=0)
                if visits[0].get('failure_detail'):errors.append({'episode':task['episode_ordinal'],'stage':'Core','detail':visits[0]['failure_detail']})
                for e in events:
                    name=e.get('payload',{}).get('requested_test')
                    if name not in names:continue
                    if e['event_type']=='test_ordered':orders.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='test_started':starts.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='result_received':received.setdefault(name,[]).append(e['minute'])
                for name,minutes in received.items():
                    if not orders.get(name) or not starts.get(name) or min(starts[name])<min(orders[name]) or min(minutes)<min(starts[name])+5:
                        violations.append({'episode':task['episode_ordinal'],'code':'SOURCE_ORDER_START_RELEASE'})
            decisions=read(folder/'validated_ER_decisions.json');row['selected_actions']=dict(Counter(d['action'] for d in decisions))
            row['executed_actions']=(task.get('result') or {}).get('action_counts')
            terminal=next((d for d in reversed(decisions) if d['action']=='diagnose'),None)
            if terminal:private.append({'episode_ordinal':task['episode_ordinal'],'condition':task['condition'],'terminal':terminal,
                'Source_Reference':case['evaluation'],'actual_status':task['status']})
            if task['status']=='COMPLETED':
                assert terminal is not None and row['export_status']=='EXPORTED'
                pred=task['result']['prediction'];assert pred['diagnosis']==(terminal['target'] if terminal['certainty']=='supported' else None)
                assert pred['department']==terminal['department'];counters['terminal_backend_export_consistent']+=1
        for bp in sorted(folder.glob('Request*_before.json')) if folder else []:
            ap=bp.with_name(bp.name.replace('_before','_after'))
            b=read(bp);req=b['request'];payload=json.loads(req['messages'][1]['content']);observation=payload['observation']
            if not ap.exists():errors.append({'episode':task['episode_ordinal'],'request':bp.name,'code':'AFTER_MISSING'});continue
            a=read(ap);assert b['request_fingerprint']==digest(req)==a['request_fingerprint']
            assert b['response_wire_contract']==p['response_contract'];counters['checkpoint_fingerprints']+=1
            assert not {'evaluation','gold','provenance','task_annotations','source_event_store'}&set(payload)
            memory=observation.get('longitudinal_memory',{});records=memory.get('records',[])
            memcounts.append(len(records));membytes.append(len(json.dumps(records,ensure_ascii=False).encode('utf-8')))
            sets.append([r['memory_id'] for r in records]);row['memory_record_occurrences']+=len(records)
            if task['condition']=='B0':assert not records
            if task['condition']=='B1':
                assert sets[-1]==plan['selected_ids']
                assert memory['eligible_count']==plan['candidate_count'] and memory['budget_excluded_count']==plan['excluded_count']
                assert all(all(r.get(k)==v for k,v in original[r['memory_id']].items()) for r in records)
                counters['B1_stable_complete_Source_payloads']+=1
            if task['condition']=='B2' and records:
                assert verify_delivered_memory(case,memory)['status']=='PASS';counters['B2_Source_cutoff_Top5_payloads']+=1
            env=a.get('response_envelope',{});ident=a.get('identity_after',{})
            row['HTTP_wall_seconds']+=a.get('wall_seconds',0);counters['HTTP200']+=a.get('HTTP_status')==200
            if ident:
                assert ident['context_length']==32768 and 'sha256:'+ident['digest'].removeprefix('sha256:')==p['model_identifier']
                counters['actual_Context_Digest_match']+=1
            request={'episode_ordinal':task['episode_ordinal'],'condition':task['condition'],'request':bp.name,'HTTP_status':a.get('HTTP_status'),
                'client_status':a.get('status'),'error_type':a.get('error_type'),'wall_seconds':a.get('wall_seconds'),
                'memory_records':len(records),'memory_bytes':membytes[-1],'wire_code':None,'memory_cited_count':None,
                'schema_status':'NOT_CHECKED_NO_RESPONSE','prompt_tokens':env.get('prompt_eval_count'),'output_tokens':env.get('eval_count')}
            if env:
                row['requests_with_token_usage']+=1;row['prompt_tokens_reported']+=env.get('prompt_eval_count',0);row['output_tokens_reported']+=env.get('eval_count',0)
                try:
                    value=_strict_object(env['message']['content']);Draft202012Validator(req['format']).validate(value)
                    assert value[FIELD] in allowed_codes(records)
                    internal=a['internal_response'];assert digest(internal)==a['internal_response_fingerprint']
                    cited=[records[i]['memory_id'] for i,bit in enumerate(value[FIELD]) if bit=='1']
                    assert internal.get('memory_ids',[])==cited and len(cited)<=4 and len(cited)==len(set(cited))
                    ca=citation_audit(req,env);assert ca['error_code'] is None
                    counters['wire_enum_Internal_Evidence_valid']+=1
                    if records:counters['nonempty_Memory_wire_valid']+=1
                    request.update(schema_status='PASS',wire_code=value[FIELD],memory_cited_count=len(cited))
                    row['memory_citation_decisions']+=bool(cited);row['memory_cited_id_occurrences']+=len(cited)
                    evidence=payload['evidence_records']
                    row['additional_evidence_citation_decisions']+=any(any(evidence[i]['text'].startswith(n+':') for n in names)
                        for i in value.get('evidence_indices',[]) if type(i) is int and 0<=i<len(evidence))
                    if value.get('action')=='request_test' and value.get('target') in names:row['additional_test_proposals']+=1
                    memory_private.append({'request':bp.name,'provided_ids':sets[-1],'cited_ids':cited,'wire':value[FIELD]})
                except Exception as exc:
                    request['schema_status']='FAIL';errors.append({'episode':task['episode_ordinal'],'request':bp.name,'stage':'wire_or_internal_schema',
                        'error_type':type(exc).__name__,'validator':getattr(exc,'validator',None),'path':list(getattr(exc,'path',[]))})
            if a.get('status')!='VALID':errors.append({'episode':task['episode_ordinal'],'request':bp.name,'stage':'Native_transport_or_schema','error_type':a.get('error_type'),'HTTP_status':a.get('HTTP_status'),'wall_seconds':a.get('wall_seconds')})
            for finding in observation.get('objective_findings',[]):
                for name,event in names.items():
                    if finding.startswith(name+': '):
                        if not received.get(name) or observation['elapsed_minutes']<min(received[name]):violations.append({'episode':task['episode_ordinal'],'code':'PRE_RELEASE_EXPOSURE'})
                        v=json.loads(finding[len(name)+2:]);assert v['value']==event['evidence']['value'] and v['units']==event['evidence'].get('units')
                        counters['Source_value_unit_occurrences_match']+=1
            reqrows.append(request)
        if task['condition']=='B1' and sets:assert all(x==sets[0] for x in sets)
        if sets:
            row.update(memory_records_min=min(memcounts),memory_records_max=max(memcounts),memory_bytes_min=min(membytes),memory_bytes_max=max(membytes))
            row['memory_unique_records_provided']=len({x for selected in sets for x in selected})
            row['memory_unique_records_cited']=len({x for r in memory_private for x in r['cited_ids']})
            private.append({'episode_ordinal':task['episode_ordinal'],'Memory_trace':memory_private,'B1_time_range':plan['selected_time_range'] if task['condition']=='B1' else None})
        row.update(additional_test_orders=sum(map(len,orders.values())),additional_result_receipts=sum(map(len,received.values())))
        rows.append(row)
    assert len(rows)==30 and collection['generation_count']==sum(r['request_count'] for r in rows)
    summary={'run_id':RUN,'generation_attempts':collection['generation_count'],'registered_patients':10,'registered_Episodes':30,
        'states':dict(Counter(r['status'] for r in rows)),'conditions':agg['conditions'],'paired':agg['paired'],
        'freeze_criterion':agg['freeze_threshold'],'global_stop':collection['global_stop_reason'],
        'wall_seconds':read(PREP/'actual_execution_summary.json')['wall_seconds'],'checks':dict(counters),'errors':errors,'Source_time_violations':violations,
        'clinical_accuracy':'NOT_EVALUATED','Verified_Concept_Match':'NOT_EVALUATED','post_run_generation':0,
        'repeat_reason':'USER_REPORTED_DISCONNECT_NOT_INDEPENDENTLY_VERIFIED','prior_Run_results_merged':False}
    for key in ('prompt_tokens_reported','output_tokens_reported','requests_with_token_usage','identity_checks','HTTP_wall_seconds',
        'additional_test_orders','additional_result_receipts','additional_evidence_citation_decisions','memory_record_occurrences',
        'memory_citation_decisions','memory_cited_id_occurrences'):
        summary[key]=sum(r[key] for r in rows)
    PUB.mkdir(parents=True);DOC.mkdir(parents=True)
    write(PR/'actual_private_diagnosis_memory_analysis.json',private)
    write(PUB/'B012_ACTUAL_METRICS.json',{'summary':summary,'rows':rows,'aggregate':agg})
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    csv_file(PUB/'B012_EPISODE_RESULTS.csv',rows,fields)
    if reqrows:csv_file(PUB/'NATIVE_REQUEST_AUDIT.csv',reqrows,list(reqrows[0]))
    for key,pair in agg['paired'].items():csv_file(PUB/('PAIRED_'+key+'.csv'),pair['rows'],list(pair['rows'][0]))
    reports(p,summary,rows)
    protected=check(read(PR/'preservation_before.json'));assert protected['status']=='PASS'
    pins=check({k:v for g in ('input_hashes','binding_hashes','code_hashes') for k,v in p[g].items()});assert pins['status']=='PASS'
    identifiers=[x for c in cases for x in (c['patient_id'],c['agent_visible']['index_encounter_id'])]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file():assert not any(x in f.read_text(encoding='utf-8-sig') for x in identifiers)
    write(PUB/'FINAL_ACTUAL_MANIFEST.json',{'status':'SAVED_ACTUAL_RUN_RECORDED','summary':summary,'protected':protected,'pins':pins,
        'trace_reaggregation':'EXACT_MATCH','Run_hashes':{str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file()},
        'reports':{f.name:sha(f) for f in DOC.glob('*.md')},'analyzer_SHA256':sha(Path(__file__)),'at':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paired','errors')},ensure_ascii=True))


def reports(p,s,rows):
    table='|조건|등록|완료|실패|미완료|미실행|진단 Source 일치/등록|Normalized 일치/등록|진료과 Source 일치/등록|요청 수|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for c,t in s['conditions'].items():
        st=t['states'];rs=[r for r in rows if r['condition']==c]
        table+=f'|{p["condition_names"][c]}|10|{st.get("COMPLETED",0)}|{sum(v for k,v in st.items() if k in ("CONTRACT_FAILED","TIMEOUT","CONTEXT_BLOCKED"))}|{st.get("INCOMPLETE",0)}|{st.get("NOT_ATTEMPTED",0)}|{t["Source_exact_registered_success_count"]}/10|{sum(r["Normalized"] is True for r in rs)}/10|{t["Source_department_registered_success_count"]}/10|{t["generation"]}|\n'
    conditional='|조건|Supported 진단 Source 일치|진료과 Source 일치/완료|Unresolved|\n|---|---:|---:|---:|\n'
    for c,t in s['conditions'].items():
        done=t['states'].get('COMPLETED',0)
        a=f'{t["Source_exact_registered_success_count"]}/{t["supported_scorable"]}' if t['supported_scorable'] else 'NOT_EVALUATED'
        b=f'{t["Source_department_registered_success_count"]}/{done}' if done else 'NOT_EVALUATED'
        conditional+=f'|{p["condition_names"][c]}|{a}|{b}|{t["unresolved"]}|\n'
    foot=f'''

## 공통 근거

Run `{RUN}`, Protocol `{p['fingerprint']}`. Model `{p['model_identifier']}`.
Source fingerprint `{p['source_rows_fingerprint']}`, Catalog SHA `{p['catalog_SHA256']}`.
Code Pin 집합 `{digest(p['code_hashes'])}`. 같은10명, 기존 임상Prompt/Scorer/Source/가중치/수식/Validator 유지.
Context32768,temperature0,seed42,출력1024,요청timeout120초,Retry/Repair/Warmup0.
환자별B0→B1_CONTEXT_BUDGETED_HISTORY→B2,등록30/각12/최대360회,wall45분/Python4GiB/output1GiB/no-resume.
각 조건의 등록10명 일치 수율과 완료·supported 분모를 구분한다. 실패·미완료·미실행을 임상 오답으로 바꾸지 않는다.
Source 일치는 합성 Reference 일치이며 임상 정확도/Verified Concept Match는 NOT_EVALUATED다.
기존254개 관련 회귀/30건Source Mock 검증은 변경 없는 Runtime 코드의 준비 근거로 재사용하며 전체 저장소 회귀나독립임상평가로확대하지않는다.
이전Run001은응답6/요청7/완료0의실패기록으로보존했고이번결과에병합하지않았다.
그7회를포함한두시도의누적요청수는{s['generation_attempts']+7}회이며,이번Run 요청수는{s['generation_attempts']}회다.
연결끊김은사용자의재실행사유진술로기록했으며Timeout의인과관계를별도로확정하지않았다.
환자별원문·진단·Reference·Memory ID/시점범위는Private actual_private_diagnosis_memory_analysis.json과원래요청Checkpoint에보존했다.
'''
    complete=sum(r['status']=='COMPLETED' for r in rows)
    md('B012_FINAL_RESULTS.md','# 새 Run 실제 결과\n\n'+table+
        '\n위 표의 등록 일치 수율은 실패를 오답 라벨로 만든 임상 정확도가 아니다. 조건부 분모는 다음과 같다.\n\n'+conditional+
        f'\n완료{complete}/30, 실제요청{s["generation_attempts"]}회, wall{s["wall_seconds"]:.3f}초, 중단이유 `{s["global_stop"]}`. '
        '현재Run의실제상태를사용하고과거Run이나Mock완료로보충하지않는다.\n\n'
        +json.dumps({'freeze_criterion':s['freeze_criterion'],'checks':s['checks'],'errors':s['errors'],
            'Source_time_violations':s['Source_time_violations']},ensure_ascii=False,indent=2)+foot)
    memtable='|선정환자|B1 후보|B1 선택|B1 제외|실제제공 B1 records 범위|B1 records bytes 범위|B2 records 범위|B2 records bytes 범위|\n|---|---:|---:|---:|---|---|---|---|\n'
    for i in range(10):
        a,b=rows[3*i+1],rows[3*i+2]
        memtable+=f'|{i+1}|{a["B1_candidate_count"]}|{a["B1_selected_count"]}|{a["B1_excluded_count"]}|{a["memory_records_min"]}~{a["memory_records_max"]}|{a["memory_bytes_min"]}~{a["memory_bytes_max"]}|{b["memory_records_min"]}~{b["memory_records_max"]}|{b["memory_bytes_min"]}~{b["memory_bytes_max"]}|\n'
    md('B1_HISTORY_SELECTION_VALIDATION.md','# B1 실제 선택/제공 검증\n\n'+memtable+
        '\nNone은미실행/미관측이며0개제공으로바꾸지않는다. B1은시간·Context 예산기준으로실행전고정한완전한기록을선택하고방문내ID집합을유지했다. '
        '후보/선택/제외는사전계획,실제제공범위는실제요청에서읽었다. 모델에제공된각기록은원래Source필드와같고원천값·단위·시각·원문을삭제하지않았다. '
        'B1은원래Full History가아니며B2는기존관련성Top5다. 정보량차이가있어순수검색알고리즘인과효과로해석하지않는다. '
        '행별CSV의memory_unique_records_provided/cited는방문내고유기록수이고occurrences는반복제공/인용횟수다. '
        '시점범위/선택ID/제외사유는Private history_selection_plans.json 및 실제Memory trace에보존한다.'+foot)
    md('LIVE_MEMORY_SUBSET_CONTRACT_VALIDATION.md','# 실제 부분집합 enum 검증\n\n'
        '각실제응답의원문JSON Schema enum,허용코드,최대4개,미제공/superseded차단,Raw→Internal ID목록/fingerprint,기존Evidence 인용계약을교차확인했다. '
        f'기록된검증수: {json.dumps(s["checks"],ensure_ascii=False)}. '
        '비어있지않은Memory의검증수는nonempty_Memory_wire_valid이며0/미존재라면해당범위Live검증을주장하지않는다. '
        '없는응답은Schema실패가아닌NOT_CHECKED로남긴다. 잘못된코드/인용을삭제하거나상한을늘리는후처리를하지않았다. '
        '이번Run에서관측된통과가모든입력/모델실행의영구적인안정성을보장하지는않는다.'+foot)
    pairtable='|비교|완료쌍|Source 진단 scorable쌍|Source 진단 차이(후자−전자)|\n|---|---:|---:|---|\n'
    for name,a in s['paired'].items():pairtable+=f'|{name}|{a["complete_pairs"]}/10|{a["scorable_pairs"]}/10|{a["difference"] if a["difference"] is not None else "NOT_EVALUATED"}|\n'
    md('B012_MEMORY_COMPARISON.md','# 실제 paired Memory 비교\n\n'+table+'\n'+conditional+'\n'+pairtable+
        '\n차이는비율단위이며1.0=100pp다. 완료/지원진단쌍조건부효과와등록전체의결과를함께보여준다. '
        '진료과차이와Normalized를추가로읽을때도같은환자별완료분모를사용해야한다. '
        '진료과Source일치는ER가반환한진료과라벨과Source정책의일치이며Adaptive Routing 정확도가아니다. '
        'B1의1~3개기록과B2의최대5개는정보량이다르며Workflow/행동길이도다를수있다. '
        'Memory만의Token은서버에서분리계측되지않았으므로전체prompt token과Source record bytes를분리보고한다.'+foot)
    md('SOURCE_TIME_AND_PRIVACY_AUDIT.md','# Source·시간·공개 경계\n\n'
        f'관측된Source 시간위반: {json.dumps(s["Source_time_violations"],ensure_ascii=False)}. '
        '추가검사order/start/result/그후모델관측의순서를읽고모델에전달된value/units를원천SourceEvent와대조했다. '
        'Memory는같은환자/이전방문/index cutoff/Source값을확인하고B0에기록이없음을검사했다. '
        'evaluation/gold/task_annotations/provenance/전체source_event_store는실제모델payload의최상위에없다. '
        'Source의임상공개시각·합성5분TAT·Recorded-Time Proxy가독립임상검증된것은아니다. '
        'ordinal12의다음날Source Reference와개발패널반복노출한계를유지하며공개보고서에원본환자/방문ID·원문임상값을출력하지않는다.'+foot)
    md('B012_RESOURCE_AND_EXECUTION_AUDIT.md','# 호출·Token·시간·실패 기록\n\n'
        +json.dumps({k:v for k,v in s.items() if k not in ('conditions','paired')},ensure_ascii=False,indent=2)+
        '\n\nToken은응답이저장된요청에서서버가보고한합계다. 응답없는요청의미관측Token을0이라고추정하지않는다. '
        '설정timeout120초와요청전체hard wall deadline은동일하지않으며자원검사는호출전후에이뤄진다. '
        '초기Memory retrieval minute은기록되지않아재구성하지않는다. 실제요청과identity검사/정책결정/실행Action을서로다른계수로보고한다.'+foot)
    reached=s['freeze_criterion']['any_condition_reached']
    message=('등록10명중Source 진단또는진료과일치2건이상인조건이있어사용자의개발최적화중단기준을달성했다. '
        '이설정을고정하고추가Prompt/Memory최적화를자동으로진행하지않는다. ' if reached else
        '현재등록10명중Source 진단또는진료과일치2건이상이라는개발중단기준은달성하지못했다. '
        '불리한결과를보존하고자동새Run/정답매핑/Prompt반복탐색을하지않는다. ')
    md('FINAL_B0_B4_READINESS.md','# 실제 결과에 따른 고정·후속 판정\n\n'+table+'\n'+message+
        '전체완료·실패·미완료·미실행상태와기준달성은서로별개다. '
        'B1은Context-Budgeted History이며원래B1 Full History 결과로바꾸지않는다. '
        'B3는적격Routing Utility/독립Correctness/현재Agent Binding 학습자료가부족해BLOCKED다. '
        'B4도B3차단및확대Native Action/likelihood/관측encoding불일치로BLOCKED이며이번범위에서실행하지않았다. '
        '기존VoI합성평가를이번B4진단정확도로대체하지않는다.'+foot)
    md('FINAL_PROFESSOR_BRIEFING.md','# 교수님 실제 재실행 보고\n\n'+table+'\n'+message+
        f'이번Run은요청{s["generation_attempts"]}회, 완료{complete}/30, 전체wall{s["wall_seconds"]:.3f}초다. '
        '기존연결끊김/Timeout Run은보존하고새Run으로독립저장했다. '
        'Runtime가중치·임상Prompt·Source·Scorer·Memory/Routing/VoI수식은그대로다. '
        'B1은원자기록과Context 예산으로선택한부분이력, B2는기존Top5라는차이를명시한다. '
        'Source 일치와실제임상정확도는다르며기술적완료와진단개선을별도로보고한다.'+foot)


if __name__=='__main__':main()
