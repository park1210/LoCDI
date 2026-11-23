"""Read-only actual stability pilot closeout. Never resumes or calls a model."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_b012_memory_stability import PUB as PREP,PR,DOC as PREPDOC,RUN,DATA
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.b012_pilot import aggregate
from app.locdi.llm_setup import digest
from app.locdi.current_ablation_llm_backend import _strict_object
from app.locdi.memory_subset_wire import FIELD,allowed_codes
from app.locdi.dataset_loader import load_cases
from app.locdi.memory_paired_recovery import citation_audit
from jsonschema import Draft202012Validator

PUB=PREP/'actual_results_v1';DOC=PREPDOC/'actual_results_v1'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_ANALYSIS_REQUIRED')
    p=read(PREP/'B012_EXECUTION_PROTOCOL.json');run=PR/RUN
    collection=read(run/'collection.json');panel=read(PR/'panel.json')
    trace=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    assert trace==collection['rows']
    agg=aggregate(panel,trace);assert agg==read(run/'aggregate.json')
    source={digest(r['source_event']):r['source_event'] for r in read(PR/'source_rows.json')}
    counts=Counter();errors=[];boundary=[];rows=[];private=[];request_rows=[]
    for task in trace:
        row={k:task[k] for k in ('episode_ordinal','patient_ordinal','condition','status','request_count','error_type','backend_episode_ordinal')}
        row.update(condition_policy=p['condition_names'][task['condition']],Core_status=None,Export='NOT_ATTEMPTED',
            valid_decisions=0,recorded_retrievals=None,Source_exact=None,Normalized=None,Source_department=None,
            reported_prompt_tokens=0,reported_output_tokens=0,token_usage_observed_requests=0,
            observed_HTTP_wall_seconds=0,additional_orders=0,additional_results_received=0,
            memory_records_provided_occurrences=0,valid_memory_citation_decisions=0,identity_checks=0)
        folder=run/'requests'/('Episode%04d'%task['backend_episode_ordinal']) if task['backend_episode_ordinal'] else None
        raw=read(folder/'raw_runner_result.json') if folder and (folder/'raw_runner_result.json').exists() else None
        names={};received={};started={};ordered={}
        if raw:
            row['Core_status']=raw['status'];row['Export']='EXPORTED' if (folder/'episode_and_raw.json').exists() else 'RAW_ONLY_NO_COMPLETED_EPISODE_EXPORT'
            row['recorded_retrievals']=len(raw.get('telemetry',{}).get('retrievals',[]))
            row['identity_checks']=raw.get('llm_execution',{}).get('identity_check_count',0)
            decisions=read(folder/'validated_ER_decisions.json');row['valid_decisions']=len(decisions)
            row['selected_action_counts']=dict(Counter(r['action'] for r in decisions))
            names={r['test_name']:source[r['source_fingerprint']] for r in raw.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])}
            visits=(raw.get('execution') or {}).get('visits',[])
            if visits:
                events=visits[0]['final_state']['event_log'];row['event_counts']=dict(Counter(e['event_type'] for e in events))
                row['simulation_last_recorded_minute']=max((e['minute'] for e in events),default=None)
                for e in events:
                    name=e.get('payload',{}).get('requested_test')
                    if name not in names:continue
                    if e['event_type']=='test_ordered':ordered.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='test_started':started.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='result_received':received.setdefault(name,[]).append(e['minute'])
                for name,minutes in received.items():
                    if not ordered.get(name) or not started.get(name) or min(minutes)<min(started[name])+5:
                        boundary.append({'episode':task['episode_ordinal'],'code':'ORDER_START_RELEASE'})
                if visits[0].get('failure_detail'):errors.append({'episode':task['episode_ordinal'],'Core_failure':visits[0]['failure_detail']})
            private.append({'episode_ordinal':task['episode_ordinal'],'validated_decisions':decisions})
        for before_path in sorted(folder.glob('Request*_before.json')) if folder else []:
            after_path=before_path.with_name(before_path.name.replace('_before','_after'))
            b=read(before_path);req=b['request'];payload=json.loads(req['messages'][1]['content'])
            counts['BEFORE_checkpoints']+=1
            if not after_path.exists():errors.append({'episode':task['episode_ordinal'],'request':before_path.name,'error':'AFTER_MISSING'});continue
            a=read(after_path);counts['AFTER_checkpoints']+=1
            assert b['request_fingerprint']==digest(req)==a['request_fingerprint']
            assert b['response_wire_contract']==p['response_contract']
            counts['request_fingerprints_match']+=1
            env=a.get('response_envelope',{});ident=a.get('identity_after',{})
            recs=payload['observation'].get('longitudinal_memory',{}).get('records',[])
            row['memory_records_provided_occurrences']+=len(recs)
            if task['condition']=='B0':assert not recs
            assert not {'gold','evaluation','task_annotations','provenance','source_event_store'}&set(payload)
            entry={'episode_ordinal':task['episode_ordinal'],'request':before_path.name,'client_status':a.get('status'),
                'HTTP_status':a.get('HTTP_status'),'response_body_recorded':bool(env),
                'wall_seconds':a.get('wall_seconds'),'error_type':a.get('error_type'),
                'schema_status':'NOT_CHECKED_NO_RESPONSE','wire_code':None,'provided_memory_records':len(recs),
                'prompt_tokens':env.get('prompt_eval_count'),'output_tokens':env.get('eval_count')}
            row['observed_HTTP_wall_seconds']+=a.get('wall_seconds',0)
            counts['HTTP200']+=a.get('HTTP_status')==200
            if ident:
                assert ident['context_length']==32768 and 'sha256:'+ident['digest'].removeprefix('sha256:')==p['model_identifier']
                counts['actual_loaded_Context_Digest_match']+=1
            if env:
                row['token_usage_observed_requests']+=1
                row['reported_prompt_tokens']+=env.get('prompt_eval_count',0);row['reported_output_tokens']+=env.get('eval_count',0)
                try:
                    value=_strict_object(env['message']['content']);Draft202012Validator(req['format']).validate(value)
                    assert value[FIELD] in allowed_codes(recs)
                    ca=citation_audit(req,env);assert ca['error_code'] is None
                    internal=a['internal_response'];assert digest(internal)==a['internal_response_fingerprint']
                    expected=[recs[i]['memory_id'] for i,bit in enumerate(value[FIELD]) if bit=='1']
                    assert internal.get('memory_ids',[])==expected and len(expected)<=4
                    entry.update(schema_status='PASS',wire_code=value[FIELD]);counts['wire_enum_schema_and_evidence_valid']+=1
                    counts['wire_internal_mapping_match']+=1
                    row['valid_memory_citation_decisions']+=bool(expected)
                    private.append({'episode_ordinal':task['episode_ordinal'],'request':before_path.name,'internal_response':internal})
                except Exception as exc:
                    entry['schema_status']='FAIL';errors.append({'episode':task['episode_ordinal'],'request':before_path.name,'validation_error_type':type(exc).__name__})
            else:
                errors.append({'episode':task['episode_ordinal'],'request':before_path.name,'client_error_type':a.get('error_type'),
                    'JSON_Schema_Enum':'NOT_CHECKED_NO_RESPONSE','HTTP_status':'NOT_RECORDED','elapsed_wall_seconds':a.get('wall_seconds')})
            for finding in payload['observation'].get('objective_findings',[]):
                for name,event in names.items():
                    if finding.startswith(name+': '):
                        if not received.get(name) or payload['observation']['elapsed_minutes']<min(received[name]):boundary.append({'episode':task['episode_ordinal'],'code':'PRE_RELEASE_MODEL_INPUT'})
                        val=json.loads(finding[len(name)+2:]);assert val['value']==event['evidence']['value'] and val['units']==event['evidence'].get('units')
                        counts['Source_value_unit_occurrences_match']+=1
            request_rows.append(entry)
        row.update(additional_orders=sum(map(len,ordered.values())),additional_results_received=sum(map(len,received.values())))
        rows.append(row)
    assert len(rows)==30 and collection['generation_count']==sum(r['request_count'] for r in rows)
    import httpx
    timeout={'configured_seconds':120,'local_httpx_version':httpx.__version__,'httpx_timeout_configuration':httpx.Timeout(120).as_dict(),
        'observed_failed_request_wall_seconds':[r['wall_seconds'] for r in request_rows if r['error_type']],
        'hard_per_request_wall_deadline_implemented':False,
        'interpretation':'Configured timeout and observed wall are distinct. Saved data cannot attribute329s among scheduling, transport, server or suspension. Do not invent a timing breakdown.'}
    summary={'run_id':RUN,'generation':collection['generation_count'],'registered_patients':10,'registered_episodes':30,
        'states':dict(Counter(r['status'] for r in rows)),'conditions':agg['conditions'],'paired':agg['paired'],
        'freeze_threshold':agg['freeze_threshold'],'global_stop':collection['global_stop_reason'],
        'wall_seconds':read(PREP/'actual_execution_summary.json')['wall_seconds'],'checks':dict(counts),
        'execution_errors':errors,'Source_memory_boundary_violations':boundary,'timeout':timeout,
        'B1_live_validation':'NOT_RUN','B2_live_validation':'NOT_RUN',
        'real_enum_validation_scope':'successful B0 no-memory code only; nonempty Memory subset enum NOT_CHECKED',
        'Source_accuracy':'NOT_EVALUATED_NO_COMPLETED_EPISODE','clinical_accuracy':'NOT_EVALUATED','Verified_Concept_Match':'NOT_EVALUATED',
        'generation_after_stop':0,'analysis_generation':0}
    for key in ('reported_prompt_tokens','reported_output_tokens','token_usage_observed_requests','observed_HTTP_wall_seconds',
        'additional_orders','additional_results_received','memory_records_provided_occurrences','valid_memory_citation_decisions','identity_checks'):
        summary[key]=sum(r[key] for r in rows)
    PUB.mkdir(parents=True);DOC.mkdir(parents=True)
    write(PR/'actual_private_action_analysis.json',private)
    write(PUB/'ACTUAL_STABILITY_METRICS.json',{'summary':summary,'rows':rows,'aggregate':agg})
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    csv_file(PUB/'B012_EPISODE_RESULTS.csv',rows,fields)
    csv_file(PUB/'NATIVE_REQUEST_AUDIT.csv',request_rows,list(request_rows[0]))
    for key,pair in agg['paired'].items():csv_file(PUB/('PAIRED_'+key+'.csv'),pair['rows'],list(pair['rows'][0]))
    reports(p,summary)
    protected=check(read(PR/'preservation_before.json'));assert protected['status']=='PASS'
    pins=check({k:v for group in ('input_hashes','binding_hashes','code_hashes') for k,v in p[group].items()});assert pins['status']=='PASS'
    ids=[c['patient_id'] for c in load_cases(DATA)]+[c['agent_visible']['index_encounter_id'] for c in load_cases(DATA)]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file():assert not any(x in f.read_text(encoding='utf-8-sig') for x in ids)
    write(PUB/'FINAL_ACTUAL_MANIFEST.json',{'status':'LIVE_TIMEOUT_PARTIAL_NO_AUTOMATIC_RESUME','summary':summary,
        'protocol':p['fingerprint'],'protected':protected,'pins':pins,'trace_reaggregation':'EXACT_MATCH',
        'Run_hashes':{str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file()},
        'reports':{f.name:sha(f) for f in DOC.glob('*.md')},'analyzer_SHA256':sha(Path(__file__)),
        'recorded_at':datetime.now(timezone.utc).isoformat()})
    for base in (PREP,PREPDOC):
        with (base/'ACTUAL_RESULT_POINTER.md').open('x',encoding='utf-8') as f:
            f.write('# Actual result\n\nRun synthea_b012_subset_stable_001 ended after7 requests with ReadTimeout. Failed1/not_attempted29; completed0. Use actual_results_v1/. Do not resume or infer live B1/B2 readiness from Mock PASS.\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paired','execution_errors')},ensure_ascii=True))


def reports(p,s):
    common=f'\n\nRun `{RUN}`, Protocol `{p["fingerprint"]}`, Model `{p["model_identifier"]}`, Source `{p["source_rows_fingerprint"]}`. '
    common+=f'Code Pin 집합 `{digest(p["code_hashes"])}`. 임상 Prompt·Scorer·Core수식·Evidence Validator와기존Run은변경하지않았다. '
    common+='실제새요청7회후종료, 추가요청·재시도·자동Resume0.\n'
    table='|조건|등록|완료|실패|미실행|실제요청|진단·진료과평가|\n|---|---:|---:|---:|---:|---:|---|\n|B0|10|0|1|9|7|NOT_EVALUATED|\n|B1_CONTEXT_BUDGETED_HISTORY|10|0|0|10|0|NOT_RUN|\n|B2|10|0|0|10|0|NOT_RUN|\n'
    md('B012_FINAL_RESULTS.md','# 실제 실행 결과 — 첫 ReadTimeout에서 중단\n\n'+table+
        '\n첫B0의7번째요청에서ReadTimeout이발생했다. 앞선6개HTTP200응답만본문·Schema·enum·actual Context/Digest를확인할수있다. '
        '마지막요청은완전한응답본문과HTTP status가저장되지않아JSON/Schema/enum을판정할수없다. '
        '미수신응답을Schema실패나잘못된진단으로취급하지않는다. 완료Episode/Export0, 완료쌍0이다. '
        'Source진단·진료과일치와Memory효과는NOT_EVALUATED이며2/10 고정기준은미달/미검증이다.\n\n'
        +json.dumps(s,ensure_ascii=False,indent=2)+common)
    md('B1_CONTEXT_BUDGETED_HISTORY_DESIGN.md','# B1 정책의 실제 검증 범위\n\n'
        '모든적격과거기록을후보로최신순완전한기록을선택하고큰기록/명시적원자블록은통째로제외후다음기록을검토한다. '
        '선택결과는시간순으로제공하고방문내ID집합을고정한다. 값·단위·시각·원문을자르거나요약하지않는다. '
        'System/현재정보/Action Menu/Schema/출력1024/margin512 및추가후속정보2048 bytes를예약하고Memory+enum공동예산은최대4096 bytes다. '
        '원래B1 Full History가아닌별도Context-Budgeted History다. 오프라인에서1~3개선택과30Core경로를검증했지만실제B1은요청0회라Live안정성은미검증이다.'+common)
    md('B1_HISTORY_SELECTION_VALIDATION.md','# B1 선택 및 정보량의 한계\n\n'
        '실행전고정선택은환자별[3,3,3,3,3,1,3,3,3,3]개다. 후보는260~12,058개이며제외수·이유·Source fingerprint와시점범위는Private계획에보존했다. '
        '254개관련회귀와30개Source Mock검증에서원천필드·환자/방문/cutoff·완전기록·안정적집합·후속guard를검사했다. '
        '실제B1 Memory제공량/Token은NOT_RUN이다. B2 Top5와정보량및선택기준이다르다는한계를유지하며이번실제효과차이를계산하지않는다.'+common)
    md('B2_MEMORY_SUBSET_WIRE_CONTRACT.md','# 부분집합 코드 계약의 실제 검증 범위\n\n'
        '현재Memory위치에대한0/1코드를단일enum으로정의한다. 최대4개·superseded제외·현재제공기록만허용하고5개선택코드는없다. '
        '유효코드만원래내부ID목록으로변환후원래Schema/EvidenceValidator를재검증하며잘못된응답을삭제/보정하지않는다. '
        '실제성공6응답은B0의Memory없는코드0만검증했다. 비어있지않은B1/B2부분집합enum의실제서버준수는NOT_CHECKED다. '
        '마지막요청은응답이없어새계약실패나통과로분류하지않는다.'+common)
    md('B2_MEMORY_CITATION_REGRESSION.md','# 인용 회귀와 실제 결과 구분\n\n'
        '기존실패응답5개인용은보존했다. 오프라인0~4개허용/5개거부/중복표현불가/미등록·superseded차단/anyOf 분기/Native Fake Raw→Internal을검증했다. '
        '실제성공응답6개는빈Memory인용만통과했다. 비어있지않은Memory인용의Live오류율을0%라고주장하지않는다. '
        '실제Timeout1건은인용오류와별도이며임상정오판정으로변환하지않는다.'+common)
    md('B012_FULL_PIPELINE_REGRESSION.md','# 오프라인과 실제 경로 구분\n\n'
        '기존관련254/254 PASS, Source Mock Core/Scheduler/Exporter30/30 PASS, 저장Core결과의Backend/Scorer후처리PASS를보존한다. '
        '전체저장소회귀나실제30건성공으로확대하지않는다. 실제Runner는첫B0부분실행과raw/error/checkpoint보존까지만확인됐다. '
        '완료Exporter0, 실제임상Scorer분모0, 저장Trace재집계는EXACT_MATCH다.\n\n'
        +json.dumps(s['checks'],ensure_ascii=False,indent=2)+common)
    md('B012_NEXT_EXECUTION_PROTOCOL.md','# 실행 후 판단 — 자동 다음 Run 없음\n\n'
        '승인된새Run의1회시도는실행됐다. 같은Run 재사용·Resume·추가요청은하지않는다. '
        '실행중지의직접근거는HTTP ReadTimeout이며부하/서버/중단·절전/네트워크중원인을저장자료만으로확정할수없다. '
        '실제응답없는처리시간구간을만들지않고서버설정·Timeout·모델을자동변경하지않는다. '
        '재실행을선택하려면Timeout운영조건과새사용하지않은Run/예산을별도로확정하고새승인을받아야한다. '
        '이번문서는새실행명령을실행하거나새승인을만든것이아니다.'+common)
    md('B012_EXECUTION_APPROVAL_REQUEST.md','# 승인 사용 및 잔여 범위\n\n'
        '사용자는다른Ollama작업없음과같은30Episode/최대360회계획을명시적으로승인했다. '
        '그승인으로실제7회요청후첫Timeout에서중단했다. 남은상한을새Run/새코드에자동재사용하지않는다. '
        '서버변경·다운로드·추가모델호출·임상Prompt최적화·다른환자·B3/B4·학습/fitting은수행하지않았다. '
        '실제승인기록과preflight는Private에그대로보존했다.'+common)
    md('FINAL_B0_B4_READINESS.md','# 최종 준비 상태\n\n'+table+
        '\nB1선택/새인용enum의오프라인구현은완료했지만실제B0/B1/B2평가완료와안정성은확보하지못했다. '
        '새실행의완료진단이없어2/10 기준을달성했다고판정하지않는다. B3/B4는이번범위에포함하지않았고기존Utility/Source/likelihood 차단도유지한다. '
        '기존자료·수식·Scorer·Clinical Gate를그대로보존하고부족한결과를성공으로합성하지않는다.'+common)
    md('TIMEOUT_OBSERVATION_LIMITATION.md','# Timeout 설정과 실제 wall 기록\n\n'
        '설정120초는현재httpx timeout의connect/read/write/pool에전달된다. 별도의요청전체hard wall deadline은구현돼있지않다. '
        '실패한요청의저장wall은약329초다. 설정120초이내에항상함수가반환됐다고보고하지않는다. '
        '저장된기록에는그시간을서버로딩/Prompt처리/생성/수신/OS대기로나눌정보가없다. '
        '사용자의턴중단과지연사이의인과관계도확정하지않는다. HTTP status/응답본문이없다는것을서버가전혀처리하지않았다는증거로사용하지않는다.\n\n'
        +json.dumps(s['timeout'],ensure_ascii=False,indent=2)+common)


if __name__=='__main__':main()
