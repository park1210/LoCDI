"""Analyze the stopped Pilot without resuming, regenerating or changing scores."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_synthea_sequential_pilot import PUB as PREP,PR,DOC as PREPDOC,RUN,load_protocol,DATA
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.dataset_loader import load_cases
from app.locdi.memory_paired_recovery import verify_delivered_memory,citation_audit
from app.locdi.synthea_diagnosis_audit import matches
from app.locdi.llm_setup import digest
from jsonschema import Draft202012Validator

PUB=PREP/'actual_results_v1';DOC=PREPDOC/'actual_results_v1'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    started=time.monotonic()
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_ACTUAL_ANALYSIS_REQUIRED')
    p=load_protocol();run=PR/RUN;collection=read(run/'collection.json');agg=read(run/'aggregate.json')
    cases=load_cases(DATA);index=read(PR/'case_index.json');out=[];private=[];counts=Counter();violations=[]
    for task in collection['rows']:
        row={'episode_ordinal':task['episode_ordinal'],'patient_ordinal':task['patient_ordinal'],
            'condition':task['condition'],'collection_status':task['status'],'generation':task['request_count'],
            'Core_status':None,'export_status':'NOT_ATTEMPTED','actual_terminal_certainty':None,
            'Source_exact_for_completed_Pilot':None,'normalized_for_completed_Pilot':None,
            'Source_department_for_completed_Pilot':None,'additional_LLM_requests':0,'additional_orders':0,
            'additional_started':0,'additional_received':0,'additional_citation_decisions':0,
            'temporal_violations':0,'Memory_payloads':0,'Memory_record_occurrences':0,
            'Memory_boundary_errors':0,'logged_retrievals':None,'initial_retrieval_minute':'NOT_RECORDED',
            'prompt_tokens':0,'output_tokens':0,'HTTP_wall_seconds':0,'identity_checks':0,
            'error_type':task['error_type']}
        folder=run/'requests'/('Episode%04d'%task['episode_ordinal'])
        rawpath=folder/'raw_runner_result.json';raw=read(rawpath) if rawpath.exists() else None
        case=cases[index[task['case_ref']]-1]
        names=set();received={};ordered={};starts={};details=[]
        if raw:
            row['Core_status']=raw['status']
            row['export_status']='EXPORTED' if (folder/'episode_and_raw.json').exists() else 'EXPORT_FAILED_AFTER_CORE_COMPLETION' if raw['status']=='completed' else 'NO_COMPLETE_EPISODE_EXPORT'
            names={x['test_name'] for x in raw.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])}
            visits=(raw.get('execution') or {}).get('visits',[])
            events=visits[0]['final_state']['event_log'] if visits else []
            for e in events:
                name=e.get('payload',{}).get('requested_test')
                if name not in names:continue
                if e['event_type']=='test_ordered':ordered[name]=e['minute']
                if e['event_type']=='test_started':starts[name]=e['minute']
                if e['event_type']=='result_received':received[name]=e['minute']
            for name,minute in received.items():
                if name not in ordered or name not in starts or not ordered[name]<=starts[name]<=minute or minute<starts[name]+5:
                    violations.append({'episode':task['episode_ordinal'],'code':'REQUEST_START_RELEASE_ORDER'})
            row.update(additional_orders=len(ordered),additional_started=len(starts),additional_received=len(received),
                logged_retrievals=len(raw.get('telemetry',{}).get('retrievals',[])))
            decisions=read(folder/'validated_ER_decisions.json')
            terminal=next((d for d in reversed(decisions) if d.get('action')=='diagnose'),None)
            if terminal:
                row['actual_terminal_certainty']=terminal.get('certainty')
                m=matches(terminal['target'] if terminal.get('certainty')=='supported' else None,case['evaluation']['diagnosis_labels'])
                # Keep recovered Core diagnostics separate. A failed actual
                # Pilot/export is never promoted into the completed denominator.
                if task['status']=='COMPLETED':
                    row['Source_exact_for_completed_Pilot']=m['source_exact'];row['normalized_for_completed_Pilot']=m['normalized_match']
                    row['Source_department_for_completed_Pilot']=terminal['department'] in case['evaluation']['accepted_departments_source_policy']
                private.append({'episode_ordinal':task['episode_ordinal'],'terminal':terminal,
                    'unscored_Core_Source_comparison':m,'reference':case['evaluation']['diagnosis_labels'],
                    'unscored_Core_department_match':terminal['department'] in case['evaluation']['accepted_departments_source_policy'],
                    'do_not_promote_failed_export_to_completed':True})
        for before in sorted(folder.glob('Request*_before.json')):
            after=before.with_name(before.name.replace('_before','_after'))
            if not after.exists():continue
            b=read(before);a=read(after);req=b['request'];payload=json.loads(req['messages'][1]['content'])
            counts['stored_request_response_pairs']+=1
            if digest(req)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint']:
                violations.append({'episode':task['episode_ordinal'],'code':'CHECKPOINT_FINGERPRINT'})
            counts['HTTP200']+=a.get('HTTP_status')==200
            env=a.get('response_envelope',{});row['prompt_tokens']+=env.get('prompt_eval_count',0);row['output_tokens']+=env.get('eval_count',0)
            row['HTTP_wall_seconds']+=a.get('wall_seconds',0)
            identity=a.get('identity_after',{})
            counts['context_and_digest_match']+=(identity.get('context_length')==32768 and
                'sha256:'+str(identity.get('digest','')).removeprefix('sha256:')==p['model_identifier'])
            if env.get('message'):
                response=json.loads(env['message']['content']);Draft202012Validator(req['format']).validate(response)
                counts['schema_valid']+=1;ca=citation_audit(req,env)
                counts['citation_contract_valid']+=ca['error_code'] is None
                records=payload['evidence_records'];indices=response.get('evidence_indices',[])
                extra_cited=any(records[i]['kind']=='objective_finding' and any(records[i]['text'].startswith(name+':') for name in names) for i in indices if type(i) is int and 0<=i<len(records))
                row['additional_citation_decisions']+=extra_cited
                if response.get('action')=='request_test' and response.get('target') in names:row['additional_LLM_requests']+=1
                details.append({'request':before.name,'action':response['action'],'target':response['target'],
                    'evidence_indices':indices,'extra_cited':extra_cited,'minute':payload['observation']['elapsed_minutes']})
            observation=payload['observation'];minute=observation['elapsed_minutes']
            for finding in observation.get('objective_findings',[]):
                for name in names:
                    if finding.startswith(name+':') and (name not in received or minute<received[name]):
                        row['temporal_violations']+=1;violations.append({'episode':task['episode_ordinal'],'code':'PRE_RECEIPT_MODEL_EXPOSURE'})
            memory=observation.get('longitudinal_memory',{})
            if memory.get('records'):row['Memory_payloads']+=1
            row['Memory_record_occurrences']+=len(memory.get('records',[]))
            if task['condition']=='B0' and memory.get('records'):
                row['Memory_boundary_errors']+=1
            if task['condition']=='B2' and memory.get('records'):
                row['Memory_boundary_errors']+=verify_delivered_memory(case,memory)['status']!='PASS'
        if raw:row['identity_checks']=(raw.get('llm_execution') or {}).get('identity_check_count',0)
        if details:private.append({'episode_ordinal':task['episode_ordinal'],'request_action_source_trace':details})
        out.append(row)
    write(PR/'post_failure_repair_v1/actual_private_trace_analysis.json',private)
    totals={'registered_episodes':20,'registered_patients':10,'states':dict(Counter(r['collection_status'] for r in out)),
        'Core_completed':sum(r['Core_status']=='completed' for r in out),'actual_exported':sum(r['export_status']=='EXPORTED' for r in out),
        'export_failed_after_Core_completed':sum(r['export_status']=='EXPORT_FAILED_AFTER_CORE_COMPLETION' for r in out),
        'generation':collection['generation_count'],'global_stop':collection['global_stop_reason'],
        'complete_pairs':agg['complete_pairs'],'scorable_pairs':agg['source_scorable_pairs'],
        'B2_minus_B0_Source_exact':None,'clinical_accuracy':'NOT_EVALUATED','concept_match':'NOT_EVALUATED',
        'additional_LLM_requests':sum(r['additional_LLM_requests'] for r in out),
        'additional_orders':sum(r['additional_orders'] for r in out),'additional_started':sum(r['additional_started'] for r in out),
        'additional_received':sum(r['additional_received'] for r in out),
        'extra_citation_decisions':sum(r['additional_citation_decisions'] for r in out),
        'temporal_violations':sum(r['temporal_violations'] for r in out),'Memory_boundary_errors':sum(r['Memory_boundary_errors'] for r in out),
        'prompt_tokens':sum(r['prompt_tokens'] for r in out),'output_tokens':sum(r['output_tokens'] for r in out),
        'HTTP_wall_seconds':sum(r['HTTP_wall_seconds'] for r in out),
        'executed_wall_seconds':read(PREP/'pilot_execution_summary.json')['wall_seconds']}
    write(PUB/'actual_pilot_metrics.json',{'summary':totals,'rows':out,'original_aggregate':agg,
        'request_validation':dict(counts),'boundary_violations':violations})
    csv_file(PUB/'actual_episode_metrics.csv',out,list(out[0]))
    csv_file(PUB/'SYNTHEA_PILOT_PAIRED_RESULTS.csv',agg['pairs'],list(agg['pairs'][0]))
    protected=check(read(PR/'preservation_before_actual_pilot.json'))
    # Explicitly record the one post-run offline exporter repair, if the prior
    # manifest happened to pin that code. Do not rehash old completed manifests.
    exporter=ROOT/'app/locdi/episode_exporter.py'
    beforefix=read(PR/'post_failure_repair_v1/before_fix.json')
    repair={'file':'app/locdi/episode_exporter.py','before_SHA256':beforefix['SHA256'],'after_SHA256':sha(exporter),
        'version_before':'1.3.1','version_after':'1.3.2','timing':'after actual Run stopped','generation_after_fix':0}
    write(PUB/'post_run_exporter_repair.json',repair)
    md('SYNTHEA_PILOT_RESULTS.md','# 실제 순차 Pilot — 첫 기술 오류에서 중단\n\n'
        f'Run `{RUN}`, 실제Generation{totals["generation"]}회. 등록20 Episode 중CONTRACT_FAILED1, NOT_ATTEMPTED19다. '
        '첫환자 B0 Core는completed이고마지막certainty는supported였지만, Exporter에서Configuration fingerprint mismatch가발생했다. '
        '실제완료Pilot Episode/Export0, 완료쌍0이다. Core완료와Pilot/Export완료를같게취급하지않는다. '
        'B2는전부미실행이므로Memory 효과·SourceExact/Normalized·진료과비교·B2−B0 효과는NOT_EVALUATED다. '
        'Source측정분모를성공한Core1건으로바꾸거나, Export실패를Source 진단오답으로변환하지않는다.\n\n'
        '실제 측정 집계:\n\n'+json.dumps(totals,ensure_ascii=False,indent=2)+
        '\n\n추가검사주문/수신/인용과시간경계는실제로저장된첫B0의범위만검사했다. 미실행B2의Memory 동작을실제검증했다고주장하지않는다. '
        'Memory token만분리한서버계측은없으며request전체prompt token만보고한다. 초기retrieval minute의미기록상태를유지한다. '
        'SourceAgreement가있는저장진단도실패한Pilot의성공으로승격하지않고Private 보조분석으로분리했다.\n')
    md('SYNTHEA_EXPORT_FAILURE_AND_OFFLINE_REPAIR.md','# Export 인터페이스 오류와 비생성 수정\n\n'
        '원인: run_simulation에새로기록한configuration.sequential_source_view가episode_exporter._configuration의허용키에없었다. '
        'Exporter의set(raw)==set(out) 및configuration_fingerprint검증에서거부됐다. Runner/Scheduler 회귀에Exporter 통합경로가누락됐다. '
        '이오류를저장된raw_runner_result로Generation0회재현했다. 모델·Scheduler·Source 결과공개실패가아닌Export 메타데이터계약오류다.\n\n'
        '수정: app/locdi/episode_exporter.py::_configuration에새optionalmetadata의정확한7개키/타입/정책명/5분/clinicalFalse/grant0/64hex fingerprint만허용했다. '
        '기존전체fingerprint검증·unknown key차단을유지하며스킵하지않는다. Exporter1.3.1→1.3.2. '
        '정적/기존옵션은변경없고Core/수식/Source/스코어/Prompt/Evidence Validator는변경하지않았다. '
        '수정전원문은Private post_failure_repair_v1/episode_exporter_before_fix.py에보존했다. '
        '저장raw의수정후projection은OFFLINE_ONLY파일로분리했고기존Run/실패상태를고치지않았다. '
        '추가생성·동일Run resume·자동재시도는0이다.\n')
    md('FINAL_B0_B2_DECISION.md','# 최종 판단\n\n'
        '이번실제Pilot은첫Export 오류로중단되어B0/B2비교를완료하지못했다. 정적D4 4/20을이Pilot 또는Memory 효과로대체하지않는다. '
        '실제첫B0에서검사요청→수신→이후관측경로가실행됐으나, B2/완료쌍이없어진단개선가능성이나Memory 효과는판정불가다. '
        '나머지10명확장·480회·기존D0~D4 재실행은하지않았다.\n\n'
        '확인된기술오류만비생성오프라인으로수정/검증했다. 기존Run을재사용하거나남은19건만resume하지않는다. '
        '재실행이필요하면수정된Exporter를포함하는새Code/Protocol fingerprint와미사용Run ID로별도운영자승인을받아야한다. '
        '이번승인을새코드/새Run에자동재사용하지않는다. 임상공개/v1 grant/임상Reference는여전히미검토다. '
        '오늘종료자료에는실제실패와미실행19건을그대로보존하고, 후속연구단계를자동으로추가하지않는다.\n')
    tests=read(PR/'post_failure_repair_v1/test_results.json')
    md('SYNTHEA_SEQUENTIAL_REGRESSION_REPORT.md','# 실행 후 비생성 회귀\n\n'+json.dumps(tests,ensure_ascii=False,indent=2)+
        '\n\n실행전관련221개통과기록은이전문서에보존했다. 수정후6개Exporter 경계테스트를추가했다. '
        '실행후회귀는테스트응답과저장Trace만사용했으며모델재호출0이다. 전체저장소테스트PASS로확대하지않는다. '
        '실제Run의성공으로소급하지않는다.\n')
    ids=[c['patient_id'] for c in cases]+[c['agent_visible']['index_encounter_id'] for c in cases]
    leaks=[]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.csv','.md'):
                t=f.read_text(encoding='utf-8-sig')
                if any(x in t for x in ids):leaks.append(str(f))
    assert not leaks
    write(PUB/'actual_delivery_manifest.json',{'status':'ACTUAL_PILOT_STOPPED_EXPORT_FAILED_OFFLINE_FIX_VERIFIED',
        'summary':totals,'request_validation':dict(counts),'boundary_violations':violations,'public_identifier_leaks':leaks,
        'protected':protected,'post_run_code_change':repair,'tests_after_fix':tests,
        'new_generation_in_analysis':0,'elapsed_analysis_seconds':time.monotonic()-started,
        'old_NOT_RUN_reports_preserved':True,'actual_failed_Run_preserved':True,
        'reports':{f.name:sha(f) for f in DOC.glob('*.md')},'analyzer_SHA256':sha(Path(__file__))})
    print(json.dumps({'status':'RECORDED','summary':totals,'request_checks':dict(counts),'violations':violations,'protected':protected}))


if __name__=='__main__':main()
