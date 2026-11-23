"""Read-only saved pilot analysis; no transport, model or fitting calls."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_locdi_pilot_improvement import PUB as PREP,PR,DOC as PREPDOC,RUN,DATA
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.llm_setup import digest
from app.locdi.dataset_loader import load_cases
from app.locdi.memory_paired_recovery import aggregate,citation_audit,verify_delivered_memory
from app.locdi.synthea_diagnosis_audit import matches
from jsonschema import Draft202012Validator

PUB=PREP/'actual_results_v1';DOC=PREPDOC/'actual_results_v1'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def main():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_ACTUAL_ANALYSIS_REQUIRED')
    p=read(PREP/'SYNTHEA_PILOT_PROTOCOL.json');run=PR/RUN
    collection=read(run/'collection.json');panel=read(PR/'panel.json');index=read(PR/'case_index.json')
    cases=load_cases(DATA);catalog=read(Path(p['catalog']['path']))
    agg=aggregate(panel,collection['rows']);assert agg==read(run/'aggregate.json')
    allsource={digest(x['source_event']):x['source_event'] for x in read(PR/'source_rows.json')}
    diagnostics=[];rows=[];violations=[];checks=Counter()
    for task in collection['rows']:
        case=cases[index[task['case_ref']]-1]
        folder=run/'requests'/('Episode%04d'%task['episode_ordinal'])
        row={k:task[k] for k in ('episode_ordinal','patient_ordinal','condition','status','request_count','error_type')}
        row['original20_ordinal']=p['stratified_selection']['selected_original20_ordinals'][task['patient_ordinal']-1]
        row.pop('case_ref',None)
        row.update(Core_status=None,export_status='NOT_ATTEMPTED',certainty=None,source_exact=None,normalized=None,
            source_department=None,simulation_minutes=None,prompt_tokens=0,completion_tokens=0,HTTP_wall_seconds=0,
            additional_requested=0,additional_ordered=0,additional_started=0,additional_received=0,
            additional_citation_decisions=0,memory_record_occurrences=0,memory_boundary_errors=0,
            memory_retrievals=None,initial_retrieval_minute='NOT_RECORDED',source_value_unit_checks=0,
            actual_context_digest_checks=0,temporal_violations=0)
        rawpath=folder/'raw_runner_result.json';raw=read(rawpath) if rawpath.exists() else None
        names=set();receipts={};starts={};orders={};source_expected={}
        decisions=read(folder/'validated_ER_decisions.json') if (folder/'validated_ER_decisions.json').exists() else []
        terminal=next((d for d in reversed(decisions) if d['action']=='diagnose'),None)
        row['selected_action_counts']=dict(Counter(d['action'] for d in decisions))
        row['actual_action_counts']=(task.get('result') or {}).get('action_counts')
        if raw:
            row['Core_status']=raw['status']
            row['export_status']='EXPORTED' if (folder/'episode_and_raw.json').exists() else 'NOT_EXPORTED'
            row['memory_retrievals']=len(raw.get('telemetry',{}).get('retrievals',[]))
            added=raw.get('synthetic_source_view_audit',[{}])[0].get('added_tests',[])
            names={x['test_name'] for x in added}
            source_expected={x['test_name']:allsource[x['source_fingerprint']]['evidence'] for x in added}
            if raw.get('execution') and raw['execution'].get('visits'):
                final=raw['execution']['visits'][0]['final_state'];events=final['event_log']
                row['simulation_minutes']=max((e['minute'] for e in events),default=0)
                for e in events:
                    name=e.get('payload',{}).get('requested_test')
                    if name not in names:continue
                    if e['event_type']=='test_ordered':orders.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='test_started':starts.setdefault(name,[]).append(e['minute'])
                    if e['event_type']=='result_received':receipts.setdefault(name,[]).append(e['minute'])
                for name,times in receipts.items():
                    if not starts.get(name) or not orders.get(name) or min(times)<min(starts[name])+5 or min(starts[name])<min(orders[name]):
                        violations.append({'episode':task['episode_ordinal'],'code':'SOURCE_ORDER_START_RECEIPT'})
        if terminal:
            scored=matches(terminal['target'] if terminal.get('certainty')=='supported' else None,case['evaluation']['diagnosis_labels'])
            dept=terminal['department'] in case['evaluation']['accepted_departments_source_policy']
            diagnostics.append({'episode_ordinal':task['episode_ordinal'],'Source':case['evaluation'],
                'terminal':terminal,'Core_Source_comparison':scored,'Core_department_comparison':dept,
                'actual_exported':row['export_status']=='EXPORTED','actual_status':task['status']})
            if task['status']=='COMPLETED':
                row.update(certainty=terminal['certainty'],source_exact=scored['source_exact'],normalized=scored['normalized_match'],source_department=dept)
                expected=(task.get('result') or {})['prediction']
                assert expected['diagnosis']==(terminal['target'] if terminal['certainty']=='supported' else None)
                assert expected['department']==terminal['department']
                checks['backend_prediction_matches_terminal']+=1
        for bfile in sorted(folder.glob('Request*_before.json')):
            afile=bfile.with_name(bfile.name.replace('_before','_after'))
            if not afile.exists():
                violations.append({'episode':task['episode_ordinal'],'code':'AFTER_CHECKPOINT_MISSING'});continue
            b=read(bfile);a=read(afile);req=b['request'];payload=json.loads(req['messages'][1]['content'])
            assert digest(req)==b['request_fingerprint']==a['request_fingerprint']
            checks['before_after_fingerprint_match']+=1
            assert payload.get('diagnosis_catalog')==catalog
            assert not {'gold','evaluation','task_annotations','source_event_store','provenance'} & set(payload)
            checks['common_catalog_and_private_fields']+=1
            env=a.get('response_envelope',{});identity=a.get('identity_after',{})
            if env.get('message'):
                try:
                    response=json.loads(env['message']['content']);Draft202012Validator(req['format']).validate(response)
                    ca=citation_audit(req,env)
                    if ca['error_code'] is not None:raise ValueError(ca['error_code'])
                except Exception as exc:
                    checks['response_contract_invalid']+=1
                    violations.append({'episode':task['episode_ordinal'],'code':'MODEL_RESPONSE_CONTRACT','error_type':type(exc).__name__})
                else:
                    checks['schema_and_citation_valid']+=1
                    if response.get('action')=='request_test' and response.get('target') in names:row['additional_requested']+=1
                    records=payload['evidence_records']
                    row['additional_citation_decisions']+=any(any(records[i]['text'].startswith(n+':') for n in names)
                        for i in response.get('evidence_indices',[]) if type(i) is int and 0<=i<len(records))
            row['prompt_tokens']+=env.get('prompt_eval_count',0);row['completion_tokens']+=env.get('eval_count',0)
            row['HTTP_wall_seconds']+=a.get('wall_seconds',0)
            checks['HTTP200']+=a.get('HTTP_status')==200
            match=identity.get('context_length')==32768 and 'sha256:'+str(identity.get('digest','')).removeprefix('sha256:')==p['model_identifier']
            row['actual_context_digest_checks']+=match
            if a.get('status')=='VALID':assert match
            o=payload['observation']
            for finding in o.get('objective_findings',[]):
                for name in names:
                    if finding.startswith(name+': '):
                        if not receipts.get(name) or o['elapsed_minutes']<min(receipts[name]):
                            row['temporal_violations']+=1
                            violations.append({'episode':task['episode_ordinal'],'code':'PRE_RECEIPT_EXPOSURE'})
                        value=json.loads(finding[len(name)+2:]);expected=source_expected[name]
                        assert value['value']==expected['value'] and value['units']==expected.get('units')
                        row['source_value_unit_checks']+=1
            memory=o.get('longitudinal_memory',{})
            row['memory_record_occurrences']+=len(memory.get('records',[]))
            if task['condition']=='B0':assert not memory.get('records')
            elif memory.get('records'):
                audit=verify_delivered_memory(case,memory)
                row['memory_boundary_errors']+=audit['status']!='PASS'
                if audit['status']!='PASS':violations.append({'episode':task['episode_ordinal'],'code':'MEMORY_BOUNDARY'})
        row.update(additional_ordered=sum(map(len,orders.values())),additional_started=sum(map(len,starts.values())),
            additional_received=sum(map(len,receipts.values())))
        rows.append(row)
    assert len(rows)==20
    for actual,stored in zip(rows,agg['episodes']):
        assert actual['source_exact']==stored['source_diagnosis_exact']
        assert actual['source_department']==stored['source_department_agreement']
    paired=[]
    for i,original in enumerate(agg['pairs']):
        a,b=rows[2*i:2*i+2];item=dict(original)
        item.update(original20_ordinal=a['original20_ordinal'],B0_normalized=a['normalized'],B2_normalized=b['normalized'],
            normalized_difference=int(b['normalized'])-int(a['normalized'])
                if type(a['normalized']) is bool and type(b['normalized']) is bool else None)
        paired.append(item)
    totals={}
    for condition in ('B0','B2'):
        rs=[r for r in rows if r['condition']==condition];done=[r for r in rs if r['status']=='COMPLETED']
        supported=[r for r in done if r['certainty']=='supported']
        totals[condition]={'registered':10,'states':dict(Counter(r['status'] for r in rs)),
            'completed':len(done),'supported':len(supported),'unresolved':sum(r['certainty']=='unresolved' for r in done),
            'Source_exact_correct_registered':{'numerator':sum(r['source_exact'] is True for r in rs),'denominator':10},
            'Source_exact_supported':{'numerator':sum(r['source_exact'] is True for r in supported),'denominator':len(supported)},
            'Normalized_correct_registered':{'numerator':sum(r['normalized'] is True for r in rs),'denominator':10},
            'Source_department_completed':{'numerator':sum(r['source_department'] is True for r in done),'denominator':len(done)}}
        for key in ('request_count','prompt_tokens','completion_tokens','HTTP_wall_seconds','additional_requested',
            'additional_ordered','additional_started','additional_received','additional_citation_decisions',
            'source_value_unit_checks','memory_record_occurrences','temporal_violations','memory_boundary_errors','actual_context_digest_checks'):
            totals[condition][key]=sum(r[key] for r in rs)
    state={'run_id':RUN,'protocol':p['fingerprint'],'model':p['model'],'model_identifier':p['model_identifier'],
        'Source_rows_fingerprint':p['source_rows_fingerprint'],'Catalog_SHA256':p['catalog']['SHA256'],
        'Code_pins_fingerprint':digest(p['new_code_hashes']),'Data_SHA256':sha(DATA),
        'configuration':{k:p[k] for k in ('context','temperature','seed','max_output','timeout_seconds','max_generations',
            'per_episode','wall_seconds','process_bytes','output_bytes','Memory','Routing','VoI','Retry','Repair','Warmup')},
        'generation':collection['generation_count'],
        'global_stop':collection['global_stop_reason'],'conditions':totals,'complete_pairs':agg['complete_pairs'],
        'scorable_pairs':agg['source_scorable_pairs'],'paired_Source_exact_B2_minus_B0':agg['paired_source_exact_B2_minus_B0'],
        'clinical_accuracy':'NOT_EVALUATED','Verified_Concept_Match':'NOT_EVALUATED',
        'checks':dict(checks),'violations':violations,'wall_seconds':read(PREP/'pilot_execution_summary.json')['wall_seconds'],
        'new_generation_in_analysis':0}
    write(PR/'actual_diagnostic_analysis.json',diagnostics)
    write(PUB/'actual_pilot_metrics.json',{'summary':state,'rows':rows,'aggregate':agg})
    csv_file(PUB/'FINAL_EPISODE_RESULTS.csv',rows,list(rows[0]))
    csv_file(PUB/'FINAL_PAIRED_RESULTS.csv',paired,list(paired[0]))
    common='\n\nSource 일치는 합성 Source Reference와의 일치이며 임상 정확도가 아니다. 기존 실패 Run과 정적D4 결과는 변경하거나 합산하지 않는다. 추가 생성·재시도·Prompt 최적화·나머지10명 실행은 하지 않는다.\n'
    common+=f'\nRun `{RUN}`, Protocol `{p["fingerprint"]}`, Model `{p["model_identifier"]}`, '
    common+=f'Source `{p["source_rows_fingerprint"]}`, Code Pin 집합 `{digest(p["new_code_hashes"])}`. '
    common+='실제설정·전체코드별SHA·분모·행별실패상태는actual_pilot_metrics.json 및 상위SYNTHEA_PILOT_PROTOCOL.json에보존했다.\n'
    table='|조건|등록|완료|Supported|Unresolved|Source Exact/등록|Normalized/등록|진료과/완료|Generation|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for c,t in totals.items():
        table+=f'|{c}|10|{t["completed"]}|{t["supported"]}|{t["unresolved"]}|{t["Source_exact_correct_registered"]["numerator"]}/10|{t["Normalized_correct_registered"]["numerator"]}/10|{t["Source_department_completed"]["numerator"]}/{t["completed"]}|{t["request_count"]}|\n'
    md('FINAL_SEQUENTIAL_PILOT_RESULTS.md', '# 실제 신규 순차 Pilot 결과\n\n'+table+'\n'+json.dumps(state,ensure_ascii=False,indent=2)+common)
    md('FINAL_B0_B2_MEMORY_ANALYSIS.md','# 실제 B0/B2 paired 결과\n\n'+table+
        f'\n완료쌍 {agg["complete_pairs"]}/10, supported 진단 Source scorable쌍 {agg["source_scorable_pairs"]}/10. '
        f'기존scorer의완료/scorable조건부 B2−B0={agg["paired_source_exact_B2_minus_B0"]}. '
        '실패·미실행·unresolved를임상오답라벨로변환하지않으며 registered 성공수율과분리한다. '
        'Source 진단·부서·검사결과 선택 차이는10명개발패널의탐색결과로만해석한다. '
        'Memory token만의서버분리계측은없어전체request prompt token을보고한다.'+common)
    md('FINAL_SYNTHEA_ACCURACY_IMPROVEMENT.md','# 실제 진단 결과와 개선 판정\n\n'+table+
        '\n기존D4 정적4/20과동일지표라도공개방식·분모·행동상호작용이다르므로직접효과차이로빼지않는다. '
        '이번조건은공통15Catalog/확대Source/합성공개정책을양쪽에동일하게썼고이전실패순차실행대비Exporter만수정했다. '
        '높은정확도를확보했다고가정하지않고위실제정답수로판정한다. '
        '새Prompt나환자별Gold검사선택없이실행한최종기술/개발평가다.'+common)
    for name in ('PILOT_DATA_USAGE_COMPARISON.md','CURRENT_SYNTHEA_INPUT_IMPROVEMENT.md','FINAL_EXPORTER_VERIFICATION.md'):
        md(name,(PREP/name).read_text(encoding='utf-8')+'\n\n## 실제 실행 후 추가 확인\n\n'+
            f'Run {RUN}, generation {state["generation"]}, 완료쌍{agg["complete_pairs"]}/10. 저장Checkpoint 검사: '+
            json.dumps(dict(checks),ensure_ascii=False)+f'. 공개/Memory 위반기록 {len(violations)}건. 실제세부는actual_pilot_metrics.json/Private Checkpoint를참조한다.'+common)
    md('FINAL_THESIS_RESULTS_COMPARISON.md','# 논문 결과 대비 최종 기록\n\n'+table+
        '\n역사적B0/B1/B2/B3/B4=38/52/44/48/90%는현재Run재현값아니다. 기존정적D4=4/20, 이번순차선정10명결과, 기존실패Run등록20/완료쌍0을별도보고한다. '
        '현재3B·ER-only·VoI OFF·개발노출Synthea·Source 일치라는연구범위를명시하고공식matched B0~B4나임상우월성으로승격하지않는다. '
        '초기chief complaint/narrative부족, 합성5분및recorded proxy, ordinal12다음날Reference, 독립임상검토없음한계가남는다.'+common)
    md('FINAL_PROFESSOR_BRIEFING.md','# 실제 최종 실행 보고\n\n'+table+
        f'\n승인된새Run {RUN}을1회실행했고실제요청{state["generation"]}회, 완료쌍{agg["complete_pairs"]}/10이다. '
        f'전체중단이유 {state["global_stop"]}. '
        'Exporter 인터페이스를고쳐실제저장가능성을검증하되증거이상의진단개선/Memory효과를주장하지않는다. '
        '수식·모델·Prompt·Reference·Scorer는동결상태다. 결과가불리해도보존하며추가최적화실험을자동으로시작하지않는다.'+common)
    protected=check(read(PR/'preservation_before.json'));assert protected['status']=='PASS'
    pins=check({k:v for g in ('input_hashes','new_code_hashes','binding_hashes') for k,v in p[g].items()})
    ids=[c['patient_id'] for c in cases]+[c['agent_visible']['index_encounter_id'] for c in cases]
    for base in (PUB,DOC):
        for f in base.rglob('*'):
            if f.is_file():assert not any(i in f.read_text(encoding='utf-8-sig') for i in ids)
    write(PUB/'FINAL_ACTUAL_VERIFICATION.json',{'status':'RECORDED_NO_AUTOMATIC_FOLLOWUP','actual':state,
        'protected':protected,'pins':pins,'reports':{f.name:sha(f) for f in DOC.glob('*.md')},
        'Run_hashes':{str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file()},
        'analyzer_SHA256':sha(Path(__file__)),'at':datetime.now(timezone.utc).isoformat()})
    print(json.dumps(state,ensure_ascii=True))


if __name__=='__main__':main()
