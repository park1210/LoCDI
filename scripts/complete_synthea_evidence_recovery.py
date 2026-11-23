"""Bounded source audit, proposed distinct proxy policy, authorized D2/D4 only.

Existing source_check/v1 remains fail-closed; no old clinical release grant is
created or emulated. Default operation is offline preparation. Python3.9.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file,metadata,MODEL3,DIGEST3
from scripts.run_synthea_diagnostic_snapshots import packet,frozen_record
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.routing_data_reuse import SYNTHEA
from app.locdi.synthea_diagnostic_snapshot import safe_observation,prepare_snapshot,shared_catalog
from app.locdi.synthea_development_proxy import POLICY,row_checks,resolve_duplicates,draft_observation,verify_authorization
from app.locdi.current_ablation_llm_backend import BudgetedNativeERClient,input_guard
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes
from app.locdi.ollama_adapter import ModelIdentityError
from app.locdi.synthea_diagnosis_audit import matches
from app.core.longitudinal_memory import timestamp

NS='FINAL_SYNTHEA_COMPLETION_v1'
PUB=ROOT/'outputs/research_evaluation'/NS
PR=ROOT/'outputs/private/research_evaluation'/NS
DOC=ROOT/'docs/연구 평가'/NS
BC=ROOT/'outputs/private/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'
BCPUB=ROOT/'outputs/research_evaluation/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'
RUN='synthea_d2_d4_001'
CODE=['app/locdi/synthea_development_proxy.py','scripts/complete_synthea_evidence_recovery.py',
      'tests/locdi/test_synthea_development_proxy.py']


def now():return datetime.now(timezone.utc).isoformat()


def report_file(name,text):
    for base in (PUB,DOC):
        p=base/name;p.parent.mkdir(parents=True,exist_ok=True)
        content=text.rstrip()+'\n'
        if p.exists():
            if p.read_text(encoding='utf-8')!=content:raise ValueError('EXISTING_REPORT_DIFFERENT_NO_OVERWRITE')
            continue
        with p.open('x',encoding='utf-8') as f:f.write(content)


def plan():
    final=PUB/'d2_d4_protocol_final.json'
    return read(final if final.exists() else PUB/'d2_d4_protocol.json')


def prepare():
    if PUB.exists() or PR.exists() or DOC.exists():raise ValueError('FRESH_COMPLETION_NAMESPACE_REQUIRED')
    start=time.monotonic()
    def deadline():
        if time.monotonic()-start>3600:raise ValueError('EVIDENCE_REVIEW_60MIN_LIMIT')
    pins=read(BC/'preservation_before.json')
    for folder in (BC,BCPUB,ROOT/'docs/연구 평가/FINAL_SYNTHEA_DIAGNOSIS_RECOVERY_BC_v1'):
        pins.update({str(f):sha(f) for f in folder.rglob('*') if f.is_file()})
    # The unchanged strict validator, model adapter, scorer, Prompt, and formulas
    # are protected, not silently re-pinned within a completed old manifest.
    for f in ('app/locdi/synthea_diagnostic_snapshot.py','app/locdi/current_ablation_llm_backend.py',
        'app/locdi/synthea_diagnosis_audit.py','scripts/run_synthea_diagnostic_snapshots.py'):
        pins[str(ROOT/f)]=sha(ROOT/f)
    if check(pins)['status']!='PASS':raise ValueError('EXISTING_ASSET_CHANGED')
    write(PR/'preservation_before.json',pins)
    inventory=read(BC/'observation_inventory.json')
    candidates=[r for r in inventory if r['additional378_candidate']
                and str(r['source_event']['target']['code'])!='72514-3']
    if len(candidates)!=358:raise ValueError('FIXED358_CHANGED')
    wanted=defaultdict(set)
    for r in candidates:
        m=r['source_event']['source_metadata'];wanted[m['source_file']].add(m['source_row'])
    actual={};raw_hashes={}
    for name,numbers in wanted.items():
        path=(SYNTHEA/name).resolve()
        if SYNTHEA.resolve() not in path.parents or not path.is_file():raise ValueError('RAW_SOURCE_FILE_MISSING')
        raw_hashes[str(path)]=sha(path);deadline()
        last_row=max(numbers)
        with path.open(encoding='utf-8-sig',newline='') as f:
            for n,row in enumerate(csv.DictReader(f),2):
                if n%10000==0:deadline()
                if n in numbers:actual[name,n]=row
                if n>=last_row:break
    catalog=shared_catalog(read(CATALOG),expected_sha=sha(CATALOG),path=CATALOG)
    audit=[];qualified=[];reference=[]
    for ordinal in range(1,21):
        d=packet(ordinal);ident=d['private_identity']
        ref=d['reference_sources_and_time']
        reference.append({'patient_ordinal':ordinal,'index':ident['started_at'],'reference_sources':ref,
            'reference_status':'UNCHANGED_SOURCE_NOT_CLINICAL_GOLD'})
        for r in [x for x in candidates if x['patient_ordinal']==ordinal]:
            e=r['source_event'];m=e['source_metadata']
            verified=actual.get((m['source_file'],m['source_row']))==m['source_record']
            a=row_checks(e,patient=ident['patient_id'],visit=ident['visit_id'],
                index_ms=int(timestamp(ident['started_at']).timestamp()*1000),cutoff=r['cutoff_minute'],
                source_row_matches=verified,catalog=catalog)
            a.update(patient_ordinal=ordinal,source_event_id=r['source_event_id'])
            a['reference_order_status']='BEFORE_SOURCE_CONDITION_DAY' if all(
                e['timestamp'][:10]<x['source']['source_record']['START'][:10] for x in ref) else 'SAME_DAY_INTRADAY_ORDER_UNKNOWN'
            audit.append(a)
            if a['structural_candidate']:qualified.append(r)
    proposed,duplicates=resolve_duplicates(qualified)
    write(PR/'source_row_validation.json',audit)
    write(PR/'source_duplicate_audit.json',duplicates)
    write(PR/'source_reference_timing.json',reference)
    write(PR/'proposed_source_rows_NOT_AUTHORIZED.json',proposed)
    write(PR/'raw_file_hashes.json',raw_hashes)
    groups=defaultdict(list)
    for r in candidates:
        e=r['source_event'];groups[(str(e['target']['code']),e['target']['description'])].append(r)
    accepted_ids={r['source_event_id'] for r in proposed}
    types=[]
    for (code,name),rr in sorted(groups.items()):
        types.append({'code':code,'description':name,'patients':len({x['patient_ordinal'] for x in rr}),
            'records':len(rr),'units':sorted({str(x['source_event']['evidence'].get('units')) for x in rr}),
            'proposed_records':sum(x['source_event_id'] in accepted_ids for x in rr),
            'repeats_existing_four':code in ('8462-4','8480-6','8867-4','9279-1')})
    perpatient=[{'patient_ordinal':i,'candidate_records':sum(r['patient_ordinal']==i for r in candidates),
        'candidate_codes':len({r['source_event']['target']['code'] for r in candidates if r['patient_ordinal']==i}),
        'proposed_records':sum(r['patient_ordinal']==i for r in proposed),
        'proposed_codes':len({r['source_event']['target']['code'] for r in proposed if r['patient_ordinal']==i})}
        for i in range(1,21)]
    summary={'candidate_rows':358,'types':types,'type_count':len(types),'patients':perpatient,
        'proposed_rows':len(proposed),'structurally_qualified':len(qualified),
        'checks':{k:sum(x['checks'][k] for x in audit) for k in audit[0]['checks']},
        'strict_v1_states':dict(Counter(x['strict_v1_status'] for x in audit)),
        'duplicate_status_counts':dict(Counter(x['status'] for x in duplicates)),
        'reference_order_counts':dict(Counter(x['reference_order_status'] for x in audit)),
        'actual_clinical_release_or_semantic_review':'NOT_REVIEWED','grants_created':0}
    write(PUB/'additional_test_type_summary.json',summary)
    config=read(BC/'llm_config.json');prepared=validate_setup(config)
    if prepared['static_readiness']!='READY':raise ValueError('EXISTING_3B_SETUP_BLOCKED')
    write(PR/'llm_config.json',config)
    oldslots=read(BC/'snapshot_requests.json');drafts=[];comparability=[]
    for ordinal in range(1,21):
        deadline();d=packet(ordinal)
        old1=next(x for x in oldslots if x['patient_ordinal']==ordinal and x['condition']=='D1')
        old3=next(x for x in oldslots if x['patient_ordinal']==ordinal and x['condition']=='D3')
        base=safe_observation(d['terminal_observation'])
        # Verify unchanged baseline Prompt/schema/Core projection BEFORE adding
        # the sole new evidence variable. No old calls are replayed.
        for label,old in ((None,old1),(catalog,old3)):
            capture=prepare_snapshot(base,registry=prepared['prepared']['registry'],catalog=label)
            if digest(frozen_record(capture))!=old['fingerprint']:
                raise ValueError('D1_D3_BASELINE_CONTRACT_NOT_REPRODUCED')
        add=[r for r in proposed if r['patient_ordinal']==ordinal]
        draft=draft_observation(base,add)  # OFFLINE proposal, never source_view/grant.
        captures=[]
        for condition,label in (('D2',None),('D4',catalog)):
            capture=prepare_snapshot(draft,registry=prepared['prepared']['registry'],catalog=label)
            record=frozen_record(capture)
            guard=input_guard(record['system'],json.dumps(record['payload'],ensure_ascii=False),record['schema'])
            drafts.append({'patient_ordinal':ordinal,'condition':condition,'status':'DRAFT_NOT_AUTHORIZED',
                'cutoff_minute':old1['cutoff_minute'],'additional_records':len(add),
                'source_fingerprints':[digest(r['source_event']) for r in add],
                'contract':record,'fingerprint':digest(record),'input_guard':guard})
            captures.append(capture)
        for key in ('observation','evidence','evidence_records','supported_diagnosis_eligible_indices'):
            if captures[0]['payload'][key]!=captures[1]['payload'][key]:raise ValueError('D2_D4_EVIDENCE_MISMATCH')
        for new,old in zip(captures,(old1,old3)):
            if new['system']!=old['contract']['system']:raise ValueError('SYSTEM_PROMPT_CHANGED')
            # Only evidence and resulting eligible citation indices/array bounds
            # differ; the output's diagnosis/certainty contract is unchanged.
            for key in ('instruction','diagnosis_catalog'):
                if new['payload'].get(key)!=old['contract']['payload'].get(key):raise ValueError('PROMPT_OR_CATALOG_CHANGED')
        comparability.append({'patient_ordinal':ordinal,'D1_D3_baseline_exact_reconstruction':True,
            'system_and_instruction_unchanged':True,'D2_D4_same_evidence':True,'cutoff_minute':old1['cutoff_minute'],
            'additional_count':len(add),'measurement_view_applied':False})
    write(PR/'draft_snapshot_requests_NOT_AUTHORIZED.json',drafts)
    write(PUB/'baseline_and_paired_input_verification.json',comparability)
    write(PUB/'evidence_review_timing.json',{'started_at':now(),'elapsed_seconds':time.monotonic()-start,
        'budget_seconds':3600,'completed_within_budget':time.monotonic()-start<=3600,'new_generation':0})
    protocol={'policy':POLICY,'namespace':NS,'run_id':RUN,'purpose':'SYNTHETIC_STATIC_D2_D4_ONLY',
        'status':'PROPOSED_NEW_POLICY_AND_EXECUTION_NOT_AUTHORIZED','registered_patients':20,'conditions':['D2','D4'],
        'cohort_original_ordinals':list(range(11,31)),'order':'patient input1..20; each D2 then D4',
        'candidate_rows':358,'proposed_rows':len(proposed),'type_count':len(types),
        'duplicate_rule':'same patient/visit/code/description/time/unit: exclude all different-value rows; identical value retain first raw row',
        'source_selection':'ALL qualified rows by common rule, never reference-dependent; no per-case gold or diagnosis hints',
        'recorded_time_proxy':'verified timestamp second upper bound <= same original cutoff45; actual clinical release NOT_REVIEWED',
        'post_diagnosis_order':'18 same-day cases intraday unknown; not a prospective initial diagnosis benchmark',
        'reference_limits':'ordinal12,16 Source Condition next-day; keep20 and original reference, no clinical approval',
        'strict_v1':'UNCHANGED_BLOCKED_NO_GRANTS; distinct operator-authorized synthetic snapshot policy only',
        'model':MODEL3,'model_identifier':DIGEST3,'context':32768,'temperature':0,'seed':42,
        'max_output':1024,'timeout':120,'retry':0,'repair':0,'warmup':0,'max_generation':40,'per_slot':1,
        'wall_seconds':1800,'process_bytes':4*1024**3,'output_bytes':1024**3,
        'Memory':'OFF','Routing':'ER_ONLY_NO_SELECTION','VoI':'OFF','measurement_view':False,
        'no_resume':True,'no_server_restart_download_or_unload':True,
        'clinical_release_or_reference_approval':False,'concept_match':'NOT_EVALUATED',
        'stop':'Source/Identity/global violation => all stop; local response errors retained, no retry/repair',
        'bounds':'pre/post request guards; individual bounded call/write may transiently overshoot',
        'input_hashes':{str(PR/n):sha(PR/n) for n in ('draft_snapshot_requests_NOT_AUTHORIZED.json','proposed_source_rows_NOT_AUTHORIZED.json','source_row_validation.json','raw_file_hashes.json','llm_config.json')},
        'source_and_old_code_pins':{str(CATALOG):sha(CATALOG),**{str(ROOT/n):sha(ROOT/n) for n in
            ('app/locdi/synthea_diagnostic_snapshot.py','app/locdi/current_ablation_llm_backend.py','app/locdi/synthea_diagnosis_audit.py')}},
        'new_code_pins':{n:sha(ROOT/n) for n in CODE},
        'created_at':now()}
    protocol['fingerprint']=digest(protocol);write(PUB/'d2_d4_protocol.json',protocol)
    write(PUB/'authorization_status.json',{'status':'PENDING_NEW_POLICY_AND_NEW_RUN_APPROVAL',
        'clinical_review':'NOT_REVIEWED','strict_v1_grants':0,'generations':0})
    print(json.dumps({'status':'PREPARED_NOT_AUTHORIZED','proposed_rows':len(proposed),'fingerprint':protocol['fingerprint']}))


def build_draft(slot,prepared,catalog):
    d=packet(slot['patient_ordinal'])
    source=[r for r in read(PR/'proposed_source_rows_NOT_AUTHORIZED.json') if r['patient_ordinal']==slot['patient_ordinal']]
    base=draft_observation(safe_observation(d['terminal_observation']),source)
    capture=prepare_snapshot(base,registry=prepared['prepared']['registry'],catalog=catalog if slot['condition']=='D4' else None)
    if digest(frozen_record(capture))!=slot['fingerprint']:raise ValueError('PROPOSED_SNAPSHOT_CHANGED')
    return capture


def execute(authorization_path):
    p=plan();authorization=read(authorization_path);verify_authorization(authorization,p)
    body=deepcopy(p);fingerprint=body.pop('fingerprint')
    if digest(body)!=fingerprint:raise ValueError('PROTOCOL_CHANGED')
    for group in ('input_hashes','source_and_old_code_pins','new_code_pins'):
        for name,expected in p[group].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(path)!=expected:raise ValueError('PIN_CHANGED')
    tests=read(PUB/'test_results.json')
    if tests['failures'] or tests['errors']:raise ValueError('REGRESSION_FAILED')
    protected=read(PR/'preservation_before.json')
    if check(protected)['status']!='PASS':raise ValueError('PROTECTED_FILE_CHANGED')
    run=PR/RUN;run.mkdir(parents=True,exist_ok=False)
    write(run/'operator_authorization.json',authorization)
    loaded=metadata('ps');tags=[m for m in metadata('tags').get('models',[]) if m.get('name')==MODEL3]
    if any(m.get('name')!=MODEL3 for m in loaded.get('models',[])) or len(tags)!=1:
        raise ValueError('SERVER_OR_OTHER_MODEL_BLOCKED')
    if 'sha256:'+tags[0]['digest'].removeprefix('sha256:')!=DIGEST3:raise ValueError('MODEL_DIGEST_MISMATCH')
    write(run/'preflight.json',{'server':metadata('version'),'loaded':loaded,'model':tags,
        'planned_context':32768,'observed_context':'CHECK_AFTER_FIRST_FIXED_TASK_AND_EVERY_REQUEST'})
    config=read(PR/'llm_config.json');prepared=validate_setup(config);catalog=read(CATALOG)
    slots=read(PR/'draft_snapshot_requests_NOT_AUTHORIZED.json');results=[];start=time.monotonic();stop=None
    class GlobalStop(ValueError):pass
    raw_hashes=read(PR/'raw_file_hashes.json')
    # Raw bytes verified immediately before execution, plus metadata before
    # each call. A detected mutation stops all; nothing is silently re-pinned.
    if any(sha(Path(k))!=v for k,v in raw_hashes.items()):raise ValueError('RAW_SOURCE_CHANGED')
    raw_stat={k:(Path(k).stat().st_size,Path(k).stat().st_mtime_ns) for k in raw_hashes}
    def guard():
        if time.monotonic()-start>p['wall_seconds']:raise GlobalStop('GLOBAL_WALL_LIMIT')
        if _process_private_bytes()>p['process_bytes']:raise GlobalStop('GLOBAL_MEMORY_LIMIT')
        if sum(f.stat().st_size for root in (PUB,PR) for f in root.rglob('*') if f.is_file())>p['output_bytes']:raise GlobalStop('GLOBAL_OUTPUT_LIMIT')
        if any((Path(k).stat().st_size,Path(k).stat().st_mtime_ns)!=v for k,v in raw_stat.items()):raise GlobalStop('RAW_SOURCE_METADATA_CHANGED')
        for group in ('input_hashes','source_and_old_code_pins','new_code_pins'):
            for name,expected in p[group].items():
                path=Path(name) if Path(name).is_absolute() else ROOT/name
                if sha(path)!=expected:raise GlobalStop('PIN_CHANGED')
    budget=GenerationBudget(40,1,guard);token=_BUDGET.set(budget)
    try:
        for slot in slots:
            ordinal,condition=slot['patient_ordinal'],slot['condition']
            row={'patient_ordinal':ordinal,'condition':condition,'status':'not_attempted','generation':0,
                'supported':None,'source_exact':None,'normalized_match':None,'source_department':None,
                'valid_citation':None,'additional_evidence_cited':None,'additional_records':slot['additional_records'],
                'prompt_tokens':None,'wall_seconds':None,'error_type':None,'concept_match':'NOT_EVALUATED'}
            if stop:row['reason']='GLOBAL_STOP';results.append(row);continue
            client=None;begin=time.monotonic();budget.episode_used=0
            try:
                guard();capture=build_draft(slot,prepared,catalog)
                client=BudgetedNativeERClient(config,authorized=True,contract_fingerprint=p['fingerprint'],
                    checkpoint_root=run/('Patient%03d_%s'%(ordinal,condition)))
                content=client.chat_json(capture['system'],json.dumps(capture['payload'],sort_keys=True,ensure_ascii=False),
                    model=capture['model'],schema=capture['schema'])
                validated=capture['validate'](content)
                supported=validated['certainty']=='supported';target=validated['target'] if supported else None
                d=packet(ordinal);match=matches(target,d['Source_reference_diagnoses'])
                base_count=len(d['terminal_observation']['objective_findings'])
                # Original Core puts objective findings first; histories follow.
                extras=set(range(base_count,base_count+slot['additional_records']))
                cited=set(validated.get('evidence_indices',[]))
                row.update(status='completed',supported=supported,source_exact=match['source_exact'] if supported else False,
                    normalized_match=match['normalized_match'] if supported else False,
                    source_department=validated['department'] in d['Source_reference_departments'],valid_citation=True,
                    additional_evidence_cited=bool(cited&extras))
                write(client.checkpoint_root/'private_scoring.json',{'original_output':json.loads(content),'validated':validated,
                    'reference':d['Source_reference_diagnoses'],'source_department_reference':d['Source_reference_departments'],
                    'additional_indices':sorted(extras),'clinical_correctness':None})
                guard()
            except Exception as exc:
                row.update(status='failed',error_type=type(exc).__name__)
                if isinstance(exc,(GlobalStop,ModelIdentityError)) or 'SNAPSHOT_CHANGED' in str(exc):stop=type(exc).__name__
                if client:write(client.checkpoint_root/'failure.json',{'error_type':type(exc).__name__,'private_detail':str(exc)})
            finally:
                row['generation']=client.request_count if client else 0;row['wall_seconds']=time.monotonic()-begin
                if client:
                    row['identity_checks']=client.identity_check_count
                    row['prompt_tokens']=(client.last_server_usage or {}).get('prompt_tokens');client.close()
                results.append(row);write(run/('Result%03d_%s.json'%(ordinal,condition)),row)
                print(json.dumps({'ordinal':ordinal,'condition':condition,'status':row['status'],'generation':row['generation']}),flush=True)
    finally:
        _BUDGET.reset(token);write(run/'results.json',results)
        write(PUB/'execution_summary.json',{'generation':budget.used,'wall_seconds':time.monotonic()-start,
            'global_stop':stop,'retry':0,'repair':0,'strict_v1_grants':0,'new_clinical_approvals':0})
        write(PUB/'preservation_after_execution.json',check(protected))


def reports(preapproval=False):
    p=plan();s=read(PUB/'additional_test_type_summary.json')
    table='|code|Source description|환자 수|기록 수|원본 단위|제안 입력 행|\n|---|---|---:|---:|---|---:|\n'
    for t in s['types']:table+='|%s|%s|%s|%s|%s|%s|\n'%(t['code'],t['description'],t['patients'],t['records'],', '.join(t['units']),t['proposed_records'])
    report_file('SYNTHEA_ADDITIONAL_TEST_TYPE_SUMMARY.md','# 추가 검사 유형 전수 집계\n\n'+table+
        '\n358행/50종은 기존 활력4종의 반복이 아니다. 신체계측, 혈당/HbA1c, 신장, 전해질, 지질, 혈구, 소변, 간기능 등이다. '
        '통증20행은 이미 제외하여 Ask 전용으로 유지했다. 환자별 code 수와 행 수는 additional_test_type_summary.json에 있고 '
        '실제 값·단위·기록 시각·Source 위치는 Private observation_inventory와 proposed_source_rows_NOT_AUTHORIZED.json에만 보존했다. '
        '동일 키 상이값3그룹6행 전체 제외, 완전동일7행 입력 중복 제거, 제안345행이다. 기존358행을 삭제하거나 정답에 따라 고르지 않았다.\n')
    report_file('SYNTHEA_DEVELOPMENT_PROXY_CONTRACT.md','# 합성 개발 Proxy 별도 계약 제안\n\n'
        f'정책 `{POLICY}`, 계획 `{p["fingerprint"]}`. 기존 source_check/v1 및 그 임상·공개 grant 계약은 변경하지 않는다. '
        'v1은358행 모두 RELEASE_NOT_REVIEWED다. 제안은 v1 통과나 임상 검토가 아닌, 사용자가 구체적으로 수용해야 하는 별도 합성 정적 Snapshot 정책이다. '
        '현재 승인 없음/새 grant0/실제 추가 Source View 실행0.\n\n'
        '공통 규칙: Source CSV358행 전체 동등성, 동일환자/방문, numeric laboratory/vital storage, 통증 제외, 원문 code/description/value/unit/date 일치, '
        '실제 timestamp에서 재계산한 second upper bound 검증, proxy minute1≤기존cutoff45, 공통15 진단명의 직접문구 차단, '
        '동일 code/description/time/unit의 상이값 전부 제외 및 완전중복 제거. 미래·불명확 timestamp·값 불일치는 제외한다. '
        '환자별 Gold/Condition/Department로 선택하지 않고 검토용 Reference는 입력에서 분리한다.\n\n'
        '실제 clinical available_at은358행 모두 미검증이다. 같은날18명은 진단과 검사의 intraday 순서를 확인할 수 없다. '
        '12/16은 Condition 날짜가 하루 뒤이므로 Reference는 회고적 Source label 해석에 한정한다. '
        '관측값이 질환을 시사하는 것과 정답 문자열 누출을 구분하되, 임상적 의미 누출 없음이나 prospective diagnosis 가능성을 자동 승인하지 않는다. '
        '이 한계를 명시적으로 수용한 합성 개발 비교에만 제한하고, 임상 안전성·공식B0~B4·학습/VoI likelihood 승인으로 확장하지 않는다.\n')
    report_file('SYNTHEA_SOURCE_ELIGIBILITY_VALIDATION.md','# Source 적격성 검증\n\n'+
        json.dumps({k:s[k] for k in ('candidate_rows','structurally_qualified','proposed_rows','checks','strict_v1_states','duplicate_status_counts','reference_order_counts')},ensure_ascii=False,indent=2)+
        '\n\n위 PASS는 구조적 사실 확인만 의미한다. 임상적 공개시각·의미·Reference 적합성은 NOT_REVIEWED다. '
        'D1/D3 기존40개 고정계약은 로컬 재구성 Hash가20명 모두 일치했다. D2/D4는 같은 추가Evidence·같은cutoff이며 '
        '원래 Prompt/관찰 표현, Catalog, 기존 의미검증기와 scorer를 유지한다. measurement_view는 적용하지 않는다. '
        '입력 context guard40건 모두 통과. baseline_and_paired_input_verification.json 및 test_results.json 참조.\n')
    report_file('SYNTHEA_RELEASE_APPROVAL_REQUEST.md','# 실제 승인 요청서 — 아직 승인되지 않음\n\n'
        f'계획 `{p["fingerprint"]}`. 정확한50종 목록은 SYNTHEA_ADDITIONAL_TEST_TYPE_SUMMARY.md와 고정 JSON에 있다. '
        '358원천행 중 동일키 상이값6행 제외·완전중복7행 제거 후345행, 원래20명 전원 유지. '
        'Recorded-Time upper-bound Proxy를 합성 정적 개발 입력의 공개 가정으로 수용할지 실제 운영자/연구 책임자의 판단이 필요하다. '
        '18명의 진단/검사 intraday 순서 미확정 및2명의 다음날 Source Reference 한계를 그대로 수용해야 한다. '
        '임상 reviewer·v1 grant·Clinical Gold 승인을 만드는 요청이 아니다. 기존source_check는 계속미통과이며 새정책의 명시적 승인이 없으면 실행하지 않는다.\n\n'
        '승인 요청 범위: D2→D4를 환자별1..20, 각1회/총40회 이하, 기존3B/Digest·Context32768·temperature0·seed42·output1024·timeout120초·Retry/Repair/Warmup0, '
        'Memory/Routing/VoI OFF, 기존관찰표현, 전역wall30분·process4GiB·output1GiB. 새Namespace/Run, no-resume. '
        'Source/Identity/전역상한 위반 전체중단, 단일 bounded 호출/쓰기 중 일시초과 가능. '
        '다른 Ollama 작업이 없을 때만 실행하며, 첫 고정Task 로딩 직후 actualContext/Digest를 검증하고 후속 호출을 허용한다. '
        '서버변경·모델다운로드·기존D0/D1/D3 재호출·480회B0/B2·fitting은 포함하지 않는다.\n')
    if preapproval:
        print(json.dumps({'status':'REVIEWABLE_SOURCE_CONTRACT_AND_APPROVAL_REQUEST_WRITTEN','fingerprint':p['fingerprint']}))
        return
    if (PR/RUN/'results.json').exists():results=read(PR/RUN/'results.json')
    else:results=[{'patient_ordinal':i,'condition':c,'status':'not_attempted','generation':0,
        'source_exact':None,'normalized_match':None,'reason':'NEW_DEVELOPMENT_POLICY_AND_PROTOCOL_APPROVAL_PENDING'}
        for i in range(1,21) for c in ('D2','D4')]
    summaries=[]
    for c in ('D2','D4'):
        rr=[r for r in results if r['condition']==c];cc=[r for r in rr if r['status']=='completed']
        summaries.append({'condition':c,'registered':20,'completed':len(cc),'failed':sum(r['status']=='failed' for r in rr),
            'not_attempted':sum(r['status']=='not_attempted' for r in rr),'supported':sum(r.get('supported') is True for r in cc),
            'unresolved':sum(r.get('supported') is False for r in cc),'source_exact':sum(r.get('source_exact') is True for r in cc) if cc else None,
            'normalized':sum(r.get('normalized_match') is True for r in cc) if cc else None,
            'source_department':sum(r.get('source_department') is True for r in cc) if cc else None,
            'additional_evidence_cited':sum(r.get('additional_evidence_cited') is True for r in cc) if cc else None,
            'valid_citation':sum(r.get('valid_citation') is True for r in cc) if cc else None,
            'generation':sum(r['generation'] for r in rr),'prompt_tokens':sum(r.get('prompt_tokens') or 0 for r in rr),
            'wall_seconds':sum(r.get('wall_seconds') or 0 for r in rr),'concept_match':'NOT_EVALUATED'})
    old=read(BC/'synthea_snapshot_001/results.json');pairs=[]
    for before,after in (('D1','D2'),('D3','D4'),('D2','D4')):
        for i in range(1,21):
            a=next(r for r in (old if before in ('D1','D3') else results) if r['patient_ordinal']==i and r['condition']==before)
            b=next(r for r in results if r['patient_ordinal']==i and r['condition']==after)
            complete=a['status']==b['status']=='completed'
            pairs.append({'patient_ordinal':i,'comparison':after+'-'+before,'registered_denominator':20,'pair_completed':complete,
                'before_status':a['status'],'after_status':b['status'],'source_exact_before':a.get('source_exact'),
                'source_exact_after':b.get('source_exact'),'exact_transition':
                    ('%s_to_%s'%(bool(a['source_exact']),bool(b['source_exact']))) if complete else 'NOT_EVALUATED',
                'source_exact_delta':int(b['source_exact'])-int(a['source_exact']) if complete else None,
                'normalized_delta':int(b['normalized_match'])-int(a['normalized_match']) if complete else None})
    csv_file(PUB/'SYNTHEA_D2_D4_PAIRED_RESULTS.csv',pairs,list(pairs[0]))
    write(PUB/'d2_d4_results.json',{'conditions':summaries,'rows':results,'paired_comparisons':[
        {'comparison':name,'registered_denominator':20,'complete_pairs':sum(r['pair_completed'] for r in pairs if r['comparison']==name),
         'source_exact_net_count':sum(r['source_exact_delta'] for r in pairs if r['comparison']==name and r['pair_completed']) if any(r['pair_completed'] for r in pairs if r['comparison']==name) else None}
        for name in ('D2-D1','D4-D3','D4-D2')]})
    report_file('SYNTHEA_D2_D4_RESULTS.md','# D2/D4 결과\n\n'+json.dumps(summaries,ensure_ascii=False,indent=2)+
        '\n\n실행 승인/공개 계약 미확정 시 NOT_RUN이며0% 성능으로 해석하지 않는다. 등록20명과 완료쌍을 구분한다. '
        'Source Exact/Normalized/진료과 일치는 합성 Source reference agreement이며 임상정확도가 아니다. '
        'Verified Concept Match는 NOT_EVALUATED. 지원되는 응답도 임상 Correctness 승인을 의미하지 않는다. '
        '기존 D1/D3는 재실행하거나 변경하지 않았다. 공개 paired CSV에는 비식별 ordinal과 지표만 포함했다.\n')
    ran=sum(r['generation'] for r in results)
    report_file('SYNTHEA_FINAL_RUN_DECISION.md','# 최종 실행 판단\n\n'
        '1. 실제 추가 Evidence는50종358행이며 기존 활력4종의 반복이 아니다. 공통 무결성/중복 규칙 적용 후 제안345행이다.\n'
        '2. 구조적 Source 검증은 가능하지만 실제 임상공개/진단전 관측임을 증명하지 못한다. 별도 합성 Recorded-Time Proxy 정책과 신규Protocol 승인이 있어야 해당 제한적 개발 평가에 사용할 수 있다. 원래v1은계속차단하고 승인전grant0이다.\n'
        f'3. 이번 신규 Generation은{ran}회다. D2/D4가 미실행이면 확대Evidence 효과는미평가이며, 최종480회B0/B2 실행가치를 판단할 근거가 추가되지 않았다. '
        '실행된 경우에도 효과 크기와 오류·Reference 한계를 검토한 뒤 별도 B0/B2 계획/승인이 필요하다. 자동재평가하지 않는다.\n\n'
        '실행 명령(실제 승인 JSON이 있을 때만): `python -B scripts/complete_synthea_evidence_recovery.py execute --execute --allow-llm --authorization <실제승인파일>`; '
        '현재 승인 상태는 authorization_status.json, 보호검증은 preservation_verification.json. 기존Core/공식/데이터/완료결과/Prompt/Catalog/scorer는 변경하지 않았다.\n')
    write(PUB/'preservation_verification.json',check(read(PR/'preservation_before.json')))
    write(PUB/'delivery_manifest.json',{'status':'D2_D4_RECORDED' if ran else 'D2_D4_NOT_RUN_APPROVAL_PENDING',
        'generation':ran,'new_clinical_reviews':0,'strict_v1_grants':0,'protocol_fingerprint':p['fingerprint'],
        'tests':read(PUB/'test_results.json'),'reports':{f.name:sha(f) for f in DOC.glob('*.md')},
        'preservation':check(read(PR/'preservation_before.json'))})
    print(json.dumps({'generation':ran,'results':summaries}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','review','execute','reports'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true')
    parser.add_argument('--authorization')
    a=parser.parse_args()
    if a.mode=='execute':
        if not (a.execute and a.allow_llm and a.authorization):parser.error('EXPLICIT_EXECUTE_ALLOW_LLM_AND_AUTHORIZATION_REQUIRED')
        execute(Path(a.authorization))
    elif a.mode=='prepare':prepare()
    elif a.mode=='review':reports(preapproval=True)
    else:reports()
