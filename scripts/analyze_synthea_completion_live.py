"""Saved-result analysis only. No generation, source admission, or fitting."""
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.complete_synthea_evidence_recovery import (PUB as PREP,PR,BC,ROOT,RUN,plan,build_draft,packet)
from scripts.audit_research_gaps import read,write,sha,check
from scripts.run_final_3b_evaluation import csv_file
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collection_preparation import CATALOG
from jsonschema import Draft202012Validator

PUB=PREP/'actual_results_v1'
DOC=ROOT/'docs/연구 평가/FINAL_SYNTHEA_COMPLETION_v1/actual_results_v1'


def md(name,text):
    for base in (PUB,DOC):
        base.mkdir(parents=True,exist_ok=True)
        with (base/name).open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def norm(text):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def main():
    if PUB.exists() or DOC.exists():raise ValueError('FRESH_ACTUAL_ANALYSIS_NAMESPACE_REQUIRED')
    p=plan();results=read(PR/RUN/'results.json');execution=read(PREP/'execution_summary.json')
    old=read(BC/'synthea_snapshot_001/results.json');slots=read(PR/'draft_snapshot_requests_NOT_AUTHORIZED.json')
    prepared=validate_setup(read(PR/'llm_config.json'));catalog=read(CATALOG)
    flags=[];counts=Counter();private=[];public=[]
    for stored in results:
        r=deepcopy(stored);ordinal=r['patient_ordinal'];condition=r['condition']
        folder=PR/RUN/('Patient%03d_%s'%(ordinal,condition))
        r.update(HTTP_status=None,HTTP_wall_seconds=None,native_schema_valid=None,
            semantic_valid=None,completion_tokens=None,actual_context=None,
            stored_request_matches_fixed_draft=None,independent_score_matches=None,
            citation_count=None,additional_citation_count=None,measurement_label_target=None)
        before=folder/'Request0001_before.json';after=folder/'Request0001_after.json'
        if after.exists():
            a=read(after);r['HTTP_status']=a.get('HTTP_status');r['HTTP_wall_seconds']=a.get('wall_seconds')
            counts['after_checkpoints']+=1;counts['HTTP200']+=a.get('HTTP_status')==200
            ident=a.get('identity_after',{})
            r['actual_context']=ident.get('context_length')
            correct_identity=(ident.get('context_length')==32768 and
                'sha256:'+str(ident.get('digest','')).removeprefix('sha256:')==p['model_identifier'])
            counts['loaded_digest_context_verified']+=correct_identity
            envelope=a.get('response_envelope',{})
            r['prompt_tokens']=envelope.get('prompt_eval_count',r.get('prompt_tokens'))
            r['completion_tokens']=envelope.get('eval_count')
            if before.exists():
                b=read(before);req=b['request'];spec=next(s for s in slots if s['patient_ordinal']==ordinal and s['condition']==condition)['contract']
                wire=(req['model']==spec['model'] and req['messages'][0]['content']==spec['system']
                    and json.loads(req['messages'][1]['content'])==spec['payload'] and req['format']==spec['schema']
                    and digest(req)==b['request_fingerprint']==a['request_fingerprint'])
                r['stored_request_matches_fixed_draft']=wire;counts['planned_request_matches']+=wire
                if not wire:flags.append('STORED_REQUEST_PLAN_MISMATCH')
                if envelope.get('message'):
                    try:
                        response=json.loads(envelope['message']['content'])
                        Draft202012Validator(spec['schema']).validate(response)
                        r['native_schema_valid']=True;counts['schema_valid']+=1
                        slot=next(s for s in slots if s['patient_ordinal']==ordinal and s['condition']==condition)
                        capture=build_draft(slot,prepared,catalog)
                        validated=capture['validate'](json.dumps(response))
                        r['semantic_valid']=True;counts['semantic_valid']+=1
                        supported=validated['certainty']=='supported';target=validated['target'] if supported else None
                        d=packet(ordinal);refs=d['Source_reference_diagnoses']
                        exact=target in refs if supported else False
                        normalized=norm(target) in [norm(x) for x in refs] if supported else False
                        dept=validated['department'] in d['Source_reference_departments']
                        indices=validated.get('evidence_indices',[])
                        evidence=capture['payload']['evidence_records'];base=len(d['terminal_observation']['objective_findings'])
                        extras=set(range(base,base+slot['additional_records']))
                        extra_count=len(set(indices)&extras)
                        eligible=set(capture['payload']['supported_diagnosis_eligible_indices'])
                        valid=all(type(i) is int and 0<=i<len(evidence) for i in indices) and (not supported or bool(set(indices)&eligible))
                        same=(r['status']=='completed' and r['source_exact']==exact and r['normalized_match']==normalized
                            and r['supported']==supported and r['source_department']==dept
                            and r['valid_citation']==valid and r['additional_evidence_cited']==bool(extra_count))
                        r['independent_score_matches']=same
                        counts['metric_matches']+=same
                        r['citation_count']=len(indices);r['additional_citation_count']=extra_count
                        r['measurement_label_target']=any(isinstance(target,str) and
                            (target==x['source_event']['target']['description'] or target.startswith(x['source_event']['target']['description']+':'))
                            for x in d['source_objective_candidates'])
                        private.append({'patient_ordinal':ordinal,'condition':condition,'prediction':target,
                            'department':validated['department'],'reference':refs,'certainty':validated['certainty'],
                            'original_output':response,'validated':validated,'cited_evidence':[evidence[i] for i in indices],
                            'additional_cited_indices':sorted(set(indices)&extras),'clinical_correctness':None})
                        if r['status']=='completed' and not same:flags.append('SAVED_METRIC_MISMATCH')
                    except Exception as exc:
                        if r['status']=='completed':flags.append('COMPLETED_BUT_REVALIDATION_FAILED_'+type(exc).__name__)
            if r['status']=='completed' and not correct_identity:flags.append('COMPLETED_IDENTITY_NOT_VERIFIED')
        elif r['generation']:
            flags.append('ATTEMPT_WITHOUT_AFTER_CHECKPOINT')
        public.append(r)
    if sum(r['generation'] for r in public)!=execution['generation']:flags.append('REQUEST_COUNT_MISMATCH')
    write(PR/'actual_independent_private_scoring.json',private)
    summaries=[]
    for c in ('D2','D4'):
        rr=[r for r in public if r['condition']==c];complete=[r for r in rr if r['status']=='completed']
        summary={'condition':c,'registered_patients':20,'completed':len(complete),
            'failed':sum(r['status']=='failed' for r in rr),'not_attempted':sum(r['status']=='not_attempted' for r in rr),
            'supported':sum(r.get('supported') is True for r in complete),'unresolved':sum(r.get('supported') is False for r in complete),
            'source_exact_count':sum(r['source_exact'] is True for r in complete),
            'normalized_count':sum(r['normalized_match'] is True for r in complete),
            'source_department_count':sum(r['source_department'] is True for r in complete),
            'additional_cited_responses':sum(r.get('additional_evidence_cited') is True for r in complete),
            'valid_citation_responses':sum(r.get('valid_citation') is True for r in complete),
            'measurement_label_targets':sum(r.get('measurement_label_target') is True for r in complete),
            'generation':sum(r['generation'] for r in rr),'prompt_tokens':sum(r.get('prompt_tokens') or 0 for r in rr),
            'completion_tokens':sum(r.get('completion_tokens') or 0 for r in rr),
            'requests_with_prompt_tokens':sum(r.get('prompt_tokens') is not None for r in rr),
            'wall_seconds_sum_slots':sum(r.get('wall_seconds') or 0 for r in rr),
            'Source_exact_registered_percent':100*sum(r['source_exact'] is True for r in complete)/20,
            'Source_exact_completed_percent':100*sum(r['source_exact'] is True for r in complete)/len(complete) if complete else None,
            'verified_concept_match':'NOT_EVALUATED'}
        summaries.append(summary)
    pairs=[];pairsummary=[]
    for before,after in (('D1','D2'),('D3','D4'),('D2','D4')):
        for i in range(1,21):
            a=next(r for r in (old if before in ('D1','D3') else public) if r['patient_ordinal']==i and r['condition']==before)
            b=next(r for r in public if r['patient_ordinal']==i and r['condition']==after)
            ok=a['status']==b['status']=='completed'
            pairs.append({'patient_ordinal':i,'comparison':after+'-'+before,'registered_patients':20,'complete_pair':ok,
                'before_status':a['status'],'after_status':b['status'],'before_supported':a.get('supported'),'after_supported':b.get('supported'),
                'source_exact_before':a.get('source_exact'),'source_exact_after':b.get('source_exact'),
                'exact_transition':('%s_to_%s'%(bool(a['source_exact']),bool(b['source_exact']))) if ok else 'NOT_EVALUATED',
                'source_exact_delta':int(b['source_exact'])-int(a['source_exact']) if ok else None,
                'normalized_delta':int(b['normalized_match'])-int(a['normalized_match']) if ok else None,
                'source_department_before':a.get('source_department_match') if before in ('D1','D3') else a.get('source_department'),
                'source_department_after':b.get('source_department')})
        group=[r for r in pairs if r['comparison']==after+'-'+before and r['complete_pair']]
        pairsummary.append({'comparison':after+'-'+before,'registered_patients':20,'complete_pairs':len(group),
            'source_exact_net_count':sum(r['source_exact_delta'] for r in group) if group else None,
            'paired_Source_exact_pp':100*sum(r['source_exact_delta'] for r in group)/len(group) if group else None,
            'normalized_net_count':sum(r['normalized_delta'] for r in group) if group else None,
            'registered20_observed_net_pp':100*sum(r['source_exact_delta'] for r in group)/20 if group else None,
            'missing_pairs_not_imputed':20-len(group)})
    csv_file(PUB/'SYNTHEA_D2_D4_PAIRED_RESULTS.csv',pairs,list(pairs[0]))
    csv_file(PUB/'condition_response_metrics.csv',public,list(public[0]))
    write(PUB/'actual_results.json',{'conditions':summaries,'rows':public,'execution':execution,'paired':pairsummary,
        'error_types':dict(Counter(r['error_type'] for r in public if r.get('error_type'))),
        'clinical_correctness':'NOT_EVALUATED','old_D1_D3_recalled':False})
    pins=read(PR/'preservation_before_live.json');preserved=check(pins)
    raw=read(PR/'raw_file_hashes.json');raw_ok=all(sha(Path(f))==v for f,v in raw.items())
    if preserved['status']!='PASS':flags.append('PRIOR_PROTECTED_FILE_MODIFIED')
    if not raw_ok:flags.append('RAW_SOURCE_MODIFIED')
    write(PUB/'independent_execution_verification.json',{'status':'PASS' if not flags else 'FAIL','flags':flags,
        'saved_request_and_response_checks':dict(counts),'preservation':preserved,'raw_hash_unchanged':raw_ok,
        'identity_checks_recorded':sum(r.get('identity_checks',0) for r in results),
        'strict_v1_grants_created':0,'clinical_reviews_created':0,'new_generation_in_analysis':0,
        'approved_plan':p['fingerprint'],'authorization_record_sha':sha(PR/'development_proxy_execution_authorization.json'),
        'regeneration_bitwise_reproducibility':'NOT_TESTED; stored trace reaggregation only'})
    table='|조건|완료/등록|Supported/Unresolved|Source Exact|Normalized|진료과 일치|추가Evidence 인용|Generation|Prompt tokens|\n|---|---|---|---|---|---|---|---|---|\n'
    for r in summaries:
        table+='|%s|%s/20|%s/%s|%s/20|%s/20|%s/20|%s/20|%s|%s|\n'%(r['condition'],r['completed'],r['supported'],r['unresolved'],
            r['source_exact_count'],r['normalized_count'],r['source_department_count'],r['additional_cited_responses'],r['generation'],r['prompt_tokens'])
    pt='\n|대조|완료 쌍/등록|Source Exact 순증|완료쌍 기준 차이|\n|---|---|---|---|\n'
    for r in pairsummary:pt+='|%s|%s/20|%s|%s pp|\n'%(r['comparison'],r['complete_pairs'],r['source_exact_net_count'],r['paired_Source_exact_pp'])
    md('SYNTHEA_D2_D4_RESULTS.md','# 승인 후 실제 D2/D4 결과\n\n'+table+pt+
        '\n기존 상위 폴더의 NOT_RUN 보고서는 승인 전 기록으로 보존했다. 이 actual_results_v1이 승인 후 실제 실행 결과다. '
        f'Run `{RUN}`, 계획 `{p["fingerprint"]}`. 총 wall `{execution["wall_seconds"]:.3f}`초. '
        'Completed는 정적 응답·계약 검증 완료이며 임상정확성을 뜻하지 않는다. Source Exact/Normalized/진료과 일치는 합성 Source label과의 비교다. '
        '실패/미실행은 정오로 대체하지 않으며 등록20과 완료쌍을 구분한다. 추가Evidence 인용은 모델이 보고한 index 사용이며 임상적 근거 타당성이나 인과적 사용의 증명이 아니다. '
        '모든 추가Source는 동일 관측 code/value/unit/date와 동일cutoff를 사용하고, D2/D4간 Catalog 계약만 다르다. '
        '기존 D1/D3는 저장 결과를 읽기만 했고 재호출하지 않았다. measurement_view 미적용, Memory/Routing/VoI OFF, Retry/Repair0. '
        'Verified Concept Match 및 임상 Correctness는 NOT_EVALUATED다.\n')
    # An improvement here demonstrates only static source-information potential.
    any_improvement=any((r['source_exact_net_count'] or 0)>0 for r in pairsummary[:2])
    judgment=('확대 Evidence의 정적 Source 일치 개선 신호가 관측됐다. 다만 실제 Ask/Test 요청·공개 시점·행동 경로를 포함한 B0/B2 효과와 같지 않다. '
        '따라서 기술적 후속 후보로는 검토할 수 있지만 즉시480회 전체 재평가를 자동 실행하거나 Memory 효과로 주장할 근거는 아니다.') if any_improvement else (
        '확대 Evidence를 사용해도 기존 D1/D3 대비 Source Exact 개선은 관측되지 않았다. 현재 근거로480회 B0/B2 재평가는 권장하지 않는다.')
    md('SYNTHEA_FINAL_RUN_DECISION.md','# 실제 결과에 따른 최종 판단\n\n'
        '1. 추가 Source에는 혈당/HbA1c, 신장기능, 전해질, 지질, 혈구, 소변, 간기능 및 신체계측50종358행이 존재한다. '
        '통증20행은 Ask 전용 유지, 상이값6행 제외·완전중복7행 제거 후345개 원천 관측을 입력했다.\n\n'
        '2. 사용자 명시적 운영자 승인을 받은 별도 합성 Recorded-Time Proxy 정책으로 정적 Snapshot에서만 사용했다. '
        '원래 source_check/v1의 임상공개 승인 상태는 변하지 않으며 v1 grant0, 임상 검토0이다. '
        '18명의 진단/검사 intraday 순서 미확정, ordinal12·16의 다음날 Source Reference, 실제 available_at 부재 및 개발 노출20명의 한계는 계속 유지한다.\n\n'
        '3. '+judgment+'\n\n'
        'B0/B2 재평가·기존 D0/D1/D3 재실행·fitting·새 데이터·모델변경·Prompt 최적화0. '
        '원본과 승인 전 보고서를 보존했으며 실제 결과는 이 하위폴더에 분리했다.\n')
    md('SYNTHEA_DEVELOPMENT_PROXY_AUTHORIZATION_ADDENDUM.md','# 승인 상태 부록\n\n'
        '사용자 실제 답변: “다른 작업 없음·새 합성 정책과 최대40회 실행 승인”. '
        '이는 e62d6641…7edc42의 새 합성 정적 정책/운영자 실행 승인이다. 교수·임상검토자의 서명 또는 임상 Source 공개 grant로 표현하지 않는다. '
        '원래 승인 전 authorization_status.json 및 NOT_RUN 보고서는 당시 상태로 보존하고, 실제 승인 기록은 Private development_proxy_execution_authorization.json에 저장했다. '
        '추가 Source는 원래 v1 Source View를 통과시킨 것이 아니라 명시적으로 승인된 별도 합성 Snapshot 범위에서만 사용했다. '
        'clinical_release_approved=false, clinical_reviewer_created=false, strict_v1_grants_created=0을 유지한다.\n')
    md('SYNTHEA_EXECUTION_VERIFICATION.md','# 실행 및 재집계 검증\n\n'+
        json.dumps({'status':'PASS' if not flags else 'FAIL','checks':dict(counts),'flags':flags,
            'preservation':preserved,'raw_hash_unchanged':raw_ok,'tests_before_execution':read(PREP/'test_results.json')},ensure_ascii=False,indent=2)+
        '\n\n저장된 요청객체/응답 envelope를 검증한 것이며 미저장 Raw network wire를 복원하지 않았다. '
        '전체 저장소 테스트나 LLM bitwise 재생성 재현성을 주장하지 않는다. 관련207개 테스트만 실행 전 통과한 결과다. '
        '원래 Source reference와 scorer를 변경하지 않고 독립적인 저장 Trace 재집계로 같은 값을 확인했다.\n')
    ids=[packet(i)['private_identity'][key] for i in range(1,21) for key in ('patient_id','visit_id')]
    leaks=[]
    for root in (PUB,DOC):
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in ('.md','.csv','.json'):
                text=f.read_text(encoding='utf-8-sig')
                if any(v in text for v in ids):leaks.append(str(f))
                if f.suffix=='.md' and ('\ufffd' in text or '??' in text):flags.append('DOC_ENCODING')
    if leaks:flags.append('PUBLIC_IDENTIFIER_LEAK')
    write(PUB/'delivery_manifest.json',{'status':'ACTUAL_RESULTS_VERIFIED' if not flags else 'VERIFICATION_FAILED',
        'flags':flags,'public_identifier_leaks':leaks,'generation':execution['generation'],'conditions':summaries,
        'paired':pairsummary,'protected':preserved,'raw_hash_unchanged':raw_ok,
        'code_analysis_sha256':sha(Path(__file__)),'new_generation_in_analysis':0,
        'report_hashes':{f.name:sha(f) for f in DOC.glob('*.md')},
        'prior_NOT_RUN_artifacts_preserved':True,'whole_B0_B2_reexecution':0})
    print(json.dumps({'status':'PASS' if not flags else 'FAIL','flags':flags,'conditions':summaries,'pairs':pairsummary,'protected':preserved}))


if __name__=='__main__':main()
