"""Fixed Synthea pilot recovery: offline first; explicit new authority only.

Reuses the existing collector/backend/Source policy. Does not change a model,
clinical payload, Prompt, Reference, scorer or algorithm. Python 3.9.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, write, sha, check
import scripts.prepare_synthea_sequential_pilot as previous
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import build_episode, EXPORTER_VERSION
from app.locdi.current_ablation_runner import features_for, _validate_result
from app.locdi.current_ablation_llm_backend import make_er_backend, input_guard
from app.locdi.memory_paired_recovery import aggregate, citation_audit, verify_delivered_memory
from app.locdi.synthea_diagnosis_audit import matches
from app.locdi.synthea_sequential_proxy import SequentialSyntheticSourceView, policy_contract
from app.locdi.routing_collection_preparation import CATALOG
from app.locdi.routing_data_reuse import EXPERIMENTS
from app.locdi.llm_setup import digest

NS = 'FINAL_LOCDI_PILOT_BASED_ACCURACY_IMPROVEMENT_v2'
PUB = ROOT / 'outputs/research_evaluation' / NS
PR = ROOT / 'outputs/private/research_evaluation' / NS
DOC = ROOT / 'docs/연구 평가' / NS
OLDPUB, OLDPR = previous.PUB, previous.PR
OLD_RUN = previous.RUN
RUN = 'synthea_sequential_002'
DATA = previous.DATA
MODULES = ['test_current_ablation_interface', 'test_current_ablation_llm_backend',
    'test_current_ablation_runner', 'test_information_leakage', 'test_memory_paired_recovery',
    'test_routing_voi_readiness', 'test_sequential_export_contract', 'test_simulation',
    'test_synthea_development_proxy', 'test_synthea_diagnosis_audit',
    'test_synthea_diagnostic_snapshot', 'test_synthea_sequential_proxy', 'test_voi_stop_consistency']


def md(name, text):
    for base in (PUB, DOC):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            f.write(text.rstrip() + '\n')


def saved_backend_replay(case, config, folder):
    """Real backend postprocessing + scorer, stored raw only, no transport."""
    raw = read(folder / 'raw_runner_result.json')
    decisions = read(folder / 'validated_ER_decisions.json')
    class StoredClient:
        def __init__(self, config, **kwargs):
            from app.locdi.llm_setup import validate_setup
            self.prepared = validate_setup(config)['prepared']
            self.request_count = len(decisions)
            self.identity_check_count = 24
            Path(kwargs['checkpoint_root']).mkdir(parents=True, exist_ok=False)
        def close(self):
            pass
        def public_metadata(self):
            return {'TEST_ONLY_SAVED_REPLAY': True, 'new_generation': 0}
    def stored_run(*args, **kwargs):
        import app.core.hospital_llm_policy as policy
        for d in decisions:
            policy.role_call(role='er', payload={'TEST_ONLY': True}, saved_decision=d)
        return deepcopy(raw)
    with patch('app.locdi.current_ablation_llm_backend.BudgetedNativeERClient', StoredClient), \
         patch('app.locdi.run_simulation.run_case', stored_run), \
         patch('app.llm.role_call.role_call', side_effect=lambda **kw: deepcopy(kw['saved_decision'])):
        backend = make_er_backend({'TEST_ONLY_CASE': case}, config=config,
            authorized=True, contract_fingerprint='a'*64,
            checkpoint_root=PR/'TEST_ONLY_saved_backend')
        result = backend({'case_ref': 'TEST_ONLY_CASE'}, features_for('B0'))
    _validate_result(result, len(decisions))
    terminal = next(d for d in reversed(decisions) if d['action'] == 'diagnose')
    assert result['prediction']['diagnosis'] == terminal['target']
    score = matches(result['prediction']['diagnosis'], case['evaluation']['diagnosis_labels'])
    panel = read(OLDPR/'panel.json')[:1]
    from app.locdi.memory_paired_recovery import schedule
    scored_rows = []
    for task in schedule(panel):
        r = {**task, 'status': 'COMPLETED' if task['condition']=='B0' else 'NOT_ATTEMPTED',
            'result':result if task['condition']=='B0' else None,
            'request_count':len(decisions) if task['condition']=='B0' else 0}
        r['row_fingerprint'] = digest(r)
        scored_rows.append(r)
    scored = aggregate(panel, scored_rows)
    assert scored['episodes'][0]['source_diagnosis_exact'] == score['source_exact']
    assert scored['complete_pairs'] == 0
    write(PR/'TEST_ONLY_saved_backend_scoring.json', {'backend':result, 'score':score, 'aggregate':scored})
    return {'status':'PASS', 'new_generation':0, 'stored_decisions':len(decisions),
        'runner_raw_to_exporter_to_backend_to_scorer':'PASS',
        'actual_failed_Run_reclassified':False, 'clinical_accuracy':'NOT_EVALUATED'}


def prepare():
    if any(p.exists() for p in (PUB, PR, DOC)):
        raise ValueError('FRESH_NAMESPACE_REQUIRED')
    started = time.monotonic()
    old = read(OLDPUB/'SYNTHEA_PILOT_PROTOCOL.json')
    protected = read(OLDPR/'preservation_before_actual_pilot.json')
    for base in (OLDPUB, OLDPR, previous.DOC):
        protected.update({str(p):sha(p) for p in base.rglob('*') if p.is_file()})
    for base in ('app/core','prompts','configs'):
        protected.update({str(p):sha(p) for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    assert check(protected)['status'] == 'PASS'
    write(PR/'preservation_before.json', protected)
    for name in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json'):
        write(PR/name, read(OLDPR/name))
    source = read(PR/'source_rows.json'); panel = read(PR/'panel.json'); index = read(PR/'case_index.json')
    config = read(PR/'llm_config.json'); catalog = read(CATALOG); cases = load_cases(DATA)
    assert EXPORTER_VERSION == '1.3.2'
    assert len(source)==345 and digest(source)==old['source_rows_fingerprint']
    spec = policy_contract(source)
    assert spec == read(OLDPUB/'sequential_source_policy_final.json')
    write(PUB/'sequential_source_policy_final.json', spec)
    # No model/server client is constructed in any preparation operation.
    replay = saved_backend_replay(cases[index[panel[0]['case_ref']]-1], config,
        OLDPR/OLD_RUN/'requests/Episode0001')
    write(PUB/'saved_backend_replay.json', replay)
    sys.path.insert(0, str(ROOT/'tests/locdi'))
    with (PR/'related_regression.log').open('x',encoding='utf-8') as f:
        suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    tests={'tests_run':result.testsRun,'pass':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'skip':len(result.skipped),
        'scope':'13 relevant modules, not entire repository','generation':0}
    write(PUB/'test_results_final.json', tests)
    if not result.wasSuccessful() or result.testsRun != 227:raise ValueError('EXPECTED227_REGRESSION_NOT_PASS')
    from test_simulation import SourceActionClient,smoke_options
    from app.locdi.run_simulation import run_case
    tspec=policy_contract(source,test_only=True)
    view=SequentialSyntheticSourceView(source,spec=tspec,authorization=previous.test_authority(tspec),
        plan_fingerprint='TEST_ONLY_PROBE',catalog=catalog)
    class Client(SourceActionClient):
        def __init__(self):
            super().__init__();self.bounds=[]
        def chat_json(self,system,user,model=None,*,schema=None):
            payload=view.compact_payload(json.loads(user))
            user=json.dumps(payload,ensure_ascii=False,sort_keys=True)
            self.bounds.append(input_guard(system,user,schema))
            return super().chat_json(system,user,model=model,schema=schema)
    probes=[];source_facts=[]
    for ordinal,p in enumerate(panel,1):
        case=cases[index[p['case_ref']]-1];adapted=adapt_case(case)
        source_names={e['target']['description'] for e in case['environment']['source_event_store']
            if e.get('event_type')=='observations' and str(e.get('target',{}).get('code'))=='72514-3'}
        initial=adapted.runtime_case['patient_visible_info'].get('associated_symptoms',[])
        source_facts.append({'selected_patient_ordinal':ordinal,
            'chief_complaint_present':bool(adapted.runtime_case['doctor_visible_info'].get('chief_complaint')),
            'initial_measurement_labels':len(set(initial)&source_names),
            'initial_narrative_other_than_measurement_labels':len(set(initial)-source_names),
            'source_catalog_coverage':all(x in catalog for x in case['evaluation']['diagnosis_labels'])})
        for condition in ('B0','B2'):
            client=Client()
            raw=run_case(case,options=smoke_options(client,runner_options=features_for(condition),
                diagnosis_catalog=catalog,max_memory_chars=None,sequential_source_view=view))
            episode=build_episode(raw,context=adapted)
            assert raw['status']=='completed' and episode['exporter_version']=='1.3.2'
            added=raw['synthetic_source_view_audit'][0]['added_tests'];names={x['test_name'] for x in added}
            source_by_fp={digest(r['source_event']):r['source_event'] for r in source}
            expected={x['test_name']:source_by_fp[x['source_fingerprint']]['evidence'] for x in added}
            events=raw['execution']['visits'][0]['final_state']['event_log']
            received={e['payload']['requested_test']:e['minute'] for e in events
                if e['event_type']=='result_received' and e['payload'].get('requested_test') in names}
            starts={e['payload']['requested_test']:e['minute'] for e in events
                if e['event_type']=='test_started' and e['payload'].get('requested_test') in names}
            for name,minute in received.items():assert minute>=starts[name]+5
            delivered=0;memory_checks=0
            for call in client.calls:
                payload=call['payload'];o=payload['observation']
                assert payload['diagnosis_catalog']==catalog
                assert not {'gold','evaluation','provenance','task_annotations','source_event_store'} & set(payload)
                for finding in o['objective_findings']:
                    for name in names:
                        if finding.startswith(name+': '):
                            assert name in received and o['elapsed_minutes']>=received[name]
                            value=json.loads(finding[len(name)+2:])
                            assert value['value']==expected[name]['value'] and value['units']==expected[name].get('units')
                            delivered+=1
                memory=o.get('longitudinal_memory',{})
                if condition=='B0':assert not memory.get('records')
                elif memory.get('records'):
                    assert verify_delivered_memory(case,memory)['status']=='PASS';memory_checks+=1
            write(PR/('TEST_ONLY_Patient%02d_%s.json'%(ordinal,condition)),
                {'raw':raw,'episode':episode,'model_payloads':client.calls,'input_bounds':client.bounds,'TEST_ONLY':True})
            probes.append({'patient_ordinal':ordinal,'condition':condition,'Core':'completed','Export':'PASS',
                'additional_menu':len(names),'additional_received':len(received),'source_value_unit_occurrences':delivered,
                'memory_boundary_checked_payloads':memory_checks,'temporal_violations':0,
                'max_input_bound':max(x['conservative_input_token_bound'] for x in client.bounds)})
    write(PUB/'TEST_ONLY_selected10_full_pipeline.json', probes)
    write(PUB/'source_input_facts.json', source_facts)
    upstream={}
    for rel in ('30_source_enriched_study_50/build_study.py','34_live_action_policy/prepare_package.py',
        '24_action_replay_100/build_action_replay.py'):
        path=EXPERIMENTS/rel
        upstream[rel]={'exists':path.exists(),'SHA256':sha(path) if path.exists() else None}
    write(PUB/'upstream_source_code_inventory.json',upstream)
    code=list(old['new_code_hashes'])+['app/locdi/episode_exporter.py','app/locdi/memory_paired_recovery.py',
        'app/locdi/case_adapter.py','app/locdi/synthea_diagnosis_audit.py',
        'scripts/prepare_locdi_pilot_improvement.py','tests/locdi/test_sequential_export_contract.py']
    # Pin all clinical Core/LLM implementations used by this closure as well.
    code+= [str(p.relative_to(ROOT)).replace('\\','/') for base in ('app/core','app/llm')
        for p in (ROOT/base).glob('*.py')]
    protocol=deepcopy(old);protocol.pop('fingerprint')
    protocol.update(namespace=NS,run_id=RUN,version='synthea.sequential.export_recovery.v1',
        prior_failed_Run=OLD_RUN,prior_protocol=old['fingerprint'],
        changed_variable='Exporter metadata compatibility only (1.3.1 -> 1.3.2); no clinical payload or Prompt change',
        improvement_evidence='Static D4 4/20 is contextual evidence; efficacy of sequential Source/Memory is NOT_YET_EVALUATED',
        small_additional_generation_budget=0,extra_diagnostic_experiments='NOT_PLANNED',
        new_code_hashes={f:sha(ROOT/f) for f in sorted(set(code))},
        input_hashes={str(PR/n):sha(PR/n) for n in ('source_rows.json','panel.json','case_index.json','llm_config.json','raw_file_hashes.json')},
        protected_manifest_sha=sha(PR/'preservation_before.json'),
        TEST_ONLY_max_input_bound=max(x['max_input_bound'] for x in probes),
        status='OFFLINE_READY_NEW_EXECUTION_APPROVAL_REQUIRED',
        created_at=datetime.now(timezone.utc).isoformat())
    protocol['binding_hashes'].update({str(PUB/'test_results_final.json'):sha(PUB/'test_results_final.json'),
        str(PUB/'TEST_ONLY_selected10_full_pipeline.json'):sha(PUB/'TEST_ONLY_selected10_full_pipeline.json')})
    protocol['fingerprint']=digest(protocol);write(PUB/'SYNTHEA_PILOT_PROTOCOL.json',protocol)
    write(PUB/'preservation_verification.json',check(protected))
    reports(protocol,tests,probes,source_facts,replay)
    write(PUB/'delivery_manifest.json',{'status':'OFFLINE_READY_NEW_EXECUTION_APPROVAL_REQUIRED','generation':0,
        'protocol':protocol['fingerprint'],'tests':tests,'saved_replay':replay,'Core_probes':len(probes),
        'protected':check(protected),'elapsed_seconds':time.monotonic()-started,
        'clinical_metrics':'NOT_EVALUATED','reports':{f.name:sha(f) for f in DOC.glob('*.md')}})
    print(json.dumps({'status':'OFFLINE_READY_APPROVAL_PENDING','tests':tests,'Core_probes':len(probes),
        'protocol':protocol['fingerprint'],'max_bound':protocol['TEST_ONLY_max_input_bound']}))


def reports(p,tests,probes,facts,replay):
    common=f'\n\n신규 Namespace `{NS}`, Run `{RUN}`. 실제 신규 Generation **0**, 실제 새 Pilot **NOT_RUN**. 계획 `{p["fingerprint"]}`. '
    common+='기존 synthea_sequential_001 실패1/미실행19/완료쌍0과 D4 정적4/20을 변경하지 않는다. 개발노출 합성 패널이며 임상 정확도·Concept Match는 NOT_EVALUATED다.\n'
    md('PILOT_DATA_USAGE_COMPARISON.md','# 기존 Pilot과 현재 Native ER\n\n'
        '|항목|기존 experiment24/30/34 실제 코드|현재 Native ER|\n|---|---|---|\n'
        '|초기 상태|24에서 측정항목명을associated_symptoms, 수치는Ask에 분리|동일 구조를Adapter가 전달; 없는chief complaint는 비움|\n'
        '|검사|30 transform이 모든진단 cell 지원 행동만 cases에 남기고 추가검사는 review view로 분리|VoI OFF 별도 승인정책의 원천345행 중 선정10명175행 메뉴|\n'
        '|공개|available_minute는 재생 가정, 실제 임상 시각 아님|요청→시작→max(start+5,source proxy)수신→모델 공개|\n'
        '|과거 기록|환자·방문 연결 past_visits/memory|기존 cutoff·token-intersection Top5, B0 none/B2 selective|\n'
        '|진단|30 공통15종; 34가 같은 cases/registry/model을 복사|같은15종을명시적으로 binding, source labels/department는평가전용|\n'
        '|확률/정책|30은학습cell 관측·median bin·smoothing 모델, 34는MemoryAwareLLMRouter 계약도 명시|이실험Routing OFF/VoI OFF; 과거확률·성능을재사용하지않음|\n'
        '\n근거: experiments/24_action_replay_100/build_action_replay.py::main(초기명칭/Ask), '
        '30_source_enriched_study_50/build_study.py::main 내부transform, '
        '34_live_action_policy/prepare_package.py::main. 실제경로와SHA는upstream_source_code_inventory.json. '
        '원래38/52/44/48/90%의원본Run과현재패널은동일평가로인정하지않는다.'+common)
    md('CURRENT_SYNTHEA_INPUT_IMPROVEMENT.md','# 현재 입력 전달 사실과 변경 경계\n\n'
        f'선정10명 chief complaint 존재 {sum(x["chief_complaint_present"] for x in facts)}/10. '
        f'측정항목명 외 초기 narrative 존재 {sum(x["initial_narrative_other_than_measurement_labels"]>0 for x in facts)}/10. '
        f'공통Catalog Source label coverage {sum(x["source_catalog_coverage"] for x in facts)}/10. '
        '통증 측정항목명은 실제 통증 호소·병명이 아니다. 현재값을 임의 증상 문장으로 보충하지 않는다. '
        '과거measurement_view는 별도정적표현실험으로보존하며 이번순차실행에동시에적용하지않는다. '
        '실제 Source test명은 code/full description을포함하고 결과 value/units/recorded_at을그대로보낸다. '
        '20개 TEST_ONLY Core→Export에서 수신후 전달값·단위 및 Memory 경계를검증했다. '
        '신규임상 View/Prompt 변경0; 이번의한변경요소는이미수정된Exporter 계약의실제경로연결이다. '
        '진단에충분한현재증상이없다는Source 한계와정보전달결함을혼동하지않는다.'+common)
    md('FINAL_EXPORTER_VERIFICATION.md','# Exporter 및 전체 후처리 검증\n\n'
        '현재episode_exporter.EXPORTER_VERSION=1.3.2. _configuration은optional sequential_source_view의정확한7키·정책·타입·5분·clinicalFalse·grant0·SHA형식을검사하고전체fingerprint/unknown key차단을유지한다. '
        '저장raw→build_episode→기존make_er_backend후처리→_validate_result→기존aggregate scorer 및 별도matches까지비생성으로검증했다. '
        'Runner 재실행으로과거모델출력을재생성하지않고, 저장된Runner결과만후처리했다. '
        '추가20개 Fake SourceActionClient probe는실제Runner/Scheduler/Adapter/Exporter를사용했으며 성능평가아니다.\n\n'
        +json.dumps({'related_regression':tests,'saved_result_replay':replay,'mock_Core_exports':len(probes)},ensure_ascii=False,indent=2)+common)
    md('FINAL_SYNTHEA_ACCURACY_IMPROVEMENT.md','# 진단 개선 상태와 제한\n\n'
        '확보한효과는기존정적D4 Source Exact4/20, D1/D2/D3각0/20이다. 확대Evidence와Catalog결합의개발신호이며 순차진단·Memory효과로대체하지않는다. '
        '첫실패순차B0는12요청·추가검사6수신·2결정에서추가Evidence인용, 최종진단Source불일치/진료과일치였지만Export실패로공식완료분모에승격하지않는다. '
        '결함없는값전달을확인하고Exporter문제부터해결했다. Prompt최적화·환자별Gold후보·동의어·검사값수정없음. '
        '추가one-shot진단실험은0회로고정하고 새순차Pilot1회만승인요청한다. '
        '정확도상승은아직확인되지않았으며 결과가개선되지않으면후속최적화/자동확장을하지않는다.'+common)
    md('FINAL_SEQUENTIAL_PILOT_RESULTS.md','# 신규 순차 Pilot 실행 준비\n\n'
        'NOT_RUN — 새운영자실행승인필요. 이전승인은새Run에재사용하지않는다. '
        '동일선정ordinal[1,2,5,7,10,12,13,14,15,19], endocrine5/neuro5. 환자별B0→B2총20Episode. '
        '각12/총240회,ctx32768/output1024/temp0/seed42/timeout120초/retry0/repair0/warmup0, '
        'wall2700초/process4GiB/output1GiB. 첫기술오류에전체중단/no-resume, 부분실패·미실행을보존한다. '
        'Source/Gold/미래정보·모델Identity·매요청Context/byteguard는그대로다. '
        '최대bytebound여유825는Mock정책범위이며실제후속입력은기존guard에서실패할수있다. '
        '실행명령: `python -B scripts/prepare_locdi_pilot_improvement.py execute --execute --allow-llm --authorization <새실제승인JSON>`.'+common)
    md('FINAL_B0_B2_MEMORY_ANALYSIS.md','# Memory 평가 계약\n\n'
        '현재새실제완료쌍0, paired차이NOT_EVALUATED. 완료율분모각10, 등록전체20과완료조건부결과를별도보고한다. '
        'Source Exact는supported정답수/등록수와supported응답중일치율을분리한다. unresolved/실패/미실행을자동오답라벨로바꾸지않는다. '
        'Source department는실제완료후진료과예측이있는분모, Normalized는별도기존matches함수, Concept/Clinical은NOT_EVALUATED. '
        'B0/B2같은공개Source/Catalog/Prompt/scorer/행동상한, Memory만none대selectiveTop5. '
        '정적D4·기존실패Run·과거20명실험과새결과를합쳐분모를바꾸지않는다.'+common)
    md('FINAL_THESIS_RESULTS_COMPARISON.md','# 역사적 논문 수치와 현재 증거\n\n'
        '|자료|분모/조건|현재 해석|\n|---|---|---|\n'
        '|논문 B0~B4|역사적38/52/44/48/90%, 과거모델/Run|현재실험재현값아님, 단순대체금지|\n'
        '|정적D4|기존20명/3B/확대Evidence+15Catalog|Source Exact4/20; 순차/Memory효과아님|\n'
        '|기존순차001|등록20, 실패1/미실행19/쌍0|추가Test경로실행; Export실패로B0/B2미평가|\n'
        '|새순차002|선정10명,20Episode,같은3B|NOT_RUN; 승인후실제상태/정답분모로기록|\n'
        '동일개발패널반복노출, 초기임상narrative부족, recorded proxy/합성TAT, ordinal12다음날Reference, '
        '독립임상검토없음을한계로유지한다.'+common)
    md('FINAL_PROFESSOR_BRIEFING.md','# 교수님 설명 자료\n\n'
        '과거논문성능을재현하거나90%에맞추는작업이아니다. 같은Synthea/3B로실제정보전달과순차진단가능성을검증한다. '
        '추가345행을Source검증하고선정10명175행을동일규칙으로연결했으며정적D4에서4/20신호가있다. '
        '첫순차실행은모델12요청이정상이어도Exporter누락으로실패했다. 현재1.3.2의후처리전체경로와227회귀/20MockEpisode는통과했으나새실제성능은미확보다. '
        '다음결정은같은설정/새Run으로20Episode최대240회재실행할지여부한가지다. '
        'Prompt·모델·수식·Source/Scorer/임상Gate를변경하지않고개선여부를있는그대로종료보고한다.'+common)


def execute(authorization):
    # A single existing bounded collector, rebound to a fresh namespace.
    previous.PUB, previous.PR, previous.DOC, previous.RUN = PUB, PR, DOC, RUN
    previous.execute(authorization)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','execute'])
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true')
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    if args.mode=='execute':
        if not(args.execute and args.allow_llm and args.authorization):parser.error('NEW_EXPLICIT_AUTHORITY_REQUIRED')
        execute(args.authorization)
    else:prepare()
