"""C4-36 evidence audit and separately fixed TEST_ONLY Hospital probe.

Default audit creates new exclusive files, no policy scoring or real model calls.
Completed stages cannot be overwritten or resumed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_research_gaps import read, write, sha, check
from app.locdi.llm_setup import digest
from app.locdi.dataset_loader import load_cases
from app.locdi.integrated_framework_readiness import (
    official_contract, memory_voi_compatibility, source_summary, callable_inventory,
)
from app.locdi.integrated_development_evaluation import technical_protocol, run_technical, block_source_evaluation

PUB = ROOT / 'outputs/research_evaluation/c4_36_v1'
PR = ROOT / 'outputs/private/research_evaluation/c4_36_v1'
U35 = ROOT / 'outputs/research_evaluation/c4_35_v1'
P35 = ROOT / 'outputs/private/research_evaluation/c4_35_v1'
M5 = ROOT / 'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
DATA = ROOT / 'datasets/raw/longitudinal_benchmark_50.jsonl'
CODE = ['app/locdi/integrated_framework_readiness.py', 'app/locdi/integrated_development_evaluation.py',
        'scripts/evaluate_integrated_framework_readiness.py', 'tests/locdi/test_integrated_framework_readiness.py']
AUDIT_CODE = ['app/core/longitudinal_memory.py', 'app/core/clinical_observation.py',
    'app/core/longitudinal_runner.py', 'app/core/memory_retrieval.py', 'app/core/agent_registry.py',
    'app/core/routing_utility.py', 'app/core/diagnostic_belief.py', 'app/core/voi_planning.py',
    'app/core/information_value.py', 'app/core/information_outcomes.py', 'app/core/hospital_llm_policy.py',
    'app/core/hospital_case_runtime.py', 'app/core/hospital_parallel_scheduler.py',
    'app/locdi/dynamic_horizon.py', 'app/locdi/integrated_runtime.py', 'app/locdi/handoff_integration.py',
    'app/locdi/run_simulation.py', 'app/locdi/voi_source_replay_environment.py',
    'app/locdi/voi_new_cohort_evaluation.py', 'app/locdi/new_cohort_execution.py',
    'app/locdi/concept_diverse_memory.py', 'app/locdi/case_adapter.py',
    'app/locdi/episode_exporter.py', 'experiments/run_locdi_ablation.py',
    'validation/artifact_readiness.py', 'validation/research_artifact_gate.py',
    'validation/routing_correctness_gate.py']


def evidence(ns, name):
    return read(ROOT / 'outputs/research_evaluation' / ns / name)


def formula_rows():
    values = [
        ('M1', 'eligible(r)=same_patient AND other_visit AND visit_started_at<cutoff AND available_at<=cutoff',
         'app/core/longitudinal_memory.py', 'PatientMemoryStore.retrieve', 'record identity/timestamps', 'eligible records', 'test_simulation.py'),
        ('M2', 's(r,q)=|unique_tokens(r.text) intersect unique_tokens(q)|; sort(-s,-available_at,memory_id)',
         'app/core/longitudinal_memory.py', 'PatientMemoryStore.retrieve', 'public query, past eligible text', 'Selective Top-K', 'test_memory_concept_evaluation.py'),
        ('M3', 'first distinct exact verified concept in unchanged rank; then unchanged-rank fill to5',
         'app/locdi/concept_diverse_memory.py', 'select', 'sidecar-enriched past candidates, development opt-in', '5 IDs', 'test_concept_diverse_memory.py'),
        ('R1', 'J(A,B)=|A intersect B|/|A union B|; empty union ->0',
         'app/core/routing_utility.py', 'jaccard_similarity', 'observable token sets', 'similarity', 'test_routing_voi_readiness.py'),
        ('R2', 'p_hat=(1+sum_i J_i*correct_i)/(2+sum_i J_i), only positive matches and min_samples support',
         'app/core/routing_utility.py', 'RoutingUtility.score', 'reviewed correctness samples and public tokens', 'uncalibrated correctness estimate or None', 'test_routing_evaluation.py'),
        ('R3', 'rule_score(dept)=sum_keyword 1[keyword_tokens subset observable_tokens]; zero ->abstain; ties lexical',
         'app/core/agent_registry.py', 'AgentRegistry.route', 'public observation, registry candidates, public memory', 'selected/abstained/unavailable/disabled', 'test_runtime_integration.py'),
        ('V1', 'P(o|b,a)=sum_d b(d)L(o|d,a)', 'app/core/voi_planning.py', 'outcome_branches',
         'belief/action likelihood', 'predictive mass and branch posterior', 'test_voi_stop_consistency.py'),
        ('V2', 'b_o(d)=b(d)L(o|d,a)/sum_j b(j)L(o|j,a)', 'app/core/diagnostic_belief.py', 'posterior',
         'normalized prior and supplied likelihood', 'posterior,mass', 'test_voi_probability_validation.py'),
        ('V3', 'T(b)=max_c sum_d b(d)U(c,d); U(c,d)=1[c=d] validation only',
         'app/core/diagnostic_belief.py', 'terminal_choice/validation_utility', 'belief,U', 'choice,value', 'test_voi_stop_consistency.py'),
        ('V4', 'V_0=T; Q_h(a)=sum_o P(o|b,a)V_(h-1)(b_o,A\\{a}); V_h=max(T,max_a Q_h)',
         'app/core/voi_planning.py', 'plan_voi', 'belief,remaining actions,U,h1..3', 'Q,VoI,root action,states', 'test_dynamic_horizon.py'),
        ('V5', 'VoI=Q-T; CONTINUE iff bestQ>T+1e-12; ties action_id lexical',
         'app/core/voi_planning.py', 'plan_voi', 'Bellman root scores', 'STOP or selected ID', 'test_voi_stop_consistency.py'),
        ('V6', 'Bayesian selector obeys planner root; IG selector maximizes IG independently',
         'app/core/information_value.py', 'select_action', 'executable candidates,belief,artifact', 'action/unavailable', 'test_voi_stop_consistency.py'),
        ('D1', 'requested=max(entropy_band(H/logK),margin_band(p1-p2)); selected=min(requested,remaining_actions,3)',
         'app/locdi/dynamic_horizon.py', 'DynamicHorizonPolicy.select', 'public belief/menu/context; thresholds1/3,2/3',
         'selected/effective horizon or terminal/fallback/blocked', 'test_dynamic_horizon.py'),
    ]
    return [{'formula_id': i, 'formula': f, 'file': p, 'function': fn, 'actual_inputs': inp, 'output': out,
             'tests': t, 'code_alignment': 'VERIFIED_BY_SOURCE_AND_RELATED_TESTS',
             'paper_equation_number': None, 'paper_alignment': 'NOT_CHECKED_NO_VERIFIED_MANUSCRIPT',
             'changed': False} for i,f,p,fn,inp,out,t in values]


def claim_matrix():
    memory = evidence('c4_22_v1', 'same_budget_comparison.json')
    primary = evidence('c4_35_v1', 'fresh_cohort_primary_comparison.json')
    voi = evidence('c4_35_v1', 'fresh_cohort_native_policy_metrics.json')
    compact = evidence('c4_27_v1', 'combined_25_task_metrics.json')
    rows = []
    def add(i, component, definition, equations, implementation, tests, source, level, numbers,
            denominator, baseline, statistics, gate, limitations, allowed):
        rows.append({'claim_id': i, 'component': component,
            'paper_definition': 'NOT_CHECKED_NO_VERIFIED_MANUSCRIPT', 'code_protocol_definition': definition,
            'formulas': equations, 'implementation': implementation, 'tests': tests,
            'source_evaluation': source, 'evidence_level': level, 'numbers': numbers,
            'denominator': denominator, 'baseline': baseline, 'statistics': statistics,
            'existing_gate': gate, 'limitations': limitations, 'permitted_claim': allowed,
            'external_clinical_validation': 'MISSING'})
    add('MEM-1','Memory','Cutoff-bounded lexical Selective Top5',['M1','M2'],
        ['app/core/longitudinal_memory.py:PatientMemoryStore.retrieve'], ['test_simulation.py','test_memory_concept_evaluation.py'],
        ['c4_21_v1/record_baseline_reproduction.json','c4_22_v1/same_budget_comparison.json'], 'DEVELOPMENT_EVALUATED',
        {'precision':memory['baseline']['record_precision_micro'],'concept_coverage':memory['baseline']['concept_coverage_micro']},
        'Original50,72590 memories; precision250 returns; concept40 complete-positive queries/200 concepts',
        'OFF/FullHistory with unequal return budgets; Selective replay reproduction', 'descriptive, not independent clinical relevance',
        'structural cutoff supported; clinical review missing', ['minute45 source-grounded query','identity reference not clinical usefulness'],
        'Source identity retrieval measured; runtime cutoff contract implemented.')
    add('MEM-2','Memory','Concept-Diverse opt-in same5-record budget',['M2','M3'],
        ['app/locdi/concept_diverse_memory.py:select'], ['test_concept_diverse_memory.py'],
        ['c4_22_v1/same_budget_comparison.json','c4_23_v1/temporal_information_loss.json'], 'DEVELOPMENT_GAIN_SUPPORTED',
        {'baseline_coverage':memory['baseline']['concept_coverage_micro'],
         'diverse_coverage':memory['concept_diverse']['concept_coverage_micro'],
         'baseline_precision':memory['baseline']['record_precision_micro'],
         'diverse_precision':memory['concept_diverse']['record_precision_micro'],
         'temporal':evidence('c4_23_v1','temporal_information_loss.json')},
        'coverage40queries/200concepts; precision50queries/250returns; temporal245mapped records per mode',
        'Selective Top5','development descriptive; same exposed cohort', 'production opt-in adapter NOT_DEPLOYED',
        ['precision declines','47 queries lose times','metadata availability unreviewed'],
        'Source concept coverage improves30% to92%, with precision and temporal tradeoffs; default unchanged.')
    add('ROUTE-1','Routing','Rule baseline and accuracy-only Adaptive with support abstention',['R1','R2','R3'],
        ['app/core/agent_registry.py:select_specialty/route','app/core/routing_utility.py:RoutingUtility.score'],
        ['test_routing_evaluation.py','test_routing_correctness_gate.py'],
        ['c4_27_v1/utility_feasibility.json','c4_28_v1/role_matched_comparison.json'], 'BLOCKED',
        {'compact25':compact['outcomes']['counts'],'fit_ready':False},
        '5 exposed questions x5 agents; answer supported2/25, correct0; not patient routing Gold',
        'fixed agent/rule/oracle potential only', 'no validated utility fit or independent specialty accuracy',
        'MISSING_UTILITY', ['unresolved remains null correctness','weight comparison19/20 attempted, invalid1,not_attempted1'],
        'Routing software/abstention exists; adaptive clinical routing benefit not demonstrated.')
    add('VOI-MATH','VoI','Cost-free finite horizon Bayes/Bellman/root stop consistency',['V1','V2','V3','V4','V5','V6'],
        ['app/core/voi_planning.py:plan_voi','app/core/information_value.py:select_action'],
        ['test_voi_stop_consistency.py','test_dynamic_horizon.py'], ['C4_03 numerical consistency repair'],
        'MATHEMATICALLY_VERIFIED', {'epsilon':1e-12,'horizon_cap':3,'clinical_utility':False},
        'contract/boundary fixtures; not clinical cohort', 'planner versus selector, IG preserved',
        'exact unit/integration contracts', 'research likelihood/utility review not implied',
        ['U1/0 dimensionless','conditional independence assumption'], 'Implemented equations and numerical root contract verified.')
    add('VOI-DEV','VoI','Matched-budget sequential Source acquisition',['V1','V2','V3','V4','V5'],
        ['app/locdi/voi_new_cohort_evaluation.py:policy_episode','app/locdi/voi_source_replay_environment.py:acquire'],
        ['test_voi_sequential_policy_evaluation.py'], ['c4_31_v1/primary_comparison.json'],
        'NEGATIVE_OR_INCONCLUSIVE_RESULT', evidence('c4_31_v1','primary_comparison.json'),
        '63patients92complete visits out of173patients229visits', 'H3 minus Global EIG',
        'patient bootstrap2000; primary CI includes0','development only',
        ['complete-case selection','no clinical Gold','Budget5 reverses'], 'Development strong-baseline superiority inconclusive; weak baselines show gains.')
    add('VOI-FRESH','VoI','Frozen M5, new-seed patient-quarantined Source panel',['V1','V2','V3','V4','V5'],
        ['scripts/evaluate_admitted_voi_cohort_bounded.py:execute'], ['test_c435_bounded_runner.py','test_c435_execution_boundary.py'],
        ['c4_35_v1/actual_evaluation_summary.json','c4_35_v1/actual_evaluation_verification.json'],
        'FRESH_SYNTHETIC_GAIN_SUPPORTED',
        {'H3_AUBC':voi['aubc']['voi_h3']['patient_macro']['top1'],
         'EIG_AUBC':voi['aubc']['global_eig']['patient_macro']['top1'],'primary':primary['patient_macro']['top1'],
         'secondary':evidence('c4_35_v1','fresh_cohort_paired_policy_comparison.json')},
        '1000generated;86/160structural;51/89quarantined;35patients71visits;3834slots17040replays;9/15classes',
        'H3 minus Global EIG (primary)', 'patient cluster bootstrap2000 seed42 percentile95%',
        'C4_35 Approval A/B completed for THAT RUN ONLY; GateA remains0',
        ['400 old patients identity-only','same generator','quarantine selection','Budget5 and Top3 inferiority','NLL/Brier CI includes0'],
        'Frozen H3 yields positive primary Source AUBC difference on the bounded fresh synthetic panel, not full-framework or clinical gain.')
    add('DYNAMIC','Dynamic Horizon','Existing entropy/margin/feasibility rule',['D1'],
        ['app/locdi/dynamic_horizon.py:DynamicHorizonPolicy.select'], ['test_dynamic_horizon_source_evaluation.py'],
        ['c4_32_v1/horizon_saturation_analysis.json','c4_35_v1/fresh_cohort_horizon_comparison.json'],
        'NEGATIVE_OR_INCONCLUSIVE_RESULT', {'fresh_dynamic_H3_difference':0,'development_H3_requests':551,'eligible':552},
        'C4_32 552 episodes; C4_35 71visits x6budgets', 'Dynamic vs H3/H1',
        'fresh H3 difference0; no equivalence margin', 'Source replay only; Hospital hook missing',
        ['H3 saturation','no meaningful H3 computation saving','H2/H3 vs H1 CI includes0'],
        'Rule implemented and evaluated; extra depth/clinical adaptation benefit not established.')
    add('INTEGRATION','Framework','Existing Hospital Runner connects components with explicit test artifacts',
        ['M1','M2','R2','V4'], ['app/locdi/integrated_runtime.py:run_integrated','app/locdi/handoff_integration.py:run_handoff_integrated'],
        ['test_runtime_integration.py','test_handoff_integration.py','test_integrated_framework_readiness.py'],
        ['C4_36 TEST_ONLY execution'], 'IMPLEMENTED_ONLY', {'clinical_scores':None},
        'one actual Original50 source case x6 TEST_ONLY conditions; not6patients', 'OFF/ON fixed technical conditions',
        'no performance inference', 'OFFICIAL_ABLATION_BLOCKED',
        ['toy likelihoods','no independent utility','no memory-M5 likelihood','not live LLM'],
        'Technical module/trace integration only; no full LoCDI effect estimate.')
    return {'claims':rows,'manuscript_status':'NOT_CHECKED_NO_VERIFIED_THESIS','equation_ID_scope':'audit-local IDs, not paper numbering'}


def audit():
    before = check(read(PR / 'protected_hashes.json'))
    if before['changed_or_missing']:
        raise ValueError('PROTECTED_ASSET_CHANGED')
    write(PR / 'initial_preservation.json', before)
    code = {n:sha(ROOT/n) for n in CODE}
    cases = load_cases(DATA); manifest = read(P35/'approved_admission.json'); model = read(M5)
    if sha(M5) != '7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c':
        raise ValueError('FROZEN_M5_CHANGED')
    if check(read(U35/'approval_B_completion_manifest.json')['files'])['changed_or_missing']:
        raise ValueError('C4_35_COMPLETION_CHANGED')
    protocol = technical_protocol(cases[0], sha(DATA), code)
    protocol['runtime_dependency_hashes'] = {n:sha(ROOT/n) for n in AUDIT_CODE}
    protocol['fixture_hashes'] = {n:sha(ROOT/n) for n in ['tests/locdi/runtime_integration_support.py',
        'tests/locdi/test_routing_voi_readiness.py','tests/locdi/test_simulation.py']}
    protocol.pop('fingerprint'); protocol['fingerprint']=digest(protocol)
    write(PUB/'integrated_evaluation_protocol.json', protocol)
    docs = {}
    for i in [1,18,19]+list(range(21,36)):
        for p in (ROOT/'docs').rglob('C4_%02d*.md'%i):
            text = p.read_text(encoding='utf-8')
            docs[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'lines':len(text.splitlines()),
                'read_scope':'full UTF8 report; figures not reinterpreted'}
    manuscript_candidates = [p.relative_to(ROOT).as_posix() for p in (ROOT/'docs').rglob('*')
                             if p.suffix.lower() in {'.pdf','.docx','.tex'}]
    write(PUB/'source_document_inventory.json', {'reports':docs, 'manuscript_candidates':manuscript_candidates,
        'manuscript_status':'NOT_FOUND_VERIFIED_THESIS_WITHIN_WORKSPACE; two037 PDFs are implementation reports',
        'paper_claims_and_equation_numbers':'NOT_CHECKED', 'C4_30_actual_namespace':'c4_30_v2'})
    write(PUB/'research_evidence_matrix.json', claim_matrix())
    write(PUB/'formula_to_code_mapping.json', {'formulas':formula_rows(),
        'callables':callable_inventory(ROOT,AUDIT_CODE),'source_sha256':protocol['runtime_dependency_hashes']})
    contracts = [official_contract(read(ROOT/('experiments/configs/B%d.json'%i))) for i in range(5)]
    write(PUB/'official_b0_b4_contract.json', {'conditions':contracts,'R0':'optional, not officialB0..B4',
        'comparison_confounders':['B0-B1 information volume/context length','B1-B2 amount and retrieval',
         'B2-B3 adaptive routing plus specialty access','B3-B4 VoI selection plus memory acquisition timing'],
        'official_sources':['experiments/configs/B0.json..B4.json','experiments/run_locdi_ablation.py:CONDITIONS']})
    write(PUB/'official_ablation_readiness.json', {'status':'OFFICIAL_ABLATION_BLOCKED',
        'all_conditions_blocked':True,'condition_checks':{c['condition']:c['execution_status'] for c in contracts},
        'blockers':['ready=false/all common model,version,decoding,registry,reference,split etc null',
            'official runner not implemented','research_evaluation purpose explicitly rejected',
            'B3/B4 MISSING_UTILITY','B4 runtime encoding/clinical support/source mapping not cleared'],
        'gate_override':False, 'official_runs':0})
    records = [r for v in cases[0]['agent_visible']['past_visits'] for r in v['records']]
    compatibility = memory_voi_compatibility(model, records)
    write(PUB/'memory_voi_compatibility.json', compatibility)
    write(PUB/'artifact_gate_matrix.json', {'Final_M5':compatibility['runtime_artifact_gate'],
        'numeric_source_replay':'C4_35 completed; not Hospital runtime gate',
        'memory_concept_sidecar':'C4_23 not production deployed',
        'routing_utility':'MISSING_UTILITY','clinical_correctness_gate':'unchanged, not automatically reviewed',
        'TEST_ONLY_artifacts':'technical execution only; never promote', 'official_status':'BLOCKED'})
    source = source_summary(cases, manifest['panel'], model)
    raw = ROOT/'outputs/private/research_evaluation/c4_34_v1/lossless_source_cache_v2.sqlite'
    db=sqlite3.connect(raw.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        links=[]
        for v in manifest['panel']:
            enc=db.execute('SELECT data FROM encounters WHERE pid=? AND eid=?',(v['patient_id'],v['visit_id'])).fetchall()
            if len(enc)!=1:raise ValueError('ADMITTED_RAW_LINK')
            current=read_json_text(enc[0][0])
            prior=sum(read_json_text(row[0])['START']<current['START'] for row in
                      db.execute('SELECT data FROM encounters WHERE pid=?',(v['patient_id'],)))
            links.append({'patient_id':v['patient_id'],'visit_id':v['visit_id'],'past_raw_encounters':prior})
    finally:db.close()
    write(PR/'admitted_patient_history_links.json',links)
    source['C4_35'].update(raw_patient_visit_links_verified=len(links),visits_with_past_raw_encounter=sum(r['past_raw_encounters']>0 for r in links))
    source['development_sources']={'C4_29_C4_30_base':'173patients229visits, action-specific rows; source references not independent clinical Gold',
        'C4_30_augmentation':'126patients142visits;87new_to_fit patients98visits,39oldpatients44visits; development exposed',
        'C4_31_C4_32':'63patients92complete visits;11/15classes; same exposed development Source',
        'training_overlap':'Routing validation5 intersects VoI train5, inherited issue not resolved',
        'full_same_patient_integrated_scoring':'BLOCKED_NO_CLEARED_MEMORY_LIKELIHOOD_ROUTING_REFERENCE'}
    write(PUB/'integrated_source_eligibility.json',source)
    write(PUB/'routing_integration_status.json', {'RuleSimilarity':'IMPLEMENTED_RUNTIME_AND_TECHNICAL_PROBE',
        'Adaptive':'TEST_ONLY_RUNTIME; MISSING_UTILITY for research','training':'BLOCKED',
        'reference':'independent specialty reference missing; answer exact match not referral Gold',
        'Gold_in_router':'not supplied by this probe; existing observable-only route implementation inspected',
        'logical_primary_role':'distinct from source-backed Consult and sequential live handoff',
        'C4_27_outcomes':evidence('c4_27_v1','combined_25_task_metrics.json')['outcomes'],
        'C4_28_counts':evidence('c4_28_v1','model_level_metrics.json')['all_registered']['counts'],
        'clinical_metrics':None})
    graph = [
        ('Source','load_cases/adapt_case','app/locdi/case_adapter.py','ACTUAL_RUNTIME'),
        ('State','prepare_encounter/doctor_observation','app/core/clinical_observation.py','ACTUAL_RUNTIME'),
        ('Memory','run_longitudinal -> PatientMemoryStore.retrieve -> memory_retriever','app/core/longitudinal_runner.py','ACTUAL_RUNTIME'),
        ('Prompt','HospitalLLMPolicy._decision -> role_call','app/core/hospital_llm_policy.py','ACTUAL_RUNTIME_LIVE_OR_OFFLINE_CLIENT'),
        ('Belief','InformationPolicy.propose -> InformationOutcomes.observe','app/core/information_value.py','RUNTIME_REQUIRES_COMPATIBLE_ARTIFACT'),
        ('Action menu','candidate_actions/source targets','app/core/information_value.py','ACTUAL_RUNTIME'),
        ('Planner','select_action -> plan_voi','app/core/voi_planning.py','ACTUAL_RUNTIME_TEST_ONLY_VERIFIED'),
        ('Release','HospitalParallelScheduler -> HospitalCaseRuntime','app/core/hospital_parallel_scheduler.py','ACTUAL_RUNTIME'),
        ('Posterior','observe recomputes from prior on released snapshot','app/core/information_outcomes.py','ACTUAL_RUNTIME'),
        ('Routing','AgentRegistry.select_specialty -> route, before role call','app/core/hospital_llm_policy.py','RULE_RUNTIME_ADAPTIVE_TEST_ONLY'),
        ('Terminal','existing action/evidence validator and scheduler finalization','app/core/hospital_llm_policy.py','ACTUAL_RUNTIME'),
        ('Export','build_episode/export_episode/export_summary','app/locdi/episode_exporter.py','ACTUAL_RUNTIME'),
        ('Numeric replay','policy_episode -> SourceReplayEnvironment.acquire -> posterior -> scoring after terminal',
         'app/locdi/voi_new_cohort_evaluation.py','SEPARATE_DEVELOPMENT_SOURCE_REPLAY_NOT_HOSPITAL'),
        ('Dynamic','DynamicHorizonPolicy -> source-policy planner wrapper','app/locdi/dynamic_horizon.py','SOURCE_REPLAY_ONLY_NO_HOSPITAL_HOOK'),
    ]
    write(PUB/'actual_runtime_call_graph.json',{'nodes':[{'component':c,'call':f,'file':p,'status':s} for c,f,p,s in graph],
        'order_caution':'Hospital routing may precede role/VoI; do not impose a fictional Memory->Routing->VoI sequence',
        'C4_35_equals_live_HospitalCaseRuntime':False})
    write(PUB/'module_integration_status.json', {'runtime_graph':'actual_runtime_call_graph.json',
        'Memory':'RUNTIME_CONNECTED; Concept-Diverse offline opt-in only',
        'Routing':'Rule connected, Adaptive TEST_ONLY; no research utility',
        'VoI':'Hospital schema exists; Final M5 Source replay evidence not automatically Hospital-ready',
        'Dynamic':'Source replay evaluated; no existing Hospital hook',
        'memory_to_M5_posterior':'NO_MATCHED_MEMORY_ACTION',
        'full_framework_performance':'BLOCKED', 'technical_probe':'registered separate T0..T5'})
    write(PUB/'voi_final_evidence_summary.json', {'status':'FRESH_SYNTHETIC_GAIN_SUPPORTED',
        'source_files':['c4_35_v1/actual_evaluation_summary.json','c4_35_v1/fresh_cohort_paired_policy_comparison.json'],
        'actual':read(U35/'actual_evaluation_summary.json'),
        'comparisons':read(U35/'fresh_cohort_paired_policy_comparison.json'),
        'limitations':['Budget5 H3 below EIG/fixed','H3 Top3 below EIG/IG','NLL/Brier CI includes0',
            'Dynamic same asH3','H3-H1 CI includes0','quarantine selection','9of15referenceclasses',
            '400 legacy identity-only','same Synthea family','unreviewed clinical reference'],
        'C4_35_preserved':True,'new_scores_calculated':False})
    write(PUB/'integrated_evaluation_approval_status.json',{'technical_probe':'AUTHORIZED_BY_C4_36_OFFLINE_TEST_SCOPE',
        'actual_integrated_policy_evaluation':'NOT_APPROVED_AND_CONTRACT_BLOCKED',
        'prior_C4_35_approval_reused':False,'performance_experiment':'NOT_RUN'})
    write(PUB/'audit_completion.json',{'status':'READINESS_AUDIT_COMPLETE','technical_protocol':protocol['fingerprint'],
        'source_scoring':0,'new_LLM_generation':0,'protected_precheck':before})
    print(json.dumps({'status':'READINESS_AUDIT_COMPLETE','protocol':protocol['fingerprint'],
                      'official_ablation':'BLOCKED','Memory_M5_binding_count':compatibility['memory_action_count']}))


def read_json_text(text):
    return json.loads(text)


def run_test_only():
    protocol=read(PUB/'integrated_evaluation_protocol.json')
    for group in ['code_hashes','runtime_dependency_hashes','fixture_hashes']:
        if check(protocol[group])['changed_or_missing']:raise ValueError('FIXED_PROBE_CODE_CHANGED')
    if sha(DATA)!=protocol['source_sha256']:raise ValueError('SOURCE_CHANGED')
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from runtime_integration_support import options_for
    case=load_cases(DATA)[0]
    write(PR/'technical_started.json',{'at':datetime.now(timezone.utc).isoformat(),'protocol':protocol['fingerprint']})
    result, private=run_technical(case,protocol,options_for,output_root=PUB/'technical_runtime')
    write(PR/'technical_inputs_and_raw_results.json',private)
    write(PUB/'actual_integrated_execution.json',result)
    write(PUB/'development_ablation_results.json',{'status':'TECHNICAL_INTEGRATION_ONLY',
        'conditions':result['conditions'],'clinical_metrics':'NOT_EVALUATED','performance_metrics':None,
        'actual_source_performance_evaluation':'NOT_RUN_CONTRACT_BLOCKED',
        'official_B0_B4':False,'new_fitted_model':False,'actual_admitted35_policy_replay':0})
    if not result['all_technical_completed']:raise ValueError('TECHNICAL_INTEGRATION_PARTIAL')
    print(json.dumps({'status':'TECHNICAL_INTEGRATION_ONLY','conditions':len(result['conditions']),
                      'actual_source_quality_evaluations':0,'LLM_generation':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','run-test-only','execute-source'],nargs='?',default='audit')
    args=parser.parse_args()
    def guard(event, _):
        if event in {'socket.connect','socket.connect_ex','socket.bind','subprocess.Popen','os.system'}:
            raise RuntimeError('EXTERNAL_EXECUTION_DISABLED')
    sys.addaudithook(guard)
    if args.stage=='audit':audit()
    elif args.stage=='run-test-only':run_test_only()
    else:print(json.dumps(block_source_evaluation()));sys.exit(2)
