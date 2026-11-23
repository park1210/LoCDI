"""Exclusive C4-38 preparation exports. Never executes a Source or model policy."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.llm_setup import digest
from app.locdi.current_ablation_runner import features_for
PUB=ROOT/'outputs/research_evaluation/c4_38_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_38_v1'


def text(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def table(name,columns,rows):
    with (PUB/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def main():
    er=read(PUB/'er_execution_contract.json'); ep=read(PUB/'er_execution_protocol.json')
    sp=read(PUB/'source_time_execution_protocol.json'); tests=read(PUB/'test_results.json')
    assert not tests['failures'] and not tests['errors']
    assert check(ep['code_hashes'])['changed_or_missing']==0
    assert check(sp['code_hashes'])['changed_or_missing']==0
    for name in ['routing_utility_source_inventory.json','routing_utility_eligibility.json','routing_utility_provenance.json']:
        write(PUB/name,read(PUB/'track_a'/name))
    for name in ['final_formula_code_mapping.json','final_formula_code_mapping.md']:
        value=(PUB/'track_c'/name).read_text(encoding='utf-8')
        text(PUB/name,value)
    changes=['experiments/run_locdi_ablation.py','app/locdi/source_time_hospital_runtime.py',
        'app/locdi/current_ablation_runner.py','app/locdi/current_ablation_llm_backend.py',
        'scripts/run_current_source_time_probe.py','scripts/run_current_er_ablation.py',
        'tests/locdi/test_source_time_hospital_runtime.py','tests/locdi/test_current_ablation_runner.py',
        'tests/locdi/test_current_ablation_llm_backend.py','tests/locdi/test_current_ablation_interface.py',
        'scripts/finalize_c4_38_preparation.py']
    code={p:sha(ROOT/p) for p in changes}
    write(PUB/'final_implementation_manifest.json',{'changed_or_added_code':code,
        'existing_modified':['experiments/run_locdi_ablation.py'],'Core_math_modified':[],
        'old_wrapper_snapshot':'outputs/private/research_evaluation/c4_38_v1/interface_baseline/run_locdi_ablation.py',
        'official_JSON_modified':False,'final_M5_modified':False,'clinical_gate_modified':False})
    write(PUB/'final_research_contract.json',{'version':'c4_38.current_completion.v1',
        'original_manuscript_sha256':'d28e51965188a22b6a87d9cb8a52d1dfde49d942400f31ef716badafe7eebc99',
        'required':['longitudinal Memory context','selective word-overlap Top5','Adaptive Routing',
                    'Bayesian belief','finite-horizon VoI H2','original B0-B4'],
        'optional_extensions_not_retuned':['Concept-Diverse','H3','Dynamic Horizon'],
        'formal_B0_B4':'BLOCKED','Source_time_protocol_fingerprint':sp['fingerprint'],
        'ER_partial_contract_fingerprint':er['fingerprint'],'new_clinical_gold':0,'fitting':0,
        'execution_approval':'AWAITING_NEW_USER_APPROVAL','LLM_generation':0})
    write(PUB/'final_m5_runtime_integration.json',{'implementation':'IMPLEMENTED_CONTRACT_TESTED',
        'model_sha256':'7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c',
        'encoding':'c4_30_source_discrete_v2','encoding_tag_changed':False,
        'binding_reused':{'patients':35,'visits':71,'actions':568,'bound':568,'past_Memory_instances':123960},
        'actual_runtime':'HospitalCaseRuntime + InformationPolicy.propose + original plan_voi',
        'scheduler':'separate SourceTimeScheduler, recorded-time proxy, not Hospital resource simulation',
        'live_or_Source5_execution':'NOT_RUN_AWAITING_NEW_APPROVAL','source5_protocol':sp['fingerprint'],
        'terminal':{'algorithmic_choice':'terminal_choice(belief)',
            'supported_diagnosis':None,'unresolved':'existing allow_unresolved branch',
            'Source_agreement':'after execution only, not objective Evidence'},
        'memory':'same patient past Observation projection, Top5 context only; no likelihood/action added',
        'existing_research_artifact_gate':'BLOCKED_UNCHANGED'})
    write(PUB/'runtime_interface_change_log.json',{'new_interfaces':[
        {'path':'app/locdi/source_time_hospital_runtime.py','functions':['public_visit','validate_development_encoding',
            'SourceBoundOutcomes.observe','SourceTimeScheduler.apply','run_source_time_visit'],
         'purpose':'raw/released exact identity, proxy Source-time, separate terminal semantics'},
        {'path':'app/locdi/current_ablation_runner.py','functions':['make_contract','run_paired_ablation','reaggregate_trace'],
         'purpose':'same-panel pinned comparison, budgets, exclusive writes and independent aggregation'},
        {'path':'app/locdi/current_ablation_llm_backend.py','functions':['safe_input_bound','make_er_backend'],
         'purpose':'existing real Runner Native transport, evidence schema, identity/context/budget per request'}],
        'existing_interface':{'path':'experiments/run_locdi_ablation.py','function':'run_ablation',
           'before':'unconditionally NotImplementedError after validation',
           'after':'pinned official contract plus unchanged research artifact gate, default dry-run'},
        'math_Bellman_Posterior_Memory_Routing_modified':False,'original_defaults_modified':False})
    write(PUB/'runtime_replay_equivalence.json',{'actual_Source5_status':'NOT_RUN_AWAITING_NEW_APPROVAL',
        'tolerance':1e-12,'fixture_tests_module':'tests/locdi/test_source_time_hospital_runtime.py',
        'fixture_tests':9,'passed':9,'evidence_level':'TEST_ONLY_CONTRACT_NOT_ACTUAL_SOURCE_PERFORMANCE',
        'comparison':['prior','menus','released raw/category','posterior','T','Q','VoI','STOP','terminal_choice'],
        'old_trace_Q':'NOT_RECORDED; not retroactively reconstructed as an old measurement',
        'approved_future_computation':'new current Source runtime and independently captured original SourceReplay planner',
        'actual_max_difference':None})
    write(PUB/'final_ablation_contract.json',{'official_features':{c:features_for(c) for c in ['B0','B1','B2','B3','B4']},
        'H2':2,'official_conditions_changed':False,'original_configs_ready_changed':False,
        'official_shared_missing':['eligible current Routing Utility','reviewed clinical Reference/visibility',
            'shared all-condition Action binding/Registry','same-panel full-history context feasibility'],
        'actual_prepared_development_contract':er,'official_research_evaluation_ready':False})
    write(PUB/'final_source_panel_eligibility.json',{'Original50':{'registered':50,'preserved':50,
        'common_initial_input_guard_eligible_B0_B1_B2':1,'ordinal':[30],
        'reason':'untruncated FullHistory conservative UTF8 upper bound includes messages/schema/framing',
        'not_proven_tokenizer_overflow':True,'selection_before_new_outputs':True,
        'later_turn_fit':'NOT_GUARANTEED_PER_REQUEST_CHECK','Memory_annotations':'independent review unavailable'},
        'C4_35':{'patients':35,'visits':71,'development_exposed':True,'fresh_holdout_claim':False,
            'M5_binding':'568/568 reused','Source_time_probe_selected':5,'all_B0_B4':'BLOCKED'},
        'cross_panel_matched_ablation':False,'source_merging':False,
        'Clinical_Gold':'NOT_APPROVED','Source_reference':'separate exploratory agreement only'})
    write(PUB/'final_ablation_runner_readiness.json',{'Harness':'IMPLEMENTED_CONTRACT_TESTED',
        'existing_wrapper_NotImplementedError':'REPLACED_BY_PINNED_INTERFACE',
        'B0_B1_B2_development':{'status':'CONTRACT_READY_REQUIRES_EXECUTION_APPROVAL',
            'cases':1,'episodes':3,'run_id':ep['run_id'],'maximum_generation':36},
        'B3_B4':'ROUTING_UTILITY_BLOCKED','official_B0_B4':'BLOCKED',
        'Source_time5':'AWAITING_NEW_EXECUTION_APPROVAL','clinical_metrics':'NOT_EVALUATED'})
    write(PUB/'actual_execution_approval_status.json',{'status':'AWAITING_NEW_USER_APPROVAL',
        'reason':'C4-38 sections10/14 require new actual generation and approval-bound Source evaluation approval',
        'prior_approvals':'old distinct runs/scopes not reused','single_consolidated_request':True,
        'independent_contracts':[{'scope':sp['scope'],'plan_fingerprint':sp['fingerprint'],'generation':0},
             {'scope':ep['scope'],'plan_fingerprint':er['fingerprint'],'maximum_generation':36}],
        'approved':False,'server_identity_or_context_current':'NOT_CHECKED','actual_generation':0})
    write(PUB/'final_ablation_execution_manifest.json',{'status':'NOT_RUN_AWAITING_NEW_APPROVAL',
        'planned_source_probe_visits':5,'planned_development_ER_cases':1,'planned_ER_episodes':3,
        'executed_current_episodes':0,'LLM_generation':0,'fitting':0,'new_clinical_verdict':0,
        'actual_run_ids':[],'reserved_run_ids':[sp['run_id'],ep['run_id']],
        'B3_B4':'BLOCKED','old_results_used_as_new_measurements':False})
    conditions=['B0','B1','B2','B3','B4']
    for filename,metric in [('final_diagnosis_results.csv','diagnosis_Source_agreement'),
        ('final_department_results.csv','final_department_Source_agreement'),
        ('final_memory_results.csv','annotation_precision_recall'),
        ('final_action_results.csv','executed_Ask_Test_Consult_Diagnose'),
        ('final_voi_results.csv','actual_H2_VoI_STOP')]:
        rows=[]
        for c in conditions:
            rows.append({'condition':c,'metric':metric,'status':'NOT_RUN' if c in conditions[:3] else 'BLOCKED',
                'prepared_panel_cases':1 if c in conditions[:3] else '', 'executed_cases':0,'value':'',
                'reason':'NEW_APPROVAL_PENDING' if c in conditions[:3] else 'ROUTING_UTILITY_BLOCKED',
                'clinical_metrics':'NOT_EVALUATED'})
        table(filename,list(rows[0]),rows)
    rows=[]
    for i,c in enumerate(conditions):
        rows.append({'condition':c,'historical_diagnosis_percent':[38,52,44,48,90][i],
            'historical_department_percent':[44,58,54,46,94][i],
            'historical_mean_actions':[2.90,3.14,2.60,2.12,3.52][i],
            'current_diagnosis':'','current_department':'','current_actions':'',
            'current_denominator':0,'status':'NOT_RUN' if i<3 else 'BLOCKED','comparison':'NOT_COMPARABLE',
            'historical_run_verified':'NOT_ESTABLISHED'})
    table('thesis_table2_recalculated.csv',list(rows[0]),rows)
    table('thesis_table3_recalculated.csv',['metric','historical_subset','current_eligible_reviewed_cases','value','status'],[
        {'metric':metric,'historical_subset':20,'current_eligible_reviewed_cases':0,'value':'','status':'NOT_EVALUATED_NO_INDEPENDENT_ANNOTATION'}
        for metric in ['annotation_precision','annotation_recall','stale','conflict']])
    historical='''# Historical vs current

The original three-page PDF reports B0-B4 diagnosis38/52/44/48/90%, department44/58/54/46/94%, mean actions2.90/3.14/2.60/2.12/3.52. These are historical manuscript values, not C4-38 measurements. The exact historical run has not been established. The bounded old-run inquiry is reused; old49/50 B4, TopK15/router differences do not identify the paper run.

C4-38 has no new actual Source or LLM execution before the new approval. Current numerical results remain null/NOT_RUN. The prepared one-case B0/B1/B2 comparison is a development input-size feasibility subset, not a replacement50-case clinical table. B3/B4 remain blocked. The manuscript Table3 annotation subset20 must not be replaced by Source-identity precision50. C4-35 AUBC is a separate sequential Source-agreement result, not B4 final diagnostic accuracy.

Data, candidate universe, model binding, Router, Memory, action timing/source visibility, probability artifact, scoring/reference and actual attempted denominator must be compared before attributing any future numerical difference to reproducibility. No old result or manuscript was edited.
'''
    text(PUB/'historical_vs_current_results.md',historical)
    commands='''# C4-38 exact commands

Preparation already completed (exclusive files; do not rerun into the completed namespace):
`python -B scripts/run_current_source_time_probe.py prepare`
`python -B scripts/run_current_er_ablation.py prepare`

Only AFTER new explicit approvals with each exact scope/fingerprint:
`python -B scripts/run_current_source_time_probe.py execute --approval-file outputs/private/research_evaluation/c4_38_v1/source_time_execution_approval.json`
`python -B scripts/run_current_er_ablation.py execute --approval-file outputs/private/research_evaluation/c4_38_v1/er_execution_approval.json`

Approval files currently DO NOT exist. They may record only a real subsequent user decision, not an invented approval. No-resume/no-overwrite. Run IDs c4_38_source_time_001 and c4_38_er_b0_b2_001 are reserved, NOT executed. The two independent scopes are listed in actual_execution_approval_status.json.

After an ER trace exists, model-free independent aggregation:
`python -B scripts/run_current_er_ablation.py reaggregate`

Related regression module list and actual count are in test_results.json. Full prior C4-36 six-condition evaluation was not repeated. Native requests use existing Qwen/Digest/ER Prompt/Registry, temperature0 seed42 output1024 Context32768 timeout120 retry0 repair0. First fixed task loads the model; /api/ps actual Context and Digest must pass before any later request. No warmup/server restart/download. B1 is never truncated. First technical error stops the entire prepared paired run and retains partial records.
'''
    text(PUB/'actual_execution_commands.md',commands)
    text(PUB/'remaining_blockers.md','''# Remaining blockers

1. New C4-38 execution approval is missing for the fixed Source-time5 protocol and separate limited ER comparison. Old approvals are not reused.
2. ROUTING_UTILITY_BLOCKED: zero current eligible complete scorable Train groups, no independently reviewed Reference/decision-time correctness, legacy binding mismatch. No fitting. Existing75 cannot be made complete by converting53 unresolved tofalse. Review existing Source first; a future arithmetic minimum2Train+1Validation by5agents=15requests needs separate reviewed data and approval, and is not statistical sufficiency.
3. Original50 full-history strict context guard admits only ordinal30 for all B0/B1/B2 initial requests. This is a conservative byte-bound operational limit, not an exact tokenizer impossibility proof. A safe long-context strategy or a separately approved differently specified experiment is needed for full50; silently shortening B1 changes the experiment.
4. All-condition shared clinical Reference/visibility, Specialty bindings/Utility and shared Action/Source contract are not ready. No mixed-panel B0-B4 table and no arbitrary Utility.
5. M5 has no supports_diagnoses outcomes. Algorithmic terminal choice cannot become a supported Hospital diagnosis. Source-time development does not establish hospital resource turnaround/cost or clinical validity.
6. Independent Memory relevance/stale/conflict annotations are not ready. Source identity and concept coverage cannot replace manuscript annotation precision/recall.
7. Historical paper run lineage remains unestablished; current seeded LLM results cannot be promised bitwise reproducible.
''')
    env=read(PR/'environment.json')
    write(PUB/'final_reproducibility_manifest.json',{'status':'CURRENT_PREPARATION_PINNED_ACTUAL_EXECUTION_PENDING',
        'code_sha256':code,'Source_protocol':sp['fingerprint'],'ER_contract':er['fingerprint'],
        'data_sha256':er['common']['dataset_sha256'],'memory_sha256':er['common']['memory_sha256'],
        'model_bindings':er['common']['model_bindings'],'registry_sha256':er['common']['registry_sha256'],
        'prompt_sha256':er['common']['prompt_sha256'],'Utility':None,
        'Final_M5_sha256':'7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c',
        'python':'3.9.7','package_versions':env['packages'],'environment_private_sha256':sha(PR/'environment.json'),
        'seed':42,'commands':'actual_execution_commands.md','actual_run_ids':[],
        'scoring_code_sha256':sha(ROOT/'app/locdi/current_ablation_runner.py'),
        'aggregation':'locdi.source_agreement_and_workflow.c4_38.v1',
        'trace_reaggregation':'EXACT in TEST_ONLY; actual trace NOT_RUN',
        'LLM_bitwise_reproduction_guaranteed':False})
    text(PUB/'final_research_completion_status.md','''# C4-38 current status

Implementation and related regression are complete. New actual Source/LLM execution awaits a consolidated user decision. This is PARTIAL / EXECUTION_APPROVAL_PENDING, not research completion.

A: ROUTING_UTILITY_BLOCKED. B: Source-time Hospital interface IMPLEMENTED_CONTRACT_TESTED, actual5 NOT_RUN. C: original unnumbered manuscript equations mapped to current exact functions/line ranges/SHA. D: pinned paired Harness implemented, official wrapper connected, one-case B0/B1/B2 Native development plan prepared; B3/B4 and official research BLOCKED.

Actual new LLM Generation0; fitting0; new clinical verdict0; old result edits0; mathematical Core changes0. Test and preservation counts are in test_results.json and verification.json. Historical/current CSV cells remain empty where no measurement exists.
''')
    print(json.dumps({'status':'PREPARATION_EXPORTS_CREATED','required_files_created':29,'generation':0}))


if __name__=='__main__':main()
