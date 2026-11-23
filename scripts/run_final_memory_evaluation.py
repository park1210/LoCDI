"""C4-39 final bounded B0/B2 development comparison using existing real Runner.

prepare is non-generating. No output-dependent selection or old result merging.
Execution requires a newly approved exact contract and a fresh no-resume run.
"""
import argparse,json
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.llm_setup import digest,load_setup
from app.locdi.dataset_loader import load_cases
from app.locdi.current_ablation_runner import make_contract,run_paired_ablation,reaggregate_trace
from app.locdi.current_ablation_llm_backend import static_preflight,make_er_backend
from app.locdi.new_cohort_execution import approval
PUB=ROOT/'outputs/research_evaluation/c4_39_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_39_v1'
DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
RUN='c4_39_memory_b0_b2_001'


def prepare():
    eligibility=read(PUB/'context_eligibility_final.json')
    if eligibility['status']!='COMPLETED':raise ValueError('INITIAL_PREFLIGHT_INCOMPLETE')
    # Fixed smallest original input ordinals, no performance/prediction selection.
    ordinals=[n for n in eligibility['B0_B2_common_eligible_ordinals'] if n!=30][:10]
    if len(ordinals)!=10:raise ValueError('TEN_COMMON_CASES_NOT_AVAILABLE')
    config=load_setup();static=static_preflight(config)
    if static['status']!='STATIC_CONTRACT_READY':raise ValueError('STATIC_LLM_SETUP_BLOCKED')
    cases=load_cases(DATA);panel=[];index={}
    for n in ordinals:
        case=cases[n-1];ref='FinalMemoryCase%03d'%n;index[ref]=n
        panel.append({'case_ref':ref,'patient_ref':'Patient'+digest({'dataset':sha(DATA),'patient':case['patient_id']})[:16],
            'execution_case':{'case_ref':ref},'reference':{
                'accepted_diagnoses':case['evaluation']['diagnosis_labels'],
                'accepted_departments':case['evaluation']['accepted_departments_source_policy']}})
    common=deepcopy(read(ROOT/'outputs/research_evaluation/c4_38_v1/er_execution_contract.json')['common'])
    common.update(dataset_sha256=sha(DATA),patient_split_sha256=digest([r['patient_ref'] for r in panel]),
        memory_sha256=digest([cases[n-1]['agent_visible']['past_visits'] for n in ordinals]),
        request_budget={'total':240,'per_episode':12,'retry':0,'repair':0},
        resource_budget={'wall_seconds':3600,'output_bytes':1024**3,'memory_bytes':4*1024**3})
    contract=make_contract(common,panel,conditions=('B0','B2'))
    write(PR/'memory_panel.json',panel);write(PR/'memory_case_index.json',index)
    code=['scripts/run_final_memory_evaluation.py','app/locdi/current_ablation_runner.py',
        'app/locdi/current_ablation_llm_backend.py','app/locdi/run_simulation.py',
        'app/locdi/case_adapter.py','app/locdi/episode_exporter.py','app/core/longitudinal_memory.py',
        'app/core/hospital_llm_policy.py','app/core/hospital_parallel_scheduler.py']
    paths=[DATA,ROOT/'configs/development_llm.json',ROOT/config['registry_path'],
        ROOT/config['patient_prompt'],PUB/'context_eligibility_final.json',PR/'memory_panel.json',PR/'memory_case_index.json']
    paths.extend(ROOT/b['prompt'] for b in config['agent_bindings'])
    protocol={'version':'c4_39.memory_actual.development.v1','run_id':RUN,
        'scope':'C4_38_CURRENT_ABLATION_DEVELOPMENT','approval_gate_scope_note':'reuse existing gate implementation; new C439 contract fingerprint, not old approval',
        'contract_fingerprint':contract['fingerprint'],'conditions':['B0','B2'],
        'registered_original_cases':50,'selected_ordinals':ordinals,'patients':10,'episodes':20,
        'selection':'first10 common byte-guard initial-input eligible ordinals, excluding previously run Case030; fixed before new outputs',
        'B1':'not in this supplementary panel; only1 common all3 initial case under unchanged guard',
        'Case030':'prior exposure preserved, not reused/merged/newly generated',
        'reason_for_ten_not_fifty':'bounded final development run 240 requests/one-hour monitored wall; sample feasibility not representative accuracy',
        'request_budget':common['request_budget'],'resource_budget':common['resource_budget'],
        'timeout_seconds':120,'output_tokens':1024,'context':32768,'margin':512,'seed':42,'temperature':0,
        'context_guard':'unchanged Native byte guard every turn, no exact-token substitute',
        'first_technical_error':'stop entire batch; retain original registered denominator and remaining NOT_ATTEMPTED',
        'resource_enforcement':'before/after episode and each generation reservation; single call/write can temporarily overshoot',
        'sequential_order':'all B0 input order, then all B2 same input order; latency order-confounded',
        'clinical_metrics':'NOT_EVALUATED','Source_agreement':'existing exact string scorer; independent annotation absent',
        'old_call_latency_seconds':[57.047,2.484,2.844,3.735,1.438,2.734],
        'estimate':'warm observed mean2.6478s/request on only Case030; 240 at that rate about10.6min plus overhead; not a guarantee',
        'uncapped_sum_of_request_timeouts_seconds':28800,'global_wall_cap_seconds':3600,
        'server_restart_download_warmup':False,'new_fitting':0,'retry':0,'repair':0,'resume':False,
        'input_hashes':{str(p.resolve()):sha(p) for p in paths},'code_hashes':{n:sha(ROOT/n) for n in code}}
    write(PUB/'memory_final_evaluation_protocol.json',protocol)
    write(PUB/'memory_execution_contract.json',contract)
    write(PUB/'memory_dry_run.json',run_paired_ablation(contract,panel))
    print(json.dumps({'status':'PREPARED_NOT_EXECUTED','ordinals':ordinals,'maximum_generation':240,'contract':contract['fingerprint']}))


def execute(path):
    protocol=read(PUB/'memory_final_evaluation_protocol.json');contract=read(PUB/'memory_execution_contract.json')
    if contract['fingerprint']!=protocol['contract_fingerprint']:raise ValueError('CONTRACT_CHANGED')
    for group in ['input_hashes','code_hashes']:
        if check(protocol[group])['changed_or_missing']:raise ValueError('SOURCE_CODE_CHANGED')
    record=read(Path(path));approval(record,protocol['scope'],contract['fingerprint'])
    cases=load_cases(DATA);index=read(PR/'memory_case_index.json');panel=read(PR/'memory_panel.json')
    backend=make_er_backend({r:cases[n-1] for r,n in index.items()},config=load_setup(),
        contract_fingerprint=contract['fingerprint'],checkpoint_root=PR/RUN/'requests',authorized=True)
    result=run_paired_ablation(contract,panel,run_one=backend,execute=True,approval_record=record,
        public_root=PUB/RUN,private_root=PR/RUN)
    print(json.dumps({'status':result['status'],'generation_count':result['generation_count'],
        'reaggregation':result['trace_reaggregation']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','execute','reaggregate'])
    p.add_argument('--approval-file');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='execute':
        if not a.approval_file:p.error('new exact approval required')
        execute(a.approval_file)
    else:print(json.dumps(reaggregate_trace(PR/RUN/'episodes.jsonl',read(PUB/'memory_execution_contract.json'),read(PR/'memory_panel.json'))))
