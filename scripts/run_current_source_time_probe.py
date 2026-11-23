"""C4-38 one newly approved five-visit Source-time technical run; no LLM."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from app.locdi.llm_setup import digest
from app.locdi.new_cohort_execution import approval
from app.locdi.source_time_hospital_runtime import run_source_time_visit,public_visit,validate_development_encoding
from app.locdi.frozen_m5_runtime_adapter import M5_SHA
from app.locdi.dynamic_horizon_source_evaluation import run_episode,PlannerMeter
from app.core.voi_planning import plan_voi
from app.locdi.source_runtime_comparison import compare_episode,ComparisonMismatch

PUB=ROOT/'outputs/research_evaluation/c4_38_v1'
PR=ROOT/'outputs/private/research_evaluation/c4_38_v1'
P37=ROOT/'outputs/private/research_evaluation/c4_37_v1'
M5=ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json'
SCOPE='C4_38_SOURCE_TIME_H2_FIVE_ONLY'


def reference_episode(model,values,guard,plans,*,budget=5):
    """Original reference runner requires the complete PlannerMeter protocol.

    A plain callable lacks snapshot()/state_history. Preserve the original
    meter and capture its actual computed plan via its existing sink hook.
    """
    class GuardedMeter(PlannerMeter):
        def __call__(self,*args,**kwargs):
            guard();result=super().__call__(*args,**kwargs);guard();return result
    meter=GuardedMeter(cached=False,sink=lambda row:plans.append(row['plan']),label='c4_38_reference_h2')
    return run_episode(model,values,'voi_h2',budget,meter)


def cases():
    admission=read(ROOT/'outputs/private/research_evaluation/c4_35_v1/approved_admission.json')
    panel=sorted(admission['panel'],key=lambda r:(r['patient_id'],r['visit_id']))[:5]
    import json
    memory=[]
    with (P37/'new_observation_memory_projection.jsonl').open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=5:break
            row=json.loads(line);assert row['case_ordinal']==i+1;memory.append(row['past_visits'])
    return panel,memory


def prepare(run_id='c4_38_source_time_001'):
    if not run_id.startswith(('c4_38_source_time_','c4_39_source_time_')) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in run_id):
        raise ValueError('INVALID_RUN_ID')
    if sha(M5)!=M5_SHA:raise ValueError('FROZEN_M5_CHANGED')
    panel,memory=cases();model=read(M5)
    import json
    with (P37/'engine_private_candidates.jsonl').open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=5:break
            row=json.loads(line);validate_development_encoding(model,panel[i],row['engine_candidate'])
    inputs=[M5,ROOT/'outputs/private/research_evaluation/c4_35_v1/approved_admission.json',
        P37/'new_observation_memory_projection.jsonl',P37/'engine_private_candidates.jsonl',
        ROOT/'outputs/research_evaluation/c4_37_v1/completion_manifest.json']
    code=['app/locdi/source_time_hospital_runtime.py','scripts/run_current_source_time_probe.py',
        'app/locdi/frozen_m5_runtime_adapter.py','app/core/information_value.py','app/core/information_outcomes.py',
        'app/core/diagnostic_belief.py','app/core/voi_planning.py','app/core/longitudinal_memory.py',
        'app/locdi/dynamic_horizon_source_evaluation.py','app/locdi/source_runtime_comparison.py']
    protocol={'version':'c4_38.source_time_h2.five.v1','scope':SCOPE,'run_id':run_id,
        'source':'existing C4_37 first5, same patient_id/visit_id ordering, development-exposed',
        'panel_fingerprint':digest([public_visit(v) for v in panel]),'memory_fingerprint':digest(memory),
        'horizon':2,'information_budget_per_visit':5,'runtime_visits':5,'reference_computations':5,
        'max_total_planner_calls':60,'max_total_states':1200000,'Core_max_states_per_call':20000,
        'wall_seconds':120,'private_memory_bytes':512*1024**2,'output_bytes':1024**3,
        'LLM_generation':0,'fitting':0,'retries':0,'resume':False,
        'comparison':{'absolute_tolerance':1e-12,'Action_ID_sets_STOP_outcomes':'EXACT',
            'values':'ID-keyed; duplicate IDs blocked before dict; raw policy order unchanged',
            'extra_terminal':'budget0 empty-information-menu terminal only'},
        'mode':'SOURCE_TIME_DEVELOPMENT not full Hospital resource simulation',
        'terminal':'algorithmic choice separate from unresolved Hospital; Source reference only AFTER execution',
        'clinical_evaluation':False,'artifact_gate':'existing clinical/research gate unchanged BLOCKED',
        'input_hashes':{str(p.resolve()):sha(p) for p in inputs},'code_hashes':{n:sha(ROOT/n) for n in code}}
    protocol['fingerprint']=digest(protocol);write(PUB/'source_time_execution_protocol.json',protocol)
    write(PUB/'source_time_approval_status.json',{'status':'AWAITING_NEW_USER_APPROVAL','scope':SCOPE,
        'protocol_fingerprint':protocol['fingerprint'],'Generation':0})
    print({'status':'PREPARED_NOT_EXECUTED','fingerprint':protocol['fingerprint']})


def execute(path):
    import json
    protocol=read(PUB/'source_time_execution_protocol.json')
    if digest({k:v for k,v in protocol.items() if k!='fingerprint'})!=protocol['fingerprint']:raise ValueError('PROTOCOL_CHANGED')
    for group in ['input_hashes','code_hashes']:
        if check(protocol[group])['changed_or_missing']:raise ValueError('SOURCE_OR_CODE_CHANGED')
    record=read(Path(path));approval(record,SCOPE,protocol['fingerprint'])
    folder=PR/'source_time_run';folder.mkdir(exist_ok=False)
    write(folder/'approval.json',record);write(folder/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'protocol':protocol['fingerprint']})
    panel,memory=cases();model=read(M5);start=time.monotonic();calls=states=0;rows=[];runtime_count=reference_count=0
    from scripts.evaluate_admitted_voi_cohort_bounded import private_bytes
    def guard():
        if time.monotonic()-start>=120 or private_bytes()>=protocol['private_memory_bytes']:raise ValueError('RESOURCE_CAP')
        if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())>=protocol['output_bytes']:raise ValueError('OUTPUT_CAP')
    try:
        for ordinal,(visit,past) in enumerate(zip(panel,memory),1):
            guard();actual=run_source_time_visit(model,public_visit(visit),past,max_actions=5);guard()
            write(folder/('Case%03d_runtime_before_comparison.json'%ordinal),actual)
            runtime_count+=1
            reference_plans=[]
            reference=reference_episode(model,visit['values'],guard,reference_plans)
            write(folder/('Case%03d_reference_before_comparison.json'%ordinal),{'episode':reference,'plans':reference_plans})
            reference_count+=1
            calls+=actual['planner_calls']+len(reference_plans)
            states+=actual['evaluated_states']+sum(p['evaluated_states'] for p in reference_plans)
            if calls>60 or states>1200000:raise ValueError('PLANNER_CAP')
            audit={}
            try:compare_episode(actual,reference,reference_plans,model,budget=5,audit=audit)
            except ComparisonMismatch as e:
                write(folder/'first_mismatch.json',{'ordinal':ordinal,'code':e.code,'detail':e.detail,'partial_comparison':audit})
                raise
            finally:
                for key in ['action_identity_comparison','posterior_comparison','decision_comparison']:
                    write(folder/('Case%03d_%s.json'%(ordinal,key)),audit.get(key,[]))
            write(folder/('Case%03d_comparison.json'%ordinal),audit)
            diffs=[audit['maximum_final_posterior_difference'],audit['maximum_terminal_value_difference']]
            for r in audit['action_identity_comparison']:
                diffs.extend(r[k] for k in r if k.startswith('maximum_'))
            for r in audit['posterior_comparison']:diffs.extend([r['before_max_abs'],r['after_max_abs']])
            got=[e for e in actual['actions'] if e.get('action_id') is not None]
            # Reference enters only the post-execution private scorer.
            agreement=actual['algorithmic_choice']==visit['diagnosis']
            write(folder/('Case%03d.json'%ordinal),{'actual':actual,'reference':reference,'source_reference':visit['diagnosis'],
                'source_reference_agreement':agreement,'maximum_numeric_difference':max(diffs,default=0)})
            rows.append({'ordinal':ordinal,'status':'completed','maximum_numeric_difference':max(diffs,default=0),
                'actions':len(got),'hospital_unresolved':True,'hospital_supported':False,'Source_reference_agreement':agreement})
        result={'status':'SOURCE_TIME_HOSPITAL_TECHNICAL_COMPLETED','cases':rows,'planner_calls':calls,'evaluated_states':states,
            'runtime_completed':runtime_count,'reference_completed':reference_count,'compared_cases':len(rows),
            'wall_seconds':time.monotonic()-start,'clinical_metrics':'NOT_EVALUATED','LLM_generation':0,
            'not_official_B4':True,'full_resource_simulation':False}
        write(PUB/'source_time_actual_results.json',result)
        write(folder/'private_index.json',{'files':{p.relative_to(ROOT).as_posix():sha(p) for p in folder.iterdir() if p.is_file()}})
        print(json.dumps(result))
    except Exception as e:
        write(PUB/'source_time_actual_partial.json',{'status':'EXECUTION_PARTIAL','error_type':type(e).__name__,
            'error_code':str(e),'runtime_completed':runtime_count,'reference_completed':reference_count,
            'planner_calls':calls,'evaluated_states':states,'completed_cases':rows,
            'partial_private_preserved':True,'no_resume':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','execute'],nargs='?',default='prepare');p.add_argument('--approval-file')
    p.add_argument('--namespace',default='c4_38_v1');p.add_argument('--run-id',default='c4_38_source_time_001');a=p.parse_args()
    if not a.namespace.startswith(('c4_38_','c4_39_')) or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.namespace):p.error('invalid fresh namespace')
    if a.namespace.startswith('c4_39_'):SCOPE='C4_39_SOURCE_TIME_H2_FIVE_ONLY'
    PUB=ROOT/'outputs/research_evaluation'/a.namespace;PR=ROOT/'outputs/private/research_evaluation'/a.namespace
    if a.stage=='prepare':prepare(a.run_id)
    elif a.approval_file:execute(a.approval_file)
    else:p.error('new pinned explicit approval required')
