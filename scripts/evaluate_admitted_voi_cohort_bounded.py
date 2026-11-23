"""C4-35 bounded evaluation entrypoint. Two approvals remain mandatory."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import ctypes
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check
from scripts.review_new_cohort_independence import PUB,PR,P33,U33,P34,U34,verify
from app.locdi.llm_setup import digest
from app.locdi.voi_fresh_cohort_admission import run_approved
from app.locdi.dynamic_horizon_source_evaluation import PlannerMeter
from scripts.analyze_voi_sequential_policies import aubc_rows,paired_comparison


def private_bytes():
    if sys.platform=='win32':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
        c=Counters();c.cb=ctypes.sizeof(c)
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype=ctypes.c_int
        ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb)
        if not ok:raise ValueError('PROCESS_MEMORY_NOT_OBSERVABLE')
        return c.PrivateUsage
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


class Limits:
    def __init__(self,folder,clock=time.monotonic,memory=private_bytes):
        self.folder=Path(folder);self.clock=clock;self.memory=memory;self.start=clock()
        self.peak_private=0;self.peak_output=0
    def check(self):
        elapsed=self.clock()-self.start;mem=self.memory()
        total=sum(p.stat().st_size for p in self.folder.rglob('*') if p.is_file())
        self.peak_private=max(self.peak_private,mem);self.peak_output=max(self.peak_output,total)
        if elapsed>=3600:raise ValueError('EVALUATION_WALL_CAP')
        if mem>=4*1024**3:raise ValueError('EVALUATION_MEMORY_CAP')
        if total>=10*1024**3:raise ValueError('EVALUATION_STORAGE_CAP')


def prepare():
    verify();manifest=read(PR/'approved_admission.json');base=read(PUB/'approved_panel_evaluation_protocol.json')
    if manifest['status']!='ADMISSION_APPROVED_EVALUATION_NOT_APPROVED':raise ValueError('ADMISSION_A_REQUIRED')
    protocol={**base};protocol.pop('fingerprint',None)
    protocol['bounded_driver_sha256']=sha(Path(__file__))
    protocol['secondary_comparisons']=[['voi_h1','global_eig'],['voi_h3','random'],['voi_h3','fixed'],
        ['voi_h3','information_gain'],['voi_h2','voi_h1'],['voi_h3','voi_h1'],['dynamic','voi_h1'],['dynamic','voi_h3']]
    protocol['bounds']={'planner_calls':20000,'executed_states':50000000,'wall_seconds':3600,
        'process_private_bytes':4*1024**3,'run_output_bytes':10*1024**3,
        'enforcement':'pre/post every Core plan and every replay output; one bounded Core call/serialization can overshoot before stop'}
    protocol['status']='AWAITING_SEPARATE_EVALUATION_APPROVAL'
    protocol['fingerprint']=digest(protocol)
    write(PUB/'bounded_evaluation_protocol.json',protocol)
    print(json.dumps({'status':protocol['status'],'fingerprint':protocol['fingerprint'],
        'slots':manifest['visits']*54,'paths':manifest['visits']*240,'policy_evaluation':0}))


def execute(approval_file,allow=False):
    verify()
    if not allow:
        print(json.dumps({'status':'DRY_RUN','new_policy_evaluation':0,'requires':'Approval A, bounded prepare, separate Approval B'}));return
    protocol=read(PUB/'bounded_evaluation_protocol.json')
    if protocol['fingerprint']!=digest({k:v for k,v in protocol.items() if k!='fingerprint'}):raise ValueError('EVALUATION_PROTOCOL_HASH')
    if protocol['bounded_driver_sha256']!=sha(Path(__file__)):raise ValueError('BOUNDED_DRIVER_CHANGED')
    manifest=read(PR/'approved_admission.json');model=read(P33/'final_m5_model.json')
    if sha(PR/'approved_admission.json')!=protocol['admission_manifest_sha256'] or sha(P33/'final_m5_model.json')!=protocol['model_sha256']:
        raise ValueError('PANEL_OR_MODEL_CHANGED')
    from app.locdi.new_cohort_execution import approval
    record=read(Path(approval_file)) if approval_file else None
    approval(record,'C4_35_FROZEN_POLICY_EVALUATION',protocol['fingerprint'])
    if check(read(P34/'actual_generation_receipt.json')['raw_files'])['changed_or_missing']:raise ValueError('RAW_CHANGED')
    run=PR/'evaluation_run';run.mkdir(exist_ok=False)
    write(run/'approval.json',record);write(run/'started.json',{'at':datetime.now(timezone.utc).isoformat(),'protocol_fingerprint':protocol['fingerprint']})
    guard=Limits(run)
    class GuardedPlanner(PlannerMeter):
        def __call__(self,*args,**kwargs):
            guard.check();result=super().__call__(*args,**kwargs);guard.check();return result
    try:
        with (run/'trace.jsonl').open('x',encoding='utf-8') as f:
            def sink(row):
                guard.check();f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();guard.check()
            with patch('app.locdi.new_cohort_execution.PlannerMeter',new=GuardedPlanner):
                records,membership,resources,results=run_approved(manifest,model,protocol,record,sink)
        write(run/'score_rows.json',records);write(run/'membership.json',membership);guard.check()
        comparisons=results['comparison']
        for left,right in protocol['secondary_comparisons']:
            comparisons[left+'__'+right]=paired_comparison(aubc_rows(records,left),aubc_rows(records,right),membership,repetitions=2000,seed=42)
            guard.check()
        primary=comparisons['voi_h3__global_eig']['patient_macro']['top1']
        basic=any(comparisons['voi_h3__'+p]['patient_macro']['top1']['ci95'][0]>0 for p in ('random','fixed'))
        decision='FRESH_SYNTHETIC_PRIMARY_GAIN' if primary['difference']>0 and primary['ci95'][0]>0 else (
            'BASIC_BASELINE_GAIN_ONLY' if basic else 'MIXED_OR_INCONCLUSIVE' if primary['ci95'][0]<=0<=primary['ci95'][1] else 'NO_REPRODUCED_GAIN')
        write(PUB/'fresh_cohort_native_policy_metrics.json',{**results,'decision':decision,'evidence':'new-seed same-generator synthetic Source agreement, not clinical validation'})
        write(PUB/'fresh_cohort_primary_comparison.json',comparisons['voi_h3__global_eig'])
        write(PUB/'fresh_cohort_paired_policy_comparison.json',comparisons)
        write(PUB/'fresh_cohort_patient_bootstrap.json',{'comparisons':comparisons,'repetitions':2000,'seed':42,'unit':'patient cluster'})
        write(PUB/'fresh_cohort_budget_curves.json',results['metrics'])
        write(PUB/'fresh_cohort_action_stop_analysis.json',{p:{str(k):{'actions':results['metrics'][p][str(k)]['patient_macro']['actual_actions'],
            'stop':results['metrics'][p][str(k)]['patient_macro']['stop']} for k in range(6)} for p in results['metrics']})
        horizons=Counter()
        with (run/'trace.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                for d in r.get('decisions',[]):horizons[str((r['policy'],d.get('effective_horizon')))]+=1
        write(PUB/'fresh_cohort_horizon_comparison.json',dict(horizons))
        ordered=sorted(manifest['panel'],key=lambda r:(r['patient_id'],r['visit_id']))
        class_ord={d:i for i,d in enumerate(model['diagnoses'],1)}
        diagnosis_groups={}
        from app.locdi.voi_sequential_policy_evaluation import mean_metrics
        for d,ordinal in class_ord.items():
            ids={i for i,v in enumerate(ordered,1) if v['diagnosis']==d}
            if ids:
                diagnosis_groups[str(ordinal)]={p:{str(k):mean_metrics([r for r in records if r['scenario_ordinal'] in ids and r['policy']==p and r['budget']==k],membership)
                    for k in range(6)} for p in results['metrics']}
                for table in diagnosis_groups[str(ordinal)].values():
                    for metric in table.values():metric['accuracy_denominator']='all admitted Source subgroup visits; no failed-slot imputation'
        write(PUB/'fresh_cohort_diagnosis_subgroups.json',{'class_ordinals_follow_frozen_catalog':True,'groups':diagnosis_groups,
            'rare_class_limits':'descriptive only; source labels not clinical Gold'})
        write(PUB/'fresh_cohort_resource_usage.json',{'planner':resources,'peak_sampled_private_bytes':guard.peak_private,
            'peak_sampled_run_bytes':guard.peak_output,'wall_seconds':guard.clock()-guard.start,'scope':'instrumented shared-cache run, not isolated speed comparison'})
        old_path=ROOT/'outputs/research_evaluation/c4_31_v1/budget_curve_metrics.json'
        write(PUB/'development_vs_fresh_cohort_comparison.json',{'fresh_patients':manifest['patients'],'fresh_visits':manifest['visits'],
            'development_reference':'C4-31/32 old63 patients/92 visits; different model folds and cohort, descriptive only',
            'stored_development_metrics':read(old_path) if old_path.exists() else 'MISSING',
            'development_metric_sha256':sha(old_path) if old_path.exists() else None,
            'fresh_primary':primary,'clinical_superiority':'NOT_ASSERTED'})
        files={p.relative_to(ROOT).as_posix():sha(p) for p in run.iterdir() if p.is_file()}
        write(run/'private_index.json',{'files':files,'fingerprint':digest(files)})
        outputs={p.relative_to(ROOT).as_posix():sha(p) for p in PUB.glob('fresh_*.json')}
        outputs[(run/'private_index.json').relative_to(ROOT).as_posix()]=sha(run/'private_index.json')
        write(PUB/'fresh_evaluation_completion_manifest.json',{'files':outputs,'fingerprint':digest(outputs),'status':decision})
    except Exception as e:
        write(PUB/'fresh_execution_partial.json',{'status':'EXECUTION_PARTIAL','error_type':type(e).__name__,
            'partial_private_output_preserved':True,'no_failed_patient_removed':True});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run'],nargs='?',default='run');p.add_argument('--execute',action='store_true');p.add_argument('--approval-file');a=p.parse_args()
    prepare() if a.stage=='prepare' else execute(a.approval_file,a.execute)
