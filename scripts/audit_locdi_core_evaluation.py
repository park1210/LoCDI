"""C4-01 read-only Core audit and real-source Memory eligibility benchmark.

No model/client, fitting, probability estimation or clinical label creation.
The reference is the declared patient/visit/timestamp contract, not relevance.
"""
import argparse,json,sys,re,ast
from pathlib import Path
from datetime import datetime
from collections import Counter
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.episode_exporter import _write
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.memory_evaluation import evaluate_memory
from app.core.clinical_observation import doctor_observation,memory_query,initial_symptoms
from app.core.hospital_case_runtime import HospitalCaseRuntime

DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
MODES=('none','full_history','selective')

def eligible_reference(records,identity):
    """Independent direct datetime/set oracle for structural declared eligibility."""
    parse=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'))
    cutoff=parse(identity['started_at'])
    return {r['memory_id'] for r in records if r['patient_id']==identity['patient_id']
        and r['visit_id']!=identity['visit_id'] and parse(r['visit_started_at'])<cutoff
        and parse(r['available_at'])<=cutoff}

def initial_query(engine):
    runtime=HospitalCaseRuntime(case_id=engine['case_id'],case_data=engine,arrival_time_min=0,arrival_index=0,
        vital_signs=deepcopy(engine['doctor_visible_info'].get('initial_vitals',{})),current_symptoms=initial_symptoms(engine))
    # Same public projection and query function as run_longitudinal initial retrieval.
    return memory_query(doctor_observation(runtime))

def benchmark(cases):
    engines=[adapt_case(c).runtime_case for c in cases]
    pool={};copies=0
    for e in engines:
        for visit in e['past_visits']:
            for r in visit['records']:
                if r['memory_id'] in pool:
                    if pool[r['memory_id']]!=r:raise ValueError('CONFLICTING_MEMORY_COPIES')
                    copies+=1
                pool[r['memory_id']]=r
    records=list(pool.values());store=PatientMemoryStore(records);before=digest(store.snapshot())
    rows=[];private=[];queries=0;relevant=0;stale=0
    for ordinal,(case,engine) in enumerate(zip(cases,engines),1):
        identity=engine['longitudinal'];query=initial_query(engine);queries+=bool(query)
        oracle=eligible_reference(records,identity)
        gold=case['task_annotations'].get('gold',{})
        # None means no annotation. Never turn None into an empty negative label set.
        annotation={k:gold[k] for k in ('relevant_memory_ids','stale_memory_ids') if isinstance(gold.get(k),list)}
        relevant+='relevant_memory_ids' in annotation;stale+='stale_memory_ids' in annotation
        for mode in MODES:
            context=store.retrieve(identity,query,mode,5)
            selected={r['memory_id'] for r in context['records']}
            metrics=evaluate_memory(context,[],annotation,oracle)
            boundary=len(selected-oracle)
            row={'case_ordinal':ordinal,'mode':mode,'query_fingerprint':digest(query),'eligible_count':len(oracle),
                'selected_count':len(selected),'boundary_violations':boundary,
                'full_history_missing':len(oracle-selected) if mode=='full_history' else None,
                'off_empty':not selected if mode=='none' else None,'top_k_compliant':len(selected)<=5 if mode=='selective' else None,
                'selection_method':context['selection_method'],
                'structural_eligible_coverage':len(selected&oracle)/len(oracle) if oracle else None,
                'precision':metrics['precision'],'recall':metrics['recall'],
                'stale_retrieval_rate':metrics['stale_retrieval_rate'],
                'clinical_relevance_status':'NOT_REVIEWED','time_reference':'DECLARED_ARCHIVAL_TIMESTAMPS_NOT_CLINICALLY_REVIEWED',
                'query_visibility_status':'LEGACY_INITIAL_SYMPTOM_PROJECTION_NOT_CLINICALLY_REVIEWED'}
            rows.append(row);private.append({'case_ordinal':ordinal,'mode':mode,'identity':identity,'query':query,
                'selected_ids':sorted(selected),'eligible_ids_fingerprint':digest(sorted(oracle))})
    if digest(store.snapshot())!=before:raise ValueError('MEMORY_MUTATED_DURING_BENCHMARK')
    summary={m:{'case_count':len(cases),'selected_records':sum(r['selected_count'] for r in rows if r['mode']==m),
        'boundary_violations':sum(r['boundary_violations'] for r in rows if r['mode']==m),
        'cases_with_selection':sum(r['selected_count']>0 for r in rows if r['mode']==m)} for m in MODES}
    return {'schema_version':'locdi.c4.memory_source_benchmark.v1','status':'PASS_STRUCTURAL_ONLY' if all(
        r['boundary_violations']==0 and r['full_history_missing'] in (None,0) and r['off_empty'] is not False
        and r['top_k_compliant'] is not False for r in rows) else 'FAIL',
        'case_count':len(cases),'unique_patients':len({e['longitudinal']['patient_id'] for e in engines}),
        'past_visit_count':sum(len(e['past_visits']) for e in engines),'unique_memory_records':len(records),
        'duplicate_identical_copies':copies,'query_nonempty':queries,'relevance_annotation_cases':relevant,'stale_annotation_cases':stale,
        'retrieval_calls':len(rows),'modes':summary,'rows':rows,'memory_unchanged':True,
        'quality_evaluation':'BLOCKED_MISSING_INDEPENDENT_RELEVANCE_AND_VISIBILITY_REVIEW',
        'evidence_level':'SOURCE_DERIVED_SYNTHETIC_STRUCTURAL_BENCHMARK_NOT_EXTERNAL_CLINICAL_PERFORMANCE',
        'model_generations':0,'utility_fitting':0,'clinical_metrics':'NOT_EVALUATED'},private

def snapshot():
    protected={}
    previous=ROOT/'outputs/private/research_evaluation/c3_16_v1/reports/baseline_hashes.json'
    for p in load(previous):
        path=ROOT/p
        if not path.is_file():raise ValueError('PROTECTED_FILE_MISSING')
        protected[p]=sha(path)
    manifest=load(ROOT/'outputs/research_evaluation/c3_16_v1/reports/output_manifest.json')
    for p,h in manifest['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('C3_16_MANIFEST_CHANGED')
        protected[p]=h
    index=ROOT/'outputs/private/research_evaluation/c3_16_v1/reports/private_index.json'
    for p,h in load(index)['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('C3_16_PRIVATE_CHANGED')
        protected[p]=h
    for p in (previous,index,ROOT/'outputs/research_evaluation/c3_16_v1/reports/output_manifest.json'):protected[p.relative_to(ROOT).as_posix()]=sha(p)
    for folder in ('app/core','app/locdi','validation','configs','experiments','prompts','docs','datasets/releases'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[p.relative_to(ROOT).as_posix()]=sha(p)
    protected[DATA.relative_to(ROOT).as_posix()]=sha(DATA)
    return protected

def code_inventory():
    names=['app/core/longitudinal_memory.py','app/core/memory_retrieval.py','app/core/memory_evaluation.py',
        'app/core/clinical_observation.py','app/core/agent_registry.py','app/core/routing_utility.py',
        'app/core/diagnostic_belief.py','app/core/information_outcomes.py','app/core/information_value.py','app/core/voi_planning.py',
        'app/core/voi_evaluation.py','app/core/longitudinal_runner.py','app/core/hospital_llm_policy.py',
        'app/locdi/run_simulation.py','app/locdi/integrated_runtime.py','app/locdi/sequential_handoff.py',
        'app/locdi/integrated_approval.py','experiments/run_locdi_ablation.py','experiments/locdi_metrics.py',
        'experiments/evaluation_protocol.py','validation/official_evaluation_readiness.py']
    result=[]
    for name in names:
        path=ROOT/name;tree=ast.parse(path.read_text(encoding='utf-8-sig'));functions=[]
        for node in tree.body:
            group=node.body if isinstance(node,ast.ClassDef) else [node]
            for f in group:
                if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    functions.append({'name':(node.name+'.' if isinstance(node,ast.ClassDef) else '')+f.name,
                        'line':f.lineno,'signature':ast.unparse(f.args)})
        result.append({'path':name,'sha256':sha(path),'functions':functions})
    return result

def readiness_audit():
    from app.locdi.evaluation_gate_audit import audit_actual_gate,actual_gate_inputs
    from app.locdi.development_features import check_development_features
    from app.locdi.integrated_approval import preflight_integrated
    from app.locdi.clinical_gap_analysis import initial_matrix,split_matrix,routing_groups
    from app.locdi.dataset_freeze import verify,verify_pair,PUBLIC as FREEZE_PUBLIC,PRIVATE as FREEZE_PRIVATE
    from app.locdi.source_replay import PUBLIC as D3_PUBLIC
    from app.locdi.routing_data_reuse import BANK,read
    original=load_cases(DATA);dm=load(D3_PUBLIC/'dataset_manifest.json')
    if digest({k:v for k,v in dm.items() if k!='manifest_fingerprint'})!=dm['manifest_fingerprint']:raise ValueError('D3_MANIFEST_CHANGED')
    def d3(name):
        entry=dm['files'][name];path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('D3_CANONICAL_CHANGED')
        return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()] if name.endswith('.jsonl') else load(path)
    initials,actions,refs,dcases=[d3(n) for n in ('initial_observations.jsonl','action_events.jsonl','reference_candidates.jsonl','cases.jsonl')]
    frozen_private=load(ROOT/FREEZE_PRIVATE);frozen_public=load(ROOT/FREEZE_PUBLIC)
    try:freeze=verify(full=False)
    except ValueError as exc:freeze={'status':'FAIL','error_code':str(exc),'scope':'UNCHANGED_LEGACY_FULL_PROTECTION_CHECK'}
    # This is a separately named narrower check, never a replacement PASS for verify().
    pair=verify_pair(frozen_public,frozen_private)
    data,_=actual_gate_inputs();bank=read(BANK);voi=read(BANK.parent/'voi_model.json')
    pilot=load(ROOT/'outputs/private/external_evaluation/healthbench/D5_pilot_v1/selection_manifest.json')
    split=split_matrix(frozen_private['private_memberships']['patient_memberships'],
        observed_original=[c['patient_id'] for c in original],observed_d3=[c['patient_id'] for c in dcases],
        observed_bank=[r['training_patient_id'] for r in bank],observed_voi=voi['provenance']['training_patients'],
        observed_routing=data['split_manifest']['patients'],pilot_ids=[r['sample_id'] for r in pilot['samples']],
        frozen_pilot_ids=frozen_private['private_memberships']['healthbench_excluded_pilot_ids'])
    current={'initial_information_matrix':initial_matrix(original,initials,actions,refs),
        'split_independence_matrix':split,'routing_group_analysis':routing_groups(data),
        'verification_summary':{'legacy_D7_verify':freeze,'frozen_payload_and_manifests':{'status':'PASS','scope':'VERIFY_PAIR_ONLY','result':pair}}}
    return {'routing_correctness_gate':audit_actual_gate(),
        'development_features':{m:check_development_features(m) for m in ('er_only','adaptive_routing','bayesian_voi')},
        'integrated_live':{m:preflight_integrated(run_id='c401'+('0'*28),mode=m) for m in ('adaptive_routing','bayesian_voi','adaptive_and_voi')},
        'initial_information_matrix':current['initial_information_matrix'],
        'split_independence_matrix':current['split_independence_matrix'],
        'routing_group_analysis':current['routing_group_analysis'],
        'verification_summary':current['verification_summary']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--save',action='store_true')
    parser.add_argument('--namespace',default='c4_01_v1');parser.add_argument('--full-regression',action='store_true')
    parser.add_argument('--finish-existing-offline-audit',action='store_true',help='Only finish missing audit outputs after preserved offline benchmark; no models/resume')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.namespace):parser.error('invalid namespace')
    if args.full_regression and not args.save:parser.error('regression requires --save')
    if args.finish_existing_offline_audit and not args.save:parser.error('finish requires --save')
    public=ROOT/'outputs/research_evaluation'/args.namespace;private=ROOT/'outputs/private/research_evaluation'/args.namespace
    # Deny all socket activity for this process, including accidental LLM/embedding calls.
    def guard(event,values):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C4_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    finishing=args.finish_existing_offline_audit
    if args.save and not finishing and (public.exists() or private.exists()):raise ValueError('NAMESPACE_EXISTS_NO_OVERWRITE')
    baseline=load(private/'reports/baseline_hashes.json') if finishing else snapshot()
    if finishing:
        if any(not (ROOT/p).is_file() or sha(ROOT/p)!=h for p,h in baseline.items()):raise ValueError('CURRENT_TASK_BASELINE_CHANGED')
        if (public/'reports/completion.json').exists():raise ValueError('AUDIT_ALREADY_COMPLETED')
    if args.save and not finishing:
        public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
        _write(baseline,'reports/baseline_hashes.json',private,'reports')
    result,details=(load(public/'reports/memory_benchmark.json'),None) if finishing else benchmark(load_cases(DATA))
    if args.save and not finishing:
        _write(result,'reports/memory_benchmark.json',public,'reports')
        _write(details,'reports/memory_internal_links.json',private,'reports')
        _write(code_inventory(),'reports/code_inventory.json',public,'reports')
    if args.save:
        if finishing:_write({'status':'FAILED_AFTER_BENCHMARK','error_code':'ORIGINAL_FILE_CHANGED',
            'function':'app.locdi.clinical_gap_analysis.audit -> dataset_freeze.verify',
            'preserved_benchmark':True,'full_regression_started_in_initial_attempt':False},
            'reports/initial_audit_failure.json',public,'reports')
        # Audit can contain original identities: keep the full result private.
        audit=readiness_audit();_write(audit,'reports/readiness_private.json',private,'reports')
        gate=audit['routing_correctness_gate']
        safe={'routing_gate':{k:gate.get(k) for k in ('source_reference_ready','clinical_review_ready','response_policy_ready',
            'complete_train_groups','validation_group_count','positive_similarity_support','utility_fit_ready','research_evaluation_ready','blocker_codes')},
            'feature_status':{m:{k:v.get(k) for k in ('development_readiness','research_readiness','blockers')} for m,v in audit['development_features'].items()},
            'live_integrated_blockers':{m:v['blocker_codes'] for m,v in audit['integrated_live'].items()},
            'source_dataset_sha256':sha(DATA),'clinical_metrics':'NOT_EVALUATED'}
        _write(safe,'reports/readiness_summary.json',public,'reports')
        regression={'status':'NOT_RUN'}
        if args.full_regression:
            from scripts.verify_simulator_release import run_regression
            (public/'regression').mkdir();regression=run_regression(public/'regression')
            _write(regression,'reports/regression.json',public,'reports')
        changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
        _write({'protected_count':len(baseline),'changed_count':len(changed),'changed':changed},'reports/preservation.json',private,'reports')
        _write({'benchmark_status':result['status'],'protected_count':len(baseline),'changed_count':len(changed),
            'regression':regression,'new_generation_count':0,'clinical_metrics':'NOT_EVALUATED'},'reports/completion.json',public,'reports')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=True))

if __name__=='__main__':main()
