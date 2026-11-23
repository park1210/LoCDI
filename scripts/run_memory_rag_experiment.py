"""Separate B2 M/K paired development harness. Default check; user-only LLM."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import argparse,json,re,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_common_menu_b0_b4 import prepare as prepare_base,load_plan as load_base,M5,DATA
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.original_source_inventory import memory_source_map,sha_file
from app.locdi.dataset_loader import load_cases
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET,_process_private_bytes,features_for
from app.locdi.memory_rag_backend import make_memory_rag_backend
from app.locdi.development_routing_proxy import MARKERS
from app.locdi.medical_knowledge import MedicalKnowledgeIndex

NS='MEMORY_RAG_LEGACY_RECOVERY_20261121_v1';PUBLIC=ROOT/'outputs/research_evaluation'/NS;PRIVATE=ROOT/'outputs/private/research_evaluation'/NS
CORPUS=ROOT/'outputs/research_evaluation/DIAGNOSIS_KNOWLEDGE_DESIGN_20261121_v1/medical_knowledge_corpus.json'
ARMS={'C0':('legacy_top5','none'),'C1':('historical_display','none'),'C2':('diagnostic_retrieval','none'),
      'C3':('legacy_top5','common_pack'),'C4':('legacy_top5','adaptive_bm25'),'C5':('diagnostic_retrieval','adaptive_bm25')}
STUDIES={'memory':['C0','C2'],'rag':['C0','C4'],'display':['C0','C1'],'common':['C0','C3'],'interaction':['C0','C5']}
CODE=['scripts/run_memory_rag_experiment.py','scripts/Invoke-LoCDIMemoryRAG.ps1','app/locdi/memory_rag_backend.py',
      'app/locdi/diagnostic_memory_rag.py','app/locdi/original_source_inventory.py','app/locdi/medical_knowledge.py',
      'app/locdi/medical_knowledge_catalog.py','app/locdi/common_menu_backend.py','app/core/longitudinal_memory.py']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def paths(name):
    if not re.fullmatch('[A-Za-z0-9_]{1,64}',name):raise ValueError('PLAN_ID')
    return PUBLIC/name,PRIVATE/name
def prepare(name,study='memory',patients=1):
    if study not in STUDIES or patients not in (1,10):raise ValueError('STUDY_SCOPE')
    pub,pr=paths(name)
    if pub.exists() or pr.exists():raise FileExistsError('NO_OVERWRITE')
    base_name=name+'_source';prepare_base(base_name,['B2'],None,'DEVELOPMENT_EXPERIMENT',patients,True,True,None,False,False,True)
    bp,br,base=load_base(base_name);cases=load_cases(DATA);index=read(br/'case_index.json');selected=[cases[n-1] for n in index.values()]
    sources,source_report=memory_source_map(selected)
    if source_report['missing']:raise ValueError('MEMORY_SOURCE_MISSING_BLOCKED')
    pr.mkdir(parents=True);pub.mkdir(parents=True);atomic_json(pr/'memory_sources.json',sources);atomic_json(pr/'memory_source_report.json',source_report)
    MedicalKnowledgeIndex(read(CORPUS));files=[ROOT/n for n in CODE]+[CORPUS,pr/'memory_sources.json']
    files += [Path(n) for n in source_report['csv']['files']]+[Path(source_report['database_path'])]
    plan={'version':'memory_rag_paired.v1','plan_id':name,'run_id':'run_'+name,'base_source_plan':base_name,'base_fingerprint':base['fingerprint'],
        'arms':{a:{'memory_mode':ARMS[a][0],'knowledge_mode':ARMS[a][1]} for a in STUDIES[study]},
        'registered_patients':patients,'registered_episodes':2*patients,'generation_per_episode':12,'generation_total':24*patients,
        'model_identifier':base['model_identifier'],'context':32768,'output_tokens':1024,'timeout_seconds':120,
        'evidence_wire':'evidence_index_set.v1 + memory_citation_subset_code.v1','catalog':'existing shared15',
        'source_policy':base['source_policy'],'corpus_fingerprint':MedicalKnowledgeIndex(read(CORPUS)).fingerprint,
        'wall_seconds':900 if patients==1 else 2700,'memory_bytes':4*1024**3,'output_bytes':1024**3,'retry':0,'repair':0,'warmup':0,
        'scope':base['scope'],'code_input_pins':{str(p.resolve()):sha_file(p) for p in files},**MARKERS}
    plan['fingerprint']=digest(plan);atomic_json(pub/'plan.json',plan)
    approval={**MARKERS,'scope':plan['scope'],'plan_fingerprint':plan['fingerprint'],'conditions':['B2'],
        'experiment_arms':plan['arms'],'max_generation':plan['generation_total'],'accept_source_label_proxy':False,
        'accept_recorded_time_proxy':False,'accept_common_m5_menu':False,'clinical_release_approved':False,
        'operator_approved':False,'no_other_ollama_work':False,'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
    atomic_json(pr/'approval_template.json',approval)
    return {'status':'PREPARED_NOT_EXECUTED','plan_id':name,'fingerprint':plan['fingerprint'],'arms':plan['arms'],'generation_cap':plan['generation_total'],'new_LLM_generation':0}

def check(name):
    pub,pr=paths(name);p=read(pub/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    for n,h in p['code_input_pins'].items():
        if sha_file(n)!=h:raise ValueError('NEW_PLAN_REQUIRED_PIN_CHANGED')
    _,br,b=load_base(p['base_source_plan'])
    if b['fingerprint']!=p['base_fingerprint']:raise ValueError('BASE_SOURCE_PLAN_CHANGED')
    return p,br
def authority(a,p):
    expected={**MARKERS,'scope':p['scope'],'plan_fingerprint':p['fingerprint'],'conditions':['B2'],
        'experiment_arms':p['arms'],'max_generation':p['generation_total'],'operator_approved':True,'no_other_ollama_work':True,
        'accept_source_label_proxy':True,'accept_common_m5_menu':True,'accept_recorded_time_proxy':True,'clinical_release_approved':False}
    if any(a.get(k)!=v for k,v in expected.items()) or not a.get('operator_ref') or not a.get('user_statement'):raise ValueError('ACTUAL_NEW_SCOPE_APPROVAL_REQUIRED')
    start=datetime.fromisoformat(a['recorded_at']);end=datetime.fromisoformat(a['expires_at']);now=datetime.now(timezone.utc)
    if start.tzinfo is None or end.tzinfo is None or not start<=now<end or not 0<(end-start).total_seconds()<=86400:raise ValueError('APPROVAL_TIME')

def execute(name,approval):
    p,br=check(name);a=read(approval);authority(a,p);pub,pr=paths(name);run=pr/p['run_id'];run.mkdir(exist_ok=False)
    atomic_json(run/'started.json',{'plan_fingerprint':p['fingerprint'],'at':datetime.now(timezone.utc).isoformat()})
    data=load_cases(DATA);index=read(br/'case_index.json');cases={ref:data[n-1] for ref,n in index.items()};bindings=read(br/'bindings.json');panel=read(br/'panel.json')
    started=time.monotonic()
    def guard():
        authority(a,p)
        if time.monotonic()-started>=p['wall_seconds'] or _process_private_bytes()>=p['memory_bytes']:raise ValueError('GLOBAL_RESOURCE_CAP')
        if sum(x.stat().st_size for x in run.rglob('*') if x.is_file())>=p['output_bytes']:raise ValueError('GLOBAL_OUTPUT_CAP')
    budget=GenerationBudget(p['generation_total'],12,resource_guard=guard);rows=[];stop=None
    backends={arm:make_memory_rag_backend(cases,**flags,source_map=read(pr/'memory_sources.json'),corpus=read(CORPUS),
        config=read(br/'llm_config.json'),bindings=bindings,model=read(M5),authorization=a,plan_fingerprint=p['fingerprint'],
        checkpoint_root=run/'requests'/arm,experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
        for arm,flags in p['arms'].items()}
    for i,patient in enumerate(panel,1):
        for arm in p['arms']:
            row={'patient_ordinal':i,'case_ref':patient['case_ref'],'arm':arm,'status':'NOT_ATTEMPTED','request_count':0,'result':None}
            if stop is None:
                before=budget.used;budget.episode_used=0;token=_BUDGET.set(budget)
                try:
                    guard();result=backends[arm]({'case_ref':patient['case_ref']},features_for('B2'))
                    row.update(status=result['status'].upper(),result=result);guard()
                    if result['status']!='completed':stop='FIRST_NONCOMPLETED_EPISODE'
                except Exception as e:row.update(status='FAILED',error_type=type(e).__name__,private_error_message=str(e));stop='FIRST_TECHNICAL_ERROR'
                finally:row['request_count']=budget.used-before;_BUDGET.reset(token)
            rows.append(row);atomic_json(run/('Episode%04d.json'%len(rows)),row)
    report={'status':'COMPLETED' if stop is None else 'PARTIAL_OR_FAILED','rows':rows,'actual_generation':budget.used,
        'wall_seconds':time.monotonic()-started,'stop_reason':stop,'plan_fingerprint':p['fingerprint'],**MARKERS}
    atomic_json(run/'collection.json',report)
    atomic_json(pub/'execution_summary.json',{k:v for k,v in report.items() if k!='rows'})
    return report['status']
def analyze(name):
    p,br=check(name);pub,pr=paths(name);run=pr/p['run_id'];collection=read(run/'collection.json');panel=read(br/'panel.json');rows=[]
    for r in collection['rows']:
        pred=(r.get('result') or {}).get('prediction',{});ref=panel[r['patient_ordinal']-1]['reference']
        rows.append({'patient_ordinal':r['patient_ordinal'],'arm':r['arm'],'status':r['status'],'certainty':pred.get('certainty'),
            'source_exact':pred.get('diagnosis') in ref['accepted_diagnoses'] if pred.get('diagnosis') else None,'generation':r['request_count']})
    report={'rows':rows,'diagnosis_metric':'SOURCE_EXACT_AUTHOR_SOURCE_POLICY_NOT_CLINICAL_ACCURACY',
        'registered_patients':p['registered_patients'],'actual_generation':collection['actual_generation'],**MARKERS}
    target=pub/'saved_results.json'
    if target.exists():
        if read(target)!=report:raise ValueError('EXISTING_ANALYSIS_DIFFERENT_NO_OVERWRITE')
    else:atomic_json(target,report)
    return rows
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','check','execute','analyze'],nargs='?',default='check')
    parser.add_argument('--plan-id',required=True);parser.add_argument('--study',choices=list(STUDIES),default='memory');parser.add_argument('--patients',type=int,choices=[1,10],default=1)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--allow-llm',action='store_true');parser.add_argument('--approval',type=Path);a=parser.parse_args()
    if a.mode=='prepare':out=prepare(a.plan_id,a.study,a.patients)
    elif a.mode=='check':p,_=check(a.plan_id);out={'status':'STATIC_PIN_VALID','plan_id':a.plan_id,'arms':p['arms'],'generation_cap':p['generation_total'],'new_LLM_generation':0}
    elif a.mode=='execute':
        if not(a.execute and a.allow_llm and a.approval):parser.error('USER_EXECUTE_ALLOW_LLM_AND_NEW_APPROVAL_REQUIRED')
        status=execute(a.plan_id,a.approval);out={'status':status}
        if status!='COMPLETED':print(json.dumps(out));raise SystemExit(2)
    else:out=analyze(a.plan_id)
    print(json.dumps(out))
if __name__=='__main__':main()
