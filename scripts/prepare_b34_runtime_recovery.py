"""Verify saved new collection; fit unchanged development proxy, no inference."""
from collections import Counter
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.development_routing_proxy import derive_rows,admit_development_proxy,MARKERS
from app.locdi.llm_setup import digest,validate_setup
from app.core.routing_utility import fit_utility
from app.core.agent_registry import AgentRegistry
NS='B34_RUNTIME_RECOVERY_20261121_v1'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=True,indent=2)
def main():
    PUB.mkdir(parents=True,exist_ok=False);PR.mkdir(parents=True,exist_ok=False)
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/followup_catalog_all_001'
    cp=base/'routing_training/plan.json';coll=base/'routing_training/collection_001/collection.json'
    p=read(cp);c=read(coll);config=read(base/'llm_config.json');registry=validate_setup(config)['prepared']['registry']
    patients=[b['patient_id'] for b in read(base/'bindings.json').values()]
    roots=[ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1',ROOT/'outputs/research_evaluation/B34_CONNECTION_v1',
           ROOT/'outputs/private/research_evaluation/B34_COMPLETED_RUN_ANALYSIS_20261121_v3',
           ROOT/'docs/B34_FOLLOWUP_ANALYSIS_20261121_v1',ROOT/'app/core',ROOT/'prompts',ROOT/'configs']
    protected={str(f.resolve()):sha(f) for r in roots for f in r.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    for f in (ROOT/'outputs/private/research_evaluation/c4_33_v1/final_m5_model.json',ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'):
        protected[str(f)]=sha(f)
    write(PR/'protection_before.json',protected)
    derived=derive_rows(p,c,coll.parent/'requests',registry,config,patients)
    assert c['generation']==15 and len(c['records'])==15 and c['stop_reason'] is None
    if derived['complete_train_groups']<2:raise ValueError('FEWER_THAN_TWO_COMPLETE_GROUPS_NO_FIT')
    artifact=fit_utility(derived['rows'],AgentRegistry(registry),min_samples=2)
    evidence={'schema_version':'locdi.routing.source_label_proxy.v1',**MARKERS,
        'collection_plan':str(cp.resolve()),'collection_file':str(coll.resolve()),'request_root':str((coll.parent/'requests').resolve()),
        'origin_files':{str(cp.resolve()):sha(cp),str(coll.resolve()):sha(coll),**derived['checkpoint_files']},
        'routing_artifact':artifact,'fit_rows_fingerprint':digest(derived['rows']),
        'complete_train_groups':derived['complete_train_groups'],'purpose':'SOURCE_DIAGNOSIS_EXACT_AGREEMENT_PROXY_NOT_CLINICAL_CORRECTNESS'}
    admit_development_proxy(evidence,registry,patients,config)
    write(PR/'development_routing_evidence.json',evidence);write(PR/'derived_assessment.json',derived)
    write(PR/'routing_utility_source_label_proxy.json',{'artifact':artifact,**MARKERS})
    report={**MARKERS,'saved_generation':c['generation'],'responses':len(c['records']),'schema_valid':15,
        'supported':sum(r['response']['certainty']=='supported' for r in c['records']),
        'unresolved':sum(r['response']['certainty']=='unresolved' for r in c['records']),
        'groups':[{'ordinal':i,'complete':g['complete_proxy_group'],'statuses':dict(Counter(x['status'] for x in g['outcomes']))} for i,g in enumerate(derived['groups'],1)],
        'complete_train_groups':derived['complete_train_groups'],'fitted_rows':len(derived['rows']),
        'positive_by_agent':dict(Counter(r['agent_id'] for r in derived['rows'] if r['correct'])),
        'negative_by_agent':dict(Counter(r['agent_id'] for r in derived['rows'] if not r['correct'])),
        'fit_calls':1,'new_LLM_generation':0,'previous_responses_merged':False,
        'fit_authorization':'Current user request to continue after collection, with prior explicit permission for fitting valid saved responses; not clinical approval',
        'protection_files':len(protected)}
    assert all(sha(f)==h for f,h in protected.items());write(PUB/'collection_and_fit.json',report);print(json.dumps(report))
if __name__=='__main__':main()
