"""Prepare 3 unused existing Train patients; reuse unchanged 15-call collector.

No client, labels, fit or execution. Reference stratification is independent of
Agent outputs. Existing Validation patients and all Original50 are excluded.
"""
from copy import deepcopy
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_common_menu_b0_b4 import load_plan,M5
from scripts.collect_common_routing_training import training_paths
from scripts.audit_research_gaps import read,write,sha
from app.locdi.llm_setup import digest,validate_setup
from app.locdi.routing_collection_preparation import project_input
from app.locdi.m5_common_source import verify_csv
from app.locdi.development_routing_proxy import MARKERS
from app.core.routing_utility import observed_tokens

def prepare(name):
    _,epdir,ep=load_plan(name);pub,pr=training_paths(name)
    if ep.get('experiment_kind')!='DEVELOPMENT_EXPERIMENT':raise ValueError('DEVELOPMENT_ONLY')
    if pub.exists() or pr.exists():raise ValueError('NEW_TRAIN_PREPARATION_ONLY')
    inventory=ROOT/'outputs/private/research_evaluation/B34_FOLLOWUP_ANALYSIS_20261121_v1/routing_candidates.json'
    rows=read(inventory);fresh=[r for r in rows if r['readiness']=='READY_FOR_NEW_TRAIN_COLLECTION_REVIEW' and r['single_catalog_source_label']]
    selected=[];departments=set()
    for r in fresh:
        if r['source_department'] not in departments:
            selected.append(deepcopy(r['candidate']));selected[-1]['reference_candidates']=deepcopy(r['reference_candidates']);departments.add(r['source_department'])
        if len(selected)==3:break
    if len(selected)!=3:raise ValueError('THREE_DISJOINT_SOURCE_STRATA_REQUIRED')
    config=read(epdir/'llm_config.json');config['max_requests']=15;ready=validate_setup(config)
    if ready['static_readiness']!='READY':raise ValueError('CURRENT_STATIC_SETUP_REQUIRED')
    registry=ready['prepared']['registry'];agents=[a for a in registry['agents'] if a['role']=='specialty'];model=read(M5)
    evaluation={b['patient_id'] for b in read(epdir/'bindings.json').values()}
    if len(agents)!=5 or len({c['decision']['patient_id'] for c in selected})!=3:raise ValueError('FIVE_AGENTS_THREE_PATIENTS')
    for c in selected:
        d=c['decision']
        if d['split']!='train' or d['patient_id'] in evaluation:raise ValueError('SPLIT_OVERLAP')
        verify_csv([{'source_metadata':r} for r in c['reference_candidates']['source_condition_rows']])
        d['registry_fingerprint']=digest(registry);d['generation']={'model_identifier':config['model_identifier'],'temperature':0,'seed':42,'max_output_tokens':1024,
            'prompt_fingerprints':{a['agent_id']:ready['prepared']['prompt_fingerprints'][a['agent_id']] for a in agents}}
        d['tokens']=observed_tokens(d['observation']);c['shared_payload']=project_input(d['observation'],list(model['diagnoses']))
        c['shared_input_fingerprint']=digest(c['shared_payload'])
    old=read(ROOT/'outputs/private/research_evaluation/B34_COLLECTION_RECOVERY_v1/source_collection_recovery_001/plan.json')
    plan=deepcopy(old)
    plan.update(version='locdi.routing.followup_three_train.v1',evaluation_plan_fingerprint=ep['fingerprint'],
        registry=registry,llm_config=config,catalog=list(model['diagnoses']),candidates=selected,generation_total=15,
        selection='first unused existing Train candidate per Source department in preparation order; 3 strata; no Agent-response selection',
        validation_added=False,previous_responses_merged=False,collection_id='collection_001',**MARKERS)
    plan['source_files'].update({str(inventory.resolve()):sha(inventory),str(M5.resolve()):sha(M5)})
    plan['code_files'].update({str(Path(__file__).resolve()):sha(Path(__file__))})
    plan['fingerprint']=digest({k:v for k,v in plan.items() if k!='fingerprint'});write(pr/'plan.json',plan)
    write(pr/'approval_template.json',{'scope':'CURRENT_3B_ROUTING_COLLECTION_15_ONLY','plan_fingerprint':plan['fingerprint'],
        'max_generation':15,'operator_approved':False,'accept_source_proxy_for_collection_only':False,'no_other_ollama_work':False,
        'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None,**MARKERS})
    write(pub/'preparation.json',{'status':'PREPARED_NOT_EXECUTED','train_patients':3,'source_departments':sorted(departments),
        'validation_added':False,'original50_patient_overlap':0,'previous_collections_merged':False,
        'generation_cap':15,'generation':0,'plan_fingerprint':plan['fingerprint'],**MARKERS})
    print(json.dumps({'status':'PREPARED_NOT_EXECUTED','generation':0,'train_patients':3,'request_cap':15}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan-id',required=True);prepare(p.parse_args().plan_id)
