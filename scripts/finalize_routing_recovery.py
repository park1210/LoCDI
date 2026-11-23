"""Offline validation and one-shot existing Core proxy fit. No LLM transport.

Default inspects only. --fit --allow-fit performs the non-generating operation
explicitly authorized by the user, without creating a clinical review/approval.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha
from scripts.run_common_menu_b0_b4 import load_plan
from app.locdi.development_routing_proxy import derive_rows,admit_development_proxy,MARKERS
from app.locdi.llm_setup import digest
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import fit_utility


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',action='store_true');p.add_argument('--allow-fit',action='store_true');a=p.parse_args()
    if a.fit!=a.allow_fit:p.error('BOTH_FIT_AND_ALLOW_FIT_REQUIRED')
    _,ev,_=load_plan('source_collection_recovery_001')
    folder=ROOT/'outputs/private/research_evaluation/B34_COLLECTION_RECOVERY_v1/source_collection_recovery_001'
    cp=folder/'plan.json';coll=folder/'collection_recovery_001/collection.json';plan=read(cp);collection=read(coll)
    config=read(ev/'llm_config.json');bindings=read(ev/'bindings.json');patients=[b['patient_id'] for b in bindings.values()]
    derived=derive_rows(plan,collection,coll.parent/'requests',plan['registry'],config,patients)
    status={'records':len(collection['records']),'generation_recorded':collection['generation'],
        'complete_train_groups':derived['complete_train_groups'],'fit_rows':len(derived['rows']),**MARKERS}
    if not a.fit:print(json.dumps(status));return
    if derived['complete_train_groups']<2:raise ValueError('INSUFFICIENT_TWO_COMPLETE_TRAIN_GROUPS_NO_FIT')
    output=ROOT/'outputs/private/research_evaluation/B34_LLM_TEST_READY_v1/development_proxy_fit_001'
    output.mkdir(exist_ok=False)
    write(output/'fit_execution_record.json',{'operation':'NON_GENERATING_EXISTING_CORE_UTILITY_FIT_ONCE',
        'authority_type':'USER_INSTRUCTION','authorization_evidence':'User explicitly permits Codex Utility fitting from valid stored responses; asks to verify completed collection and prepare LLM test.',
        'recorded_at':datetime.now(timezone.utc).isoformat(),'collection_sha256':sha(coll),
        'fit_rows_fingerprint':digest(derived['rows']),'max_fit_calls':1,'actual_LLM_generation':0,**MARKERS})
    artifact=fit_utility(derived['rows'],AgentRegistry(plan['registry']),min_samples=2)
    write(output/'routing_utility_source_label_proxy.json',{'artifact':artifact,**MARKERS})
    evidence={'schema_version':'locdi.routing.source_label_proxy.v1',**MARKERS,
        'collection_plan':str(cp.resolve()),'collection_file':str(coll.resolve()),'request_root':str((coll.parent/'requests').resolve()),
        'origin_files':{str(cp.resolve()):sha(cp),str(coll.resolve()):sha(coll),**derived['checkpoint_files']},
        'routing_artifact':artifact,'fit_rows_fingerprint':digest(derived['rows']),
        'complete_train_groups':derived['complete_train_groups'],'purpose':'SOURCE_DIAGNOSIS_EXACT_AGREEMENT_PROXY_NOT_CLINICAL_CORRECTNESS'}
    admitted=admit_development_proxy(evidence,plan['registry'],patients,config)
    write(output/'source_proxy_assessment.json',derived)
    write(output/'development_routing_evidence.json',evidence)
    write(ROOT/'outputs/research_evaluation/B34_LLM_TEST_READY_v1/utility_status.json',{
        **status,'fit_calls':1,'min_samples':2,'artifact_fingerprint':digest(artifact),
        'utility_status':admitted['utility_status'],'clinical_gate':'UNCHANGED_NOT_ADMITTED',
        'validation_source_scoring':'NOT_EVALUATED_AMBIGUOUS_REFERENCE','previous_partial_collection_merged':False,
        'actual_new_LLM_generation':0})
    print(json.dumps({**status,'fit_calls':1,'utility_status':admitted['utility_status'],'actual_new_LLM_generation':0}))


if __name__=='__main__':main()
