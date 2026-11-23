"""Prepare bounded new development arms, never create an LLM client."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_common_menu_b0_b4 import prepare,preflight,paths
CORPUS=ROOT/'outputs/research_evaluation/DIAGNOSIS_KNOWLEDGE_DESIGN_20261121_v1/medical_knowledge_corpus.json'
UTILITY=ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/development_routing_evidence.json'
ARMS={
 'E0R':(['B0','B1','B2','B3','B4'],False,False,False),
 'E1':(['B0','B1','B2','B3','B4'],True,False,False),
 'E2':(['B0','B1','B2'],False,True,False),
 'E3_ADAPTIVE':(['B3'],False,False,False),
 'E3_RULE':(['B3'],False,False,True),
 'E3_ER':(['B2'],False,False,False),
 'E4':(['B4'],True,False,False),
}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=list(ARMS),required=True)
    p.add_argument('--plan-id',required=True);p.add_argument('--patients',type=int,choices=[1,10],default=1)
    a=p.parse_args();conditions,knowledge,temporal,rules=ARMS[a.experiment]
    prepare(a.plan_id,conditions,UTILITY,'DEVELOPMENT_EXPERIMENT',a.patients,True,True,
            CORPUS if knowledge else None,temporal,rules,True)
    result=preflight(a.plan_id);result.update(experiment=a.experiment,generation_cap=12*a.patients*len(conditions),
        new_LLM_generation=0,clinical_accuracy='NOT_EVALUATED',evidence_wire_control_required=True)
    pub,_=paths(a.plan_id)
    with (pub/'knowledge_preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
    if result['status']=='BLOCKED':raise SystemExit(2)
if __name__=='__main__':main()
