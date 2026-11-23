"""TEST_ONLY nonclinical near-indifference diagnostic; never fits an artifact."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core.voi_planning import plan_voi
from app.core.information_value import select_action
from app.core.diagnostic_belief import validation_utility

def probe():
    b={'TEST_ONLY_A':.5,'TEST_ONLY_B':.5}
    actions=[{'id':'TEST_ONLY_T','type':'request_test','target':'TEST_ONLY_T','outcomes':[
        {'id':'TEST_ONLY_POS','likelihood':{'TEST_ONLY_A':.5000000000001,'TEST_ONLY_B':.4999999999999}},
        {'id':'TEST_ONLY_NEG','likelihood':{'TEST_ONLY_A':.4999999999999,'TEST_ONLY_B':.5000000000001}}]}]
    utility=validation_utility(b)
    planner=plan_voi(b,actions,utility,horizon=1)
    interface=[{'action_id':'TEST_ONLY_T','type':'Test','action':'request_test','target':'TEST_ONLY_T','outcome_model':'TEST_ONLY_T'},
        {'action_id':'diagnose','type':'Diagnose','action':'diagnose','target':None,'outcome_model':None}]
    selector=select_action(interface,selector='bayesian_voi',belief=b,
        artifact={'actions':actions,'terminal_utility':utility,'horizon':1})
    return {'test_only':True,'planner_selected_action_id':planner['selected_action_id'],
        'selector_selected_type':selector['selected_action']['type'],'voi':planner['action_values'][0]['voi'],
        'same_stop_decision':(planner['selected_action_id'] is None)==(selector['selected_action']['type']=='Diagnose'),
        'scope':'NONCLINICAL_NUMERICAL_CONTRACT_ONLY','generation_count':0,'fitting_count':0}

if __name__=='__main__':print(json.dumps(probe(),ensure_ascii=True))
