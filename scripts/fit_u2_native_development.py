"""Separately approved Source-proxy fitting AFTER actual Native collection.

This artifact is NOT admitted to existing B3/B4 or any clinical Gate here.
No model requests. Unresolved/failed never become negative labels.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_u2_native_collection import load,analyze,PUB,PR
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.routing_collector import atomic_json
from app.locdi.common_prediction_contract import extract_core_prediction
from app.locdi.development_routing_proxy import MARKERS
from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import fit_utility

NAME='u2_native_development_fit_001'


def prepared():
    pub,pr,p=load('Full')
    protocol={'plan_id':NAME,'scope':'NATIVE_SOURCE_PROXY_DEVELOPMENT_FIT_ONLY',
        'collection_plan_id':p['plan_id'],'collection_plan_fingerprint':p['fingerprint'],
        'fit_code_sha256':sha_file(__file__),'min_samples':2,'five_agent_complete_train_groups_required':2,
        'U0_U1_unchanged':True,'B3_B4_admission':'NOT_INSTALLED',
        'B4_blocker':'ROUTING_VALIDATION_VOI_TRAIN_OVERLAP','fit_calls_max':1,**MARKERS}
    protocol['fingerprint']=digest(protocol)
    target=PUB/NAME/'plan.json'
    if target.exists():
        if read(target)!=protocol:raise ValueError('FIT_PLAN_CHANGED_NEW_ID_REQUIRED')
    else:
        target.parent.mkdir(exist_ok=False)
        atomic_json(target,protocol)
    return pub,pr,p,protocol


def check():
    pub,pr,p,protocol=prepared();run=pr/p['run_id'];blocks=[]
    if not (run/'collection.json').exists():blocks.append('ACTUAL_NATIVE_FULL_COLLECTION_NOT_AVAILABLE')
    if (PR/NAME).exists():blocks.append('FIT_NAMESPACE_USED_NO_OVERWRITE')
    return {'status':'BLOCKED' if blocks else 'ACTUAL_RESPONSE_REVALIDATION_AND_SEPARATE_APPROVAL_REQUIRED',
            'blocker_codes':blocks,'fit_plan_id':NAME,'fit_plan_fingerprint':protocol['fingerprint'],
            'fit_called':False,'LLM_generation':0,'B3_B4_admission':'NOT_INSTALLED'}


def fit(operator,statement):
    state=check()
    if state['status']=='BLOCKED':raise ValueError('FIT_PREFLIGHT_BLOCKED')
    if not operator.strip() or not statement.strip():raise ValueError('ACTUAL_OPERATOR_FITTING_APPROVAL_REQUIRED')
    pub,pr,p,protocol=prepared();run=pr/p['run_id'];summary=analyze('Full');details=read(run/'analysis_private.json')
    patients={r['ordinal']:r for r in read(pr/'patients.json')};agents={a['agent_id']:a for a in p['agents']};rows=[]
    if summary['status']!='COMPLETED':raise ValueError('COMPLETE_NATIVE_COLLECTION_REQUIRED')
    eligible={r['ordinal'] for r in details['groups'] if r['split']=='train' and r['complete_scorable']}
    if len(eligible)<2:raise ValueError('TWO_COMPLETE_SCORABLE_TRAIN_GROUPS_REQUIRED')
    cfg=read(pr/'llm_config.json');files={}
    for r in details['rows']:
        folder=run/('Patient%02d'%r['ordinal'])/r['department']/'Episode0001'
        if r['status']!='COMPLETED':raise ValueError('COMPLETE_NATIVE_EPISODES_REQUIRED')
        for bfile in sorted(folder.glob('Request*_before.json')):
            afile=Path(str(bfile).replace('_before.json','_after.json'));b,a=read(bfile),read(afile)
            req=b['request'];payload=json.loads(req['messages'][1]['content'])
            if (b['contract']!=p['fingerprint'] or req['model']!=cfg['model'] or req['options']!={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42}
                or a['status']!='VALID' or a['HTTP_status']!=200 or digest(req)!=a['request_fingerprint']):raise ValueError('ACTUAL_MODEL_REQUEST_LINEAGE')
            if any(normalize_model_identifier(i['digest'])!=cfg['model_identifier'] for i in (b['identity_before'],a['identity_after'])) or a['identity_after']['context_length']!=32768:raise ValueError('ACTUAL_MODEL_CONTEXT')
            from app.llm.prompt_loader import specialty_system_prompt
            if req['messages'][0]['content']!=specialty_system_prompt(r['department']) or payload['role']!='specialty' or payload['department']!=r['department']:raise ValueError('ACTUAL_PROMPT_AGENT_BINDING')
        actual=extract_core_prediction(read(folder/'actual_core_decisions.json'),r['result']['status'])
        if actual!=r['result']['prediction']:raise ValueError('ACTUAL_CORE_TERMINAL_PREDICTION')
        expected=(actual['diagnosis']==patients[r['ordinal']]['reference']['label']) if actual['certainty']=='supported' else None
        if type(expected)!=type(r['correctness_proxy']) or expected!=r['correctness_proxy']:raise ValueError('ACTUAL_SOURCE_PROXY_MATCH')
        files.update({str(f.resolve()):sha_file(f) for f in folder.rglob('*') if f.is_file()})
        if r['ordinal'] not in eligible:continue
        initial=json.loads(read(folder/'Request0001_before.json')['request']['messages'][1]['content'])['observation']
        rows.append({'patient_id':patients[r['ordinal']]['case']['patient_id'],
            'decision_id':patients[r['ordinal']]['case']['case_id']+':initial_native_episode','agent_id':r['agent_id'],
            'split':'train','observation':initial,'correct':r['correctness_proxy'],
            'model':agents[r['agent_id']]['model'],'version':agents[r['agent_id']]['version']})
    # Freeze the actual operator's new fit authority before the one fitting call.
    out=PR/NAME;out.mkdir(exist_ok=False)
    atomic_json(out/'operator_fitting_approval.json',{'scope':protocol['scope'],'plan_fingerprint':protocol['fingerprint'],
        'operator_ref':operator,'user_statement':statement,'recorded_at':datetime.now(timezone.utc).isoformat(),
        'approval_source':'USER_EXECUTED_EXPLICIT_FIT_COMMAND','clinical_approval':False})
    atomic_json(out/'fit_rows_private.json',rows)
    artifact=fit_utility(rows,AgentRegistry(p['registry']),min_samples=2)
    result={'schema_version':'locdi.routing.native_source_proxy.v1','artifact':artifact,
        'fit_plan_fingerprint':protocol['fingerprint'],'collection_plan_fingerprint':p['fingerprint'],
        'complete_scorable_train_groups':len(eligible),'fit_row_count':len(rows),'origin_files':files,
        'B3_B4_admission':'NOT_INSTALLED','clinical_gate':'UNCHANGED_NOT_ADMITTED',
        'validation_M5_training_overlap':p['validation_M5_training_overlap'],**MARKERS}
    atomic_json(out/'U2_native_development_utility.json',result)
    public={'fit_called':True,'fit_rows':len(rows),'complete_train_groups':len(eligible),
            'artifact_sha256':sha_file(out/'U2_native_development_utility.json'),'B3_B4_admission':'NOT_INSTALLED',**MARKERS}
    atomic_json(PUB/NAME/'fit_result.json',public);return public


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--check',action='store_true');q.add_argument('--approve-development-fit',action='store_true')
    q.add_argument('--operator-ref');q.add_argument('--statement');a=q.parse_args()
    if a.check:out=check()
    else:
        if not a.approve_development_fit or not a.operator_ref or not a.statement:q.error('SEPARATE_EXPLICIT_USER_FITTING_APPROVAL_REQUIRED')
        out=fit(a.operator_ref,a.statement)
    print(json.dumps(out))
    if out.get('status')=='BLOCKED':raise SystemExit(2)
