"""Revalidate stored Native five-Agent responses without calling a model/fit."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_original50_full_readiness import PUB,PR,write_new,snapshot_status
from scripts.run_u2_native_collection import load
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.common_prediction_contract import extract_core_prediction
from app.llm.prompt_loader import specialty_system_prompt
from jsonschema import Draft202012Validator


def verify(stage):
    pub,pr,p=load(stage);run=pr/p['run_id']
    if not (run/'collection.json').exists():return {'status':'NOT_RUN','stage':stage}
    c=read(run/'collection.json');lines=[json.loads(s) for s in (run/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    if c['rows']!=lines or c['plan_fingerprint']!=p['fingerprint']:raise ValueError('SAVED_COLLECTION_ROWS')
    started=read(run/'started.json');approval=read(run/'approval.json')
    if started['TEST_ONLY'] is not False or approval['plan_fingerprint']!=p['fingerprint'] or approval['operator_approved'] is not True:raise ValueError('ACTUAL_STARTED_AND_APPROVAL_REQUIRED')
    begin=datetime.fromisoformat(started['at'])
    if not datetime.fromisoformat(approval['recorded_at'])<=begin<datetime.fromisoformat(approval['expires_at']):raise ValueError('APPROVAL_NOT_VALID_AT_START')
    cfg=read(pr/'llm_config.json');patients={r['ordinal']:r for r in read(pr/'patients.json')};agents={r['agent_id']:r for r in p['agents']}
    seen=set();rows=[];initial={};total=0
    for r in lines:
        key=r['ordinal'],r['agent_id']
        if key in seen or key[0] not in patients or key[1] not in agents or agents[key[1]]['department']!=r['department']:raise ValueError('REGISTERED_AGENT_SLOT')
        seen.add(key);correct=None
        folder=run/('Patient%02d'%r['ordinal'])/r['department']/'Episode0001'
        bs=sorted(folder.glob('Request*_before.json'));afs=sorted(folder.glob('Request*_after.json'))
        if len(bs)!=r['generation'] or len(afs)!=len(bs):raise ValueError('GENERATION_CHECKPOINT_COUNT')
        total+=len(bs)
        for bfile,afile in zip(bs,afs):
            b,a=read(bfile),read(afile);req=b['request'];payload=json.loads(req['messages'][1]['content'])
            if (digest(req)!=b['request_fingerprint'] or a['request_fingerprint']!=b['request_fingerprint'] or b['contract']!=p['fingerprint']
                or a['status']!='VALID' or a['HTTP_status']!=200):raise ValueError('ACTUAL_HTTP_RESPONSE_LINKAGE')
            if (req['model']!=cfg['model'] or req['options']!={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42}
                or req['messages'][0]['content']!=specialty_system_prompt(r['department'])
                or payload['role']!='specialty' or payload['department']!=r['department']):raise ValueError('MODEL_PROMPT_AGENT_BINDING')
            if any(normalize_model_identifier(i['digest'])!=cfg['model_identifier'] for i in (b['identity_before'],a['identity_after'])) or a['identity_after']['context_length']!=32768:raise ValueError('DIGEST_CONTEXT')
            Draft202012Validator(req['format']).validate(json.loads(a['response_envelope']['message']['content']))
        if bs:initial.setdefault(r['ordinal'],set()).add(digest(json.loads(read(bs[0])['request']['messages'][1]['content'])['observation']))
        if r['status']=='COMPLETED':
            prediction=extract_core_prediction(read(folder/'actual_core_decisions.json'),r['result']['status'])
            if any(prediction[k]!=r['result']['prediction'][k] for k in ('certainty','diagnosis','department')):raise ValueError('CORE_FINAL_PREDICTION')
            if prediction['certainty']=='supported':correct=prediction['diagnosis']==patients[r['ordinal']]['reference']['label']
        rows.append({**r,'verified_correctness_proxy':correct})
    if len(seen)!=len(patients)*5 or total!=c['generation'] or any(len(v)!=1 for v in initial.values()):raise ValueError('FIVE_AGENT_COMPLETE_SAME_INITIAL_STATE')
    train_groups=0;split_positive={}
    for ordinal,patient in patients.items():
        group=[r for r in rows if r['ordinal']==ordinal]
        if patient['split']=='train' and all(type(r['verified_correctness_proxy']) is bool for r in group):train_groups+=1
    for split in ('train','validation'):
        split_positive[split]={a['department']:dict(Counter(str(r['verified_correctness_proxy']) for r in rows if r['split']==split and r['agent_id']==a['agent_id'])) for a in p['agents']}
    out={'status':'SAVED_RESPONSES_REVALIDATED','collection_status':c['status'],'registered_patients':len(patients),
         'episodes':len(rows),'generation_saved':total,'new_generation':0,'fit_called':False,
         'complete_scorable_train_groups':train_groups,'initial_fingerprints_equal_per_patient':True,
         'current_Model_Prompt_Digest_Context_Schema_Core_prediction':'PASS',
         'by_split_agent':split_positive,'clinical_accuracy':'NOT_EVALUATED','independent_clinical_review':False}
    write_new(PR/('U2_'+stage+'_verified_private.json'),rows)
    return out


if __name__=='__main__':
    result={s:verify(s) for s in ('Smoke','Full')}
    write_new(PUB/'U2_saved_response_verification.json',result)
    write_new(PUB/'latest_saved_execution_status.json',snapshot_status())
    print(json.dumps(result))
