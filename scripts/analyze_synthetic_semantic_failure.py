"""Read saved v1-schema records only; no model, fitting or historical writes."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from jsonschema import Draft202012Validator
from app.locdi.synthetic_pilot_collection import read,M5,ZIP
from app.locdi.synthetic_pilot_source import load_zip,score_response
from app.locdi.routing_collection_preparation import response_schema as legacy_schema
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.llm.prompt_loader import specialty_system_prompt
from app.locdi.routing_collector import atomic_json

def main():
    ns='SYNTHETIC_PILOT_SEMANTIC_RECOVERY_20261121_v1';pub=ROOT/'outputs/research_evaluation'/ns;private=ROOT/'outputs/private/research_evaluation'/ns
    base=ROOT/'outputs/private/research_evaluation/SYNTHETIC_PILOT_UTILITY_20261121_v1';model=read(M5);bundle=load_zip(ZIP,model['diagnoses'])
    summaries=[];details=[];matrix=[]
    for name in ('syn15_initial_smoke_003','syn15_initial_full_003'):
        p=read(base/name/'plan.json');run=base/name/p['run_id'];status=read(run/'collection_status.json')
        assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
        assert p['zip_sha256']==bundle['zip_sha256']
        attempted=normal=invalid=http=prompt_tokens=output_tokens=0;groups=[];per_agent={}
        for task in p['tasks']:
            scored=[];present=[]
            for agent in p['agents']:
                folder=run/('Case%03d'%task['ordinal'])/agent['department'];record=read(folder/'response.json') if (folder/'response.json').exists() else None
                match=None;semantic='NOT_ATTEMPTED';response=None
                if record:
                    before=read(folder/'Request0001_before.json');after=read(folder/'Request0001_after.json');request=before['request'];attempted+=1
                    assert record['plan_fingerprint']==p['fingerprint'] and record['shared_input_fingerprint']==task['shared_fingerprint']
                    assert before['request_fingerprint']==after['request_fingerprint']==digest(request)
                    assert request['format']==legacy_schema(p['catalog'])
                    assert request['messages']==[{'role':'system','content':specialty_system_prompt(agent['department'])},
                        {'role':'user','content':json.dumps({'department':agent['department'],**task['shared']},sort_keys=True)}]
                    assert request['model']==p['config']['model'] and request['options']=={'num_ctx':32768,'temperature':0,'seed':42,'num_predict':1024}
                    assert after['HTTP_status']==200 and after['status']=='VALID';http+=1
                    for identity in (before['identity_before'],after['identity_after']):assert normalize_model_identifier(identity['digest'])==p['config']['model_identifier']
                    assert after['identity_after']['context_length']==32768
                    env=after['response_envelope'];assert env['done'] is True and env['done_reason']=='stop'
                    response=json.loads(env['message']['content']);Draft202012Validator(legacy_schema(p['catalog'])).validate(response)
                    prompt_tokens+=env['prompt_eval_count'];output_tokens+=env['eval_count']
                    try:match,semantic=score_response(response,bundle['references'][task['case_id']],p['catalog'])
                    except ValueError as e:
                        assert str(e)=='CERTAINTY_DIAGNOSIS_CONTRADICTION';invalid+=1;semantic='CERTAINTY_DIAGNOSIS_CONTRADICTION'
                        assert record['status']=='FAILED' and record['response'] is None
                    else:
                        assert record['status']=='RESPONSE_RECORDED' and record['response']==response
                        assert digest(response)==record['response_fingerprint'];normal+=1
                    details.append({'plan_id':name,'ordinal':task['ordinal'],'agent':agent['department'],
                        'raw_response':response,'reference':bundle['references'][task['case_id']],
                        'reference_match':match,'semantic_status':semantic})
                per_agent.setdefault(agent['department'],{'normal':0,'semantic_invalid':0,'not_attempted':0,'true':0,'false':0,'null':0})
                a=per_agent[agent['department']];a['normal']+=record is not None and semantic!='CERTAINTY_DIAGNOSIS_CONTRADICTION'
                a['semantic_invalid']+=semantic=='CERTAINTY_DIAGNOSIS_CONTRADICTION';a['not_attempted']+=record is None
                a['true']+=match is True;a['false']+=match is False;a['null']+=match is None
                scored.append(match);present.append(record is not None)
                matrix.append({'plan_id':name,'ordinal':task['ordinal'],'agent':agent['department'],'reference_match':match,'status':semantic})
            if any(present):groups.append({'ordinal':task['ordinal'],'responses_or_markers':sum(present),
                'complete_scorable':all(type(v) is bool for v in scored),'positive':sum(v is True for v in scored),'negative':sum(v is False for v in scored),'null':sum(v is None for v in scored)})
        assert attempted==status['generation_attempts'] and normal==status['normal_responses']
        summary={'plan_id':name,'status':status['status'],'planned':p['generation_cap'],'generation':attempted,'normal':normal,
            'HTTP200':http,'JSON_and_legacy_schema_valid':http,'semantic_invalid':invalid,'not_attempted':status['not_attempted'],
            'wall_seconds':status['wall_seconds'],'prompt_tokens_reported':prompt_tokens,'output_tokens_reported':output_tokens,
            'identity_snapshots_verified':2*attempted,'loaded_context':32768,'by_agent':per_agent,'groups':groups,
            'complete_scorable_train_groups':sum(g['complete_scorable'] for g in groups),'fit_ready':sum(g['complete_scorable'] for g in groups)>=2,
            'clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False}
        summaries.append(summary)
    atomic_json(pub/'actual_run_analysis.json',{'runs':summaries,'runs_merged':False,'stored_generations_total':sum(r['generation'] for r in summaries),
        'new_LLM_generation':0,'fit_called':False,'smoke_technical_success':True,'full_collection_completed':False})
    atomic_json(pub/'response_status_matrix.json',matrix)
    atomic_json(private/'response_reference_details.json',details)
    print(json.dumps(summaries,ensure_ascii=True))
if __name__=='__main__':main()
