"""Read-only audit of actual user-run Routing/VoI smokes. No generation/fitting."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import json,csv,sys,re
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_b34_transfer_smoke import load,IDS
from scripts.analyze_original50_d0_d2 import facts
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.current_ablation_runner import _no_reference
from jsonschema import Draft202012Validator

NS='B34_U1_SMOKE_COMPLETED_ANALYSIS_v2';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS

def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def dt(s):
    # .NET approval timestamps use seven fractional digits. Python3.9 accepts
    # six. Only normalize a local parsed view; never change saved approvals.
    s=s.replace('Z','+00:00');s=re.sub(r'(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)',r'\1',s)
    return datetime.fromisoformat(s)

def main():
    PUB.mkdir(exist_ok=False);PR.mkdir(exist_ok=False);before={}
    for name in IDS.values():
        pub,pr,_,_=load(name)
        for base in (pub,pr):
            for file in base.rglob('*'):
                if file.is_file():before[str(file.resolve())]=sha_file(file)
    write(PR/'preservation_before.json',before)
    reports=[];details=[];starts=[];repeat=[];audit_counts=Counter()
    for stage,name in IDS.items():
        pub,pr,p,bp=load(name);run=pr/p['run_id'];c=read(run/'collection.json');saved=read(pub/'saved_results.json');tasks=read(pr/'tasks.json');approval=read(run/'approval.json');started=read(run/'started.json')
        assert c['status']=='COMPLETED' and c['TEST_ONLY'] is False and len(c['rows'])==len(tasks)==2
        assert c['generation']==saved['generation']==sum(r['request_count'] for r in c['rows'])<=24
        assert c['plan_fingerprint']==started['fingerprint']==approval['plan_fingerprint']==p['fingerprint']
        assert dt(approval['recorded_at'])<=dt(started['at'])<dt(approval['expires_at'])
        assert approval['operator_approved'] is True and approval['accept_authored_utility_transfer'] is True and approval['clinical_release_approved'] is False
        rows=[]
        for row,task in zip(c['rows'],tasks):
            assert row['status']=='COMPLETED' and all(row[k]==v for k,v in task.items())
            folder=run/'requests'/row['branch']/('Episode%04d'%row['branch_episode']);f=facts(folder,row['result'],read(bp/'panel.json')[0]['reference']);runtime=read(folder/'runtime_snapshot_private.json')
            assert f['request_count_checkpoint']==row['request_count'];usage=Counter();payloads=[];proposals=[];fingerprints=[]
            tests={t['summary']:t['observed_minute'] for t in runtime['test_results']};asks={q['answer']:q['observed_minute'] for q in runtime['question_history'] if q['answer_status']=='provided'}
            bfs=sorted(folder.glob('Request*_before.json'));afs=sorted(folder.glob('Request*_after.json'));assert len(bfs)==len(afs)==row['request_count']
            for bf,af in zip(bfs,afs):
                b=read(bf);a=read(af);request=b['request'];payload=json.loads(request['messages'][1]['content']);payloads.append(payload);_no_reference(payload)
                assert digest(request)==b['request_fingerprint']==a['request_fingerprint']
                assert b['contract']==p['fingerprint'] and a['status']=='VALID' and a['HTTP_status']==200
                assert request['options']=={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42}
                for identity in (b['identity_before'],a['identity_after']):assert normalize_model_identifier(identity['digest'])==p['Model']
                assert a['identity_after']['context_length']==32768
                raw=json.loads(a['response_envelope']['message']['content']);Draft202012Validator(request['format']).validate(raw)
                core=read(bf.with_name(bf.name.replace('_before','_core_response')))['decoded_response'];selected=core['evidence_indices'];assert len(selected)<=4 and len(selected)==len(set(selected))
                assert all(0<=i<len(payload['evidence']) for i in selected)
                for memory in payload['observation']['longitudinal_memory']['records']:
                    assert dt(memory['available_at'])<=dt(payload['observation']['longitudinal_memory']['as_of']);audit_counts['memory_occurrences_checked']+=1
                for e in payload['evidence_records']:
                    receipt=tests.get(e['text'],asks.get(e['text']));assert receipt is not None
                    assert payload['observation']['elapsed_minutes']>=receipt;audit_counts['evidence_receipt_occurrences_checked']+=1
                env=a['response_envelope'];usage.update(prompt_tokens=env['prompt_eval_count'],output_tokens=env['eval_count'])
                audit_counts['HTTP200_schema_model_context_valid']+=1;proposals.append(core);fingerprints.append({'request':b['request_fingerprint'],'response_content':digest(env['message']['content'])})
            meta=row['result']['client_metadata'];assert not meta['error_types'] and meta['retry']==meta['repair']==0 and meta['request_count']==row['request_count']
            route=f['routing_trace'];last=f['voi_trace'][-1] if f['voi_trace'] else None
            result={'stage':stage,'branch':row['branch'],'condition':row['condition'],'status':row['status'],'Source_Exact':f['source_exact'],'Source_department':f['department_match'],
                'certainty':f['prediction']['certainty'],'selected_initial_specialty':route[0]['selected_specialty'],'routing_trace_counts':dict(Counter(t['selected_specialty'] or 'ABSTAIN' for t in route)),
                'generation':row['request_count'],'identity_checks':meta['identity_check_count'],'actions':row['result']['action_counts'],'usage':dict(usage),
                'core_wall_seconds':row['result']['computation']['wall_seconds'],'planner_calls':row['result']['computation']['planner_calls'],
                'STOP':row['result']['voi_STOP_count'],'information_actions':row['result']['voi_information_action_count'],
                'algorithmic_Source_Exact_separate':f['algorithmic_source_exact'] if row['condition']=='B4' else None,
                'initial_memory_records':len(payloads[0]['observation']['longitudinal_memory']['records']),'final_memory_records':len(f['last_memory'])}
            assert result['Source_Exact']==next(s['Source_Exact'] for s in saved['rows'] if s['branch']==row['branch'] and s['condition']==row['condition'])
            rows.append(result);details.append({'run':name,'row':row,'facts':f,'payloads':payloads,'proposals':proposals,'STOP':last})
            starts.append({'stage':stage,'branch':row['branch'],'condition':row['condition'],'initial_state':row['result']['initial_state_fingerprint'],
                'observation':digest(payloads[0]['observation']),'evidence':digest(payloads[0]['evidence']),'system_prompt':digest(read(bfs[0])['request']['messages'][0]['content'])})
            if row['branch']=='R1' and row['condition']=='B3':repeat.append(fingerprints)
        reports.append({'stage':stage,'plan_id':name,'status':c['status'],'episodes':rows,'generation':c['generation'],'wall_seconds':c['wall_seconds'],
            'start_KST':dt(started['at']).astimezone(timezone(timedelta(hours=9))).isoformat()})
    assert len({x['initial_state'] for x in starts})==1
    # Raw request/response equality here is a stored comparison, not a future LLM reproducibility promise.
    summary={'runs':reports,'total_episodes':4,'unique_evaluation_patients':1,'total_generation':sum(r['generation'] for r in reports),
        'all_same_initial_state':True,'all_same_initial_observation':len({x['observation'] for x in starts})==1,
        'U1_initial_role_system_prompt_same':len({x['system_prompt'] for x in starts if x['branch']=='R1'})==1,
        'U1_B3_repeated_saved_request_and_raw_content_equal':repeat[0]==repeat[1],
        'audit_counts':dict(audit_counts),'new_generation':0,'fitting':0,'clinical_accuracy':'NOT_EVALUATED',
        'independent_clinical_review':False,'research_evaluation_ready':False,'D2_label_conditioned':True,'Full10_performance_claim':False}
    write(PR/'episode_details.json',details);write(PR/'input_comparison.json',starts);write(PUB/'results_summary.json',summary)
    table=[row for r in reports for row in r['episodes']]
    with (PUB/'comparison_results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        keys=['stage','branch','condition','status','Source_Exact','Source_department','certainty','selected_initial_specialty','generation','core_wall_seconds','planner_calls','STOP','information_actions']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(table)
    changed=[f for f,h in before.items() if sha_file(f)!=h];assert not changed
    write(PUB/'preservation.json',{'files':len(before),'changed':0,'LLM_generation':0})
    print(json.dumps(summary))

if __name__=='__main__':main()
