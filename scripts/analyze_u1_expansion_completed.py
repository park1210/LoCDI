"""Read-only actual full10 U1 paired analysis; no model, fitting or rescoring edits."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import json,csv,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_u1_voi_expansion import load,NAME,PR as SOURCE_PRIVATE
from app.locdi.original_source_inventory import read,sha_file
from app.locdi.llm_setup import digest,normalize_model_identifier
from app.locdi.current_ablation_runner import _no_reference
from app.locdi.synthea_diagnosis_audit import matches
from jsonschema import Draft202012Validator

NS='U1_VOI_EXPANSION_COMPLETED_ANALYSIS_20261123_v1';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS

def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def dt(s):return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)',r'\1',s.replace('Z','+00:00')))

def main():
    PUB.mkdir(exist_ok=False);PR.mkdir(exist_ok=False);pub,pr,p,bp=load();run=pr/p['run_id']
    before={str(f.resolve()):sha_file(f) for parent in (pub,pr) for f in parent.rglob('*') if f.is_file()};write(PR/'preservation_before.json',before)
    c=read(run/'collection.json');saved=read(pub/'saved_results.json');details=read(run/'analysis_private.json');tasks=read(pr/'tasks.json');lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert c['status']=='COMPLETED' and c['TEST_ONLY'] is False and len(c['rows'])==20 and c['rows']==lines and sum(r['request_count'] for r in lines)==c['generation']==saved['generation']==123
    assert c['plan_fingerprint']==p['fingerprint']
    approval=read(run/'approval.json');started=read(run/'started.json');assert approval['plan_fingerprint']==started['plan_fingerprint']==p['fingerprint']
    assert approval['operator_approved'] is True and approval['clinical_release_approved'] is False and approval['accept_routing_abstention_er_fallback'] is True
    assert dt(approval['recorded_at'])<=dt(started['at'])<dt(approval['expires_at'])
    audits=Counter();look={};private=[];rows=[];stop_counts=Counter();core_proposals=Counter()
    for row,t in zip(details,tasks):
        assert row['status']=='COMPLETED' and all(row[k]==v for k,v in t.items());folder=run/'requests'/('Episode%04d'%row['episode_ordinal']);rt=read(folder/'runtime_snapshot_private.json');f=row['facts'];initial=[];proposals=[]
        tests={x['summary']:x['observed_minute'] for x in rt['test_results']};ask={x['answer']:x['observed_minute'] for x in rt['question_history'] if x['answer_status']=='provided'}
        files=sorted(folder.glob('Request*_before.json'));assert len(files)==len(list(folder.glob('Request*_after.json')))==row['request_count']
        for n,bf in enumerate(files):
            b=read(bf);a=read(bf.with_name(bf.name.replace('_before','_after')));request=b['request'];body=json.loads(request['messages'][1]['content']);_no_reference(body)
            assert b['contract']==p['fingerprint'] and digest(request)==b['request_fingerprint']==a['request_fingerprint'];assert a['status']=='VALID' and a['HTTP_status']==200
            assert request['options']=={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42}
            for ident in (b['identity_before'],a['identity_after']):assert normalize_model_identifier(ident['digest'])==p['Model']
            assert a['identity_after']['context_length']==32768
            raw=json.loads(a['response_envelope']['message']['content']);Draft202012Validator(request['format']).validate(raw)
            core=read(bf.with_name(bf.name.replace('_before','_core_response')))['decoded_response'];indices=core['evidence_indices'];assert len(indices)<=4 and len(indices)==len(set(indices)) and all(0<=i<len(body['evidence']) for i in indices)
            for memory in body['observation']['longitudinal_memory']['records']:
                assert dt(memory['available_at'])<=dt(body['observation']['longitudinal_memory']['as_of']);audits['memory_record_occurrences_checked']+=1
            for ev in body['evidence_records']:
                receipt=tests.get(ev['text'],ask.get(ev['text']))
                if receipt is not None:assert body['observation']['elapsed_minutes']>=receipt;audits['known_evidence_receipt_occurrences_checked']+=1
                else:audits['initial_or_other_evidence_not_receipt_matched']+=1
            trace=f['routing_trace'][n];assert body['role']==('er' if trace['selected_specialty'] is None else 'specialty');audits['actual_role_routing_matches']+=1
            if n==0:initial=body
            proposals.append(core);audits['HTTP200_schema_model_context_valid']+=1
        meta=row['result']['client_metadata'];assert not meta['error_types'] and meta['retry']==meta['repair']==0
        last=f['voi_trace'][-1] if row['condition']=='B4' else None;candidate=proposals[-1];overridden=[]
        if last:
            stop_counts[last.get('terminal_origin','UNKNOWN')]+=1
            for q,h in zip(proposals,f['voi_trace']):
                if q['action']=='diagnose' and matches(q['target'],f['reference']['accepted_diagnoses'])['source_exact'] is True:
                    core_proposals['source_matching_diagnose_proposals']+=1
                    if h['selected_action']!='diagnose':overridden.append(h['minute']);core_proposals['source_matching_diagnose_proposals_overridden']+=1
        result={'panel_ordinal':row['panel_ordinal'],'condition':row['condition'],'Source_Exact':f['source_exact'],'certainty':f['prediction']['certainty'],
            'Source_department':f['department_match'],'initial_selection':f['routing_trace'][0]['selected_specialty'],'initial_role':initial['role'],
            'final_role':('er' if f['routing_trace'][-1]['selected_specialty'] is None else 'specialty'),'last_LLM_action':candidate['action'],
            'STOP_origin':last.get('terminal_origin') if last else None,'STOP_minute':last['minute'] if last else None,
            'M5_Source_Exact':f['algorithmic_source_exact'] if last else None,'generation':row['request_count'],'source_matching_proposals_overridden':len(overridden)}
        rows.append(result);look[row['panel_ordinal'],row['condition']]=(row,result,initial);private.append({'row':row,'raw_model_proposals':proposals,'source_matching_overridden_minutes':overridden})
    transitions=[];extra=Counter();group=Counter()
    for pair in saved['pairs']:
        i=pair['panel_ordinal'];x=look[i,'B3'][1];y=look[i,'B4'][1];a,b=x['Source_Exact'],y['Source_Exact']
        if a is True and b is None:tag='CORRECT_TO_UNRESOLVED'
        elif a is False and b is None:tag='INCORRECT_TO_UNRESOLVED'
        elif a is False and b is True:tag='INCORRECT_TO_CORRECT'
        elif a is True and b is False:tag='CORRECT_TO_INCORRECT'
        elif a is True and b is True:tag='CORRECT_RETAINED'
        else:tag='INCORRECT_RETAINED'
        group[tag]+=1
        for k in ('extra_tests_received','extra_tests_final_provided','extra_tests_final_cited'):extra[k]+=pair[k]
        transitions.append({**pair,'transition':tag,'B3_final_role':x['final_role'],'B4_final_role':y['final_role'],'B4_last_LLM_action':y['last_LLM_action'],'B4_STOP_origin':y['STOP_origin']})
    assert group['CORRECT_TO_UNRESOLVED']==2 and group['INCORRECT_TO_CORRECT']==1
    # Compare same-patient initial requests to prior technical Smoke without merging outcomes.
    smoke=ROOT/'outputs/private/research_evaluation/B34_UTILITY_VOI_SMOKE_v1/u1_voi_smoke_002/run_u1_voi_smoke_002/requests/R1';repeat=[]
    for condition,folder in (('B3','Episode0001'),('B4','Episode0002')):
        current=look[1,condition][0];base=run/'requests'/('Episode%04d'%current['episode_ordinal']);old=read(smoke/folder/'Request0001_before.json');new=read(base/'Request0001_before.json')
        oa=read(smoke/folder/'Request0001_after.json');na=read(base/'Request0001_after.json')
        repeat.append({'condition':condition,'same_initial_request':old['request_fingerprint']==new['request_fingerprint'],'same_first_raw_response':oa['response_envelope']['message']['content']==na['response_envelope']['message']['content']})
    # U0 historical M5 path is only a replay reference, not a matched new utility comparison.
    historic=read(ROOT/'outputs/private/research_evaluation/ORIGINAL50_D0_D2_EXPERIMENT_v1/original50_d0d2_full_003/run_original50_d0d2_full_003/analysis_private.json')
    same=0
    for i in range(1,11):
        old=next(r['facts']['voi_trace'] for r in historic['rows'] if r['dataset_view']=='D2' and r['condition']=='B4' and r['patient_ordinal']==i);now=look[i,'B4'][0]['facts']['voi_trace']
        keys=lambda h:[(s['selected_action'],s.get('action_id'),s['belief']) for s in h]
        same+=keys(old)==keys(now)
    report={'plan_id':NAME,'status':c['status'],'episodes_completed':20,'registered_patients':10,'generation':123,'wall_seconds':c['wall_seconds'],
        'start_KST':dt(started['at']).astimezone(timezone(timedelta(hours=9))).isoformat(),'conditions':saved['conditions'],'patient_transitions':transitions,
        'transition_counts':dict(group),'extra_test_counts':dict(extra),'STOP_origins':dict(stop_counts),'proposal_counts':dict(core_proposals),
        'audit_counts':dict(audits),'prior_smoke_initial_comparison':repeat,'same_historical_M5_path_and_belief_pairs':same,
        'B_collection_found':(SOURCE_PRIVATE/'u2_source_snapshot_collect_001/run_u2_source_snapshot_collect_001/collection.json').exists(),
        'new_generation':0,'fitting':0,'clinical_accuracy':'NOT_EVALUATED','research_evaluation_ready':False}
    write(PR/'patient_details.json',private);write(PUB/'results_analysis.json',report)
    with (PUB/'paired_transitions.csv').open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(transitions[0]));w.writeheader();w.writerows(transitions)
    changed=[f for f,h in before.items() if sha_file(f)!=h];assert not changed;write(PUB/'preservation.json',{'files_checked':len(before),'changed':0,'new_generation':0})
    print(json.dumps({k:v for k,v in report.items() if k not in ('patient_transitions','conditions')}))

if __name__=='__main__':main()
