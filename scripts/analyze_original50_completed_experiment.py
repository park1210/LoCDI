"""Read-only completed D0/D2 audit; no network, generation, fitting or changes to runs."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_original50_d0_d2 import load,smoke_proof,audit_requests
from app.locdi.original_source_inventory import read,original_cases
from app.locdi.current_ablation_runner import _no_reference
from app.locdi.evidence_subset_wire import FIELD,decode
from app.locdi.llm_setup import digest
from jsonschema import Draft202012Validator

NS='ORIGINAL50_D0_D2_RESULTS_ANALYSIS_v1'
PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dt(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    PUB.mkdir(exist_ok=False);PR.mkdir(exist_ok=False)
    names=('original50_d0d2_smoke_005','original50_d0d2_full_003');protected={}
    for name in names:
        pub,pr,_,_=load(name)
        for base in (pub,pr):
            for f in base.rglob('*'):
                if f.is_file():protected[str(f.resolve())]=sha(f)
    write(PR/'preservation_before.json',protected)
    smoke_proof(names[0]);run_reports={};violations=[];detail=[];before_counts=Counter();after_counts=Counter()
    for name in names:
        pub,pr,plan,bp=load(name);run=pr/plan['run_id'];c=read(run/'collection.json');saved=read(pub/'saved_results.json');analysis=read(run/'analysis_private.json')
        lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        assert lines==c['rows'] and c['generation']==saved['actual_generation']==sum(r['request_count'] for r in lines)
        assert c['status']=='COMPLETED' and len(lines)==plan['registered_episodes'] and all(r['status']=='COMPLETED' for r in lines)
        assert c['plan_fingerprint']==plan['fingerprint'];assert len(analysis['rows'])==len(lines)
        total=0;memory_checks=0;receipt_checks=0;unmatched=0
        for row in lines:
            folder=run/'requests'/row['dataset_view']/('Episode%04d'%row['view_episode_ordinal']);audit=read(folder/'dataset_view_audit.json')
            audit_requests(folder,row,plan,audit['actual_initial_state']);runtime=read(folder/'runtime_snapshot_private.json')
            receipts={r['summary']:r['observed_minute'] for r in runtime['test_results']}
            for bf in sorted(folder.glob('Request*_before.json')):
                b=read(bf);af=bf.with_name(bf.name.replace('_before','_after'));a=read(af);request=b['request'];payload=json.loads(request['messages'][1]['content'])
                _no_reference(payload);assert b['request_fingerprint']==digest(request);assert a['status']=='VALID' and a['HTTP_status']==200
                value=json.loads(a['response_envelope']['message']['content']);Draft202012Validator(request['format']).validate(value)
                code=value[FIELD];selected=[] if code=='NONE' else [int(x) for x in code.split(',')]
                assert len(selected)<=4 and len(selected)==len(set(selected)) and all(0<=i<len(payload['evidence']) for i in selected)
                assert value['action']=='diagnose' or not selected
                core=read(bf.with_name(bf.name.replace('_before','_core_response')))['decoded_response'];assert core['evidence_indices']==selected
                memory=payload['observation']['longitudinal_memory'];cutoff=dt(memory['as_of'])
                for m in memory['records']:
                    memory_checks+=1
                    if dt(m['available_at'])>cutoff:violations.append('FUTURE_MEMORY')
                minute=payload['observation']['elapsed_minutes']
                for e in payload['evidence_records']:
                    if e['text'] in receipts:
                        receipt_checks+=1
                        if minute<receipts[e['text']]:violations.append('BEFORE_RESULT_RECEIPT')
                    else:unmatched+=1
                total+=1;before_counts[name]+=1;after_counts[name]+=1
        assert total==c['generation']
        run_reports[name]={'status':c['status'],'registered_episodes':len(lines),'completed':len(lines),'generation':total,'wall_seconds':c['wall_seconds'],
            'matched_before_after':total,'HTTP200_schema_valid':total,'memory_timestamp_checks':memory_checks,'evidence_receipt_checks':receipt_checks,
            'evidence_records_not_matched_to_test_receipts':unmatched,'model_context_pin_checks':'PASS','current_code_source_pins':'PASS','source_reference_timing_review':'NOT_REVIEWED'}
        if name==names[-1]:full=(plan,bp,c,saved,analysis)
    assert not violations,violations
    plan,bp,c,saved,analysis=full;lookup={(r['patient_ordinal'],r['condition'],r['dataset_view']):r for r in analysis['rows']}
    utility=read(ROOT/'outputs/private/research_evaluation/B34_RUNTIME_RECOVERY_20261121_v1/development_routing_evidence.json')
    samples=utility['routing_artifact']['samples'];agent_counts={agent:dict(Counter(str(s['correct']) for s in samples if s['agent_id']==agent)) for agent in sorted({s['agent_id'] for s in samples})}
    routing={};memory={};paired={};b4=[];private=[]
    for cond in plan['conditions']:
        p=[x for x in analysis['paired'] if x['condition']==cond]
        paired[cond]={'registered_pairs':10,'correct_retained':sum(x['CORRECT_RETAINED'] for x in p),'gain':sum(x['INCORRECT_TO_CORRECT'] for x in p),
            'loss':sum(x['CORRECT_TO_INCORRECT'] for x in p),'memory_ids_changed':sum(x['memory_ids_changed'] for x in p)}
        for view in plan['dataset_views']:
            rows=[r for r in analysis['rows'] if r['condition']==cond and r['dataset_view']==view]
            memory[view+'_'+cond]={'final_selected_counts':dict(Counter(len(r['facts']['last_memory']) for r in rows)),
                'server_prompt_tokens':saved['conditions'][view][cond]['reported_usage']['prompt_tokens']}
        for i in range(1,11):
            x=lookup[i,cond,'D0'];y=lookup[i,cond,'D2']
            private.append({'panel_ordinal':i,'condition':cond,'D0_prediction':x['facts']['prediction'],'D2_prediction':y['facts']['prediction'],
                'source_reference':x['facts']['reference'],'D0_exact':x['facts']['source_exact'],'D2_exact':y['facts']['source_exact'],
                'D0_actions':x['facts']['action_sequence'],'D2_actions':y['facts']['action_sequence'],
                'D0_memory_ids':[r['memory_id'] for r in x['facts']['last_memory']],'D2_memory_ids':[r['memory_id'] for r in y['facts']['last_memory']]})
            if cond=='B4':
                hx=x['facts']['voi_trace'];hy=y['facts']['voi_trace']
                path=lambda h:[(t.get('selected_action'),t.get('action_id'),t.get('target')) for t in h]
                same_path=path(hx)==path(hy);same_beliefs=[t['belief'] for t in hx]==[t['belief'] for t in hy]
                b4.append({'panel_ordinal':i,'same_voi_path':same_path,'same_belief_trajectory':same_beliefs,
                    'D0_Algorithm_Hospital_same':x['result']['prediction']['algorithmic_diagnosis_choice']==x['facts']['prediction']['diagnosis'],
                    'D2_Algorithm_Hospital_same':y['result']['prediction']['algorithmic_diagnosis_choice']==y['facts']['prediction']['diagnosis']})
    for view in ('D0','D2'):
        for cond in ('B3','B4'):
            rows=[r for r in analysis['rows'] if r['condition']==cond and r['dataset_view']==view];traces=[t for r in rows for t in r['facts']['routing_trace']]
            margins=[];weights=[];matches=[]
            for t in traces:
                scores=sorted(t['specialty_scores'].values(),reverse=True);margins.append(scores[0]-scores[1])
                for value in t['utility_estimates'].values():weights.append(value['effective_weight']);matches.append(value['matched_samples'])
            routing[view+'_'+cond]={'trace_decisions':len(traces),'selected_distribution':dict(Counter(t['selected_specialty'] for t in traces)),
                'first_selection_patient_distribution':dict(Counter(r['facts']['routing_trace'][0]['selected_specialty'] for r in rows)),
                'margin_range':[min(margins),max(margins)],'matched_samples_range':[min(matches),max(matches)],'effective_weight_range':[min(weights),max(weights)]}
    report={'scope':'COMPLETED_RUN_READ_ONLY_ANALYSIS','runs':run_reports,'Full_only_conditions':saved['conditions'],'Full_only_paired':paired,
        'D2_minus_D0':saved['D2_minus_D0_by_condition'],'routing':routing,'utility_train_groups':utility['complete_train_groups'],'agent_proxy_label_counts':agent_counts,
        'memory':memory,'B4_same_path_pairs':sum(r['same_voi_path'] for r in b4),'B4_same_belief_pairs':sum(r['same_belief_trajectory'] for r in b4),
        'B4_Algorithm_Hospital_same_counts':{v:sum(r[v+'_Algorithm_Hospital_same'] for r in b4) for v in ('D0','D2')},
        'Smoke_merged_into_Full':False,'clinical_accuracy':'NOT_EVALUATED','independent_clinical_review':False,'D2_label_conditioned':True,
        'new_LLM_generation':0,'utility_fitting':0,'boundary_violations_observed':violations}
    write(PR/'patient_level_analysis.json',{'paired':private,'B4':b4});write(PUB/'results_analysis.json',report)
    changed=[f for f,h in protected.items() if not Path(f).exists() or sha(f)!=h];assert not changed
    write(PUB/'preservation_verification.json',{'files_checked':len(protected),'changed':len(changed),'actual_LLM_generation':0})
    print(json.dumps({'runs':run_reports,'paired':paired,'routing':routing,'B4_same_paths':report['B4_same_path_pairs'],'B4_same_beliefs':report['B4_same_belief_pairs'],'B4_Algorithm_Hospital_same_counts':report['B4_Algorithm_Hospital_same_counts'],'artifacts_unchanged':len(protected)}))


if __name__=='__main__':main()
