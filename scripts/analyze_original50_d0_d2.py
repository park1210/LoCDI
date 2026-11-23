"""Read saved paired traces only. No model, fitting or Source label changes."""
from collections import Counter
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read
from app.locdi.common_prediction_contract import extract_core_prediction
from app.locdi.synthea_diagnosis_audit import matches
from scripts.run_original50_d0_d2 import load,write_equal

def facts(folder,result,reference):
    decisions=read(folder/'actual_core_decisions.json') if (folder/'actual_core_decisions.json').exists() else []
    status=(result or {}).get('status','not_attempted');pred=extract_core_prediction(decisions,status)
    if result and any(pred[k]!=result['prediction'][k] for k in ('certainty','diagnosis','department')):raise ValueError('SAVED_CORE_PREDICTION_MISMATCH')
    usage=Counter();payloads=[];response_rows=[]
    for before in sorted(folder.glob('Request*_before.json')):payloads.append(json.loads(read(before)['request']['messages'][1]['content']))
    for after in sorted(folder.glob('Request*_after.json')):
        a=read(after);e=a.get('response_envelope',{});usage.update(prompt_tokens=e.get('prompt_eval_count',0),output_tokens=e.get('eval_count',0),
            received=a.get('response_received') is True,HTTP_wall_seconds=a.get('wall_seconds',0))
        response_rows.append({'status':a.get('status'),'HTTP_status':a.get('HTTP_status'),'error_type':a.get('error_type')})
    runtime=read(folder/'runtime_snapshot_private.json') if (folder/'runtime_snapshot_private.json').exists() else {};voi=runtime.get('voi_state',{})
    exact=matches(pred['diagnosis'],reference['accepted_diagnoses'])
    last=payloads[-1] if payloads else {};routing=voi.get('primary_routing_history',[]);information=voi.get('history',[])
    return {'prediction':pred,'source_exact':exact['source_exact'] if status=='completed' else None,
        'normalized':exact['normalized_match'] if status=='completed' else None,
        'department_match':pred['department'] in reference['accepted_departments'] if status=='completed' else None,
        'algorithmic_source_exact':matches((result or {}).get('prediction',{}).get('algorithmic_diagnosis_choice'),reference['accepted_diagnoses'])['source_exact'],
        'usage':dict(usage),'action_sequence':decisions,'final_minute':decisions[-1]['minute'] if decisions else None,
        'memory_snapshots':[p['observation'].get('longitudinal_memory',{}) for p in payloads],
        'last_memory':last.get('observation',{}).get('longitudinal_memory',{}).get('records',[]),
        'released_evidence':last.get('evidence',[]),'routing_trace':routing,'voi_trace':information,
        'responses':response_rows,'request_count_checkpoint':len(payloads),'reference':reference}

def transitions(first,second):
    x=first['source_exact'];y=second['source_exact']
    return {'CORRECT_TO_INCORRECT':x is True and y is False,'INCORRECT_TO_CORRECT':x is False and y is True,
        'CORRECT_RETAINED':x is True and y is True,
        'SUPPORTED_TO_UNRESOLVED':first['prediction']['certainty']=='supported' and second['prediction']['certainty']=='unresolved',
        'unscored_or_not_completed_present':x is None or y is None}

def analyze(name):
    pub,pr,p,bp=load(name);run=pr/p['run_id'];c=read(run/'collection.json');lines=[json.loads(s) for s in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if lines!=c['rows'] or c['plan_fingerprint']!=p['fingerprint'] or sum(r['request_count'] for r in lines)!=c['generation']:raise ValueError('SAVED_COLLECTION_TRACE_COUNTER_MISMATCH')
    expected=read(pr/'tasks.json')
    if len(lines)!=len(expected) or any(any(row.get(k)!=v for k,v in task.items()) for row,task in zip(lines,expected)):raise ValueError('REGISTERED_PAIRED_TASKS_MISMATCH')
    if any(r['status'] not in ('COMPLETED','FAILED','INCOMPLETE','NOT_ATTEMPTED') for r in lines):raise ValueError('UNKNOWN_EPISODE_STATUS')
    panel=read(bp/'panel.json');detailed=[];scored=[]
    for r in lines:
        folder=run/'requests'/r['dataset_view']/('Episode%04d'%r['view_episode_ordinal']);f=facts(folder,r['result'],panel[r['patient_ordinal']-1]['reference'])
        if r['status']!='COMPLETED':
            for key in ('source_exact','normalized','department_match','algorithmic_source_exact'):f[key]=None
        if f['request_count_checkpoint']!=r['request_count']:raise ValueError('ROW_CHECKPOINT_COUNTER_MISMATCH')
        detailed.append({**r,'facts':f});scored.append({k:r[k] for k in ('patient_ordinal','dataset_view','condition','status','request_count')} | {k:f[k] for k in ('source_exact','normalized','department_match','algorithmic_source_exact')})
    summary={}
    for view in p['dataset_views']:
        summary[view]={}
        for condition in p['conditions']:
            chosen=[r for r in detailed if r['dataset_view']==view and r['condition']==condition];rr=[r['facts'] for r in chosen];usage=Counter()
            for r in rr:usage.update(r['usage'])
            choices=Counter(t.get('selected_specialty') or 'ABSTAINED_OR_NO_SELECTION' for r in rr for t in r['routing_trace'])
            information=Counter(t.get('selected_action') for r in rr for t in r['voi_trace'])
            correct=sum(r['source_exact'] is True for r in rr)
            actions=Counter()
            for row in chosen:actions.update((row.get('result') or {}).get('action_counts',{}))
            summary[view][condition]={'registered':p['registered_patients'],'states':dict(Counter(r['status'] for r in chosen)),
                'source_exact_successes':correct,'source_exact_agreement_registered':correct/p['registered_patients'],
                'supported':sum(r['status']=='COMPLETED' and r['facts']['prediction']['certainty']=='supported' for r in chosen),
                'unresolved':sum(r['status']=='COMPLETED' and r['facts']['prediction']['certainty']=='unresolved' for r in chosen),
                'source_department_successes':sum(r['department_match'] is True for r in rr),
                'algorithmic_source_success_separate':sum(r['algorithmic_source_exact'] is True for r in rr) if condition=='B4' else None,
                'generation':sum(r['request_count'] for r in chosen),'reported_usage':dict(usage),'action_counts':dict(actions),
                'routing_selection_distribution':dict(choices),'voi_selection_distribution':dict(information),
                'planner_calls':sum((r.get('result') or {}).get('computation',{}).get('planner_calls',0) for r in chosen),
                'core_episode_wall_seconds':sum((r.get('result') or {}).get('computation',{}).get('wall_seconds',0) for r in chosen)}
    paired=[];lookup={(r['patient_ordinal'],r['condition'],r['dataset_view']):r for r in detailed}
    for i in range(1,p['registered_patients']+1):
        for condition in p['conditions']:
            x=lookup[i,condition,'D0'];y=lookup[i,condition,'D2'];a=x['facts'];b=y['facts']
            paired.append({'patient_ordinal':i,'condition':condition,'D0_status':x['status'],'D2_status':y['status'],
                'D0_source_exact':a['source_exact'],'D2_source_exact':b['source_exact'],**transitions(a,b),
                'memory_ids_changed':[r['memory_id'] for r in a['last_memory']]!=[r['memory_id'] for r in b['last_memory']],
                'released_evidence_changed':a['released_evidence']!=b['released_evidence'],
                'D0_generation':x['request_count'],'D2_generation':y['request_count']})
    private={'plan_fingerprint':p['fingerprint'],'rows':detailed,'paired':paired,'clinical_accuracy':'NOT_EVALUATED'}
    write_equal(run/'analysis_private.json',private)
    # Separate historical comparison, never merge old outcomes or call it a matched effect.
    historical_path=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001/run_runtime_recovery_all_001/collection.json'
    if historical_path.exists():
        historical=read(historical_path);old={(r['patient_ordinal'],r['condition']):r for r in historical['rows']}
        comp=[]
        for r in detailed:
            if r['dataset_view']!='D0':continue
            o=old.get((r['patient_ordinal'],r['condition']));comp.append({'patient_ordinal':r['patient_ordinal'],'condition':r['condition'],
                'same_contract':False,'previous_plan':'runtime_recovery_all_001','current_plan':name,'previous_result':o,'current':r['facts'],
                'interpretation':'SEPARATE_HISTORICAL_CONTEXT_DIFFERENT_WIRE_CODE_NOT_CAUSAL'})
        write_equal(run/'historical_comparison_private.json',comp)
    report={'run_id':p['run_id'],'plan_fingerprint':p['fingerprint'],'conditions':summary,'paired':paired,
        'registered_patients':p['registered_patients'],'actual_generation':c['generation'],'wall_seconds':c['wall_seconds'],
        'source_exact_agreement_primary':'SUCCESS_COUNT / REGISTERED_PATIENTS; statuses/unresolved separately retained; no false correctness fitting labels',
        'clinical_accuracy':'NOT_EVALUATED','independent_clinical_review':False,'research_evaluation_ready':False,
        'D2_is_label_conditioned':True,'independent_generalization_claim':False,'existing_results_merged':False}
    report['D2_minus_D0_by_condition']={cond:{'registered_denominator':p['registered_patients'],
        'source_exact_agreement_difference_pp':100*(summary['D2'][cond]['source_exact_agreement_registered']-summary['D0'][cond]['source_exact_agreement_registered']),
        'completed_pairs':sum(r['D0_status']==r['D2_status']=='COMPLETED' for r in paired if r['condition']==cond),
        'correct_retained':sum(r['CORRECT_RETAINED'] for r in paired if r['condition']==cond),
        'correct_to_incorrect':sum(r['CORRECT_TO_INCORRECT'] for r in paired if r['condition']==cond),
        'supported_to_unresolved':sum(r['SUPPORTED_TO_UNRESOLVED'] for r in paired if r['condition']==cond)} for cond in p['conditions']}
    write_equal(pub/'saved_results.json',report)
    csvpath=run/'paired_results.csv'
    if not csvpath.exists():
        with csvpath.open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    return {'status':'SAVED_TRACE_ANALYSIS_COMPLETE','plan_id':name,'actual_generation':c['generation'],'new_LLM_generation':0,
        'public_summary':str((pub/'saved_results.json').relative_to(ROOT)),'private_comparison':str((run/'analysis_private.json').relative_to(ROOT))}
if __name__=='__main__':
    import argparse
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('--plan-id',required=True);print(json.dumps(analyze(q.parse_args().plan_id)))
