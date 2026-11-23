"""Read-only saved C3-16 A/C audit; lexical scoring requires new study approval."""
import json,sys,statistics
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.episode_exporter import _write
from app.locdi.agentclinic_state_pilot import StudyApproval
from app.locdi.agentclinic_pilot_readiness import output_paths
from app.locdi.agentclinic_scenario_adapter import load_scenarios
from app.locdi.agentclinic_live_preflight import SOURCE
from app.locdi.agentclinic_benchmark_scoring import _score
from app.locdi.agentclinic_doctor_exporter import public_doctor_episode

PUBLIC=ROOT/'outputs/research_evaluation/c3_16_v1'
PRIVATE=ROOT/'outputs/private/research_evaluation/c3_16_v1'

def analyze():
    evidence=PRIVATE/'approvals/study_authorization.json';gate=StudyApproval(evidence,sha(evidence));gate.verify('A')
    bundle=load_scenarios(SOURCE);keys={k.scenario_id:k for k in bundle.scoring_keys};conditions={};private_scores={}
    for c in ('A','C'):
        public,private=output_paths('c3_16_'+c+'_001')
        if not (public/'metrics/summary.json').is_file():
            conditions[c]={'status':'NOT_EXECUTED','planned':5};continue
        gate.verify(c);summary=load(public/'metrics/summary.json');rows=[];all_events=[];turns=[];checks=[];scores=[]
        for original in summary['rows']:
            row=dict(original);ordinal=row['ordinal']
            if row['status']=='NOT_EXECUTED':rows.append(row);continue
            name='Scenario%03d'%ordinal;episode=load(private/('episodes/'+name+'.json'));policy=load(private/('policy_traces/'+name+'.json'))
            events=policy['events'];all_events+=events;turns.append(len(events))
            checks.append(policy['episode_fingerprint']==digest(episode))
            checks.append(public_doctor_episode(episode)==load(public/('episodes/'+name+'.json')))
            checks.append(policy==load(public/('policy_traces/'+name+'.json')))
            check_events=[]
            for event in events:
                i=event['source_event_index']
                if i is None:check_events.append(event['executed_action'] is None and event['new_observation_count']==0)
                else:
                    source=episode['trace'][i];check_events.append(source['action']==event['proposed_action'] and source['target']==event['proposed_target'])
            checks.append(all(check_events));checks.append(episode['prompt_fingerprint']==summary_prompt())
            score=_score(episode,keys[episode['scenario_ref']],None);scores.append(score)
            row.update(exact_match=score['exact_match'],scoring_state=score['state'],
                successful_unique_observations=len({e['observation_ref'] for e in episode['trace'] if e['status']=='EXECUTED' and e['observation_ref']}),
                executed_repeats=sum(e['repeat_of'] is not None for e in episode['trace']),
                invalid_source_actions=sum(e['validation']!='VALID' for e in episode['trace']),
                action_sequence=[{'turn':e['turn_index'],'action':e['proposed_action'],'target':e['proposed_target'],
                    'status':e['action_status'],'new_observation_count':e['new_observation_count']} for e in events])
            rows.append(row)
        afters=[load(p) for p in sorted((private/'checkpoints').glob('*_after.json'))]
        befores=[load(p) for p in sorted((private/'checkpoints').glob('*_before.json'))]
        contexts=[a['context_observation'] for a in afters if a.get('context_observation')]
        information=sum(e['proposed_action'] in ('ASK','EXAMINE','TEST') for e in all_events)
        repeated=sum(e['repeated_proposal'] for e in all_events)
        counts=Counter(e['proposed_action'] for e in all_events)
        conditions[c]={'rows':rows,'planned':5,'executed':len(turns),
            'completed':sum(r['status']=='completed' for r in rows),
            'valid_diagnosis_completed':sum(s['state']=='completed' for s in scores),
            'incomplete':sum(r['status']=='incomplete' for r in rows),'failed':sum(r['status']=='failed' for r in rows),
            'not_executed':sum(r['status']=='NOT_EXECUTED' for r in rows),
            'unresolved_terminals':sum(s['state']=='diagnosis_unresolved' for s in scores),
            'generation_count':summary['generation_count'],'identity_count':summary['identity_count'],
            'information_proposals':information,'repeated_proposals':repeated,
            'repeated_proposal_rate':repeated/information if information else 'NOT_APPLICABLE',
            'unique_observations':sum(r.get('successful_unique_observations',0) for r in rows),
            'executed_repeats':sum(r.get('executed_repeats',0) for r in rows),
            'consumed_rejections':sum(e['rejection_reason']=='ALREADY_OBSERVED' for e in all_events),
            'proposal_counts':dict(counts),'decision_turns_mean':statistics.mean(turns) if turns else None,
            'decision_turns_median':statistics.median(turns) if turns else None,
            'exact_match_count':sum(s['exact_match'] for s in scores),'exact_registered_denominator':5,
            'valid_prediction_denominator':sum(s['state']=='completed' for s in scores),
            'retry':0,'repair':0,'export_link_checks_pass':bool(checks) and all(checks),
            'before_count':len(befores),'after_count':len(afters),'actual_context_valid_count':sum(o['status']=='VALID' for o in contexts),
            'context_values':sorted({o.get('loaded_model',{}).get('context_length',0) for o in contexts}),
            'stop_reason':summary['stop_reason'],'integrity_stop':summary['integrity_stop']}
        private_scores[c]=scores
    paired=[i for i in range(1,6) if all(any(r['ordinal']==i and r['status']!='NOT_EXECUTED' for r in conditions[c].get('rows',[])) for c in ('A','C'))]
    completed=all(conditions[c].get('completed')==5 for c in ('A','C'))
    report={'status':'C3_16_COMPARISON_COMPLETED' if completed else 'C3_16_PARTIAL',
        'conditions':conditions,'observed_pairs':paired,'full_paired_execution':len(paired)==5,
        'generation_count':sum(c.get('generation_count',0) for c in conditions.values()),
        'clinical_metrics':'NOT_EVALUATED','clinical_equivalence':'NOT_VERIFIED','research_evaluation_ready':False,
        'repetition_interpretation':'Catalog and constrained JSON schema are part of the policy; no independent cognitive improvement claim',
        'exact_match_scope':'NEW_USER_APPROVED_OFFLINE_LEXICAL_ONLY; no aliases, grading, clinical verdict or fitting'}
    _write(private_scores,'reports/private_scoring.json',PRIVATE,'reports')
    _write(report,'reports/comparison_analysis.json',PUBLIC,'reports')
    return report

def summary_prompt():
    from app.locdi.agentclinic_doctor_client import PROMPT_SHA256
    return PROMPT_SHA256

if __name__=='__main__':
    r=analyze();print(json.dumps({'status':r['status'],'generation_count':r['generation_count'],
        'conditions':{c:{k:v for k,v in r['conditions'][c].items() if k!='rows'} for c in r['conditions']}},ensure_ascii=True))
