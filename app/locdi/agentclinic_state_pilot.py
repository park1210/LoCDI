"""Pinned A/C study gate and immutable external-policy pilot. No auto approval."""
from datetime import datetime,timezone,timedelta
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_freeze import sha,load
from app.locdi.external_reference_readiness import pinned_private_json
from app.locdi.episode_exporter import _write
from app.locdi.agentclinic_pilot_readiness import prepare_plan,output_paths,PinnedPilotApproval
from app.locdi.agentclinic_live_preflight import proposal,SOURCE
from app.locdi.agentclinic_doctor_client import create_live_client
from app.locdi.agentclinic_doctor_exporter import export_doctor_episode
from app.locdi.agentclinic_scenario_adapter import load_scenarios
from app.locdi.agentclinic_state_runtime import run_state_episode
from app.locdi.agentclinic_context_bootstrap import observe_context

def plan(condition):
    if condition not in ('A','C'):raise ValueError('STATE_CONDITION_REQUIRED')
    base=prepare_plan('c3_16_'+condition+'_001',proposal())
    files=['app/locdi/agentclinic_state_actions.py','app/locdi/agentclinic_state_runtime.py','app/locdi/agentclinic_state_pilot.py']
    return {'schema_version':'agentclinic.state_study_plan.v1','condition':condition,'base_plan':base,
        'policy_enabled':condition=='C','code_hashes':{p:sha(ROOT/p) for p in files},
        'cohort':[1,2,3,4,5],'condition_budget':50,'combined_budget':100,'decision_limit':10,
        'retry':0,'repair':0,'order':['A','C'],'resume':False}

class StudyApproval:
    def __init__(self,path,expected):self.path,self.expected=path,expected
    def verify(self,condition):
        r=pinned_private_json(self.path,self.expected);now=datetime.now(timezone.utc)
        if (r.get('schema_version')!='agentclinic.state_study_authorization.v1'
            or r.get('approval_source')!='EXPLICIT_CURRENT_USER_C3_16_APPROVAL'
            or r.get('approved_by')!='USER_LOCAL_SERVER_OPERATOR' or not r.get('user_statement')
            or r.get('execute_authorized') is not True or r.get('order')!=['A','C']
            or r.get('cohort')!=[1,2,3,4,5] or r.get('combined_budget')!=100
            or r.get('condition_budget')!=50 or r.get('retry')!=0 or r.get('repair')!=0
            or r.get('scoring_scope')!='OFFLINE_EXACT_MATCH_LEXICAL_ONLY'):
            raise ValueError('NEW_STATE_STUDY_AUTHORIZATION_REQUIRED')
        issued=datetime.fromisoformat(r['issued_at']);expires=datetime.fromisoformat(r['expires_at'])
        if issued.tzinfo is None or expires.tzinfo is None or not issued<=now<expires or expires-issued>timedelta(hours=24):
            raise ValueError('STATE_STUDY_AUTHORIZATION_EXPIRED')
        current=plan(condition)
        if r['plans'].get(condition)!=digest(current):raise ValueError('STATE_STUDY_PLAN_CHANGED')
        if condition=='C':
            public,_=output_paths('c3_16_A_001');prior=load(public/'metrics/summary.json')
            if (prior.get('study_plan_fingerprint')!=r['plans']['A'] or prior.get('mode')!='LIVE'
                or prior.get('integrity_stop') is not False):raise ValueError('A_TERMINAL_AND_INTEGRITY_REQUIRED')
        return current

def run_live(condition,*,study=None,authority=None,execute=False,allow_llm=False):
    if execute is not True or allow_llm is not True or not isinstance(study,StudyApproval) or not isinstance(authority,PinnedPilotApproval):
        raise ValueError('STATE_AND_PINNED_PILOT_AUTHORIZATION_REQUIRED')
    p=study.verify(condition);base=p['base_plan'];public,private=output_paths(base['namespace'])
    if public.exists() or private.exists():raise ValueError('OUTPUT_NAMESPACE_EXISTS_NO_RESUME')
    doctor_authority,counter,_=authority.verify(base)
    client=create_live_client(proposal(),execute=True,allow_llm=True,authority=doctor_authority)
    public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    _write(p,'reports/plan.json',public,'reports')
    rows=[{'ordinal':i,'status':'NOT_EXECUTED'} for i in range(1,6)];stop=None;integrity=False
    attempt=0;total_tokens=0
    try:
        for index,s in enumerate(load_scenarios(SOURCE).scenarios[:5]):
            def checkpoint(phase,data):
                nonlocal attempt,total_tokens
                if phase=='before':
                    study.verify(condition);authority.verify(base,load_tokenizer=False)
                    attempt+=1;total_tokens+=data['input_tokens']
                    if attempt>50 or total_tokens>base['max_total_input_tokens']:raise ValueError('GLOBAL_BUDGET_EXHAUSTED')
                _write(dict(data,ordinal=index+1,study_plan_fingerprint=digest(p)),
                    'checkpoints/Request%04d_%s.json'%(attempt,phase),private,'checkpoints')
            episode,trace=run_state_episode(s,client,enabled=condition=='C',counter=counter,checkpoint=checkpoint,
                context_observer=lambda:observe_context(base['doctor_plan']['definition']['legacy_config']))
            row=rows[index];result=episode['result']
            row.update(status=result['status'],stop_reason=result['stop_reason'],diagnosis_unresolved=result['diagnosis_unresolved'],
                generation_count=episode['real_generation_calls'],source_turn_count=result['turn_count'],
                decision_turn_count=trace['decision_turn_count'],
                information_proposals=sum(e['proposed_action'] in ('ASK','EXAMINE','TEST') for e in trace['events']),
                repeated_proposals=sum(e['repeated_proposal'] for e in trace['events']),
                rejections=sum(e['action_status']=='REJECTED' for e in trace['events']),
                unique_observations=sum(e['new_observation_count'] for e in trace['events']),
                action_counts={k:sum(e['action']==k for e in episode['trace']) for k in ('ASK','EXAMINE','TEST','DIAGNOSE')})
            try:
                export_doctor_episode(episode,public_root=public,private_root=private,name='Scenario%03d'%(index+1))
                for root in (public,private):_write(trace,'policy_traces/Scenario%03d.json'%(index+1),root,'policy_traces')
                row['export_status']='EXPORTED'
            except Exception:
                row.update(status='failed',export_status='FAILED_OR_PARTIAL');stop='EXPORT_FAILED';integrity=True
            if result['status']!='completed' or stop:
                stop=stop or result['stop_reason']
                integrity=integrity or result['stop_reason'] in {'ACTUAL_CONTEXT_UNVERIFIED_OR_MISMATCH','EXPORT_FAILED'}
                # Generic client/contract errors are conservatively pair-stopping:
                # a new execution approval is required to investigate uncertainty.
                integrity=integrity or result['status']=='failed'
                break
    finally:client.close()
    summary={'schema_version':'agentclinic.state_study_summary.v1','mode':'LIVE','condition':condition,
        'study_plan_fingerprint':digest(p),'rows':rows,'stop_reason':stop,'integrity_stop':integrity,
        'generation_count':client.transport.request_count,'identity_count':client.transport.identity_check_count,
        'retry':0,'repair':0,'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}
    _write(summary,'metrics/summary.json',public,'metrics');return summary
