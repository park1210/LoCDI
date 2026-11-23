"""Explicit Frozen-M5 development interface over HospitalCaseRuntime/InformationPolicy.

This is Source-time execution, not the HospitalParallelScheduler resource model.
The original clinical/research artifact gate stays unchanged and BLOCKED.
"""
from copy import deepcopy
import math
from unittest.mock import patch

from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.clinical_observation import prepare_encounter,doctor_observation,memory_query,encounter_evidence
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.information_outcomes import InformationOutcomes
from app.core.information_value import InformationPolicy,candidate_actions
from app.core.diagnostic_evidence import diagnostic_evidence
from app.core.diagnostic_belief import terminal_choice
from app.core.voi_planning import plan_voi
from app.locdi.frozen_m5_runtime_adapter import ReleasedCategoryView,bind_value,candidate_case
from app.locdi.llm_setup import digest

VERSION='locdi.frozen_m5.source_time_hospital.development.v1'


def public_visit(visit):
    return {k:deepcopy(visit[k]) for k in ('patient_id','visit_id','case_id','decision_time','values')}


def validate_development_encoding(model,visit,candidate):
    """Validate Source bindings, not just an encoding string or reviewer declaration."""
    if model['provenance'].get('observation_encoding')!='c4_30_source_discrete_v2':
        raise ValueError('NOT_FROZEN_SOURCE_ENCODING')
    if model['horizon']!=2 or len(model['actions'])!=8:raise ValueError('FIXED_H2_MENU8_REQUIRED')
    if any(o['supports_diagnoses'] for a in model['actions'] for o in a['outcomes']):
        raise ValueError('UNREVIEWED_DIAGNOSIS_SUPPORT_NOT_ALLOWED')
    if visit['patient_id'] in model['provenance']['training_patients']:raise ValueError('FIT_EVALUATION_PATIENT_OVERLAP')
    expected=candidate_case(model,public_visit(visit),candidate['past_visits'])
    if digest(candidate)!=digest(expected):raise ValueError('CANDIDATE_SOURCE_OR_REFERENCE_CHANGED')
    rows=[bind_value(model,visit,k) for k in model['provenance']['action_key_mapping']]
    return {'status':'DEVELOPMENT_ENCODING_CONTRACT_VALID','source_bindings':len(rows),
        'clinical_research_gate':'UNCHANGED_BLOCKED','research_evaluation_ready':False,
        'source_encoding_tag_changed':False,'source_time':'RECORDED_TIME_PROXY',
        'terminal':'algorithmic_choice_separate_from_Hospital_unresolved'}


class SourceBoundOutcomes:
    """Exact raw-to-category view with private selected/released Source capability."""
    def __init__(self,model,visit):
        self.original=InformationOutcomes(model);self.view=ReleasedCategoryView(model,public_visit(visit))
        self.fingerprint=self.original.fingerprint;self.remaining_budget=5
    @property
    def artifact(self):
        # Runtime depth clipping is the same finite remaining-budget cap used
        # by the existing Source Replay, not a model/threshold rewrite.
        result=deepcopy(self.original.artifact)
        result['horizon']=max(1,min(2,self.remaining_budget))
        return result
    def observe(self,observed):
        expected_findings=[];expected_history=[]
        for row in self.view.released:
            raw=next(e for e in self.events if e['target']==row['target'])
            if raw['action']=='Test':expected_findings.append(raw['target']+': '+raw['evidence'])
            else:expected_history.append((raw['target'],raw['evidence']))
        if observed.get('objective_findings',[])!=expected_findings:
            raise ValueError('UNBOUND_OR_UNRELEASED_OBJECTIVE_FINDING')
        if [(r['question'],r['answer']) for r in observed.get('history',[])]!=expected_history:
            raise ValueError('UNBOUND_OR_UNRELEASED_HISTORY')
        canonical=self.view.observation()
        canonical['longitudinal_memory']=deepcopy(observed.get('longitudinal_memory',{}))
        return self.original.observe(canonical)


class SourceTimeScheduler:
    """Selection->Source-clock advance->actual event release; no invented resource delay."""
    def __init__(self,runtime,model_view):self.runtime=runtime;self.model_view=model_view;self.events=[]
    def apply(self,proposal):
        rt=self.runtime;kind=proposal['action'];target=proposal['target']
        if kind=='diagnose':
            if proposal.get('certainty')!='unresolved' or proposal.get('evidence_indices'):
                raise ValueError('SUPPORTED_DIAGNOSIS_NOT_AUTHORIZED')
            rt.is_completed=True;rt.latest_action='er_finalize';rt.workflow_state='completed'
            self.events.append({'kind':'terminal_unresolved','minute':rt.current_time_min,'supported':False})
            return
        event_kind={'ask_patient':'Ask','request_test':'Test'}.get(kind)
        source=next((e for e in rt.case_data['current_encounter_events'] if e['action']==event_kind and e['target']==target),None)
        if source is None:raise ValueError('ACTION_NOT_IN_EXACT_SOURCE_MENU')
        aid=next(a['id'] for a in self.model_view.original.artifact['actions'] if a['type']==kind and a['target']==target)
        selected_minute=rt.current_time_min
        rt.current_time_min=max(rt.current_time_min,source['available_minute'])
        released=encounter_evidence(rt,event_kind,target)
        if released is None:raise ValueError('SOURCE_NOT_RELEASED')
        row=self.model_view.view.release(aid,selected=aid,minute=rt.current_time_min)
        if kind=='ask_patient':
            rt.question_history.append({'question':target,'answer':released['evidence'],'answer_status':'known'})
            rt.latest_action='ask_history'
        else:
            rt.completed_tests.append(target);rt.test_results.append({'test_name':target,'result':released['evidence']})
            rt.objective_findings.append(target+': '+released['evidence']);rt.latest_action='order_test'
        self.events.append({'kind':kind,'action_id':aid,'outcome_id':row['outcome_id'],
            'selected_at_minute':selected_minute,'released_at_minute':rt.current_time_min,
            'available_minute':source['available_minute'],'availability_basis':'RECORDED_TIME_PROXY'})


def run_source_time_visit(model,visit,past_visits,*,max_actions=5):
    """No LLM, no Reference/scoring. Call only from approved CLI or TEST_ONLY tests."""
    if type(max_actions) is not int or not 0<=max_actions<=5:raise ValueError('INFORMATION_BUDGET')
    original=digest(model);v=public_visit(visit);case=candidate_case(model,v,past_visits)
    gate=validate_development_encoding(model,v,case)
    rt=HospitalCaseRuntime(case_id=v['case_id'],case_data=prepare_encounter(case),arrival_time_min=0,arrival_index=0,
        health_state='UNKNOWN_SOURCE',triage_level=0,workflow_state='development_source_time')
    store=PatientMemoryStore([r for p in past_visits for r in p['records']])
    policy=InformationPolicy(model,selector='bayesian_voi');bound=SourceBoundOutcomes(model,v)
    bound.events=case['current_encounter_events'];policy.model=bound
    scheduler=SourceTimeScheduler(rt,bound);decisions=[];plans=[];retrievals=[]
    used=0
    while not rt.is_completed:
        observed=doctor_observation(rt)
        rt.memory_context=store.retrieve(case['longitudinal'],memory_query(observed),'selective',5)
        observed['longitudinal_memory']=deepcopy(rt.memory_context)
        retrievals.append({'minute':rt.current_time_min,'selected_count':len(rt.memory_context['records']),
            'record_ids':[r['memory_id'] for r in rt.memory_context['records']]})
        bound.remaining_budget=max_actions-used
        kinds=['diagnose'] if used>=max_actions else ['ask_patient','request_test','diagnose']
        tests=[] if used>=max_actions else [e['target'] for e in case['current_encounter_events']
                                            if e['action']=='Test' and e['target'] not in rt.completed_tests]
        evidence,_=diagnostic_evidence(rt)
        # Existing HospitalLLMPolicy also permits unresolved when an
        # InformationPolicy is active (its documented terminal comparison path).
        calls=[]
        def measured(b,actions,u,**kw):
            value=plan_voi(b,actions,u,**kw);calls.append(deepcopy(value));return value
        with patch('app.core.information_value.plan_voi',new=measured):
            proposal=policy.propose(rt,observed,actions=kinds,tests=tests,departments=[],
                evidence_records=evidence,allow_unresolved=True)
        if proposal is None:raise ValueError('SOURCE_POLICY_UNAVAILABLE_NO_FALLBACK')
        candidates=candidate_actions(rt,kinds,tests,[],bound.artifact)
        if not any(c['action']==proposal['action'] and (c['target'] is None or c['target']==proposal['target']) for c in candidates):
            raise ValueError('PROPOSAL_OUTSIDE_ACTION_VALIDATOR_MENU')
        before=deepcopy(rt.voi_state['belief']);selected=proposal['action']
        scheduler.apply(proposal)
        after,_=bound.observe(doctor_observation(rt))
        decisions.append({'belief_before':before,'belief_after':after,'T_before':terminal_choice(before,model['terminal_utility'])[1],
            'T_after':terminal_choice(after,model['terminal_utility'])[1],'proposal':proposal,
            'remaining_budget':max_actions-used,'plans':calls,'hospital_unresolved_allowed':True})
        plans.extend(calls)
        if selected!='diagnose':used+=1
        if len(decisions)>max_actions+1:raise ValueError('DECISION_BUDGET')
    belief,_=bound.observe(doctor_observation(rt));choice,T=terminal_choice(belief,model['terminal_utility'])
    if digest(model)!=original:raise ValueError('FROZEN_MODEL_MUTATED')
    return {'mode':VERSION,'status':'completed','hospital_runtime_class':'HospitalCaseRuntime',
        'information_policy_class':'InformationPolicy','scheduler_mode':'SourceTimeScheduler_not_resource_simulation',
        'development_gate':gate,'algorithmic_choice':choice,'belief':belief,'terminal_value':T,
        'hospital_supported_diagnosis':None,'hospital_unresolved':True,'runtime_completed':rt.is_completed,
        'actions':scheduler.events,'decisions':decisions,'memory_retrievals':retrievals,'planner_calls':len(plans),
        'evaluated_states':sum(p['evaluated_states'] for p in plans),'LLM_generation':0,'clinical_metrics':'NOT_EVALUATED'}
