"""Opt-in semantic display and explicit LLM finalization after Frozen M5 STOP.

No probabilities, Source facts, clinical supports or validators are modified.
"""
from copy import deepcopy
import json
from app.locdi.llm_setup import digest
from app.locdi.m5_common_source import NativeM5InformationPolicy

VERSION='released_measurements_llm_at_voi_stop.v1'

def released_measurement_view(payload,binding):
    """Label only exactly released indexed evidence; preserve all original text."""
    p=deepcopy(payload);obs=p['observation'];rows=[]
    for index,text in enumerate(p.get('evidence',[])):
        for r in binding['rows']:
            if r['kind']=='Test':
                released=(text==r['target']+': '+r['evidence'] and text in obs.get('objective_findings',[]))
            else:
                released=(text==r['evidence'] and any(h.get('question')==r['target'] and h.get('answer')==r['evidence']
                    and h.get('answer_status')!='unknown' for h in obs.get('history',[])))
            if not released:continue
            if obs['elapsed_minutes']<r['available_minute']:raise ValueError('MEASUREMENT_NOT_YET_AVAILABLE')
            value=json.loads(r['evidence'])
            if value.get('units')!=r['unit']:raise ValueError('RELEASED_MEASUREMENT_UNIT_MISMATCH')
            rows.append({'evidence_index':index,'kind':r['kind'],'description':r['description'],
                'code':r['code'],'value':value['value'],'unit':value['units']})
    p['released_measurements']=rows;p['observation_display_contract']=VERSION
    p['instruction']=p.get('instruction','')+(' released_measurements names the already released evidence at its unchanged index; '
        'use description, value and unit together. It is not a diagnosis or clinical support label. '
        'Select a common catalog diagnosis using current evidence only; normal or nonspecific measurements '
        'do not establish a chronic diagnosis. Retain unresolved when candidates cannot be distinguished.')
    return p

class StopThenLLMInformationPolicy(NativeM5InformationPolicy):
    """New hybrid terminal contract; never pretend M5 supports a clinical label.

    HospitalLLMPolicy already validates the current LLM row before propose(),
    and validates our returned row again. All information actions and STOP come
    unchanged from the existing M5 planner. No extra LLM call or repair.
    """
    def __init__(self,model,binding,selector,terminal_candidate):
        super().__init__(model,binding,selector);self.terminal_candidate=terminal_candidate

    def propose(self,runtime,observed,**kwargs):
        proposal=super().propose(runtime,observed,**kwargs)
        if proposal['action']!='diagnose':return proposal
        trace=runtime.voi_state['history'][-1];record=self.terminal_candidate()
        if not record or record['observation_fingerprint']!=digest(observed):
            raise ValueError('CURRENT_LLM_TERMINAL_SNAPSHOT_REQUIRED')
        candidate=record['decision'];trace['terminal_contract']=VERSION
        trace['m5_clinical_support']=trace.get('clinical_support',False)
        if candidate.get('action')!='diagnose':
            trace['terminal_origin']='M5_UNRESOLVED_LLM_NOT_TERMINAL';return proposal
        trace['terminal_origin']='CURRENT_LLM_DIAGNOSE_PENDING_CORE_REVALIDATION'
        trace['llm_terminal_certainty']=candidate.get('certainty')
        return deepcopy(candidate)
