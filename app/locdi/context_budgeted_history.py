"""Opt-in stable complete-record history adapter; existing retrieval unchanged."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from unittest.mock import patch
from app.core.longitudinal_memory import PatientMemoryStore,timestamp,detect_conflicts
from app.locdi.memory_subset_wire import allowed_codes
from app.locdi.llm_setup import digest

VERSION='B1_CONTEXT_BUDGETED_HISTORY.v1'
MAX_JOINT_BYTES=4096
FUTURE_RESERVE_BYTES=2048
SCHEMA_REPLICATION_RESERVE=6
WIRE_FRAMING_BYTES=512


def source_records(records):
    return [{k:deepcopy(v) for k,v in r.items() if k!='retrieval_score'} for r in records]


def projected(context,selected,joint_budget):
    out=deepcopy({k:v for k,v in context.items() if k!='records'})
    out['records']=deepcopy(selected);out['conflicts']=detect_conflicts(selected)
    out.update(selected_count=len(selected),budget_excluded_count=len(context['records'])-len(selected),
        selection_method=VERSION,max_chars=None,
        records_json_chars=len(json.dumps(selected,ensure_ascii=False)))
    out['context_budgeted_history']={'policy':VERSION,'joint_memory_schema_bytes':joint_budget,
        'stable_within_visit':True,'complete_records_only':True,'candidate_count':len(context['records']),
        'selected_count':len(selected),'excluded_count':len(context['records'])-len(selected)}
    return out


def joint_cost(context,records):
    codes=allowed_codes(records)
    prop={'type':'string','enum':codes}
    return (len(json.dumps(context,ensure_ascii=False,sort_keys=True).encode('utf-8'))+
        SCHEMA_REPLICATION_RESERVE*len(json.dumps(prop,separators=(',',':')).encode('utf-8'))+
        WIRE_FRAMING_BYTES+32*len(records))


def build_plan(identity,context,budget,*,atomic_groups=None):
    """Newest available complete units; skip oversize, then inspect smaller ones."""
    if type(budget) is not int or not 0<=budget<=MAX_JOINT_BYTES:raise ValueError('EXPLICIT_JOINT_BYTE_BUDGET_REQUIRED')
    records=context['records'];atomic_groups=atomic_groups or {};groups={}
    if len({r['memory_id'] for r in records})!=len(records):raise ValueError('DUPLICATE_SOURCE_MEMORY')
    for r in records:
        if (r['patient_id']!=identity['patient_id'] or r['visit_id']==identity['visit_id'] or
            timestamp(r['visit_started_at'])>=timestamp(identity['started_at']) or
            timestamp(r['available_at'])>timestamp(identity['started_at']) or
            timestamp(r['observed_at'])>timestamp(r['available_at'])):raise ValueError('HISTORY_SOURCE_BOUNDARY')
        groups.setdefault(atomic_groups.get(r['memory_id'],'record:'+r['memory_id']),[]).append(r)
    units=sorted(groups.values(),key=lambda rs:(-max(timestamp(r['available_at']).timestamp() for r in rs),min(r['memory_id'] for r in rs)))
    chosen=[];excluded=[]
    for unit in units:
        candidate=sorted(chosen+unit,key=lambda r:(timestamp(r['available_at']),r['memory_id']))
        try:cost=joint_cost(projected(context,candidate,budget),candidate)
        except ValueError:cost=budget+1
        if cost<=budget:chosen=candidate
        else:excluded.extend({'memory_id':r['memory_id'],'reason':'COMPLETE_UNIT_JOINT_CONTEXT_BUDGET'} for r in unit)
    out=projected(context,chosen,budget)
    # Empty history still has a fixed runtime frame. If that alone cannot fit,
    # fail closed instead of labeling an unbudgeted representation as ready.
    if joint_cost(out,chosen)>budget:raise ValueError('NO_SPACE_FOR_EMPTY_HISTORY_FRAME')
    plan={'policy':VERSION,'identity':deepcopy(identity),'joint_budget_bytes':budget,
        'candidate_count':len(records),'candidate_source_fingerprint':digest(source_records(records)),
        'selected_ids':[r['memory_id'] for r in chosen],
        'selected_source_fingerprint':digest(source_records(chosen)),'selected_count':len(chosen),
        'excluded_count':len(excluded),'excluded':excluded,'atomic_groups':deepcopy(atomic_groups),
        'joint_cost_bytes':joint_cost(out,chosen),'selected_json_bytes':len(json.dumps(chosen,ensure_ascii=False).encode('utf-8')),
        'selected_time_range':[min((r['available_at'] for r in chosen),default=None),max((r['available_at'] for r in chosen),default=None)]}
    plan['fingerprint']=digest(plan)
    return plan


class ContextBudgetedHistory:
    def __init__(self,plans):
        self.plans={}
        for p in deepcopy(plans):
            if p.get('policy')!=VERSION or p.get('fingerprint')!=digest({k:v for k,v in p.items() if k!='fingerprint'}):
                raise ValueError('HISTORY_PLAN_FINGERPRINT')
            key=(p['identity']['patient_id'],p['identity']['visit_id'])
            if key in self.plans:raise ValueError('DUPLICATE_HISTORY_VISIT')
            self.plans[key]=p
        self.fingerprint=digest(plans)
    def metadata(self):
        return {'policy':VERSION,'selection_fingerprint':self.fingerprint,'context':32768,
            'future_reserve_bytes':FUTURE_RESERVE_BYTES,'max_joint_bytes':MAX_JOINT_BYTES,
            'selection_unit':'complete_record_or_declared_visit_block','stable_within_visit':True}
    def project(self,identity,context):
        plan=self.plans.get((identity['patient_id'],identity['visit_id']))
        if plan is None or plan['identity']!=identity:raise ValueError('UNREGISTERED_HISTORY_IDENTITY')
        records=context['records']
        if digest(source_records(records))!=plan['candidate_source_fingerprint']:raise ValueError('HISTORY_SOURCE_CHANGED')
        lookup={r['memory_id']:r for r in records};selected=[lookup[x] for x in plan['selected_ids']]
        if digest(source_records(selected))!=plan['selected_source_fingerprint']:raise ValueError('HISTORY_SELECTION_CHANGED')
        out=projected(context,selected,plan['joint_budget_bytes'])
        # Ignore changing lexical scores for budgeting (never used to select
        # B1); the per-record allowance covers their bounded serialization.
        if joint_cost(out,selected)>plan['joint_budget_bytes']+32*len(selected):raise ValueError('HISTORY_RUNTIME_BYTE_BUDGET')
        return out
    @contextmanager
    def installed(self):
        original=PatientMemoryStore.retrieve;policy=self
        def retrieve(store,identity,query,mode='selective',top_k=5,**kwargs):
            result=original(store,identity,query,mode,top_k,**kwargs)
            # Core's explicit unused full_history transport top_k is1. Its
            # separate eligibility lookup uses the default5 and stays full.
            if mode=='full_history' and top_k==1:
                if kwargs.get('max_chars') is not None:raise ValueError('NO_SECOND_HISTORY_TRUNCATION')
                return policy.project(identity,result)
            return result
        with patch.object(PatientMemoryStore,'retrieve',new=retrieve):yield
