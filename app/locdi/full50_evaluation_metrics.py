"""Pure aggregation of validated same-Run development rows; never a scorer.

Source Exact values must come from the unchanged private scorer. Cause tags
describe stored facts/candidates and never establish clinical causality.
"""
from collections import Counter

CONDITIONS=('B0','B1','B2','B3','B4')


def aggregate(rows,ordinals,*,plan_fingerprint):
    ids=set(ordinals)
    if len(ids)!=len(ordinals) or not ids:raise ValueError('UNIQUE_REGISTERED_PATIENTS_REQUIRED')
    lookup={}
    for r in rows:
        k=r['ordinal'],r['condition']
        if k in lookup or k[0] not in ids or k[1] not in CONDITIONS:raise ValueError('DUPLICATE_OR_UNREGISTERED_EPISODE')
        if r['plan_fingerprint']!=plan_fingerprint:raise ValueError('CROSS_RUN_MIXING_FORBIDDEN')
        for field in ('source_exact','source_department'):
            if r.get(field) is not None and type(r[field]) is not bool:raise ValueError('BOOLEAN_OR_NULL_SCORE_REQUIRED')
        lookup[k]=r
    output={};pairs=[]
    for condition in CONDITIONS:
        rr=[lookup.get((i,condition),{'ordinal':i,'status':'NOT_ATTEMPTED'}) for i in ordinals]
        complete=[r for r in rr if r['status']=='COMPLETED'];correct=sum(r.get('source_exact') is True for r in complete)
        actions=Counter();roles=Counter();usage=Counter();selected=abstained=route_calls=0;tags=Counter()
        for r in rr:
            actions.update(r.get('action_counts',{}));roles.update(r.get('role_counts',{}));usage.update(r.get('usage',{}))
            route_calls+=r.get('routing_calls',0);selected+=r.get('routing_selected',0);abstained+=r.get('routing_abstained',0)
            causes=[]
            if r.get('chief_complaint_missing'):causes.append('MISSING_CHIEF_COMPLAINT_FACT')
            if r.get('current_evidence_count')==0:causes.append('NO_CURRENT_EVIDENCE_FACT')
            if r.get('memory_eligible_count',0)>0 and r.get('memory_selected_count')==0:causes.append('RETRIEVAL_FAILURE_CANDIDATE_NOT_RELEVANCE_JUDGMENT')
            if r.get('specialty_called'):causes.append('SPECIALTY_EXECUTION_OBSERVED_NOT_CAUSE_PROVEN')
            if r.get('terminal_origin')=='M5_UNRESOLVED_LLM_NOT_TERMINAL':causes.append('STOP_NONTERMINAL_CONTRACT_FACT')
            if r['status']=='COMPLETED' and r.get('source_exact') is False:causes.append('SOURCE_DIAGNOSIS_DISAGREEMENT_FACT')
            tags.update(causes)
        output[condition]={'registered':len(ids),'states':dict(Counter(r['status'] for r in rr)),
            'Source_Exact':correct,'Source_Exact_registered_rate':correct/len(ids),
            'Source_department':sum(r.get('source_department') is True for r in complete),
            'supported':sum(r.get('certainty')=='supported' for r in complete),
            'unresolved':sum(r.get('certainty')=='unresolved' for r in complete),
            'algorithmic_Source_Exact':sum(r.get('algorithmic_source_exact') is True for r in complete),
            'routing_calls':route_calls,'routing_selected':selected,'routing_abstained':abstained,
            'routing_coverage':selected/route_calls if route_calls else None,
            'role_counts':dict(roles),'action_counts':dict(actions),'usage':dict(usage),
            'generation':sum(r.get('generation',0) for r in rr),'wall_seconds':sum(r.get('wall_seconds',0) for r in rr),
            'received_tests':sum(r.get('received_test_count',0) for r in rr),
            'final_provided_test_results':sum(r.get('final_provided_test_count',0) for r in rr),
            'final_cited_test_results':sum(r.get('final_cited_test_count',0) for r in rr),
            'Planner':sum(r.get('planner_calls',0) for r in rr),'STOP':sum(r.get('stop_count',0) for r in rr),
            'terminal_extra_requests':sum(r.get('terminal_extra_requests',0) for r in rr),
            'pre_STOP_model_actions':dict(Counter(r['pre_stop_model_action'] for r in rr if r.get('pre_stop_model_action'))),
            'memory_cutoff_violations':sum(r.get('memory_cutoff_violations',0) for r in rr),
            'nonexclusive_observation_tags':dict(tags)}
    for i in ordinals:
        for left,right in (('B2','B3'),('B3','B4')):
            a=lookup.get((i,left),{});b=lookup.get((i,right),{});x=a.get('source_exact');y=b.get('source_exact')
            if a.get('status')!='COMPLETED' or b.get('status')!='COMPLETED':label='INCOMPLETE_PAIR'
            elif x is True:label='CORRECT_RETAINED' if y is True else 'CORRECT_TO_INCORRECT' if y is False else 'CORRECT_TO_UNRESOLVED_OR_UNSCORED'
            elif y is True:label='INCORRECT_OR_UNSCORED_TO_CORRECT'
            else:label='NO_REGISTERED_CORRECT_RESULT_BOTH'
            pairs.append({'ordinal':i,'comparison':left+'_'+right,'transition':label})
    return {'conditions':output,'pairs':pairs,'clinical_accuracy':'NOT_EVALUATED',
            'independent_clinical_review':False,'source_label_proxy':True,'causality':'NOT_ESTABLISHED'}


def aggregate_by_exposure(rows,ordinals,old_development_ordinals,*,plan_fingerprint):
    """Keep the full denominator and show the existing10/other40 separately."""
    old=set(old_development_ordinals);result={}
    for label,selected in (('all_registered',list(ordinals)),
                           ('old_development10',[i for i in ordinals if i in old]),
                           ('other40_also_development_exposed',[i for i in ordinals if i not in old])):
        result[label]=(aggregate([r for r in rows if r['ordinal'] in selected],selected,plan_fingerprint=plan_fingerprint)
                       if selected else {'status':'NOT_APPLICABLE_EMPTY_STRATUM'})
    return result
