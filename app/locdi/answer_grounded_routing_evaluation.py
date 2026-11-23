"""Answer-only metrics/TEST_ONLY collection. No Core fitting or clinical gate calls."""
from collections import Counter
from copy import deepcopy
import json
from app.locdi.answer_grounded_routing import STATES,model_input,answer_schema,score
from app.locdi.llm_setup import digest
from app.core.routing_utility import jaccard_similarity,observed_tokens


def rate(n,d):return {'numerator':n,'denominator':d,'value':n/d if d else None,'status':'DEFINED' if d else 'NOT_APPLICABLE'}


def metrics(rows):
    if any(r['state'] not in STATES for r in rows):raise ValueError('UNKNOWN_OUTCOME_STATE')
    counts=Counter(r['state'] for r in rows);n=len(rows)
    supported=counts['CORRECT_ANSWER']+counts['INCORRECT_ANSWER']
    return {'registered':n,'counts':{s:counts[s] for s in STATES},
        'answer_coverage':rate(supported,n),'selective_answer_accuracy':rate(counts['CORRECT_ANSWER'],supported),
        'correct_supported_answer_rate':rate(counts['CORRECT_ANSWER'],n),
        'unresolved_rate':rate(counts['UNRESOLVED'],n),
        'failure_rate':rate(counts['INVALID_RESPONSE']+counts['GENERATION_FAILED'],n),
        'not_attempted_rate':rate(counts['NOT_ATTEMPTED'],n)}


def evaluate(questions,bindings,rows,*,selected_agents=None):
    qids=[q['question_id'] for q in questions];aids=[a['agent_id'] for a in bindings]
    if len(set(qids))!=len(qids) or len(set(aids))!=len(aids):raise ValueError('DUPLICATE_REGISTERED_ID')
    expected={(q,a) for q in qids for a in aids};index={}
    for r in rows:
        k=(r['question_id'],r['agent_id'])
        if k not in expected or k in index:raise ValueError('DUPLICATE_OR_UNREGISTERED_OUTCOME')
        if r['state'] not in STATES:raise ValueError('OUTCOME_STATE')
        should={'CORRECT_ANSWER':True,'INCORRECT_ANSWER':False}.get(r['state'])
        if r.get('answer_correct') is not should:raise ValueError('OUTCOME_CORRECTNESS_MEANING')
        index[k]=r
    # Missing rows are explicitly retained, never a fabricated complete group.
    allrows=[index.get((q,a),{'question_id':q,'agent_id':a,'state':'NOT_ATTEMPTED','answer_correct':None}) for q in qids for a in aids]
    index={(r['question_id'],r['agent_id']):r for r in allrows}
    groups=[];complete=[]
    for qid in qids:
        rs=[index[qid,a] for a in aids]
        collected=all(r['state'] not in ('NOT_ATTEMPTED','GENERATION_FAILED','INVALID_RESPONSE') for r in rs)
        scorable=all(r['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER') for r in rs)
        signatures=[r.get('answer_signature') for r in rs]
        comparable=collected and all(s is not None for s in signatures)
        g={'question_id':qid,'complete_collection':collected,'complete_scorable':scorable,
            'oracle_success':any(r['state']=='CORRECT_ANSWER' for r in rs) if collected else None,
            'all_agents_same_answer':len(set(signatures))==1 if comparable else None}
        groups.append(g)
        if collected:complete.append(g)
    chosen=[]
    if selected_agents is not None:
        if set(selected_agents)!=set(qids):raise ValueError('PAIRED_SELECTION_SCOPE')
        for q,a in selected_agents.items():
            if a not in aids:raise ValueError('SELECTED_AGENT_NOT_CANDIDATE')
            chosen.append(index[q,a])
    comparable=[g for g in groups if g['all_agents_same_answer'] is not None]
    return {'all_agents':metrics(allrows),'per_agent':{a:metrics([r for r in allrows if r['agent_id']==a]) for a in aids},
        'groups':groups,'complete_collection_groups':sum(g['complete_collection'] for g in groups),
        'complete_scorable_groups':sum(g['complete_scorable'] for g in groups),
        'evaluation_only_oracle':{'scope':'Posthoc complete observed groups only; not Router performance',
            'success_upper_bound':rate(sum(g['oracle_success'] for g in complete),len(complete))},
        'agent_disagreement_rate':rate(sum(not g['all_agents_same_answer'] for g in comparable),len(comparable)),
        'selected_agent_metrics':metrics(chosen) if selected_agents is not None else None,
        'clinical_routing_metrics':'NOT_EVALUATED','utility_fit_ready':False}


def collect_test_only(plan,questions,refs,bindings,client_factory,*,execute=False,test_only=False,max_requests=25):
    """Injected Fake only, fail closed before constructing any transport by default.

    Deliberately cannot authorize a live model via a JSON flag or caller bool.
    A live runner/approval implementation is a separate subsequent task.
    """
    if not execute:return {'mode':'DRY_RUN','generation_attempts':0,'client_constructed':False}
    if test_only is not True:raise ValueError('LIVE_EXECUTION_DISABLED_C4_24')
    if not 0<=max_requests<=25:raise ValueError('GLOBAL_BUDGET_25')
    if plan['fingerprint']!=digest({k:v for k,v in plan.items() if k!='fingerprint'}):raise ValueError('PLAN_FINGERPRINT')
    if digest(list(zip(questions,refs)))!=plan['question_reference_fingerprint'] or digest(bindings)!=plan['bindings_fingerprint']:
        raise ValueError('PLAN_INPUT_BINDING')
    client=client_factory()
    if getattr(client,'TEST_ONLY',False) is not True:raise ValueError('TEST_ONLY_CLIENT_REQUIRED')
    # No production transport class is accepted; explicit injected fixture function only.
    if type(client).__module__.startswith(('app.','openai','httpx','requests')):raise ValueError('REAL_CLIENT_FORBIDDEN')
    qi={q['question_id']:q for q in questions};ri={r['question_id']:r for r in refs};ai={a['agent_id']:a for a in bindings}
    outputs=[];checkpoints=[];attempts=0;stopped=False
    for t in plan['tasks']:
        q=qi[t['question_id']];ref=ri[t['question_id']];agent=ai[t['agent_id']]
        if stopped or attempts>=min(max_requests,plan['max_generations']):
            result=score(q,ref,attempted=False)
        else:
            public=model_input(q)
            if digest(public)!=t['input_fingerprint'] or digest(answer_schema(q))!=t['schema_fingerprint']:
                raise ValueError('TASK_INPUT_OR_SCHEMA_CHANGED')
            attempts+=1
            checkpoints.append({'stage':'BEFORE','request_ordinal':attempts,'input_fingerprint':digest(public),
                'prompt_fingerprint':agent['benchmark_prompt_fingerprint'],'question_id':q['question_id'],'agent_id':agent['agent_id']})
            try:
                raw=client.generate(agent['system_prompt'],deepcopy(public),answer_schema(q))
                result=score(q,ref,raw)
                if result['state'] in ('CORRECT_ANSWER','INCORRECT_ANSWER'):
                    parsed=json.loads(raw) if isinstance(raw,str) else raw
                    from app.locdi.agentclinic_benchmark_scoring import normalize
                    result['answer_signature']=digest(parsed['answer_id'] if ref['answer_type']=='option_id' else normalize(parsed['diagnosis']))
                elif result['state']=='UNRESOLVED':result['answer_signature']=digest('UNRESOLVED')
            except Exception as exc:
                result=score(q,ref,error_type=type(exc).__name__)
            checkpoints.append({'stage':'AFTER','request_ordinal':attempts,'state':result['state']})
            if result['state'] in ('INVALID_RESPONSE','GENERATION_FAILED'):stopped=True
        outputs.append({**result,'agent_id':agent['agent_id']})
    return {'mode':'TEST_ONLY','generation_attempts':attempts,'actual_llm_generations':0,
        'rows':outputs,'checkpoints':checkpoints,'evaluation':evaluate(questions,bindings,outputs),
        'first_failure_stopped':stopped,'clinical_metrics':'NOT_EVALUATED'}


def support_audit(questions,*,min_samples=2):
    """Potential shared-public-text similarity only; no labels or Utility fitting."""
    rows=[]
    for q in questions:
        public=model_input(q)
        tokens=observed_tokens({'chief_complaint':json.dumps(public,sort_keys=True,ensure_ascii=False)})
        rows.append((q,tokens))
    return {'scope':'Potential leave-one-question-out text support, not training samples or calibrated utility',
        'min_samples':min_samples,'rows':[{'question_id':q['question_id'],
            'other_questions_with_positive_jaccard':sum(jaccard_similarity(ts,other)>0 for oq,other in rows if oq['question_id']!=q['question_id'])}
            for q,ts in rows],'fit_calls':0,'clinical_gate_ready':False}
