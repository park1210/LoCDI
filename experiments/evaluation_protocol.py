"""Draft metric semantics and engineering-condition inventory, NOT an evaluator.

No clinical values are calculated here. Original B0-B4 switches remain user
confirmed; paper correspondence and the research evaluation policy are unapproved.
"""
from experiments.run_locdi_ablation import CONDITIONS,FEATURE_KEYS

METRICS=(
 ('diagnosis_correctness','Diagnosis','sum(correct)/N independently reviewed supported evaluable responses',
  'Independent current-decision diagnosis reference + human response judgment; unresolved is excluded, not false',
  'Private response/reference/review; public result alone has no diagnosis text','experiments.locdi_metrics.evaluate_episode (NOT_IMPLEMENTED)',
  'Legacy paper_evaluation/hospital_evaluation/voi_evaluation can score unresolved as 0; DO_NOT_REUSE_AS_THIS_METRIC'),
 ('unresolved_rate','Diagnosis','N finalized unresolved / N finalized episodes with explicit unresolved status',
  'No clinical gold for descriptive rate; full terminal coverage and approved denominator policy required',
  'result.finalized, result.diagnosis_unresolved, result.status','Episode fields; official translator NOT_IMPLEMENTED',
  'Failed unfinalized episode with false unresolved flag is not resolved; report failures separately'),
 ('finalization_rate','Diagnosis','N finalized / N started episodes with known terminal state',
  'Complete execution ledger, no diagnosis correctness inference','result.finalized, result.status','Episode fields; official translator NOT_IMPLEMENTED','Missing execution is NOT_EXECUTED, not finalized=false'),
 ('completion_rate','Diagnosis','N completed / N started episodes with known status',
  'Complete execution ledger only','result.status','Episode fields; official translator NOT_IMPLEMENTED','Completed is not clinical correctness'),
 ('memory_precision_at_k','Memory','|ranked_retrieval[1:K] intersect eligible_relevant| / K',
  'Independent relevance at exact decision/cutoff; ordered retrieval and K',
  'memory.retrievals[].memory_refs + private reference map','app.core.memory_evaluation.evaluate_memory (ADAPTATION_REQUIRED)',
  'DRAFT fixed K denominator; legacy precision uses returned set size and final snapshot; not identical when underfilled'),
 ('memory_recall_at_k','Memory','|ranked_retrieval[1:K] intersect eligible_relevant| / |eligible_relevant|',
  'Independent complete eligible relevance labels at decision cutoff','memory.retrievals + evaluator-private labels',
  'app.core.memory_evaluation.evaluate_memory (snapshot adaptation required)','No relevant eligible records => NOT_APPLICABLE; missing annotation => BLOCKED'),
 ('memory_mrr','Memory','mean(1/first relevant rank; 0 if none retrieved) over queries with nonempty reviewed eligible relevance',
  'Ranked retrieval, independent relevance, stable private ID linkage','memory.retrievals[].memory_refs ordered',
  'NOT_IMPLEMENTED','Empty relevant set is NA; missing reference is not a zero rank'),
 ('strict_routing_accuracy','Routing','N selected specialty equal unique reviewed target / N selected evaluable routing decisions',
  'Independent unique target department; explicit routing status','handoff audit.routing, private review',
  'NOT_IMPLEMENTED (legacy final_department_accuracy is not this decision metric)',
  'DRAFT conditional accuracy; abstention coverage reported separately, missing Utility is not abstention'),
 ('acceptable_specialty_accuracy','Routing','N selected specialty in reviewed acceptable set / N selected evaluable routing decisions',
  'Independent acceptable specialty set for the exact decision','handoff audit.routing, private review',
  'NOT_IMPLEMENTED (legacy consult precision measures another event)',
  'Logical Handoff selection and physical Consult must be separated'),
 ('routing_abstention_rate','Routing','N abstained / N available autonomous routing decisions (selected+abstained)',
  'Explicit statuses, Artifact availability and candidate scope; no clinical label for descriptive rate',
  'handoff audit.routing[].status','Audit fields; official translator NOT_IMPLEMENTED',
  'Disabled/unavailable/missing Artifact are separate categories, not abstentions'),
 ('routing_success','Routing','N successful independently reviewed contributions / N contribution-evaluable routing decisions',
  'Separate routing-success policy/reference and independent contribution judgment','Private contribution review + linked routing/action events',
  'NOT_IMPLEMENTED','Diagnosis correctness is not routing success; no automatic label from Agent diagnosis'),
 ('action_quality','VoI','N executed actions in independently reviewed state-specific acceptable set / N reviewed executed actions',
  'Independent acceptable next actions and public-state/cutoff binding','workflow.decisions/event links + private review',
  'NOT_IMPLEMENTED','LLM proposed action is not necessarily the VoI-selected executed action'),
 ('information_gain','VoI','expected IG=H(b)-sum_o P(o|b,a)H(b_o); realized entropy change is a separate statistic',
  'Validated prior/likelihood/outcome model; independently reviewed interpretation',
  'Private voi_state history; public trace contains hashes/counts not all numeric beliefs',
  'app.core.voi_planning.plan_voi; paper_evaluation.episode_rows (private translator required)',
  'Toy/model IG is not empirical clinical benefit; expected IG is not realized IG'),
 ('unnecessary_test_rate','VoI','N ordered tests independently judged unnecessary / N ordered tests evaluable for necessity',
  'Independent necessity labels at order time, not absence from a source list','test_ordered events + private necessity review',
  'NOT_IMPLEMENTED_AS_DEFINED (hospital_evaluation.uti uses different numerator/denominator)',
  'Legacy uti is nonacceptable received tests / required-test count, not this proposed rate'),
 ('action_count','VoI','count of linked actual Ask/Test/Consult/Finalize execution events per episode',
  'Complete action/event mapping, agreed action inclusion policy','result.action_counts; workflow.events',
  'Episode counts; official translator NOT_IMPLEMENTED','Do not count logical Initial/Specialty judgments or planning rows again'),
 ('time_to_final_diagnosis','System','terminal diagnosis event minute - arrival minute, conditional on finalized resolved workflow',
  'Complete clock/event trace and censoring policy; clinical correctness not implied',
  'workflow.events case_started/case_completed + result.finalized/diagnosis_unresolved',
  'app.core.hospital_evaluation.evaluate_case.ttd (LoCDI translation required)',
  'Incomplete/failed/unresolved are censored/NA categories; simulation elapsed includes scheduler endpoint and is not necessarily TTD'),
 ('generation_requests','System','sum of actual adapter generation request counters across all stages',
  'Real adapter counters/checkpoints and complete run ledger','llm_execution.request_count; handoff request journal',
  'Adapter telemetry; official translator NOT_IMPLEMENTED','Identity checks and Core decision count are separate; Offline dispatch is not generation'),
 ('repair_count','System','count repair requests by specialty-semantic versus Core-action kind',
  'Explicit repair flags and repair-of linkage','handoff.requests[].repair_kind, repair_of_request_id',
  'Audit counts; official translator NOT_IMPLEMENTED','Missing legacy repair flag => NOT_RECORDED, not observed zero'),
 ('retry_count','System','count explicitly recorded retry attempts',
  'Retry policy and actual attempt ledger','handoff.requests[].retry; llm_execution',
  'Audit counts; official translator NOT_IMPLEMENTED','Repair is not retry; configuration retries=0 is not raw-wire evidence'),
 ('failure_rate','System','N failed / N started episodes with known status',
  'Complete planned/started/blocked/missing execution ledger','result.status + preflight/started/pilot records',
  'Episode statuses; official translator NOT_IMPLEMENTED','Missing runs are not successes; blocked/unstarted are separate coverage categories'),
 ('budget_exhaustion_rate','System','N explicitly budget-exhausted episodes / N started episodes with complete budget audit',
  'Explicit request-budget blocker and full attempt records','GLOBAL_REQUEST_BUDGET_EXHAUSTED or REQUEST_LIMIT_REACHED',
  'Audit blocker codes; official translator NOT_IMPLEMENTED','Request count equal to cap does not by itself establish exhaustion'),
)


def metric_contracts():
    return [{'metric_id':m,'category':c,'definition':d,'required_reference':r,'observed_fields':o,
        'calculation_function':f,'limitation':l,'policy_status':'DRAFT_NOT_PAPER_APPROVED',
        'missing_rule':'MISSING_REFERENCE_OR_LOG=>BLOCKED_OR_NOT_CHECKED; TRUE_EMPTY_DENOMINATOR=>NOT_APPLICABLE',
        'clinical_paper_use':'BLOCKED_PENDING_POLICY_REFERENCE_AND_TRANSLATOR'} for m,c,d,r,o,f,l in METRICS]


def condition_contracts(configs):
    """Engineering values only. Missing Handoff is NOT silently inferred."""
    rows=[]
    for name in CONDITIONS:
        config=configs.get(name,{})
        rows.append({'condition':name,'features':config.get('features'),
            'sequential_handoff':'NOT_SPECIFIED_BY_EXISTING_OFFICIAL_JSON',
            'switch_definition_source':'USER_CONFIRMED_ENGINEERING_CONDITIONS',
            'paper_chapter_equation':'NOT_CHECKED_NO_CONFIRMED_PAPER',
            'evaluation_protocol_approved':False,'execution_ready':False})
    return rows
