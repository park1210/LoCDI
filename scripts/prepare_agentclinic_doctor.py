"""C3-12 preparation: DRY_RUN by default, optional offline evidence, NEVER live."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.agentclinic_doctor_client import (FakeDoctorClient, OfflineJsonTransport,
    build_request, PROMPT_VERSION, PROMPT_SHA256, CATALOG_POLICY)
from app.locdi.agentclinic_doctor_runtime import run_doctor_scenario
from app.locdi.agentclinic_doctor_exporter import export_doctor_episode
from app.locdi.agentclinic_live_preflight import SOURCE, proposal, preflight
from app.locdi.agentclinic_scenario_adapter import load_scenarios, adapt_test_record, action_catalog
from app.locdi.agentclinic_turn_runtime import initial_observation, TurnOptions
from app.locdi.agentclinic_benchmark_scoring import evaluate_test_only, load_private_keys
from app.locdi.episode_exporter import _write
from app.locdi.dataset_freeze import sha
from app.locdi.llm_setup import digest


class ScriptedJson(OfflineJsonTransport):
    """Only predetermined nonclinical actions; receives no scoring key."""
    def __init__(self, actions):
        self.actions, self.seen = deepcopy(actions), []

    def chat_json(self, system, user, model=None, *, schema=None):
        self.seen.append({'system':system, 'payload':json.loads(user), 'schema':schema})
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return json.dumps(action) if isinstance(action, dict) else action


def source_plan(scenario):
    c = action_catalog(scenario)
    plan = [{'action':'ASK','target':'history'},
            {'action':'EXAMINE','target':c['EXAMINE'][0]['target']}]
    if c['TEST']:
        plan.append({'action':'TEST','target':c['TEST'][0]['target']})
    plan.append({'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_FINAL_NOT_A_CLINICAL_JUDGMENT'})
    return plan


def fixture_record():
    return {'OSCE_Examination':{'Objective_for_Doctor':'TEST_ONLY_OBJECTIVE',
        'Patient_Actor':{'Demographics':'TEST_ONLY_ACTOR','History':'TEST_ONLY_HISTORY',
                         'Symptoms':{'Primary_Symptom':'TEST_ONLY_SYMPTOM'},
                         'Past_Medical_History':['TEST_ONLY_PAST'],
                         'Social_History':'TEST_ONLY_SOCIAL','Review_of_Systems':'TEST_ONLY_REVIEW'},
        'Physical_Examination_Findings':{'Exam':'TEST_ONLY_EXAM'},
        'Test_Results':{'Test':'TEST_ONLY_RESULT'}, 'Correct_Diagnosis':'TEST_ONLY_PRIVATE_KEY'}}


def pilot_plan():
    definition = proposal()
    result = preflight(definition)
    cfg = definition['legacy_config']
    return {'status':'PROPOSED_NOT_APPROVED', 'purpose':'DEVELOPMENT_EXTERNAL_DOCTOR_PILOT',
        'definition':definition, 'plan_fingerprint':result['plan_fingerprint'],
        'proposed_cases':5, 'approved_case_count':0,
        'selection_rule':'FIRST_FIVE_SOURCE_ORDINALS_FIXED_BEFORE_MODEL_OUTPUT',
        'observed_server_digest':'UNKNOWN_NOT_CHECKED',
        'generation_requests_max':50, 'identity_get_requests_max':50,
        'total_model_server_requests_max':100, 'retry_requests_max':0,'repair_requests_max':0,
        'output_tokens_per_request_max':cfg['max_output_tokens'],
        'total_output_tokens_max':50*cfg['max_output_tokens'],
        'input_budget':{'per_request_utf8_bytes':definition['max_prompt_bytes'],
                        'total_utf8_bytes_max':50*definition['max_prompt_bytes'],
                        'actual_tokenizer_counts':'NOT_CHECKED',
                        'conservative_input_unit_upper_bound':50*definition['max_prompt_bytes'],
                        'conservative_input_plus_output_units_max':50*(definition['max_prompt_bytes']+cfg['max_output_tokens']),
                        'server_num_ctx':'NOT_SET_BY_LEGACY_ADAPTER_REQUIRES_OPERATOR_EVIDENCE'},
        'timeout_seconds_per_http_request':cfg['timeout_seconds'],
        'retry_repair_policy':'NONE; fail case, preserve partial export, stop live batch for review',
        'exports':{'public':'outputs/research_evaluation/c3_13_<NEW_RUN>/','private':'outputs/private/research_evaluation/c3_13_<NEW_RUN>/'},
        'approval_requirements':['PINNED_PRIVATE_EXECUTION_APPROVAL_BOUND_TO_PLAN',
            'PINNED_EXPOSURE_AND_SERVER_CONTEXT_REVIEW','LIVE_IDENTITY_CHECK_BEFORE_EACH_GENERATION',
            'SEPARATE_SCORING_EXECUTION_AUTHORIZATION'],
        'stop_conditions':['SOURCE_MODEL_PROMPT_BINDING_CHANGED','IDENTITY_FAILED',
            'JSON_ACTION_CONTRACT_ERROR','CLIENT_TIMEOUT_OR_ERROR','GLOBAL_BUDGET_EXHAUSTED','EXPORT_FAILED'],
        'proposed_indicators':['TERMINAL_STATE','REQUESTS','ACTION_COUNTS','RELEASE_BOUNDARY',
            'LEXICAL_EXACT_MATCH_ONLY_AFTER_SEPARATE_SCORING_APPROVAL'],
        'clinical_metrics':'NOT_EVALUATED','research_evaluation_ready':False}


def execute_preparation(public_root, private_root):
    """New immutable evidence only. Actual 107 labels never select Fake actions."""
    public_root, private_root = Path(public_root), Path(private_root)
    bundle = load_scenarios(SOURCE)
    notice = SOURCE.parent/'LICENSE.txt'
    if sha(notice)!='1585a9d786b35544fd18c9632582ff9258691fb7a8b317071c8f7691740cdec4':
        raise ValueError('PINNED_MIT_NOTICE_HASH')
    with (public_root/'MIT_NOTICE.txt').open('x',encoding='utf8') as f:
        f.write(notice.read_text(encoding='utf8'))
    _write(preflight(proposal()),'reports/live_preflight.json',public_root,'reports')
    plan = pilot_plan()
    _write(plan,'reports/c3_13_pilot_proposal.json',public_root,'reports')
    _write({'status':'PROPOSED_NOT_APPROVED','selection_rule':plan['selection_rule'],
            'proposed_scenario_refs':[s.scenario_id for s in bundle.scenarios[:5]],
            'plan_fingerprint':plan['plan_fingerprint'],'model_results_seen_before_selection':False},
           'reports/pilot_selection.json',private_root,'reports')
    _write({'source_sha256':sha(SOURCE),'source_revision':bundle.scenarios[0].source['revision'],
        'evaluation_group':bundle.scenarios[0].source['evaluation_group'],
        'development_exposed_ordinals':list(range(1,108)),
        'independent_test_split':'NOT_ESTABLISHED','model_pretraining_contamination':'UNKNOWN',
        'overlap_with_existing_paper_data':'NOT_REVIEWED',
        'prompt_version':PROMPT_VERSION,'prompt_sha256':PROMPT_SHA256,
        'prior_prompt_history':'C3-11 source-lookup Fake; C3-12 first versioned Doctor system prompt',
        'catalog_policy':CATALOG_POLICY,'catalog_policy_scope':'ALL_SOURCE_TARGET_LABELS_AND_EXACT_IDS; NO_VALUES',
        'catalog_indirect_hint_risk':'SOURCE_BRANCH_LABELS_ARE_CASE_SPECIFIC; NOT_BLINDED',
        'model_prompt_optimization_from_correctness':'NOT_PERFORMED',
        'cross_case_memory':False,'clinical_time_and_resources':'NOT_SUPPORTED',
        'live_execution_approved':False,'clinical_reference_approved':False},
        'reports/exposure_manifest.json',public_root,'reports')
    episodes, rows, boundary = [], [], []
    for ordinal,s in enumerate(bundle.scenarios,1):
        tr=ScriptedJson(source_plan(s)); doctor=FakeDoctorClient(tr)
        episode=run_doctor_scenario(s,doctor)
        out=export_doctor_episode(episode,public_root=public_root,private_root=private_root,name='Scenario%03d'%ordinal)
        episodes.append(episode)
        # Recheck accepted source values reach the NEXT request only.
        for index,seen in enumerate(tr.seen):
            expected=[t['observation'] for t in episode['trace'][:index] if t['observation'] is not None]
            if seen['payload']['observations']!=expected:
                raise ValueError('SOURCE_TO_NEXT_REQUEST_LINK')
        initial=initial_observation(s);initial['remaining_requests']=20
        system,payload,_=build_request(initial)
        # The answer is consulted only by an offline leakage assertion, never by Fake.
        from scripts.audit_agentclinic_dataset import lexical_hit
        key=bundle.scoring_keys[ordinal-1]
        if lexical_hit(key.correct_diagnosis,system+json.dumps(payload,ensure_ascii=False)):
            raise ValueError('INITIAL_PRIVATE_LABEL_EXPOSURE')
        boundary.append({'ordinal':ordinal,'initial_private_label_absent':True,
                         'source_to_next_request_linked':True,'prompt_payload_fingerprints':
                         [digest(x['payload']) for x in tr.seen]})
        rows.append({'ordinal':ordinal,'scenario_ref':s.scenario_id,**out['result'],
                     'doctor_request_attempts':out['doctor_request_attempts']})
    # Read the pre-existing private scoring store only AFTER all doctor executions.
    key_path=ROOT/'outputs/private/research_evaluation/c3_11_v1/reports/scoring_keys.json'
    keys=load_private_keys(key_path,sha(key_path))
    if keys!=bundle.scoring_keys:
        raise ValueError('PRIVATE_KEY_SOURCE_LINEAGE')
    real_scores=evaluate_test_only(bundle.scenarios,episodes,keys)
    _write(real_scores,'metrics/benchmark_scoring_status.json',public_root,'metrics')
    _write({'rows':boundary,'all_107_initial_prompts_checked':True,
            'actual_wire_payloads':'NOT_CAPTURED_NO_LIVE_REQUESTS'},'reports/prompt_boundary.json',public_root,'reports')
    _write({'rows':rows,'status_counts':dict(Counter(r['status'] for r in rows)),
            'doctor_request_attempts':sum(r['doctor_request_attempts'] for r in rows),
            'empty_test_catalog_ordinals':[i for i,s in enumerate(bundle.scenarios,1) if not s.test_targets],
            'fake_policy':'ASK_HISTORY_EXAM_FIRST_TEST_FIRST_IF_AVAILABLE_FIXED_NONCLINICAL_FINAL',
            'clinical_metrics':'NOT_EVALUATED','real_generation_calls':0,'grading_calls':0},
           'metrics/offline_source_execution.json',public_root,'metrics')
    pairs=[adapt_test_record(fixture_record(),i) for i in range(1,7)]
    fixture_episodes=[]
    plans=[([{'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_PRIVATE_KEY'}],TurnOptions()),
           ([{'action':'DIAGNOSE','predicted_diagnosis':'TEST_ONLY_OTHER'}],TurnOptions()),
           ([{'action':'DIAGNOSE','predicted_diagnosis':''}],TurnOptions()),
           ([{'action':'ASK','target':'history'}],TurnOptions(1,1)),
           ([TimeoutError('TEST_ONLY_PRIVATE_FAILURE')],TurnOptions())]
    for i,(actions,options) in enumerate(plans):
        episode=run_doctor_scenario(pairs[i][0],FakeDoctorClient(ScriptedJson(actions)),options=options)
        export_doctor_episode(episode,public_root=public_root,private_root=private_root,name='TEST_ONLY_Scoring%02d'%(i+1))
        fixture_episodes.append(episode)
    scores=evaluate_test_only([s for s,k in pairs],fixture_episodes,[k for s,k in pairs])
    _write(scores,'metrics/TEST_ONLY_lexical_scoring.json',public_root,'metrics')
    _write({'schema_version':'agentclinic.private_fixture_keys.v1','TEST_ONLY':True,
            'keys':[{'scenario_ref':k.scenario_id,'correct_diagnosis':k.correct_diagnosis,
                     'record_fingerprint':k.source_record_fingerprint} for s,k in pairs]},
           'reports/TEST_ONLY_fixture_keys.json',private_root,'reports')
    return {'source_scenarios':len(rows),'source_status_counts':dict(Counter(r['status'] for r in rows)),
            'fake_doctor_requests':sum(r['doctor_request_attempts'] for r in rows),
            'fixture_registered':6,'fixture_executed':5,'fixture_state_counts':scores['state_counts'],
            'benchmark_scoring':real_scores['status'],'real_generation_calls':0,'grading_calls':0,
            'live_execution_ready':False,'research_evaluation_ready':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true',help='Fake Client only; no live execution option')
    parser.add_argument('--namespace',help='NEW immutable namespace, required with --offline')
    args=parser.parse_args()
    def guard(event,values):
        if event in {'socket.connect','socket.connect_ex','socket.bind'}:
            raise RuntimeError('C312_NO_NETWORK')
    sys.addaudithook(guard)
    if not args.offline:
        if args.namespace:parser.error('--namespace requires --offline')
        print(json.dumps(preflight(proposal()),ensure_ascii=True));return
    if not args.namespace or not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.namespace):
        parser.error('NEW_NAMESPACE_REQUIRED')
    pub=ROOT/'outputs/research_evaluation'/args.namespace
    priv=ROOT/'outputs/private/research_evaluation'/args.namespace
    if pub.exists() or priv.exists():parser.error('NAMESPACE_ALREADY_EXISTS')
    pub.mkdir(parents=True,exist_ok=False);priv.mkdir(parents=True,exist_ok=False)
    print(json.dumps(execute_preparation(pub,priv)))


if __name__=='__main__':
    main()
