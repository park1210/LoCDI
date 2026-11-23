"""C4-24 freeze/prepare/dry-run; production generation intentionally unavailable."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check,locations
from scripts.evaluate_memory_concepts import offline
from app.locdi.llm_setup import digest,inspect_local_model
from app.locdi.answer_grounded_routing import (VERSION,TARGET,TASK_PROMPT,read_agentclinic,
    model_input,bind_agents,build_plan,split_audit,answer_schema)
from app.locdi.answer_grounded_routing_evaluation import support_audit
from app.llm.prompt_loader import specialty_system_prompt
from app.locdi.routing_data_reuse import HOSPITAL,BANK,HISTORICAL

NS='c4_24_v1'
SOURCE=ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'
REGISTRY=ROOT/'configs/agent_registry.specialty_development.json'
CONFIG=ROOT/'configs/development_specialty_llm.json'
CODE=['app/locdi/answer_grounded_routing.py','app/locdi/answer_grounded_routing_evaluation.py',
      'scripts/prepare_answer_grounded_routing.py','tests/locdi/test_answer_grounded_routing.py']
SETTINGS={'temperature':0,'seed':42,'context':32768,'max_output_tokens':1024,'timeout_seconds':120,'retry':0,'repair':0}


def freeze():
    pub,pr=locations(NS)
    if pub.exists() or pr.exists():raise ValueError('NAMESPACE_ALREADY_USED_NO_RESUME')
    parent=ROOT/'outputs/private/research_evaluation/c4_23_v1'
    pins=read(parent/'protected_hashes.json');sources=read(parent/'source_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_23_v1/output_manifest.json';m=read(mp)
    assert digest({k:v for k,v in m.items() if k!='fingerprint'})==m['fingerprint']
    ip=ROOT/m['private_index_path'];assert sha(ip)==m['private_index_sha256']
    for n,h in {**m['files'],**read(ip)['files']}.items():
        if n in pins:assert pins[n]==h
        pins[n]=h
    for p in (mp,ip):pins[p.relative_to(ROOT).as_posix()]=sha(p)
    a,b=check(pins),check(sources);assert a['status']==b['status']=='PASS'
    write(pr/'protected_hashes.json',pins);write(pr/'source_hashes.json',sources)
    write(pr/'initial_preservation.json',{'protected':a,'sources':b})
    write(pr/'code_hashes.json',{n:sha(ROOT/n) for n in CODE})
    protocol={'version':VERSION,'target':TARGET,'evidence_level':'DEVELOPMENT_BENCHMARK_INFRASTRUCTURE',
        'preferred':'Explicit ordered options and answer ID from admitted QA data; no such Raw admitted in C3-09',
        'available_fallback':'Separate AgentClinic-MedQA static full-source diagnostic problem; no fabricated options',
        'projection':['Objective_for_Doctor','Patient_Actor','Physical_Examination_Findings','Test_Results'],
        'release':'All projected source facts disclosed at once; no actions, delays or longitudinal memories invented',
        'eligibility_before_responses':['Pinned official107-row source and existing schema','Explicit nonempty Correct_Diagnosis',
            'Exclude whole problem on normalized word-boundary exact reference phrase in full static input',
            'Reject duplicate ID, normalized question, input fingerprint or source case group; no model-output-dependent filtering'],
        'selection':'First5 eligible source ordinals in original order; fixed before any new response',
        'reference':'DATASET_DECLARED_ANSWER, private only; free-text exact lexical normalization reuses agentclinic.lexical_scoring.v1, not clinical equivalence',
        'split':'Original split UNKNOWN preserved; all107 development-exposed by C3-11; no new independent Test designation',
        'sft_and_pretraining':'Only bounded local artifacts can be audited; absence of local training corpus does not prove no pretraining exposure',
        'prompt_version':'locdi.answer_grounded.task_prompt.v1','common_task_prompt_fingerprint':digest(TASK_PROMPT),
        'settings':SETTINGS,'max_generations':25,'generation':0,'fitting':0,
        'core_injection':False,'clinical_gate_bypass':False,'live_runner':'NOT_IMPLEMENTED_IN_C4_24_TEST_ONLY_COLLECTOR',
        'pilot_namespace':'c4_24_v1','planned_run_id':'c4_24_pilot_001'}
    protocol['fingerprint']=digest(protocol);write(pub/'benchmark_protocol.json',protocol)
    print(json.dumps({'freeze':'PASS','protected':len(pins),'sources':len(sources)}),flush=True)


def prepare():
    pub,pr=locations(NS)
    if (pr/'preparation_started.json').exists():raise ValueError('NO_RESUME')
    assert check(read(pr/'code_hashes.json'))['status']=='PASS'
    protocol=read(pub/'benchmark_protocol.json')
    assert protocol['fingerprint']==digest({k:v for k,v in protocol.items() if k!='fingerprint'})
    write(pr/'preparation_started.json',{'protocol_fingerprint':protocol['fingerprint'],'generation':0})
    sourcepins={str(p):sha(p) for p in [SOURCE,REGISTRY,CONFIG,BANK,HISTORICAL,
        SOURCE.parent/'LICENSE.txt',HOSPITAL/'data/raw/_medqa.jsonl'] if p.is_file()}
    lic=read(ROOT/'outputs/research_evaluation/c3_10_v1/license_and_source_notes.json')
    assert sha(SOURCE.parent/'LICENSE.txt')==lic['license_file_sha256']
    entries,excluded=read_agentclinic(SOURCE)
    questions=[q for _,q,_ in entries];refs=[r for _,_,r in entries]
    audit=split_audit(questions);assert not audit['issues']
    registry=read(REGISTRY);config=read(CONFIG)['llm']
    bindings=bind_agents(registry,config,specialty_system_prompt)
    local=inspect_local_model(config['model'],config['models_root'])
    # Inspect the configured manifest's layer types, never model weight contents.
    modelroot=Path(config['models_root'] or os.environ.get('OLLAMA_MODELS',str(Path.home()/'.ollama/models')))
    name,tag=config['model'].split(':');manifest=modelroot/'manifests/registry.ollama.ai/library'/name/tag
    layers=read(manifest).get('layers',[]) if manifest.is_file() else []
    adapter_layers=[r.get('mediaType') for r in layers if 'adapter' in r.get('mediaType','').lower() or 'lora' in r.get('mediaType','').lower()]
    if manifest.is_file():sourcepins[str(manifest)]=sha(manifest)
    scans=[];training_candidates=[]
    # Scoped file-name inventory; do not inspect venv, private outputs or model weights as corpora.
    for root in (ROOT/'datasets',ROOT/'artifacts',HOSPITAL/'data',HOSPITAL/'training',HOSPITAL/'adapters',ROOT.parent/'001_synthea-docker/experiments'):
        count=0
        if root.is_dir():
            for directory,dirs,files in os.walk(root):
                dirs[:]=[d for d in dirs if d not in ('.venv','venv','.git','node_modules','__pycache__')]
                for f in files:
                    count+=1
                    if any(t in f.lower() for t in ('sft','lora','adapter_config','medqa')):
                        training_candidates.append(str(Path(directory)/f))
        scans.append({'path':str(root),'exists':root.exists(),'files_scanned':count})
    write(pr/'local_training_inventory.json',{'roots':scans,'name_candidates':training_candidates,
        'complete_training_corpus_available':False,'current_manifest_adapter_layers':adapter_layers})
    bank=[json.loads(l) for l in BANK.read_text(encoding='utf-8-sig').splitlines() if l.strip()] if BANK.is_file() else []
    historical=read(HISTORICAL) if HISTORICAL.is_file() else []
    inventory=[{'dataset':'MedQA QA-only','actual_admitted_files':0,'eligible_questions':0,'status':'ACQUISITION_BLOCKED_IN_C3_09','answer_format':'OPTION_ID_NOT_LOCALLY_ADMITTED'},
        {'dataset':'AgentClinic-MedQA pinned official','records':107,'explicit_source_answers':107,'public_options':0,'eligible_questions':len(entries),'excluded_reference_literal':len(excluded),
         'original_split':'UNKNOWN','source_revision':protocol.get('source_revision',questions[0]['dataset_version'] if questions else None),
         'source_sha256':sha(SOURCE),'reference':'DATASET_DECLARED_FREE_TEXT_EXACT_ONLY','license':lic['license'],'license_evidence_sha256':sha(SOURCE.parent/'LICENSE.txt'),
         'usage_condition':'Retain MIT copyright/permission notice; project execution approval for new static benchmark pending'},
        {'dataset':'Synthea Training Bank','records':len(bank),'patients':len({r['training_patient_id'] for r in bank}),'eligible_explicit_answer_questions':0,'reason':'Source Condition labels not independently approved problem answers'},
        {'dataset':'Hospital historical routing_train','records':len(historical),'eligible_explicit_answer_questions':0,'reason':'Historical model outcomes/boolean labels are not a question-answer benchmark'},
        {'dataset':'HealthBench','records':5000,'eligible_explicit_answer_questions':0,'status':'EXISTING_RUBRIC_BENCHMARK_OUTSIDE_EXACT_DIAGNOSTIC_ANSWER_CONTRACT'}]
    write(pub/'dataset_inventory.json',{'datasets':inventory,'bounded_search':True,'no_new_acquisition':True,
        'qa_only_external_candidate':'awinml/medqa pinned afdb980b627374a7f10a0a11ed68455596c52742; unresolved acquisition/use evidence unchanged',
        'current_sft_corpus':'NOT_FOUND_IN_SCANNED_ROOTS_NOT_PROOF_OF_NO_TRAINING','pretraining_overlap':'UNKNOWN'})
    write(pr/'question_reference_manifest.json',{'protocol_fingerprint':protocol['fingerprint'],
        'questions':questions,'references':refs,'original_ordinals':[i for i,_,_ in entries],'excluded':excluded,
        'source_pins':sourcepins,'license_evidence_path':str(SOURCE.parent/'LICENSE.txt')})
    write(pub/'question_reference_manifest.json',{'eligible_questions':len(entries),'explicit_answer_id_questions':0,
        'explicit_free_text_reference_questions':len(entries),'excluded':len(excluded),
        'private_manifest_path':'outputs/private/research_evaluation/c4_24_v1/question_reference_manifest.json',
        'private_manifest_sha256':sha(pr/'question_reference_manifest.json'),'source_fields_not_published':True})
    write(pr/'split_and_exposure_audit.json',audit)
    write(pub/'split_and_exposure_audit.json',{'registered_source_cases':107,'eligible':len(entries),'duplicate_issues':len(audit['issues']),
        'original_split':'UNKNOWN_PRESERVED','evaluation_role':'DEVELOPMENT_ONLY','independent_test_ready':False,
        'development_exposed_source_cases':107,'prior_live_doctor_cohort_ordinals':[1,2,3,4,5],
        'same_source_case_reuse':'All eligible cases are derived from previously adapted C3-11 source cases',
        'prior_sft_corpus_overlap':'UNKNOWN_COMPLETE_CORPUS_NOT_AVAILABLE','current_lora_adapter_binding':adapter_layers,
        'model_pretraining_exposure':'UNKNOWN','patient_level_disjointness':'UNKNOWN_NO_PATIENT_IDENTIFIERS_IN_SOURCE',
        'question_id_text_input_case_group_checks':'PASS_WITHIN_ELIGIBLE_CURRENT_SOURCE','original_split_reassigned':False})
    write(pr/'agent_binding_inventory.json',{'registry':registry,'llm_config':config,'bindings':bindings,'disk_model_check':local,
        'model_layer_types':[r.get('mediaType') for r in layers]})
    write(pub/'agent_binding_inventory.json',{'registry_sha256':sha(REGISTRY),'registry_fingerprint':digest(registry),
        'agents':[{k:v for k,v in a.items() if k!='system_prompt'} for a in bindings],
        'shared_base_model':len({a['model'] for a in bindings})==1,'shared_digest':len({a['model_identifier'] for a in bindings})==1,
        'disk_model_check':local,'disk_digest_matches_pin':local.get('model_identifier')==config['model_identifier'],
        'adapter_lora_layers_in_configured_manifest':adapter_layers,'trained_specialty_models_evidenced':False,
        'server_identity_context':'NOT_CHECKED_NO_MODEL_OR_SERVER_CALLS'})
    n=min(5,len(entries));pilotq=questions[:n];pilotr=refs[:n]
    plan=build_plan(pilotq,pilotr,bindings,SETTINGS) if n==5 else None
    write(pr/'pilot_execution_plan.json',{'plan':plan,'questions':pilotq,'references':pilotr,'bindings':bindings,
        'selected_source_ordinals':[o for o,_,_ in entries[:n]],'source_pins':sourcepins})
    inputs=[]
    for q in pilotq:
        for a in bindings:inputs.append({'question_id':q['question_id'],'agent_id':a['agent_id'],
            'system':a['system_prompt'],'user':model_input(q),'schema':answer_schema(q),'input_fingerprint':q['input_fingerprint']})
    write(pr/'pilot_model_inputs.json',inputs)
    write(pr/'potential_similarity_support.json',support_audit(pilotq))
    write(pub/'pilot_execution_plan.json',{'status':'PREPARED_DEVELOPMENT_FREE_TEXT_PLAN_LIVE_BLOCKED' if plan else 'INSUFFICIENT_ELIGIBLE_QUESTIONS',
        'question_count':n,'agent_count':len(bindings),'max_generations':25 if plan else 0,'planned_run_id':'c4_24_pilot_001',
        'pilot_namespace':'c4_24_v1','protocol_fingerprint':protocol['fingerprint'],'plan_fingerprint':plan['fingerprint'] if plan else None,
        'settings':SETTINGS,'actual_generation':0,'execution_authorized':False,'fitting_authorized':False,
        'private_plan_path':'outputs/private/research_evaluation/c4_24_v1/pilot_execution_plan.json',
        'input_comparison':'Same question/options per5 agents; system prompt differs by specialty',
        'live_blockers':['NEW_OPERATOR_EXECUTION_APPROVAL_REQUIRED','CURRENT_SERVER_DIGEST_CONTEXT_RECHECK_REQUIRED','DEDICATED_APPROVAL_GATED_LIVE_RUNNER_NOT_IMPLEMENTED'],
        'original_split':'UNKNOWN_PRESERVED','development_only':True,'reuse_previous_c3_c4_execution_approval':False})
    write(pub/'benchmark_eligibility.json',{'status':'C4_24_PARTIAL_BENCHMARK_COVERAGE','source_answer_records':107,
        'eligible_static_diagnostic_questions':len(entries),'mcq_eligible':0,'reference_literal_excluded':len(excluded),
        'diagnosis_eligibility':'All source OSCE objectives are diagnosis tasks by existing validated source schema; all projected facts disclosed; no clinical suitability inferred',
        'scoring':'Exact lexical answer agreement only, no synonyms/clinical equivalence','independent_test':False,
        'can_compare_agents':'On explicit dataset-answer agreement after authorized collection; current real performance NOT_EVALUATED',
        'utility_fit_ready':False,'clinical_routing_reference_ready':False,'generation':0})
    write(pub/'answer_scoring_spec.json',{'version':VERSION,'schema':'answer_schema(question) JSON + semantic supported/unresolved checks',
        'states':list(__import__('app.locdi.answer_grounded_routing',fromlist=['STATES']).STATES),
        'mcq':'Exact option ID; supported requires nonnull in-catalog ID and diagnosis=null',
        'free_text':'Existing NFC/whitespace/casefold/trailing-period normalization exact equality; mismatch means lexical incorrect only',
        'unresolved':'Both answer_id and diagnosis null; correctness=null',
        'invalid':'JSON/schema/semantic failure; correctness=null; stop collection',
        'generation_failed':'Transport failure distinct from invalid response; correctness=null',
        'not_attempted':'Never called, including post-stop/budget slots; correctness=null',
        'rationale':'Stored privately; not adjudicated for medical truth','clinical_correctness_created':False})
    write(pub/'evaluation_metric_spec.json',{'denominator':'Registered question-agent slots; selected-agent metrics one registered choice per question',
        'answer_coverage':'(CORRECT_ANSWER+INCORRECT_ANSWER)/N',
        'selective_answer_accuracy':'CORRECT_ANSWER/(CORRECT_ANSWER+INCORRECT_ANSWER)',
        'correct_supported_answer_rate':'CORRECT_ANSWER/N; unresolved is no successful answer, not incorrect bool',
        'unresolved_rate':'UNRESOLVED/N','failure_rate':'(INVALID_RESPONSE+GENERATION_FAILED)/N',
        'not_attempted_rate':'NOT_ATTEMPTED/N','zero_denominator':'null / NOT_APPLICABLE',
        'complete_collection':'All5 schema-valid including unresolved','complete_scorable':'All5 supported and deterministic dataset-answer-scorable',
        'agent_disagreement':'Different normalized answer/unresolved signatures / fully collected comparable questions',
        'oracle':'Any correct among5 / completely collected questions only; posthoc evaluation upper bound',
        'static':'Pre-fixed agent ID, no answer access','rule_similarity':'Possible reuse AgentRegistry.route mode=rule on public task projection with fixed candidate set',
        'adaptive':'MISSING_UTILITY until independent task-scoped target/data/validation/approval contract; not implemented benchmark fitting',
        'clinical_routing_accuracy':'NOT_EVALUATED'})
    print(json.dumps({'prepared':'PASS','eligible':len(entries),'excluded':len(excluded),'mcq':0,'pilot':n,'generation':0}),flush=True)


def dry_run(execute=False):
    pub,pr=locations(NS)
    p=read(pr/'pilot_execution_plan.json')
    if execute:raise ValueError('LIVE_EXECUTION_DISABLED_C4_24_SEPARATE_APPROVAL_GATED_RUNNER_REQUIRED')
    assert p['plan']['fingerprint']==digest({k:v for k,v in p['plan'].items() if k!='fingerprint'})
    assert all(sha(Path(n))==h for n,h in p['source_pins'].items())
    print(json.dumps({'mode':'DRY_RUN','tasks':len(p['plan']['tasks']),'max_generations':25,'actual_generation':0,
        'planned_run_id':p['plan']['run_id'],'execution_authorized':False,'live_runner':'BLOCKED_NOT_IMPLEMENTED'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','prepare','dry-run'],nargs='?',default='dry-run');p.add_argument('--execute',action='store_true')
    args=p.parse_args();socket.create_connection=offline;socket.socket.connect=offline
    if args.execute and args.phase!='dry-run':p.error('execute flag is disabled for preparation')
    try:
        if args.phase=='freeze':freeze()
        elif args.phase=='prepare':prepare()
        else:dry_run(args.execute)
    except ValueError as e:
        print(json.dumps({'status':'BLOCKED','error_code':str(e) if str(e).isupper() else type(e).__name__,'generation':0}));sys.exit(2)
