"""C3-06 frozen documentary notes and executable contract inventory. No acquisition."""
from app.locdi.llm_setup import digest
from app.locdi.reference_acquisition import fingerprint,validate_plan
from app.locdi.external_reference_task import TASK_SCHEMA,medcase_spec
from app.locdi.external_reference_readiness import evaluate_gate


def metadata_notes():
    # Evidence snippets are documentation, never accepted operator licenses/approval.
    common={'reviewed_on':'2026-11-02','timezone':'Asia/Seoul','new_raw_downloaded':False,
      'accepted_usage_policy':None,'approval_reference':None,'acquired_raw_schema':'UNVERIFIED',
      'contamination_status':'UNKNOWN','patient_clinical_gold_transfer_allowed':False}
    return [dict(common,dataset_id='MedQA',current_documentary_review='CONFIRMED',priority=1,
      official_urls=['https://github.com/jind11/MedQA',
       'https://raw.githubusercontent.com/jind11/MedQA/27b02f66aac217933c9648a06f82e9f720377925/README.md',
       'https://raw.githubusercontent.com/jind11/MedQA/27b02f66aac217933c9648a06f82e9f720377925/LICENSE',
       'https://raw.githubusercontent.com/jind11/MedQA/master/IR/aristomini/common/models.py'],
      observations=['Official README links QA+textbook Google Drive archive, JSONL, train/dev/test and four-option variant',
       'Official current reader uses question, options indexed by A.. and answer_idx letter; pinned reader fetch unavailable',
       'Repository MIT software license is not acceptance of archive/textbook usage rights'],
      code_revision='27b02f66aac217933c9648a06f82e9f720377925',data_revision=None,archive_sha256=None,
      expected_files=None,expected_file_bytes=None,download_availability='DOCUMENTED_LINK_NOT_REQUESTED',
      license_status='DATA_ARCHIVE_SCOPE_REVIEW_REQUIRED',license_excerpt='MIT License',
      index_convention='medqa_letter',split_names=['train','dev','test'],
      split_mapping='Raw dev preserved as original_split dev; explicit adapter contract may use validation alias only after pinned mapping'),
      dict(common,dataset_id='MedMCQA',current_documentary_review='CONFIRMED',priority=2,
      official_urls=['https://github.com/medmcqa/medmcqa',
       'https://raw.githubusercontent.com/medmcqa/medmcqa/c59ef14ca1990266c4107c7864b45a20fd93e5e0/README.md',
       'https://huggingface.co/datasets/openlifescienceai/medmcqa/raw/main/README.md',
       'https://raw.githubusercontent.com/medmcqa/medmcqa/main/dataset.py'],
      observations=['Pinned official README says test ground truth not released',
       'Current official loader subtracts1 from cop; HF feature ClassLabel0..3; text description1..4 conflicts',
       'Repo MIT vs HF Card Apache-2.0 conflict; README Val/Test summary headings differ from HF counts',
       'HF aggregate download_size is not exact per-file expected bytes or accepted file hashes'],
      code_revision='c59ef14ca1990266c4107c7864b45a20fd93e5e0',data_revision='91c6572c454088bf71b679ad90aa8dffcd0d5868',
      pinned_card_recheck='UNAVAILABLE_VIA_RESEARCH_TOOL; current main documentary comparison only',
      archive_sha256=None,expected_files=None,expected_file_bytes=None,
      download_availability='DOCUMENTED_DRIVE_AND_HF_DISTRIBUTION_NO_REQUEST',license_status='CONFLICT_REQUIRES_SCOPE_RESOLUTION',
      split_names=['train','validation','test'],public_label_splits=['train','validation'],withheld_label_splits=['test'],
      index_convention=['medmcqa_csv_one_based','medmcqa_hf_zero_based']),
      dict(common,dataset_id='MedCaseReasoning',current_documentary_review='PARTIAL_PINNED_CARD_UNAVAILABLE',priority=3,
      official_urls=['https://github.com/kevinwu23/Stanford-MedCaseReasoning',
       'https://raw.githubusercontent.com/kevinwu23/Stanford-MedCaseReasoning/main/README.md',
       'https://huggingface.co/datasets/zou-lab/MedCaseReasoning',
       'https://huggingface.co/datasets/zou-lab/MedCaseReasoning/blob/469a5365bc534b5b2b9cfbc52b2ef2a10f43a339/README.md'],
      observations=['Current HF page metadata MIT vs repo Dataset CC-BY4/code MIT; per-article source rights unresolved',
       'Pinned Card unavailable in current research tool; C3-05 prior observation retained, not newly verified revision',
       'Current page lists case_prompt/final_diagnosis/diagnostic_reasoning/pmcid/publication_date; raw schema remains UNVERIFIED',
       'Dataset page incidentally renders public example rows; none acquired locally or reused as test fixtures'],
      code_revision=None,data_revision='469a5365bc534b5b2b9cfbc52b2ef2a10f43a339',archive_sha256=None,
      expected_files=None,expected_file_bytes=None,download_availability='PUBLIC_PARQUET_METADATA_ONLY_NOT_REQUESTED',
      license_status='CONFLICT_REQUIRES_SCOPE_AND_ARTICLE_RIGHTS_RESOLUTION',split_names=['train','val','test'],
      index_convention='NOT_APPLICABLE_FREE_TEXT')]


def acquisition_plans():
    plans=[]
    for note in metadata_notes():
        p={'schema_version':'locdi.reference_acquisition_plan.v1','dataset_id':note['dataset_id'],
          'canonical_url':note['official_urls'][0],'pinned_revision':note['data_revision'],
          'expected_files':[],'accepted_license_use_policy_evidence':None,'maximum_bytes':None,
          'destination_private_namespace':'c3_06_'+note['dataset_id'].lower()+'_reserved',
          'expected_split':{'MedQA':'dev','MedMCQA':'validation','MedCaseReasoning':'val'}[note['dataset_id']],
          'approval_reference':None,'network_request_budget':1,'timeout_seconds':30,'download_attempt_limit':1}
        p['plan_fingerprint']=fingerprint(p);plans.append(p)
    return plans


def contract_outputs(gates=None):
    plans=acquisition_plans()
    notes=metadata_notes()
    return {
     'reference_acquisition_plan':{'schema_version':'locdi.c3_06.plans.v1','mode':'DRY_RUN',
       'actual_downloads':0,'plans':[{'plan':p,'validation':validate_plan(p)} for p in plans],
       'current_phase_execute_allowed':False,'live_cli':'BLOCKED_C3_06_TEST_ONLY',
       'future_interface':'reference_acquisition.acquire(plan, execute=True, allow_download=True, authority=PinnedApproval(...))',
       'attempt_policy':'One dispatch per file, no retry/redirect/extraction; bounded streaming; failure receipt preserved; version never overwritten'},
     'reference_license_and_revision':{'schema_version':'locdi.c3_06.metadata.v1',
       'sources':[{k:v for k,v in n.items() if k!='license_excerpt'} for n in notes],
       'rights_approval_generated':0,'metadata_access_is_data_acquisition':False},
     'medqa_adapter_contract':{'reference_type':'EXAM_GOLD','status':'IMPLEMENTED_TEST_ONLY_RAW_SCHEMA_UNVERIFIED',
       'input_allowlist':['question','options'],'private_fields':['answer_idx','original sample reference','source fingerprint','original split','exposure/contamination'],
       'index':'Exact letter, explicit pinned row/index contract;4 or5 unique options A..D/E in recorded order',
       'duplicate_policy':'Reject same ID or normalized question; absent ID uses private source row ordinal',
       'original_split_policy':'Never reassign; dev accepted as dev, not silently renamed validation',
       'qa_client_contract':'Existing LocalDialogueClient.chat_messages(messages, model=bound_model) interface shape; future exact binding/digest approval required',
       'generation_enabled':False,'patient_runtime_used':False},
     'medmcqa_adapter_contract':{'reference_type':'EXAM_GOLD','status':'IMPLEMENTED_TEST_ONLY_RAW_SCHEMA_UNVERIFIED',
       'label_splits':['train','validation'],'blocked_split':'test','input_allowlist':['question','opa','opb','opc','opd'],
       'private_fields':['cop','exp','subject_name','topic_name','original sample reference','split','exposure'],
       'index':'Explicit medmcqa_csv_one_based1..4 OR medmcqa_hf_zero_based0..3; no autodetect even for overlapping1..3',
       'choice_policy':'single only, multi rejected pending dedicated contract',
       'training_exposure':'TRAINING_SPLIT_DEVELOPMENT_EXPOSURE; no independent clinical Test admission',
       'subject_topic_is_routing_gold':False,'generation_enabled':False},
     'medcase_adapter_spec':medcase_spec(),
     'external_task_schema':TASK_SCHEMA,
     'leakage_guard_contract':{'schema_version':'locdi.c3_06.leakage.v1','model_projection':'Only one user message with question/options, or separately reviewed case_prompt design',
       'public_projection':'Ordinal/count/hash/task-type/split/status only; no item ID, answer or private path',
       'fingerprints':['Exact original row','Pinned index contract and option order','Adapter source','Messages','Private answer key/source relationship'],
       'guards':['Strict field allowlists','Explicit answer/metadata markers','Known complete explanation inclusion','No patient simulation','No Gold/Rubric/Canary sidecars'],
       'limitations':'Lexical/structural guards do not establish semantic absence of answer clues, clinical equivalence or model contamination; row review still required',
       'new_clinical_verdicts':0},
     'evaluation_readiness_gate':{'schema_version':'locdi.c3_06.gates.v1',
       'datasets':gates if gates is not None else [evaluate_gate(p) for p in plans],
       'existing_HealthBench':'D7 pinned raw reused/hash checked; no new grading or readiness promotion',
       'clinical_reference_ready':False,'routing_artifact_ready':False,'official_B0_B4_ready':False}}
