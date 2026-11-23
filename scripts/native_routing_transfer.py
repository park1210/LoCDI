"""Native menu-feature reuse -> disjoint Source validation -> approved transfer.

No default generation or fitting. Each executable action needs a new operator
approval; only the user invokes these modes. Existing Gates/files stay intact.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
from unittest.mock import patch
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest,validate_setup,normalize_model_identifier
from app.locdi.native_menu_routing import VERSION,public_menu,tokens,fit_observation,fitting_feature_scope
from app.locdi.development_routing_proxy import MARKERS
from scripts.run_original50_d0_d2 import M5

NS='NATIVE_ROUTING_TRANSFER_20261123_v1'
PUB=ROOT/'outputs/research_evaluation'/NS; PR=ROOT/'outputs/private/research_evaluation'/NS
VALIDATION='c435_native_validation2_002';FIT='native_public_menu_fit_002'
EVIDENCE=PR/'admission_002/native_transfer_evidence.json'
_VALIDATED_EVIDENCE_FP=None

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:raise FileExistsError('EXISTING_ARTIFACT_DIFFERS_NEW_NAMESPACE_REQUIRED')
    else:atomic_json(path,value)

def native_training():
    from scripts.run_u2_native_collection import load
    return load('Full')

def derive_training():
    from app.locdi.common_prediction_contract import extract_core_prediction
    from app.llm.prompt_loader import specialty_system_prompt
    _,pr,p=native_training();cfg=read(pr/'llm_config.json');c=read(pr/p['run_id']/'collection.json')
    if c['status']!='COMPLETED' or c['plan_fingerprint']!=p['fingerprint'] or len(c['rows'])!=50:raise ValueError('ACTUAL_NATIVE50_REQUIRED')
    patients={r['ordinal']:r for r in read(pr/'patients.json')};rows=[];proofs=[];origins={**p['pins']}
    for ordinal,patient in patients.items():
        rr=[r for r in c['rows'] if r['ordinal']==ordinal]
        if len(rr)!=5 or {r['agent_id'] for r in rr}!={a['agent_id'] for a in p['agents']}:raise ValueError('FIVE_AGENT_GROUP_REQUIRED')
        signatures=set()
        for row in rr:
            folder=pr/p['run_id']/('Patient%02d'%ordinal)/row['department']/'Episode0001'
            bb=sorted(folder.glob('Request*_before.json'));aa=sorted(folder.glob('Request*_after.json'))
            if row['status']!='COMPLETED' or not bb or len(bb)!=len(aa) or len(bb)!=row['generation']:raise ValueError('COMPLETE_ACTUAL_NATIVE_RESPONSE_REQUIRED')
            for bf,af in zip(bb,aa):
                b,a=read(bf),read(af);q=b['request'];body=json.loads(q['messages'][1]['content'])
                if b['contract']!=p['fingerprint'] or digest(q)!=b['request_fingerprint'] or a.get('request_fingerprint')!=b['request_fingerprint'] or a.get('status')!='VALID' or a.get('HTTP_status')!=200:raise ValueError('RAW_RESPONSE_LINEAGE')
                if q['model']!=cfg['model'] or q['messages'][0]['content']!=specialty_system_prompt(row['department']) or body['role']!='specialty' or body['department']!=row['department']:raise ValueError('ACTUAL_PROMPT_MODEL_ROLE')
                if q['options']!={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42}:raise ValueError('ACTUAL_DECODING')
                if a['identity_after']['context_length']!=32768 or any(normalize_model_identifier(x['digest'])!=cfg['model_identifier'] for x in (b['identity_before'],a['identity_after'])):raise ValueError('ACTUAL_IDENTITY_CONTEXT')
                origins[str(bf.resolve())]=sha_file(bf);origins[str(af.resolve())]=sha_file(af)
            initial=json.loads(read(bb[0])['request']['messages'][1]['content']);menu=public_menu(initial['source_action_metadata'])
            if menu!=public_menu(patient['binding']['rows']):raise ValueError('INITIAL_PUBLIC_MENU_NOT_EXACT_SOURCE')
            obs=initial['observation'];signatures.add(digest({'observation':obs,'menu':menu,'evidence':initial.get('evidence')}))
            actual=extract_core_prediction(read(folder/'actual_core_decisions.json'),row['result']['status'])
            if actual!=row['result']['prediction']:raise ValueError('FINAL_CORE_PREDICTION_MISMATCH')
            correct=actual['diagnosis']==patient['reference']['label'] if actual['certainty']=='supported' else None
            if patient['split']=='train':
                rows.append({'patient_id':patient['case']['patient_id'],'decision_id':patient['case']['case_id']+':initial_native_episode',
                    'agent_id':row['agent_id'],'split':'train','observation':fit_observation(obs,menu),'correct':correct})
            proofs.append({'ordinal':ordinal,'split':patient['split'],'agent_id':row['agent_id'],
                'legacy_token_count':len(__import__('app.core.routing_utility',fromlist=['observed_tokens']).observed_tokens(obs)),
                'menu_token_count':len(tokens(obs,menu)),'correct':correct,'certainty':actual['certainty']})
        if len(signatures)!=1:raise ValueError('FIVE_AGENT_INITIAL_PUBLIC_INFORMATION_DIFFERS')
    for f in (pr/'patients.json',pr/'llm_config.json',pr/p['run_id']/'collection.json'):origins[str(f.resolve())]=sha_file(f)
    if any(type(r['correct']) is not bool for r in rows):raise ValueError('UNSCORABLE_TRAIN_GROUP_NO_IMPUTATION')
    if len(rows)!=40 or any(not tokens(r['observation'],r['observation']['_routing_public_menu']) for r in rows):raise ValueError('NONEMPTY_TRAIN8_REQUIRED')
    return rows,proofs,origins,p,cfg

def prepare():
    if (PUB/'preparation_002.json').exists():raise FileExistsError('PREPARATION_EXISTS_USE_STATUS')
    PUB.mkdir(parents=True,exist_ok=True);PR.mkdir(parents=True,exist_ok=True)
    rows,proofs,origins,np,cfg=derive_training();model=read(M5)
    from app.locdi.c435_native_validation_source import wrapper,reference
    candidates_path=ROOT/'outputs/private/research_evaluation/ROUTING_ROOT_CAUSE_20261123_v1/C4_35_validation_source_candidates_private.json'
    admitted_path=ROOT/'outputs/private/research_evaluation/c4_35_v1/approved_admission.json'
    admitted=read(admitted_path)
    if admitted['status']!='ADMISSION_APPROVED_EVALUATION_NOT_APPROVED' or digest(admitted['panel'])!=admitted['panel_fingerprint']:raise ValueError('EXISTING_C435_ADMISSION_NOT_VERIFIED')
    candidates=read(candidates_path);source={digest(v) for v in admitted['panel']}
    if any(digest(c['visit']) not in source for c in candidates):raise ValueError('C435_CANDIDATE_NOT_IN_APPROVED_PANEL')
    eval_ids={c['patient_id'] for c in original_cases()};train_ids={r['patient_id'] for r in rows};m5_ids=set(model['provenance']['training_patients'])
    all_eligible={}
    for c in sorted(candidates,key=lambda c:(c['visit']['decision_time'],c['visit_id'])):
        pid=c['patient_id']
        if pid in eval_ids|train_ids|m5_ids:raise ValueError('C435_PATIENT_OVERLAP')
        all_eligible.setdefault(pid,c)
    selected=sorted(all_eligible,key=lambda pid:digest(['c435-native-validation',42,pid]))[:2]
    if len(selected)!=2:raise ValueError('TWO_DISJOINT_VALIDATION_PATIENTS_REQUIRED')
    validation=[]
    for i,pid in enumerate(selected,1):
        c=all_eligible[pid];case,binding=wrapper(c,model)
        obs_file=Path(next(iter(c['visit']['values'].values()))['source_file']);ref=reference(c,obs_file.with_name('conditions.csv'))
        validation.append({'ordinal':i,'split':'validation','case':case,'binding':binding,'reference':ref,
            'provenance':{'prior_VoI_evaluation_exposed':True,'untouched_holdout':False,'routing_outcomes_not_yet_observed':True,
                          'patient_disjoint_M5_Train_RoutingTrain_Original50':True,'bounded_export_independence_only':True}})
    vp=PR/VALIDATION;vpub=PUB/VALIDATION
    write_new(vp/'patients.json',validation);write_new(vp/'llm_config.json',cfg)
    write_new(PR/'train_fit_rows_private.json',rows);write_new(PR/'training_reuse_audit_private.json',proofs)
    collection_index={'candidates':[{'decision':{'patient_id':pid,'split':'train'}} for pid in sorted(train_ids)]+[
        {'decision':{'patient_id':r['case']['patient_id'],'split':'validation'}} for r in validation],
        'purpose':'NATIVE_SOURCE_PROXY_ROUTING_ONLY','feature_version':VERSION,'clinical_reviewed':False,
        'validation_exposure':'PREVIOUS_VOI_DEVELOPMENT_EVALUATION_NOT_UNTOUCHED_HOLDOUT'}
    write_new(PR/'collection_index.json',collection_index)
    origins.update({str(f.resolve()):sha_file(f) for f in (candidates_path,admitted_path,PR/'train_fit_rows_private.json',vp/'patients.json',vp/'llm_config.json',PR/'collection_index.json',M5)})
    for r in validation:
        for f,h in r['case']['provenance']['source_sha256'].items():origins[str(Path(f).resolve())]=h
        f=Path(r['reference']['source_file']);origins[str(f.resolve())]=sha_file(f)
    code=['app/locdi/native_menu_routing.py','app/locdi/c435_native_validation_source.py','scripts/native_routing_transfer.py',
          'scripts/run_native_menu_full50.py','scripts/Invoke-LoCDINativeTransfer.ps1','scripts/run_u2_native_collection.py']
    origins.update({str((ROOT/f).resolve()):sha_file(ROOT/f) for f in code})
    write_new(PR/'source_code_pins_002.json',origins)
    agents=[a for a in np['registry']['agents'] if a['role']=='specialty']
    p={'plan_id':VALIDATION,'run_id':'run_'+VALIDATION,'scope':'B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY',
        'experiment_scope':'PATIENT_DISJOINT_C435_NATIVE_ROUTING_VALIDATION_ONLY','conditions':['B2'],
        'features_note':'fixed5-specialty intervention, not B2 evaluation','feature_version':VERSION,
        'agents':agents,'registry':np['registry'],'registered_patients':2,'registered_episodes':10,'generation_total':120,
        'episode_generation':12,'wall_seconds':3600,'process_bytes':4*1024**3,'output_bytes':2*1024**3,
        'validation_M5_training_overlap':0,'B4_admission':'PENDING_FIT_AND_ACTUAL_NATIVE_VALIDATION',
        'pins':origins,'retry':0,'repair':0,'warmup':0,'no_resume':True,**MARKERS}
    p['fingerprint']=digest(p);write_new(vpub/'plan.json',p)
    a={**MARKERS,'scope':p['scope'],'conditions':p['conditions'],'plan_fingerprint':p['fingerprint'],
        'experiment_scope':p['experiment_scope'],'max_generation':120,'clinical_release_approved':False,
        'operator_ref':None,'user_statement':None,'recorded_at':None,'expires_at':None}
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy','accept_common_m5_menu','accept_native_training_intervention','accept_source_reference_date_proxy','accept_prior_VoI_exposure_and_bounded_independence'):a[k]=False
    write_new(vp/'approval_template.json',a)
    report={'status':'SOURCE_AND_FEATURES_PREPARED_NOT_ADMITTED','old_Native_episodes_verified':50,'reused_train_groups':8,'train_rows':40,
        'legacy_initial_nonempty_groups':sum(next(x for x in proofs if x['ordinal']==i)['legacy_token_count']>0 for i in range(1,9)),
        'new_initial_nonempty_groups':8,'feature_version':VERSION,'feature_is_menu_not_performed_tests':True,
        'new_native_validation_patients':2,'new_validation_episodes':10,'validation_generation_cap':120,
        'M5_RoutingTrain_Original50_patient_overlap':0,'untouched_validation_holdout':False,
        'validation_past_VoI_development_exposure':True,'new_LLM_generation':0,'new_fitting':0,
        'training_outcomes':{a['department']:dict(Counter(str(r['correct']) for r in rows if r['agent_id']==a['agent_id'])) for a in agents},**MARKERS}
    write_new(PUB/'preparation_002.json',report)
    write_new(PUB/'stage_contracts_not_executable.json',{s:{'patients':n,'episodes':n*5,'max_generation':n*60,
        'status':'REQUIRES_ACTUAL_VALIDATION_FITTING_ADMISSION_AND_NEW_PINNED_PLAN'} for s,n in [('Smoke5',5),('Verify10',10),('Full50',50)]})
    return report

def load_validation():
    p=read(PUB/VALIDATION/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('VALIDATION_PLAN_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('SOURCE_CODE_PIN_CHANGED')
    return PUB/VALIDATION,PR/VALIDATION,p

def validation_check():
    pub,pr,p=load_validation();blocks=[]
    if (pr/p['run_id']).exists():blocks.append('RUN_USED_NO_RESUME')
    proof=PUB/'TEST_ONLY_validation_and_full250.json'
    if not proof.exists() or read(proof).get('status')!='PASS':blocks.append('FRESH_NATIVE_FAKE_VALIDATION_REQUIRED')
    return {'status':'BLOCKED' if blocks else 'STATIC_READY_USER_APPROVAL_REQUIRED','blocker_codes':blocks,
        'plan_id':VALIDATION,'registered_episodes':10,'generation_cap':120,'new_generation':0,
        'independence':'PATIENT_DISJOINT_PREVIOUS_VOI_EXPOSURE_NOT_UNTOUCHED_CLINICAL_HOLDOUT'}

@contextmanager
def validation_collector_scope():
    from scripts import run_u2_native_collection as old
    original_permission=old.permission
    def permission(a,p):
        original_permission(a,p)
        if a.get('accept_prior_VoI_exposure_and_bounded_independence') is not True:raise ValueError('NEW_VALIDATION_SOURCE_PURPOSE_ACCEPTANCE_REQUIRED')
    with patch.object(old,'load',new=lambda stage:load_validation()),patch.object(old,'check',new=lambda stage:validation_check()),patch.object(old,'permission',new=permission):yield old

def validation_analyze():
    with validation_collector_scope() as collector:return collector.analyze('Validation')

def fit_check():
    rows,proofs,origins,p,cfg=derive_training();blocks=[]
    if (PR/FIT).exists():blocks.append('FIT_NAMESPACE_ALREADY_USED_NO_DUPLICATE_FITTING')
    frozen=read(PR/'train_fit_rows_private.json')
    if digest(rows)!=digest(frozen):blocks.append('FROZEN_TRAIN_ROWS_CHANGED')
    for f,h in read(PR/'source_code_pins_002.json').items():
        if sha_file(f)!=h:raise ValueError('SOURCE_CODE_PIN_CHANGED')
    return {'status':'BLOCKED' if blocks else 'FIT_READY_SEPARATE_USER_APPROVAL_REQUIRED','blocker_codes':blocks,
        'complete_scorable_train_groups':8,'fit_rows':40,'feature_version':VERSION,'fit_calls':0,'B34_admitted':False}

def fit(operator,statement):
    if fit_check()['status']=='BLOCKED' or not operator.strip() or not statement.strip():raise ValueError('EXPLICIT_NEW_FITTING_APPROVAL_REQUIRED')
    from app.core.routing_utility import fit_utility
    from app.core.agent_registry import AgentRegistry
    p=read(PUB/VALIDATION/'plan.json');out=PR/FIT;out.mkdir(exist_ok=False)
    write_new(out/'operator_approval.json',{'operator_ref':operator,'user_statement':statement,'scope':'NATIVE_MENU_FEATURE_TRAIN8_SOURCE_PROXY_FIT_ONCE',
        'recorded_at':datetime.now(timezone.utc).isoformat(),'clinical_approval':False})
    rows=read(PR/'train_fit_rows_private.json')
    with fitting_feature_scope():artifact=fit_utility(rows,AgentRegistry(p['registry']),min_samples=2)
    value={'schema_version':'locdi.native_menu_candidate.v1','artifact':artifact,'feature_version':VERSION,
        'source_pins':read(PR/'source_code_pins_002.json'),'fit_rows_fingerprint':digest(rows),'fit_scope':'TRAIN8_ONLY_OLD_VALIDATION2_EXCLUDED',
        'B34_admitted':False,**MARKERS}
    write_new(out/'candidate_utility.json',value)
    return {'status':'CANDIDATE_FITTED_NOT_ADMITTED','artifact_sha256':sha_file(out/'candidate_utility.json'),'fit_calls':1,'new_generation':0}

def verify_validation_prediction(row,patient,folder):
    from app.locdi.common_prediction_contract import extract_core_prediction
    actual=extract_core_prediction(read(Path(folder)/'actual_core_decisions.json'),row['result']['status'])
    if actual!=row['result']['prediction']:raise ValueError('VALIDATION_FINAL_CORE_PREDICTION_MISMATCH')
    expected=actual['diagnosis']==patient['reference']['label'] if actual['certainty']=='supported' else None
    if type(expected)!=type(row['correctness_proxy']) or expected!=row['correctness_proxy']:raise ValueError('VALIDATION_SOURCE_SCORE_MISMATCH')
    return expected

def admission_check(*,write=False):
    global _VALIDATED_EVIDENCE_FP
    blocks=[];fit_path=PR/FIT/'candidate_utility.json';_,vpr,vp=load_validation();run=vpr/vp['run_id']
    if not fit_path.exists():blocks.append('ACTUAL_SEPARATELY_APPROVED_FIT_REQUIRED')
    if not (run/'collection.json').exists():blocks.append('ACTUAL_DISJOINT_NATIVE_VALIDATION_REQUIRED')
    if blocks:return {'status':'BLOCKED','blocker_codes':blocks,'new_generation':0,'fit_calls':0}
    candidate=read(fit_path);rows,_,_,_,cfg=derive_training()
    if candidate['feature_version']!=VERSION or candidate['fit_rows_fingerprint']!=digest(rows):raise ValueError('FEATURE_TRAIN_LINEAGE_CHANGED')
    from app.core.agent_registry import AgentRegistry
    from app.core.routing_utility import RoutingUtility
    expected=[{**{k:r[k] for k in ('patient_id','decision_id','agent_id','correct')},'tokens':tokens(r['observation'],r['observation']['_routing_public_menu'])} for r in rows]
    if sorted(map(digest,expected))!=sorted(map(digest,candidate['artifact']['samples'])) or candidate['artifact']['min_samples']!=2:raise ValueError('EXACT_ACTUAL_FIT_LINEAGE')
    registry=AgentRegistry(vp['registry']);RoutingUtility(candidate['artifact'],registry)
    summary=validation_analyze()
    details=read(run/'analysis_private.json');patients=read(vpr/'patients.json');val_ids={r['case']['patient_id'] for r in patients}
    excluded={r['patient_id'] for r in rows}|set(read(M5)['provenance']['training_patients'])|{c['patient_id'] for c in original_cases()}
    if val_ids&excluded or len(val_ids)!=2:raise ValueError('VALIDATION_M5_TRAIN_EVAL_PATIENT_OVERLAP')
    if summary['status']!='COMPLETED' or len(details['groups'])!=2 or any(not g['complete_collection'] for g in details['groups']):return {'status':'BLOCKED','blocker_codes':['TWO_COMPLETE_NATIVE_VALIDATION_GROUPS_REQUIRED'],'fit_calls':0,'new_generation':0}
    # Validation unresolved stays null, with registered denominators reported.
    # Boolean outcomes are required for the existing TRAIN fitting Gate only.
    approval=read(run/'approval.json')
    if approval.get('plan_fingerprint')!=vp['fingerprint'] or approval.get('accept_prior_VoI_exposure_and_bounded_independence') is not True or approval.get('operator_approved') is not True:raise ValueError('ACTUAL_SOURCE_PURPOSE_OPERATOR_APPROVAL_REQUIRED')
    if read(run/'started.json').get('TEST_ONLY') is not False:raise ValueError('REAL_NON_TEST_VALIDATION_REQUIRED')
    from app.llm.prompt_loader import specialty_system_prompt
    origins=dict(candidate['source_pins']);origins[str(fit_path.resolve())]=sha_file(fit_path)
    for f in run.rglob('*'):
        if f.is_file():origins[str(f.resolve())]=sha_file(f)
    validations=[]
    from app.locdi.native_menu_routing import registry_type
    for patient in patients:
        rr=[r for r in details['rows'] if r['ordinal']==patient['ordinal']];signatures=set();initial=None
        for row in rr:
            folder=run/('Patient%02d'%row['ordinal'])/row['department']/'Episode0001'
            verify_validation_prediction(row,patient,folder)
            for bf in sorted(folder.glob('Request*_before.json')):
                b=read(bf);a=read(Path(str(bf).replace('_before.json','_after.json')));q=b['request']
                if a.get('status')!='VALID' or a.get('HTTP_status')!=200 or q['messages'][0]['content']!=specialty_system_prompt(row['department']) or normalize_model_identifier(a['identity_after']['digest'])!=cfg['model_identifier'] or a['identity_after']['context_length']!=32768:raise ValueError('VALIDATION_ACTUAL_ROLE_MODEL_CONTEXT')
                if (b['contract']!=vp['fingerprint'] or q['model']!=cfg['model'] or
                    q['options']!={'num_ctx':32768,'num_predict':1024,'temperature':0,'seed':42} or
                    normalize_model_identifier(b['identity_before']['digest'])!=cfg['model_identifier']):raise ValueError('VALIDATION_REQUEST_BINDING_OPTIONS')
            body=json.loads(read(folder/'Request0001_before.json')['request']['messages'][1]['content']);initial=body
            if public_menu(body['source_action_metadata'])!=public_menu(patient['binding']['rows']):raise ValueError('VALIDATION_PUBLIC_MENU_CHANGED')
            signatures.add(digest({'observation':body['observation'],'menu':body['source_action_metadata']}))
        if len(signatures)!=1:raise ValueError('VALIDATION_AGENT_INPUT_DIFFERS')
        router=registry_type(public_menu(initial['source_action_metadata']))(vp['registry'],candidate['artifact'])
        decision=router.route(initial['observation'],[a['department'] for a in vp['agents']],mode='adaptive')
        if any(x is None for x in decision.get('utility_estimates',decision.get('estimates',{})).values()):raise ValueError('VALIDATION_SUPPORT_MISSING')
        selected=decision.get('selected_specialty');chosen=next((r for r in rr if r['department']==selected),None)
        validations.append({'ordinal':patient['ordinal'],'decision_status':decision['decision_status'],'selected_specialty':selected,
            'selected_Source_Exact':chosen['correctness_proxy'] if chosen else None,'route':decision})
    evidence={'schema_version':'locdi.routing.native_menu_transfer.v1','feature_version':VERSION,'routing_artifact':candidate['artifact'],
        'model_identifier':cfg['model_identifier'],'prompt_fingerprints':validate_setup(cfg)['prepared']['prompt_fingerprints'],
        'registry':vp['registry'],'origin_files':origins,'collection_plan':str((PR/'collection_index.json').resolve()),
        'actual_validation_generation':read(run/'collection.json')['generation'],'validation_report':validations,
        'validation_status':'PATIENT_DISJOINT_PROSPECTIVE_NATIVE_OUTCOMES_PRIOR_VOI_EXPOSURE',
        'clinical_gate':'UNCHANGED_NOT_ADMITTED','performance_improvement_required_or_claimed':False,**MARKERS}
    if EVIDENCE.exists() and digest(read(EVIDENCE))!=digest(evidence):raise ValueError('ADMISSION_EVIDENCE_TAMPERED')
    if write:write_new(EVIDENCE,evidence)
    if EVIDENCE.exists():_VALIDATED_EVIDENCE_FP=digest(evidence)
    return {'status':'NATIVE_SOURCE_PROXY_TRANSFER_READY' if write else 'ADMISSION_CHECK_PASS_NEW_EXECUTION_PLAN_REQUIRED',
        'artifact_sha256':sha_file(fit_path),'evidence_sha256':sha_file(EVIDENCE) if EVIDENCE.exists() else None,
        'validation_patients':2,'validation_registered_agent_episodes':10,'validation_Source_outcomes':summary['by_agent'],
        'validation_not_untouched_holdout':True,'clinical_accuracy':'NOT_EVALUATED','new_generation':0,'fit_calls':0}

def admitted(evidence,registry,patients,cfg):
    if not EVIDENCE.exists() or digest(evidence)!=digest(read(EVIDENCE)) or digest(evidence)!=_VALIDATED_EVIDENCE_FP:raise ValueError('EXACT_DERIVED_ADMITTED_NATIVE_EVIDENCE_REQUIRED')
    if evidence['schema_version']!='locdi.routing.native_menu_transfer.v1' or evidence['feature_version']!=VERSION or evidence['registry']!=registry or evidence['model_identifier']!=cfg['model_identifier']:raise ValueError('NATIVE_TRANSFER_BINDING')
    if evidence['prompt_fingerprints']!=validate_setup(cfg)['prepared']['prompt_fingerprints']:raise ValueError('PROMPT_BINDING')
    train={r['patient_id'] for r in evidence['routing_artifact']['samples']}
    index=read(evidence['collection_plan']);val={r['decision']['patient_id'] for r in index['candidates'] if r['decision']['split']=='validation'}
    if train&set(patients) or val&(train|set(patients)|set(read(M5)['provenance']['training_patients'])):raise ValueError('SOURCE_SPLIT_OVERLAP')
    return {'artifact':evidence['routing_artifact'],'scope':'NATIVE_SOURCE_PROXY_DEVELOPMENT_ONLY','clinical_research_admission':False}

def main():
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['prepare','status','fit-check','fit','validation-check','validation-execute','validation-analyze','admission-check','admit']);q.add_argument('--approval',type=Path);q.add_argument('--execute',action='store_true');q.add_argument('--approve-fit',action='store_true');q.add_argument('--operator-ref');q.add_argument('--statement');a=q.parse_args()
    if a.mode=='prepare':out=prepare()
    elif a.mode=='status':out={'preparation':read(PUB/'preparation_002.json'),'admission':admission_check()}
    elif a.mode=='fit-check':out=fit_check()
    elif a.mode=='fit':
        if not a.approve_fit or not a.operator_ref or not a.statement:q.error('EXPLICIT_USER_FITTING_APPROVAL_REQUIRED')
        out=fit(a.operator_ref,a.statement)
    elif a.mode=='validation-check':out=validation_check()
    elif a.mode=='validation-analyze':out=validation_analyze()
    elif a.mode=='validation-execute':
        if not a.execute or not a.approval:q.error('NEW_USER_VALIDATION_APPROVAL_REQUIRED')
        with validation_collector_scope() as collector:out=collector.execute('Validation',a.approval)
    else:out=admission_check(write=a.mode=='admit')
    print(json.dumps(out,ensure_ascii=True))
    if out.get('status') in ('BLOCKED','PARTIAL_OR_FAILED'):raise SystemExit(2)

if __name__=='__main__':main()
