"""Original50 read-only readiness, saved-result verification and blocked stages.

No generation/fitting entry point. Clinical and cross-training Gates stay closed.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,native_index,read,sha_file,memory_source_map,validate_csv_claims,eligible,DATA
from app.locdi.original50_augmentation import validate_augmentation,initial_preview,recover_symptom_events
from app.locdi.m5_common_source import bind_cases
from app.locdi.case_adapter import adapt_case
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.routing_data_reuse import BANK,SOURCE,read as bank_read
from app.locdi.development_routing_proxy import MARKERS
from scripts.run_original50_d0_d2 import AUG,M5
from scripts.run_u1_voi_expansion import load as parent_load
from scripts.prepare_u2_source_collection import reference_check
from scripts.run_terminal_only_replay import load as terminal_load
from scripts.run_u2_native_collection import load as native_load

NS='ORIGINAL50_FULL_EVALUATION_READINESS_20261123_v1'
PUB=ROOT/'outputs/research_evaluation'/NS
PR=ROOT/'outputs/private/research_evaluation'/NS
STAGES={'Smoke5':5,'Verify10':10,'Full50':50}


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:raise FileExistsError('EXISTING_ANALYSIS_DIFFERS_NO_OVERWRITE')
    else:atomic_json(path,value)


def snapshot_status():
    """Inspect completed collections, never starts/resumes a model request."""
    statuses={};_,tpr,tp,_,_=terminal_load();trun=tpr/tp['run_id']
    if (trun/'collection.json').exists():
        c=read(trun/'collection.json');rows=c['rows'];details=[]
        if c['plan_fingerprint']!=tp['fingerprint'] or len(rows)!=10 or sum(r['new_generation'] for r in rows)!=c['generation']:raise ValueError('TERMINAL_COLLECTION_LINKAGE')
        for r in rows:
            folder=trun/('Panel%02d'%r['panel_ordinal']);ledger=read(folder/'replay_ledger.json')
            if r['status']!='COMPLETED' or not r['terminal_core_revalidated'] or ledger['new_generation_attempts']!=1 or ledger['new_terminal_posts']!=1:raise ValueError('TERMINAL_COMPLETION_PROOF')
            last=sorted((folder/'Episode0001').glob('Request*_after.json'))[-1];a=read(last)
            if a['status']!='VALID' or a['HTTP_status']!=200 or a['identity_after']['context_length']!=32768:raise ValueError('TERMINAL_NEW_RESPONSE_PROOF')
            x,y=r['old_source_exact'],r['source_exact']
            transition=('RETAINED_CORRECT' if y is True else 'CORRECT_LOST') if x is True else ('BECAME_CORRECT' if y is True else 'NOT_CORRECT_BOTH')
            details.append({'panel_ordinal':r['panel_ordinal'],'before':x,'after':y,'transition':transition,
                            'certainty':r['prediction']['certainty']})
        statuses['terminal']={'status':c['status'],'registered':10,'new_generation':c['generation'],
            'T0_Source_Exact':sum(r['old_source_exact'] is True for r in rows),'T1_Source_Exact':sum(r['source_exact'] is True for r in rows),
            'T0_unresolved':sum(r['old_source_exact'] is None for r in rows),
            'T1_unresolved':sum(r['prediction']['certainty']=='unresolved' for r in rows),
            'transitions':dict(Counter(r['transition'] for r in details)),
            'comparison':'SAVED_STOP_REPLAY_NOT_FULL_NATIVE_B4_ABLATION','rows':details}
    else:statuses['terminal']={'status':'NOT_RUN'}
    for stage in ('Smoke','Full'):
        pub,pr,p=native_load(stage);run=pr/p['run_id'];before=[f for f in run.rglob('Request*_before.json')] if run.exists() else []
        after=[read(f) for f in run.rglob('Request*_after.json')] if run.exists() else []
        item={'run_exists':run.exists(),'approval_exists':(pr/'operator_approval.json').exists(),
              'before_count':len(before),'after_count':len(after),'valid_HTTP200':sum(a.get('status')=='VALID' and a.get('HTTP_status')==200 for a in after)}
        if (run/'collection.json').exists():
            c=read(run/'collection.json')
            if c['plan_fingerprint']!=p['fingerprint'] or sum(r['generation'] for r in c['rows'])!=c['generation']:raise ValueError('NATIVE_COLLECTION_COUNT')
            item.update(status=c['status'],generation=c['generation'],episode_states=dict(Counter(r['status'] for r in c['rows'])))
            if (pub/'saved_results.json').exists():item['saved_analysis']=read(pub/'saved_results.json')
        else:item['status']='RUN_IN_PROGRESS_OR_INTERRUPTED' if run.exists() else 'NOT_STARTED'
        statuses['U2_'+stage]=item
    f=ROOT/'outputs/private/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/u2_native_development_fit_001/U2_native_development_utility.json'
    statuses['U2_artifact']={'exists':f.exists(),'path_private_only':True,'B3_B4_admission':'NOT_INSTALLED'}
    return statuses


def audit():
    if (PUB/'data_readiness.json').exists():raise FileExistsError('AUDIT_EXISTS_USE_CHECK_OR_NEW_NAMESPACE')
    PUB.mkdir(parents=True,exist_ok=True);PR.mkdir(parents=True,exist_ok=True)
    _,_,parent,bp=parent_load();protected=dict(parent['pins'])
    for d in ('u1_stop_terminal_replay_001','u2_native_5agent_smoke_001'):
        root=ROOT/'outputs/private/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1'/d/('run_'+d)
        if (root/'collection.json').exists():protected.update({str(f.resolve()):sha_file(f) for f in root.rglob('*') if f.is_file()})
    write_new(PR/'protection_before.json',protected)
    cases=original_cases();raw_index=native_index();model=read(M5)
    augmentation=[json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(augmentation)!=50 or len({r['case_ordinal'] for r in augmentation})!=50:raise ValueError('D2_COMPLETE50_REQUIRED')
    aug={r['case_ordinal']:r for r in augmentation}
    from app.locdi.literature_augmentation_catalog import evidence_catalog
    catalog=evidence_catalog();refs=[];checks=[];private=[];bindings={};wrappers={}
    old_ordinals=set(read(bp/'case_index.json').values())
    print('Validating 50 Original/D2 Source wrappers...',flush=True)
    for i,c in enumerate(cases,1):
        ref='Original%03d'%i;wrappers[ref]=c;av=c['agent_visible'];row=aug[i]
        adapted=adapt_case(c)
        validate_augmentation(c,row,catalog)
        entry=raw_index[i];path=Path(entry['raw_path']);native_sha=sha_file(path)
        if native_sha!=entry['declared_raw_sha256'] or row['native_source']['file_sha256']!=native_sha:raise ValueError('NATIVE_SOURCE_HASH')
        native=read(path)
        facts,_=recover_symptom_events(c,native,path,native_sha)
        if facts!=row['source_recovered_findings']:raise ValueError('D2_RECOVERED_SOURCE_MISMATCH')
        src=c['provenance']['source_case_audit']['diagnosis_sources']
        labels=c['evaluation']['diagnosis_labels'];start=av['index_encounter_start'];patient=c['patient_id'];visit=av['index_encounter_id']
        if any(s['source_record']['PATIENT']!=patient or s['source_record']['ENCOUNTER']!=visit for s in src):raise ValueError('REFERENCE_PATIENT_VISIT')
        refs.extend((s['source_file'],s['source_row'],s['source_record']) for s in src)
        future=sum(s['source_record']['START'][:10]>start[:10] for s in src)
        memory=[r for v in av['past_visits'] for r in v['records']]
        bad=sum(not eligible(r,av['longitudinal']) for r in memory)
        d0=initial_preview(c,row,catalog,view='D0');d2=initial_preview(c,row,catalog,view='D2',accept_development_proxy=True)
        item={'ordinal':i,'old_development10':i in old_ordinals,'D0_present':True,'D2_present':True,
            'D2_provenance_valid':True,'D2_label_conditioned':True,'D2_release_minute':0,
            'D2_generated_numeric_tests':len(row['literature_informed_synthetic']['synthetic_test_results']),
            'D2_symptom_count':len(row['literature_informed_synthetic']['symptoms']),
            'Reference_in_catalog':bool(labels) and all(x in model['diagnoses'] for x in labels),
            'Reference_future_date_rows':future,'Reference_intraday_clinically_reviewed':False,
            'memory_records':len(memory),'memory_cutoff_violations':bad,'structural_adaptation':'PASS_WITH_RECORDED_WARNINGS',
            'D0_unchanged':d0==av['current_initial_state']}
        checks.append(item);private.append({'summary':item,'case_ref':ref,'patient_id':patient,'visit_id':visit,
            'reference':{'accepted_diagnoses':labels,'accepted_departments':c['evaluation']['accepted_departments_source_policy']},
            'reference_source':src,'D0_state':d0,'D2_state':d2,'native_source_sha256':native_sha,
            'case_fingerprint':digest(c),'augmentation_fingerprint':row['row_fingerprint']})
    print('Verifying grouped CSV claims and 72,590 Memory records...',flush=True)
    reference_proof=validate_csv_claims(refs);memory_map,memory_proof=memory_source_map(cases)
    bindings=bind_cases(wrappers,model)
    for i,r in enumerate(checks,1):
        b=bindings['Original%03d'%i];r.update(common_actions=len(b['rows']),missing_actions=len(b['missing_actions']),
            objective_tests=sum(x['kind']=='Test' for x in b['rows']),patient_report_Ask=sum(x['kind']=='Ask' for x in b['rows']),
            Source_values_units_verified=True,Source_cutoff_minutes=45,
            input_Source_ready=bool(b['rows']) and r['D0_unchanged'] and r['D2_provenance_valid'] and r['memory_cutoff_violations']==0)
    write_new(PR/'case_audit_private.json',private);write_new(PR/'bindings50.json',bindings)
    write_new(PR/'reference_source_validation.json',reference_proof);write_new(PR/'memory_source_validation.json',memory_proof)
    write_new(PR/'config.json',read(bp/'llm_config.json'))
    # Exhaustive Bank patient/visit independence, without changing splits.
    bank=bank_read(BANK);sources={r['case_id']:r for r in bank_read(SOURCE)};m5=set(model['provenance']['training_patients'])
    original_ids={c['patient_id'] for c in cases};bank_ids={b['training_patient_id'] for b in bank}
    old=read(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json')
    reserved={pid for values in old['split_manifest']['patients'].values() for pid in values}
    eligibility=Counter();available=[]
    for b in bank:
        pid=b['training_patient_id'];label,status=reference_check(b,sources[b['training_case_id']],model['diagnoses'])
        if label is None:eligibility[status]+=1;continue
        eligibility['REFERENCE_DATE_PROXY_ELIGIBLE']+=1
        if pid in original_ids:eligibility['ORIGINAL50_PATIENT_OVERLAP']+=1;continue
        if pid in reserved:eligibility['PRIOR_ROUTING_RESERVATION']+=1;continue
        if pid in m5:eligibility['M5_TRAIN_PATIENT_OVERLAP']+=1;continue
        available.append(b)
    _,upr,up=native_load('Full');urows=read(upr/'patients.json');utrain={r['case']['patient_id'] for r in urows if r['split']=='train'};uval={r['case']['patient_id'] for r in urows if r['split']=='validation'}
    split={'Bank_visits':len(bank),'Bank_patients':len(bank_ids),'M5_train_patients':len(m5),
        'M5_training_visits':model['provenance']['training_visits'],'Bank_M5_patient_intersection':len(bank_ids&m5),
        'Bank_outside_M5':len(bank_ids-m5),'Original50_M5_overlap':len(original_ids&m5),
        'Original50_Bank_overlap':len(original_ids&bank_ids),'U2_train_validation_overlap':len(utrain&uval),
        'U2_Original50_overlap':len((utrain|uval)&original_ids),'U2_validation_M5_overlap':len(uval&m5),
        'earlier_routing_reserved_patients':len(reserved),'new_validation_patients':len({r['training_patient_id'] for r in available}),
        'new_validation_visits':len(available),'reference_and_exclusion_counts':dict(eligibility),
        'patient_level_exclusion_covers_other_encounters':True,'new_validation_plan_created':False,
        'blocker_codes':['NO_TRAINING_BANK_VALIDATION_PATIENT_OUTSIDE_M5','ROUTING_VALIDATION_VOI_TRAIN_OVERLAP']}
    write_new(PUB/'validation_independence.json',split)
    # Existing 40 were all in the same authored augmentation, not untouched Holdout.
    report={'Original50':50,'unique_patients':50,'original_dataset_sha256':sha_file(DATA),
        'D0_present':50,'D2_present':50,'Source_runtime_ready':sum(r['input_Source_ready'] for r in checks),
        'D2_validated':50,'new_augmentation_generated':0,'label_conditioned':True,
        'old_development_patients':len(old_ordinals),'other_patients':50-len(old_ordinals),
        'other40_independent_holdout':False,'reference_future_date_cases':sum(r['Reference_future_date_rows']>0 for r in checks),
        'reference_same_or_earlier_date_cases':sum(r['Reference_future_date_rows']==0 for r in checks),
        'clinical_reference_reviewed':0,'Memory_total':sum(r['memory_records'] for r in checks),
        'Memory_cutoff_violations':sum(r['memory_cutoff_violations'] for r in checks),
        'Memory_exact_source_verified':memory_proof['verified'],'rows':checks,**MARKERS}
    write_new(PUB/'data_readiness.json',report)
    # This selection uses neither reference diagnoses nor stored model correctness.
    order=sorted(range(1,51),key=lambda n:digest(['original50-stage-v1',42,cases[n-1]['patient_id']]))
    pins={str(Path(__file__).resolve()):sha_file(__file__),str(DATA.resolve()):sha_file(DATA),str(M5.resolve()):sha_file(M5),
          str((PR/'bindings50.json').resolve()):sha_file(PR/'bindings50.json'),str((PR/'case_audit_private.json').resolve()):sha_file(PR/'case_audit_private.json')}
    for stage,n in STAGES.items():
        name='original50_d2_'+stage.lower()+'_001'
        tasks=[{'original_ordinal':i,'case_ref':'Original%03d'%i,'condition':cond,'old_development10':i in old_ordinals}
               for i in order[:n] for cond in ('B0','B1','B2','B3','B4')]
        write_new(PR/name/'tasks.json',tasks)
        protocol={'plan_id':name,'stage':stage,'status':'STAGED_CONTRACT_BLOCKED_NOT_EXECUTABLE',
            'patients':n,'episodes':n*5,'dataset_view':'D2','D0_separate_contract_required':True,
            'generation_per_episode_including_terminal':12,'B4_terminal_requests_max':n,
            'generation_total':n*5*12,'terminal_calls_are_included_not_unbudgeted':True,
            'terminal_budget_exhaustion':'FAIL_NO_EXTRA_REQUEST_NO_FORCED_UNRESOLVED',
            'retry':0,'repair':0,'warmup':0,'context':32768,'output_tokens':1024,'timeout_seconds':120,
            'temperature':0,'seed':42,'no_resume':True,'wall_seconds':1800 if n==5 else 3600 if n==10 else 14400,
            'process_bytes':4*1024**3,'output_bytes':4*1024**3,'selection':'SEEDED_PATIENT_HASH_NO_REFERENCE_OR_ACCURACY',
            'actual_utility':'U2_REQUIRED_NOT_AVAILABLE','B3_B4_same_utility_required':True,'approval_created':False,
            'pins':pins,'blocker_codes':['U2_NATIVE_FULL_AND_FIT_NOT_VERIFIED','U2_NATIVE_ARTIFACT_ADMISSION_NOT_INSTALLED',
                'NO_INDEPENDENT_VALIDATION_OUTSIDE_M5','NATIVE_B4_T_NEW_EXECUTION_CONTRACT_NOT_READY'],**MARKERS}
        protocol['fingerprint']=digest(protocol);write_new(PUB/name/'plan.json',protocol)
    write_new(PUB/'saved_execution_status.json',snapshot_status())
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}));print(json.dumps(split))


def check(stage):
    p=read(PUB/('original50_d2_'+stage.lower()+'_001')/'plan.json')
    if p['fingerprint']!=digest({k:v for k,v in p.items() if k!='fingerprint'}):raise ValueError('STAGED_CONTRACT_FINGERPRINT')
    for f,h in p['pins'].items():
        if sha_file(f)!=h:raise ValueError('STAGED_CONTRACT_PIN_CHANGED')
    return {'status':'BLOCKED','plan_id':p['plan_id'],'patients':p['patients'],'episodes':p['episodes'],
            'generation_cap_proposed':p['generation_total'],'blocker_codes':p['blocker_codes'],
            'execute_entry_point_created':False,'LLM_generation':0,'fitting':0}


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['audit','check','status']);q.add_argument('--stage',choices=STAGES,default='Full50');a=q.parse_args()
    if a.mode=='audit':audit()
    else:
        out=check(a.stage) if a.mode=='check' else snapshot_status();print(json.dumps(out))
        if out.get('status')=='BLOCKED':raise SystemExit(2)
