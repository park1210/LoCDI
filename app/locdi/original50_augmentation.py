"""Separate Source recovery and label-conditioned authored development previews.

No Core runtime registration, Source grant, likelihood, scoring or model client.
"""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib,math
from app.core.longitudinal_memory import timestamp
from app.locdi.llm_setup import digest

VERSION='original50_augmentation.v1'
SCOPE='LABEL_CONDITIONED_SYNTHETIC_DEVELOPMENT'
FORBIDDEN=('alzheimer','ischemic heart disease','atrial fibrillation','metabolic syndrome','osteoporosis',
    'hypertriglyceridemia','diabetes','hyperlipidemia','chronic kidney disease','emphysema','asthma',
    'chronic obstructive bronchitis','cardiology','endocrinology','nephrology','neurology','pulmonology')

def pointer_part(s):return str(s).replace('~','~0').replace('/','~1')
def ms_iso(ms):
    # Windows fromtimestamp() can reject valid negative Unix times. Arithmetic
    # preserves the actual Source epoch, including pre-1970 historical events.
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=ms)).isoformat(timespec='milliseconds')
def directly_discloses_label(text):return any(w in str(text).casefold() for w in FORBIDDEN)

def recover_symptom_events(case,raw,raw_path,raw_sha):
    """Recover timed positive events only with a unique actual encounter interval.

    Visit linkage is explicitly derived, not a nonexistent recorded visit ID.
    Final resolved/lastUpdated states are never replayed backwards.
    """
    av=case['agent_visible'];identity=av['longitudinal'];start=timestamp(identity['started_at']).timestamp()*1000
    cutoff=start+45*60000
    if raw.get('attributes',{}).get('id')!=identity['patient_id']:raise ValueError('NATIVE_PATIENT_MISMATCH')
    encounters=raw['record']['encounters'];good=[];reject=[]
    for term,entry in raw.get('symptoms',{}).items():
        for module,s in entry.get('sources',{}).items():
            for key,event in s.get('timeInfos',{}).items():
                ptr='/symptoms/'+pointer_part(term)+'/sources/'+pointer_part(module)+'/timeInfos/'+pointer_part(key)
                t=event.get('time');value=event.get('value');reason=None
                if (type(t) not in (int,float) or type(value) not in (int,float) or
                    not math.isfinite(t) or not math.isfinite(value)):reason='INVALID_EVENT_TIME_OR_VALUE'
                elif str(int(t))!=key:reason='EVENT_KEY_TIME_MISMATCH'
                elif t>cutoff:reason='FUTURE_SYMPTOM_EVENT'
                elif value<=0:reason='NONPOSITIVE_ENGINE_VALUE_NOT_A_PATIENT_DENIAL'
                elif directly_discloses_label(term):reason='DIAGNOSIS_OR_SPECIALTY_TERM_NOT_A_PUBLIC_SYMPTOM'
                links=[] if reason else [e for e in encounters if isinstance(e.get('start'),(int,float)) and
                    isinstance(e.get('stop'),(int,float)) and e['start']<=t<=e['stop']]
                if not reason and len(links)!=1:reason='NO_UNIQUE_SOURCE_ENCOUNTER_INTERVAL'
                if not reason and (t<start and links[0]['uuid']==identity['visit_id']):reason='CURRENT_ENCOUNTER_EVENT_BEFORE_INDEX_REQUIRES_REVIEW'
                if not reason and (t>=start and links[0]['uuid']!=identity['visit_id']):reason='CURRENT_TIME_DIFFERENT_ENCOUNTER'
                source={'file':str(raw_path),'file_sha256':raw_sha,'json_pointer':ptr,'event_fingerprint':digest(event)}
                if reason:
                    reject.append({'candidate_type':'SOURCE_RECOVERY','reason':reason,'source':source});continue
                scope='CURRENT' if links[0]['uuid']==identity['visit_id'] else 'HISTORICAL'
                fact={'finding_id':'SRC_SYM_'+digest((identity['patient_id'],ptr))[:24],
                    'origin':'SOURCE_RECOVERED','kind':'SOURCE_ENGINE_SYMPTOM_EVENT','patient_id':identity['patient_id'],
                    'visit_id':links[0]['uuid'],'scope':scope,'symptom_term':term,'engine_value':value,
                    'engine_value_interpretation':'NOT_A_VALIDATED_CLINICAL_SEVERITY_SCALE',
                    'recorded_at':ms_iso(t),'recorded_minute_from_index':(t-start)/60000,
                    'actual_clinical_available_at':None,'clinical_reviewed':False,'execution_release_approved':False,
                    'availability_basis':'RECORDED_ENGINE_TIME_PROXY_REQUIRES_SEPARATE_RELEASE_CONTRACT',
                    'source_patient_link':'NATIVE_ATTRIBUTES_ID_EXACT','visit_link_basis':'DERIVED_UNIQUE_NATIVE_ENCOUNTER_INTERVAL',
                    'source_time_verified':True,'source':source}
                good.append(fact)
    good.sort(key=lambda r:(r['recorded_at'],r['finding_id']))
    return good,reject

def scenario(case_ordinal,card,catalog,policy,*,used_chiefs=None):
    """Authored narratives only; no numeric tests, genetics or past history invented."""
    number=int(hashlib.sha256((str(policy['seed'])+'|'+str(case_ordinal)).encode()).hexdigest(),16)
    ids=list(card['symptom_feature_ids']);mode=number%4
    if not ids:mode=0
    elif "Alzheimer" in card['disease'] and mode==0:mode=1
    elif card['disease']=='Chronic kidney disease stage 4 (disorder)' and mode==0:mode=1
    count=0 if mode==0 else 1 if mode in (1,2) else min(2,len(ids))
    chosen=[]
    if count:
        start=(number//7)%len(ids)
        for offset in range(count):chosen.append(deepcopy(catalog['features'][ids[(start+offset)%len(ids)]]))
    # Do not imply that absent history is a verified absence of disease.
    if not chosen:
        core=['No new symptom is reported.','No specific current complaint is described.','Current symptoms are not reported.',
              'No particular new symptom is mentioned.','No symptom concern is supplied today.'][number%5]
    else:core=' '.join(f['text'] for f in chosen)
    frames=['Reason for this visit: {}','The concern to review today is: {}','At this consultation, the reported concern is: {}',
            'The person would like an assessment of the following: {}','The presenting account is: {}',
            'The visit is for a review of this concern: {}','The account supplied for this visit is: {}',
            'The current discussion begins with this report: {}','The main issue described at this visit is: {}',
            'The person gives this account for review: {}']
    used_chiefs=used_chiefs if used_chiefs is not None else set()
    available=[f.format(core) for f in frames if f.format(core) not in used_chiefs]
    if not available:raise ValueError('REPEATED_NARRATIVE_REQUIRES_AUTHOR_REVIEW')
    chief=available[(number//11)%len(available)];used_chiefs.add(chief)
    if directly_discloses_label(chief):raise ValueError('AUTHORED_LABEL_DISCLOSURE')
    sourceids=sorted({s for f in chosen for s in f['sources']} or set(card['sources']))
    for i,f in enumerate(chosen):f.update(origin='LITERATURE_INFORMED_SYNTHETIC',finding_id='SYN_%02d_%02d'%(case_ordinal,i+1),
        available_minute=0,time_basis='AUTHORED_SCENARIO_TIME_NOT_ORIGINAL_SOURCE',clinical_reviewed=False)
    return {'chief_complaint':{'text':chief,'origin':'LITERATURE_INFORMED_SYNTHETIC','sources':sourceids,
            'available_minute':0,'clinical_reviewed':False},'symptoms':chosen,
        'symptom_history_mode':'NO_CURRENT_SYMPTOM_REPORTED_SYNTHETIC' if not chosen else 'SPARSE_OR_NONSPECIFIC_SYNTHETIC' if count==1 else 'SELECTED_LITERATURE_SYMPTOMS_SYNTHETIC',
        'candidate_tests':[],'synthetic_test_results':[],
        'not_generated':['NUMERIC_TEST_RESULTS','FRACTURE_HISTORY','FAMILY_HISTORY','GENETIC_VARIANTS','AGE_OF_ONSET','CONFIRMED_DIAGNOSIS'],
        'clinical_reviewed':False}

def validate_augmentation(case,row,catalog,*,verify_source_files=False):
    if row.get('version')!=VERSION or row.get('original_case_fingerprint')!=digest(case):raise ValueError('ORIGINAL_CASE_FINGERPRINT_MISMATCH')
    if row.get('clinical_reviewed') is not False or row.get('clinical_diagnosability')!='UNREVIEWED':raise ValueError('UNSUPPORTED_CLINICAL_APPROVAL')
    if row.get('evaluation_scope')!=SCOPE:raise ValueError('LABEL_CONDITIONED_SCOPE_REQUIRED')
    if row.get('literature_fingerprint')!=digest(catalog):raise ValueError('LITERATURE_FINGERPRINT')
    if row.get('row_fingerprint')!=digest({k:v for k,v in row.items() if k!='row_fingerprint'}):raise ValueError('AUGMENTATION_ROW_FINGERPRINT')
    originals=[e for e in case['environment']['source_event_store'] if e['event_type']=='observations']
    if digest(row['original_findings'])!=digest(originals):raise ValueError('ORIGINAL_OBSERVATION_MODIFIED')
    generated=row['literature_informed_synthetic']
    if generated['synthetic_test_results']:raise ValueError('NUMERIC_OR_OTHER_SYNTHETIC_RESULTS_NOT_AUTHORED_IN_V1')
    if directly_discloses_label(generated['chief_complaint']['text']):raise ValueError('DIRECT_DIAGNOSIS_OR_SPECIALTY_DISCLOSURE')
    for item in [generated['chief_complaint'],*generated['symptoms']]:
        if item.get('origin')!='LITERATURE_INFORMED_SYNTHETIC' or item.get('clinical_reviewed') is not False:raise ValueError('SYNTHETIC_ORIGIN_OR_REVIEW')
        if item.get('available_minute')!=0:raise ValueError('AUTHORED_TIME_V1_CONTRACT')
        if not item.get('sources') or any(s not in catalog['sources'] or catalog['sources'][s]['verification']!='PRIMARY_PAGE_CONTENT_OBSERVED' for s in item['sources']):raise ValueError('UNVERIFIED_LITERATURE')
        if 'feature_id' in item and (item['feature_id'] not in catalog['features'] or item['text']!=catalog['features'][item['feature_id']]['text'] or item['sources']!=catalog['features'][item['feature_id']]['sources']):raise ValueError('UNSUPPORTED_SYNTHETIC_FEATURE')
        if directly_discloses_label(item['text']):raise ValueError('DIRECT_DIAGNOSIS_OR_SPECIALTY_DISCLOSURE')
    identity=case['agent_visible']['longitudinal'];cutoff=timestamp(identity['started_at']).timestamp()*1000+45*60000
    for f in row['source_recovered_findings']:
        if f['origin']!='SOURCE_RECOVERED' or f['patient_id']!=identity['patient_id'] or not f['source_time_verified'] or not f['source'].get('event_fingerprint'):raise ValueError('RECOVERED_SOURCE_CONTRACT')
        if timestamp(f['recorded_at']).timestamp()*1000>cutoff:raise ValueError('RECOVERED_FUTURE_EVENT')
        if f['execution_release_approved'] is not False:raise ValueError('NO_CLINICAL_SOURCE_GRANT_IN_AUGMENTATION')
        if directly_discloses_label(f['symptom_term']):raise ValueError('RECOVERED_DIAGNOSIS_NAME_DISCLOSURE')
    if verify_source_files:
        from app.locdi.original_source_inventory import native_index,read,sha_file
        entry=native_index()[row['case_ordinal']];path=entry['raw_path'];expected=entry['declared_raw_sha256']
        if row['native_source']['file']!=str(path) or row['native_source']['file_sha256']!=expected or sha_file(path)!=expected:raise ValueError('NATIVE_SOURCE_PIN_MISMATCH')
        raw=read(path);facts,_=recover_symptom_events(case,raw,path,expected)
        if facts!=row['source_recovered_findings']:raise ValueError('RECOVERED_FACT_SOURCE_REPLAY_MISMATCH')
    return {'structural_status':'PASS','clinical_diagnosability':'UNREVIEWED','clinical_reviewed':False}

def initial_preview(case,row,catalog,*,view='D0',accept_development_proxy=False):
    """Same initial-state schema; NOT an admitted runtime, evidence or M5 view.

    Historical symptoms are a separate audit sidecar. They are never turned into
    current symptoms, Memory IDs or patient Evidence here. Literature test menus
    are never added to the original environment.
    """
    if view not in ('D0','D1','D2'):raise ValueError('VIEW')
    state=deepcopy(case['agent_visible']['current_initial_state'])
    if view=='D0':return state
    if accept_development_proxy is not True:raise ValueError('EXPLICIT_NEW_DEVELOPMENT_PREVIEW_CONTRACT_REQUIRED')
    validate_augmentation(case,row,catalog,verify_source_files=view=='D1')
    if view=='D1':
        symptoms=[f['symptom_term'] for f in row['source_recovered_findings'] if f['scope']=='CURRENT' and f['recorded_minute_from_index']<=0]
        state['patient_visible_info']['associated_symptoms']=list(dict.fromkeys(state['patient_visible_info'].get('associated_symptoms',[])+symptoms))
    else:
        if state['doctor_visible_info'].get('chief_complaint'):raise ValueError('EXISTING_SOURCE_CHIEF_COMPLAINT_MUST_NOT_BE_REPLACED')
        synthetic=row['literature_informed_synthetic'];state['doctor_visible_info']['chief_complaint']=synthetic['chief_complaint']['text']
        state['patient_visible_info']['associated_symptoms']=list(dict.fromkeys(
            state['patient_visible_info'].get('associated_symptoms',[])+[s['text'] for s in synthetic['symptoms']]))
    # No hidden Source/Reference, candidate tests, authoring prompts or provenance.
    return state
