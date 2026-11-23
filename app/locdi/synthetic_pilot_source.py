"""Authored synthetic ZIP: private Source capability, never Synthea provenance."""
from collections import Counter
from copy import deepcopy
import json,math,zipfile,hashlib
from pathlib import Path
from app.locdi.llm_setup import digest
from app.core.clinical_observation import prepare_encounter,doctor_observation,encounter_evidence
from app.core.hospital_parallel_scheduler import HospitalParallelScheduler
from app.core.hospital_clinical_policy import HospitalClinicalPolicy
from app.core.hospital_llm_policy import HospitalLLMPolicy

VERSION='AUTHORED_SYNTHETIC_PILOT_SOURCE.v1'
MEMBERS={'README.md','manifest.json','training_cases_public.jsonl','reference_private.jsonl','provenance.jsonl'}
DEPARTMENTS=('cardiology','endocrinology','nephrology','neurology','pulmonology')
MEASURES={'heart_rate':('8867-4','/min'),'systolic_bp':('8480-6','mm[Hg]'),
 'diastolic_bp':('8462-4','mm[Hg]'),'respiratory_rate':('9279-1','/min'),
 'hba1c':('4548-4','%'),'creatinine':('38483-4','mg/dL'),'egfr':('33914-3','mL/min/{1.73_m2}')}
MISSING_CONFIRMATION='ILLUSTRATIVE_LABEL_WITH_MISSING_CONFIRMATORY_EVIDENCE'

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def strict(text):
    from app.locdi.current_ablation_llm_backend import _strict_object
    return _strict_object(text)

def load_zip(path,catalog):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=5 or set(z.namelist())!=MEMBERS:raise ValueError('ZIP_MEMBER_SCOPE')
        if any(i.file_size>2**20 or i.flag_bits&1 for i in z.infolist()):raise ValueError('ZIP_SIZE_OR_ENCRYPTION')
        raw={n:z.read(n) for n in MEMBERS}
    manifest=strict(raw['manifest.json'].decode('utf-8-sig'))
    tables={n:[strict(s) for s in raw[n].decode('utf-8-sig').splitlines() if s.strip()]
            for n in ('training_cases_public.jsonl','reference_private.jsonl','provenance.jsonl')}
    cases=tables['training_cases_public.jsonl'];refs=tables['reference_private.jsonl'];prov=tables['provenance.jsonl']
    if any(len(v)!=15 for v in tables.values()) or manifest.get('case_count')!=15:raise ValueError('COUNT_NOT_15')
    ids=[c['case_id'] for c in cases]
    if len(set(ids))!=15 or any(len({r['case_id'] for r in t})!=15 or {r['case_id'] for r in t}!=set(ids) for t in (refs,prov)):raise ValueError('IDENTITY_ALIGNMENT')
    refs={r['case_id']:r for r in refs};prov={r['case_id']:r for r in prov};seen=[]
    for c in cases:
        s=c['static_case'];r=refs[c['case_id']]
        if r['reference_diagnosis'] not in catalog or r['reference_specialty'] not in DEPARTMENTS:raise ValueError('CATALOG_OR_DEPARTMENT')
        if c.get('not_synthea') is not True or c['dataset_version']!=manifest['dataset_id']:raise ValueError('NOT_SYNTHETIC_CONTRACT')
        if s['arrival_minute']!=0 or s['initial_vitals_visible'] is not False or s['initial_objective_findings']!=[]:raise ValueError('INITIAL_VISIBILITY_CHANGED')
        if not s['chief_complaint'] or not isinstance(s['current_symptoms'],list) or not isinstance(s['history_visible_at_start'],list):raise ValueError('INITIAL_FIELDS')
        if not 0<s['age']<130 or s['sex'] not in ('male','female'):raise ValueError('DEMOGRAPHICS')
        if c['access_policy']!={'reference_and_teacher_rationale_visible_to_agent':False,'measurement_values_visible_only_after_action':True,'knowledge_base_links_visible_to_agent':False}:raise ValueError('ACCESS_POLICY')
        actions=c['test_timeline']
        if len(actions)!=7 or {x['action_key'] for x in actions}!=set(MEASURES):raise ValueError('ACTION_MENU')
        for x in actions:
            if not x['event_id'].startswith(c['case_id']+'-OBS-'):raise ValueError('CROSS_CASE_SOURCE_EVENT')
            if (x['code'],x['unit'])!=MEASURES[x['action_key']] or x['kind']!='Test' or x['synthetic_value'] is not True:raise ValueError('MEASUREMENT_BINDING')
            if type(x['value']) not in (int,float) or not math.isfinite(x['value']):raise ValueError('NUMERIC_VALUE')
            if any(type(x[k]) is not int for k in ('recorded_minute','available_minute')) or not 0<=x['recorded_minute']<=x['available_minute']:raise ValueError('SOURCE_TIME')
            if x['release_rule']!='ON_REQUEST_AND_AFTER_AVAILABLE':raise ValueError('RELEASE_POLICY')
            seen.append(x['event_id'])
        temporal=c['temporal_profile']
        if temporal['time_unit']!='minute':raise ValueError('CLOCK_UNIT')
        for e in temporal['events']:
            if type(e['minute']) is not int or e['minute']<0 or e['kind'] not in ('arrival','history_detail'):raise ValueError('HISTORY_CLOCK')
        if prov[c['case_id']]['source_row_provenance'] is not None or prov[c['case_id']]['clinical_reviewer'] is not None:raise ValueError('PROVENANCE_CONTRACT_CHANGED')
    if len(set(seen))!=105:raise ValueError('DUPLICATE_SOURCE_EVENT')
    return {'version':VERSION,'zip_sha256':file_sha(path),'member_hashes':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
            'manifest':manifest,'cases':cases,'references':refs,'provenance':prov}

def native_case(case,model):
    """Source-free-of-Reference engine projection; clock anchor is technical only."""
    s=case['static_case'];targets={}
    for key,aid in model['provenance']['action_key_mapping'].items():
        kind,_,code,unit,_=json.loads(key)
        if kind=='Test':targets[code,unit]=next(a['target'] for a in model['actions'] if a['id']==aid)
    events=[];binding=[]
    for x in case['test_timeline']:
        if (x['code'],x['unit']) not in targets:raise ValueError('NO_COMMON_ACTION_MAPPING')
        target=targets[x['code'],x['unit']]
        evidence=json.dumps({'value':x['value'],'units':x['unit'],'recorded_minute':x['recorded_minute']},sort_keys=True,separators=(',',':'))
        events.append({'event_id':x['event_id'],'action':'Test','target':target,'available_minute':x['available_minute'],'evidence':evidence})
        binding.append({**deepcopy(x),'target':target,'evidence':evidence,'source_fingerprint':digest(x)})
    for i,e in enumerate(case['temporal_profile']['events']):
        if e['kind']=='history_detail':events.append({'event_id':case['case_id']+'-H'+str(i),'action':'Ask',
            'target':'Read authored history detail %d?'%i,'available_minute':e['minute'],'evidence':e['text']})
    clock='2000-01-01T00:00:00Z';cid='synthetic:'+case['case_id'];visit=cid+':visit0'
    engine={'case_id':cid,'index_encounter_id':visit,'index_encounter_start':clock,
        'longitudinal':{'patient_id':cid,'visit_id':visit,'started_at':clock},'past_visits':[],
        'current_initial_state':{'doctor_visible_info':{'chief_complaint':s['chief_complaint'],
           'demographics':{'age':s['age'],'sex':s['sex']},'initial_vitals':{}},
           'patient_visible_info':{'chief_complaint':s['chief_complaint'],
              'associated_symptoms':s['current_symptoms']+[s['patient_statement']],'history':s['history_visible_at_start']}},
        'current_encounter_events':events,'resource_profile':{'test_turnaround':{},'test_occupancy':{},'test_resources':{}}}
    for r in binding:
        # Declared synthetic delay, not a measured clinical TAT. Guarantees receipt >= absolute availability.
        engine['resource_profile']['test_turnaround'][r['target']]=max(1,r['available_minute'])
        engine['resource_profile']['test_occupancy'][r['target']]=1
        engine['resource_profile']['test_resources'][r['target']]='lab_slot'
    return prepare_encounter(engine),binding

class AcquisitionPolicy(HospitalClinicalPolicy):
    """Fixed source-only test acquisition; never diagnoses or reads References."""
    def evaluate_er_case(self,runtime,*,doctor_id):
        remaining=[t for t in runtime.case_data['available_tests'] if t not in runtime.completed_tests]
        return {'action':'order_test' if remaining else 'er_finalize','target':remaining[0] if remaining else 'Unresolved',
          'target_department':'general_medicine','assessment_summary':'Unresolved','recommended_tests':remaining,
          'department_candidates':[],'alternatives':[],'certainty':'unresolved','rationale':'Fixed source acquisition, no diagnostic model.'}
    build_objective_finding=staticmethod(HospitalLLMPolicy.build_objective_finding)
    @staticmethod
    def build_test_result_summary(runtime,test_name):
        result=HospitalLLMPolicy.build_test_result_summary(runtime,test_name);result.update(synthetic=True,source=VERSION)
        if not result['metrics'] and not HospitalLLMPolicy.build_objective_finding(runtime,test_name):raise ValueError('MISSING_SOURCE_NO_FALLBACK')
        return result

def snapshot(case,model,*,view='full',stage='initial'):
    if view not in ('full','source_compatible') or stage not in ('initial','all_tests'):raise ValueError('VIEW_OR_STAGE')
    engine,binding=native_case(case,model);scheduler=HospitalParallelScheduler();scheduler.clinical_policy=AcquisitionPolicy()
    state=scheduler.initialize([engine],run_id='AUTHORED_SOURCE_ACQUISITION');rt=next(iter(state.cases.values()))
    if stage=='all_tests':
        scheduler.run_until(state,120)
        if set(rt.completed_tests)!=set(engine['available_tests']) or not rt.is_completed:raise ValueError('NATIVE_ACQUISITION_INCOMPLETE')
    obs=doctor_observation(rt)
    # Source says these histories are initially visible. This is not an executed Ask.
    if view=='full':
        obs['history']=[{'question':'Initially disclosed history','answer':h,'answer_status':'provided'} for h in case['static_case']['history_visible_at_start']]
    else:
        obs['chief_complaint']='';obs['current_symptoms']=[];obs['history']=[]
    # No case/visit IDs, raw temporal Source, author metadata, Reference or URLs in shared payload.
    clean={k:deepcopy(obs[k]) for k in ('chief_complaint','demographics','elapsed_minutes','vital_signs','current_symptoms','history','objective_findings')}
    clean['longitudinal_memory']={'mode':'none','records':[]}
    measurements=[]
    for i,text in enumerate(clean['objective_findings']):
        r=next((r for r in binding if text==r['target']+': '+r['evidence']),None)
        if r is None or r['target'] not in rt.completed_tests or rt.current_time_min<r['available_minute']:raise ValueError('UNRECEIVED_RESULT')
        measurements.append({'evidence_index':i,**{k:r[k] for k in ('description','code','value','unit')}})
    return {'observation':clean,'released_measurements':measurements}, {'policy':VERSION,'stage':stage,'view':view,
        'source_case_fingerprint':digest(case),'binding':binding,'events':state.event_log,
        'initial_history_is_disclosed_not_asked':True,'native_unresolved_acquisition_only':stage=='all_tests',
        'actual_LLM_generation':0,'clock_anchor_is_not_source_date':True}

def score_response(response,reference,catalog):
    from jsonschema import Draft202012Validator
    from app.locdi.routing_collection_preparation import response_schema
    Draft202012Validator(response_schema(catalog)).validate(response)
    if (response['certainty']=='unresolved')!=(response['diagnosis']=='Unresolved'):raise ValueError('CERTAINTY_DIAGNOSIS_CONTRADICTION')
    if response['certainty']=='unresolved':return None,'UNRESOLVED'
    if reference.get('reference_status')!='NOT_INDEPENDENTLY_CLINICALLY_VERIFIED':return None,'REFERENCE_NOT_ELIGIBLE'
    if reference.get('reference_diagnosis') not in catalog:return None,'REFERENCE_NOT_ELIGIBLE'
    return response['diagnosis']==reference['reference_diagnosis'],'AUTHOR_REFERENCE_EXACT_PROXY'
