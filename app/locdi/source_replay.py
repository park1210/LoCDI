"""D3 source-only development dataset. Clinical review is never manufactured.

Canonical identities/evidence stay in a separate private dataset. The legacy
v34.1 wrapper is a deliberately lossy, explicitly authorized proxy projection.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import csv
import json
import math
import os
from pathlib import Path
import tempfile

from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_suitability import COHORT, load_memberships, sha, millis, reference_timing
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.core.longitudinal_memory import PatientMemoryStore

VERSION='simulation_source_replay_v1'
PRIVATE=ROOT/'outputs/private/simulation_datasets'/VERSION
PUBLIC=ROOT/'datasets/versions'/VERSION
STRICT='STRICT_SOURCE'
PROXY='EXPLICIT_DEVELOPMENT_PROXY'
ASSUMPTION='D3_RECORDED_INTERVAL_UPPER_BOUND_ACTION_GATED_V1'
PAIN_REPORTS={'72514-3','38208-5','75893-8'}
PRIORITY={'emergency':0,'urgentcare':1,'outpatient':2}
# This policy is fixed before examining eligibility or outcomes. No quotas.
POLICY={'version':'D3_SELECTION_V1','purpose':'DEVELOPMENT_ONLY',
    'patient_scope':'D2_368_OUTSIDE_KNOWN_EXPOSURE_CONTEXT_MEMORY',
    'priority_classes':['emergency','urgentcare','outpatient'],
    'wellness':'COMPARISON_ONLY','per_patient':'ONE_EARLIEST_ELIGIBLE_ENCOUNTER_IN_HIGHEST_PRIORITY_CLASS',
    'required':['verified_identity_hash','prior_encounter','past_recorded_observation',
                'nonempty_current_observation','source_condition','allowlisted_source_action'],
    'chief_complaint_required':False,'clinical_gold_required_for_development':False,
    'first_value_per_binding':True,'binding':['category','code','description','units','type'],
    'ask_codes':sorted(PAIN_REPORTS),'test_categories':['vital-signs','laboratory','exam'],
    'max_release_minute':110,'simulation_minute_limit':120,
    'initial_clinical_values':'NO_UNKNOWN_AVAILABILITY_VALUES',
    'proxy_static_demographics':'EXPLICIT_SOURCE_METADATA_ASSUMPTION',
    'memory_proxy':'RECORDED_INTERVAL_UPPER_BOUND_NOT_OBSERVED_AVAILABILITY',
    'consult':'DISABLED','assigned_split':None,'clinical_evaluation_ready':False}


def iso(ms):
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=ms)).isoformat(timespec='milliseconds').replace('+00:00','Z')


def source_ref(table,number,row,hashes):
    name='csv/'+table
    return {'source_file':'experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1/'+name,
            'source_row':number,'source_record':deepcopy(row),'file_sha256':hashes[name],
            'record_fingerprint':digest(row)}


def read_sources():
    prior=json.loads((ROOT/'outputs/reports/D2_suitability_v1.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'outputs/private/dataset_audit/D2_suitability_v1/manifests/split_relations.json').read_text(encoding='utf-8'))
    if manifest['public_summary_fingerprint']!=digest(prior):raise ValueError('D2_PAIR_CHANGED')
    candidates={p['patient_id'] for p in manifest['patients'] if not p['memberships']}
    memberships,_=load_memberships()
    excluded=set().union(*memberships.values())
    if len(candidates)!=368 or candidates&excluded:raise ValueError('EXCLUSION_SCOPE_CHANGED')
    hashes={s['path']:s['sha256'] for s in prior['sources']}
    for name,expected in hashes.items():
        if sha(COHORT/name)!=expected:raise ValueError('D2_SOURCE_HASH_CHANGED')
    def rows(table):
        with (COHORT/'csv'/table).open(encoding='utf-8-sig',newline='') as stream:
            for n,r in enumerate(csv.DictReader(stream),2):
                if r.get('PATIENT',r.get('Id')) in candidates:yield n,r
    patients={r['Id']:(n,r) for n,r in rows('patients.csv')}
    encounters={r['Id']:{'row':r,'number':n,'observations':[],'conditions':[],'procedures':[],
                           'imaging':[],'symptoms':[]} for n,r in rows('encounters.csv')}
    for table,key in [('observations.csv','observations'),('conditions.csv','conditions'),
                      ('procedures.csv','procedures'),('imaging_studies.csv','imaging')]:
        for n,r in rows(table):
            e=encounters.get(r.get('ENCOUNTER'))
            if e is not None:
                if e['row']['PATIENT']!=r['PATIENT']:raise ValueError('SOURCE_PATIENT_MISMATCH')
                e[key].append((n,r))
    json_refs={}; by_patient=defaultdict(list)
    for eid,e in encounters.items():by_patient[e['row']['PATIENT']].append(eid)
    for f in manifest['json_files']:
        # Read only the 368 candidate patients; filename is not treated as identity.
        if not any(pid in f['file'] for pid in candidates):continue
        path=COHORT/f['file']
        if sha(path)!=f['sha256']:raise ValueError('JSON_SOURCE_CHANGED')
        d=json.loads(path.read_text(encoding='utf-8'));pid=d['attributes']['id']
        if pid not in candidates:raise ValueError('JSON_FILENAME_ID_MISMATCH')
        json_refs[pid]={'source_file':'experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1/'+f['file'],
                        'file_sha256':f['sha256']}
        for i,entry in enumerate(d['record']['encounters']):
            e=encounters.get(entry['uuid'])
            if e is None:continue
            if e['row']['PATIENT']!=pid:raise ValueError('JSON_ENCOUNTER_PATIENT_MISMATCH')
            e.update(start_ms=entry['start'],end_ms=entry.get('stop') if entry.get('stop',0)>=entry['start'] else None,
                     json_pointer='/record/encounters/'+str(i),json_source=json_refs[pid])
        updates=[]
        for name,s in d.get('symptoms',{}).items():
            for module,m in s.get('sources',{}).items():
                for key,event in m.get('timeInfos',{}).items():
                    updates.append((event['time'],name,module,key,event['value']))
        updates.sort();state={};at=0
        for eid in sorted(by_patient[pid],key=lambda x:(encounters[x]['start_ms'],x)):
            e=encounters[eid]
            while at<len(updates) and updates[at][0]<=e['start_ms']:
                t,name,module,key,value=updates[at];state[name,module]=(t,key,value);at+=1
            for (name,module),(t,key,value) in sorted(state.items()):
                if value>0:
                    escape=lambda v:str(v).replace('~','~0').replace('/','~1')
                    e['symptoms'].append({'symptom_name':name,'value':value,'source_recorded_at':iso(t),
                        'json_pointer':'/symptoms/'+escape(name)+'/sources/'+escape(module)+'/timeInfos/'+escape(key),
                        'semantic_status':'ASOF_ENCODED_STATE_NOT_CURRENT_COMPLAINT'})
    if set(json_refs)!=candidates or any('start_ms' not in e for e in encounters.values()):raise ValueError('JSON_LINK_MISSING')
    return {'patients':patients,'encounters':encounters,'hashes':hashes,'json_refs':json_refs,
            'memberships':memberships,'prior_manifest_fingerprint':digest(manifest)}


def action_candidates(e,hashes):
    allowed=[]; rejected=[]; seen=set();start=e['start_ms']
    for number,row in sorted(e['observations'],key=lambda x:(x[1]['DATE'],x[0])):
        category,code=row['CATEGORY'],row['CODE'];when=millis(row['DATE']);upper=when+999
        release=max(1,math.ceil((upper-start)/60000))
        binding=tuple(row[k] for k in ('CATEGORY','CODE','DESCRIPTION','UNITS','TYPE'))
        reason=None
        if not row['VALUE'].strip():reason='EMPTY_SOURCE_VALUE'
        elif code in PAIN_REPORTS:
            if when>start:reason='PATIENT_REPORT_AFTER_DECISION_UNREVIEWED'
            kind='Ask'
        elif category in POLICY['test_categories']:kind='Test'
        else:reason='CATEGORY_OR_REPORTED_SEMANTICS_NOT_ALLOWLISTED'
        if reason is None and release>POLICY['max_release_minute']:reason='SOURCE_OUTSIDE_REPLAY_WINDOW'
        if reason is None and binding in seen:reason='REPEATED_BINDING_FIRST_VALUE_ONLY'
        display=row['DESCRIPTION']+' ('+row['UNITS']+'; '+row['TYPE']+')'
        target=('What recorded value is available for '+display+' ['+code+']?') if code in PAIN_REPORTS else display+' ['+code+']'
        if reason is None and len(target)>160:reason='TARGET_LENGTH_CONTRACT'
        source=source_ref('observations.csv',number,row,hashes)
        item={'action_id':'D3Action'+str(number),'source_record_id':'D3Obs'+str(number),
            'source_category':category,'code':code,'description':row['DESCRIPTION'],'value':row['VALUE'],
            'units':row['UNITS'],'value_type':row['TYPE'],'recorded_at':row['DATE'],
            'actual_available_at':None,'proxy_available_minute':release,'release_assumption_id':ASSUMPTION,
            'review_status':'NOT_REVIEWED','availability_policy':PROXY,'source':source,
            'decision_time':iso(start),'timestamp_precision':'second','canonical_target':target,
            'reason':reason,'execution_allowlisted':reason is None}
        if reason is None:
            item['action']=kind;item['source_display']=display;seen.add(binding);allowed.append(item)
        else:rejected.append(item)
    return allowed,rejected


def memory_record(number,row,visit,hashes):
    observed=millis(row['DATE']);upper=observed+999
    source=source_ref('observations.csv',number,row,hashes)
    return {'memory_id':'D3Memory'+str(number),'patient_id':row['PATIENT'],'visit_id':row['ENCOUNTER'],
        'source_id':'D3Obs'+str(number),'source_recorded_at':row['DATE'],'observed_at':row['DATE'],
        'available_at':None,'availability_unknown':True,'proxy_available_at':iso(upper),
        'visit_started_at':iso(visit['start_ms']),'visit_ended_at':iso(visit['end_ms']) if visit.get('end_ms') is not None else None,
        'source':source,'content_type':'recorded_observation',
        'visit_source':{'json_source':deepcopy(visit['json_source']),'json_pointer':visit['json_pointer'],
                        'source_encounter':source_ref('encounters.csv',visit['number'],visit['row'],hashes)},
        'review_status':'NOT_REVIEWED','availability_policy':PROXY,
        'text':json.dumps({k:row[k] for k in ('CATEGORY','CODE','DESCRIPTION','VALUE','UNITS','TYPE')},ensure_ascii=False,sort_keys=True)}


def runtime_memory(record):
    return {**{k:record[k] for k in ('memory_id','patient_id','visit_id','visit_started_at','source_id','text')},
            'observed_at':record['observed_at'],'available_at':record['proxy_available_at'],
            'kind':'objective_finding','supersedes':[]}


def assemble(inputs):
    enc=inputs['encounters'];hashes=inputs['hashes'];by_patient=defaultdict(list)
    for eid,e in enc.items():by_patient[e['row']['PATIENT']].append(eid)
    eligibility=[];eligible=[];memories={};per_enc_memory={};per_enc_actions={};rejections=Counter()
    for eid,e in sorted(enc.items(),key=lambda x:(x[1]['row']['PATIENT'],x[1]['start_ms'],x[0])):
        pid=e['row']['PATIENT'];reasons=[]
        cls=e['row']['ENCOUNTERCLASS'].lower()
        if cls not in PRIORITY:reasons.append('WELLNESS_COMPARISON_ONLY' if cls=='wellness' else 'NON_PRIORITY_ENCOUNTER_CLASS')
        past=[enc[x] for x in by_patient[pid] if enc[x]['start_ms']<e['start_ms']]
        if not past:reasons.append('NO_PRIOR_ENCOUNTER')
        ms=[]
        for visit in sorted(past,key=lambda v:(v['start_ms'],v['row']['Id'])):
            for n,r in visit['observations']:
                observed=millis(r['DATE'])
                ended=visit.get('end_ms')
                if (r['VALUE'].strip() and visit['start_ms']<=observed and observed+999<=e['start_ms']
                        and (ended is None or observed+999<=ended<=e['start_ms'])):
                    ms.append((n,r,visit))
        if not ms:reasons.append('NO_CUTOFF_SAFE_MEMORY')
        if not e['observations']:reasons.append('NO_SOURCE_OBSERVATION')
        if not e['conditions']:reasons.append('NO_SOURCE_CONDITION_LINK')
        actions,rejected=action_candidates(e,hashes)
        if not actions:reasons.append('NO_ALLOWLISTED_REPLAY_ACTION')
        item={'patient_id':pid,'encounter_id':eid,'encounter_class':cls,'decision_time':iso(e['start_ms']),
              'eligibility_reasons':reasons,'eligible_development_replay':not reasons,'selected':False,
              'assigned_split':None,'clinical_evaluation_ready':False,'selection_policy_fingerprint':digest(POLICY),
              'source_encounter':source_ref('encounters.csv',e['number'],e['row'],hashes),
              'past_encounter_count':len(past),'cutoff_memory_candidates':len(ms),'allowlisted_actions':len(actions),
              'action_exclusion_reasons':dict(Counter(a['reason'] for a in rejected))}
        eligibility.append(item)
        if not reasons:
            eligible.append((PRIORITY[cls],e['start_ms'],eid,item));per_enc_memory[eid]=ms;per_enc_actions[eid]=(actions,rejected)
    selected=[];used=set()
    for priority,start,eid,item in sorted(eligible):
        if item['patient_id'] in used:item['selection_status']='ELIGIBLE_NOT_SELECTED_PATIENT_PRIORITY'
        else:
            used.add(item['patient_id']);item['selected']=True;item['selection_status']='SELECTED';selected.append((eid,item))
    for item in eligibility:
        if not item['eligible_development_replay']:item['selection_status']='EXCLUDED'
        rejections.update(item['eligibility_reasons'])
    cases=[];events=[];references=[];ledger=[];initial=[];private_actions=[];wrappers=[];strict_wrappers=[]
    for ordinal,(eid,item) in enumerate(selected,1):
        e=enc[eid];pid=item['patient_id'];case_id='D3Case{:04d}'.format(ordinal);start=e['start_ms']
        item['case_ref']=case_id
        patient_number,patient=inputs['patients'][pid]
        birthday=datetime.fromisoformat(patient['BIRTHDATE']).date();day=(datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=start)).date()
        age=day.year-birthday.year-((day.month,day.day)<(birthday.month,birthday.day))
        current={'case_id':case_id,'patient_id':pid,'visit_id':eid,'decision_time':iso(start),'chief_complaint':None,
            'demographics':{'age_at_index':age,'sex':patient['GENDER']},'encounter_class':item['encounter_class'],
            'initial_vitals':{},'objective_findings':[],'current_symptoms':[],
            'encoded_symptom_candidates':e['symptoms'],'clinical_review_status':'NOT_REVIEWED',
            'actual_available_at':None,'source_recorded_at':None,'timestamp_precision':'UNKNOWN',
            'availability_policy':STRICT,'source_encounter':item['source_encounter'],
            'source_demographics':{**source_ref('patients.csv',patient_number,patient,hashes),'derivation':'AGE_FROM_BIRTHDATE_AT_INDEX'},
            'json_source':e['json_source'],'json_pointer':e['json_pointer']}
        initial.append(current)
        memory_ids=[]
        for n,r,visit in per_enc_memory[eid]:
            m=memory_record(n,r,visit,hashes);memories[m['memory_id']]=m;memory_ids.append(m['memory_id'])
        action_ids=[]
        allowed,rejected=per_enc_actions[eid]
        for a in allowed+rejected:
            a.update(case_id=case_id,patient_id=pid,visit_id=eid)
            private_actions.append(a)
            if a['execution_allowlisted']:events.append(a);action_ids.append(a['action_id'])
        ref_ids=[]
        for n,r in e['conditions']:
            rid=case_id+'Reference'+str(n);ref_ids.append(rid)
            references.append({'reference_candidate_id':rid,'case_id':case_id,'patient_id':pid,
                'condition_code':r['CODE'],'condition_description':r['DESCRIPTION'],'source_condition_start':r['START'],
                'source_encounter':eid,'timing_status':reference_timing(r['START'],iso(start)),
                'independent_review_status':'NOT_REVIEWED','clinical_gold_status':'UNAPPROVED',
                'source':source_ref('conditions.csv',n,r,hashes),'correct':None})
        for n,r in e['procedures']+e['imaging']:
            private_actions.append({'case_id':case_id,'patient_id':pid,'visit_id':eid,
                'execution_allowlisted':False,'reason':'METADATA_NOT_RESULT',
                'source':source_ref('procedures.csv' if 'START' in r else 'imaging_studies.csv',n,r,hashes),
                'actual_available_at':None,'review_status':'NOT_REVIEWED'})
        c={'case_id':case_id,'patient_id':pid,'visit_id':eid,'decision_time':iso(start),
           'initial_observation_id':case_id,'memory_ids':memory_ids,'action_ids':action_ids,
           'reference_candidate_ids':ref_ids,'purpose':'DEVELOPMENT_ONLY','clinical_evaluation_ready':False}
        cases.append(c)
        wrapper=project_wrapper(c,current,[memories[x] for x in memory_ids],allowed,hashes)
        adapt_case(wrapper);wrappers.append(wrapper)
        strict=project_wrapper(c,current,[],[],hashes,policy=STRICT);adapt_case(strict);strict_wrappers.append(strict)
        for scope,records in [('initial_observation',[current]),
                              ('initial_demographics',[{'source':current['source_demographics']}]),
                              ('initial_encounter_metadata',[{'source':current['source_encounter']}]),
                              ('memory',[memories[x] for x in memory_ids]),
                              ('action',allowed+rejected),('reference',[r for r in references if r['case_id']==case_id]),
                              ('encoded_symptom_candidate',e['symptoms'])]:
            for i,r in enumerate(records,1):
                ledger.append({'ledger_id':case_id+':'+scope+':'+str(i),'case_id':case_id,'patient_id':pid,'visit_id':eid,
                    'scope':scope,'source':deepcopy(r.get('source',r.get('json_source',e['json_source']))),
                    'source_recorded_at':r.get('recorded_at',r.get('source_recorded_at',r.get('source_condition_start'))),
                    'timestamp_precision':'date' if scope=='reference' else r.get('timestamp_precision','second' if scope in {'action','memory'} else 'millisecond' if scope=='encoded_symptom_candidate' else 'UNKNOWN'),
                    'actual_available_at':None,'decision_time':iso(start),
                    'proxy_available_minute':r.get('proxy_available_minute'),
                    'proxy_available_at':r.get('proxy_available_at'),
                    'availability_policy':PROXY if scope in {'memory','action'} else STRICT,
                    'release_assumption_id':ASSUMPTION if scope in {'memory','action','initial_observation','initial_demographics','initial_encounter_metadata'} else None,
                    'strict_source_status':'UNKNOWN_NOT_EXPORTED',
                    'proxy_export_scope':'ACTION_AFTER_EXECUTION' if scope=='action' and r.get('execution_allowlisted') else 'PAST_MEMORY_ONLY' if scope=='memory' else 'STATIC_DEMOGRAPHICS_METADATA_ONLY' if scope in {'initial_observation','initial_demographics','initial_encounter_metadata'} else 'PRIVATE_REVIEW_ONLY',
                    'clinical_review_status':'NOT_REVIEWED'})
    counts=Counter(a['action'] for a in events)
    summary={'schema_version':'locdi.source_replay.D3.v1','version':VERSION,'purpose':'DEVELOPMENT_ONLY',
        'selection_policy':POLICY,'candidate_patients':len(inputs['patients']),'candidate_encounters':len(enc),
        'eligible_encounters':len(eligible),'eligible_patients':len(used),'selected_encounters':len(cases),
        'selected_patients':len(used),'selected_class_counts':dict(Counter(x[1]['encounter_class'] for x in selected)),
        'exclusion_reason_counts_nonexclusive':dict(rejections),'eligible_not_selected':len(eligible)-len(cases),
        'initial_chief_complaint':0,'initial_vitals':0,'initial_objective_findings':0,'initial_current_symptoms':0,
        'demographics_metadata_cases':len(cases),'encoded_symptom_candidate_cases':sum(bool(x['encoded_symptom_candidates']) for x in initial),
        'action_counts':dict(counts),'ask_target_kinds':len({a['code'] for a in events if a['action']=='Ask'}),
        'test_binding_kinds':len({a['canonical_target'] for a in events if a['action']=='Test'}),
        'consult_events':0,'memory_records':len(memories),'memory_case_references':sum(len(c['memory_ids']) for c in cases),
        'reference_candidates':len(references),'reviewed_reference_count':0,'clinical_gold_count':0,
        'ledger_rows':len(ledger),'source_linked_records':len(events)+len(memories)+len(references),
        'source_link_denominator':len(events)+len(memories)+len(references),
        'strict_action_replay_ready':0,'strict_source_blocked_cases':len(cases),'proxy_replay_ready':len(cases),
        'assigned_split_count':0,'legacy_scope_patient_overlap':0,'new_llm_generations':0,
        'utility_fit_count':0,'voi_probability_estimates':0,'clinical_verdicts_generated':0,'research_readiness':False}
    private={'cases.jsonl':cases,'initial_observations.jsonl':initial,'action_events.jsonl':events,
        'action_candidates.jsonl':private_actions,'reference_candidates.jsonl':references,'candidate_selection.jsonl':eligibility,
        'source_visibility_ledger.jsonl':ledger,'generated_proxy_wrappers.jsonl':wrappers,
        'generated_strict_wrappers.jsonl':strict_wrappers,
        'memory.json':{'schema_version':'locdi.canonical_memory.D3.v1','records':list(memories.values())},
        'patient_split_manifest.json':{'schema_version':'locdi.source_split.D3.v1','assigned_splits':False,
            'legacy_patient_sets':{k:sorted(v) for k,v in inputs.get('memberships',{}).items()},
            'patients':[{'patient_id':pid,'assigned_split':None,'selected_encounters':[c['visit_id'] for c in cases if c['patient_id']==pid],
                         'independent_clinical_test_approved':False} for pid in sorted(inputs['patients'])]}}
    return {'summary':summary,'private':private,'source_hashes':hashes,
            'source_json_hashes':{r['source_file']:r['file_sha256'] for r in inputs['json_refs'].values()}}


def project_wrapper(c,initial,memories,actions,hashes,policy=PROXY):
    visits={}
    for m in memories:
        record=runtime_memory(m)
        visits.setdefault(m['visit_id'],{'visit_id':m['visit_id'],'started_at':m['visit_started_at'],
                                        'ended_at':m.get('visit_ended_at'),'records':[]})['records'].append(record)
    PatientMemoryStore([runtime_memory(m) for m in memories])
    events=[];sources=[];symptoms=[];start=millis(c['decision_time'])
    for a in actions:
        t=millis(a['recorded_at']);eid=c['case_id']+':'+a['action_id']
        source={'event_id':eid,'case_id':c['case_id'],'patient_id':c['patient_id'],'encounter_id':c['visit_id'],
            'timestamp':a['recorded_at'],'timing_precision':'second','timestamp_lower_epoch_ms':t,'timestamp_upper_epoch_ms':t+999,
            'evidence':{'category':a['source_category'],'kind':'recorded_observation','units':a['units'],'value':a['value']},
            'target':{'description':a['source_display'],'code':a['code'],'system':None},'source_metadata':deepcopy(a['source']),
            'availability_status':'UNKNOWN_EXPLICIT_DEVELOPMENT_PROXY','reveal_status':'ACTION_GATED',
            'relative_to_decision':'after_decision' if t>start else 'before_decision' if t+999<start else 'same_precision_interval_uncertain',
            'event_type':'observations','action_type_candidate':a['action']}
        sources.append(source)
        if a['action']=='Ask':
            fact={'active_status':None,'code':a['code'],'description':a['description'],'duration':None,'onset':None,
                  'recorded_at':a['recorded_at'],'units':a['units'],'value':a['value']}
            evidence={'facts':[fact],'kind':'recorded_patient_report','replay_bin':'UNBINNED_SOURCE_VALUE_NO_VOI'}
            symptoms.append({'source_id':a['source_record_id'],'source_timestamp':a['recorded_at'],
                'kind':'recorded_patient_report','evidence_grade':'SOURCE_LINKED_DEVELOPMENT_ONLY',
                'source_metadata':{'_source':a['source']},'visible_value':{'symptom_name':a['description'],'value':a['value']},
                'visibility_allowed':True})
        else:
            evidence={'records':[{'code':a['code'],'description':a['source_display'],'event_type':'observations',
                'recorded_at':a['recorded_at'],'recorded_evidence':source['evidence'],'system':None,'timing_precision':'second'}],
                'replay_bin':'UNBINNED_SOURCE_VALUE_NO_VOI'}
        events.append({'event_id':eid,'action':a['action'],'target':a['canonical_target'],
                       'available_minute':a['proxy_available_minute'],'evidence':json.dumps(evidence,ensure_ascii=False,sort_keys=True)})
    doctor={'arrival_info':{'clinical_information_need':None,'encounter_class':initial['encounter_class'],
                            'reviewed_generic_purposes':[],'safe_initial_observations':[]},'initial_vitals':{}}
    patient={'associated_symptoms':[]}
    if policy==PROXY:
        doctor['demographics']=deepcopy(initial['demographics']);patient.update(age=initial['demographics']['age_at_index'],sex=initial['demographics']['sex'])
    return {'schema_version':'longitudinal_single_file_v34.1','case_id':c['case_id'],'patient_id':c['patient_id'],
        'agent_visible':{'case_id':c['case_id'],'current_initial_state':{'doctor_visible_info':doctor,'patient_visible_info':patient},
            'index_encounter_id':c['visit_id'],'index_encounter_start':c['decision_time'],
            'longitudinal':{'patient_id':c['patient_id'],'visit_id':c['visit_id'],'started_at':c['decision_time']},
            'past_visits':sorted(visits.values(),key=lambda v:(v['started_at'],v['visit_id']))},
        'environment':{'current_encounter_events':events,'source_event_store':sources,
            'data_replay':{'policy':policy,'original_decision_time':c['decision_time'],'observed_release_time':False,
                           'requires_time_series_loader_for_latest_value':False}},
        'evaluation':{'diagnosis_labels':[],'accepted_departments_source_policy':[],'observed_index_diagnoses':[]},
        'task_annotations':{'annotation':{'review_status':'unreviewed','source':'D3_SOURCE_REPLAY_NO_GOLD'},
            'gold':{k:None for k in ('acceptable_next_actions','accepted_departments','accepted_diagnoses','relevant_memory_ids','stale_memory_ids')},
            'metric_readiness':{k:False for k in ('diagnosis_accuracy','memory_retrieval','routing_accuracy','voi_regret')}},
        'provenance':{'source_package':VERSION,'source_sha256':{'experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1/'+k:v for k,v in hashes.items()},
            'visibility_policy':policy+':'+ASSUMPTION,
            'source_case_audit':{'case_id':c['case_id'],'clinical_review_status':'unreviewed','diagnosis_sources':[],
                'symptom_sources':symptoms,'other_current_source_candidates_not_projected':[],
                'source_annotation_store':'NOT_APPLICABLE_NO_CLINICAL_ANNOTATIONS','source_policy_primary_specialty_candidate':'NOT_REVIEWED',
                'selection_score':None}}}


def atomic_rows(root,name,rows,jsonl=True):
    root=Path(root).resolve()
    if not root.is_relative_to(ROOT.resolve()) or Path(name).name!=name:raise ValueError('OUTPUT_PATH')
    path=root/name
    if path.exists():raise FileExistsError('NO_CLOBBER')
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',dir=str(root),delete=False) as stream:
            tmp=Path(stream.name)
            if jsonl:
                for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
            else:stream.write(json.dumps(rows,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.link(str(tmp),str(path))
    finally:
        if tmp is not None:tmp.unlink()
    return path


def save_bundle(bundle):
    from validation.source_replay_validator import validate_source_bundle
    validate_source_bundle(bundle)
    for root in (PRIVATE,PUBLIC):
        if root.exists():raise FileExistsError('VERSION_ALREADY_EXISTS')
        if not root.resolve().is_relative_to(ROOT.resolve()):raise ValueError('OUTPUT_REDIRECTED')
    PRIVATE.mkdir(parents=True,exist_ok=False);PUBLIC.mkdir(parents=True,exist_ok=False)
    files={}
    for name,value in bundle['private'].items():
        path=atomic_rows(PRIVATE,name,value,jsonl=name.endswith('.jsonl'))
        files[name]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'private':True}
    result={'schema_version':'locdi.source_replay_manifest.D3.v1','dataset_version':VERSION,
        'status':'DEVELOPMENT_ONLY','files':files,'coverage':bundle['summary'],
        'source_hashes':bundle['source_hashes'],'source_json_hashes_fingerprint':digest(bundle['source_json_hashes']),
        'clinical_review_status':'NOT_REVIEWED','license':'UNKNOWN_NOT_APPROVED_FOR_REDISTRIBUTION',
        'consult_enabled':False,'research_evaluation_ready':False}
    result['manifest_fingerprint']=digest(result)
    atomic_rows(PUBLIC,'dataset_manifest.json',result,jsonl=False)
    return result


def load_replay_cases(*,policy=PROXY,accept_development_proxy=False):
    if policy not in {STRICT,PROXY}:raise ValueError('UNSUPPORTED_POLICY')
    if policy==STRICT:raise ValueError('STRICT_SOURCE_ACTUAL_AVAILABILITY_UNVERIFIED')
    if accept_development_proxy is not True:raise ValueError('EXPLICIT_PROXY_ACCEPTANCE_REQUIRED')
    manifest=json.loads((PUBLIC/'dataset_manifest.json').read_text(encoding='utf-8'))
    fingerprint=manifest.get('manifest_fingerprint')
    if digest({k:v for k,v in manifest.items() if k!='manifest_fingerprint'})!=fingerprint:raise ValueError('MANIFEST_INTEGRITY')
    entry=manifest['files']['generated_proxy_wrappers.jsonl'];path=(ROOT/entry['path']).resolve()
    if not path.is_relative_to(PRIVATE.resolve()) or sha(path)!=entry['sha256']:raise ValueError('WRAPPER_INTEGRITY')
    return load_cases(path)
