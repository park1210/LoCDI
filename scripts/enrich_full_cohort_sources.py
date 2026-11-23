"""C4-05 full-size source enrichment, offline release snapshots and review pools.

Private source extraction is not clinical approval or automatic Doctor exposure.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import timedelta
from functools import lru_cache
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.dataset_loader import load_cases
from app.locdi.reference_readiness import source_path,verify_d3
from app.locdi.routing_data_reuse import BANK,SOURCE,read
from app.locdi.memory_reference_evaluation import VERSION as REVIEW_VERSION,PROCEDURE,load_bundle,evaluate,released_query
from app.locdi.source_enrichment import (VERSION,PARTIAL_VERSION,PARTIAL_PROCEDURE,POOL_SEED,
    fact_row,validate_fact,iso_ms,scalar_count,released_projection,validate_candidate,make_pool,validate_partial_review,judged_metrics)
from app.core.longitudinal_memory import PatientMemoryStore,timestamp
from scripts.audit_locdi_core_evaluation import benchmark,eligible_reference

DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
@lru_cache(maxsize=1)
def data_sha():return sha(DATA)
COLLECTIONS={'observations':'Observation','reports':'SyntheaReport','conditions':'Condition',
    'procedures':'Procedure','medications':'Medication','allergies':'Allergy', 'careplans':'CarePlan',
    'medicationRequests':'MedicationRequest','questionnaires':'Questionnaire','questionnaireResponses':'QuestionnaireResponse',
    'immunizations':'Immunization','imagingStudies':'ImagingStudy','devices':'Device','supplies':'Supply'}
TYPES=['Encounter','EncounterReason','AttributeReasonState',*COLLECTIONS.values()]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as s:json.dump(value,s,ensure_ascii=False,indent=2);s.write('\n')


def rows(path,values):
    with path.open('x',encoding='utf8',newline='\n') as s:
        for value in values:s.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')


def baseline():
    prior=ROOT/'outputs/private/research_evaluation/c4_04_v1'
    m=load(ROOT/'outputs/research_evaluation/c4_04_v1/reports/output_manifest.json');index=load(prior/'private_index.json')
    for n,h in {**m['files'],**index['files']}.items():
        if sha(ROOT/n)!=h:raise ValueError('C4_04_SEALED_ASSET_CHANGED')
    result=load(prior/'baseline_hashes.json');result.update(m['files']);result.update(index['files'])
    for folder in ['outputs/research_evaluation/c4_04_v1','outputs/private/research_evaluation/c4_04_v1']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():result[p.relative_to(ROOT).as_posix()]=sha(p)
    if any(sha(ROOT/n)!=h for n,h in result.items()):raise ValueError('PROTECTED_ASSET_CHANGED')
    return result


def descriptors(original,d3private,training):
    trace=load(ROOT/'outputs/private/research_evaluation/c4_02_v1/reports/source_trace.json')
    cases=[]
    for ordinal,(c,t) in enumerate(zip(original,trace),1):
        cases.append({'dataset':'Original50','case_ref':'OriginalCase%03d'%ordinal,'ordinal':ordinal,
            'patient_id':c['patient_id'],'visit_id':c['agent_visible']['index_encounter_id'],
            'decision_time':c['agent_visible']['index_encounter_start'],'raw_path':t['raw_json_path'],
            'declared_raw_sha256':t['raw_json_sha256'],'legacy_reference_text':json.dumps(c['provenance'])})
    initial={r['case_id']:r for r in d3private['initial_observations.jsonl']}
    for ordinal,c in enumerate(d3private['cases.jsonl'],1):
        source=initial[c['case_id']]['json_source']
        cases.append({'dataset':'D3','case_ref':'D3Case%03d'%ordinal,'ordinal':ordinal,'patient_id':c['patient_id'],
            'visit_id':c['visit_id'],'decision_time':c['decision_time'],'raw_path':str(source_path(source['source_file'])),
            'declared_raw_sha256':source['file_sha256'],'legacy_reference_text':''})
    source_index={r['case_id']:r for r in read(SOURCE)}
    for ordinal,r in enumerate(training,1):
        s=source_index[r['training_case_id']];ref=s['symptom_sources'][0]['source_metadata']['_source']
        if s['patient_id']!=r['training_patient_id']:raise ValueError('BANK_SOURCE_PATIENT_MISMATCH')
        cases.append({'dataset':'TrainingBank','case_ref':'TrainingVisit%03d'%ordinal,'ordinal':ordinal,
            'patient_id':s['patient_id'],'visit_id':s['index_encounter']['encounter_id'],
            'decision_time':s['index_encounter']['start_time'],'raw_path':str(source_path(ref['raw_json_file'])),
            'declared_raw_sha256':None,'legacy_reference_text':json.dumps(s)})
    return cases


def resource_entries(enc,path):
    yield 'Encounter',path,enc
    for k in ('reason','reasons','chief_complaint','patient_statement','symptom_narrative'):
        if enc.get(k) not in (None,'',[],{}):yield 'EncounterReason',path+'/'+k,enc[k]
    def nested(values,prefix):
        for i,value in enumerate(values):
            yield 'Observation',prefix+'/'+str(i),value
            yield from nested(value.get('observations',[]),prefix+'/'+str(i)+'/observations')
    for collection,kind in COLLECTIONS.items():
        for i,value in enumerate(enc.get(collection,[])):
            p=path+'/'+collection+'/'+str(i)
            yield kind,p,value
            if collection in ('observations','reports'):
                yield from nested(value.get('observations',[]),p+'/observations')


def scan_sources(cases,private,public):
    documents={};pins={};stats={};case_rows=[];total=Counter();unique_resources=set();current_index={}
    out=private/'source_facts.jsonl'
    with out.open('x',encoding='utf8') as stream:
        for case in cases:
            path=Path(case['raw_path'])
            if str(path) not in documents:
                documents[str(path)]=load(path)
                if str(path) not in pins:pins[str(path)]=sha(path)
                # Raw JSON spans >1 GB; keep a bounded parse cache, never drop patients.
                if len(documents)>2:documents.pop(next(iter(documents)))
            document=documents[str(path)];h=pins[str(path)]
            if case['declared_raw_sha256'] and h!=case['declared_raw_sha256']:raise ValueError('DECLARED_RAW_JSON_HASH')
            if document['attributes']['id']!=case['patient_id']:raise ValueError('RAW_PATIENT_LINK')
            encounters=document['record']['encounters'];indices=[i for i,e in enumerate(encounters) if e['uuid']==case['visit_id']]
            if len(indices)!=1:raise ValueError('RAW_CURRENT_ENCOUNTER_LINK')
            index=indices[0];enc=encounters[index];base='/record/encounters/'+str(index)
            if abs(timestamp(iso_ms(enc['start'])).timestamp()-timestamp(case['decision_time']).timestamp())>1:
                raise ValueError('RAW_DECISION_START_MISMATCH')
            ds=case['dataset'];case_count=Counter();all_count=Counter();field_count=0;new_count=0;new_fields=0;reason_fields=0
            for eidx,e in enumerate(encounters):
                relation='current' if eidx==index else 'past' if e['start']<enc['start'] else 'future'
                for kind,p,value in resource_entries(e,'/record/encounters/'+str(eidx)):
                    key=(ds,kind)
                    if key not in stats:stats[key]={'counts':Counter(),'patients_any':set(),'patients_current':set()}
                    s=stats[key];s['counts'][relation+'_occurrences']+=1;s['patients_any'].add(case['patient_id']);all_count[kind]+=1
                    if relation!='current':continue
                    s['patients_current'].add(case['patient_id']);case_count[kind]+=1
                    r=fact_row(case,path,h,p,value,kind,base);validate_fact(r,document,h)
                    if r['recorded_at']:s['counts']['with_recorded_at']+=1
                    if r['available_at']:s['counts']['with_explicit_available_at']+=1
                    if r['recorded_at'] and timestamp(r['recorded_at'])>timestamp(case['decision_time']):s['counts']['recorded_after_initial_decision']+=1
                    r['fact_ref']='SourceFact%08d'%(total['records']+1)
                    uid=value.get('uuid') if isinstance(value,dict) else None
                    legacy=bool(uid and uid in case['legacy_reference_text'])
                    r['legacy_reference_present']=legacy
                    r['link_status']='EXISTING_SOURCE_REFERENCE_EXPANDED' if legacy else 'NEW_LINK_OUTSIDE_EXISTING_REFERENCE_PROJECTION'
                    n=scalar_count(value);field_count+=n
                    if not legacy:new_count+=1;new_fields+=n
                    if kind=='EncounterReason':reason_fields+=1
                    stream.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n');total['records']+=1
                    if uid:unique_resources.add((ds,str(path),uid))
            # Final simulator attributes are inspected but never promoted as current reports.
            attribute_reasons=0
            for key,value in document['attributes'].items():
                if 'reason' not in key.casefold() or value in (None,'',[],{}):continue
                p='/attributes/'+key.replace('~','~0').replace('/','~1')
                r=fact_row(case,path,h,p,value,'AttributeReasonState',base);validate_fact(r,document,h)
                r.update(fact_ref='SourceFact%08d'%(total['records']+1),temporal_status='FINAL_STATE_NOT_ASOF_DECISION',
                         link_status='QUARANTINED_FINAL_ATTRIBUTE_STATE',legacy_reference_present=False)
                stream.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n');total['records']+=1;attribute_reasons+=1
                key=(ds,'AttributeReasonState')
                if key not in stats:stats[key]={'counts':Counter(),'patients_any':set(),'patients_current':set()}
                stats[key]['patients_any'].add(case['patient_id']);stats[key]['counts']['quarantined_final_state_occurrences']+=1
            current_index[case['case_ref']]={'source_file':str(path),'source_sha256':h,'encounter_pointer':base,
                'encounter_fingerprint':digest(enc),'raw_encounter_start':iso_ms(enc['start'])}
            case_rows.append({'dataset':ds,'case_ref':case['case_ref'],'current_resource_occurrences':dict(case_count),
                'all_encounter_resource_occurrences':dict(all_count),'raw_encounters_scanned':len(encounters),
                'scalar_field_links':field_count,'new_record_links_outside_legacy':new_count,
                'new_scalar_field_links_outside_legacy':new_fields,'encounter_reason_field_count':reason_fields,
                'explicit_chief_complaint_fields':int(enc.get('chief_complaint') not in (None,'',[],{})),
                'explicit_patient_narrative_fields':sum(enc.get(k) not in (None,'',[],{}) for k in ('patient_statement','symptom_narrative')),
                'quarantined_attribute_reason_fields':attribute_reasons,'verified_chief_complaint_count':0,
                'strict_new_doctor_exposure_count':0,'clinical_review_status':'UNREVIEWED'})
    coverage=[]
    for ds in ('Original50','D3','TrainingBank'):
        for kind in TYPES:
            s=stats.get((ds,kind),{'counts':Counter(),'patients_any':set(),'patients_current':set()})
            coverage.append({'dataset':ds,'source_resource_type':kind,'patients_with_any_source':len(s['patients_any']),
                'patients_with_current_source':len(s['patients_current']),**dict(s['counts']),
                'source_format':'NATIVE_SYNTHEA_JSON_NOT_FHIR','current_doctor_exposure':'NOT_AUTHORIZED_FROM_RECORDED_TIME_ALONE',
                'independent_clinical_review':'REQUIRED','query_role':'QUARANTINED_UNLESS_SEPARATELY_REPLAY_RELEASED',
                'label_risk':'REFERENCE_RESTRICTED' if kind in ('Condition','EncounterReason','AttributeReasonState') else 'EXPOSURE_REVIEW_REQUIRED'})
    rows(private/'case_source_index.jsonl',[{k:v for k,v in c.items() if k!='legacy_reference_text'} for c in cases])
    write(private/'raw_source_hashes.json',pins);write(private/'current_encounter_links.json',current_index)
    write(public/'reports/source_resource_coverage.json',coverage);write(public/'reports/source_case_coverage.json',case_rows)
    summary={'raw_json_files':len(pins),'source_case_visits':len(cases),'extracted_resource_occurrences':total['records'],
        'unique_current_resource_ids_by_dataset':len(unique_resources),'scope':'All native encounters scanned; current resources and quarantined final reason attributes extracted; no FHIR synthesis',
        'by_dataset':{ds:{'case_visits':sum(r['dataset']==ds for r in case_rows),
            'scalar_field_links':sum(r['scalar_field_links'] for r in case_rows if r['dataset']==ds),
            'new_record_links_outside_legacy':sum(r['new_record_links_outside_legacy'] for r in case_rows if r['dataset']==ds),
            'new_scalar_field_links_outside_legacy':sum(r['new_scalar_field_links_outside_legacy'] for r in case_rows if r['dataset']==ds),
            'encounter_reason_field_count':sum(r['encounter_reason_field_count'] for r in case_rows if r['dataset']==ds),
            'verified_chief_complaint_count':0} for ds in ('Original50','D3','TrainingBank')}}
    write(public/'reports/source_enrichment_summary.json',summary)
    return summary,pins


def extract_past_context(cases,private,public):
    """Additional source history sidecar; never append to the fixed Memory pool."""
    counts=Counter();patients=defaultdict(set);temporal=Counter();cache={};fields=Counter()
    with (private/'historical_context_source_facts.jsonl').open('x',encoding='utf8') as out:
        for case in cases:
            path=Path(case['raw_path']);key=str(path)
            if key not in cache:
                cache[key]=(load(path),sha(path))
                if len(cache)>2:cache.pop(next(iter(cache)))
            document,h=cache[key];cutoff=timestamp(case['decision_time'])
            if document['attributes']['id']!=case['patient_id']:raise ValueError('HISTORY_PATIENT_LINK')
            for index,encounter in enumerate(document['record']['encounters']):
                if encounter['uuid']==case['visit_id'] or timestamp(iso_ms(encounter['start']))>=cutoff:continue
                base='/record/encounters/'+str(index)
                for collection in ('allergies','medications','careplans'):
                    kind=COLLECTIONS[collection]
                    for i,value in enumerate(encounter.get(collection,[])):
                        source_case=dict(case,visit_id=encounter['uuid'])
                        r=fact_row(source_case,path,h,base+'/'+collection+'/'+str(i),value,kind,base)
                        validate_fact(r,document,h)
                        relation=('RECORDED_BY_DECISION' if timestamp(r['recorded_at'])<=cutoff else 'RECORDED_AFTER_DECISION_QUARANTINED') if r['recorded_at'] else 'RECORD_TIME_UNKNOWN'
                        r.update(decision_visit_reference=case['visit_id'],decision_time=case['decision_time'],
                            prior_encounter_before_cutoff=True,record_time_relation=relation,
                            contains_future_stop_field=bool(value.get('stop') and timestamp(iso_ms(value['stop']))>cutoff),
                            projection_role='PRIVATE_HISTORICAL_SOURCE_NOT_FIXED_MEMORY_OR_AGENT_INPUT',
                            history_interpretation='Recorded historical event; no claim of current allergy/medication/care-plan status')
                        out.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
                        counts[(case['dataset'],kind)]+=1;patients[(case['dataset'],kind)].add(case['patient_id'])
                        temporal[relation]+=1;fields[case['dataset']]+=scalar_count(value)
    result={'scope':'All prior-encounter Allergy/Medication/CarePlan source occurrences; no changes to original Memory and no automatic current-status inference',
        'record_occurrences':sum(counts.values()),'temporal_counts':dict(temporal),'strict_doctor_exposure_count':0,
        'by_dataset':{ds:{'records':sum(n for (d,_),n in counts.items() if d==ds),
            'scalar_field_occurrences':fields[ds],
            'resources':{kind:{'records':counts[(ds,kind)],'patients':len(patients[(ds,kind)])} for kind in ('Allergy','Medication','CarePlan')}}
            for ds in ('Original50','D3','TrainingBank')}}
    write(public/'reports/historical_context_enrichment.json',result);return result


def full_packet(c,ordinal,history,minute):
    identity=c['agent_visible']['longitudinal'];event=next(e for e in c['environment']['current_encounter_events'] if e['action']=='Ask')
    query,clean=released_query(history,snapshot_minute=minute,source_event=event)
    memories=[deepcopy(r) for v in c['agent_visible']['past_visits'] for r in v['records']]
    ids=eligible_reference(memories,identity);memories=sorted([r for r in memories if r['memory_id'] in ids],key=lambda r:(timestamp(r['available_at']),r['memory_id']))
    return {'schema_version':'locdi.memory_blinded_packet.v1','snapshot_id':'Snapshot%03d'%ordinal,'case_ordinal':ordinal,
        'private_identity':identity,'decision_minute':minute,'decision_time':(timestamp(identity['started_at'])+timedelta(minutes=minute)).isoformat(),
        'memory_cutoff':identity['started_at'],'chief_complaint':None,'complaint_status':'MISSING_SOURCE_COMPLAINT',
        'released_history':deepcopy(history),'released_source_event':event,'query_input_projection':clean,
        'query':query,'query_fingerprint':digest(query),'eligible_memory':memories,
        'review_status':'UNREVIEWED','model_prediction_included':False,'retrieval_output_included':False,
        'memory_provenance':[{'memory_id':r['memory_id'],'record_fingerprint':digest(r),'source_id':r['source_id'],
                              'original_dataset_sha256':data_sha()} for r in memories]}


def blank_full(packet,packet_sha):
    return {'snapshot_id':packet['snapshot_id'],'packet_sha256':packet_sha,'query_fingerprint':packet['query_fingerprint'],
        'review_status':'UNREVIEWED','reviewer_private_ref':None,'reviewer_independent':None,'reviewed_at':None,
        'procedure_version':PROCEDURE,'query_suitable':None,'query_visibility_reviewed':None,'whole_eligible_pool_reviewed':None,
        'model_or_retrieval_outputs_seen':None,'visibility_scope':None,'recorded_time_proxy_explicitly_reviewed':None,'rationale':None,
        'records':[{'memory_id':r['memory_id'],'relevant':None,'stale':None,'conflict':None,'rationale':None,'not_evaluable_reason':None}
                   for r in packet['eligible_memory']]}


def run_original(original,private,public,source_pins):
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from test_simulation import SourceActionClient,smoke_options
    from app.locdi.run_simulation import run_case
    bundle=private/'full_pool_ask_v1';bundle.mkdir();judged=private/'judged_pool_v1';judged.mkdir()
    full_entries=[];full_reviews=[];pool_entries=[];pool_reviews=[];candidate_rows=[];pool_ledger=[];public_rows=[];runtime=[]
    memory_links=[];source_gold_mentions=0
    for ordinal,c in enumerate(original,1):
        client=SourceActionClient();options=smoke_options(client)
        options.runner_options['memory_mode']='none'
        result=run_case(c,options=options)
        if not result['execution']['visits']:raise ValueError('SOURCE_REPLAY_NO_COMMITTED_VISIT')
        state=next(iter(result['execution']['visits'][0]['final_state']['cases'].values()))
        identity=c['agent_visible']['longitudinal'];events=c['environment']['current_encounter_events']
        records=[r for v in c['agent_visible']['past_visits'] for r in v['records']]
        for vi,v in enumerate(c['agent_visible']['past_visits']):
            for ri,r in enumerate(v['records']):memory_links.append({'case_ref':'OriginalCase%03d'%ordinal,'memory_id':r['memory_id'],
                'source_id':r['source_id'],'record_fingerprint':digest(r),'dataset_sha256':data_sha(),
                'original_record_pointer':'/%d/agent_visible/past_visits/%d/records/%d'%(ordinal-1,vi,ri)})
        candidates=[]
        for call in client.calls:
            observed=call['payload']['observation']
            if not observed.get('history') and not observed.get('objective_findings'):continue
            candidate=released_projection(observed,events,state['test_results'],identity)
            validate_candidate(candidate,events,identity)
            candidate.update(case_ref='OriginalCase%03d'%ordinal,snapshot_ref='OriginalCase%03d_Decision%03d'%(ordinal,len(candidates)+1))
            candidates.append(candidate);candidate_rows.append(candidate)
        if not candidates:raise ValueError('NO_SOURCE_RELEASED_SNAPSHOT_KEEP_CASE_BLOCKED')
        first_observed=next(call['payload']['observation'] for call in client.calls if call['payload']['observation'].get('history'))
        p=full_packet(c,ordinal,first_observed['history'],first_observed['elapsed_minutes']);name=p['snapshot_id']+'.json'
        write(bundle/name,p);full_entries.append({'snapshot_id':p['snapshot_id'],'file':name,'sha256':sha(bundle/name)})
        full_reviews.append(blank_full(p,sha(bundle/name)))
        final=candidates[-1]
        selected,ledger,counts=make_pool(records,identity,final['query'],final['source_facts'],ordinal)
        packet={'schema_version':PARTIAL_VERSION,'snapshot_id':'OriginalCase%03d'%ordinal,'private_identity':identity,
            'query_candidate':final,'candidate_memories':[{k:v for k,v in r.items() if k not in ('retrieval_score','status_as_of_visit')} for r in selected],
            'full_eligible_count':counts['eligible_count'],'original_memory_dataset_sha256':data_sha(),
            'review_status':'UNREVIEWED','method_membership_hidden':True,'model_predictions_included':False}
        name=packet['snapshot_id']+'.json';write(judged/name,packet)
        pool_entries.append({'snapshot_id':packet['snapshot_id'],'file':name,'sha256':sha(judged/name),'packet_fingerprint':digest(packet)})
        pool_reviews.append({'schema_version':PARTIAL_VERSION,'snapshot_id':packet['snapshot_id'],'packet_fingerprint':digest(packet),
            'query_fingerprint':final['query_fingerprint'],'review_status':'UNREVIEWED','reviewer_private_ref':None,
            'reviewer_independent':None,'reviewed_at':None,'procedure_version':PARTIAL_PROCEDURE,'query_suitable':None,
            'visibility_reviewed':None,'recorded_time_proxy_reviewed':None,'methods_or_predictions_seen':None,'rationale':None,
            'records':[{'memory_id':r['memory_id'],'relevant':None,'stale':None,'conflict':None,'rationale':None,'not_evaluable_reason':None} for r in selected]})
        pool_ledger.append({'snapshot_id':packet['snapshot_id'],'methods':ledger,'counts':counts})
        public_rows.append({'case_ref':packet['snapshot_id'],'snapshots':len(candidates),'eligible_records':counts['eligible_count'],
            'query_source_type':'RELEASED_TEST_RESULT_BASED' if any(x['action']=='Test' for x in final['source_links']) else 'RELEASED_HISTORY_BASED',
            'released_observation_count':len(final['source_links']),'reviewed_relevance_count':0,'unjudged_count':counts['eligible_count'],
            'pooled_records':counts['pooled_count'],'unpooled_fraction':counts['unpooled_fraction'],'pool_seed':counts['seed'],
            'full_eligibility':'REFERENCE_REQUIRED','partial_eligibility':'REFERENCE_REQUIRED','evaluation_status':'NOT_RUN',
            'clinical_suitability':'UNREVIEWED','decision_minute':final['decision_minute'],'runtime_status':result['status']})
        runtime.append({'case_ref':packet['snapshot_id'],'status':result['status'],'offline_dispatches':len(client.calls),
            'completed_tests':len(state['completed_tests']),'test_receipt_minutes':[r['observed_minute'] for r in state['test_results']],
            'diagnosis_unresolved':'diagnosis_unresolved' in state['alerts'],'actual_llm_generations':0})
    hashes={str(DATA):data_sha(),**source_pins}
    for target,entries in [(bundle,full_entries),(judged,pool_entries)]:
        m={'schema_version':'locdi.memory_review_bundle.v1' if target==bundle else PARTIAL_VERSION,
            'packets':entries,'source_hashes':hashes,'cohort_rule':'ALL_ORIGINAL50_INPUT_ORDER_NO_PATIENT_REMOVED',
            'quality_reference_status':'UNREVIEWED'}
        m['fingerprint']=digest(m);write(target/'manifest.json',m)
        write(target/'review_results_blank.json',{'schema_version':REVIEW_VERSION if target==bundle else PARTIAL_VERSION,
            'manifest_fingerprint':m['fingerprint'],'snapshots':full_reviews if target==bundle else pool_reviews})
    load_bundle(bundle)
    rows(private/'query_candidates.jsonl',candidate_rows);rows(private/'original_memory_links.jsonl',memory_links)
    write(private/'pool_method_ledger.json',pool_ledger);write(public/'reports/all50_snapshot_inventory.json',public_rows)
    write(public/'reports/offline_runtime_checks.json',runtime)
    return {'query_candidates':len(candidate_rows),'patients_with_released_candidates':len(public_rows),
        'history_only_candidates':sum(not any(x['action']=='Test' for x in r['source_links']) for r in candidate_rows),
        'test_result_candidates':sum(any(x['action']=='Test' for x in r['source_links']) for r in candidate_rows),
        'memory_links':len(memory_links),'judged_pool_records':sum(r['pooled_records'] for r in public_rows),
        'eligible_records_all50':sum(r['eligible_records'] for r in public_rows),'offline_dispatches':sum(r['offline_dispatches'] for r in runtime),
        'runtime_completed':sum(r['status']=='completed' for r in runtime),'strict_clinical_available_query_count':0,
        'independently_reviewed_queries':0,'actual_llm_generations':0}


def prepare(namespace,use_captured=False):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',namespace):raise ValueError('INVALID_NAMESPACE')
    public=ROOT/'outputs/research_evaluation'/namespace;private=ROOT/'outputs/private/research_evaluation'/namespace
    if use_captured:
        if not (private/'baseline_hashes.json').is_file() or list((public/'reports').glob('*')) or set(p.name for p in private.iterdir())!={'baseline_hashes.json'}:
            raise ValueError('CAPTURED_BASELINE_ONLY_NO_RESUME')
    else:
        if public.exists() or private.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
        b=baseline();public.mkdir(parents=True);private.mkdir(parents=True);write(private/'baseline_hashes.json',b)
    protected=load(private/'baseline_hashes.json')
    if any(sha(ROOT/n)!=h for n,h in protected.items()):raise ValueError('PROTECTED_ASSET_CHANGED')
    original=load_cases(DATA);dm,d3,validation,_=verify_d3();training=read(BANK)
    cases=descriptors(original,d3,training);summary,source_pins=scan_sources(cases,private,public)
    history=extract_past_context(cases,private,public)
    source_pins.update({str(BANK):sha(BANK),str(SOURCE):sha(SOURCE),str(DATA):data_sha()})
    for item in dm['files'].values():source_pins[str(ROOT/item['path'])]=item['sha256']
    write(private/'all_source_hashes.json',source_pins)
    replay=run_original(original,private,public,source_pins)
    structural,internal=benchmark(original);write(public/'reports/structural_150_replay.json',structural)
    write(private/'structural_internal_links.json',internal)
    groups={ds:{r['patient_id'] for r in cases if r['dataset']==ds} for ds in ('Original50','D3','TrainingBank')}
    prep=load(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json');split=prep['split_manifest']['patients']
    overlap={'Original50_D3':len(groups['Original50']&groups['D3']),'Original50_TrainingBank':len(groups['Original50']&groups['TrainingBank']),
        'D3_TrainingBank':len(groups['D3']&groups['TrainingBank']),'routing_train_validation':len(set(split['train'])&set(split['validation'])),
        'routing_train_holdout':len(set(split['train'])&set(split['test'])),'routing_validation_holdout':len(set(split['validation'])&set(split['test'])),
        'routing_validation_in_voi_training_bank':len(set(split['validation'])&groups['TrainingBank']),
        'new_independent_test_cohort_created':False,'original50_status':'DEVELOPMENT_EXPOSED_NOT_BLIND_HOLDOUT'}
    write(public/'reports/split_overlap.json',overlap)
    result={'status':'C4_05_SOURCE_ENRICHMENT_COMPLETED','scope':'SOURCE_LAYER_AND_DEVELOPMENT_REPLAY_NOT_CLINICAL_APPROVAL',
        'source_summary':summary,'historical_context_summary':history,'replay_summary':replay,'original_patients':len(groups['Original50']),
        'original_memory_records':structural['unique_memory_records'],'preserved_review_cohorts':[15,5],
        'structural_retrieval_calls':structural['retrieval_calls'],'structural_status':structural['status'],
        'new_chief_complaints':0,'reviewed_reference_count':0,'clinical_quality':'NOT_RUN','utility_fitting':0,'actual_llm_generations':0}
    write(public/'reports/preparation_result.json',result);return result


def partial_evaluate(root,reference,expected_sha,output):
    if reference is None or expected_sha is None:raise ValueError('REVIEWED_PARTIAL_REFERENCE_REQUIRED')
    root=Path(root).resolve();reference=Path(reference).resolve();output=Path(output).resolve()
    if not root.is_relative_to(ROOT/'outputs/private') or not reference.is_relative_to(ROOT/'outputs/private'):
        raise ValueError('PRIVATE_PATH_REQUIRED')
    if not output.is_relative_to(ROOT/'outputs/research_evaluation'):raise ValueError('PUBLIC_OUTPUT_PATH_REQUIRED')
    if output.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if not reference.is_file() or sha(reference)!=expected_sha:raise ValueError('REFERENCE_HASH_REQUIRED')
    m=load(root/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='fingerprint'})!=m['fingerprint']:raise ValueError('MANIFEST_FINGERPRINT')
    for p,h in m['source_hashes'].items():
        if sha(p)!=h:raise ValueError('SOURCE_HASH_MISMATCH')
    review=load(reference)
    expected_ids=['OriginalCase%03d'%i for i in range(1,51)]
    if [r['snapshot_id'] for r in m['packets']]!=expected_ids:raise ValueError('ORIGINAL50_COHORT_REQUIRED')
    if review.get('manifest_fingerprint')!=m['fingerprint'] or review.get('schema_version')!=PARTIAL_VERSION:raise ValueError('REFERENCE_MANIFEST_BINDING')
    if [r['snapshot_id'] for r in review['snapshots']]!=[r['snapshot_id'] for r in m['packets']]:raise ValueError('ALL50_REFERENCE_COHORT_REQUIRED')
    original=load_cases(DATA)
    if len(original)!=50:raise ValueError('ORIGINAL50_REQUIRED')
    validated=[]
    for e,r,c in zip(m['packets'],review['snapshots'],original):
        if Path(e['file']).name!=e['file'] or sha(root/e['file'])!=e['sha256']:raise ValueError('PACKET_HASH')
        packet=load(root/e['file'])
        if digest(packet)!=e['packet_fingerprint']:raise ValueError('PACKET_FINGERPRINT')
        if packet['snapshot_id']!=e['snapshot_id'] or packet['query_candidate'].get('case_ref')!=e['snapshot_id']:
            raise ValueError('CASE_REFERENCE_BINDING')
        identity=c['agent_visible']['longitudinal']
        if identity!=packet['private_identity']:raise ValueError('PATIENT_VISIT_REFERENCE_MISMATCH')
        validate_candidate(packet['query_candidate'],c['environment']['current_encounter_events'],identity)
        original_records=[r for v in c['agent_visible']['past_visits'] for r in v['records']]
        eligible=eligible_reference(original_records,identity);by_id={r['memory_id']:r for r in original_records}
        if packet['full_eligible_count']!=len(eligible) or any(r['memory_id'] not in eligible or by_id[r['memory_id']]!=r for r in packet['candidate_memories']):
            raise ValueError('POOL_MEMORY_SOURCE_BOUNDARY')
        judgments=validate_partial_review(packet,r);validated.append((packet,judgments))
    out=[]
    for (packet,judgments),c in zip(validated,original):
        identity=c['agent_visible']['longitudinal']
        if identity!=packet['private_identity']:raise ValueError('PATIENT_VISIT_REFERENCE_MISMATCH')
        records=[r for v in c['agent_visible']['past_visits'] for r in v['records']];store=PatientMemoryStore(records)
        for mode in ('none','full_history','selective'):
            selected=store.retrieve(identity,packet['query_candidate']['query'],mode,5)['records']
            out.append({'snapshot_id':packet['snapshot_id'],'mode':mode,
                **judged_metrics([r['memory_id'] for r in selected],[r['memory_id'] for r in packet['candidate_memories']],judgments)})
    output.mkdir(parents=True);write(output/'reports/judged_metrics.json',{'schema_version':PARTIAL_VERSION,'scope':'PARTIAL_JUDGED_ONLY_NOT_FULL_RECALL',
        'reference_sha256':expected_sha,'rows':out,'clinical_metrics':'NOT_EVALUATED'})
    return {'status':'PARTIAL_JUDGED_DEVELOPMENT_EVALUATED','rows':len(out)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--namespace',default='c4_05_v1');p.add_argument('--use-captured-baseline',action='store_true')
    e=sub.add_parser('partial-evaluate');e.add_argument('--packets',required=True);e.add_argument('--reference',required=True);e.add_argument('--reference-sha256',required=True);e.add_argument('--output-namespace',required=True)
    a=parser.parse_args()
    def guard(event,values):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C4_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    if a.command=='prepare':result=prepare(a.namespace,a.use_captured_baseline)
    else:
        if not re.fullmatch('[A-Za-z0-9_-]{1,64}',a.output_namespace):raise ValueError('INVALID_NAMESPACE')
        result=partial_evaluate(a.packets,a.reference,a.reference_sha256,ROOT/'outputs/research_evaluation'/a.output_namespace)
    print(json.dumps(result,ensure_ascii=True))
