"""D4 blinded source-reference review preparation. No clinical judgments."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from app.locdi.llm_setup import ROOT,digest
from app.locdi.dataset_suitability import COHORT,sha,millis,reference_timing
from app.locdi.source_replay import PUBLIC as D3_PUBLIC,PRIVATE as D3_PRIVATE,atomic_rows,iso
from validation.source_replay_validator import validate_source_bundle
from validation.routing_correctness_gate import empty_review_results,POLICY

FORBIDDEN={'agent_reviews','agent_response','agent_responses','response_reviews','responses','llm_audit','evaluation','gold','task_annotations'}


def empty_reference_form():
    return {'reference_timing_status':'REFERENCE_TIMING_NOT_REVIEWED','reference_diagnosis':None,
        'actual_available_at':None,'current_decision_reference_valid':None,
        'information_sufficiency':None,'clinical_evidence_rationale':None,
        'review_status':'NOT_REVIEWED','reviewer':None,'reviewer_independent':None,
        'reviewed_at':None,'clinical_verdict':None,'procedure_version':POLICY['reference_procedure_version']}


def validate_blinded_packet(packet):
    def walk(x):
        if isinstance(x,dict):
            if set(x)&FORBIDDEN:raise ValueError('REVIEW_NOT_BLINDED')
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(packet)
    if packet.get('clinical_gold_approved',False) is not False:raise ValueError('REVIEW_PROMOTION_FORBIDDEN')
    if packet['review_form']!=empty_reference_form():raise ValueError('REVIEW_PROMOTION_FORBIDDEN')
    if packet['agent_responses_hidden'] is not True:raise ValueError('REVIEW_NOT_BLINDED')
    return True


def source_path(ref):
    prefix='experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1/'
    if not ref.startswith(prefix):raise ValueError('SOURCE_PREFIX')
    path=(COHORT/ref[len(prefix):]).resolve()
    if not path.is_relative_to(COHORT.resolve()):raise ValueError('SOURCE_ESCAPE')
    return path


def json_pointer(document,pointer):
    current=document
    for key in pointer.lstrip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~');current=current[int(key)] if isinstance(current,list) else current[key]
    return current


def verify_d3():
    manifest=json.loads((D3_PUBLIC/'dataset_manifest.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in manifest.items() if k!='manifest_fingerprint'})!=manifest['manifest_fingerprint']:raise ValueError('D3_MANIFEST_HASH')
    private={}
    for name,entry in manifest['files'].items():
        p=(ROOT/entry['path']).resolve()
        if not p.is_relative_to(D3_PRIVATE.resolve()) or sha(p)!=entry['sha256']:raise ValueError('D3_FILE_HASH')
        private[name]=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] if name.endswith('.jsonl') else json.loads(p.read_text(encoding='utf-8'))
    for name,expected in manifest['source_hashes'].items():
        p=(COHORT/name).resolve()
        if not p.is_relative_to(COHORT.resolve()) or sha(p)!=expected:raise ValueError('D3_ORIGINAL_CSV_HASH')
    validation=validate_source_bundle({'private':private})
    documents={};symptom_rows=[]
    for initial in private['initial_observations.jsonl']:
        p=source_path(initial['json_source']['source_file'])
        if sha(p)!=initial['json_source']['file_sha256']:raise ValueError('D3_JSON_HASH')
        document=json.loads(p.read_text(encoding='utf-8'));documents[initial['case_id']]=document
        for s in initial['encoded_symptom_candidates']:
            source=json_pointer(document,s['json_pointer'])
            valid=(source['value']==s['value'] and iso(source['time'])==s['source_recorded_at']
                   and source['time']<=millis(initial['decision_time']) and '/timeInfos/' in s['json_pointer'])
            if not valid:raise ValueError('ENCODED_SYMPTOM_SOURCE_MISMATCH')
            symptom_rows.append({'case_id':initial['case_id'],'patient_id':initial['patient_id'],
                'candidate':deepcopy(s),'json_source':deepcopy(initial['json_source']),
                'source_time_verified':True,'asof_state_verified':True,'final_state_used':False,
                'semantic_status':'ENCODED_SIMULATOR_STATE_NOT_VERIFIED_CURRENT_PATIENT_REPORT',
                'patient_report_status':'NOT_REVIEWED','clinical_availability':'NOT_REVIEWED',
                'current_symptom_approved':False})
    return manifest,private,validation,symptom_rows


def make_packet(version,ordinal,*,patient_id,visit_id,decision_time,initial,memory,conditions,actions,ledger,source_hashes):
    packet={'schema_version':'locdi.D4.blinded_reference.v1','dataset_version':version,'case_ordinal':ordinal,
        'private_identity':{'patient_id':patient_id,'visit_id':visit_id},'decision_time':decision_time,
        'encounter_start':decision_time,'current_initial_projection':deepcopy(initial),
        'initial_projection_clinical_review':'NOT_REVIEWED','memory_candidates':deepcopy(memory),
        'source_condition_candidates':deepcopy(conditions),'source_action_candidates':deepcopy(actions),
        'source_visibility_ledger':deepcopy(ledger),'source_hashes':deepcopy(source_hashes),
        'agent_responses_hidden':True,'clinical_gold_approved':False,'review_form':empty_reference_form()}
    validate_blinded_packet(packet);packet['source_packet_fingerprint']=digest(packet);return packet


def prepare_internal(*,write=False):
    manifest,p,validation,symptoms=verify_d3()
    legacy_path=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
    legacy=[json.loads(s) for s in legacy_path.read_text(encoding='utf-8').splitlines()]
    packets=[];summary={};warnings=[];source_hashes=manifest['source_hashes']
    for version,cases in [('longitudinal_benchmark_50',legacy),('simulation_source_replay_v1',p['cases.jsonl'])]:
        references=Counter();counts=Counter();initial_counts=Counter()
        for ordinal,c in enumerate(cases,1):
            if version=='longitudinal_benchmark_50':
                a=c['agent_visible'];pid=c['patient_id'];visit=a['index_encounter_id'];decision=a['index_encounter_start']
                initial=deepcopy(a['current_initial_state']);memory=[r for v in a['past_visits'] for r in v['records']]
                conditions=[]
                for source in c['provenance']['source_case_audit']['diagnosis_sources']:
                    r=source['source_record'];relation=reference_timing(r['START'],decision)
                    conditions.append({'source':deepcopy(source),'condition_code':r['CODE'],
                        'condition_description':r['DESCRIPTION'],'source_condition_start':r['START'],
                        'timing_status':relation,'independent_review_status':'NOT_REVIEWED','clinical_gold_status':'UNAPPROVED'})
                actions=deepcopy(c['environment']['current_encounter_events']);ledger=[]
                for event in c['environment']['source_event_store']:
                    match=next((r for r in actions if r['event_id']==event['event_id']),{})
                    ledger.append({'source':deepcopy(event),'recorded_at':event.get('timestamp'),
                        'timestamp_precision':event.get('timing_precision'),'actual_request_at':None,'actual_receipt_at':None,
                        'actual_availability_proof':None,'proxy_available_minute':match.get('available_minute'),
                        'review_status':'NOT_REVIEWED'})
                h={'wrapper_sha256':sha(legacy_path),'original_csv_sha256':source_hashes}
            else:
                pid=c['patient_id'];visit=c['visit_id'];decision=c['decision_time'];cid=c['case_id']
                current=next(i for i in p['initial_observations.jsonl'] if i['case_id']==cid)
                initial={'doctor_visible_info':{'chief_complaint':current['chief_complaint'],'initial_vitals':current['initial_vitals'],
                    'demographics':current['demographics'],'objective_findings':current['objective_findings']},
                    'patient_visible_info':{'associated_symptoms':current['current_symptoms']}}
                memory=[r for r in p['memory.json']['records'] if r['memory_id'] in c['memory_ids']]
                conditions=[r for r in p['reference_candidates.jsonl'] if r['case_id']==cid]
                actions=[r for r in p['action_events.jsonl'] if r['case_id']==cid]
                ledger=[r for r in p['source_visibility_ledger.jsonl'] if r['case_id']==cid]
                h={'dataset_manifest_fingerprint':manifest['manifest_fingerprint'],'source_json':current['json_source'],
                    'original_csv_sha256':source_hashes}
            doctor=initial['doctor_visible_info'];patient=initial['patient_visible_info']
            initial_counts.update({k:int(bool(v)) for k,v in {'chief_complaint':doctor.get('chief_complaint'),
                'initial_vitals':doctor.get('initial_vitals'),'demographics':doctor.get('demographics'),
                'objective_findings':doctor.get('objective_findings'),'projected_symptom_candidates':patient.get('associated_symptoms')}.items()})
            for ref in conditions:
                relation=ref['timing_status'];references[relation]+=1
                if relation=='AFTER_ENCOUNTER_DATE':warnings.append({'dataset_version':version,'case_ordinal':ordinal,
                    'warning':'SOURCE_CONDITION_AFTER_DECISION_DATE','reference_timing_status':'REFERENCE_TIMING_NOT_REVIEWED'})
            counts.update({'cases':1,'reference_candidates':len(conditions),'memory_candidates':len(memory),
                'ask_candidates':sum(r.get('action')=='Ask' for r in actions),'test_candidates':sum(r.get('action')=='Test' for r in actions)})
            packet=make_packet(version,ordinal,patient_id=pid,visit_id=visit,decision_time=decision,initial=initial,
                memory=memory,conditions=conditions,actions=actions,ledger=ledger,source_hashes=h)
            packets.append(packet)
        summary[version]={**dict(counts),'initial_projected_coverage':dict(initial_counts),
            'reference_date_relations':dict(references),'clinically_reviewed_references':0,'clinical_gold':0}
    previous=ROOT/'outputs/private/routing_collection_review/phase24c_review_v2/review_packet.json'
    routing=json.loads(previous.read_text(encoding='utf-8'))
    candidate=next(r for r in routing['items'] if r['case_ref']=='Candidate010')
    candidate_packet={'schema_version':'locdi.D4.routing_reference_blinded.v1','case_ref':'Candidate010',
        'private_decision':deepcopy(candidate['private_decision']),'source_condition_candidates':deepcopy(candidate['reference_candidates']),
        'visible_snapshot':deepcopy(candidate['agent_visible_snapshot']),'visibility_ledger':deepcopy(candidate['visibility_ledger']),
        'source_packet_hash':sha(previous),'agent_responses_hidden':True,'review_form':empty_reference_form()}
    validate_blinded_packet(candidate_packet)
    report={'schema_version':'locdi.D4.internal_reference_readiness.v1','status':'PREPARED_NOT_REVIEWED',
        'd3_manifest_fingerprint':manifest['manifest_fingerprint'],'d3_validation':validation,
        'd3_original_csv_hash_checks':len(source_hashes),'d3_json_hash_checks':len(p['cases.jsonl']),
        'datasets':summary,'date_warnings':warnings,'routing_candidate010':'REFERENCE_TIMING_NOT_REVIEWED',
        'encoded_symptom_cases':len({r['case_id'] for r in symptoms}),'encoded_symptom_items':len(symptoms),
        'encoded_symptom_source_verified':len(symptoms),'encoded_symptom_clinically_approved':0,
        'd3_action_count':len(p['action_events.jsonl']),'d3_actual_request_time_known':0,
        'd3_actual_receipt_time_known':0,'d3_proxy_minutes':dict(Counter(r['proxy_available_minute'] for r in p['action_events.jsonl'])),
        'strict_source':'BLOCKED_ACTUAL_AVAILABILITY_UNVERIFIED','clinical_review_ready':False,
        'training_readiness':'BLOCKED','new_llm_generations':0,'utility_fit_count':0,'clinical_verdict_count':0}
    if write:
        destination=ROOT/'outputs/private/reference_review/D4_reference_v1'
        if destination.exists():raise FileExistsError('NO_REVIEW_OUTPUT_REUSE')
        destination.mkdir(parents=True,exist_ok=False)
        for packet in packets:
            atomic_rows(destination,packet['dataset_version']+'_case{:03d}.json'.format(packet['case_ordinal']),packet,jsonl=False)
        atomic_rows(destination,'routing_Candidate010_blinded.json',candidate_packet,jsonl=False)
        atomic_rows(destination,'encoded_symptom_audit.jsonl',symptoms)
        visibility=[{**deepcopy(r),'actual_request_at':None,'actual_receipt_at':None,'actual_availability_proof':None,
            'time_status':'DEVELOPMENT_PROXY_ONLY','clinical_review_status':'NOT_REVIEWED'} for r in p['action_events.jsonl']]
        atomic_rows(destination,'action_visibility_audit.jsonl',visibility)
        atomic_rows(destination,'routing_review_results_blank.json',empty_review_results(),jsonl=False)
        files={r.name:{'sha256':sha(r),'path':r.relative_to(ROOT).as_posix()} for r in destination.iterdir()}
        atomic_rows(destination,'manifest.json',{'schema_version':'locdi.D4.private_review_manifest.v1',
            'files':files,'review_status':'NOT_REVIEWED','agent_responses_hidden':True},jsonl=False)
        atomic_rows(ROOT/'outputs/reports','D4_reference_v1.json',report,jsonl=False)
    return report
