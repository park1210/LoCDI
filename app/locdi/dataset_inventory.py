"""D1 read-only inventory. Public output has counts, not identities/evidence.

Existing labels/proxies/default review worksheets never become clinical Gold.
Scope: project data plus explicitly referenced adjacent Synthea/Hospital data.
"""
from collections import Counter
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3

from app.locdi.llm_setup import ROOT, digest
from app.locdi.dataset_loader import load_cases
from app.locdi.case_adapter import adapt_case
from app.core.hospital_case_runtime import HospitalCaseRuntime
from app.core.clinical_observation import doctor_observation, encounter_evidence
from app.core.longitudinal_memory import timestamp
from app.locdi.routing_data_reuse import (SYNTHEA, EXPERIMENTS, BANK, SOURCE, DATABASE, HISTORICAL, OSCE,
                                        HOSPITAL, read, audit_existing_data)
from app.locdi.routing_training_setup import HOLDOUT
from app.locdi.evaluation_gate_audit import audit_actual_gate
from app.locdi.development_features import check_development_features
from validation.run_validation import validate_dataset

MISSING = object()


def value_state(value):
    if value is MISSING: return 'MISSING'
    if value is None: return 'NULL'
    if isinstance(value,str) and not value.strip(): return 'EMPTY_STRING'
    if isinstance(value,list) and not value: return 'EMPTY_ARRAY'
    if isinstance(value,dict) and not value: return 'EMPTY_OBJECT'
    return 'POPULATED'


def coverage(name, values, *, linked=None, temporal=None, proxy=None, reviewed=None):
    n=len(values); counts=Counter(value_state(v) for v in values)
    present=[value_state(v)=='POPULATED' for v in values]
    def flagged(flags):
        return sum(a and b for a,b in zip(present,flags)) if flags is not None else 'UNKNOWN'
    return {'field':name,'denominator_cases':n,'populated_cases':sum(present),
        'empty_cases':n-sum(present),'empty_breakdown':{k:v for k,v in counts.items() if k!='POPULATED'},
        'populated_percent':100*sum(present)/n if n else 'NOT_APPLICABLE',
        'source_linked_cases':flagged(linked),'recorded_time_structurally_checked_cases':flagged(temporal),
        'development_proxy_cases':flagged(proxy),'clinically_reviewed_cases':flagged(reviewed),
        'unusable_due_to_empty_cases':n-sum(present)}


def source_references(value):
    if isinstance(value,dict):
        if {'source_file','source_row','source_record'} <= set(value):
            yield value
        for v in value.values():yield from source_references(v)
    elif isinstance(value,list):
        for v in value:yield from source_references(v)


def checked_csv_links(cases):
    wanted={}; expected={}
    for case in cases:
        for ref in source_references(case):
            key=(ref['source_file'].replace('\\','/'),ref['source_row'])
            expected[key]=ref['source_record'];wanted.setdefault(key[0],set()).add(key[1])
    actual={};missing=[]
    for name, rows in wanted.items():
        path=(SYNTHEA/name).resolve()
        if SYNTHEA.resolve() not in path.parents or not path.is_file():
            missing.append(name);continue
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for i,row in enumerate(csv.DictReader(stream),2):
                if i in rows:actual[name,i]=row
    return {k:actual.get(k)==v for k,v in expected.items()},missing


def refs_linked(value, links):
    refs=list(source_references(value))
    return bool(refs) and all(links.get((r['source_file'].replace('\\','/'),r['source_row']),False) for r in refs)


def analyze_locdi(cases, *, csv_links=None, original_memory=None):
    """Pure aggregation; original_memory is an optional independently read index."""
    n=len(cases);links=csv_links or {};fields={};flags={};counts=Counter();per_case=[]
    question_types=set();test_types=set();memory_ids=set();past_visits=set();all_visits=set();source_events=set()
    diagnosis_codes=set();source_patients=set();arrival_classes=Counter()
    identity_pairs=[];initial_boundaries=[]
    def add(name,value,link=False,time=False,proxy=False,review=False):
        fields.setdefault(name,[]).append(value)
        for key,val in [('linked',link),('temporal',time),('proxy',proxy),('reviewed',review)]:
            flags.setdefault((name,key),[]).append(val)
    for ordinal,c in enumerate(cases,1):
        av,env,ann=c['agent_visible'],c['environment'],c['task_annotations'];audit=c['provenance']['source_case_audit']
        start=timestamp(av['index_encounter_start']);doctor=av['current_initial_state']['doctor_visible_info']
        reviewed=audit.get('clinical_review_status')=='expert_reviewed'
        patient=c['patient_id'];visit=av['index_encounter_id'];all_visits.add((patient,visit));identity_pairs.append((patient,visit))
        source_patients.add(patient);sources={e['event_id']:e for e in env['source_event_store']}
        events=env['current_encounter_events'];kinds={k:[e for e in events if e['action']==k] for k in ('Ask','Test','Consult')}
        symptoms=audit['symptom_sources'];diag=audit['diagnosis_sources'];arrival_classes[doctor.get('arrival_info',{}).get('encounter_class','UNKNOWN')]+=1
        diagnosis_codes.update((r['source_record'].get('SYSTEM'),r['source_record'].get('CODE')) for r in diag)
        offsets=[(date.fromisoformat(r['source_record']['START'])-start.date()).days for r in diag]
        counts['diagnosis_source_after_encounter_cases']+=any(x>0 for x in offsets)
        counts['diagnosis_same_patient_encounter_cases']+=all(r['source_record'].get('PATIENT')==patient and r['source_record'].get('ENCOUNTER')==visit for r in diag)
        counts['diagnosis_date_only_rows']+=len(diag)
        add('patient_id',patient,link=refs_linked(diag,links),review=reviewed)
        add('visit_id',visit,link=refs_linked(diag,links),time=True,review=reviewed)
        add('encounter_start',av['index_encounter_start'],link=refs_linked(diag,links),time=True,proxy=True,review=reviewed)
        add('observed_arrival_time',doctor.get('arrival_info',{}).get('arrival_time',MISSING))
        for key in ('chief_complaint','initial_vitals','objective_findings','history'):
            add('initial.'+key,doctor.get(key,MISSING),review=reviewed)
        add('initial.demographics',doctor.get('demographics',MISSING),review=reviewed)
        add('initial.symptom_labels',av['current_initial_state']['patient_visible_info'].get('associated_symptoms',MISSING),
            link=refs_linked(symptoms,links),time=True,proxy=True,review=reviewed)
        add('initial.safe_initial_observations',doctor.get('arrival_info',{}).get('safe_initial_observations',MISSING),review=reviewed)
        add('initial.clinical_information_need',doctor.get('arrival_info',{}).get('clinical_information_need',MISSING),review=reviewed)
        records=[r for v in av['past_visits'] for r in v['records']]
        counts['past_visit_references']+=len(av['past_visits']);counts['memory_record_references']+=len(records)
        valid_memory=True
        for v in av['past_visits']:
            past_visits.add((patient,v['visit_id']));all_visits.add((patient,v['visit_id']))
            counts['past_visit_end_missing']+=v.get('ended_at') is None
            for r in v['records']:
                memory_ids.add(r['memory_id']);counts['memory_supersedes_links']+=len(r['supersedes'])
                good=(r['patient_id']==patient and r['visit_id']==v['visit_id']!=visit
                    and timestamp(r['visit_started_at'])==timestamp(v['started_at'])<start
                    and timestamp(r['visit_started_at'])<=timestamp(r['observed_at'])<=timestamp(r['available_at'])<=start)
                valid_memory &= good;counts['memory_time_or_identity_violations']+=not good
                if original_memory is not None:
                    counts['memory_matches_upstream_record']+=original_memory.get(r['memory_id'])==r
        for name,vals in [('past_visits',av['past_visits']),('ids',[r['memory_id'] for r in records]),
                          ('observed_at',[r['observed_at'] for r in records]),('available_at',[r['available_at'] for r in records])]:
            add('memory.'+name,vals,link=all(original_memory.get(r['memory_id'])==r for r in records) if records and original_memory is not None else False,
                time=valid_memory,proxy=True,review=False)
        for key in ('relevant_memory_ids','stale_memory_ids','conflict_memory_ids'):
            add('annotation.'+key,ann['gold'].get(key,MISSING))
        for kind in ('Ask','Test','Consult'):
            es=kinds[kind];answers=[];record_times=[];good_links=[]
            for e in es:
                payload=json.loads(e['evidence']);entries=payload.get('records',payload.get('facts',[]))
                answers.extend(entries);record_times.extend(x.get('recorded_at') for x in entries)
                counts[kind.lower()+'_events']+=1;counts[kind.lower()+'_evidence_records']+=len(entries)
                if kind=='Ask':
                    question_types.add(e['target']);good_links.append(refs_linked(symptoms,links))
                else:
                    if kind=='Test':test_types.add(e['target'])
                    good_links.append(e['event_id'] in sources and refs_linked(sources[e['event_id']],links))
                counts[kind.lower()+'_linked_events']+=good_links[-1]
                counts[kind.lower()+'_recorded_after_projected_release']+=any(timestamp(x['recorded_at'])>start+timedelta(minutes=e['available_minute']) for x in entries)
            add(kind.lower()+'.events',es,link=bool(es) and all(good_links),time=True,proxy=not env['data_replay']['observed_release_time'],review=reviewed)
            add(kind.lower()+'.results',answers,link=bool(es) and all(good_links),time=True,proxy=not env['data_replay']['observed_release_time'],review=reviewed)
            add(kind.lower()+'.recorded_at',record_times,link=bool(es) and all(good_links),time=all(record_times),proxy=True,review=reviewed)
        add('reference.condition',diag,link=refs_linked(diag,links),time=False,review=reviewed)
        add('reference.diagnosis_codes',[r['source_record'].get('CODE') for r in diag],link=refs_linked(diag,links),time=False,review=reviewed)
        add('reference.condition_start',[r['source_record'].get('START') for r in diag],link=refs_linked(diag,links),time=True,proxy=True,review=reviewed)
        add('reference.independent_gold',ann['gold'].get('accepted_diagnoses',MISSING),review=reviewed)
        for key,value in ann['metric_readiness'].items():counts['metric_ready_'+key]+=value is True
        counts['clinical_reviewed_cases']+=reviewed;counts['actual_release_observed_cases']+=env['data_replay']['observed_release_time'] is True
        for e in env['source_event_store']:source_events.add(e['event_id'])
        context=adapt_case(c)
        runtime=HospitalCaseRuntime(context.case_id,context.runtime_case,0,0)
        observed=doctor_observation(runtime)
        # Static projection check only; never run Scheduler/model/generation.
        forbidden={'gold','evaluation','provenance','task_annotations','source_event_store','current_encounter_events','hidden_test_results'}
        def keys(x):
            if isinstance(x,dict):return set(x)|set().union(*(keys(v) for v in x.values()),set())
            if isinstance(x,list):return set().union(*(keys(v) for v in x),set())
            return set()
        counts['initial_projection_private_key_violations']+=bool(keys(observed)&forbidden)
        counts['initial_projected_test_results']+=len(observed.get('objective_findings',[]))
        boundary=True
        for e in kinds['Test']:
            runtime.current_time_min=e['available_minute']-1
            boundary &= encounter_evidence(runtime,'Test',e['target']) is None
            runtime.current_time_min=e['available_minute']
            boundary &= encounter_evidence(runtime,'Test',e['target'])==e
        initial_boundaries.append(boundary)
        per_case.append({'case_ordinal':ordinal,'past_visit_count':len(av['past_visits']),'memory_records':len(records),
            'ask_events':len(kinds['Ask']),'test_events':len(kinds['Test']),'consult_events':len(kinds['Consult']),
            'source_events':len(sources),'reference_condition_start_after_encounter':any(x>0 for x in offsets),
            'memory_recorded_cutoff_valid':valid_memory,'release_boundary_contract_valid':boundary,
            'adapter_status':'STRUCTURAL_PASS_CLINICAL_NOT_REVIEWED'})
    return {'case_count':n,'unique_patients':len(source_patients),'unique_index_encounters':len(set(identity_pairs)),
        'duplicate_patient_index_encounter_rows':len(identity_pairs)-len(set(identity_pairs)),
        'unique_past_patient_encounters':len(past_visits),'unique_total_patient_encounters':len(all_visits),
        'unique_memory_ids':len(memory_ids),'unique_source_event_ids':len(source_events),
        'distinct_ask_question_templates':len(question_types),'distinct_test_types':len(test_types),
        'distinct_declared_diagnosis_codes':len(diagnosis_codes),'encounter_class_counts':dict(arrival_classes),
        'coverage':[coverage(name,vals,**{k:flags[name,k] for k in ('linked','temporal','proxy','reviewed')}) for name,vals in fields.items()],
        'counts':dict(counts),'cases':per_case,'release_boundary_pass_cases':sum(initial_boundaries),
        'clinical_evaluation_usable_cases':0,'clinical_evaluation_status':'BLOCKED_INDEPENDENT_GOLD_NOT_REVIEWED'}


def _hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _path(path):
    for base,label in ((ROOT,'PROJECT'),(SYNTHEA,'SYNTHEA'),(HOSPITAL,'HOSPITAL')):
        try:return label+'/'+path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:pass
    return 'UNKNOWN_OUTSIDE_DECLARED_SCOPE'


def inventory_entry(path, *, name, rows='UNKNOWN', patients='UNKNOWN', encounters='UNKNOWN',
                    version='UNKNOWN', role='UNKNOWN', origin='UNKNOWN', method='UNKNOWN', overlap='UNKNOWN',
                    status='AVAILABLE_UNREVIEWED_CLINICAL', duplicates='UNKNOWN'):
    return {'dataset_name':name,'dataset_version':version,'source_path':_path(path),'format':path.suffix,
        'file_status':'FOUND' if path.is_file() else 'NOT_FOUND','rows':rows,'unique_patients':patients,
        'encounters':encounters,'duplicates':duplicates,'origin':origin,'transformation':method,
        'clinical_review':'NOT_REVIEWED','current_purpose':role,'split_role':role,'holdout_patient_overlap':overlap,
        'license':'UNKNOWN','access_restriction':'LOCAL_PRIVATE_NOT_APPROVED_FOR_REDISTRIBUTION',
        'research_use':'BLOCKED_PENDING_REFERENCE_LICENSE_AND_SCOPE_REVIEW','status':status,
        'bytes':path.stat().st_size if path.is_file() else None,'sha256':_hash(path) if path.is_file() and path.stat().st_size<200_000_000 else 'NOT_COMPUTED_LARGE_SOURCE'}


def audit_episode_orders(documents):
    """Public log evidence only, not reconstruction of unsaved model payloads."""
    groups={}
    for mode,doc in documents:
        out=groups.setdefault(mode,{'episodes':0,'test_results':0,'before_order_violations':0,'unlinked_results':0})
        out['episodes']+=1;ordered={}
        for event in doc['workflow']['events']:
            target=event.get('target_ref');minute=event['minute']
            if event['event_type']=='test_ordered':ordered[target]=minute
            elif event['event_type']=='result_received':
                out['test_results']+=1
                if target is None or target not in ordered:out['unlinked_results']+=1
                elif minute<ordered[target]:out['before_order_violations']+=1
    return groups


def audit_saved_boundaries():
    cases=load_cases(HOLDOUT);mentions=0
    for c in cases:
        runtime=HospitalCaseRuntime(c['case_id'],adapt_case(c).runtime_case,0,0)
        visible=json.dumps(doctor_observation(runtime),ensure_ascii=False).casefold()
        mentions+=any(label.casefold() in visible for label in c['evaluation']['diagnosis_labels'])
    docs=[]
    for p in sorted((ROOT/'outputs/episodes').glob('*.json')):
        d=read(p);docs.append(('mock' if d['mock_execution'] else 'real_llm',d))
    for p in sorted((ROOT/'outputs/test_only').glob('*/episodes/*.json')):
        docs.append(('TEST_ONLY',read(p)))
    preparation=read(ROOT/'outputs/private/routing_collection/phase24b_candidates_v2/preparation.json')
    split=preparation['split_manifest']['patients']
    voi_train=set(read(BANK.parent/'voi_model.json')['provenance']['training_patients'])
    return {'schema_version':'locdi.dataset_boundaries.D1.v1','initial_projection_cases':len(cases),
        'initial_declared_label_text_mentions':mentions,'saved_episode_order_checks':audit_episode_orders(docs),
        'cross_module_overlap':{'routing_train_in_voi_train':len(set(split['train'])&voi_train),
            'routing_validation_in_voi_train':len(set(split['validation'])&voi_train),
            'holdout_in_voi_train':len(set(split['test'])&voi_train),
            'shared_independent_validation_status':'NOT_ESTABLISHED'},
        'full_llm_wire_payload':'NOT_CHECKED_NOT_SAVED','new_llm_generations':0,'utility_fit_count':0,
        'clinical_gold_generated':0,'clinical_metrics':'NOT_EVALUATED'}


def audit_inventory():
    cases=load_cases(HOLDOUT);patients={c['patient_id'] for c in cases}
    links,missing=checked_csv_links(cases)
    upstream=read(BANK.parent/'memory.json')['records'];memory_index={r['memory_id']:r for r in upstream}
    detail=analyze_locdi(cases,csv_links=links,original_memory=memory_index)
    validation=validate_dataset(HOLDOUT)
    detail['validators']={'counts':validation.counts,'issue_counts':validation.issue_counts,'simulation_readiness':validation.simulation_readiness}
    declared={}
    for c in cases:
        for name,h in c['provenance']['source_sha256'].items():declared.setdefault(name.replace('\\','/'),set()).add(h)
    detail['declared_original_hashes']=[{'path':'SYNTHEA/'+name,'status':'NOT_FOUND' if not (SYNTHEA/name).is_file() else
        'MATCH' if len(hs)==1 and _hash(SYNTHEA/name) in hs else 'MISMATCH'} for name,hs in declared.items()]
    detail['csv_link_check']={'unique_references':len(links),'matches':sum(links.values()),'missing_files':missing,
                             'clinical_truth_verified':False}
    reused=audit_existing_data()
    gate=audit_actual_gate()
    features=check_development_features('bayesian_voi')
    bank=read(BANK);bank_patients={r['training_patient_id'] for r in bank}
    inventory=[inventory_entry(HOLDOUT,name='LoCDI longitudinal wrapper',rows=len(cases),patients=len(patients),
        encounters=detail['unique_index_encounters'],version=','.join(sorted({c['schema_version'] for c in cases})),
        role='TEST_HOLDOUT_DEVELOPMENT_EXPOSED',origin='Synthea experiment 34 + experiment 30 audit',
        method='existing wrapper builder UNKNOWN; declared source hash checked',overlap=len(patients),duplicates=0),
        inventory_entry(BANK,name='Synthea Training Bank',rows=len(bank),patients=len(bank_patients),encounters=reused['bank']['unique_visits'],
        version=BANK.parent.name,role='TRAIN_VALIDATION_CANDIDATE_POOL',origin='Synthea conditions/contexts/source observations',
        method='existing prepare_package / source study preparation',overlap=len(bank_patients&patients),duplicates=len(bank)-len(bank_patients)),
        inventory_entry(SOURCE,name='Training source linkage',rows=len(read(SOURCE)),patients=len(bank_patients),
        encounters=reused['bank']['unique_visits'],role='PRIVATE_TRAIN_SOURCE',origin='experiment 30 audit',overlap=len(bank_patients&patients)),
        inventory_entry(BANK.parent/'memory.json',name='Upstream fixed memory',rows=len(upstream),patients=len({r['patient_id'] for r in upstream}),
        encounters=len({(r['patient_id'],r['visit_id']) for r in upstream}),role='TEST_HISTORY',origin='Synthea observations/records',
        overlap=len({r['patient_id'] for r in upstream}&patients),duplicates=len(upstream)-len(memory_index))]
    # All JSON/JSONL data files in the explicitly referenced v29/v30/v34 packages.
    package_roots=[EXPERIMENTS/'29_prepared_study_50/output/locdi_50_study_v1',
        EXPERIMENTS/'30_source_enriched_study_50/output/locdi_50_study_v2',
        EXPERIMENTS/'30_source_enriched_study_50/output/locdi_50_study_v3_qwen',BANK.parent]
    known={i['source_path'] for i in inventory}
    file_manifest=[]
    for root in package_roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.jsonl'}:
                file_manifest.append({'path':_path(path),'bytes':path.stat().st_size,'role':'REFERENCED_ARCHIVE_NOT_MERGED'})
    for root in package_roots:
        for filename in ('cases.json','routing_probes.json','labels.json','gold_departments.json','evaluation_labels.json','evaluation_departments.json','voi_training_rows.json','voi_model.json','annotations.json','memory.json'):
            p=root/filename
            if not p.is_file() or _path(p) in known:continue
            obj=read(p);rs=obj.get('records',[]) if isinstance(obj,dict) and 'records' in obj else obj if isinstance(obj,list) else None
            ids={r.get('patient_id',r.get('longitudinal',{}).get('patient_id')) for r in rs if isinstance(r,dict)}-{None} if rs is not None else set()
            visits={r.get('index_encounter_id',r.get('longitudinal',{}).get('visit_id',r.get('decision_id'))) for r in rs if isinstance(r,dict)}-{None} if rs is not None else set()
            rows=len(rs) if rs is not None else len(obj) if filename in {'labels.json','gold_departments.json','evaluation_labels.json','evaluation_departments.json','annotations.json'} else 'NOT_APPLICABLE'
            inventory.append(inventory_entry(p,name=root.name+'/'+filename,rows=rows,
                patients=len(ids) if ids else 'UNKNOWN',encounters=len(visits) if visits else 'UNKNOWN',
                version=root.name,role='ARCHIVED_DEVELOPMENT_NOT_INDEPENDENT_NEW_DATA',origin='Synthea prior package',
                overlap=len(ids&patients) if ids else 'UNKNOWN',status='DEVELOPMENT_ONLY'))
            known.add(_path(p))
    # Database metadata and review worksheets: no SELECT update or file repair.
    con=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        tables={t:con.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ('contexts','memory','case_memory','sources','events','raw_refs')}
        context_rows=[json.loads(r[0]) for r in con.execute('SELECT payload FROM contexts')]
        context_patients={r['patient_id'] for r in context_rows}
    finally:con.close()
    inventory.append(inventory_entry(DATABASE,name='Source annotation SQLite',rows=tables['contexts'],patients=len(context_patients),
        encounters=tables['contexts'],version='run_v2',role='UNREVIEWED_ANNOTATION_POOL',origin='Synthea cohort context builder',
        overlap=len(context_patients&patients)))
    worksheets=[]
    for p in sorted(DATABASE.parent.glob('*.csv')):
        count=0;reviewers=0;states=Counter()
        with p.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                count+=1;reviewers+=bool(row.get('reviewer','').strip())
                for key in ('decision','label'):
                    if key in row:
                        value=row[key]
                        states[key+':'+(value if value in {'REVIEW','unreviewed','ACCEPT','REJECT','relevant','irrelevant','stale','conflict'} else 'OTHER_OR_EMPTY')]+=1
        worksheets.append({'file':_path(p),'rows':count,'reviewer_nonempty_rows':reviewers,'states':dict(states)})
        inventory.append(inventory_entry(p,name=p.name,rows=count,version='run_v2',role='REVIEW_WORKSHEET_NOT_GOLD',origin='annotation preparation',status='DEVELOPMENT_ONLY'))
    inventory.extend([inventory_entry(HISTORICAL,name='Historical routing outcomes',rows=reused['historical']['rows'],patients=reused['historical']['unique_patients'],
        encounters=reused['historical']['decision_groups'],role='HISTORICAL_INCOMPATIBLE_BINDINGS',origin='HospitalWorkflow replay',overlap=reused['historical']['holdout_overlap'],status='DEVELOPMENT_ONLY'),
        inventory_entry(OSCE,name='OSCE wrapper named _medqa',rows=reused['osce']['rows'],role='UNREVIEWED_SEPARATE_CANDIDATE',origin='UNKNOWN',method='original transformation UNKNOWN; local loader exists')])
    for p in sorted((ROOT/'outputs/private/routing_collection').glob('*/preparation.json')):
        b=read(p);ids={c['decision']['patient_id'] for c in b['candidates']}
        inventory.append(inventory_entry(p,name=p.parent.name,rows=len(b['candidates']),patients=len(ids),encounters=len(b['candidates']),
            version=b.get('schema_version','UNKNOWN'),role='TRAIN10_VALIDATION5_PREPARATION_VERSION_NOT_NEW_PATIENTS',
            origin='Training Bank selected patients',overlap=len(ids&patients),status='DEVELOPMENT_ONLY'))
    real_run=ROOT/'outputs/private/routing_collection/phase24b_collect_001'
    outcomes=[read(p) for p in real_run.glob('Candidate*.json')]
    inventory.append(inventory_entry(real_run/'run_complete.json',name='Specialty collection phase24b_collect_001',rows=len(outcomes),
        patients=len({r['patient_id'] for r in outcomes}),encounters=len({r['decision_id'] for r in outcomes}),
        role='TRAIN_VALIDATION_RESPONSES_UNREVIEWED',origin='real Specialty generation; not correctness',overlap=len({r['patient_id'] for r in outcomes}&patients)))
    evidence=ROOT/'evidence/clinical_evidence.csv'
    with evidence.open(encoding='utf-8-sig',newline='') as stream:evidence_count=sum(1 for _ in csv.DictReader(stream))
    inventory.append(inventory_entry(evidence,name='Clinical evidence worksheet',rows=evidence_count,role='REFERENCE_WORKSHEET',origin='local evidence schema',status='DEVELOPMENT_ONLY'))
    for p in (ROOT/'outputs/private/routing_collection_review/phase24c_review_v2/review_packet.json',
              ROOT/'outputs/private/unresolved_analysis/phase24d_analysis_v1/independent_review_template.json'):
        obj=read(p);rows=obj['items']
        inventory.append(inventory_entry(p,name=p.parent.name+'/'+p.name,rows=len(rows),patients=15,encounters=15,
            version=obj['schema_version'],role='PRIVATE_REVIEW_PREPARATION_NOT_REVIEWED_LABELS',
            origin='same 15 selected Training Bank patients; not new samples',overlap=0,status='DEVELOPMENT_ONLY'))
    for mock in (True,False):
        episodes=[];paths=[]
        for p in sorted((ROOT/'outputs/episodes').glob('*.json')):
            obj=read(p)
            if obj.get('mock_execution') is mock:episodes.append(obj);paths.append(p)
        if paths:
            entry=inventory_entry(paths[0],name='Existing public '+('Mock' if mock else 'real LLM')+' Episodes',rows=len(episodes),
                version=','.join(sorted({x['schema_version'] for x in episodes})),role='DEVELOPMENT_RESULTS_NOT_GOLD',
                origin='existing Runner/Exporter',status='DEVELOPMENT_ONLY')
            entry.update(source_path='PROJECT/outputs/episodes/*.json [mock_execution='+str(mock)+']',
                         sha256='MULTI_FILE_SEE_LOCAL_MANIFEST',bytes=sum(p.stat().st_size for p in paths))
            inventory.append(entry)
    rawroot=SYNTHEA/'experiments/15_symptom_cohort_1000/output/runs/cohort_1000_json_v1/csv'
    raw_csv_tables=[]
    for p in sorted(rawroot.glob('*.csv')):
        ids=set();visits=set();rows=0
        with p.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                rows+=1
                identity=row.get('PATIENT',row.get('PATIENTID'))
                if p.name=='patients.csv':identity=row.get('Id')
                if identity:ids.add(identity)
                if p.name=='encounters.csv' and row.get('Id'):visits.add(row['Id'])
                elif row.get('ENCOUNTER'):visits.add(row['ENCOUNTER'])
        item=inventory_entry(p,name='Raw cohort/'+p.name,rows=rows,patients=len(ids) if ids else 'NOT_APPLICABLE',
            encounters=len(visits) if visits else 'UNKNOWN',version=rawroot.parent.name,
            role='RAW_SOURCE_NOT_ASSIGNED_TO_TRAIN',origin='existing Synthea cohort CSV',
            overlap=len(ids&patients) if ids else 'NOT_APPLICABLE',status='PARTIALLY_AVAILABLE')
        inventory.append(item);raw_csv_tables.append({'file':_path(p),'rows':rows,'unique_patients':len(ids),
            'holdout_overlap':len(ids&patients),'financial_or_organizational_only':p.name in {'claims.csv','claims_transactions.csv','payers.csv','payer_transitions.csv','providers.csv','organizations.csv'}})
    voi=read(BANK.parent/'voi_model.json');provenance=voi['provenance'];train_ids=set(provenance.get('training_patients',[]))
    source_mins=[v for a in provenance.get('empirical_support',{}).values() for v in a.values()]
    # No arbitrary Artifact creation, statistical refitting, or clinical approval.
    voi_summary={'source':_path(BANK.parent/'voi_model.json'),'schema_version':voi['schema_version'],
        'diagnosis_candidates':len(voi['diagnoses']),'prior_present':bool(voi.get('prior')),
        'action_models':len(voi['actions']),'outcome_rows':sum(len(a['outcomes']) for a in voi['actions']),
        'horizon':voi['horizon'],'fit_method_declared':provenance.get('fit_method','UNKNOWN'),
        'training_patients':len(train_ids),'training_visits_declared':provenance.get('training_visits','UNKNOWN'),
        'training_rows_declared':provenance.get('training_rows','UNKNOWN'),'holdout_overlap':len(train_ids&patients),
        'bank_patient_set_matches':train_ids==bank_patients,'review_status':provenance['review_status'],
        'minimum_empirical_cell_support':min(source_mins) if source_mins else 'UNKNOWN',
        'statistical_likelihood_status':'EXISTING_ESTIMATE_NOT_REESTIMATED',
        'utility_scope':provenance.get('utility_scope','UNKNOWN'),'clinical_support':provenance.get('clinical_support','UNKNOWN'),
        'clinical_validated_probability_cells':0,'runtime_feature_check':features}
    # Native current-project files, excluding new D1 exports, are discoverable.
    local_manifest=[{'path':_path(p),'bytes':p.stat().st_size,'scope':'EXISTING_LOCAL_DATA_OR_RESULT'}
        for base in ('datasets','evidence','configs','experiments','outputs/private','outputs/episodes','outputs/metrics','outputs/reports','outputs/test_only')
        for p in sorted((ROOT/base).rglob('*')) if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.sqlite'} and 'D1_inventory' not in p.name]
    return {'schema_version':'locdi.dataset_inventory.D1.v1','scope':'project_and_explicit_referenced_archives_not_all_disks',
        'inventory':inventory,'local_file_manifest':local_manifest,'archive_file_manifest':file_manifest,
        'locdi':detail,'source_sqlite':{'tables':tables,'unique_context_patients':len(context_patients),'holdout_patients_in_context_pool':len(context_patients&patients),
            'bank_patients_in_context_pool':len(context_patients&bank_patients),'worksheets':worksheets},
        'routing_sources':reused,'actual_routing_gate':gate,'voi':voi_summary,
        'raw_csv_tables':raw_csv_tables,
        'evidence_csv_rows':evidence_count,'clinical_review_readiness':'BLOCKED','research_readiness':False,
        'new_llm_generations':0,'utility_fit_count':0,'new_voi_probability_estimates':0,'clinical_gold_generated':0}
