"""C4-02 source tracing, blinded packets and explicitly gated offline evaluation."""
import argparse,csv,json,sys
from pathlib import Path
from datetime import timedelta
from collections import Counter
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import load,sha
from app.locdi.dataset_loader import load_cases
from app.locdi.episode_exporter import _write
from app.locdi.routing_data_reuse import SYNTHEA
from app.locdi.reference_readiness import source_path
from app.core.longitudinal_memory import timestamp
from app.locdi.memory_reference_evaluation import released_query,validate_packet,evaluate,VERSION,PROCEDURE
from scripts.audit_locdi_core_evaluation import eligible_reference

DATA=ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'
D4=ROOT/'outputs/private/reference_review/D4_reference_v1'

def baseline():
    prior=ROOT/'outputs/private/research_evaluation/c4_01_v1/reports'
    files=load(prior/'baseline_hashes.json')
    manifest_path=ROOT/'outputs/research_evaluation/c4_01_v1/reports/output_manifest.json'
    manifest=load(manifest_path)
    index_path=prior/'private_index.json'
    files.update(manifest['files']);files.update(load(index_path)['files'])
    for p in (index_path,manifest_path):files[p.relative_to(ROOT).as_posix()]=sha(p)
    if any(not (ROOT/n).is_file() or sha(ROOT/n)!=h for n,h in files.items()):raise ValueError('PRIOR_PROTECTED_ASSET_CHANGED')
    return files

def source_trace(cases):
    hashes={};rows=[];wanted={}
    for c in cases:
        for name,h in c['provenance']['source_sha256'].items():
            path=(SYNTHEA/name).resolve()
            if not path.is_relative_to(SYNTHEA.resolve()):raise ValueError('UPSTREAM_PATH_ESCAPE')
            if str(path) in hashes and hashes[str(path)]!=h:raise ValueError('DECLARED_HASH_CONFLICT')
            hashes[str(path)]=h
        for s in c['provenance']['source_case_audit']['symptom_sources']:
            ref=s['source_metadata']['_source'];wanted[ref['source_row']]=ref['source_record']
    for name,h in hashes.items():
        if sha(name)!=h:raise ValueError('DECLARED_UPSTREAM_HASH_MISMATCH')
    csv_path=source_path(cases[0]['provenance']['source_case_audit']['symptom_sources'][0]['source_metadata']['_source']['source_file'])
    csv_sha=sha(csv_path);found={}
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        for number,r in enumerate(csv.DictReader(stream),2):
            if number in wanted:found[number]=r
    if set(found)!=set(wanted) or any(found[n]!=r for n,r in wanted.items()):raise ValueError('SOURCE_CSV_ROW_MISMATCH')
    hashes[str(csv_path)]=csv_sha
    for ordinal,c in enumerate(cases,1):
        symptoms=c['provenance']['source_case_audit']['symptom_sources'];reasons=[]
        if len(symptoms)!=1:reasons.append('ONE_REPORTED_MEASUREMENT_REQUIRED')
        if reasons:rows.append({'ordinal':ordinal,'source_candidate':False,'reasons':reasons});continue
        s=symptoms[0];ref=s['source_metadata']['_source'];record=ref['source_record'];v=s['visible_value']
        raw_path=source_path(ref['raw_json_file']);document=load(raw_path);hashes[str(raw_path)]=sha(raw_path)
        encounters=[e for e in document['record']['encounters'] if e['uuid']==ref['raw_encounter_id']];matches=[]
        def walk(x):
            if isinstance(x,dict):
                if x.get('uuid')==ref['raw_observation_uuid']:matches.append(x)
                for child in x.values():walk(child)
            elif isinstance(x,list):
                for child in x:walk(child)
        if len(encounters)==1:walk(encounters[0])
        checks={'source_patient':record['PATIENT']==c['patient_id']==document['attributes']['id'],
            'source_visit':record['ENCOUNTER']==c['agent_visible']['index_encounter_id']==ref['raw_encounter_id'],
            'single_raw_observation':len(matches)==1,'value_present':v.get('value') not in (None,''),
            'value_preserved_in_provenance':v.get('value')==record['VALUE'],
            'declared_visibility':s['visibility_allowed'] is True,
            'report_not_future':timestamp(s['source_timestamp'])<=timestamp(c['agent_visible']['index_encounter_start']),
            'not_a_complaint':not c['agent_visible']['current_initial_state']['doctor_visible_info'].get('chief_complaint')}
        if matches:
            raw=matches[0];checks.update(raw_value=str(raw['value'])==record['VALUE'],raw_time=raw['start']==ref['raw_timestamp'],
                raw_code=any(code.get('code')==record['CODE'] for code in raw['codes']))
        events=[e for e in c['environment']['current_encounter_events'] if e['action']=='Ask']
        checks['ask_value_retained']=len(events)==1 and any(f.get('value')==record['VALUE'] and f.get('code')==record['CODE']
            for f in json.loads(events[0]['evidence']).get('facts',[]))
        candidate=all(value for key,value in checks.items() if key!='not_a_complaint')
        rows.append({'ordinal':ordinal,'source_candidate':candidate,'checks':checks,
            'source':deepcopy(s),'raw_json_sha256':hashes[str(raw_path)],'raw_json_path':str(raw_path),
            'source_event':events[0] if len(events)==1 else None,'source_value_type':'REPORTED_NUMERIC_MEASUREMENT_NOT_CHIEF_COMPLAINT',
            'complaint_status':'MISSING_SOURCE_COMPLAINT','actual_clinical_release_status':'NOT_OBSERVED',
            'source_file':str(csv_path),'source_file_sha256':csv_sha})
    if sha(csv_path)!=csv_sha:raise ValueError('SOURCE_CHANGED_DURING_SCAN')
    return rows,hashes

def prepare(namespace,continue_existing=False):
    public=ROOT/'outputs/research_evaluation'/namespace;private=ROOT/'outputs/private/research_evaluation'/namespace
    if not continue_existing and (public.exists() or private.exists()):raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if continue_existing:
        if (public/'reports/preparation_result.json').exists() or any((private/'review_packets').glob('*')):raise ValueError('PACKET_OUTPUT_EXISTS_NO_RESUME')
        protected=load(private/'reports/baseline_hashes.json')
        if any(sha(ROOT/n)!=h for n,h in protected.items()):raise ValueError('PRIOR_PROTECTED_ASSET_CHANGED')
    else:
        protected=baseline();public.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
        _write(protected,'reports/baseline_hashes.json',private,'reports')
    cases=load_cases(DATA)
    if continue_existing:
        traces=load(private/'reports/source_trace.json');preselection=load(private/'reports/preselection.json');source_hashes=preselection['source_hashes']
        if sha(DATA)!=preselection['source_dataset_sha256'] or any(sha(n)!=h for n,h in source_hashes.items()):raise ValueError('SOURCE_CHANGED_AFTER_AUDIT')
        _write({'status':'INITIAL_PREPARATION_FAILED_BEFORE_PACKET_WRITE','error_code':'SOURCE_FACT_FIELDS',
            'cause':'New validator omitted existing nullable onset/duration/active_status fields; canonical query now strips these and replay-bin metadata',
            'model_generations':0,'existing_files_overwritten':False},'reports/initial_preparation_failure.json',public,'reports')
    else:
        traces,source_hashes=source_trace(cases);_write(traces,'reports/source_trace.json',private,'reports')
    dm=load(D4/'manifest.json')
    if dm.get('agent_responses_hidden') is not True:raise ValueError('D4_NOT_BLINDED')
    d4checks={}
    for name,entry in dm['files'].items():
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('D4_PACKET_HASH_MISMATCH')
        d4checks[name]=entry['sha256']
    # Cohort fixed using source linkage, chronology and presence ONLY, before any search or replay output.
    selected=[r['ordinal'] for r in traces if r['source_candidate']][:15]
    if not continue_existing:_write({'ordinals':selected,'selection_rule':'FIRST_15_SOURCE_VALID_ORIGINAL_ORDINALS_BEFORE_REPLAY_OR_RETRIEVAL',
        'source_dataset_sha256':sha(DATA),'source_hashes':source_hashes,'clinical_eligibility':'NOT_ESTABLISHED'},
        'reports/preselection.json',private,'reports')
    if continue_existing and selected!=preselection['ordinals']:raise ValueError('FIXED_COHORT_CHANGED')
    if not continue_existing:_write({'selected_ordinals':selected,'cohort_count':len(selected),'cohort_use':'REVIEW_CANDIDATES_ONLY_NOT_CLINICALLY_APPROVED'},
        'reports/cohort.json',public,'reports')
    memory_name=next(n for n in cases[0]['provenance']['source_sha256'] if n.endswith('memory.json'))
    original_memory=load(SYNTHEA/memory_name)['records'];memory_index={r['memory_id']:(i,r) for i,r in enumerate(original_memory)}
    # Existing deterministic source-backed client: never generates a diagnosis or source evidence.
    sys.path.insert(0,str(ROOT/'tests/locdi'))
    from test_simulation import SourceActionClient,smoke_options
    from app.locdi.run_simulation import run_case
    packet_root=private/'review_packets';packet_root.mkdir(exist_ok=continue_existing);packet_refs=[];review_rows=[];runtime_rows=[]
    for ordinal in selected:
        c=cases[ordinal-1];trace=traces[ordinal-1];client=SourceActionClient()
        run=run_case(c,options=smoke_options(client,minute_limit=10))
        calls=[call for call in client.calls if call['payload']['observation']['history']]
        if not calls:
            runtime_rows.append({'ordinal':ordinal,'status':'NO_RELEASED_ASK_SNAPSHOT','runtime_status':run['status']});continue
        observed=calls[0]['payload']['observation'];minute=observed['elapsed_minutes']
        query,clean=released_query(observed['history'],snapshot_minute=minute,source_event=trace['source_event'])
        prior=D4/('longitudinal_benchmark_50_case%03d.json'%ordinal);d4=load(prior)
        check=dict(d4);fingerprint=check.pop('source_packet_fingerprint')
        if digest(check)!=fingerprint:raise ValueError('D4_INTERNAL_FINGERPRINT')
        records=[r for v in c['agent_visible']['past_visits'] for r in v['records']]
        if records!=d4['memory_candidates']:raise ValueError('D4_ORIGINAL_MEMORY_MISMATCH')
        identity=c['agent_visible']['longitudinal'];eligible=eligible_reference(records,identity)
        records=sorted([deepcopy(r) for r in records if r['memory_id'] in eligible],key=lambda r:(timestamp(r['available_at']),r['memory_id']))
        record_links=[]
        for r in records:
            i,original=memory_index[r['memory_id']]
            if original!=r:raise ValueError('UPSTREAM_MEMORY_RECORD_MISMATCH')
            record_links.append({'memory_id':r['memory_id'],'upstream_records_index':i,'record_fingerprint':digest(r),
                'artifact_reference':'UPSTREAM_MEMORY_JSON','source_id':r['source_id']})
        sid='Snapshot%03d'%ordinal
        packet={'schema_version':'locdi.memory_blinded_packet.v1','snapshot_id':sid,'case_ordinal':ordinal,
            'private_identity':identity,'decision_minute':minute,
            'decision_time':(timestamp(identity['started_at'])+timedelta(minutes=minute)).isoformat(),
            'memory_cutoff':identity['started_at'],'complaint_status':'MISSING_SOURCE_COMPLAINT',
            'chief_complaint':None,'current_information_type':'REPLAY_RELEASED_PATIENT_REPORTED_PAIN_MEASUREMENT',
            'released_history':deepcopy(observed['history']),'query_input_projection':clean,'released_source_event':trace['source_event'],
            'query':query,'query_fingerprint':digest(query),
            'source_visibility_evidence':{'source_report':trace['source'],'replay_release_minute':minute,
                'physical_clinical_release_observed':False,'release_policy':'RECORDED_TIME_PROXY_PLUS_OBSERVED_OFFLINE_ASK',
                'raw_json_sha256':trace['raw_json_sha256'],'csv_sha256':trace['source_file_sha256']},
            'eligible_memory':records,'memory_provenance':record_links,
            'shared_provenance':{'upstream_memory_path':str((SYNTHEA/memory_name).resolve()),
                'upstream_memory_sha256':sha(SYNTHEA/memory_name),'D4_packet_sha256':sha(prior)},
            'review_burden':{'eligible_records':len(records),'all_eligible_included':True,'truncated':False,
                'source_text_characters':sum(len(r['text']) for r in records)},
            'review_status':'UNREVIEWED','model_prediction_included':False,'retrieval_output_included':False}
        validate_packet(packet);name=sid+'.json'
        # Atomic no-overwrite writer expects a category/name pair.
        _write(packet,'review_packets/'+name,private,'review_packets');packet_refs.append({'snapshot_id':sid,'file':name,'sha256':sha(packet_root/name)})
        review_rows.append({'snapshot_id':sid,'packet_sha256':sha(packet_root/name),'query_fingerprint':digest(query),
            'review_status':'UNREVIEWED','reviewer_private_ref':None,'reviewer_independent':None,'reviewed_at':None,
            'procedure_version':PROCEDURE,'query_suitable':None,'query_visibility_reviewed':None,
            'whole_eligible_pool_reviewed':None,'model_or_retrieval_outputs_seen':None,'visibility_scope':None,
            'recorded_time_proxy_explicitly_reviewed':None,'rationale':None,
            'records':[{'memory_id':r['memory_id'],'relevant':None,'stale':None,'conflict':None,
                        'rationale':None,'not_evaluable_reason':None} for r in records]})
        runtime_rows.append({'ordinal':ordinal,'snapshot_id':sid,'status':'REVIEW_CANDIDATE_ONLY','runtime_status':run['status'],
            'decision_minute':minute,'offline_dispatch_count':len(client.calls),'eligible_records':len(records),
            'future_test_results_in_query':0,'source_value_verified':True,'actual_generation_count':0})
    source_hashes[DATA.relative_to(ROOT).as_posix()]=sha(DATA)
    source_hashes.update({str(D4/name):h for name,h in d4checks.items() if name.startswith('longitudinal_benchmark_50_case')})
    source_hashes[str(D4/'manifest.json')]=sha(D4/'manifest.json')
    manifest={'schema_version':'locdi.memory_review_bundle.v1','packets':packet_refs,'source_hashes':source_hashes,
        'cohort_rule':'FIRST_15_SOURCE_VALID_ORIGINAL_ORDINALS','retrieval_has_run':False,'quality_reference_status':'UNREVIEWED'}
    manifest['fingerprint']=digest(manifest);_write(manifest,'review_packets/manifest.json',private,'review_packets')
    _write({'schema_version':VERSION,'manifest_fingerprint':manifest['fingerprint'],'snapshots':review_rows},
        'review_packets/review_results_blank.json',private,'review_packets')
    _write(runtime_rows,'reports/replay_snapshot_checks.json',public,'reports')
    result={'status':'C4_02_REFERENCE_PREPARED' if packet_refs else 'C4_02_SOURCE_BLOCKED','original_cases':len(cases),
        'source_value_candidate_queries':sum(r['source_candidate'] for r in traces),'replay_snapshot_count':len(packet_refs),
        'clinically_approved_queries':0,'observed_physical_clinical_release_count':0,'missing_source_complaint_count':50,
        'reviewed_annotations':0,'reference_status':'UNREVIEWED','quality_evaluation':'BLOCKED_REVIEWED_REFERENCE_REQUIRED',
        'precision':None,'recall_at_5':None,'stale_retrieval_rate':None,'retrieval_quality_calls':0,
        'missing_annotation_count':sum(len(r['records']) for r in review_rows),'source_hash_count':len(source_hashes),
        'manifest_fingerprint':manifest['fingerprint'],'model_generation_count':0,'utility_fitting_count':0,
        'core_changed_count':0,'clinical_metrics':'NOT_EVALUATED'}
    _write(result,'reports/preparation_result.json',public,'reports')
    changed=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    _write({'protected_count':len(protected),'changed':changed,'changed_count':len(changed)},'reports/preservation.json',private,'reports')
    if changed:raise ValueError('PROTECTED_ASSET_CHANGED')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--namespace',default='c4_02_v1');p.add_argument('--continue-existing-source-audit',action='store_true')
    e=sub.add_parser('evaluate');e.add_argument('--packets',required=True);e.add_argument('--reference');e.add_argument('--reference-sha256')
    e.add_argument('--output-namespace',required=True);args=parser.parse_args()
    namespace=args.namespace if args.command=='prepare' else args.output_namespace
    if not namespace or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in namespace) or len(namespace)>64:parser.error('invalid namespace')
    def guard(event,values):
        if event in ('socket.connect','socket.connect_ex','socket.bind'):raise RuntimeError('C4_OFFLINE_NO_NETWORK')
    sys.addaudithook(guard)
    if args.command=='prepare':result=prepare(namespace,args.continue_existing_source_audit)
    else:
        output=ROOT/'outputs/research_evaluation'/namespace
        if output.exists():raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
        result=evaluate(args.packets,args.reference,args.reference_sha256)
        output.mkdir(parents=True,exist_ok=False);_write(result,'reports/retrieval_quality.json',output,'reports')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
