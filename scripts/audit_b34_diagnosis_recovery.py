"""Private cross-run diagnosis audit and disjoint Native training feasibility.

No model requests, fitting, Source edits or clinical labels.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_terminal_only_replay import PUB,PR
from scripts.run_u1_voi_expansion import load as load_parent
from scripts.run_original50_d0_d2 import load as load_old,M5
from scripts.prepare_u2_source_collection import load as load_static
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.routing_collector import atomic_json
from app.locdi.llm_setup import digest
from app.locdi.routing_data_reuse import BANK,SOURCE,DATABASE,read as source_read
from app.locdi.m5_common_source import verify_csv,_candidate_case_binding
from app.locdi.source_replay import project_wrapper
from app.locdi.case_adapter import adapt_case
from app.core.longitudinal_memory import PatientMemoryStore,timestamp
import sqlite3
from dataclasses import asdict


def save(path,value):
    if Path(path).exists():
        if read(path)!=value:raise FileExistsError('EXISTING_ANALYSIS_DIFFERS')
    else:atomic_json(path,value)


def native_candidate(bank,source,context,connection,model):
    """Fresh wrapper from actual Source facts; never copy another patient's case."""
    pid=bank['training_patient_id'];visit=source['index_encounter'];cutoff=timestamp(visit['start_time'])
    starts={v['encounter_id']:v['start_time'] for v in context['past_encounters']}
    records=[];rejected=Counter()
    for raw, in connection.execute('SELECT m.payload FROM case_memory c JOIN memory m ON c.rid=m.rid WHERE c.cid=?',(context['case_id'],)):
        r=json.loads(raw);v=r['encounter_id']
        if r['patient_id']!=pid or v not in starts or timestamp(starts[v])>=cutoff or timestamp(r['available_at'])>cutoff:
            rejected['IDENTITY_OR_CUTOFF']+=1;continue
        record={'memory_id':r['record_id'],'patient_id':pid,'visit_id':v,'visit_started_at':starts[v],
                'observed_at':r['observed_at_upper_bound'],'available_at':r['available_at'],'source_id':r['record_id'],
                'kind':'objective_finding' if r['type']=='observations' else 'history','text':r['normalized_text'],'supersedes':[]}
        try:PatientMemoryStore([record])
        except ValueError:rejected['MEMORY_SOURCE_CHRONOLOGY']+=1;continue
        records.append(record)
    c={'case_id':'U2Native:'+bank['training_case_id'],'patient_id':pid,'visit_id':visit['encounter_id'],'decision_time':visit['start_time']}
    # Reuse only the validated generic wrapper structure. Empty facts stay empty.
    wrapper=project_wrapper(c,{'encounter_class':visit['encounter_class'],'demographics':context['demographics']},[],[],{})
    past={}
    for r in records:past.setdefault(r['visit_id'],{'visit_id':r['visit_id'],'started_at':r['visit_started_at'],'records':[]})['records'].append(r)
    wrapper['agent_visible']['past_visits']=sorted(past.values(),key=lambda r:(r['started_at'],r['visit_id']))
    events=[{**deepcopy(e),'case_id':c['case_id']} for e in source['source_events'] if e['event_type']=='observations']
    # Patient-reported pain remains Ask. Do not promote it to an objective Test.
    for s in source['symptom_sources']:
        meta=s['source_metadata']['_source'];raw=meta['source_record']
        if raw.get('CODE')!='72514-3':continue
        t=timestamp(raw['DATE']);ms=int(t.timestamp()*1000)
        events.append({'event_id':s['source_id'],'case_id':c['case_id'],'patient_id':pid,'encounter_id':visit['encounter_id'],
            'event_type':'observations','timestamp':raw['DATE'],'timing_precision':'second',
            'timestamp_lower_epoch_ms':ms,'timestamp_upper_epoch_ms':ms+999,
            'target':{'code':raw['CODE'],'description':raw['DESCRIPTION'],'system':None},
            'evidence':{'category':raw['CATEGORY'],'kind':'recorded_observation','value':raw['VALUE'],'units':raw['UNITS']},
            'source_metadata':meta,'availability_status':'UNKNOWN_EXPLICIT_DEVELOPMENT_PROXY',
            'relative_to_decision':'same_precision_interval_uncertain','reveal_status':'ACTION_GATED','action_type_candidate':'Ask'})
    wrapper['environment']['source_event_store']=events
    files={e['source_metadata']['source_file'] for e in events}
    from app.locdi.source_paths import resolve_source_file
    wrapper['provenance']['source_package']='U2_NATIVE_SOURCE_BANK_DECLARED_DEVELOPMENT_PROXY.v1'
    wrapper['provenance']['source_sha256']={f:sha_file(resolve_source_file(f)) for f in files}
    wrapper['provenance']['source_case_audit']['symptom_sources']=deepcopy(source['symptom_sources'])
    binding=_candidate_case_binding(wrapper,model)
    projected=[]
    for r in binding['rows']:
        e=r['source_event'];raw=e['source_metadata']['source_record']
        if r['kind']=='Ask':
            symptom=next(s for s in source['symptom_sources'] if s['source_metadata']['_source']['source_record']==raw)
            delta=(timestamp(symptom['source_timestamp'])-timestamp(raw['DATE'])).total_seconds()
            if not 0<=delta<1:raise ValueError('ASK_SOURCE_TIME_OUTSIDE_RECORDED_SECOND')
            value={'kind':'recorded_patient_report','replay_bin':'UNBINNED_SOURCE_VALUE_NO_VOI',
                'facts':[{'active_status':None,'code':raw['CODE'],'description':raw['DESCRIPTION'],
                    'duration':None,'onset':None,'recorded_at':symptom['source_timestamp'],'units':raw['UNITS'],'value':raw['VALUE']}]}
        else:
            value={'replay_bin':'UNBINNED_SOURCE_VALUE_NO_VOI','records':[{'code':raw['CODE'],
                'description':raw['DESCRIPTION'],'event_type':'observations','recorded_at':raw['DATE'],
                'recorded_evidence':deepcopy(e['evidence']),'system':None,'timing_precision':'second'}]}
        source_target='{} [{}]'.format(raw['DESCRIPTION'],raw['CODE']) if r['kind']=='Test' else r['target']
        projected.append({'event_id':e['event_id'],'action':r['kind'],'target':source_target,
            'available_minute':r['available_minute'],'evidence':json.dumps(value,sort_keys=True)})
    wrapper['environment']['current_encounter_events']=projected
    return wrapper,binding,dict(rejected)


def main():
    PUB.mkdir(parents=True,exist_ok=True);PR.mkdir(parents=True,exist_ok=True)
    _,newpr,newplan,bp=load_parent();newrun=newpr/newplan['run_id']
    _,oldpr,oldplan,oldbp=load_old('original50_d0d2_full_003');oldrun=oldpr/oldplan['run_id']
    old=[r for r in read(oldrun/'analysis_private.json')['rows'] if r['dataset_view']=='D2' and r['condition']=='B2']
    new=[r for r in read(newrun/'analysis_private.json') if r['condition']=='B3'];lookup={r['case_ref']:r for r in new}
    private=[];public=[];protected={}
    for root in (newrun,oldrun):
        protected.update({str(f.resolve()):sha_file(f) for f in root.rglob('*') if f.is_file()})
    for r in old:
        n=lookup[r['case_ref']];a,b=r['facts'],n['facts'];i=n['panel_ordinal']
        folder=oldrun/'requests'/'D2'/('Episode%04d'%r['view_episode_ordinal'])
        if not folder.exists():raise ValueError('OLD_EPISODE_PATH')
        nf=newrun/'requests'/('Episode%04d'%n['episode_ordinal'])
        op=[json.loads(read(f)['request']['messages'][1]['content']) for f in sorted(folder.glob('Request*_before.json'))]
        np=[json.loads(read(f)['request']['messages'][1]['content']) for f in sorted(nf.glob('Request*_before.json'))]
        if not op or not np:raise ValueError('ACTUAL_REQUESTS_MISSING')
        transition=('CORRECT_RETAINED' if b['source_exact'] is True else 'CORRECT_TO_INCORRECT' if b['source_exact'] is False else 'CORRECT_TO_UNRESOLVED') if a['source_exact'] is True else ('INCORRECT_TO_CORRECT' if b['source_exact'] is True else 'NOT_CORRECT_BOTH')
        row={'panel_ordinal':i,'B2_Source_Exact':a['source_exact'],'B3_Source_Exact':b['source_exact'],'transition':transition,
             'same_reference':a['reference']==b['reference'],'same_initial_observation':digest(op[0]['observation'])==digest(np[0]['observation']),
             'initial_memory_same':digest(a['memory_snapshots'][0])==digest(b['memory_snapshots'][0]),
             'final_memory_same':digest(a['last_memory'])==digest(b['last_memory']),
             'B2_roles':dict(Counter(p['role'] for p in op)),'B3_roles':dict(Counter(p['role'] for p in np)),
             'B3_initial_route':b['routing_trace'][0]['selected_specialty'] if b['routing_trace'] else None,
             'B3_route_changes':dict(Counter(t['selected_specialty'] or 'ABSTAIN_ER' for t in b['routing_trace'])),
             'B2_action_types':dict(Counter(t['action'] for t in a['action_sequence'])),
             'B3_action_types':dict(Counter(t['action'] for t in b['action_sequence'])),
             'B2_final_minute':a['final_minute'],'B3_final_minute':b['final_minute'],
             'B2_generation':r['request_count'],'B3_generation':n['request_count'],
             'final_evidence_same':digest(op[-1].get('evidence'))==digest(np[-1].get('evidence'))}
        public.append(row);private.append({'summary':row,'B2':r,'U1_B3':n,'B2_payloads':op,'B3_payloads':np})
    save(PR/'cross_run_patient_details.json',private)
    report={'comparison':'CROSS_RUN_DEVELOPMENT_AUDIT_NOT_MATCHED_ABLATION','rows':public,
        'B2_correct':sum(r['B2_Source_Exact'] is True for r in public),'B3_correct':sum(r['B3_Source_Exact'] is True for r in public),
        'transitions':dict(Counter(r['transition'] for r in public)),
        'same_model_config':read(bp/'llm_config.json')==read(oldbp/'llm_config.json'),
        'same_source_binding':read(bp/'bindings.json')==read(oldbp/'bindings.json'),
        'parent_run_fingerprints_differ':newplan['fingerprint']!=oldplan['fingerprint'],'actual_new_generation':0,'fitting':0}
    save(PUB/'cross_run_audit.json',report)
    _,staticpr,staticplan=load_static();candidates=read(staticpr/'candidates.json');bank={b['training_case_id']:b for b in source_read(BANK)};sources={s['case_id']:s for s in source_read(SOURCE)};model=read(M5)
    refs={r['ordinal']:r for r in read(staticpr/'reference_private.json')};native=[];checks=[]
    connection=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        for c in candidates:
            cid=c['source_context_id'];b=bank[cid];s=sources[cid];context=json.loads(connection.execute('SELECT payload FROM contexts WHERE cid=?',(cid,)).fetchone()[0])
            row={'ordinal':c['ordinal'],'split':'train' if c['ordinal']<=8 else 'validation','status':'NOT_VALIDATED'}
            try:
                case,binding,rejected=native_candidate(b,s,context,connection,model)
                adapted=adapt_case(case)
                row.update(status='STRUCTURAL_PASS',validation=[{'validator':r.validator,'status':r.status} for r in adapted.validation_results],
                           common_actions=len(binding['rows']),past_records=sum(len(v['records']) for v in case['agent_visible']['past_visits']),memory_rejected=rejected)
                native.append({'ordinal':c['ordinal'],'split':row['split'],'case':case,'binding':binding,'reference':refs[c['ordinal']]})
            except Exception as e:
                row.update(status='BLOCKED',error_type=type(e).__name__)
                if hasattr(e,'results'):row['validation']=[{'validator':r.validator,'status':r.status,'issues':[asdict(i) for i in r.issues]} for r in e.results]
                else:row['private_error']=str(e)
            checks.append(row)
    finally:connection.close()
    if native:verify_csv([r['source_event'] for n in native for r in n['binding']['rows']])
    save(PR/'U2_native_candidate_wrappers_v2.json',native)
    save(PR/'U2_native_validation_details.json',checks)
    train={c['patient_id'] for c in candidates[:8]};val={c['patient_id'] for c in candidates[8:]};holdout={c['patient_id'] for c in original_cases()}
    audit={'candidate_patients':len(candidates),'train_patients':len(train),'validation_patients':len(val),
        'train_validation_overlap':len(train&val),'Original50_overlap':len((train|val)&holdout),
        'validation_M5_train_overlap':len(val&set(model['provenance']['training_patients'])),
        'Native_structurally_valid':len(native),'checks':[{k:v for k,v in c.items() if k in ('ordinal','split','status','common_actions','past_records','error_type')} for c in checks],
        'actual_Native_episodes_collected':0,'Utility_fitting':0,'U2_artifact_created':False,
        'proposed_max_generation':len(candidates)*5*12,
        'blocker_codes':['NATIVE_COLLECTION_RUNNER_AND_FAKE_FIVE_AGENT_VERIFICATION_NOT_COMPLETED',
                         'U2_ACTUAL_NATIVE_RESPONSES_NOT_COLLECTED','U2_NATIVE_FIT_AND_TRANSFER_ADMISSION_NOT_VERIFIED']}
    if audit['validation_M5_train_overlap']:audit['blocker_codes'].append('ROUTING_VALIDATION_VOI_TRAIN_OVERLAP')
    if len(native)!=len(candidates):audit['blocker_codes'].append('NATIVE_WRAPPER_VALIDATION_INCOMPLETE')
    atomic_json(PUB/'U2_native_readiness.json',audit)
    protected.update({f:h for f,h in newplan['pins'].items()});atomic_json(PR/'preservation_before.json',protected)
    unchanged={f:sha_file(f)==h for f,h in protected.items()}
    atomic_json(PUB/'preservation.json',{'checked':len(unchanged),'unchanged':sum(unchanged.values()),'changed':sum(not v for v in unchanged.values())})
    print(json.dumps({'cross_run':{k:v for k,v in report.items() if k!='rows'},'U2':audit,'protected_files':len(protected)}))
if __name__=='__main__':main()
