"""Read-only verification of completed recovery traces; never calls a model."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_memory_paired_recovery import PUB,PR,RUN,load_protocol
from scripts.finalize_memory_paired_recovery import compute
from scripts.audit_research_gaps import read,rows,sha,write
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest


def verify(write_seal=True):
    value=compute();p=load_protocol();root=PR/RUN
    independent=value['summary']==read(root/'aggregate.json')
    cases=load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl');panel=read(PR/'panel.json');index=read(PR/'case_index.json')
    trace=rows(root/'episodes.jsonl');identities=[];rendering=[];fingerprints=[];visit_cutoffs=[]
    request_attempts=0;HTTP=Counter();schema=Counter();error_types=Counter()
    for row in trace:
        folder=root/'requests'/('Episode%04d'%row['episode_ordinal'])
        case=cases[index[row['case_ref']]-1]
        raw_path=folder/'raw_runner_result.json';raw=read(raw_path) if raw_path.exists() else {}
        visits=(raw.get('execution') or {}).get('visits',[])
        events=visits[0]['final_state']['event_log'] if visits else []
        received=[e for e in events if e['event_type']=='result_received']
        initial=case['agent_visible']['current_initial_state'].get('objective_findings',[])
        for before in sorted(folder.glob('Request*_before.json')):
            b=read(before);request=b['request'];u=json.loads(request['messages'][1]['content'])
            fingerprints.append(b['request_fingerprint']==digest(request))
            def instant(s):return datetime.fromisoformat(s.replace('Z','+00:00'))
            cutoff=instant(case['agent_visible']['index_encounter_start'])
            for record in u.get('observation',{}).get('longitudinal_memory',{}).get('records',[]):
                visit_cutoffs.append(instant(record['visit_started_at'])<cutoff
                    and instant(record['available_at'])<=cutoff and instant(record['observed_at'])<=cutoff)
            for finding in u.get('observation',{}).get('objective_findings',[]):
                if finding in initial:
                    rendering.append({'source':'INITIAL_PUBLIC_SOURCE','source_and_time_match':True});continue
                matches=[e for e in received if finding in (e['payload']['test_result'].get('summary'),e['payload']['test_result'].get('impression'))]
                exact=[]
                for event in matches:
                    name=event['payload']['test_result']['test_name']
                    for source in case['environment']['current_encounter_events']:
                        if source['action']=='Test' and finding=='%s: %s'%(name,source['evidence']):
                            exact.append(event['minute']<=u['observation']['elapsed_minutes'] and source['available_minute']<=event['minute'])
                rendering.append({'source':'ACTUAL_RECEIVED_TEST','source_and_time_match':bool(exact) and any(exact)})
            after=before.with_name(before.name.replace('_before','_after'))
            if not after.exists():continue
            a=read(after);request_attempts+=a.get('generation_attempted',True)
            HTTP[str(a.get('HTTP_status'))]+=1;schema[a['status']]+=1
            if a.get('error_type'):error_types[a['error_type']]+=1
            fingerprints.append(a.get('request_fingerprint')==digest(request))
            if a['status']=='VALID':
                bi=b['identity_before'];ai=a['identity_after']
                identities.append({'digest_match':bi['digest'].removeprefix('sha256:')==p['model_identifier'].removeprefix('sha256:')==ai['digest'].removeprefix('sha256:'),
                                   'context_match':ai['context_length']==32768,'model_match':a['response_envelope']['model']==p['model']})
    public=[f for f in PUB.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.md')]
    docs=ROOT/'docs/연구 평가/FINAL_EXP_RECOVERY'
    public+=list(docs.glob('*.md'))+[ROOT/'docs/연구 평가/FINAL_EXP_RECOVERY_REPORT.md']
    identifiers=set()
    for case in cases:identifiers.update([case['patient_id'],case['case_id'],case['agent_visible']['index_encounter_id']])
    leaked=[]
    for file in public:
        if not file.exists():continue
        text=file.read_text(encoding='utf-8-sig')
        if any(identifier in text for identifier in identifiers if len(identifier)>6):leaked.append(str(file.relative_to(ROOT)))
    checks={'registered40_rows':len(trace)==40,'registered20_pairs':len(value['pairs'])==20,
        'independent_saved_trace_reaggregation':independent,'request_accounting_match':request_attempts==value['reported_generation_count']==value['generation_count'],
        'request_fingerprints_match':all(fingerprints),'successful_request_identity_and_context':all(all(i.values()) for i in identities),
        'Gold_private_keys_absent':value['Gold_boundary_scan'],'Memory_source_patient_visit_cutoff':value['Memory_boundary_status']=='PASS',
        'Memory_records_before_original_index_encounter':all(visit_cutoffs),
        'finding_exact_Source_and_receipt_time':all(r['source_and_time_match'] for r in rendering),
        'protected_inputs_unchanged':value['protected']['status']=='PASS','execution_pins_unchanged':value['input_pins']['status']=='PASS',
        'original_patient_visit_case_ids_absent_from_public':not leaked,
        'concept_not_automatically_approved':all(e['verified_concept_agreement'] is None for e in value['summary']['episodes'])}
    output={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'scope':'export/source-boundary/reaggregation integrity; not all clinical outcomes or all Episodes successful',
        'actual_generation_requests':request_attempts,'HTTP_status_counts':dict(HTTP),'native_status_counts':dict(schema),
        'request_error_types':dict(error_types),'successful_identities_checked':len(identities),
        'actual_test_finding_occurrences_checked':len(rendering),
        'delivered_memory_occurrences_checked':sum(m['record_occurrences'] for m in value['memory_boundary_checks']),
        'index_encounter_cutoff_occurrences_checked':len(visit_cutoffs),
        'complete_pairs':value['summary']['complete_pairs'],'source_scorable_pairs':value['summary']['source_scorable_pairs'],
        'protected_files':value['protected'],'input_pins':value['input_pins'],'public_identifier_leaked_files':leaked,
        'source_trace_sha256':sha(root/'episodes.jsonl'),'protocol_fingerprint':p['fingerprint'],
        'check_command':'python -B scripts/verify_memory_paired_recovery.py --check-only',
        'verification_generations':0,'clinical_reference':'NOT_REVIEWED','actual_clinical_release_time':'NOT_REVIEWED; existing recorded-time proxy',
        'files':{str(f.relative_to(ROOT)):sha(f) for f in public if f.exists()}}
    if write_seal:write(PUB/'verification_final.json',output)
    print(json.dumps({k:v for k,v in output.items() if k!='files'}))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');a=parser.parse_args()
    verify(write_seal=not a.check_only)
