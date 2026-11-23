"""C4-10 offline source-coded identity audit, separate from clinical relevance."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.dataset_freeze import load, sha
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest
from app.locdi.routing_data_reuse import DATABASE, SYNTHEA
from app.locdi.source_enrichment import validate_candidate
from app.core.longitudinal_memory import PatientMemoryStore, timestamp
from scripts.enrich_full_cohort_sources import DATA, write
from scripts.classify_pre_review_evidence import jsonl, memory_checks
from scripts.audit_memory_review_progress import check_hashes

VERSION = 'locdi.source_measurement_identity.c4_10.v1'
URI = 'http://loinc.org'
TARGETS = {'72514-3': '{score}', '8462-4': 'mm[Hg]', '8480-6': 'mm[Hg]',
           '8867-4': '/min', '9279-1': '/min'}
SOURCES = {
    'namespace': 'https://hl7.org/fhir/R4/loinc.html',
    'coding': 'https://hl7.org/fhir/R4/datatypes.html#Coding',
    'export': 'https://raw.githubusercontent.com/synthetichealth/synthea/master/src/main/java/org/mitre/synthea/export/ExportHelper.java',
    'target_definitions': ['https://loinc.org/' + c for c in TARGETS]}


def read_rows(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def protect():
    parent = ROOT / 'outputs/private/research_evaluation/c4_09_v1'
    pins = load(parent / 'protected_hashes.json')
    mp = ROOT / 'outputs/research_evaluation/c4_09_v1/output_manifest.json'
    manifest = load(mp)
    if digest({k:v for k,v in manifest.items() if k != 'fingerprint'}) != manifest['fingerprint']:
        raise ValueError('PARENT_MANIFEST')
    ip = parent / 'private_index.json'
    if sha(ip) != manifest['private_index_sha256']: raise ValueError('PARENT_INDEX')
    pins.update(manifest['files']); pins.update(load(ip)['files'])
    for p in (mp, ip): pins[p.relative_to(ROOT).as_posix()] = sha(p)
    # Preserve actual current human work, not an obsolete assumption about null judgments.
    for p in (ROOT / 'outputs/private/research_evaluation/c4_06_reviews').rglob('*'):
        if p.is_file(): pins[p.relative_to(ROOT).as_posix()] = sha(p)
    check_hashes(pins)
    return pins


def export_value(value):
    """Documented CSV numeric rendering, no clinical numeric tolerance/conversion."""
    if isinstance(value, bool) or value is None: return None
    if isinstance(value, float):
        rendered = str(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    elif isinstance(value, dict): rendered = value.get('display')
    else: rendered = str(value)
    # CSVExporter.clean: replace each CRLF/CR/LF/comma with one space, then trim.
    return re.sub(r'\r\n|\r|\n|,', ' ', rendered).strip() if rendered is not None else None


def native_index(document):
    index = defaultdict(list)
    def visit(observations, encounter, base):
        for i, obs in enumerate(observations):
            pointer = base + '/' + str(i)
            if obs.get('value') is None:
                visit(obs.get('observations', []), encounter, pointer + '/observations')
                continue
            codes = obs.get('codes') or []
            if not codes or type(obs.get('start')) is not int: continue
            entry = {'coding': codes[0], 'all_codings': codes, 'unit': obs.get('unit') or '',
                     'value': export_value(obs['value']), 'native_value': obs['value'],
                     'start': obs['start'], 'resource_id': obs.get('uuid'), 'pointer': pointer,
                     'encounter_start': encounter['start'], 'resource_type': 'NativeSyntheaObservation',
                     'native_type': obs.get('type'), 'coding_version': codes[0].get('version', 'UNKNOWN')}
            index[(encounter['uuid'], str(codes[0].get('code')), obs['start'] // 1000)].append(entry)
    for i, enc in enumerate(document['record']['encounters']):
        visit(enc.get('observations', []), enc, '/record/encounters/%d/observations' % i)
    return index


def align(index, encounter, code, recorded_at, value, unit):
    if not code or not recorded_at or value in (None, ''):
        return {'status': 'UNKNOWN', 'reason': 'MISSING_JOIN_FIELDS'}
    try: second = int(timestamp(recorded_at).timestamp() // 1)
    except (TypeError, ValueError): return {'status':'UNKNOWN','reason':'INVALID_TIME'}
    candidates = index.get((encounter, str(code), second), [])
    matches = [e for e in candidates if e['value'] == str(value) and e['unit'] == (unit or '')]
    # Distinct native resources at the same timestamp/value are not silently collapsed.
    if len(matches) != 1:
        return {'status': 'UNKNOWN', 'reason': 'AMBIGUOUS_NATIVE_JOIN' if matches else 'NATIVE_VALUE_UNIT_TIME_NOT_MATCHED',
                'candidate_count': len(candidates), 'matching_count': len(matches)}
    e = matches[0]
    system = e['coding'].get('system')
    if system != 'LOINC':
        return {'status':'UNKNOWN','reason':'UNVERIFIED_EXPLICIT_NAMESPACE','native':e}
    return {'status':'VERIFIED_SOURCE_CODING','reason':'EXPLICIT_NATIVE_CODING_NOT_CODE_PATTERN_INFERENCE',
            'system':system, 'system_uri':URI, 'code':str(e['coding']['code']), 'unit':e['unit'], 'native':e,
            'terminology_release':e['coding_version']}


def identity_label(source_type, aligned, targets, structural=True, query_complete=True):
    if not structural: return 'UNKNOWN', 'SOURCE_OR_BOUNDARY_NOT_VERIFIED'
    if not query_complete: return 'UNKNOWN', 'QUERY_CODING_INCOMPLETE'
    if source_type != 'observations':
        return ('NEGATIVE', 'EXPLICIT_RESOURCE_OUTSIDE_MEASUREMENT_TASK') if source_type else ('UNKNOWN', 'MISSING_RESOURCE_TYPE')
    if aligned.get('status') != 'VERIFIED_SOURCE_CODING': return 'UNKNOWN', aligned.get('reason', 'CODING_UNKNOWN')
    key = (aligned['system_uri'], aligned['code'])
    if key not in targets: return 'NEGATIVE', 'OUTSIDE_DECLARED_EXACT_CODE_SET_NOT_CLINICAL_IRRELEVANCE'
    if not aligned['unit'] or aligned['unit'] != targets[key]: return 'UNKNOWN', 'TARGET_UNIT_NOT_IDENTICAL_NO_CONVERSION'
    return 'POSITIVE', 'EXPLICIT_SOURCE_CODE_AND_RECORDED_UNIT_IDENTITY'


def metrics(labels, selected):
    if any(x not in ('POSITIVE', 'NEGATIVE', 'UNKNOWN') for x in labels.values()): raise ValueError('LABEL')
    if len(selected) != len(set(selected)) or any(mid not in labels for mid in selected): raise ValueError('RETURNED_POOL_BOUNDARY')
    pos = sum(labels[mid] == 'POSITIVE' for mid in selected)
    judged = sum(labels[mid] != 'UNKNOWN' for mid in selected)
    total_pos = sum(v == 'POSITIVE' for v in labels.values())
    unknown = sum(v == 'UNKNOWN' for v in labels.values())
    return {'returned':len(selected), 'judged':judged, 'positive_returned':pos, 'unjudged':len(selected)-judged,
            'precision':pos/judged if judged else None, 'judged_coverage':judged/len(selected) if selected else None,
            'full_pool_positive':total_pos, 'full_pool_unknown':unknown,
            'recall':pos/total_pos if not unknown and total_pos else None,
            'recall_status':'COMPUTED_COMPLETE_POOL' if not unknown and total_pos else 'NOT_RUN_INCOMPLETE_OR_NO_POSITIVE_REFERENCE'}


def load_packets():
    folder = ROOT / 'outputs/private/research_evaluation/c4_05_v1/judged_pool_v1'
    manifest = load(folder/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='fingerprint'}) != manifest['fingerprint']: raise ValueError('POOL_MANIFEST')
    packets = []
    for e in manifest['packets']:
        if Path(e['file']).name != e['file'] or sha(folder/e['file']) != e['sha256']: raise ValueError('PACKET_SHA')
        p = load(folder/e['file'])
        if digest(p) != e['packet_fingerprint']: raise ValueError('PACKET_FINGERPRINT')
        packets.append(p)
    return packets


def verify_existing(namespace, csv_projection=False):
    """Read-only independent recount/recalculation; never calls a model or rewrites results."""
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',namespace): raise ValueError('NAMESPACE')
    public=ROOT/'outputs/research_evaluation'/namespace; private=ROOT/'outputs/private/research_evaluation'/namespace
    if csv_projection: public=public/'csv_export_aligned';private=private/'csv_export_aligned'
    pins=load(private/'protected_hashes.json'); sources=load(private/'source_hashes.json')
    check_hashes(pins);check_hashes(sources)
    refs=read_rows(private/'verified_identity_reference.jsonl')
    queries=read_rows(private/'query_source_alignment.jsonl')
    comparison=read_rows(private/'old_pool_comparison.jsonl')
    trace=load(private/'retrieval_trace.json'); published=load(public/'retrieval_evaluation.json')
    if len(refs)!=72590 or len(queries)!=50 or len(comparison)!=989 or len(trace)!=150: raise ValueError('COUNTS')
    labels=defaultdict(dict)
    for r in refs:
        if digest({k:v for k,v in r.items() if k!='evidence_fingerprint'})!=r['evidence_fingerprint']: raise ValueError('EVIDENCE_FINGERPRINT')
        if r['memory_id'] in labels[r['snapshot_id']]: raise ValueError('DUPLICATE_MEMORY')
        labels[r['snapshot_id']][r['memory_id']]=r['label']
        if any(r.get(k) is not None for k in ('clinical_relevance','stale','conflict')): raise ValueError('CLINICAL_JUDGMENT')
    if len(labels)!=50: raise ValueError('ALL50')
    expected={(r['snapshot_id'],mode):r['modes'][mode] for r in published['cases'] for mode in ('none','full_history','selective')}
    for row in trace:
        if metrics(labels[row['snapshot_id']],row['memory_ids']) != expected[(row['snapshot_id'],row['mode'])]:
            raise ValueError('METRIC_RECOMPUTATION')
    task_fp=digest(load(public/'evaluation_task_manifest.json'))
    ref_sha=sha(private/'verified_identity_reference.jsonl')
    return {'status':'VERIFIED','records':len(refs),'queries':len(queries),'old_pool':len(comparison),
            'metric_rows_recomputed':len(trace),'protected_files_unchanged':len(pins),'source_hashes_checked':len(sources),
            'task_manifest_fingerprint':task_fp,'private_reference_sha256':ref_sha,
            'reference_dataset_fingerprint':digest({'task':task_fp,'reference_sha256':ref_sha,'sources':sources}),
            'generation':0,'clinical_judgments':0}


def refine_csv_projection(namespace):
    """Append-only second source pass. Keep first strict pass; reuse its hashed CSV checks and real retrievals."""
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',namespace): raise ValueError('NAMESPACE')
    pb=ROOT/'outputs/research_evaluation'/namespace; pr=ROOT/'outputs/private/research_evaluation'/namespace
    public=pb/'csv_export_aligned';private=pr/'csv_export_aligned'
    if public.exists() or private.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    base_verification=verify_existing(namespace)
    refs=read_rows(pr/'verified_identity_reference.jsonl');queries=read_rows(pr/'query_source_alignment.jsonl')
    trace=load(pr/'retrieval_trace.json'); comparisons=read_rows(pr/'old_pool_comparison.jsonl')
    task=load(pb/'evaluation_task_manifest.json')
    task.update(projection_pass='CSV_EXPORTER_DOCUMENTED_CLEAN',
        projection_rule='CSVExporter.clean replaces CRLF/CR/LF/comma with a space then trims; exact projected equality only',
        parent_strict_reference_sha256=base_verification['private_reference_sha256'])
    write(public/'evaluation_task_manifest.json',task)
    by_case=defaultdict(list);links=None
    for r in refs:by_case[r['snapshot_id']].append(r)
    qmap={q['snapshot_id']:q for q in queries}; recovery=[]
    con=sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
    try:
        for sid,rows in by_case.items():
            unknown=[r for r in rows if r['label']=='UNKNOWN' and r['source_alignment']['reason']=='NATIVE_VALUE_UNIT_TIME_NOT_MATCHED']
            if not unknown:continue
            nativepath=Path(unknown[0]['source_provenance']['native_file'])
            if sha(nativepath)!=unknown[0]['source_provenance']['native_sha256']:raise ValueError('NATIVE_HASH')
            index=native_index(load(nativepath))
            for r in unknown:
                # The already verified original record links source_id to database rid; recover it without guessing IDs.
                origin=None
                for key in ('payload_fingerprint',):
                    if not r['source_provenance'].get(key):raise ValueError('PARENT_SOURCE_FP')
                # memory_id and rid are different; use preserved original-memory link registry.
                if links is None:
                    links={x['memory_id']:x['source_id'] for x in read_rows(ROOT/'outputs/private/research_evaluation/c4_05_v1/original_memory_links.jsonl')}
                hit=con.execute('SELECT payload FROM memory WHERE rid=?',(links[r['memory_id']],)).fetchone()
                origin=json.loads(hit[0]) if hit else None
                if digest(origin)!=r['source_provenance']['payload_fingerprint']:raise ValueError('SOURCE_PAYLOAD_CHANGED')
                raw=origin['_source']['source_record']
                if digest(raw)!=r['source_provenance']['csv_row_fingerprint']:raise ValueError('SOURCE_ROW_CHANGED')
                a=align(index,origin['encounter_id'],origin['code'],origin['timestamp'],origin['value'],origin['units'])
                q=qmap[sid];targets={(a['system_uri'],a['code']):a['unit'] for a in q['alignments'] if a['status']=='VERIFIED_SOURCE_CODING'}
                structural=all(c['result']=='PASS' for c in r['checks'] if c['rule_id'] not in ('M11_EXPLICIT_SUPERSEDES','M13_EXPLICIT_SOURCE_VALUE'))
                label,reason=identity_label(r['source_type'],a,targets,structural,q['complete_source_coding'])
                before=r['evidence_fingerprint']
                r.update(source_alignment=a,label=label,reason=reason,projection_pass='CSV_EXPORTER_DOCUMENTED_CLEAN')
                r['evidence_fingerprint']=digest({k:v for k,v in r.items() if k!='evidence_fingerprint'})
                recovery.append({'snapshot_id':sid,'memory_id':r['memory_id'],'before_fingerprint':before,
                                 'after_fingerprint':r['evidence_fingerprint'],'new_label':label,
                                 'new_alignment_status':a['status'],'csv_source_unchanged':True})
    finally:con.close()
    rows_by_id={(r['snapshot_id'],r['memory_id']):r for r in refs}
    for r in comparisons:r['A1']=rows_by_id[(r['snapshot_id'],r['memory_id'])]['label']
    cases=[];tmap={(r['snapshot_id'],r['mode']):r['memory_ids'] for r in trace}
    for sid,rows in by_case.items():
        labels={r['memory_id']:r['label'] for r in rows}
        cases.append({'snapshot_id':sid,'memory_count':len(rows),'query_complete':qmap[sid]['complete_source_coding'],
            'labels':dict(Counter(labels.values())),
            'modes':{m:metrics(labels,tmap[(sid,m)]) for m in ('none','full_history','selective')}})
    summary=[]
    for mode in ('none','full_history','selective'):
        ms=[r['modes'][mode] for r in cases]
        total={k:sum(r[k] for r in ms) for k in ('returned','judged','positive_returned','unjudged','full_pool_positive','full_pool_unknown')}
        complete=[r for r in ms if r['recall_status']=='COMPUTED_COMPLETE_POOL']
        total.update(mode=mode,verified_identity_judged_precision=total['positive_returned']/total['judged'] if total['judged'] else None,
            judged_coverage=total['judged']/total['returned'] if total['returned'] else None,
            full_cohort_recall=sum(r['positive_returned'] for r in complete)/sum(r['full_pool_positive'] for r in complete) if len(complete)==50 else None,
            recall_status='COMPUTED_COMPLETE_ALL50' if len(complete)==50 else 'NOT_RUN_FULL_COHORT_INCOMPLETE',
            complete_query_count=len(complete),complete_queries_only_micro_recall=sum(r['positive_returned'] for r in complete)/sum(r['full_pool_positive'] for r in complete) if complete else None)
        summary.append(total)
    counts=Counter(r['label'] for r in refs);complete=sum(qmap[s]['complete_source_coding'] and all(r['label']!='UNKNOWN' for r in rs) for s,rs in by_case.items())
    coverage={'registered_queries':50,'full_memory':len(refs),'labels':dict(counts),'complete_queries':complete,'partial_queries':50-complete,
        'verified_source_coded_observations':sum(r['source_alignment']['status']=='VERIFIED_SOURCE_CODING' for r in refs),
        'query_source_complete':sum(q['complete_source_coding'] for q in queries),
        'per_query':[{k:v for k,v in r.items() if k!='modes'} for r in cases],
        'old_literal_positive_506':dict(Counter(r['A1'] for r in comparisons if r['A2']=='POSITIVE')),
        'old_standardized_abstain_704':dict(Counter(r['A1'] for r in comparisons if r['C4_09_standardized']=='ABSTAIN'))}
    write(public/'full_pool_judgment_coverage.json',coverage)
    evaluation=load(pb/'retrieval_evaluation.json');evaluation.update(cases=cases,summary=summary,
        retrieval_source='Parent strict pass actual 150 calls plus 150 deterministic repeats; no new retrieval policy/results synthesized',
        reference_projection_pass='CSV_EXPORTER_DOCUMENTED_CLEAN')
    write(public/'retrieval_evaluation.json',evaluation)
    audit=load(pb/'source_code_system_audit.json')
    audit['counts']['VERIFIED_SOURCE_CODING']=coverage['verified_source_coded_observations']
    audit['counts']['UNKNOWN']=sum(r['source_alignment']['status']=='UNKNOWN' for r in refs)
    audit['reasons']=dict(Counter(r['reason'] for r in refs))
    audit['explicit_native_namespace_counts']=dict(Counter(r['source_alignment']['native']['coding'].get('system','UNKNOWN') for r in refs if r['source_alignment'].get('native')))
    audit['csv_projection_recovered']=sum(r['new_alignment_status']=='VERIFIED_SOURCE_CODING' for r in recovery)
    write(public/'source_code_system_audit.json',audit)
    concepts=Counter((r['source_alignment']['system_uri'],r['source_alignment']['code']) for r in refs if r['source_alignment']['status']=='VERIFIED_SOURCE_CODING')
    write(public/'verified_measurement_mapping.json',{'target_concepts':5,'source_coded_concepts':len(concepts),
        'concepts':[{'uri':u,'code':c,'record_count':n,'official_term_lookup':'VERIFIED_CURRENT_PAGE' if c in TARGETS else 'NOT_REVIEWED','source_terminology_release':'UNKNOWN'} for (u,c),n in sorted(concepts.items())]})
    write(public/'unresolved_code_systems.json',{'unknown_records':counts['UNKNOWN'],'unknown_reasons':dict(Counter(r['reason'] for r in refs if r['label']=='UNKNOWN')),
        'terminology_release':'UNKNOWN; identifier identity does not establish versioned clinical equivalence'})
    failure=load(pb/'failure_analysis.json');failure['native_join_unknown']=dict(Counter(r['reason'] for r in refs if r['label']=='UNKNOWN'))
    failure['csv_projection_recovery']={'attempted':len(recovery),'resolved':sum(r['new_label']!='UNKNOWN' for r in recovery),
        'rule':'Exact existing CSVExporter.clean, no clinical normalization',
        'parent_strict_projection_observations':'Other descriptive diagnostics retain the explicitly preserved first-pass scope'}
    failure['memory_size_and_known_misses']=[{'snapshot_id':r['snapshot_id'],'memory_count':r['memory_count'],
        'known_positive':r['modes']['selective']['full_pool_positive'],'positive_at5':r['modes']['selective']['positive_returned'],'unknown':r['modes']['selective']['full_pool_unknown']} for r in cases]
    write(public/'failure_analysis.json',failure)
    circularity=load(pb/'circularity_audit.json')
    circularity['scoring']='PatientMemoryStore casefold + regex \\w+; integer token intersection COUNT; selective non-superseded positive-score rows ordered by score descending, available_at descending, memory_id then top_k'
    write(public/'circularity_audit.json',circularity)
    jsonl(private/'verified_identity_reference.jsonl',refs);jsonl(private/'query_source_alignment.jsonl',queries)
    jsonl(private/'old_pool_comparison.jsonl',comparisons);jsonl(private/'csv_projection_recovery.jsonl',recovery)
    for name in ('retrieval_trace.json','protected_hashes.json','source_hashes.json'):write(private/name,load(pr/name))
    result=load(pb/'result.json');result.update(status='C4_10_VERIFIED_IDENTITY_EVALUATED' if complete==50 else 'C4_10_PARTIAL_SOURCE_MAPPING',
        positive=counts['POSITIVE'],negative=counts['NEGATIVE'],unknown=counts['UNKNOWN'],complete_queries=complete,
        projection_recovered=sum(r['new_label']!='UNKNOWN' for r in recovery),parent_strict_reference_sha256=base_verification['private_reference_sha256'])
    write(public/'result.json',result)
    check_hashes(load(pr/'protected_hashes.json'));check_hashes(load(pr/'source_hashes.json'))
    return result


def run(namespace):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', namespace): raise ValueError('NAMESPACE')
    public = ROOT/'outputs/research_evaluation'/namespace
    private = ROOT/'outputs/private/research_evaluation'/namespace
    if public.exists() or private.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    pins = protect()
    source_pins = load(ROOT/'outputs/private/research_evaluation/c4_07_v1/source_hashes.json')
    check_hashes(source_pins)
    print('Protected/source hashes verified', flush=True)
    cases = load_cases(DATA); packets = load_packets()
    if len(cases)!=50 or [p['snapshot_id'] for p in packets] != ['OriginalCase%03d'%i for i in range(1,51)]: raise ValueError('COHORT50')
    prior = ROOT/'outputs/private/research_evaluation/c4_05_v1'
    descriptors = {r['case_ref']:r for r in read_rows(prior/'case_source_index.jsonl') if r['dataset']=='Original50'}
    links = {r['memory_id']:r for r in read_rows(prior/'original_memory_links.jsonl')}
    origins = {}; raw_hashes = {}; wanted = defaultdict(set)
    con = sqlite3.connect(DATABASE.resolve().as_uri()+'?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    try:
        for case in cases:
            for v in case['agent_visible']['past_visits']:
                for r in v['records']:
                    hit = con.execute('SELECT payload FROM memory WHERE rid=?', (r['source_id'],)).fetchone()
                    o = json.loads(hit[0]) if hit else None
                    origins[r['memory_id']] = o
                    if o:
                        path = (SYNTHEA/o['_source']['source_file']).resolve()
                        if not path.is_relative_to(SYNTHEA.resolve()): raise ValueError('SOURCE_PATH_ESCAPE')
                        wanted[path].add(o['_source']['source_row'])
                        hit = con.execute('SELECT digest FROM raw_refs WHERE tab=? AND n=?', (o['type'],o['_source']['source_row'])).fetchone()
                        raw_hashes[r['memory_id']] = hit[0] if hit else None
    finally: con.close()
    csv_rows = {}
    for path, numbers in wanted.items():
        source_pins[str(path)] = sha(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            for n,r in enumerate(csv.DictReader(f),2):
                if n in numbers: csv_rows[(str(path),n)] = r
    task = {'version':VERSION, 'task_A1':'EXACT_EXPLICIT_SOURCE_LOINC_CODE_AND_RECORDED_UNIT_IDENTITY',
            'task_A2':'UNCHANGED_C4_09_LITERAL_DESCRIPTION_UNIT', 'task_B':'NO_CLINICAL_JUDGMENTS_CREATED',
            'targets':TARGETS, 'target_policy':'Five actually released measurements; exact namespace/code, no synonyms/crosswalk/unit conversion; display is NOT a label operand',
            'non_target_policy':'Verified other codes and explicit non-observation resource types are negatives ONLY for this fixed identity task',
            'unknown_policy':'Missing/ambiguous join, namespace, or same-target unit mismatch stays UNKNOWN',
            'terminology_version':'UNKNOWN_IN_NATIVE_SOURCE; current official target pages verified, not a pinned release',
            'retrieval':'Unchanged C4-05 final query/cutoff/pool; PatientMemoryStore none/full_history/selective, k=5',
            'recall_gate':'Every eligible memory classified POSITIVE/NEGATIVE, complete current coding, and at least one positive, separately per Query',
            'cohort':{'patients':50,'memory':72590,'judged_pool':989,'source_snapshots':250,'exposure':'DEVELOPMENT_EXPOSED'},
            'official_sources':SOURCES, 'label_inputs_exclude':['display','query tokens','retrieval score','rank','clinical gold']}
    # Freeze the identity definition before labels/retrieval; no outcome-guided target changes.
    write(public/'evaluation_task_manifest.json',task)
    write(private/'protected_hashes.json',pins)
    reference=[]; query_rows=[]; trace=[]; case_stats=[]; pooled=[]
    reasons=Counter(); source_stats=Counter(); concepts=Counter(); display_variants=defaultdict(set)
    display_collisions=defaultdict(set); pairs=Counter(); duplicates=0; leakage=Counter(); executions=0
    original_trace = {(r['snapshot_id'],r['mode']):r for r in load(ROOT/'outputs/private/research_evaluation/c4_09_v1/retrieval_trace.json')}
    old_standard = {(r['snapshot_id'],r['memory_id']):r for r in read_rows(ROOT/'outputs/private/research_evaluation/c4_09_v1/task_a_standardized_reference.jsonl')}
    old_literal = {(r['snapshot_id'],r['memory_id']):r for r in read_rows(ROOT/'outputs/private/research_evaluation/c4_09_v1/task_a_literal_reference.jsonl')}
    for ordinal,(case,p) in enumerate(zip(cases,packets),1):
        sid=p['snapshot_id']; identity=case['agent_visible']['longitudinal']; q=p['query_candidate']
        if identity!=p['private_identity']: raise ValueError('IDENTITY_BINDING')
        validate_candidate(q,case['environment']['current_encounter_events'],identity)
        descriptor=descriptors[sid]; nativepath=Path(descriptor['raw_path'])
        if sha(nativepath)!=descriptor['declared_raw_sha256']: raise ValueError('NATIVE_HASH')
        document=load(nativepath)
        if document['attributes'].get('id')!=identity['patient_id']: raise ValueError('NATIVE_PATIENT')
        index=native_index(document); query_alignments=[]
        for fact in q['source_facts']:
            a=align(index,identity['visit_id'],fact['code'],fact['recorded_at'],fact['value'],fact['units'])
            query_alignments.append(a)
        query_complete=(len(query_alignments)==5 and all(a['status']=='VERIFIED_SOURCE_CODING' and a['code'] in TARGETS and a['unit']==TARGETS[a['code']] for a in query_alignments)
                        and {a.get('code') for a in query_alignments}==set(TARGETS))
        targets={(a['system_uri'],a['code']):a['unit'] for a in query_alignments if a['status']=='VERIFIED_SOURCE_CODING'}
        query_rows.append({'snapshot_id':sid,'query_fingerprint':q['query_fingerprint'],'complete_source_coding':query_complete,
                          'native_sha256':descriptor['declared_raw_sha256'],'alignments':query_alignments,
                          'clinical_visibility_reviewed':None,'actual_available_at':'NOT_RECORDED','recorded_time_proxy':'UNCHANGED_UNREVIEWED'})
        records=[r for v in case['agent_visible']['past_visits'] for r in v['records']]; by_id={r['memory_id']:r for r in records}
        labels={}; per=Counter(); seen=Counter(); pooled_ids={r['memory_id'] for r in p['candidate_memories']}
        for r in records:
            mid=r['memory_id']; o=origins[mid]
            path=(SYNTHEA/o['_source']['source_file']).resolve() if o else None
            raw=csv_rows.get((str(path),o['_source']['source_row'])) if o else None
            checks=memory_checks(r,identity,r,links.get(mid),o,raw, bool(raw and digest(raw)==raw_hashes.get(mid)),by_id)
            structural=all(c['result']=='PASS' for c in checks if c['rule_id'] not in ('M11_EXPLICIT_SUPERSEDES','M13_EXPLICIT_SOURCE_VALUE'))
            a=align(index,r['visit_id'],o['code'],o['timestamp'],o['value'],o['units']) if o and o['type']=='observations' else {'status':'NOT_APPLICABLE','reason':'NON_OBSERVATION'}
            label,reason=identity_label(o['type'] if o else None,a,targets,structural,query_complete)
            labels[mid]=label; per[label]+=1; reasons[reason]+=1
            source_stats['structural_pass' if structural else 'structural_not_pass']+=1
            source_stats[a['status']]+=1
            if o:
                source_stats['intermediate_namespace_missing']+=not bool(o.get('system'))
                if a['status']=='VERIFIED_SOURCE_CODING':
                    concepts[(a['system_uri'],a['code'])]+=1
                    display_variants[(a['system_uri'],a['code'])].add(o['description'])
                    display_collisions[o['description']].add((a['system_uri'],a['code']))
                    source_stats['native_vs_csv_display_different']+=a['native']['coding'].get('display')!=o['description']
                    seen[(r['visit_id'],a['code'],a['native']['start'],a['unit'],o['value'])]+=1
            row={'schema_version':VERSION,'snapshot_id':sid,'memory_id':mid,'label':label,'reason':reason,
                 'reference_type':'SOURCE_CODE_IDENTITY_NOT_CLINICAL_GOLD','source_alignment':a,
                 'source_type':o.get('type') if o else None,'original_record_fingerprint':digest(r),
                 'source_provenance':{'database_sha256':source_pins[str(DATABASE.resolve())], 'payload_fingerprint':digest(o),
                    'csv_file':str(path),'csv_sha256':source_pins.get(str(path)),
                    'csv_row':o['_source']['source_row'] if o else None,'csv_row_fingerprint':digest(raw),
                    'native_file':str(nativepath),'native_sha256':descriptor['declared_raw_sha256']},
                 'checks':checks,'clinical_relevance':None,'stale':None,'conflict':None}
            row['evidence_fingerprint']=digest(row); reference.append(row)
            if mid in pooled_ids:
                key=(sid,mid); old=old_standard[key]['silver_label']; literal=old_literal[key]['silver_label']
                pairs[(old,label)]+=1
                pooled.append({'snapshot_id':sid,'memory_id':mid,'A1':label,'A2':literal,'C4_09_standardized':old})
        duplicates+=sum(n-1 for n in seen.values() if n>1)
        st={'snapshot_id':sid,'memory_count':len(records),'query_complete':query_complete,'labels':dict(per),'modes':{}}
        store=PatientMemoryStore(records)
        for mode in ('none','full_history','selective'):
            result=store.retrieve(identity,q['query'],mode,5)['records']; executions+=1
            ids=[r['memory_id'] for r in result]
            repeated=store.retrieve(identity,q['query'],mode,5)['records']; executions+=1
            if result!=repeated: raise ValueError('NONDETERMINISTIC_RETRIEVAL')
            if ids!=original_trace[(sid,mode)]['memory_ids']: raise ValueError('C4_09_RETRIEVAL_CHANGED')
            for r in result:
                leakage['patient_visit']+=r['patient_id']!=identity['patient_id'] or r['visit_id']==identity['visit_id']
                leakage['cutoff']+=timestamp(r['available_at'])>timestamp(identity['started_at']) or timestamp(r['visit_started_at'])>=timestamp(identity['started_at'])
            st['modes'][mode]=metrics(labels,ids)
            trace.append({'snapshot_id':sid,'mode':mode,'query_fingerprint':q['query_fingerprint'],'cutoff':identity['started_at'],'memory_ids':ids})
        case_stats.append(st)
        print('Audited %d/50; records=%d; unknown=%d'%(ordinal,len(records),per['UNKNOWN']),flush=True)
    if len(reference)!=72590 or len(pooled)!=989: raise ValueError('FULL_COHORT_COUNTS')
    summary=[]
    for mode in ('none','full_history','selective'):
        rows=[r['modes'][mode] for r in case_stats]
        total={k:sum(r[k] for r in rows) for k in ('returned','judged','positive_returned','unjudged','full_pool_positive','full_pool_unknown')}
        complete=[r for r in rows if r['recall_status']=='COMPUTED_COMPLETE_POOL']
        total.update(mode=mode,verified_identity_judged_precision=total['positive_returned']/total['judged'] if total['judged'] else None,
                     judged_coverage=total['judged']/total['returned'] if total['returned'] else None,
                     full_cohort_recall=sum(r['positive_returned'] for r in complete)/sum(r['full_pool_positive'] for r in complete) if len(complete)==50 else None,
                     recall_status='COMPUTED_COMPLETE_ALL50' if len(complete)==50 else 'NOT_RUN_FULL_COHORT_INCOMPLETE',
                     complete_query_count=len(complete),complete_queries_only_micro_recall=sum(r['positive_returned'] for r in complete)/sum(r['full_pool_positive'] for r in complete) if complete else None)
        summary.append(total)
    counts=Counter(r['label'] for r in reference); complete=sum(r['query_complete'] and not r['labels'].get('UNKNOWN',0) for r in case_stats)
    status='C4_10_VERIFIED_IDENTITY_EVALUATED' if complete==50 else 'C4_10_PARTIAL_SOURCE_MAPPING' if counts['POSITIVE'] else 'C4_10_SOURCE_INSUFFICIENT'
    write(public/'source_code_system_audit.json',{'version':VERSION,'explicit_native_namespace':'LOINC','canonical_uri':URI,
        'namespace_inferred_from_code_pattern':False,'terminology_release':'UNKNOWN','source_run_version':'master-branch-latest (metadata; no pinned commit)',
        'intermediate_observation_csv_system_column':'ABSENT','intermediate_database_system':'NULL; native separately aligned',
        'counts':dict(source_stats),'reasons':dict(reasons),'official_sources':SOURCES,
        'official_lookup_scope':'Five current target term pages only; no complete versioned terminology release/API acquired',
        'source_native_json_is_not_fhir':True})
    write(public/'verified_measurement_mapping.json',{'target_concepts':len(TARGETS),'source_coded_concepts':len(concepts),
        'concepts':[{'uri':uri,'code':code,'record_count':n,'official_term_lookup':'VERIFIED_CURRENT_PAGE' if code in TARGETS else 'NOT_REVIEWED',
                     'source_terminology_release':'UNKNOWN','display_variants':len(display_variants[(uri,code)])} for (uri,code),n in sorted(concepts.items())],
        'unit_policy':'Recorded unit exact equality only; /min not automatically promoted to UCUM {beats}/min or {breaths}/min'})
    write(public/'unresolved_code_systems.json',{'unknown_records':counts['UNKNOWN'],
        'unknown_reasons':dict(Counter(r['reason'] for r in reference if r['label']=='UNKNOWN')),
        'terminology_release_unknown_records':source_stats['VERIFIED_SOURCE_CODING'],
        'source_code_identity_vs_release_semantics':'Source identifier match can be verified without claiming versioned clinical equivalence'})
    write(public/'full_pool_judgment_coverage.json',{'registered_queries':50,'full_memory':72590,'labels':dict(counts),
        'verified_source_coded_observations':source_stats['VERIFIED_SOURCE_CODING'],'complete_queries':complete,'partial_queries':50-complete,
        'query_source_complete':sum(r['query_complete'] for r in case_stats),'per_query':[{k:v for k,v in r.items() if k!='modes'} for r in case_stats],
        'old_pool_changes':[{'previous':a,'A1':b,'count':n} for (a,b),n in sorted(pairs.items())],
        'old_literal_positive_506':dict(Counter(r['A1'] for r in pooled if r['A2']=='POSITIVE')),
        'old_standardized_abstain_704':dict(Counter(r['A1'] for r in pooled if r['C4_09_standardized']=='ABSTAIN'))})
    write(public/'retrieval_evaluation.json',{'scope':'EXPLICIT_SOURCE_CODE_IDENTITY_ONLY_NOT_CLINICAL_RELEVANCE','summary':summary,
        'cases':case_stats,'actual_retrieval_calls':executions,'first_pass_calls':150,'repeat_calls':150,
        'c4_09_returned_ids_identical':True,'leakage':dict(leakage),'clinical_metrics':'NOT_EVALUATED',
        'A2_unchanged_results_file':'outputs/research_evaluation/c4_09_v1/silver_evaluation.json',
        'A2_97_2_percent_denominator':'243 positive / 250 judged returned records, labels only in old 989 pool; not full-pool recall'})
    write(public/'failure_analysis.json',{'native_join_unknown':dict(Counter(r['reason'] for r in reference if r['label']=='UNKNOWN')),
        'same_code_multiple_csv_displays':sum(len(v)>1 for v in display_variants.values()),
        'same_csv_display_multiple_codes':sum(len(v)>1 for v in display_collisions.values()),
        'native_csv_display_mismatch':source_stats['native_vs_csv_display_different'],
        'duplicate_observation_signatures_extra_records':duplicates,
        'missing_chief_complaint_queries':50,'actual_clinical_available_timestamp_missing_queries':50,
        'memory_size_and_known_misses':[{'snapshot_id':r['snapshot_id'],'memory_count':r['memory_count'],
            'known_positive':r['modes']['selective']['full_pool_positive'],'positive_at5':r['modes']['selective']['positive_returned'],
            'unknown':r['modes']['selective']['full_pool_unknown']} for r in case_stats],
        'causal_size_effect':'NOT_ESTABLISHED; k=5 cannot retrieve all repeated identity records in long histories',
        'clinical_same_text_different_meaning':'NOT_REVIEWED; namespace/code collisions are structural facts only'})
    write(public/'circularity_audit.json',{'A1':'SOURCE_CODE_REFERENCE_INDEPENDENT_OF_DISPLAY_AND_RANK; RESIDUAL_CONCEPT_TEXT_CORRELATION',
        'A2':'HIGH_SHARED_DESCRIPTION_UNIT_SIGNAL_AND_POOLED_SELECTION','B':'NO_INDEPENDENT_REFERENCE_NO_METRICS',
        'query':'Unchanged C4-05 released history/objective findings serialization; five current measurement facts; no complaint',
        'memory_text':'CSV time | resource type | description | value | units; namespace/code are not explicitly appended',
        'scoring':'Existing PatientMemoryStore casefold + regex \\w+ token sets; score is intersection COUNT (not normalized). Selective excludes superseded/zero-score, orders by descending overlap, descending available_at, memory_id, then top_k.',
        'reference':'A1 uses explicit source namespace/code/unit; no tokenizer/score/rank/display inputs',
        'pool':'Full 72590 attempted; old 989 preserved for cross-tab only; UNKNOWNs block full recall per Query',
        'residual_risks':['Code-specific display text correlates with concept labels','Original50 development exposure','Repeated records clustered by patient',
            'Record-time proxy unreviewed','Current code terminology release unknown','Task is measurement identity, not clinical usefulness'],
        'clinical_performance_proven':False})
    jsonl(private/'verified_identity_reference.jsonl',reference)
    jsonl(private/'query_source_alignment.jsonl',query_rows)
    jsonl(private/'old_pool_comparison.jsonl',pooled)
    write(private/'retrieval_trace.json',trace)
    check_hashes(pins); check_hashes(source_pins)
    write(private/'source_hashes.json',source_pins)
    result={'status':status,'patients':50,'memory_records':len(reference),'old_pool_records':len(pooled),'source_snapshots_preserved':250,
            'positive':counts['POSITIVE'],'negative':counts['NEGATIVE'],'unknown':counts['UNKNOWN'],'complete_queries':complete,
            'protected_files':len(pins),'source_assets':len(source_pins),'changed_protected_files':0,
            'generation':0,'fitting':0,'clinical_judgments_generated':0,'retrieval_calls':executions}
    write(public/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',default='c4_10_v1');parser.add_argument('--verify-existing',action='store_true')
    parser.add_argument('--refine-csv-projection',action='store_true')
    parser.add_argument('--csv-export-aligned',action='store_true',help='Select appended source-projection pass for read-only verification')
    args=parser.parse_args()
    if args.csv_export_aligned and not args.verify_existing:parser.error('--csv-export-aligned requires --verify-existing')
    def guard(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.bind'): raise RuntimeError('OFFLINE_ONLY')
    sys.addaudithook(guard)
    print(json.dumps(refine_csv_projection(args.namespace) if args.refine_csv_projection else verify_existing(args.namespace,args.csv_export_aligned) if args.verify_existing else run(args.namespace),ensure_ascii=True))
