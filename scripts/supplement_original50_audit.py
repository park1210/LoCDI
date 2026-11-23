"""Independent read-only completeness addendum. No retrieval implementation imported."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import original_cases,native_index,read
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
NS='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS

def flatten(observations,path):
    for i,o in enumerate(observations):
        ptr=path+'/'+str(i)
        if 'value' in o and o.get('codes'):
            for code in o['codes']:
                yield {'code':str(code['code']),'description':code.get('display'),'value':o['value'],'unit':o.get('unit',''),
                    'start':o.get('start'),'pointer':ptr}
        yield from flatten(o.get('observations',[]),ptr+'/observations')

def main():
    cases=original_cases();idx=native_index();traces=read(PR/'baseline_trace_details.json');details=read(PR/'patient_source_details.json')
    memorymap=read(PR/'memory_source_map.json');byordinal={r['ordinal']:r for r in details};counts=Counter();out=[];semantic=Counter()
    with (PR/'NATIVE_SOURCE_COMPLETENESS_AND_ASK_APPENDIX.md').open('x',encoding='utf-8') as f:
        f.write('# PRIVATE — Native JSON completeness and actual Ask history\n\nRoot symptom state is export-time state, NOT a verified current complaint. It is not admitted into any model input. All contents below remain private.\n')
        for n,c in enumerate(cases,1):
            raw=read(idx[n]['raw_path']);visit=c['agent_visible']['index_encounter_id']
            enc=next(e for e in raw['record']['encounters'] if e.get('uuid')==visit)
            native=list(flatten(enc.get('observations',[]),'/record/current_encounter/observations'))
            evs=[e for e in c['environment']['source_event_store'] if e['event_type']=='observations']
            csvkeys={(str(e['target']['code']),str(e['evidence'].get('value')),e['evidence'].get('units') or '') for e in evs}
            nativekeys={(e['code'],str(e['value']),e['unit'] or '') for e in native}
            extra=[e for e in native if (e['code'],str(e['value']),e['unit'] or '') not in csvkeys]
            for x in extra:x['availability']='NATIVE_ONLY_NOT_RELEASE_REVIEWED_NOT_AGENT_INPUT'
            latest=[r for r in traces['latest'] if r['original_ordinal']==n];candidateids=byordinal[n]['omitted_candidate_ids']
            candidatecodes=Counter(str(memorymap[mid].get('code')) for mid in candidateids)
            row={'patient_ordinal':n,'csv_current_measurement_rows':len(evs),'native_numeric_or_text_leaf_rows':len(native),
                'native_only_rows':extra,'csv_keys_not_in_native_leaf_count':len(csvkeys-nativekeys),'native_encounter_fields':sorted(enc),
                'native_reason':enc.get('reason'),'raw_symptom_state':raw.get('symptoms'),
                'omitted_same_measurement_candidates_by_code':dict(candidatecodes),
                'omitted_pain_candidates':candidatecodes['72514-3'],'omitted_nonpain_candidates':sum(v for k,v in candidatecodes.items() if k!='72514-3')}
            out.append(row);counts.update(patients=1,csv_current_rows=len(evs),native_leaf_rows=len(native),native_only_rows=len(extra),
                csv_keys_not_in_native_leaf_count=len(csvkeys-nativekeys),omitted_pain_candidates=row['omitted_pain_candidates'],
                omitted_nonpain_candidates=row['omitted_nonpain_candidates'])
            counts['patients_with_native_only_rows']+=bool(extra)
            f.write('\n## Patient %02d\n\n'%n)
            f.write('Current native encounter reason (not automatically a complaint): '+json.dumps(enc.get('reason'),ensure_ascii=False)+'\n\n')
            f.write('Current CSV observations %d; native leaf observations %d; native-only %d.\n\n'%(len(evs),len(native),len(extra)))
            f.write('Native-only observations (unreviewed):\n\n```json\n'+json.dumps(extra,ensure_ascii=False,indent=2)+'\n```\n')
            f.write('Export-time symptom state, NOT current symptoms:\n\n```json\n'+json.dumps(raw.get('symptoms'),ensure_ascii=False,indent=2)+'\n```\n')
            f.write('Actually disclosed Ask/question histories in saved latest payloads:\n')
            for t in latest:f.write('\n'+t['condition']+' '+t['status']+'\n```json\n'+json.dumps(t['observation'].get('history',[]),ensure_ascii=False,indent=2)+'\n```\n')
            f.write('Unselected equal-code/unit candidates: '+json.dumps(dict(candidatecodes))+'; code matching is NOT a clinical relevance label.\n')
    # Verify the derived measurement metadata against the exact Source CSV rows.
    byfile=defaultdict(dict)
    for s in memorymap.values():
        if s.get('source_type')=='observations':byfile[s['source_file']][s['source_row']]=s
    from app.locdi.source_paths import resolve_source_file
    for name,wanted in byfile.items():
        with resolve_source_file(name).open(encoding='utf-8-sig',newline='') as f:
            for n,row in enumerate(csv.DictReader(f),2):
                if n not in wanted:continue
                s=wanted[n];ok=(str(s['code'])==row['CODE'] and s['description']==row['DESCRIPTION'] and str(s['value'])==row['VALUE'] and s['unit']==row['UNITS'])
                semantic['checked']+=1;semantic['exact']+=ok
                if not ok:raise ValueError('DERIVED_MEMORY_MEASUREMENT_SEMANTIC_MISMATCH')
    atomic_json(PR/'native_completeness_details.json',out)
    summary={'counts':dict(counts),'memory_measurement_code_value_unit_description_checks':dict(semantic),
        'raw_symptom_visibility':'EXPORT_STATE_NOT_CURRENT_ENCOUNTER_REVIEWED','generation':0,'clinical_relevance':'NOT_REVIEWED'}
    atomic_json(PUB/'source_completeness_addendum.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
