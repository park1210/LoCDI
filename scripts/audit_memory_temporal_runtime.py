"""C4-23 saved-trace analysis. Offline, exclusive output; never invokes retrievers."""
import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,write,sha,check,locations
from scripts.evaluate_memory_concepts import BASE,jsonlines,native_check,offline
from scripts.evaluate_measurement_identity import load_packets
from app.locdi.llm_setup import digest
from app.locdi.memory_concept_evaluation import validate_reference,concept_key
from app.locdi.source_enrichment import validate_candidate
from app.locdi.memory_temporal_audit import measurement,compare,runtime_fragment
from app.locdi.agentclinic_pilot_readiness import LocalTokenCounter,tokenizer_tree

NS='c4_23_v1'
PARENT=ROOT/'outputs/private/research_evaluation/c4_22_v1'
CODE=['app/locdi/memory_temporal_audit.py','scripts/audit_memory_temporal_runtime.py','tests/locdi/test_memory_temporal_audit.py']
TOKENIZER=ROOT/'artifacts/external_tokenizers/c3_13c_qwen2_5_1_5b_989aa798_acquired/tokenizer'
TREE='de5d71eac66d8658dca7a53895eea69ca7180def1de601c7dc80efd30a958ad3'


def freeze():
    pub,pr=locations(NS)
    if pub.exists() or pr.exists():raise ValueError('NAMESPACE_USED_NO_RESUME')
    pins=read(PARENT/'protected_hashes.json');sources=read(PARENT/'source_hashes.json')
    mp=ROOT/'outputs/research_evaluation/c4_22_v1/output_manifest.json';m=read(mp)
    assert digest({k:v for k,v in m.items() if k!='fingerprint'})==m['fingerprint']
    ip=ROOT/m['private_index_path'];assert sha(ip)==m['private_index_sha256']
    for n,h in {**m['files'],**read(ip)['files']}.items():
        if n in pins:assert pins[n]==h
        pins[n]=h
    for p in (mp,ip):pins[p.relative_to(ROOT).as_posix()]=sha(p)
    a,b=check(pins),check(sources);assert a['status']==b['status']=='PASS'
    write(pr/'protected_hashes.json',pins);write(pr/'source_hashes.json',sources)
    write(pr/'initial_preservation.json',{'protected':a,'sources':b})
    write(pr/'code_hashes.json',{n:sha(ROOT/n) for n in CODE})
    protocol={'version':'c4_23.temporal_audit.v1','population':{'queries':50,'memory':72590},
        'selection':'C4-22 saved baseline and development IDs only; no selector/retriever execution',
        'concept':'Exact existing verified system URI/code/unit/native resource type; no normalization',
        'time':'Verified NativeSyntheaObservation.start epoch milliseconds only. Proxy observed_at/available_at never substitutes for actual measurement or actual clinical release',
        'values':'Native source JSON typed exact values; null missing; no unit conversion, text parsing or clinical trend judgment',
        'loss':'Per patient/concept set difference of actual timestamps and of canonical source values; these overlap and must not be added as unique records',
        'added':'Shared-concept additions separate from new-concept additions in private details',
        'endpoints':'Any tied record at earliest/latest actual timestamp counts retained; eligible endpoints over full cutoff-eligible verified pool, selected endpoints separately',
        'trend':'Numeric chronological transitions only when all selected timed values numeric finite and each time has one value; equal-time conflicts => not computable',
        'budget':'Unchanged Core row fields plus historical/status and original score; role_call ensure_ascii=False sort_keys=True JSON records-list fragment. No entire prompt, wrapper, schema, chat template, output or server token claim',
        'tokenizer':{'model':'Qwen/Qwen2.5-1.5B-Instruct','revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','tree_sha256':TREE,'offline_only':True,'add_special_tokens':False},
        'metadata_overhead':'Hypothetical whitelist concept_metadata on selected rows; not applied or accepted by current strict Core schema',
        'scope':'DEVELOPMENT_SOURCE_IDENTITY_TEMPORAL_FACTS_NOT_CLINICAL_EVALUATION',
        'generation':0,'fitting':0,'policy_changes':0,'source_pins_fingerprint':digest(sources)}
    protocol['fingerprint']=digest(protocol);write(pub/'temporal_preservation_protocol.json',protocol)
    print(json.dumps({'freeze':'PASS','protected':len(pins),'sources':len(sources)}),flush=True)


def run():
    pub,pr=locations(NS)
    assert check(read(pr/'code_hashes.json'))['status']=='PASS'
    if (pr/'execution_started.json').exists():raise ValueError('NO_RESUME')
    protocol=read(pub/'temporal_preservation_protocol.json')
    assert protocol['fingerprint']==digest({k:v for k,v in protocol.items() if k!='fingerprint'})
    write(pr/'execution_started.json',{'protocol_fingerprint':protocol['fingerprint'],'generation':0})
    counter=LocalTokenCounter(TOKENIZER,TREE)
    import transformers,tokenizers
    def count(s):return len(counter.tokenizer.encode(s,add_special_tokens=False))
    pools=list(jsonlines(PARENT/'candidate_pools.jsonl'));traces=list(jsonlines(PARENT/'selection_trace.jsonl'))
    cases=list(jsonlines(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl'));packets=load_packets()
    qs={q['snapshot_id']:q for q in jsonlines(BASE/'query_source_alignment.jsonl')}
    groups=iter(groupby(jsonlines(BASE/'verified_identity_reference.jsonl'),key=lambda r:r['snapshot_id']))
    public=[];tokens=[];replacements=[];negative=[];total=Counter();mapping=Counter();rawfields=Counter()
    with (pr/'temporal_record_comparison.jsonl').open('x',encoding='utf-8') as out:
        for i,(case,packet,pool,trace) in enumerate(zip(cases,packets,pools,traces),1):
            sid,rs=next(groups);refs=list(rs);assert sid==packet['snapshot_id']==pool['snapshot_id']
            ident=case['agent_visible']['longitudinal'];assert ident==packet['private_identity']
            validate_candidate(packet['query_candidate'],case['environment']['current_encounter_events'],ident)
            assert pool['query_fingerprint']==packet['query_candidate']['query_fingerprint']
            assert digest(pool['candidates'])==pool['candidate_input_fingerprint']
            assert trace['selection']['input_fingerprint']==pool['candidate_input_fingerprint']
            old=trace['baseline_ids'];new=trace['selection']['memory_ids'];assert old==pool['baseline_ids']
            assert old==[r['memory_id'] for r in pool['candidates'][:5]]
            records=[r for v in case['agent_visible']['past_visits'] for r in v['records']]
            byid={r['memory_id']:r for r in records};assert set(byid)=={r['memory_id'] for r in refs}
            nativepath=Path(refs[0]['source_provenance']['native_file']);native_sha=sha(nativepath)
            assert native_sha==qs[sid]['native_sha256'];doc=read(nativepath)
            assert doc['attributes']['id']==ident['patient_id']
            qkeys={concept_key(a,'observations')[0] for a in qs[sid]['alignments']}
            ms={};refmap={r['memory_id']:r for r in refs}
            for r in refs:
                original=byid[r['memory_id']];rawfields.update(original.keys())
                validate_reference(r,original,ident)
                assert r['source_provenance']['native_sha256']==native_sha
                total['native_pointer_checks']+=native_check(r['source_alignment'],doc,original['visit_id'])
                m=measurement(original,r,ident);ms[r['memory_id']]=m;mapping[m['mapping_status']]+=1
                total['memory']+=1
                if m['concept_key'] is not None:
                    total['mapped_source_times']+=m['measurement_epoch_ms'] is not None
                    total['mapped_missing_source_times']+=m['measurement_epoch_ms'] is None
            summary,detail=compare([ms[x] for x in old],[ms[x] for x in new],[ms[x] for x in pool['full_ids']])
            summary['query_id']='Query%03d'%i;public.append(summary)
            cs={r['memory_id']:r for r in pool['candidates']};rank={r['memory_id']:j for j,r in enumerate(pool['candidates'],1)}
            removed=[x for x in old if x not in new];added=[x for x in new if x not in old]
            rr=[]
            for change,ids in [('removed',removed),('added',added)]:
                for mid in ids:
                    row={**ms[mid],'change':change,'source_identity_label':refmap[mid]['label'],
                        'source_type':refmap[mid]['source_type'],'original_rank':rank[mid],'original_score':cs[mid]['score'],
                        'query_exact_identity':tuple(ms[mid]['concept_key']) in qkeys if ms[mid]['concept_key'] else None,
                        'selection_reason':'FIRST_VERIFIED_CONCEPT_PASS' if change=='added' else 'REPEATED_CONCEPT_DEFERRED_BEYOND_TOP5',
                        'source_reason':refmap[mid]['reason']}
                    rr.append(row);total[change+'_'+row['source_identity_label']]+=1
                    if change=='added' and row['source_identity_label']=='NEGATIVE':negative.append({'query_id':summary['query_id'],**row})
            replacements.extend({'query_id':summary['query_id'],**r} for r in rr)
            budget={'query_id':summary['query_id'],'records_per_mode':5}
            for name,ids in [('baseline',old),('diverse',new)]:
                rows,fragment=runtime_fragment(records,pool['candidates'],ids)
                extra=[]
                for row in rows:
                    row=dict(row);key=ms[row['memory_id']]['concept_key']
                    if key:row['concept_metadata']=dict(zip(('system_uri','code','unit','resource_type'),key))
                    extra.append(row)
                enriched=json.dumps(extra,ensure_ascii=False,sort_keys=True,allow_nan=False)
                n=count(fragment);en=count(enriched)
                budget[name]={'characters':len(fragment),'utf8_bytes':len(fragment.encode()),'tokens':n,
                    'tokens_per_record':n/len(ids),'fragment_fraction_of_reference_32768':n/32768,
                    'hypothetical_metadata_chars':len(enriched)-len(fragment),
                    'hypothetical_metadata_tokens':en-n}
            budget['token_delta']=budget['diverse']['tokens']-budget['baseline']['tokens'];tokens.append(budget)
            out.write(json.dumps({'query_id':summary['query_id'],'snapshot_id':sid,'identity':ident,
                'baseline_ids':old,'diverse_ids':new,'concepts':detail,'selected_records':[
                    {**ms[x],'original_rank':rank[x],'score':cs[x]['score'],'baseline_selected':x in old,'diverse_selected':x in new}
                    for x in dict.fromkeys(old+new)],'source_native_sha256':native_sha},ensure_ascii=False)+'\n')
            print('Temporal audit %d/50'%i,flush=True)
    assert len(public)==50 and total['memory']==72590
    try:next(groups);raise ValueError('EXTRA_SOURCE_GROUP')
    except StopIteration:pass
    assert total['removed_POSITIVE']==173 and total['added_POSITIVE']==153 and total['added_NEGATIVE']==20
    assert tokenizer_tree(TOKENIZER)==TREE
    aggregate={}
    for mode in ('baseline','diverse'):
        aggregate[mode]={k:sum(r['modes'][mode][k] for r in public) for k in public[0]['modes'][mode]
            if k not in ('records_per_concept','max_span_seconds')}
        aggregate[mode]['max_span_seconds']=max(r['modes'][mode]['max_span_seconds'] or 0 for r in public)
    loss={k:sum(r[k] for r in public) for k in ('lost_times','added_times','lost_values','added_values',
        'new_concepts','dropped_concepts','shared_concepts','shared_concepts_losing_times','shared_concepts_losing_values')}
    for field in ('earliest_baseline_retained','latest_baseline_retained','eligible_earliest_baseline',
                  'eligible_earliest_diverse','eligible_latest_baseline','eligible_latest_diverse'):
        num=sum(r[field]['numerator'] for r in public);den=sum(r[field]['denominator'] for r in public)
        loss[field]={'numerator':num,'denominator':den,'value':num/den if den else None}
    loss.update(queries_losing_times=sum(r['lost_times']>0 for r in public),queries_losing_values=sum(r['lost_values']>0 for r in public),
        interpretation='Technical same-concept timestamp/value loss; overlapping counts; not clinical harm',actual_release='NOT_RECORDED_PROXY_UNREVIEWED')
    write(pub/'temporal_record_comparison.json',{'aggregate':aggregate,'rows':public,'mapping':dict(mapping),'source_checks':dict(total)})
    write(pub/'temporal_information_loss.json',loss)
    write(pr/'replacement_record_analysis.json',{'rows':replacements,'source_negative20':negative})
    write(pub/'replacement_record_analysis.json',{'removed':173,'removed_positive':173,'added':173,'added_positive':153,'added_negative':20,
        'negative_exact_query_identity':sum(r['query_exact_identity'] is True for r in negative),
        'negative_key_missing':sum(r['concept_key'] is None for r in negative),
        'negative_original_rank_range':[min(r['original_rank'] for r in negative),max(r['original_rank'] for r in negative)],
        'negative_source_types':dict(Counter(r['source_type'] for r in negative)),
        'negative_source_reasons':dict(Counter(r['source_reason'] for r in negative)),
        'cause':'Unchanged positive lexical score plus first distinct verified concept pass; no query exact-identity restriction. Source-negative is not clinical irrelevance.',
        'private_rows_path':'outputs/private/research_evaluation/c4_23_v1/replacement_record_analysis.json'})
    totals={mode:{k:sum(r[mode][k] for r in tokens) for k in ('characters','utf8_bytes','tokens','hypothetical_metadata_chars','hypothetical_metadata_tokens')}
        for mode in ('baseline','diverse')}
    write(pub/'token_budget_comparison.json',{'scope':'ROLE_CALL_SORTED_JSON_MEMORY_RECORDS_LIST_FRAGMENT_ONLY','tokenizer':protocol['tokenizer'],
        'libraries':{'transformers':transformers.__version__,'tokenizers':tokenizers.__version__},'totals':totals,'rows':tokens,
        'max_fragment_tokens':{m:max(r[m]['tokens'] for r in tokens) for m in totals},
        'full_request_tokens':'NOT_CHECKED','actual_server_tokenizer_equivalence':'NOT_CHECKED',
        'actual_context_usage':'NOT_CHECKED','reference_context_denominator':32768,
        'hypothetical_metadata':'Only concept_metadata four strings; excludes source pointers/fingerprints/timestamps/extra audit schema; not production overhead total',
        'generation':0})
    write(pr/'runtime_raw_field_inventory.json',dict(rawfields))
    print(json.dumps({'status':'ANALYSIS_PASS','loss':loss,'token_totals':totals}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);args=p.parse_args()
    socket.create_connection=offline;socket.socket.connect=offline
    (freeze if args.phase=='freeze' else run)()
