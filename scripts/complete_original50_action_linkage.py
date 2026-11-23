# -*- coding: utf-8 -*-
"""Complete this new audit's action column using the actual Core action names."""
from collections import Counter
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read
from app.locdi.routing_collector import atomic_json
from scripts.finalize_original50_audit import safe,table
NS='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS
def main():
    base=ROOT/'outputs/private/research_evaluation/B34_CONNECTION_v1/runtime_recovery_all_001'
    index=read(base/'case_index.json');bindings=read(base/'bindings.json');byoriginal={n:bindings[ref] for ref,n in index.items()}
    latest={(r['original_ordinal'],r['condition']):r for r in read(PR/'baseline_trace_details.json')['latest']}
    with (PR/'ORIGINAL50_TEST_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    replace={};counts=Counter()
    for r in rows:
        if r['scope']!='CURRENT_ENCOUNTER':continue
        n=int(r['patient_ordinal']);bs=[b for b in byoriginal.get(n,{}).get('rows',[]) if b['source_event']['event_id']==r['source_event']];requested=[]
        for co in ('B0','B1','B2','B3','B4'):
            t=latest.get((n,co))
            if t is None:continue
            for b in bs:
                actions=('order_test','request_test') if b['kind']=='Test' else ('ask_history','ask_patient')
                if any(d.get('action') in actions and d.get('target')==b['target'] for d in t['decisions']):requested.append(co)
        r['requested_conditions']=';'.join(sorted(set(requested)))
        counts['current_source_rows_requested_at_least_once']+=bool(requested)
        counts['distinct_condition_source_row_orders']+=len(set(requested))
        if r['event_bound_received_conditions'] and not requested:counts['receipt_rows_without_order_in_actual_core_decisions']+=1
        locator=safe(r['source_file'])+':'+r['source_row']
        replace[locator]='|'+ '|'.join(safe(r[k]) for k in ('description','code','value','unit','recorded_at','recorded_proxy_available_minute','M5_Test_mappable','requested_conditions','event_bound_received_conditions'))+'|'+locator+'|'
    table(PR/'ORIGINAL50_TEST_RESULTS.csv',rows)
    main=PR/'ORIGINAL50_PATIENT_DIAGNOSTIC_SUMMARY.md';lines=main.read_text(encoding='utf-8').splitlines()
    for j,line in enumerate(lines):
        if line.startswith('|') and line.endswith('|'):
            locator=line.rsplit('|',2)[-2]
            if locator in replace:lines[j]=replace[locator]
    main.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report={'status':'CORE_ACTION_ALIAS_LINKED','core_test_action':'order_test','core_ask_action':'ask_history','counts':dict(counts),
        'no_source_or_action_created':True,'generation':0}
    atomic_json(PUB/'actual_action_linkage.json',report);print(json.dumps(report))
if __name__=='__main__':main()
