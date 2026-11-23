"""Distinguish export formatting differences from absent Native measurements."""
from collections import Counter
from decimal import Decimal,InvalidOperation
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.original_source_inventory import read,original_cases
from app.locdi.routing_collector import atomic_json
NS='ORIGINAL50_PATIENT_DIAGNOSTIC_AUDIT_v1';PR=ROOT/'outputs/private/research_evaluation'/NS;PUB=ROOT/'outputs/research_evaluation'/NS

def classification(native,csvrows):
    same=[r for r in csvrows if str(r['target']['code'])==native['code'] and (r['evidence'].get('units') or '')==(native['unit'] or '')]
    if not same:return 'NATIVE_CODE_UNIT_ABSENT_FROM_CURRENT_CSV'
    for r in same:
        a=str(native['value']);b=str(r['evidence'].get('value'))
        if a==b:return 'EXACT_CODE_VALUE_UNIT'
        if isinstance(native['value'],dict) and native['value'].get('display')==r['evidence'].get('value'):
            return 'CODED_VALUE_DISPLAY_EXACTLY_EQUALS_CSV'
        try:
            if Decimal(a)==Decimal(b):return 'NUMERICALLY_EQUAL_TEXT_FORMAT_DIFFERS'
            if format(float(a),'.1f')==b:return 'COMPATIBLE_WITH_ONE_DECIMAL_EXPORT_NOT_RELEASE_APPROVAL'
        except (InvalidOperation,ValueError,OverflowError):pass
    return 'SAME_CODE_UNIT_DIFFERENT_VALUE_UNRESOLVED_EXPORT_DIFFERENCE'

def main():
    data=read(PR/'native_completeness_details.json');cases=original_cases();counts=Counter();codes=Counter();rows=[]
    for d,c in zip(data,cases):
        evs=[e for e in c['environment']['source_event_store'] if e['event_type']=='observations']
        for r in d['native_only_rows']:
            status=classification(r,evs);counts[status]+=1
            if status=='NATIVE_CODE_UNIT_ABSENT_FROM_CURRENT_CSV':codes[(r['code'],r['description'],r['unit'])]+=1
            rows.append({'patient_ordinal':d['patient_ordinal'],**r,'comparison_status':status})
    atomic_json(PR/'native_export_difference_classification_refined.json',rows)
    report={'scope':'Only the 1911 literal Native code/value/unit nonmatches; not 1911 proven missing measurements.',
        'counts':dict(counts),'code_unit_absent_from_csv':[{'code':c,'description':d,'unit':u,'rows':n} for (c,d,u),n in codes.items()],
        'no_values_replaced':True,'input_release_grants_created':0,'clinical_availability':'NOT_REVIEWED','LLM_generation':0}
    atomic_json(PUB/'native_export_difference_summary_refined.json',report);print(json.dumps(report))
if __name__=='__main__':main()
