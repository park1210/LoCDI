"""No Generation unless execute AND allow-llm AND a new operator approval."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.synthetic_pilot_collection import prepare,check,collect,analyze
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','check','collect','analyze','fit'])
    p.add_argument('--plan-id',required=True);p.add_argument('--count',type=int,choices=[1,15],default=1)
    p.add_argument('--view',choices=['full','source_compatible'],default='full');p.add_argument('--stage',choices=['initial','all_tests'],default='all_tests')
    p.add_argument('--start-ordinal',type=int,default=1,help='Smoke source ordinal; full15 always starts at1')
    p.add_argument('--execute',action='store_true');p.add_argument('--allow-llm',action='store_true');p.add_argument('--approval',type=Path)
    a=p.parse_args()
    if a.mode!='collect' and (a.execute or a.allow_llm or a.approval):p.error('LIVE_FLAGS_COLLECT_ONLY')
    if a.mode!='prepare' and a.start_ordinal!=1:p.error('ORDINAL_FIXED_AT_PREPARE')
    if a.mode=='prepare':r=prepare(a.plan_id,count=a.count,view=a.view,stage=a.stage,start_ordinal=a.start_ordinal)
    elif a.mode=='check':
        plan=check(a.plan_id);r={'status':'STATIC_PINS_AND_SOURCE_VALID','fingerprint':plan['fingerprint'],'generation_cap':plan['generation_cap'],'actual_LLM_generation':0}
    elif a.mode=='collect':r=collect(a.plan_id,execute=a.execute,allow_llm=a.allow_llm,approval=a.approval)
    else:r=analyze(a.plan_id,fit=a.mode=='fit')
    print(json.dumps(r))
    if a.mode=='collect' and r.get('status')=='PARTIAL_OR_FAILED':raise SystemExit(2)
if __name__=='__main__':main()
