"""C3-13A pilot: default DRY_RUN, separate Fake and explicitly approved Live modes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.locdi.agentclinic_pilot_readiness import readiness,PinnedPilotApproval
from app.locdi.agentclinic_live_pilot_runner import run_fake,run_live


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',required=True,help='NEW namespace; no resume')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--fake',action='store_true',help='Offline TEST_ONLY JSON Doctor')
    modes.add_argument('--execute',action='store_true',help='Live requires all independent approval arguments')
    parser.add_argument('--allow-llm',action='store_true')
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--approval-sha256')
    args=parser.parse_args()
    if not args.execute:
        def guard(event,values):
            if event in {'socket.connect','socket.connect_ex','socket.bind'}:raise RuntimeError('PILOT_OFFLINE_NO_NETWORK')
        sys.addaudithook(guard)
    authority= PinnedPilotApproval(args.approval,args.approval_sha256) if args.approval and args.approval_sha256 else None
    try:
        if args.fake:
            if args.allow_llm or authority is not None:parser.error('FAKE_AND_LIVE_AUTHORITY_MUST_BE_SEPARATE')
            from scripts.prepare_agentclinic_doctor import ScriptedJson,source_plan
            from app.locdi.agentclinic_scenario_adapter import load_scenarios
            from app.locdi.agentclinic_live_preflight import SOURCE
            actions=[a for s in load_scenarios(SOURCE).scenarios[:5] for a in source_plan(s)]
            result=run_fake(args.namespace,ScriptedJson(actions))
        elif args.execute:
            result=run_live(args.namespace,authority=authority,execute=True,allow_llm=args.allow_llm)
        else:result=readiness(args.namespace,authority)
        print(json.dumps(result,ensure_ascii=True))
        return 0 if result.get('mode')=='DRY_RUN' or result.get('status') in ('COMPLETED','UNRESOLVED') else 2
    except Exception:
        print(json.dumps({'status':'BLOCKED_OR_FAILED','error_code':'PILOT_GATE_OR_STORAGE_FAILED',
                          'existing_results_preserved':True,'automatic_resume':False}))
        return 2


if __name__=='__main__':sys.exit(main())
