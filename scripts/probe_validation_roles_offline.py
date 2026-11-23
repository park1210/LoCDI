"""Network-denied six-role Core replay, no LLM/fitting."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from unittest.mock import patch
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_validation_role_snapshot import load,one,DEFAULT
from scripts.probe_terminal_only_replay_offline import TerminalFake
from app.locdi.validation_role_snapshot import VERSION
from app.locdi.current_ablation_runner import GenerationBudget,_BUDGET
from app.locdi.original_source_inventory import read
from app.locdi.routing_collector import atomic_json


class RoleFake(TerminalFake):
    def post(self,url,*,json):
        # Reuse the finite citation/Memory Schema fixture; only its task marker
        # differs from the M5-specific terminal fixture. No Source fields change.
        from copy import deepcopy
        p=__import__('json').loads(json['messages'][1]['content'])
        if p['terminal_request_contract']!=VERSION:raise AssertionError('ROLE_CONTRACT')
        wire=deepcopy(json);p['terminal_request_contract']='frozen_m5_stop_terminal_only.v1'
        wire['messages'][1]['content']=__import__('json').dumps(p)
        return super().post(url,json=wire)


def main(name,output):
    pub,pr,p=load(name);folder=pr/output;folder.mkdir(exist_ok=False)
    patients={r['ordinal']:r for r in read(pr/'patients.json')};cfg=read(pr/'config.json');a=read(pr/'approval_template.json');now=datetime.now(timezone.utc)
    for k in a:
        if k.startswith('accept_') or k in ('operator_approved','no_other_ollama_work'):a[k]=True
    a.update(operator_ref='TEST_ONLY_NOT_APPROVAL',user_statement='TEST_ONLY_NETWORK_DENIED',recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    rows=[]
    for t in read(pr/'tasks.json'):
        budget=GenerationBudget(1,1);token=_BUDGET.set(budget);fake=RoleFake(cfg)
        try:
            with patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')),\
                    patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')):
                r=one(t,patients[t['validation_ordinal']],p=p,config=cfg,authorization=a,folder=folder/('Task%02d'%t['task_ordinal']),factory=lambda _:fake)
            if r['status']!='COMPLETED' or r['certainty']!='unresolved' or r['source_exact'] is not None or budget.used!=1 or len(fake.posts)!=1:raise AssertionError('ONE_UNRESOLVED_CORE_VALIDATED_RESPONSE')
            rows.append({'task_ordinal':t['task_ordinal'],'role_ref':t['role_ref'],'status':'PASS','actual_generation':0})
        finally:_BUDGET.reset(token)
    atomic_json(pub/'TEST_ONLY_validation.json',{'status':'PASS','tasks':rows,'plan_fingerprint':p['fingerprint'],
        'actual_LLM_generation':0,'utility_fitting':0,'network':'DENIED','real_role_performance':'NOT_EVALUATED'})
    print(json.dumps({'status':'PASS','TEST_ONLY_roles':len(rows),'actual_generation':0}))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan-id',default=DEFAULT);q.add_argument('--output-id',default='TEST_ONLY_001');a=q.parse_args()
    if not a.output_id.startswith('TEST_ONLY_') or not a.output_id.replace('_','').isalnum():raise ValueError('TEST_ONLY_OUTPUT_REQUIRED')
    main(a.plan_id,a.output_id)
