"""TEST_ONLY strict-wire Core integration; deny network, never actual Smoke."""
from copy import deepcopy
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_original50_d0_d2 import load, SMOKE, AUG, PR, PUB, CONDITIONS, M5, UTILITY
from scripts.probe_original50_d0_d2_offline import Fake
from app.locdi.original_source_inventory import original_cases, read
from app.locdi.augmentation_experiment_backend import make_paired_backend
from app.locdi.context_budgeted_history import ContextBudgetedHistory
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.current_ablation_runner import GenerationBudget, _BUDGET, features_for
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.locdi.strict_evidence_selection_wire import VERSION
from jsonschema import Draft202012Validator


def main():
    _,pr,p,bp=load(SMOKE);idx=read(bp/'case_index.json');ref=next(iter(idx));case=original_cases()[idx[ref]-1]
    aug={r['case_ordinal']:r for r in (json.loads(s) for s in (AUG/'AUGMENTED_CASES.jsonl').read_text(encoding='utf-8').splitlines())}
    a=read(pr/'approval_template.json');now=datetime.now(timezone.utc)
    a.update(operator_ref='TEST_ONLY_NOT_OPERATOR_APPROVAL',user_statement='TEST_ONLY_NO_NETWORK',
             recorded_at=now.isoformat(),expires_at=(now+timedelta(hours=1)).isoformat())
    for k in ('operator_approved','no_other_ollama_work','accept_source_label_proxy','accept_recorded_time_proxy',
              'accept_common_m5_menu','accept_label_conditioned_development','accept_augmentation_initial_time_policy'):a[k]=True
    outroot=PR/'wire_recovery_v1/TEST_ONLY';outroot.mkdir(exist_ok=False);results=[];case_fp=digest(case)
    import app.locdi.evidence_selection_wire as v1
    previous_encode=v1.encode;previous_version=v1.VERSION
    for view in ('D0','D2'):
        for cond in CONDITIONS:
            name=view+'_'+cond;config=read(bp/'llm_config.json');fake=Fake(config);token=_BUDGET.set(GenerationBudget(12,12))
            try:
                with ExitStack() as s:
                    s.enter_context(patch('socket.socket.connect',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                    s.enter_context(patch('socket.create_connection',side_effect=AssertionError('NETWORK_FORBIDDEN_TEST_ONLY')))
                    s.enter_context(patch('app.locdi.current_ablation_llm_backend._native_transport',return_value=fake))
                    backend=make_paired_backend({ref:case},dataset_view=view,augmented_rows={ref:aug[idx[ref]]},catalog=evidence_catalog(),
                        config=config,bindings=read(bp/'bindings.json'),model=read(M5),authorization=a,
                        plan_fingerprint=p['fingerprint'],checkpoint_root=outroot/name,routing_evidence=read(UTILITY),
                        history_policy=ContextBudgetedHistory(read(bp/'history_selection_plans.json')),
                        experiment_kind='DEVELOPMENT_EXPERIMENT',catalog_selection=True,runtime_recovery=True,evidence_set=True)
                    result=backend({'case_ref':ref},features_for(cond))
                assert result['status']=='completed',name
                for request in fake.wires:
                    body=json.loads(request['messages'][1]['content']);assert VERSION in body['instruction']
                    for branch in request['format'].get('anyOf',[request['format']]):
                        field=branch['properties'].get('evidence_selection')
                        if field and field.get('maxProperties')==0:
                            assert field['properties']=={} and field['enum']==[{}]
                assert digest(case)==case_fp
                assert v1.encode is previous_encode and v1.VERSION==previous_version
                for after in (outroot/name/'Episode0001').glob('Request*_after.json'):
                    assert read(after)['status']=='VALID'
                results.append({'profile':name,'status':'PASS','core_status':result['status'],'fake_requests':len(fake.wires)})
            finally:_BUDGET.reset(token)
    # Replay the actual failed response against both schemas. Remains illegal.
    old=PR/'original50_d0d2_smoke_003/run_original50_d0d2_smoke_003/requests/D0/Episode0001'
    before=read(old/'Request0002_before.json');value=json.loads(read(old/'Request0002_after.json')['response_envelope']['message']['content'])
    schema=deepcopy(before['request']['format'])
    for branch in schema['anyOf']:
        field=branch['properties']['evidence_selection']
        if field.get('maxProperties')==0:field.update(properties={},enum=[{}])
    assert not Draft202012Validator(schema).is_valid(value)
    atomic_json(PUB/'wire_recovery_v1/TEST_ONLY_core_probe.json',{'TEST_ONLY':True,'profiles':results,
        'actual_failed_response_still_rejected':True,'original_case_unchanged':True,'legacy_wire_restored':True,
        'actual_LLM_generation':0,'actual_smoke_executed':False})
    print(json.dumps({'PASS':len(results),'actual_LLM_generation':0,'actual_smoke_executed':False}))


if __name__=='__main__':main()
