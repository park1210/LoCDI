"""Saved C3-16 evidence checks and finite public/private boundary audit."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.locdi.dataset_freeze import load,sha
from app.locdi.llm_setup import digest
from app.locdi.episode_exporter import _write
from app.locdi.agentclinic_scenario_adapter import load_scenarios,action_catalog
from app.locdi.agentclinic_live_preflight import SOURCE
from app.locdi.agentclinic_state_actions import KINDS
from scripts.audit_c3_01 import privacy_check

def verify(document=None):
    public=ROOT/'outputs/research_evaluation/c3_16_v1';private=ROOT/'outputs/private/research_evaluation/c3_16_v1'
    baseline=load(private/'reports/baseline_hashes.json');changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    bundle=load_scenarios(SOURCE);scenarios={s.scenario_id:s for s in bundle.scenarios};tests=[];counts={};private_text=set()
    keys={k.scenario_id:k for k in bundle.scoring_keys}
    def strings(v):
        if isinstance(v,str):
            if len(v)>=24:private_text.add(v)
        elif isinstance(v,dict):
            for x in v.values():strings(x)
        elif isinstance(v,list):
            for x in v:strings(x)
    for c in ('A','C'):
        run=ROOT/('outputs/private/research_evaluation/c3_16_'+c+'_001');out=ROOT/('outputs/research_evaluation/c3_16_'+c+'_001')
        befores=sorted((run/'checkpoints').glob('*_before.json'));afters=sorted((run/'checkpoints').glob('*_after.json'))
        tests.append(len(befores)==len(afters));token_equal=0;request_total=0;gold_mentions=0
        for bp,ap in zip(befores,afters):
            b,a=load(bp),load(ap);ordinal=b['ordinal'];s=scenarios[b['scenario_ref']];catalog=action_catalog(s)
            consumed={(x['action'],x['target']) for x in b['consumed_before']}
            expected={k:[t for t in catalog[k] if c=='A' or (k,t['target']) not in consumed] for k in KINDS}
            tests.append(b['payload']['catalog']==expected)
            tests.append(b['original_catalog_fingerprint']==digest(catalog))
            tests.append(all(o['released_turn']<b['source_turn_index'] for o in b['payload']['observations']))
            tests.append(a['context_observation']['status']=='VALID' and a['context_observation']['loaded_model']['context_length']==32768)
            tests.append('sha256:'+a['context_observation']['loaded_model']['digest']=='sha256:65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b')
            tests.append(a['identity_count']==a['generation_count']==request_total+1);request_total+=1
            tests.append(b['input_tokens']+1024+512<=32768 and b['utf8_bytes']<=16384)
            event=a['policy_event'];policy=load(run/('policy_traces/Scenario%03d.json'%ordinal));tests.append(event==policy['events'][a['turn_index']-1])
            ep=load(run/('episodes/Scenario%03d.json'%ordinal));audit=ep['doctor_audit'][a['turn_index']-1]
            meta=audit.get('response_metadata',{})
            server_tokens=meta.get('prompt_eval_count')
            if server_tokens==b['input_tokens']:token_equal+=1
            tests.append(audit['status']=='ok' or audit['status']=='success')
            strings(a['response_content']);strings(ep['result'].get('predicted_diagnosis'));strings(keys[s.scenario_id].correct_diagnosis)
            for obs in b['payload']['observations']:strings(obs['value'])
            if re.search(r'(?<!\w)'+re.escape(keys[s.scenario_id].correct_diagnosis)+r'(?!\w)',json.dumps(b['payload']),re.I):gold_mentions+=1
        counts[c]={'request_count':request_total,'hf_server_prompt_count_equal':token_equal,
            'reference_literal_mentions_in_model_payload':gold_mentions}
        tests.append(privacy_check(out)['status']=='PASS')
    roots=[public,ROOT/'outputs/research_evaluation/c3_16_A_001',ROOT/'outputs/research_evaluation/c3_16_C_001']
    files=[p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.txt')]
    if document and document.is_file():files.append(document)
    hits=[]
    for p in files:
        text=p.read_text(encoding='utf8')
        if any(v in text for v in private_text):hits.append(p.relative_to(ROOT).as_posix())
    result={'status':'PASS' if all(tests) and not changed and not hits else 'FAIL',
        'protected_file_count':len(baseline),'protected_changed_count':len(changed),
        'evidence_checks':len(tests),'evidence_checks_passed':sum(tests),'per_condition':counts,
        'public_private_literal_matches':len(hits),'public_files_scanned':len(files),
        'scope':'Saved adapter content, exact IDs, fingerprints, source release turns and finite known private literals; not raw wire or independent clinical privacy review',
        'clinical_metrics':'NOT_EVALUATED','new_generation_count':0}
    _write({'changed':changed,'literal_hit_paths':hits},'reports/verification_private_details.json',private,'reports')
    _write(result,'reports/final_verification.json',public,'reports');return result

if __name__=='__main__':
    document=ROOT/'docs/연구 평가/C3_16_AGENTCLINIC_STATE_AWARE_ACTION_CONTROL_REPORT.md'
    print(json.dumps(verify(document),ensure_ascii=True))
