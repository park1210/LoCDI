"""C4-35 freeze first, audit second; no generation, fitting or policy scores."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_research_gaps import read,rows,write,sha,check
from scripts.evaluate_dynamic_horizon_source_policy import merge_pins
from app.locdi.llm_setup import digest
from app.locdi.voi_patient_independence_review import *
from app.locdi.voi_fresh_cohort_admission import classify_collision

NS='c4_35_v1';PUB=ROOT/'outputs/research_evaluation'/NS;PR=ROOT/'outputs/private/research_evaluation'/NS
P34=ROOT/'outputs/private/research_evaluation/c4_34_v1';U34=ROOT/'outputs/research_evaluation/c4_34_v1'
P33=ROOT/'outputs/private/research_evaluation/c4_33_v1';U33=ROOT/'outputs/research_evaluation/c4_33_v1'
DEV=ROOT/'outputs/private/research_evaluation/c4_30_v2';RAW=P34/'raw'
CODE=['app/locdi/voi_patient_independence_review.py','app/locdi/voi_fresh_cohort_admission.py',
      'scripts/review_new_cohort_independence.py','scripts/evaluate_admitted_voi_cohort.py',
      'tests/locdi/test_voi_patient_independence_review.py','tests/locdi/test_voi_fresh_cohort_admission.py']


def prepare():
    if PUB.exists() or PR.exists():raise ValueError('NO_OVERWRITE')
    # Check completed manifest structure before opening any new audit results.
    pins=read(P34/'protected_hashes.json')
    for f in (U34/'actual_completion_manifest.json',P34/'actual_private_index.json'):
        m=read(f)
        if m['fingerprint']!=digest(m['files']):raise ValueError('COMPLETION_MANIFEST')
        merge_pins(pins,m['files']);merge_pins(pins,{f.relative_to(ROOT).as_posix():sha(f)})
    PUB.mkdir(parents=True);PR.mkdir(parents=True)
    write(PR/'protected_hashes.json',pins)
    known=read(P33/'prior_raw_batch_inventory.json')
    roots=[Path(r['root']) for r in known if r['status']=='STRUCTURAL_INVENTORY']
    from app.locdi.reference_readiness import COHORT
    roots.append(COHORT)
    roots=sorted(set(p.resolve() for p in roots))
    write(PR/'comparison_roots.json',[str(p) for p in roots])
    spec={'protocol':'locdi.voi_patient_independence_review.c4_35.v1','namespace':NS,
        'prepared_at':datetime.now(timezone.utc).isoformat(),'client_date':'2026-11-18 Asia/Seoul',
        'post_C434_content_counts_design':True,'fresh_policy_scores_seen':False,
        'parent_generation_receipt_sha256':sha(P34/'actual_generation_receipt.json'),
        'raw_files_fingerprint':digest(read(P34/'actual_generation_receipt.json')['raw_files']),
        'model_sha256':'7409b58581d02fad68565e8a9ac311647c07ec968ccc5070b314f26a762c722c',
        'selection_sha256':'6f92dd6ff54e7c32f047cdefc75687923ced7cf111dcd81deb8eb49c38b10608',
        'registered_panel':{'patients':86,'visits':160,'all_candidates_retained_for_audit':True},
        'existing_encounter_signature':'exact existing encounter_fingerprints; no changed code/system/values/minutes',
        'patient_signature':'all exported CSV Encounter/Condition/Observation history relative to DOB or first encounter; demographics GENDER/RACE/ETHNICITY; other resources NOT_COMPARED',
        'history_interpretation':'exact full match is reuse suspicion; nonmatch across different export coverage is not full independence proof',
        'input_signatures':['strict raw8 strings','numeric8','recorded proxy availability not actual policy order',
            'raw8+availability','frozen ternary8','reference+raw8','reference+ternary8'],
        'blocking':'raw/receipt/identity/mapping/provenance failure or full-history reuse suspicion globally; quarantine entire candidate patient for any encounter collision or exact raw8 copy',
        'nonblocking_alone':['ternary equality','availability equality','chief prior equality'],
        'classification':'COMMON_PATTERN_SUPPORTED only verified pinned module literal condition/observation/unit/value; otherwise UNRESOLVED unless mapping/provenance/reuse evidence',
        'Gate_B':'PROPOSAL ONLY; no automatic waiver. Exclude all collision-involved candidate patients, not only their collided visits; scores never used',
        'approvals':['A revised admission exact contract','B separate admitted-panel-bound policy evaluation'],
        'audit_resources':{'cache_gib_cap':15,'wall_seconds_cap':7200,'minimum_free_gib':20,'patient_batched':True},
        'evaluation':{'primary':'patient macro Top1 Source Agreement AUBC budgets1..5','comparison':['voi_h3','global_eig'],
            'random_repeats':32,'bootstrap_repetitions':2000,'bootstrap_seed':42,'budgets':list(range(6)),
            'policies':9,'planner_call_cap':20000,'executed_states_cap':50000000,'wall_seconds_cap':3600,
            'memory_process_cap_gib':4,'storage_cap_gib':10,'Core_max_states':20000,'max_horizon':3},
        'code_hashes':{n:sha(ROOT/n) for n in CODE}}
    spec['fingerprint']=digest(spec);write(PUB/'independence_review_protocol.json',spec)
    write(PUB/'admission_approval_status.json',{'status':'NOT_APPROVED','scope':'C4_35_REVISED_ADMISSION_CONTRACT'})
    write(PUB/'evaluation_approval_status.json',{'status':'NOT_APPROVED','scope':'C4_35_FROZEN_POLICY_EVALUATION'})
    print(json.dumps({'phase':'PREPARED','fingerprint':spec['fingerprint'],'policy_evaluation':0}))


def verify():
    ready=PUB/'independence_review_protocol_execution_ready.json'
    final=ready if ready.exists() else PUB/'independence_review_protocol_final.json'
    amended=PUB/'independence_review_protocol_amended.json'
    s=read(final if final.exists() else amended if amended.exists() else PUB/'independence_review_protocol.json')
    if s['fingerprint']!=digest({k:v for k,v in s.items() if k!='fingerprint'}) or check(s['code_hashes'])['changed_or_missing']:
        raise ValueError('PROTOCOL_OR_CODE_CHANGED')
    if sha(P33/'final_m5_model.json')!=s['model_sha256'] or sha(P33/'final_action_selection.json')!=s['selection_sha256']:
        raise ValueError('FROZEN_M5_CHANGED')
    return s


def audit():
    spec=verify();gc.disable();started=time.monotonic()
    write(PR/'audit_started.json',{'protocol_fingerprint':spec['fingerprint'],'policy_scores_seen':False})
    raw=read(P34/'actual_generation_receipt.json')
    protected=check(read(PR/'protected_hashes.json'))
    if protected['changed_or_missing']:raise ValueError('PROTECTED_CHANGED')
    # Existing C4-34 exact receipt/approval/container/native metadata checks.
    from scripts.evaluate_new_voi_cohort import receipt_bundle
    genplan=read(U34/'generation_plan_final.json');receipt,metadata=receipt_bundle(genplan)
    write(PUB/'generation_receipt_reverification.json',{'status':'PASS','files':len(raw['raw_files']),
        'native_JSON':len(list((RAW/'json').glob('*.json'))),'patients':metadata['patientCount'],
        'exit_code':receipt['exit_code'],'OOMKilled':False,'image_JAR_seed_approval_checkpoint':'PASS',
        'whole_raw_hash':'PASS','prior_completed_assets':protected,'new_generation':0})
    model=read(P33/'final_m5_model.json');exclusions=read(P33/'exclusion_inventory.json')
    groups=read(P34/'content_collision_private_groups.json');ledger=read(P34/'source_eligibility_ledger.json')
    index=SourceIndex(P34/'lossless_source_cache_v2.sqlite');encmap=index.encounter_map()
    sets,intersections,collided=patient_sets(groups['cross_existing'],groups['within_new'],ledger,encmap)
    write(PR/'patient_sets.json',{'sets':{k:sorted(v) for k,v in sets.items()},'intersections':intersections})
    public_sets={'status':'COMPUTED','sets':{k:len(v) for k,v in sets.items()},
        'intersections':{k:len(v) for k,v in intersections.items()},
        'dedup':'patient ID sets; all other encounters of same patient included',
        'encounter_intersection_not_assumed_patient_independence':True}
    write(PUB/'patient_collision_intersection.json',public_sets)
    newfp={};histories={};candidates=[];newinputs=[];collision_details=[]
    with (PR/'audit_progress.jsonl').open('x',encoding='utf-8') as progress:
        for n,pid in enumerate(index.patients(),1):
            t=index.tables(pid);fp=encounter_fingerprints(t);newfp.update(fp);h=full_history(t);histories[pid]=h
            vs=project_patient(t,RAW,model)
            complete_ids={r['visit_id'] for r in ledger if r['patient_id']==pid and r['complete']}
            for v in vs:
                sig=input_signatures(v,model)
                if v['visit_id'] in complete_ids:
                    if sig is None or v['diagnosis'] is None:raise ValueError('CANDIDATE_RECONSTRUCTION')
                    candidates.append(v);newinputs.append({'patient_id':pid,'visit_id':v['visit_id'],'signatures':sig})
            linkedconds=defaultdict(list);linkedobs=defaultdict(list)
            for _,r in t['conditions']:linkedconds[r['ENCOUNTER']].append(r)
            for _,r in t['observations']:linkedobs[r['ENCOUNTER']].append(r)
            for eid in set(fp)&(collided['cross_encounters']|collided['within_encounters']):
                collision_details.append({'patient_id':pid,'visit_id':eid,'fingerprint':fp[eid],
                    'conditions':linkedconds[eid],'observations':linkedobs[eid],
                    'encounter':next(r for _,r in t['encounters'] if r['Id']==eid),
                    'same_patient_is_complete_candidate':pid in sets['E'],'full_patient_history':h})
            if n%50==0:progress.write(json.dumps({'stage':'new','patients':n,'elapsed':time.monotonic()-started})+'\n');progress.flush()
        index.close()
        oldfp=set(exclusions['encounter_fingerprints'].values());allnew=defaultdict(list)
        for eid,fp in newfp.items():allnew[fp].append(eid)
        reproduction={'status':'PASS','cross_existing_rows':sum(fp in oldfp for fp in newfp.values()),
            'cross_existing_groups':len(set(newfp.values())&oldfp),
            'within_new_rows':sum(len(v) for v in allnew.values() if len(v)>1),
            'within_new_excess':len(newfp)-len(allnew),'within_new_groups':sum(len(v)>1 for v in allnew.values())}
        if [reproduction[k] for k in ['cross_existing_rows','cross_existing_groups','within_new_rows','within_new_excess','within_new_groups']] != [294,261,462,241,221]:
            raise ValueError('C434_COLLISION_COUNTS_NOT_REPRODUCED')
        write(PUB/'encounter_collision_reproduction.json',reproduction)
        write(PR/'candidate_visits.json',sorted(candidates,key=lambda v:(v['patient_id'],v['visit_id'])))
        write(PR/'candidate_input_signatures.json',newinputs);write(PR/'new_patient_history.json',histories)
        write(PR/'collision_details.json',collision_details)
        if len(candidates)!=160 or len({v['patient_id'] for v in candidates})!=86:raise ValueError('CANDIDATE_DENOMINATOR')
        oldhist=[];oldinputs=[];sourcepins={};roots=read(PR/'comparison_roots.json')
        for k,rootstr in enumerate(roots,1):
            root=Path(rootstr);cache=PR/('old_source_%02d.sqlite'%k)
            for name in REQUIRED:sourcepins[str(root/'csv'/(name+'.csv'))]=sha(root/'csv'/(name+'.csv'))
            build_index(root,cache)
            oldindex=SourceIndex(cache)
            for pid in oldindex.patients():
                t=oldindex.tables(pid);oldhist.append({'source_root':rootstr,'patient_id':pid,**full_history(t)})
                for v in project_patient(t,root,model):
                    sig=input_signatures(v,model)
                    if sig:oldinputs.append({'source_root':rootstr,'patient_id':pid,'visit_id':v['visit_id'],'signatures':sig})
            oldindex.close()
            progress.write(json.dumps({'stage':'old','roots_done':k,'old_histories':len(oldhist),'elapsed':time.monotonic()-started})+'\n');progress.flush()
            if time.monotonic()-started>7200 or sum(f.stat().st_size for f in PR.glob('*.sqlite'))>15*1024**3:raise ValueError('AUDIT_RESOURCE_CAP')
    # Add exact previously projected development inputs; do not overwrite root-derived evidence.
    for name in ['base_visits.json','additional_visits.json']:
        for v in read(DEV/name):
            sig=input_signatures(v,model)
            if sig:oldinputs.append({'source_root':'C430/'+name,'patient_id':v['patient_id'],'visit_id':v['visit_id'],'signatures':sig})
    write(PR/'old_patient_history.json',oldhist);write(PR/'old_input_signatures.json',oldinputs);write(PR/'comparison_source_hashes.json',sourcepins)
    finish_review(spec,sets,intersections,histories,oldhist,newinputs,oldinputs,candidates,collision_details,reproduction)


def finish_review(spec,sets,intersections,histories,oldhist,newinputs,oldinputs,candidates,details,reproduction):
    old_hist_map=defaultdict(list)
    for h in oldhist:old_hist_map[h['fingerprint']].append(h)
    reused={p for p,h in histories.items() if h['fingerprint'] in old_hist_map and h['encounters']>0}
    write(PR/'full_history_matches.json',{p:old_hist_map[histories[p]['fingerprint']] for p in reused})
    write(PUB/'patient_lineage_analysis.json',{'status':'EXPORTED_HISTORY_SCOPE_REVIEWED','new_patient_histories':len(histories),
        'old_exported_history_instances':len(oldhist),'old_unique_patients':len({r['patient_id'] for r in oldhist}),
        'full_exported_history_match_patients':len(reused),'candidate_history_match_patients':len(reused&sets['E']),
        'new_seed':2026111601,'new_batch':'c4_34_v1','generation_receipt_verified':True,
        'native_other_resources_full_history':'NOT_COMPARED','old_export_windows_may_differ':True,
        'nonmatch_proves_independence':False,'clinical_independence':'NOT_ASSERTED'})
    keys=['raw8','numeric8','availability','raw8_and_availability','ternary','reference_and_raw8','reference_and_ternary']
    overlaps={};duplicate_pids=set()
    for key in keys:
        prior={r['signatures'][key] for r in oldinputs if r['signatures'].get(key)}
        newgroups=defaultdict(list)
        for r in newinputs:
            if r['signatures'].get(key):newgroups[r['signatures'][key]].append(r)
        hits=[r for r in newinputs if r['signatures'].get(key) in prior and r['signatures'].get(key)]
        within=[g for g in newgroups.values() if len(g)>1]
        overlaps[key]={'old_complete_input_instances':len(oldinputs),'new_input_instances':len(newinputs),
            'cross_old_matched_visits':len(hits),'cross_old_matched_patients':len({r['patient_id'] for r in hits}),
            'within_new_groups':len(within),'within_new_rows':sum(map(len,within)),
            'within_new_cross_patient_groups':sum(len({r['patient_id'] for r in g})>1 for g in within),
            'distinct_new_patterns':len(newgroups)}
        if key=='raw8':
            duplicate_pids|={r['patient_id'] for r in hits}
            duplicate_pids|={r['patient_id'] for g in within for r in g}
    write(PR/'raw_input_duplicate_patients.json',sorted(duplicate_pids))
    write(PUB/'evaluation_input_overlap_analysis.json',{'status':'STRUCTURAL_INPUT_AUDIT_ONLY','signatures':overlaps,
        'actual_policy_disclosure_order':'NOT_RUN','availability_signature_not_actual_trace':True,
        'old_role_coverage':'all available prior raw roots plus actual C430 base229/augmentation142; Original50/D3/old evaluation represented by patient linkage to raw; prior LLM payloads not reconstructed',
        'unknown_external_development_exposure':'NOT_CHECKED'})
    write(PUB/'frozen_encoding_overlap_analysis.json',{'status':'FROZEN_THRESHOLD_APPLICATION_NO_FIT',
        'ternary':overlaps['ternary'],'reference_and_ternary':overlaps['reference_and_ternary'],
        'raw_input_match_separate':overlaps['raw8'],'finite_combinations':3**8,
        'ternary_match_proves_copy':False,'reference_used_for_audit_only':True,'policy_scores_calculated':False})
    classifications=Counter();classified=[]
    for d in details:
        status=classify_collision(full_history_match=d['patient_id'] in reused)
        classified.append({'patient_id':d['patient_id'],'visit_id':d['visit_id'],'status':status,
            'condition_count':len(d['conditions']),'observation_count':len(d['observations']),
            'module_execution_provenance':'UNKNOWN; do not infer module from code alone'})
        classifications[status]+=1
    write(PR/'collision_classification.json',classified)
    write(PUB/'collision_pattern_classification.json',{'status':'STRUCTURAL_REVIEW_NO_CLINICAL_JUDGMENT',
        'collision_encounter_occurrence_counts':dict(classifications),
        'unit':'union of collided new encounters, not pattern group count',
        'module_specific_cause':'UNKNOWN','common_pattern_supported_automatically':False,
        'record_count_distribution':dict(Counter(str((len(d['conditions']),len(d['observations']))) for d in details)),
        'full_record_reuse_suspicion_separate':len(reused)})
    global_blockers=['FULL_EXPORTED_HISTORY_REUSE_SUSPECTED'] if reused else []
    proposal=conservative_proposal(candidates,sets['C'],reused,duplicate_pids,global_blockers)
    write(PR/'proposed_panel_review.json',proposal)
    contract={'version':'locdi.patient_quarantine_admission.proposal.c4_35.v1','parent_review_fingerprint':spec['fingerprint'],
        'proposal_fingerprint':digest(proposal),'original_gate':'UNCHANGED_ADMISSION0',
        'rule':'quarantine entire patient for any collided encounter or exact raw8 input copy; global fatal integrity/provenance/reuse failures still block all',
        'unresolved_collisions':'no COMMON_PATTERN_SUPPORTED claim; all affected candidate patients excluded in proposal, no selective encounter removal',
        'limits':'CSV history only, old export windows differ, exact nonmatch not proof of independence; user scope review required',
        'reference':'UNREVIEWED_SOURCE_DIAGNOSIS','clinical_approval':False,'approval_scope':'C4_35_REVISED_ADMISSION_CONTRACT',
        'proposed_patients':proposal['proposed_patients'],'proposed_visits':proposal['proposed_visits'],
        'current_admitted_patients':0,'current_admitted_visits':0}
    contract['fingerprint']=digest(contract);write(PUB/'proposed_admission_contract.json',contract)
    write(PUB/'original_gate_reproduction.json',{'status':'SOURCE_OVERLAP_BLOCKED','Gate_A_patients':0,'Gate_A_visits':0,'counts':reproduction,'original_results_unchanged':True})
    before=Counter(v['diagnosis'] for v in candidates);after=Counter(v['diagnosis'] for v in proposal['proposed'])
    write(PUB/'admission_contract_comparison.json',{'structural_candidates':{'patients':86,'visits':160},'Gate_A':{'patients':0,'visits':0},
        'Gate_B_proposal_only':{'patients':proposal['proposed_patients'],'visits':proposal['proposed_visits']},
        'excluded_candidate_patients':proposal['excluded_patients'].__len__(),'excluded_candidate_visits':proposal['excluded_visits'],
        'reference_class_counts_before':dict(before),'reference_class_counts_after_proposal':dict(after),
        'before_visits_per_patient':dict(Counter(Counter(v['patient_id'] for v in candidates).values())),
        'after_visits_per_patient':dict(Counter(Counter(v['patient_id'] for v in proposal['proposed']).values())),
        'global_blockers':global_blockers,'approval_status':'NOT_APPROVED','clinical_reference_review':'NOT_APPROVED'})
    write(PUB/'final_admission_manifest.json',{'status':'NOT_ADMITTED_AWAITING_CONTRACT_APPROVAL','admitted_patients':0,'admitted_visits':0,
        'contract_fingerprint':contract['fingerprint'],'proposed_patients':proposal['proposed_patients'],'proposed_visits':proposal['proposed_visits']})
    ev={'version':'locdi.fresh_voi_policy_evaluation.c4_35.v1','approval_scope':'C4_35_FROZEN_POLICY_EVALUATION',
        'admission_panel_fingerprint':None,'status':'BLOCKED_PENDING_ADMISSION_APPROVAL','primary':spec['evaluation']['primary'],
        'comparison':spec['evaluation']['comparison'],'settings':spec['evaluation'],'score_selection_forbidden':True}
    ev['fingerprint']=digest(ev);write(PUB/'evaluation_protocol.json',ev)
    print(json.dumps({'status':'REVIEW_COMPLETE_ADMISSION_NOT_APPROVED','patient_sets':{k:len(v) for k,v in sets.items()},
        'intersections':{k:len(v) for k,v in intersections.items()},'proposed_patients':proposal['proposed_patients'],
        'proposed_visits':proposal['proposed_visits'],'contract_fingerprint':contract['fingerprint'],'policy_evaluation':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','audit']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else audit()
