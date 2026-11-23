"""Read-only routing/data audit and isolated authored Train-view proposals.

Never calls fit_utility, a model, clinical reviewer or an admission override.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_validation_role_snapshot import PUB,PR
from scripts.run_u2_native_collection import load as native_load
from scripts.run_u1_voi_expansion import load as eval_load
from scripts.analyze_b34_utility_voi import U1
from scripts.build_original50_literature_augmentation import POLICY
from app.locdi.original_source_inventory import read,sha_file,original_cases
from app.locdi.original50_augmentation import scenario,directly_discloses_label
from app.locdi.literature_augmentation_catalog import evidence_catalog
from app.locdi.llm_setup import digest
from app.locdi.routing_collector import atomic_json
from app.core.routing_utility import RoutingUtility,observed_tokens,jaccard_similarity
from app.core.agent_registry import AgentRegistry
from app.core.longitudinal_memory import PatientMemoryStore
from app.core.clinical_observation import memory_query


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:raise FileExistsError('NEW_ANALYSIS_NAMESPACE_REQUIRED')
    else:atomic_json(path,value)


def fit_status(registry,evaluation_patients):
    folder=ROOT/'outputs/private/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/u2_native_development_fit_001'
    f=folder/'U2_native_development_utility.json'
    if not f.exists():return {'status':'FIT_PARTIAL_OR_BLOCKED_INSPECT_BEFORE_ANY_RETRY' if folder.exists() else 'NOT_FITTED',
        'artifact_exists':False,'artifact_sha256':None,'fit_namespace_exists':folder.exists(),'U2_B3_performance':'NOT_VERIFIED','fit_called_by_this_analysis':False},None
    wrapper=read(f);artifact=wrapper['artifact'];rows=read(folder/'fit_rows_private.json')
    utility=RoutingUtility(artifact,AgentRegistry(registry))
    expected=[{'patient_id':r['patient_id'],'decision_id':r['decision_id'],'agent_id':r['agent_id'],'correct':r['correct'],'tokens':observed_tokens(r['observation'])} for r in rows]
    if sorted(map(digest,expected))!=sorted(map(digest,artifact['samples'])) or artifact['min_samples']!=2 or utility.training_patients&set(evaluation_patients):raise ValueError('EXACT_NATIVE_ARTIFACT_SAMPLES_OR_SPLIT')
    for path,h in wrapper['origin_files'].items():
        if sha_file(path)!=h:raise ValueError('U2_ORIGIN_CHANGED')
    public=ROOT/'outputs/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/u2_native_development_fit_001/fit_result.json'
    if not public.exists() or read(public)['artifact_sha256']!=sha_file(f):raise ValueError('ACTUAL_FIT_RESULT_SHA')
    return {'status':'FITTED_ORIGIN_AND_SAMPLE_SHA_VERIFIED','artifact_exists':True,'artifact_sha256':sha_file(f),
        'sample_count':len(artifact['samples']),'groups':len({(r['patient_id'],r['decision_id']) for r in artifact['samples']}),
        'B3_B4_admission':wrapper['B3_B4_admission'],'U2_B3_performance':'NOT_VERIFIED','fit_called_by_this_analysis':False},utility


def main():
    if (PUB/'routing_root_summary.json').exists():raise FileExistsError('NO_COMPLETED_ANALYSIS_OVERWRITE')
    PUB.mkdir(parents=True,exist_ok=True);PR.mkdir(parents=True,exist_ok=True)
    _,npr,np=native_load('Full');run=npr/np['run_id'];patients=read(npr/'patients.json');analysis=read(run/'analysis_private.json')
    _,epr,ep,bp=eval_load();erun=epr/ep['run_id'];evaluation=read(erun/'analysis_private.json')
    source_cases=original_cases();status,u2=fit_status(np['registry'],[c['patient_id'] for c in source_cases])
    protected=dict(np['pins']);protected.update(ep['pins'])
    protected.update({str(f.resolve()):sha_file(f) for f in run.rglob('*') if f.is_file()})
    for folder in (ROOT/'app/core',ROOT/'validation',ROOT/'prompts'):
        protected.update({str(f.resolve()):sha_file(f) for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts})
    put(PR/'preservation_before.json',protected)
    native_initial={};train=[]
    for patient in patients:
        hashes=set()
        for a in np['agents']:
            f=run/('Patient%02d'%patient['ordinal'])/a['department']/'Episode0001/Request0001_before.json'
            p=json.loads(read(f)['request']['messages'][1]['content']);hashes.add(digest(p['observation']))
            if a['department']=='cardiology':native_initial[patient['ordinal']]=p['observation']
        if len(hashes)!=1:raise ValueError('FIVE_AGENT_SAME_INITIAL_OBSERVATION')
        if patient['split']=='train':train.append(patient)
    outcomes={(r['ordinal'],r['agent_id']):r['correctness_proxy'] for r in analysis['rows']}
    counts={a['department']:dict(Counter(str(outcomes[(p['ordinal'],a['agent_id'])]) for p in train)) for a in np['agents']}
    public_evaluations=[];private_evaluations=[]
    u1_wrapper=read(U1);u1_artifact=u1_wrapper['artifact'];u1registry=AgentRegistry(np['registry'],u1_artifact)
    for row in evaluation:
        if row['condition']!='B3':continue
        folder=erun/'requests'/('Episode%04d'%row['episode_ordinal']);files=sorted(folder.glob('Request*_before.json'))
        for phase,f in (('initial',files[0]),('last',files[-1])):
            observation=json.loads(read(f)['request']['messages'][1]['content'])['observation'];tok=observed_tokens(observation)
            overlap=[{'train_ordinal':p['ordinal'],'jaccard':jaccard_similarity(tok,observed_tokens(native_initial[p['ordinal']]))} for p in train]
            route=u1registry.select_specialty(observation,observation['longitudinal_memory'],[a['department'] for a in np['agents']],router='adaptive',seed=42)
            u2_route=(AgentRegistry(np['registry'],u2.artifact).select_specialty(observation,observation['longitudinal_memory'],[a['department'] for a in np['agents']],router='adaptive',seed=42) if u2 is not None else None)
            positive_support={a['department']:sum(x['jaccard']>0 and outcomes[(x['train_ordinal'],a['agent_id'])] is True for x in overlap) for a in np['agents']}
            detail={'panel_ordinal':row['panel_ordinal'],'phase':phase,'observation':observation,'native_train_jaccard':overlap,'U1_route':route,'U2_route':u2_route,'positive_outcome_overlap_counts':positive_support}
            private_evaluations.append(detail)
            public_evaluations.append({'panel_ordinal':row['panel_ordinal'],'phase':phase,'U1_selected':route['selected_specialty'],
                'U1_status':route['decision_status'],'U1_ties':route['tied_departments'],
                'U1_matched_samples':{k:(v['matched_samples'] if v else 0) for k,v in route['utility_estimates'].items()},
                'U1_scores':route['specialty_scores'],'native_U2_feature_positive_jaccard_groups':sum(x['jaccard']>0 for x in overlap),
                'native_U2_max_jaccard':max(x['jaccard'] for x in overlap),'native_U2_positive_outcome_support':positive_support,
                'U2_routing_score':'NOT_COMPUTED_NO_FITTING' if u2 is None else u2_route})
    # Isolated proposed Train view, inherited source literature rule, no tests/results invented.
    catalog=evidence_catalog();cards={c['disease']:c for c in catalog['cards']};proposals=[];used=set()
    for number,p in enumerate(train,1):
        label=p['reference']['label'];generated=scenario(number,cards[label],catalog,POLICY,used_chiefs=used)
        obs=deepcopy(native_initial[p['ordinal']]);obs['chief_complaint']=generated['chief_complaint']['text']
        base=p['case']['agent_visible']['current_initial_state']['patient_visible_info'].get('associated_symptoms',[])
        obs['current_symptoms']=list(dict.fromkeys(base+[s['text'] for s in generated['symptoms']]))
        records=[r for visit in p['case']['agent_visible']['past_visits'] for r in visit['records']]
        obs['longitudinal_memory']=PatientMemoryStore(records).retrieve(p['case']['agent_visible']['longitudinal'],memory_query(obs),'selective',5,max_chars=None)
        if directly_discloses_label(obs['chief_complaint']) or generated['synthetic_test_results']:raise ValueError('PROPOSAL_SOURCE_OR_LABEL_LEAK')
        proposals.append({'train_ordinal':p['ordinal'],'view':'D2_COMPATIBLE_TRAIN_PROPOSAL_NOT_EXECUTABLE',
            'source_case_fingerprint':digest(p['case']),'source_reference_fingerprint':digest(p['reference']),
            'literature_fingerprint':digest(catalog),'inherited_rule_fingerprint':digest(POLICY),
            'new_population_scope':'EXISTING_U2_TRAIN8_ONLY_NOT_ORIGINAL50','label_conditioned':True,
            'clinical_reviewed':False,'clinical_release_approved':False,'execution_approved':False,
            'generated_subjective':generated,'observation_proposal':obs,'original_observation_fingerprint':digest(native_initial[p['ordinal']]),
            'objective_source_values_changed':False,'numeric_tests_added':0,'actual_agent_outcomes_for_this_view':None})
    current_stats=[]
    for phase,obs_list in [('U2_Native_Train8',[native_initial[p['ordinal']] for p in train]),
                           ('D2_compat_Train8_proposal',[r['observation_proposal'] for r in proposals]),
                           ('D2_evaluation_initial10',[r['observation'] for r in private_evaluations if r['phase']=='initial'])]:
        sets=[observed_tokens(o) for o in obs_list]
        current_stats.append({'view':phase,'patients':len(obs_list),'chief_complaint_present':sum(bool(o.get('chief_complaint')) for o in obs_list),
            'symptom_counts':dict(Counter(len(o.get('current_symptoms',[])) for o in obs_list)),
            'objective_finding_counts':dict(Counter(len(o.get('objective_findings',[])) for o in obs_list)),
            'memory_counts':dict(Counter(len(o.get('longitudinal_memory',{}).get('records',[])) for o in obs_list)),
            'mean_observed_tokens':sum(len(t) for t in sets)/len(sets),
            'numeric_token_fraction':(sum(t.isdigit() for ts in sets for t in ts)/sum(len(ts) for ts in sets)) if sum(len(ts) for ts in sets) else None,
            'empty_token_patients':sum(not t for t in sets)})
    proposal_similarity=[]
    for ev in [r for r in private_evaluations if r['phase']=='initial']:
        tokens=observed_tokens(ev['observation']);old=[jaccard_similarity(tokens,observed_tokens(native_initial[p['ordinal']])) for p in train]
        new=[jaccard_similarity(tokens,observed_tokens(p['observation_proposal'])) for p in proposals]
        proposal_similarity.append({'panel_ordinal':ev['panel_ordinal'],'current_native_max':max(old),'proposal_max':max(new),
            'current_native_positive_groups':sum(x>0 for x in old),'proposal_positive_groups':sum(x>0 for x in new),
            'proposal_is_not_new_Agent_success_evidence':True})
    post_groups=[]
    for p in patients:
        fps=[];diagnostic=True
        for a in np['agents']:
            folder=run/('Patient%02d'%p['ordinal'])/a['department']/'Episode0001'
            last=sorted(folder.glob('Request*_before.json'))[-1]
            body=json.loads(read(last)['request']['messages'][1]['content'])
            fps.append(digest({k:body.get(k) for k in ('observation','evidence','supported_diagnosis_eligible_indices')}))
            final=read(sorted(folder.glob('Request*_core_response.json'))[-1])['decoded_response']
            diagnostic=diagnostic and final.get('action')=='diagnose'
        post_groups.append({'ordinal':p['ordinal'],'split':p['split'],'strict_same_post_snapshot_all_five':len(set(fps))==1,
                            'all_actual_final_requests_diagnostic':diagnostic})
    cross=read(ROOT/'outputs/research_evaluation/B34_DIAGNOSIS_RECOVERY_20261123_v1/cross_run_audit.json')
    report={'U2':status,'native_train_groups':8,'native_train_rows':40,'actual_train_positive_rows':sum(outcomes[(p['ordinal'],a['agent_id'])] is True for p in train for a in np['agents']),
        'by_agent_train':counts,'domain_stats':current_stats,'routing_support':public_evaluations,'proposal_similarity':proposal_similarity,
        'cross_run_regression':cross,'shared_post_observation_candidate_groups':post_groups,
        'ER_vs_specialty_same_terminal_snapshot':'NOT_EVALUATED_PENDING_NEW_PLAN',
        'U3_ready':False,'U3_blockers':['NEW_VIEW_HAS_NO_ACTUAL_FIVE_AGENT_OUTCOMES','ER_COMPARATOR_OUTCOMES_NOT_AVAILABLE',
            'VALIDATION_M5_INDEPENDENCE_NOT_MET'],'new_fitting':0,'new_LLM_generation':0,'clinical_accuracy':'NOT_EVALUATED'}
    put(PR/'native_initial_observations_private.json',native_initial);put(PR/'routing_feature_probe_private.json',private_evaluations)
    put(PR/'D2_compatible_train_view_proposals.json',proposals);put(PUB/'routing_root_summary.json',report)
    print(json.dumps({'U2':status,'positive_train_rows':report['actual_train_positive_rows'],'by_agent_train':counts,'domain_stats':current_stats,'U3_ready':False,'new_generation':0,'new_fit':0}))


if __name__=='__main__':main()
