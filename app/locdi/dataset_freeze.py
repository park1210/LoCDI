"""D7 local development freeze. Read-only verification; no model or fit entry."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
from app.locdi.llm_setup import ROOT,digest
from app.locdi.source_replay import atomic_rows

EXPECTED='69b890f1a05245a981c32199a65bc1e23fd5f3643a5aed95ed0e59f51f951d1a'
CANDIDATE='outputs/private/dataset_release/D6_release_candidate_v1/manifest.json'
SUMMARY='outputs/reports/D6_release_candidate_v1.json'
PUBLIC='datasets/releases/locdi_dataset_v1.0/freeze_manifest.json'
PRIVATE='outputs/private/dataset_release/D7_locdi_dataset_v1.0/manifest.json'
BASELINE='outputs/private/D7_baseline_hashes_v1.json'
REGRESSION='outputs/reports/D7_regression_v3.json'
TOOLS=('app/locdi/dataset_freeze.py','scripts/freeze_d7_dataset.py',
       'scripts/verify_d7_dataset_freeze.py','tests/locdi/test_dataset_freeze.py')
READINESS={'SOFTWARE_READY':True,'DATASET_READY_FOR_DEVELOPMENT':True,
           'EXTERNAL_BENCHMARK_EVALUATION_READY':False,'CLINICAL_RESEARCH_EVALUATION_READY':False}
STATUSES={'A':'FROZEN_DEVELOPMENT_ONLY','B':'FROZEN_DEVELOPMENT_ONLY',
          'C':'FROZEN_EXTERNAL_INPUT_UNGRADED','D':'FROZEN_DEVELOPMENT_ONLY',
          'E':'FROZEN_OFFLINE_CONTRACT_ONLY','F':'FROZEN_REVIEW_INPUT_UNAPPROVED'}
INPUTS=(SUMMARY,CANDIDATE,'outputs/reports/D6_dataset_validation_v1.json',
        'outputs/reports/D6_readiness_v1.json','outputs/reports/D6_split_and_exposure_v1.json',
        'outputs/reports/D6_grader_environment_v1.json','outputs/reports/D6_final_verification_v1.json',
        'outputs/private/healthbench_environment/D6_v1/original_source_hash_audit.json',
        'outputs/private/healthbench_environment/D6_v1/installed_environment.json')


class FreezeError(ValueError):
    """Only a safe error code, never source text/path/identity."""


def require(condition,code):
    if not condition:raise FreezeError(code)


def load(path):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'DUPLICATE_JSON_KEY');out[key]=value
        return out
    def constant(_):raise FreezeError('NONFINITE_JSON')
    try:return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=constant)
    except FileNotFoundError:raise FreezeError('MISSING_SOURCE') from None
    except (OSError,json.JSONDecodeError):raise FreezeError('INVALID_OR_INCOMPLETE_JSON') from None


def sha(path):
    try:
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:raise FreezeError('MISSING_SOURCE') from None
    except OSError:raise FreezeError('SOURCE_UNREADABLE') from None


def safe_path(root,name):
    require(isinstance(name,str) and bool(name) and '\\' not in name and ':' not in name,'PATH_TRAVERSAL')
    p=PurePosixPath(name)
    require(not p.is_absolute() and all(x not in ('..','.') for x in p.parts) and str(p)==name,'PATH_TRAVERSAL')
    root=Path(root).resolve();target=(root/name).resolve()
    require(target.is_relative_to(root) and target!=root,'PATH_TRAVERSAL')
    return target


def entries_check(root,bundles):
    seen=set();content={};count=0
    for bundle in bundles:
        entries=bundle['files']
        require(bundle['file_count']==len(entries),'FILE_COUNT_MISMATCH')
        require(bundle['files_fingerprint']==digest(entries),'FILE_LIST_FINGERPRINT_MISMATCH')
        for entry in entries:
            name=entry['path'];path=safe_path(root,name)
            require(name not in seen,'DUPLICATE_FILE_ENTRY');seen.add(name)
            require(path.is_file(),'MISSING_SOURCE')
            require(sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'],'FILE_HASH_MISMATCH')
            require(entry['private']==name.startswith('outputs/private/'),'PRIVATE_CLASSIFICATION_MISMATCH')
            content.setdefault(entry['sha256'],[]).append(name);count+=1
    return {'files':count,'duplicate_content_groups':[v for v in content.values() if len(v)>1]}


def candidate(root=ROOT,expected_fp=EXPECTED):
    root=Path(root);d=load(root/CANDIDATE);fp=d['manifest_fingerprint']
    require(digest({k:v for k,v in d.items() if k!='manifest_fingerprint'})==fp==expected_fp,'CANDIDATE_FINGERPRINT_MISMATCH')
    summary=load(root/SUMMARY)
    require(summary['manifest_fingerprint']==fp and summary['private_manifest_sha256']==sha(root/CANDIDATE),'CANDIDATE_PUBLIC_LINK_MISMATCH')
    require([b['bundle'] for b in d['bundles']]==list('ABCDEF'),'BUNDLE_SET_MISMATCH')
    checked=entries_check(root,d['bundles'])
    if expected_fp==EXPECTED:require(checked['files']==185,'CANDIDATE_FILE_COUNT_CHANGED')
    return d,checked


def protected_check(root,baseline):
    for name,expected in baseline.items():require(sha(safe_path(root,name))==expected,'ORIGINAL_FILE_CHANGED')
    return len(baseline)


def environment_smoke(root=ROOT):
    root=Path(root);python=root/'.venv-healthbench/Scripts/python.exe'
    require(python.is_file(),'GRADER_ENVIRONMENT_NOT_INSTALLED')
    def run(args):
        try:
            r=subprocess.run([str(python),'-B']+args,cwd=str(root),capture_output=True,text=True,encoding='utf-8',timeout=120)
        except (OSError,subprocess.TimeoutExpired):raise FreezeError('GRADER_ENVIRONMENT_SMOKE_FAILED') from None
        require(r.returncode==0,'GRADER_ENVIRONMENT_SMOKE_FAILED');return r.stdout
    installed=json.loads(run(['-c','import sys,json,importlib.metadata as m; print(json.dumps({"python":sys.version.split()[0],"packages":{d.metadata["Name"]:d.version for d in m.distributions()}}))']))
    recorded=load(root/'outputs/private/healthbench_environment/D6_v1/installed_environment.json')
    require(installed['python']=='3.11.18' and installed['packages']==recorded['packages'],'INSTALLED_ENVIRONMENT_CHANGED')
    smoke=json.loads(run(['scripts/smoke_healthbench_official.py']))
    require(smoke['official_module_import']=='PASS' and smoke['official_pure_tests']=='2_PASS'
            and smoke['network_disabled'] and not smoke['grade_sample_called'] and not smoke['sampler_constructed'],'GRADER_ENVIRONMENT_SMOKE_FAILED')
    smoke['interpreter']='.venv-healthbench/Scripts/python.exe'
    return {'installed_versions':installed,'official_import_and_pure_smoke':smoke,
            'runtime_and_wheel_hash_scope':'SEPARATE_CANDIDATE_FILE_HASH_CHECK',
            'reinstalled':False,'actual_generation_calls':0,'actual_grader_calls':0}


def membership_snapshot():
    from app.locdi.dataset_suitability import load_memberships
    from app.locdi.external_dialogue import load_pilot
    from app.locdi.reference_readiness import verify_d3
    memberships,_=load_memberships();_,p,_,_=verify_d3();_,selection,_=load_pilot()
    selected={k:sorted(v) for k,v in memberships.items() if k in ('holdout','bank','routing_train','routing_validation','voi_train')}
    selected['D3_replay']=sorted({c['patient_id'] for c in p['cases.jsonl']})
    return {'patient_memberships':selected,'healthbench_excluded_pilot_ids':sorted(s['sample_id'] for s in selection['samples']),
            'new_split_assignments':False,'healthbench_assigned_split':None}


def validate_data():
    from app.locdi.dataset_release_readiness import validate_all,split_audit
    data=validate_all();splits=split_audit()
    old=load(ROOT/'outputs/reports/D6_split_and_exposure_v1.json')
    require(splits==old,'SPLIT_OR_EXPOSURE_CHANGED')
    a=data['original_locdi'];b=data['source_replay'];c=data['healthbench'];f=data['internal_review_packets']
    require(a['case_count']==50 and a['counts']['memory_record_references']==a['counts']['memory_matches_upstream_record']==72590
        and a['counts']['memory_time_or_identity_violations']==0 and a['counts']['ask_events']==50 and a['counts']['test_events']==200
        and a['source_reference_checks']['unique_references']==a['source_reference_checks']['matching']==3176,'ORIGINAL_DATA_CONTRACT_CHANGED')
    require(b['manifest_fingerprint']=='40e001271e8cd8f3e7db0095b3fe00e05ea5c85be8fb9b41536ac6f8b6cab6c2'
        and b['proxy_cases_validated']==29 and b['memory_records']==1256 and b['ledger_rows']==2690
        and b['action_counts']=={'Ask':29,'Test':301} and b['reference_candidates']==64,'D3_DATA_CONTRACT_CHANGED')
    require(b['strict_source_status']=='STRICT_SOURCE_ACTUAL_AVAILABILITY_UNVERIFIED'
        and b['actual_available_action_times']==0 and b['consult_events']==0,'VISIBILITY_POLICY_CHANGED')
    require(c['sample_count']==c['unique_sample_ids']==c['lossless_adapter_samples']==5000 and c['rubric_count']==57237
        and c['development_samples']==3 and c['nonpilot_candidates']==4997 and c['original_split']=='test','EXTERNAL_INPUT_BOUNDARY_CHANGED')
    require(f=={'packet_files_checked':81,'distinct_review_targets':80,'reviewed':0,'reviewer_fields_blank':True}
        and a['clinical_gold_approved']==b['clinical_gold_approved']==0,'UNREVIEWED_REFERENCE_PROMOTED')
    for v in (a['validators'],b['proxy_validator_counts']):
        require(not any(count for counts in v.values() for state,count in counts.items() if state in ('FAIL','BLOCKED')),'DATASET_SCHEMA_FAILED')
    source_audit=load(ROOT/'outputs/private/healthbench_environment/D6_v1/original_source_hash_audit.json')
    for source in source_audit:require(sha(source['path']) in source['expected'],'ORIGINAL_SOURCE_HASH_MISMATCH')
    require(data['routing_correctness_gate']['utility_fit_ready'] is False,'UNREVIEWED_TRAINING_PROMOTED')
    summary={'original':{'cases':50,'memory_references':72590,'ask':50,'test':200,'source_references':3176,
        'validators':a['validators'],'temporal':'WARNING','clinical_gold':0},
        'D3':{'cases':29,'memory':1256,'ask':29,'test':301,'ledger':2690,'reference_candidates':64,
        'clinical_gold':0,'consult':0,'actual_availability':0,'strict_source':b['strict_source_status'],'historical_replay_calls':b['historical_runner_calls']},
        'healthbench':{'samples':5000,'rubric_count':57237,'negative_rubric_count':c['negative_rubrics'],'prompt_projection':'LOSSLESS_PROMPT_ONLY',
        'pilot_excluded':3,'nonpilot_candidates':4997,'original_split':'test','assigned_split':None,'blind_test_approved':False,'pretraining_contamination':'UNKNOWN'},
        'review':f,'reference_status':'NOT_REVIEWED','clinical_metrics':'NOT_EVALUATED','original_source_files_verified':len(source_audit),
        'new_replay_or_model_evaluation_runs':0}
    return summary,splits


def policy(public):
    require(public['evaluation_readiness']==READINESS,'READINESS_PROMOTION_FORBIDDEN')
    require(public['clinical_review_status']=='NOT_REVIEWED_GOLD_ZERO' and public['clinical_metrics']=='NOT_EVALUATED','UNREVIEWED_REFERENCE_PROMOTED')
    require(public['final_freeze_status']=='FROZEN_DEVELOPMENT_ONLY','FREEZE_STATUS_INVALID')
    require(public['package_list']==list('ABCDEF'),'BUNDLE_SET_MISMATCH')
    require(public['license_access_status']['A']=='UNKNOWN' and public['license_access_status']['B'].startswith('UNKNOWN'),'UNKNOWN_LICENSE_PROMOTED')
    require(public['development_exposure_summary']['healthbench_excluded_pilot_count']==3
        and public['development_exposure_summary']['healthbench_nonpilot_blind_test_approved'] is False
        and public['development_exposure_summary']['routing_validation_in_voi_training']==5
        and public['development_exposure_summary']['new_splits_created'] is False,'DEVELOPMENT_EXPOSURE_CHANGED')
    require(public['dataset_validation_status']['D3']['strict_source']=='STRICT_SOURCE_ACTUAL_AVAILABILITY_UNVERIFIED','VISIBILITY_POLICY_CHANGED')


def public_boundary(public):
    forbidden={'path','private_manifest_path','patient_id','visit_id','case_id','prompt_id','sample_id','rubrics','canary','gold','private_identity'}
    def walk(value):
        if isinstance(value,dict):
            require(not set(value)&forbidden,'PUBLIC_PRIVATE_LEAKAGE')
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
        elif isinstance(value,str):require('outputs/private/' not in value and ':\\Users\\' not in value,'PUBLIC_PRIVATE_LEAKAGE')
    walk(public)


def documents(candidate_document,validation,splits,memberships,environment,baseline,*,root=ROOT,created_at=None):
    root=Path(root);bundles=deepcopy(candidate_document['bundles']);extra=[]
    for name in TOOLS:
        p=safe_path(root,name);extra.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size,'private':False})
    bundle=next(b for b in bundles if b['bundle']=='E');bundle['files']+=extra
    bundle['file_count']=len(bundle['files']);bundle['files_fingerprint']=digest(bundle['files'])
    for b in bundles:b['freeze_status']=STATUSES[b['bundle']]
    checked=entries_check(root,bundles)
    meta={'schema_version':'locdi.D7.public_freeze.v1','release_id':'locdi_dataset_v1.0','release_version':'1.0',
        'created_at':created_at or datetime.now(timezone.utc).isoformat(),'source_candidate_fingerprint':candidate_document['manifest_fingerprint'],
        'release_purpose':'LOCAL_DEVELOPMENT_REPRODUCIBILITY_NOT_CLINICAL_VALIDATION','package_list':list('ABCDEF'),
        'package_versions':{b['bundle']:b['version'] for b in bundles},'file_counts':{b['bundle']:b['file_count'] for b in bundles},
        'source_hash_summary':{b['bundle']:b['files_fingerprint'] for b in bundles},'total_file_count':checked['files'],
        'candidate_file_count':185,'file_changes':{'added':len(extra),'excluded':0,'reason':'D7_READ_ONLY_VERIFIER_AND_TEST_ONLY_CONTRACT_TOOLS_ADDED_TO_E'},
        'same_content_different_path_groups':len(checked['duplicate_content_groups']),
        'dataset_validation_status':validation,'clinical_review_status':'NOT_REVIEWED_GOLD_ZERO','clinical_metrics':'NOT_EVALUATED',
        'allowed_uses':{b['bundle']:b['allowed_use'] for b in bundles},'blocked_uses':{b['bundle']:b['blocked_use'] for b in bundles},
        'evaluation_readiness':deepcopy(READINESS),'development_exposure_summary':{'original_50':'EXPOSED_DEVELOPMENT',
            'D3_29':'EXPOSED_PROXY_REPLAY','routing_validation_in_voi_training':5,'healthbench_excluded_pilot_count':3,
            'healthbench_nonpilot_candidates':4997,'healthbench_nonpilot_blind_test_approved':False,'new_splits_created':False},
        'license_access_status':{b['bundle']:b['license_access_status'] for b in bundles},
        'package_freeze_status':{b['bundle']:b['freeze_status'] for b in bundles},'access_policy':'LOCAL_PRIVATE_SOURCES_NO_AUTOMATIC_REDISTRIBUTION',
        'final_freeze_status':'FROZEN_DEVELOPMENT_ONLY','actual_generation_calls':0,'actual_grader_calls':0}
    public_boundary(meta);policy(meta)
    locks={name:sha(safe_path(root,name)) for name in INPUTS+(BASELINE,REGRESSION)}
    private={'schema_version':'locdi.D7.private_freeze.v1','release_id':meta['release_id'],'release_version':'1.0','created_at':meta['created_at'],
        'source_candidate_fingerprint':candidate_document['manifest_fingerprint'],'bundles':bundles,'file_delta':{'added':extra,'excluded':[]},
        'duplicate_content_groups':checked['duplicate_content_groups'],'split_snapshot':splits,'private_memberships':memberships,
        'environment_verification':environment,'protected_baseline':baseline,'evidence_file_hashes':locks,
        'public_projection_fingerprint':digest(meta),'hash_policy':'SHA256_UTF8_SORT_KEYS_COMPACT_ENSURE_ASCII_FALSE_NO_NAN_OMIT_OWN_FINGERPRINT'}
    private['manifest_fingerprint']=digest(private)
    public={**meta,'private_manifest_fingerprint':private['manifest_fingerprint']}
    public['manifest_fingerprint']=digest(public)
    return public,private


def verify_pair(public,private,root=ROOT):
    require('public_manifest_fingerprint' not in private and 'private_manifest_fingerprint' not in private,'CIRCULAR_HASH_DEPENDENCY')
    require(digest({k:v for k,v in private.items() if k!='manifest_fingerprint'})==private['manifest_fingerprint'],'PRIVATE_MANIFEST_FINGERPRINT_MISMATCH')
    require(digest({k:v for k,v in public.items() if k!='manifest_fingerprint'})==public['manifest_fingerprint'],'PUBLIC_MANIFEST_FINGERPRINT_MISMATCH')
    require(public['private_manifest_fingerprint']==private['manifest_fingerprint'],'PUBLIC_PRIVATE_LINK_MISMATCH')
    projection={k:v for k,v in public.items() if k not in ('manifest_fingerprint','private_manifest_fingerprint')}
    require(digest(projection)==private['public_projection_fingerprint'],'PUBLIC_PRIVATE_LINK_MISMATCH')
    public_boundary(public);policy(public)
    checked=entries_check(root,private['bundles'])
    require(public['total_file_count']==checked['files'],'FILE_COUNT_MISMATCH')
    for key,field in [('package_versions','version'),('file_counts','file_count'),('source_hash_summary','files_fingerprint'),
                      ('license_access_status','license_access_status'),('allowed_uses','allowed_use'),
                      ('blocked_uses','blocked_use'),('package_freeze_status','freeze_status')]:
        require(public[key]=={b['bundle']:b[field] for b in private['bundles']},'PUBLIC_PRIVATE_METADATA_MISMATCH')
    return checked


def snapshot_check(public,private,validation,splits,memberships):
    require(validation==public['dataset_validation_status'],'DATA_VALIDATION_CHANGED')
    require(splits==private['split_snapshot'] and memberships==private['private_memberships'],'SPLIT_OR_EXPOSURE_CHANGED')


def verify(*,root=ROOT,full=True):
    root=Path(root);public=load(root/PUBLIC);private=load(root/PRIVATE)
    checked=verify_pair(public,private,root);c,_=candidate(root)
    actual={f['path']:f for b in private['bundles'] for f in b['files']};original={f['path']:f for b in c['bundles'] for f in b['files']}
    require(all(actual.get(k)==v for k,v in original.items()) and set(actual)-set(original)==set(TOOLS),'CANDIDATE_COMPOSITION_CHANGED')
    expected=deepcopy(c['bundles'])
    for bundle in expected:
        bundle['freeze_status']=STATUSES[bundle['bundle']]
        if bundle['bundle']=='E':
            bundle['files'] += [actual[name] for name in TOOLS]
            bundle['file_count']=len(bundle['files']);bundle['files_fingerprint']=digest(bundle['files'])
    require(expected==private['bundles'],'CANDIDATE_METADATA_CHANGED')
    for name,expected in private['evidence_file_hashes'].items():require(sha(safe_path(root,name))==expected,'FROZEN_EVIDENCE_CHANGED')
    count=protected_check(root,private['protected_baseline'])
    if full:
        require(root.resolve()==ROOT.resolve(),'FULL_VALIDATION_REQUIRES_PROJECT_ROOT')
        validation,splits=validate_data()
        snapshot_check(public,private,validation,splits,membership_snapshot())
        require(environment_smoke(root)==private['environment_verification'],'INSTALLED_ENVIRONMENT_CHANGED')
    return {'status':'PASS','final_freeze_status':public['final_freeze_status'],'files_verified':checked['files'],
        'protected_files_verified':count,'candidate_fingerprint':c['manifest_fingerprint'],'public_manifest_fingerprint':public['manifest_fingerprint'],
        'private_manifest_fingerprint':private['manifest_fingerprint'],'full_source_and_schema_validation':full,
        'evaluation_readiness':public['evaluation_readiness'],'actual_generation_calls':0,'actual_grader_calls':0}


def plan():
    c,checked=candidate();protected=load(ROOT/BASELINE);protected_check(ROOT,protected)
    validation,splits=validate_data();environment=environment_smoke();memberships=membership_snapshot()
    return c,validation,splits,memberships,environment,protected


def write_release(public,private,root=ROOT):
    root=Path(root);pub=safe_path(root,PUBLIC);priv=safe_path(root,PRIVATE)
    require(not pub.parent.exists() and not priv.parent.exists(),'RELEASE_ID_ALREADY_EXISTS')
    verify_pair(public,private,root)
    try:
        pub.parent.mkdir(parents=True,exist_ok=False);priv.parent.mkdir(parents=True,exist_ok=False)
        # Public manifest is the last commit marker. A partial write cannot be
        # considered frozen without both documents and a successful verifier.
        atomic_rows(priv.parent,priv.name,private,jsonl=False)
        atomic_rows(pub.parent,pub.name,public,jsonl=False)
    except Exception:raise FreezeError('FREEZE_STORAGE_FAILED_INCOMPLETE_RELEASE') from None


def freeze(*,write=False):
    require(not (ROOT/PUBLIC).parent.exists() and not (ROOT/PRIVATE).parent.exists(),'RELEASE_ID_ALREADY_EXISTS')
    inputs=plan()
    result={'status':'DRY_RUN','release_id':'locdi_dataset_v1.0','source_candidate_fingerprint':inputs[0]['manifest_fingerprint'],
        'candidate_files':185,'planned_files':189,'planned_added_verifier_files':list(TOOLS),'planned_exclusions':0,
        'planned_new_manifests':[PUBLIC,'PRIVATE_MANIFEST_RESTRICTED_LOCATION'],
        'evaluation_readiness':deepcopy(READINESS),'existing_files_written':0,'actual_generation_calls':0,'actual_grader_calls':0}
    if not write:return result
    regression=load(ROOT/REGRESSION);require(regression['successful'] and regression['fail']==regression['errors']==0,'REGRESSION_NOT_PASSED')
    protected_check(ROOT,inputs[-1]);candidate()
    public,private=documents(*inputs)
    write_release(public,private)
    return {**verify(),'status':'FROZEN_DEVELOPMENT_ONLY','release_id':public['release_id']}
