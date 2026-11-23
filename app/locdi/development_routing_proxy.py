"""Separate Source-label proxy contract. Never admits clinical evaluation.

No transport/fitting at import or admission. Core math/validators are unchanged.
Only complete five-agent groups enter the existing Core fit; unresolved is null.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import RoutingUtility, observed_tokens
from app.llm.prompt_loader import specialty_system_prompt
from app.locdi.llm_setup import digest, validate_setup,normalize_model_identifier
from app.locdi.m5_common_source import verify_csv
from app.locdi.routing_collection_preparation import project_input, response_schema

MARKERS={'experiment_kind':'DEVELOPMENT_EXPERIMENT','source_label_proxy':True,
         'independent_clinical_review':False,'clinical_accuracy':'NOT_EVALUATED',
         'research_evaluation_ready':False,'clinical_research_admission':False}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def derive_rows(plan,collection,request_root,registry,config,evaluation_patients):
    """Recompute labels from raw Source and actually saved native requests.

    Exact catalog text only, no synonyms, department labels or unresolved bools.
    Same-date Condition is a declared retrospective proxy, not clinical release.
    """
    ready=validate_setup(config)
    if ready['static_readiness']!='READY' or ready['prepared']['registry']!=registry:
        raise ValueError('CURRENT_PROXY_REGISTRY_CONFIG_INVALID')
    train_config=deepcopy(plan['llm_config']);runtime_config=deepcopy(config)
    for cfg in (train_config,runtime_config):
        cfg.pop('max_requests',None);cfg.pop('registry_path',None)
    if train_config!=runtime_config or plan['registry']!=registry:
        raise ValueError('CURRENT_PROXY_MODEL_PROMPT_OPTIONS_MISMATCH')
    if plan['fingerprint']!=digest({k:v for k,v in plan.items() if k!='fingerprint'}):
        raise ValueError('PROXY_COLLECTION_PLAN_CHANGED')
    for group in ('source_files','code_files'):
        for path,expected in plan[group].items():
            if file_sha(path)!=expected:raise ValueError('PROXY_SOURCE_OR_COLLECTOR_CHANGED')
    agents={a['agent_id']:a for a in registry['agents'] if a['role']=='specialty'}
    if len(agents)!=5:raise ValueError('FIVE_SPECIALTIES_REQUIRED')
    candidates=plan['candidates'];ids=[c['decision']['patient_id'] for c in candidates]
    if len(ids)!=len(set(ids)) or set(ids)&set(evaluation_patients):raise ValueError('PROXY_PATIENT_SPLIT_OVERLAP')
    if len({c['shared_input_fingerprint'] for c in candidates})!=len(candidates):raise ValueError('PROXY_DUPLICATE_SHARED_INPUT')
    records=collection['records'];by_key={};checkpoint_files={}
    for c in candidates:
        g=c['decision']['generation']
        if (g['model_identifier']!=config['model_identifier'] or g['temperature']!=0 or g['seed']!=42
            or g['max_output_tokens']!=1024 or any(g['prompt_fingerprints'].get(aid)!=ready['prepared']['prompt_fingerprints'][aid] for aid in agents)):
            raise ValueError('CURRENT_PROXY_GENERATION_BINDING_CHANGED')
    from jsonschema import Draft202012Validator
    validator=Draft202012Validator(response_schema(plan['catalog']))
    for i,record in enumerate(records,1):
        key=(record['decision_id'],record['agent_id'])
        if key in by_key:raise ValueError('DUPLICATE_PROXY_RESPONSE')
        by_key[key]=record
        c=next((c for c in candidates if c['decision']['decision_id']==key[0]),None)
        if c is None or key[1] not in agents:raise ValueError('UNREGISTERED_PROXY_RESPONSE')
        d=c['decision'];agent=agents[key[1]]
        if (record['patient_id']!=d['patient_id'] or record['observation_fingerprint']!=digest(d['observation'])
            or record['configuration_fingerprint']!=digest(d) or record['generation']!=d['generation']
            or record['model']!=agent['model'] or record['version']!=agent['version']):raise ValueError('PROXY_RESPONSE_BINDING')
        if record['execution_status']!='SUCCEEDED' or record['response_contract_status']!='VALID':continue
        before=Path(request_root)/('Request%04d_before.json'%i);after=Path(request_root)/('Request%04d_after.json'%i)
        b=read(before);a=read(after);req=b['request']
        checkpoint_files.update({str(before.resolve()):file_sha(before),str(after.resolve()):file_sha(after)})
        expected_payload={'department':agent['department'],**c['shared_payload']}
        messages=req['messages']
        if (normalize_model_identifier(a['identity_after']['digest'])!=config['model_identifier']
            or normalize_model_identifier(b['identity_before']['digest'])!=config['model_identifier']
            or a['identity_after']['context_length']!=32768):raise ValueError('ACTUAL_PROXY_MODEL_CONTEXT_MISMATCH')
        if (a['status']!='VALID' or a['HTTP_status']!=200 or a.get('response_received') is not True
            or req['model']!=agent['model'] or req['format']!=response_schema(plan['catalog'])
            or len(messages)!=2 or messages[0]!={'role':'system','content':specialty_system_prompt(agent['department'])}
            or messages[1]['role']!='user' or json.loads(messages[1]['content'])!=expected_payload
            or b['request_fingerprint']!=digest(req) or a['request_fingerprint']!=digest(req)
            or record['observed_model_identifier']!=config['model_identifier']):raise ValueError('RAW_PROXY_CHECKPOINT_MISMATCH')
        response=json.loads(a['response_envelope']['message']['content']);validator.validate(response)
        if response!=record['response'] or digest(response)!=record['response_fingerprint']:
            raise ValueError('RAW_PROXY_RESPONSE_CHANGED')
    fit_rows=[];assessments=[]
    for c in candidates:
        d=c['decision'];ref=c['reference_candidates'];source_rows=ref.get('source_condition_rows',[])
        shared=project_input(d['observation'],plan['catalog'])
        if shared!=c['shared_payload'] or digest(shared)!=c['shared_input_fingerprint']:raise ValueError('PROXY_INPUT_CHANGED')
        label=None;reason='SOURCE_REFERENCE_AMBIGUOUS_OR_MISSING'
        # Require exactly one unambiguous current-encounter Source description.
        if source_rows:
            verify_csv([{'source_metadata':r} for r in source_rows])
            raw=[r['source_record'] for r in source_rows]
            labels={r.get('DESCRIPTION') for r in raw}
            if (len(labels)==1 and None not in labels and all(r.get('PATIENT')==d['patient_id']
                and r.get('ENCOUNTER')==d['visit_id'] and r.get('CODE') and r.get('SYSTEM')
                and r.get('START') and r['START'][:10]<=c['visit_started_at'][:10] for r in raw)):
                label=next(iter(labels));reason='EXACT_SOURCE_LABEL_PROXY'
                if label not in plan['catalog']:label=None;reason='SOURCE_LABEL_OUTSIDE_SHARED_CATALOG'
        outcomes=[];group_rows=[]
        for aid,agent in agents.items():
            r=by_key.get((d['decision_id'],aid));correct=None;status=reason
            if r is None:status='MISSING_RESPONSE'
            elif r['execution_status']!='SUCCEEDED' or r['response_contract_status']!='VALID':status='REQUEST_OR_SCHEMA_FAILED'
            elif r['response']['certainty']=='unresolved' or r['response']['diagnosis']=='Unresolved':status='UNRESOLVED'
            elif r['response']['certainty']!='supported':status='CERTAINTY_NOT_SUPPORTED'
            elif label is not None:
                correct=r['response']['diagnosis']==label;status='SOURCE_LABEL_PROXY_SCORED'
            outcomes.append({'agent_id':aid,'correctness_proxy':correct,'status':status})
            if type(correct) is bool:
                group_rows.append({**{k:d[k] for k in ('patient_id','decision_id','split','observation')},
                    'agent_id':aid,'model':agent['model'],'version':agent['version'],'correct':correct})
        complete=len(group_rows)==5
        assessments.append({'case_ref':c['case_ref'],'split':d['split'],'complete_proxy_group':complete,'outcomes':outcomes})
        if complete and d['split']=='train':fit_rows.extend(group_rows)
    return {'rows':fit_rows,'groups':assessments,'checkpoint_files':checkpoint_files,
            'complete_train_groups':len(fit_rows)//5,**MARKERS}


def admit_development_proxy(evidence,registry,patients,config):
    if evidence.get('schema_version') not in ('locdi.routing.source_label_proxy.v1','locdi.routing.unfitted_source_proxy.v1') or any(evidence.get(k)!=v for k,v in MARKERS.items()):
        raise ValueError('DEVELOPMENT_PROXY_SCOPE_REQUIRED')
    if not evidence.get('origin_files'):raise ValueError('PROXY_LINEAGE_REQUIRED')
    for p,expected in evidence['origin_files'].items():
        if file_sha(p)!=expected:raise ValueError('PROXY_ORIGIN_CHANGED')
    plan=read(evidence['collection_plan'])
    unfitted=evidence['schema_version']=='locdi.routing.unfitted_source_proxy.v1'
    if unfitted:
        if evidence.get('collection_file') is not None or evidence.get('collection_status')!='NOT_COLLECTED' or evidence.get('routing_artifact') is not None:
            raise ValueError('UNFITTED_MUST_NOT_CLAIM_COLLECTION_OR_UTILITY')
        collection={'records':[]}  # Absence of responses, never an invented receipt.
    else:collection=read(evidence['collection_file'])
    derived=derive_rows(plan,collection,evidence['request_root'],registry,config,patients)
    if digest(derived['rows'])!=evidence['fit_rows_fingerprint']:raise ValueError('PROXY_LABELS_CHANGED')
    for path,expected in derived['checkpoint_files'].items():
        if evidence['origin_files'].get(path)!=expected:raise ValueError('UNPINNED_PROXY_CHECKPOINT')
    artifact=evidence['routing_artifact']
    if not derived['rows']:
        if artifact is not None:raise ValueError('NO_SYNTHETIC_UTILITY_WITHOUT_PROXY_ROWS')
    else:
        if artifact is None or artifact.get('min_samples')!=2:raise ValueError('UNCHANGED_MIN_SAMPLES_TWO_REQUIRED')
        expected=[{**{k:r[k] for k in ('patient_id','decision_id','agent_id','correct')},'tokens':observed_tokens(r['observation'])} for r in derived['rows']]
        if sorted(map(digest,artifact['samples']))!=sorted(map(digest,expected)):raise ValueError('PROXY_FIT_LINEAGE_MISMATCH')
        RoutingUtility(artifact,AgentRegistry(registry))
    return {'artifact':artifact,'correctness':{'utility_fit_ready':False},**MARKERS,
            'utility_status':'SOURCE_LABEL_PROXY_FITTED' if artifact is not None else 'NO_ELIGIBLE_GROUPS_CORE_ABSTENTION',
            'complete_train_groups':derived['complete_train_groups']}
