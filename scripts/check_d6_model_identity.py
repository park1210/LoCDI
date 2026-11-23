"""Static model binding and optional GET /api/tags only. Never sends data."""
import argparse,json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.locdi.llm_setup import load_setup,local_endpoint,inspect_local_model,normalize_model_identifier,digest
from app.locdi.healthbench_grader import GRADER_CONFIG,official_template


def inspect(probe=False):
    config=load_setup();endpoint=local_endpoint(config['endpoint'])
    result={'schema_version':'locdi.D6.model_identity.v1','generation_model':{
        **{k:config[k] for k in ('model','model_version','model_identifier','temperature','seed','max_output_tokens','timeout_seconds','request_retries')},
        'provider':'LOCAL_OLLAMA_OPENAI_COMPATIBLE','prompt_version':'ORIGINAL_HEALTHBENCH_PROMPT_ONLY_NO_ADDED_SYSTEM',
        'local_files':inspect_local_model(config['model'],config['models_root']),'server':'NOT_CHECKED','observed_digest':'NOT_RECORDED'},
        'grader_model':{**GRADER_CONFIG,'prompt_fingerprint':digest(official_template()),'status':'BLOCKED_MODEL_AND_QUALITY_UNCONFIRMED'},
        'actual_generation_calls':0,'actual_grader_calls':0,'identity_environment_http_requests':0,'healthbench_content_sent':False}
    if probe:
        result['identity_environment_http_requests']=1
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(endpoint[:-3]+'/api/tags',timeout=5) as response:entries=json.load(response)['models']
            values={normalize_model_identifier(e.get('digest')) for e in entries if e.get('name')==config['model'] or e.get('model')==config['model']}
            observed=next(iter(values)) if len(values)==1 else None
            result['generation_model'].update(server='AVAILABLE',observed_digest=observed or 'NOT_RECORDED',identity_match=observed==config['model_identifier'])
        except Exception as exc:
            result['generation_model'].update(server='UNAVAILABLE_OR_ACCESS_BLOCKED',error_type=type(exc).__name__)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe-local-server',action='store_true')
    print(json.dumps(inspect(parser.parse_args().probe_local_server),ensure_ascii=True,indent=2))
