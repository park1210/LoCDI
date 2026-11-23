"""Independent saved-wire/scoring verification. Never contacts a model."""
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_synthea_diagnostic_snapshots import PUB,PR,OLD,RUN,load_protocol,packet,build_condition
from scripts.audit_research_gaps import read,write,check,sha
from app.locdi.llm_setup import digest,validate_setup
from jsonschema import Draft202012Validator


def normalized(value):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',value).casefold()))


def main():
    protocol=load_protocol();slots=read(PR/'snapshot_requests.json')
    result=read(PR/RUN/'results.json');execution=read(PUB/'execution_summary.json')
    registry=validate_setup(read(PR/'llm_config.json'))['prepared']['registry']
    catalog=read(Path(protocol['catalog']['path']))
    flags=[];counts={'requests':0,'HTTP200':0,'native_schema_valid':0,'semantic_valid':0,
        'identity_digest_context_valid':0,'planned_wire_match':0,'metric_match':0}
    for row in result:
        if not row['generation']:
            continue
        ordinal,condition=row['patient_ordinal'],row['condition']
        slot=next(s for s in slots if s['patient_ordinal']==ordinal and s['condition']==condition)
        folder=PR/RUN/('Patient%03d_%s'%(ordinal,condition))
        before=read(folder/'Request0001_before.json');after=read(folder/'Request0001_after.json')
        counts['requests']+=1
        req=before['request'];spec=slot['contract']
        wire=(req['model']==spec['model'] and req['messages'][0]['content']==spec['system']
            and json.loads(req['messages'][1]['content'])==spec['payload'] and req['format']==spec['schema']
            and digest(req)==before['request_fingerprint']==after['request_fingerprint'])
        counts['planned_wire_match']+=wire
        if not wire:flags.append('PLANNED_WIRE_MISMATCH')
        counts['HTTP200']+=after.get('HTTP_status')==200
        identity=after.get('identity_after',{})
        identity_ok=('sha256:'+str(identity.get('digest','')).removeprefix('sha256:')==protocol['model_identifier']
            and identity.get('context_length')==32768)
        counts['identity_digest_context_valid']+=identity_ok
        if not identity_ok:flags.append('IDENTITY_NOT_VERIFIED')
        try:
            output=json.loads(after['response_envelope']['message']['content'])
            Draft202012Validator(spec['schema']).validate(output)
            counts['native_schema_valid']+=1
            d=packet(ordinal)
            capture=build_condition(d,condition,registry,catalog,slot['cutoff_minute'])
            validated=capture['validate'](json.dumps(output))
            counts['semantic_valid']+=1
            if row['status']=='completed':
                supported=validated['certainty']=='supported'
                target=validated['target'] if supported else None
                exact=target in d['Source_reference_diagnoses'] if supported else False
                norm=normalized(target) in [normalized(x) for x in d['Source_reference_diagnoses']] if supported else False
                same=(row['source_exact']==exact and row['normalized_match']==norm
                    and row['supported']==supported
                    and row['source_department_match']==(validated['department'] in d['Source_reference_departments']))
                counts['metric_match']+=same
                if not same:flags.append('SAVED_METRIC_MISMATCH')
        except Exception as exc:
            if row['status']=='completed':flags.append('COMPLETED_BUT_INVALID_'+type(exc).__name__)
    if counts['requests']!=execution['generations']:
        flags.append('GENERATION_COUNT_MISMATCH')
    boundaries=read(PUB/'pre_execution_boundary_check.json')
    ids=[]
    for ordinal in range(1,21):
        d=packet(ordinal)
        ids.extend(d['private_identity'][k] for k in ('patient_id','visit_id'))
    leaks=[]
    for file in PUB.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.csv','.md'):
            text=file.read_text(encoding='utf-8-sig')
            if any(i in text for i in ids):leaks.append(str(file.relative_to(PUB)))
    if leaks:flags.append('PUBLIC_IDENTIFIER_LEAK')
    protected=check(read(PR/'preservation_before.json'))
    if protected['status']!='PASS':flags.append('PROTECTED_ASSET_MODIFIED')
    write(PUB/'independent_verification.json', {'status':'PASS' if not flags else 'FAIL',
        'flags':flags,'counts':counts,'registered_slots':len(result),
        'completed':sum(r['status']=='completed' for r in result),
        'failed':sum(r['status']=='failed' for r in result),
        'not_attempted':sum(r['status']=='not_attempted' for r in result),
        'blocked_expansion_slots':sum(s['status']=='NOT_RUN' for s in slots),
        'clinical_review_created':0,'new_generation_in_verification':0,
        'boundary_check_sha':sha(PUB/'pre_execution_boundary_check.json'),
        'pre_execution_boundary_check_pass':boundaries['pass'],
        'public_identifier_leaks':leaks,'protected':protected,
        'LLM_bitwise_reproduction':'NOT_TESTED; saved trace verification only',
        'verifier_sha256':sha(Path(__file__))})
    print(json.dumps({'flags':flags,'counts':counts,'protected':protected['status']}))


if __name__=='__main__':main()
