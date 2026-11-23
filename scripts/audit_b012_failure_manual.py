"""Saved-response diagnostic helpers. Never constructs a model client."""
import json
from jsonschema import Draft202012Validator


def schema_diagnosis(before, after):
    """Report validators/paths, not clinical text or invalid-output repairs."""
    schema=before['request']['format']
    content=(after.get('response_envelope') or {}).get('message',{}).get('content')
    if content is None:return {'json':'NOT_RECEIVED','schema':'NOT_CHECKED','errors':[]}
    try:value=json.loads(content)
    except (ValueError,TypeError):return {'json':'INVALID','schema':'NOT_CHECKED','errors':[]}
    if not isinstance(value,dict):return {'json':'VALID','schema':'INVALID','errors':[{'validator':'type','path':[]}]}
    whole=list(Draft202012Validator(schema).iter_errors(value))
    branches=[b for b in schema.get('anyOf',[]) if value.get('action') in b.get('properties',{}).get('action',{}).get('enum',[])]
    target=branches[0] if len(branches)==1 else schema
    errors=[{'validator':e.validator,'path':list(e.path),'schema_path':list(e.schema_path)}
            for e in Draft202012Validator(target).iter_errors(value)]
    return {'json':'VALID','schema':'INVALID' if whole else 'VALID','matching_action_branches':len(branches),
            'errors':errors,'evidence_citation_count':len(value.get('evidence_indices',[])),
            'unique_evidence_citation_count':len(set(value.get('evidence_indices',[])))}
