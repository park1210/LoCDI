"""Opt-in unique memory citation wire contract; no clinical inference.

The model chooses distinct keys, not a repeated array of opaque IDs. The
internal Core list is reconstructed from those selected keys without deduping
or repairing an invalid response. Original lists remain the default contract.
"""
from copy import deepcopy

VERSION = 'memory_citation_index_set.v1'
INSTRUCTION = (
    ' Wire contract memory_citation_index_set.v1: memory_ids MUST be a JSON object, '
    'not an array. Each key is the zero-based position of a used record in '
    'observation.longitudinal_memory.records, and its value must be true. '
    'Use each key once, at most the permitted citation count; cite only non-superseded records actually used. '
    'Use {} when no memory was used. This replaces only the memory_ids array '
    'output instruction; all clinical evidence and action rules are unchanged.'
)


def _decision_nodes(value):
    """Core uses anyOf branches, each with its own closed object schema."""
    found=[]
    if isinstance(value,dict):
        props=value.get('properties',{})
        if value.get('type')=='object' and ('action' in props or 'memory_ids' in props):
            found.append(value)
        for child in value.values():found.extend(_decision_nodes(child))
    elif isinstance(value,list):
        for child in value:found.extend(_decision_nodes(child))
    return found


def encode(payload, schema):
    if not isinstance(payload, dict) or not isinstance(schema, dict):
        raise ValueError('EXPLICIT_PAYLOAD_AND_SCHEMA_REQUIRED')
    out, wire = deepcopy(payload), deepcopy(schema)
    records=out.get('observation',{}).get('longitudinal_memory',{}).get('records',[])
    if not isinstance(records,list) or any(not isinstance(r,dict) for r in records):
        raise ValueError('MEMORY_RECORD_LIST_REQUIRED')
    ids=[r.get('memory_id') for r in records]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('MEMORY_INPUT_IDENTITIES_NOT_UNIQUE')
    mapping={str(i):r['memory_id'] for i,r in enumerate(records)
        if r.get('status_as_of_visit')!='superseded'}
    nodes=_decision_nodes(wire)
    # An empty no-memory closed object is also useful in pure contract tests.
    if not nodes and wire.get('type')=='object':nodes=[wire]
    if not nodes:raise ValueError('DECISION_OBJECT_SCHEMA_REQUIRED')
    for node in nodes:
        props=node.setdefault('properties',{})
        cap=props.get('memory_ids',{}).get('maxItems',len(mapping))
        if type(cap) is not int or cap<0:raise ValueError('INVALID_ORIGINAL_CITATION_CAP')
        props['memory_ids']={'type':'object','properties':{k:{'const':True} for k in mapping},
            'additionalProperties':False,'maxProperties':min(cap,len(mapping))}
        if 'memory_ids' not in node.setdefault('required',[]):node['required'].append('memory_ids')
    out['instruction']=out.get('instruction','')+INSTRUCTION
    return out,wire,mapping


def decode(value,mapping,original_schema):
    """Reject invalid wire shapes; never accept/dedupe the legacy bad array."""
    from jsonschema import Draft202012Validator
    if not isinstance(value,dict) or not isinstance(value.get('memory_ids'),dict):
        raise ValueError('MEMORY_CITATION_OBJECT_REQUIRED')
    selected=value['memory_ids']
    if any(k not in mapping or selected[k] is not True for k in selected):
        raise ValueError('MEMORY_CITATION_KEY_OR_VALUE_INVALID')
    result=deepcopy(value)
    result['memory_ids']=[mapping[k] for k in selected]
    # Original B0 schema omits optional memory_ids. An empty new wire object
    # corresponds to absence, not an invented internal field.
    if not any('memory_ids' in n.get('properties',{}) for n in _decision_nodes(original_schema)):
        if result['memory_ids']:raise ValueError('NO_INTERNAL_MEMORY_CITATION_FIELD')
        result.pop('memory_ids')
    Draft202012Validator(original_schema).validate(result)
    return result
