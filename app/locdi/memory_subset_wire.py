"""Finite subset enum, preserving original Core citation/evidence validators."""
from copy import deepcopy
from itertools import combinations
from math import comb
from app.locdi.memory_citation_wire import _decision_nodes

VERSION='memory_citation_subset_code.v1'
FIELD='memory_subset_code'
MAX_ENUM_CODES=2048
INSTRUCTION=(' Output wire memory_citation_subset_code.v1: return memory_subset_code '
    'instead of memory_ids. Use exactly {width} binary digits; position 0 is the '
    'first supplied memory record. 1 cites that record, 0 does not. Select only '
    'an allowed schema enum code, at most {cap} records actually used; all zeros '
    'means no citation. No unprovided or superseded records. All clinical rules remain unchanged.')


def allowed_codes(records,cap=4):
    if type(cap) is not int or not 0<=cap<=4:raise ValueError('ORIGINAL_CITATION_CAP_REQUIRED')
    ids=[r.get('memory_id') for r in records]
    if any(not isinstance(x,str) or not x for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError('MEMORY_SOURCE_IDENTITIES_NOT_UNIQUE')
    eligible=[i for i,r in enumerate(records) if r.get('status_as_of_visit')!='superseded']
    count=sum(comb(len(eligible),k) for k in range(min(cap,len(eligible))+1))
    if count>MAX_ENUM_CODES:raise ValueError('SUBSET_ENUM_CONTEXT_BUDGET')
    width=max(1,len(records));codes=[]
    for k in range(min(cap,len(eligible))+1):
        for selected in combinations(eligible,k):
            bits=['0']*width
            for i in selected:bits[i]='1'
            codes.append(''.join(bits))
    return sorted(codes)


def encode(payload,schema):
    out,wire=deepcopy(payload),deepcopy(schema)
    records=out.get('observation',{}).get('longitudinal_memory',{}).get('records',[])
    nodes=_decision_nodes(wire)
    if not nodes and wire.get('type')=='object':nodes=[wire]
    if not nodes:raise ValueError('EXPLICIT_DECISION_SCHEMA_REQUIRED')
    original_caps=[]
    for node in nodes:
        props=node.setdefault('properties',{})
        cap=min(4,props.get('memory_ids',{}).get('maxItems',0 if not records else 4))
        codes=allowed_codes(records,cap);original_caps.append(cap)
        props.pop('memory_ids',None)
        props[FIELD]={'type':'string','enum':codes}
        node['required']=[x for x in node.get('required',[]) if x!='memory_ids']
        if FIELD not in node['required']:node['required'].append(FIELD)
    cap=min(original_caps)
    mapping={'records':deepcopy(records),'codes':allowed_codes(records,cap),'cap':cap}
    out['instruction']=out.get('instruction','')+INSTRUCTION.format(width=max(1,len(records)),cap=cap)
    return out,wire,mapping


def decode(value,mapping,original_schema):
    from jsonschema import Draft202012Validator
    if not isinstance(value,dict) or value.get(FIELD) not in mapping['codes']:
        raise ValueError('MEMORY_SUBSET_CODE_NOT_ALLOWED')
    if 'memory_ids' in value:raise ValueError('LEGACY_FIELD_NOT_ALLOWED_IN_SUBSET_WIRE')
    code=value[FIELD];records=mapping['records']
    selected=[records[i]['memory_id'] for i,b in enumerate(code) if b=='1']
    out=deepcopy(value);out.pop(FIELD)
    if any('memory_ids' in n.get('properties',{}) for n in _decision_nodes(original_schema)):
        out['memory_ids']=selected
    elif selected:raise ValueError('NO_INTERNAL_MEMORY_CITATION_FIELD')
    Draft202012Validator(original_schema).validate(out)
    return out
