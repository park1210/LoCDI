"""Finite citation enum: no truncation, automatic deletion or semantic repair."""
from copy import deepcopy
from itertools import combinations
from math import comb
import re
from app.locdi.memory_citation_wire import _decision_nodes

VERSION='evidence_citation_subset_code.v3'
FIELD='evidence_subset_code'
MAX_ENUM_CODES=2048


def allowed_codes(count, cap=4):
    if type(count) is not int or count<0 or type(cap) is not int or not 0<=cap<=4:
        raise ValueError('EVIDENCE_SUBSET_COUNT_OR_CAP')
    if sum(comb(count,k) for k in range(min(count,cap)+1))>MAX_ENUM_CODES:
        raise ValueError('EVIDENCE_SUBSET_ENUM_BUDGET')
    return ['NONE']+[','.join(map(str,indices)) for k in range(1,min(count,cap)+1) for indices in combinations(range(count),k)]


def encode(payload,schema):
    p,s=deepcopy(payload),deepcopy(schema);n=len(p.get('evidence',[]));nodes=_decision_nodes(s)
    if not nodes:raise ValueError('EXPLICIT_DECISION_SCHEMA_REQUIRED')
    for node in nodes:
        props=node['properties'];old=props.pop('evidence_indices')
        cap=min(4,old.get('maxItems',4)) if props['action'].get('enum')==['diagnose'] else 0
        props[FIELD]={'type':'string','enum':allowed_codes(n,cap)}
        node['required']=[FIELD if key=='evidence_indices' else key for key in node['required']]
    p['instruction']=p.get('instruction','')+(
        ' Output wire evidence_citation_subset_code.v3: instead of evidence_indices or evidence_selection, '
        'return evidence_subset_code as exactly one allowed schema enum string. '
        'NONE means no citation. Otherwise list chosen zero-based evidence indices in ascending order '
        'separated by commas, for example "0,2". Select at most four supplied indices. '
        'For every non-diagnose action use NONE. All clinical support rules remain unchanged; '
        'Memory and external literature are not patient evidence indices.'
    )
    return p,s


def decode(value,original_schema):
    from jsonschema import Draft202012Validator
    if not isinstance(value,dict) or 'evidence_indices' in value or 'evidence_selection' in value:
        raise ValueError('EVIDENCE_SUBSET_EXCLUSIVE_FIELD')
    out=deepcopy(value);code=out.pop(FIELD,None)
    if code=='NONE':indices=[]
    else:
        if not isinstance(code,str) or not re.fullmatch(r'(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*',code):
            raise ValueError('EVIDENCE_SUBSET_INVALID_CODE')
        indices=[int(i) for i in code.split(',')]
        if indices!=sorted(set(indices)) or len(indices)>4:raise ValueError('EVIDENCE_SUBSET_ORDER_DUPLICATE_OR_CAP')
    if out.get('action')!='diagnose' and indices:raise ValueError('NONDIAGNOSE_CITATIONS_FORBIDDEN')
    out['evidence_indices']=indices
    Draft202012Validator(original_schema).validate(out)
    return out
