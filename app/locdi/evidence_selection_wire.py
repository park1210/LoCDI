"""New closed index-key object; Core's original array validator remains final."""
from copy import deepcopy
from app.locdi.memory_citation_wire import _decision_nodes
VERSION='evidence_index_set.v1'
FIELD='evidence_selection'
def encode(payload,schema):
    p=deepcopy(payload);s=deepcopy(schema);n=len(p.get('evidence',[]))
    for node in _decision_nodes(s):
        props=node['properties'];old=props.pop('evidence_indices')
        non_diagnose=props['action'].get('enum')!=['diagnose']
        maximum=min(4,old.get('maxItems',4),n)
        props[FIELD]={'type':'object','properties':{str(i):{'const':True} for i in range(n)},
            'additionalProperties':False,'maxProperties':0 if non_diagnose else maximum}
        node['required']=[FIELD if k=='evidence_indices' else k for k in node['required']]
    p['instruction']=p.get('instruction','')+' Evidence wire evidence_index_set.v1: return evidence_selection as an object of cited index keys with true values; no evidence_indices array. Use {} for non-diagnose actions. Cite at most four available indices; literature has no patient evidence index.'
    return p,s
def decode(value,original_schema):
    from jsonschema import Draft202012Validator
    out=deepcopy(value);selected=out.pop(FIELD,None)
    if not isinstance(selected,dict) or 'evidence_indices' in out:raise ValueError('EVIDENCE_SET_REQUIRED')
    if any(not k.isdecimal() or str(int(k))!=k or v is not True for k,v in selected.items()):raise ValueError('EVIDENCE_SET_INVALID_KEY')
    if len(selected)>4:raise ValueError('EVIDENCE_SET_CAP')
    if out.get('action')!='diagnose' and selected:raise ValueError('NONDIAGNOSE_CITATIONS_FORBIDDEN')
    out['evidence_indices']=sorted(int(k) for k in selected)
    Draft202012Validator(original_schema).validate(out)
    return out
