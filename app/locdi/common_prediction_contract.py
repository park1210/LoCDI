"""Core terminal extraction and opt-in shared taxonomy selection.

No Source, scoring, clinical evidence or numerical policy changes.
"""
from copy import deepcopy

SELECTION_VERSION='shared_diagnosis_department_selection.v1'
TERMINALS=frozenset(('er_finalize','specialty_finalize'))

def extract_core_prediction(decisions,status):
    terminal=next((d for d in reversed(decisions) if d.get('action') in TERMINALS),None)
    success=status=='completed' and terminal is not None
    certainty=terminal.get('certainty') if success else 'not_available'
    return {'certainty':certainty,
            'diagnosis':terminal['target'].strip() if certainty=='supported' else None,
            'department':terminal.get('target_department') if success else None,
            'algorithmic_diagnosis_choice':None}

def shared_selection(payload,schema,taxonomy):
    """Narrow existing schema; never repair, infer a label or weaken evidence.

    Same 15-label/5-department taxonomy for every patient. Supported diagnose
    branches use only the common labels. The mapping is visible guidance. Unresolved
    is retained, and non-diagnostic action targets are untouched.
    """
    if (not isinstance(taxonomy,dict) or not taxonomy
        or any(not isinstance(k,str) or not isinstance(v,str) or not k or not v for k,v in taxonomy.items())):
        raise ValueError('SHARED_TAXONOMY_REQUIRED')
    p=deepcopy(payload);s=deepcopy(schema);departments=sorted(set(taxonomy.values()))
    if list(p.get('diagnosis_catalog',[]))!=list(taxonomy):
        raise ValueError('SHARED_DIAGNOSIS_CATALOG_MISMATCH')
    p.update(diagnosis_department_catalog=deepcopy(taxonomy),department_catalog=departments,
             output_selection_contract=SELECTION_VERSION)
    p['instruction']=p.get('instruction','')+' Choose supported diagnosis and department only from the shared taxonomy; this is a common candidate list, not this patient\'s answer. Preserve unresolved when evidence is insufficient.'
    branches=s.get('anyOf',[s]);out=[]
    for branch in branches:
        prop=branch['properties'];allowed=prop['department'].get('enum')
        if allowed is None or not set(departments)<=set(allowed):
            raise ValueError('SHARED_DEPARTMENTS_NOT_ALLOWED_BY_CORE')
        prop['department']['enum']=departments
        if prop['action'].get('enum')!=['diagnose']:
            out.append(branch);continue
        targets=prop['target'].get('enum',[]);certainties=prop['certainty']['enum']
        if 'supported' in certainties:
            if not set(taxonomy)<=set(targets):raise ValueError('SHARED_DIAGNOSIS_NOT_ALLOWED_BY_CORE')
            b=deepcopy(branch);b['properties']['target']['enum']=list(taxonomy)
            b['properties']['certainty']['enum']=['supported'];out.append(b)
        if 'unresolved' in certainties:
            if 'Unresolved' not in targets:raise ValueError('CORE_UNRESOLVED_CONTRACT_REQUIRED')
            b=deepcopy(branch);b['properties']['target']['enum']=['Unresolved']
            b['properties']['certainty']['enum']=['unresolved'];out.append(b)
    if not out:raise ValueError('NO_ALLOWED_SCHEMA_BRANCH')
    return p,{'anyOf':out} if len(out)>1 else out[0]
