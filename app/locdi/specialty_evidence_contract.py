"""Explicit indices over Core-released evidence, never Memory or new facts."""
from copy import deepcopy
from app.locdi.llm_setup import digest

VERSION='locdi.specialty_evidence.v2'
PARENT_ERROR='SPECIALTY_EVIDENCE_CONTRACT_ERROR'
REPAIRABLE_CODES=frozenset({'SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE',
    'SPECIALTY_EVIDENCE_REQUIRED','SPECIALTY_CERTAINTY_EVIDENCE_CONTRADICTION'})
REPAIR_VERSION='locdi.specialty_semantic_repair.v1'


class EvidenceScopeError(ValueError):
    pass


def projection(payload):
    evidence=payload.get('evidence',[]);records=payload.get('evidence_records',[])
    eligible=payload.get('supported_diagnosis_eligible_indices',[])
    if (not isinstance(evidence,list) or any(not isinstance(t,str) for t in evidence)
            or not isinstance(records,list) or len(records)!=len(evidence) or not isinstance(eligible,list)):
        raise EvidenceScopeError('SPECIALTY_EVIDENCE_SCOPE_MISMATCH')
    current=payload['observation'];minute=current['elapsed_minutes'];expected=[];table=[]
    prefixes={'objective_finding':'finding:','patient_history':'history:','vital_sign':'vital:','specialty_opinion':'consult:'}
    for i,(text,record) in enumerate(zip(evidence,records)):
        if (not isinstance(record,dict) or record.get('text')!=text or record.get('kind') not in prefixes
                or type(record.get('supports_finalization')) is not bool
                or not isinstance(record.get('source_id'),str) or not record['source_id'].startswith(prefixes[record['kind']])):
            raise EvidenceScopeError('SPECIALTY_EVIDENCE_SCOPE_MISMATCH')
        if 'available_minute' in record and (type(record['available_minute']) is not int or not 0<=record['available_minute']<=minute):
            raise EvidenceScopeError('SPECIALTY_EVIDENCE_SCOPE_MISMATCH')
        if record['supports_finalization']:
            if record['kind']!='objective_finding':raise EvidenceScopeError('SPECIALTY_EVIDENCE_SCOPE_MISMATCH')
            expected.append(i)
        if record['kind']=='objective_finding':
            # Match the original Core finding slot; do not promote a future,
            # Memory or private source text into the current evidence table.
            slot=record['source_id'].split(':',1)[1];findings=current.get('objective_findings',[])
            if not slot.isdigit() or int(slot)>=len(findings) or str(findings[int(slot)])!=text:
                raise EvidenceScopeError('SPECIALTY_EVIDENCE_SCOPE_MISMATCH')
        table.append({'index':i,'kind':record['kind'],'supports_finalization':record['supports_finalization']})
    if eligible!=expected or any(type(i) is not int for i in eligible):raise EvidenceScopeError('SPECIALTY_EVIDENCE_SCOPE_MISMATCH')
    return {'evidence':deepcopy(evidence),'evidence_records':deepcopy(records),
        'supported_diagnosis_eligible_indices':deepcopy(eligible),'evidence_index_map':table,
        'evidence_contract':{'version':VERSION,'index_base':0,'index_source':'evidence',
            'allowed_indices':list(range(len(evidence))),'supported_eligible_indices':deepcopy(eligible),
            'memory_indices_are_evidence':False,'model_opinion_is_source_evidence':False,
            'supported_requires_cited_eligible_objective_finding':True,
            'empty_evidence_requires_empty_indices':True,
            'supported_permitted':bool(eligible),
            'permitted_certainties':['supported','unresolved'] if eligible else ['unresolved'],
            'unresolved_with_empty_indices_allowed':True,'scope_fingerprint':digest({'evidence':evidence,'records':records,'eligible':eligible})}}


def validate_response(value,snapshot):
    """Safe counts/categories only. Never return clinical text or cited indices."""
    indices=value['evidence_indices'];n=len(snapshot['evidence']);eligible=snapshot['supported_diagnosis_eligible_indices']
    outside=sum(type(i) is not int or not 0<=i<n for i in indices)
    code=None
    if outside:code='SPECIALTY_EVIDENCE_INDEX_OUT_OF_RANGE'
    elif value['certainty']=='supported' and not indices:code='SPECIALTY_EVIDENCE_REQUIRED'
    elif value['certainty']=='supported' and not set(indices).intersection(eligible):code='SPECIALTY_CERTAINTY_EVIDENCE_CONTRADICTION'
    return {'contract_version':VERSION,'semantic_validation':'INVALID' if code else 'VALID',
        'error_code':PARENT_ERROR if code else None,'detail_code':code,
        'evidence_count':n,'eligible_evidence_count':len(eligible),'citation_count':len(indices),
        'out_of_range_count':outside,'cited_eligible_count':len(set(indices).intersection(eligible)),
        'reported_certainty':value['certainty'],'scope_fingerprint':snapshot['evidence_contract']['scope_fingerprint']}


def repair_payload(snapshot,summary):
    """Same approved input plus safe error/range feedback; never the bad answer."""
    if summary.get('detail_code') not in REPAIRABLE_CODES:
        raise ValueError('SPECIALTY_SEMANTIC_REPAIR_NOT_PERMITTED')
    result=deepcopy(snapshot)
    result['specialty_semantic_repair']={'error_code':summary['detail_code'],
        'allowed_evidence_indices':deepcopy(snapshot['evidence_contract']['allowed_indices']),
        'supported_eligible_indices':deepcopy(snapshot['supported_diagnosis_eligible_indices'])}
    return result
