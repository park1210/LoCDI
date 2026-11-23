"""Read-only audit helpers. No clinical equivalence, release approval or model."""
from copy import deepcopy
from datetime import date
import math
import re
import unicodedata
from jsonschema import Draft202012Validator


def normalize(text):
    if not isinstance(text,str):return None
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def matches(prediction,labels):
    return {'source_exact':prediction in labels if isinstance(prediction,str) else None,
        'normalized_match':normalize(prediction) in [normalize(x) for x in labels] if isinstance(prediction,str) else None,
        'verified_concept_match':None,'concept_status':'NOT_EVALUATED'}


def diagnosis_branch(schema):
    branches=schema.get('anyOf',[schema])
    return [b for b in branches if b.get('properties',{}).get('action',{}).get('enum')==['diagnose']]


def schema_selectable(schema,response,label):
    """Offline local schema probe only. Never send the modified object to LLM."""
    value=deepcopy(response);value['target']=label
    return Draft202012Validator(schema).is_valid(value)


def proxy_minute(source,index_epoch_ms):
    upper=source.get('timestamp_upper_epoch_ms')
    if type(upper) is not int:return None
    return max(0,math.ceil((upper-index_epoch_ms)/60000))


def reference_time(start,index):
    """Day precision does not justify inventing midnight or a release time."""
    try:delta=(date.fromisoformat(start[:10])-date.fromisoformat(index[:10])).days
    except (ValueError,TypeError):return {'status':'TIMING_UNKNOWN','day_difference':None}
    return {'status':'SOURCE_CONDITION_AFTER_INDEX_DAY' if delta>0 else 'EARLIER_DAY' if delta<0 else
            'SAME_DAY_INTRADAY_NOT_VERIFIED','day_difference':delta}


def source_observation_matches(record,event):
    target=event.get('target',{});evidence=event.get('evidence',{})
    return (str(record.get('code'))==str(target.get('code'))
        and record.get('recorded_at')==event.get('timestamp')
        and record.get('recorded_evidence')==evidence)


def typed_objective_candidate(event):
    """Gold-independent candidate whitelist; does NOT authorize disclosure."""
    e=event.get('evidence',{})
    if event.get('event_type')!='observations' or e.get('category') not in ('laboratory','vital-signs'):
        return False
    try:value=float(e['value'])
    except (ValueError,TypeError,KeyError):return False
    return math.isfinite(value)


def disclosure_status(event,cutoff,index_epoch_ms,projected=False):
    minute=proxy_minute(event,index_epoch_ms)
    if minute is None:return 'TIMESTAMP_UNKNOWN'
    if minute>cutoff:return 'AFTER_FIXED_CUTOFF'
    if projected:return 'EXISTING_PROXY_REPLAY_MENU'
    return 'PROXY_TIME_ELIGIBLE_RELEASE_NOT_REVIEWED'
