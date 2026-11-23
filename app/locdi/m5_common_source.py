"""Opt-in exact common M5 menu; original data/model/Gates remain unchanged.

No references, probabilities, clinical grants or reviewer judgments are created.
Preparation verifies actual CSV rows. Only released Runtime observations can
enter the immutable M5 outcome model. Source timing remains a declared proxy.
"""
from collections import defaultdict
from copy import deepcopy
import csv
from decimal import Decimal
import json
import math
from pathlib import Path

from app.core.longitudinal_memory import timestamp
from app.core.information_outcomes import InformationOutcomes
from app.core.information_value import InformationPolicy
from app.locdi.frozen_m5_runtime_adapter import SYSTEM, M5_SHA
from app.locdi.llm_setup import digest
from app.locdi.synthea_development_proxy import row_checks
from app.locdi.synthea_sequential_proxy import SequentialSyntheticSourceView
from app.locdi.voi_information_improvement import bucket
from app.locdi.source_paths import resolve_source_file

POLICY='M5_COMMON_NATIVE_RECORDED_TIME_PROXY.v1'


def verify_csv(events):
    grouped=defaultdict(dict)
    for event in events:
        meta=event['source_metadata'];path=resolve_source_file(meta['source_file'])
        if meta['source_row'] in grouped[path] and grouped[path][meta['source_row']]!=meta['source_record']:
            raise ValueError('CONFLICTING_RAW_ROW_CLAIMS')
        grouped[path][meta['source_row']]=meta['source_record']
    for path,wanted in grouped.items():
        found={}
        with path.open(encoding='utf-8-sig',newline='') as f:
            for n,row in enumerate(csv.DictReader(f),2):
                if n in wanted:found[n]=row
                if n>=max(wanted):break
        if found!=wanted:raise ValueError('RAW_CSV_ROW_MISMATCH')


def _candidate_case_binding(case,model):
    """Source-only selection: latest eligible timestamp, reject tied conflicts.

    Missing actions remain missing; no patient replacement or imputation.
    Condition/evaluation/task_annotations are never inspected.
    """
    visible=case['agent_visible'];identity=visible['longitudinal']
    patient=case['patient_id'];visit=visible['index_encounter_id']
    index=int(timestamp(visible['index_encounter_start']).timestamp()*1000)
    selected=[];missing=[];rejected=[]
    for key,aid in model['provenance']['action_key_mapping'].items():
        kind,system,code,unit,resource=json.loads(key)
        if system!=SYSTEM or resource!='Observation':raise ValueError('M5_SOURCE_SYSTEM_CHANGED')
        candidates=[]
        for e in case['environment']['source_event_store']:
            raw=e.get('source_metadata',{}).get('source_record',{});ev=e.get('evidence',{})
            if (e.get('event_type')!='observations' or str(e.get('target',{}).get('code'))!=code
                or raw.get('UNITS')!=unit):continue
            if (e.get('patient_id')!=patient or e.get('encounter_id')!=visit
                or raw.get('PATIENT')!=patient or raw.get('ENCOUNTER')!=visit):continue
            if e['target'].get('system') not in (None,'',SYSTEM):continue
            if raw.get('SYSTEM',SYSTEM) not in ('',SYSTEM):continue
            if raw.get('CODE')!=code or raw.get('DATE')!=e.get('timestamp') or raw.get('TYPE')!='numeric':continue
            if raw.get('VALUE')!=ev.get('value') or raw.get('UNITS')!=ev.get('units'):continue
            if (raw.get('DESCRIPTION')!=e['target'].get('description') or raw.get('CATEGORY')!=ev.get('category')
                or ev.get('kind')!='recorded_observation' or e.get('timing_precision')!='second'):continue
            try:
                number=float(raw['VALUE']);lower=int(timestamp(e['timestamp']).timestamp()*1000)
                upper=e['timestamp_upper_epoch_ms']
                if not math.isfinite(number) or upper!=lower+999 or e['timestamp_lower_epoch_ms']!=lower:continue
                available=max(0,math.ceil((upper-index)/60000))
                if available>45:continue
            except (KeyError,ValueError,TypeError):continue
            if kind=='Ask':
                if code!='72514-3':raise ValueError('UNSUPPORTED_ASK_SOURCE')
            else:
                if code=='72514-3':raise ValueError('PAIN_MUST_NOT_BECOME_TEST')
                checks=row_checks(e,patient=patient,visit=visit,index_ms=index,cutoff=45,
                                  source_row_matches=True,catalog=list(model['diagnoses']))
                if not checks['structural_candidate']:continue
            candidates.append((upper,e,available,number))
        if not candidates:missing.append(aid);continue
        latest=max(r[0] for r in candidates);tied=[r for r in candidates if r[0]==latest]
        if len({(Decimal(r[1]['source_metadata']['source_record']['VALUE']),r[1]['evidence'].get('units')) for r in tied})!=1:
            missing.append(aid);rejected.append({'action_id':aid,'reason':'LATEST_TIMESTAMP_CONFLICT'});continue
        _,e,available,number=min(tied,key=lambda r:digest(r[1]))
        action=next(a for a in model['actions'] if a['id']==aid)
        selected.append({'action_id':aid,'source_key':key,'kind':kind,'target':action['target'],
            'description':e['target']['description'],'code':code,'unit':unit,
            'source_fingerprint':digest(e),'source_event':deepcopy(e),
            'available_minute':available,'outcome_id':bucket(number,model['provenance']['binning_thresholds'][key]),
            'evidence':json.dumps({'value':e['evidence']['value'],'units':unit,'recorded_at':e['timestamp']},
                                  ensure_ascii=False,sort_keys=True,separators=(',',':'))})
    return {'patient_id':patient,'visit_id':visit,'case_id':case['case_id'],
        'started_at':visible['index_encounter_start'],'rows':selected,'missing_actions':missing,
        'rejected':rejected,'selection':'latest eligible source timestamp; no Gold access; tied conflict rejected',
        'clinical_release':'NOT_REVIEWED','policy':POLICY}


def bind_case(case,model):
    binding=_candidate_case_binding(case,model)
    verify_csv([r['source_event'] for r in binding['rows']])
    return binding


def bind_cases(cases,model):
    """Same rules as bind_case, one grouped CSV scan across all patients."""
    bindings={ref:_candidate_case_binding(case,model) for ref,case in cases.items()}
    verify_csv([r['source_event'] for binding in bindings.values() for r in binding['rows']])
    return bindings


class CommonM5SourceView(SequentialSyntheticSourceView):
    """Subclass only to use the existing typed source-view extension point."""
    def __init__(self,binding,authorization):
        if (authorization.get('scope')!='B0_B4_M5_COMMON_MENU_DEVELOPMENT_ONLY'
            or authorization.get('source_binding_fingerprint')!=digest(binding)
            or authorization.get('accept_recorded_time_proxy') is not True
            or authorization.get('accept_common_m5_menu') is not True
            or authorization.get('clinical_release_approved') is not False):
            raise ValueError('NEW_COMMON_SOURCE_OPERATOR_APPROVAL_REQUIRED')
        self.binding=deepcopy(binding)

    def metadata(self):
        return {'policy':POLICY,'fingerprint':digest(self.binding),'test_only':False,
            'turnaround_minutes':5,'clinical_release_approved':False,'v1_grants_created':0,
            'candidate_wire_projection':'exact M5 targets; source description/code/unit in separate metadata'}

    def apply(self,adapted,*,features,purpose):
        if purpose!='development_llm':raise ValueError('COMMON_MENU_DEVELOPMENT_ONLY')
        b=self.binding;runtime=deepcopy(adapted.runtime_case);identity=runtime['longitudinal']
        if (identity['patient_id'],identity['visit_id'])!=(b['patient_id'],b['visit_id']):raise ValueError('COMMON_SOURCE_IDENTITY')
        original={e['event_id']:e for e in adapted.environment['source_event_store']}
        events=[];tests=[];audit=[]
        for r in b['rows']:
            e=r['source_event']
            if original.get(e['event_id'])!=e or digest(e)!=r['source_fingerprint']:raise ValueError('COMMON_SOURCE_CHANGED')
            events.append({'event_id':'m5-common:'+r['source_fingerprint'],'action':r['kind'],
                'target':r['target'],'available_minute':r['available_minute'],'evidence':r['evidence']})
            if r['kind']=='Test':
                tests.append(r['target']);audit.append({'test_name':r['target'],'source_fingerprint':r['source_fingerprint'],
                    'available_minute':r['available_minute'],'synthetic_turnaround_minute':5})
        runtime['current_encounter_events']=events;runtime['available_tests']=tests
        profile=runtime.setdefault('resource_profile',{})
        for name in tests:
            profile.setdefault('test_turnaround',{})[name]=5
            profile.setdefault('test_occupancy',{})[name]=5
            profile.setdefault('test_resources',{})[name]='lab_slot'
        return runtime,audit


class NativeM5Outcomes:
    """Lossless raw-to-category view, bound to actual completed Runtime events."""
    def __init__(self,model,binding):
        self.original=InformationOutcomes(model);self.fingerprint=self.original.fingerprint
        self.binding=deepcopy(binding);self.runtime=None

    @property
    def artifact(self):return self.original.artifact

    def observe(self,observed):
        """Validate receipt only for Source observations actually presented.

        Test receipts require completed_tests/objective_findings/current_time_min.
        Ask receipts require question_history/current_time_min. Real Runtime
        initializes all these fields; an unobserved Ask must not read its history.
        Missing Ask history never counts as execution of a claimed Ask result.
        """
        canonical={'history':[],'objective_findings':[],
                   'longitudinal_memory':deepcopy(observed.get('longitudinal_memory',{}))}
        rt=self.runtime
        for r in self.binding['rows']:
            if r['kind']=='Test':
                text=r['target']+': '+r['evidence']
                if any(s.startswith(r['target']+': ') and s!=text for s in observed.get('objective_findings',[])):
                    raise ValueError('UNBOUND_M5_SOURCE_VALUE')
                present=text in observed.get('objective_findings',[])
            else:
                present=any(x.get('question')==r['target'] and x.get('answer')==r['evidence']
                            and x.get('answer_status')!='unknown' for x in observed.get('history',[]))
            if not present:continue
            if r['kind']=='Test':
                executed=rt is not None and r['target'] in rt.completed_tests and text in rt.objective_findings
            else:
                if rt is not None and not isinstance(getattr(rt,'question_history',None),list):
                    raise ValueError('ASK_RUNTIME_HISTORY_REQUIRED')
                executed=rt is not None and any(x.get('question')==r['target'] and x.get('answer')==r['evidence']
                    and x.get('answer_status')!='unknown' for x in rt.question_history)
            if (not executed or observed['elapsed_minutes']<r['available_minute']
                or observed['elapsed_minutes']!=rt.current_time_min):raise ValueError('UNRELEASED_M5_SOURCE')
            action=next(a for a in self.artifact['actions'] if a['id']==r['action_id'])
            outcome=next(o for o in action['outcomes'] if o['id']==r['outcome_id'])
            if r['kind']=='Test':canonical['objective_findings'].append(outcome['observation']['text'])
            else:canonical['history'].append({'question':r['target'],'answer':outcome['observation']['text'],'answer_status':'known'})
        return self.original.observe(canonical)


class NativeM5InformationPolicy(InformationPolicy):
    def __init__(self,model,binding,selector='bayesian_voi'):
        super().__init__(model,selector);self.model=NativeM5Outcomes(model,binding)

    def propose(self,runtime,observed,**kwargs):
        self.model.runtime=runtime
        proposal=super().propose(runtime,observed,**kwargs)
        if proposal is None:raise RuntimeError('BOUND_M5_UNAVAILABLE_NO_SILENT_HEURISTIC_FALLBACK')
        return proposal
