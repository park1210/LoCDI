"""Opt-in public-menu covariates; original Jaccard/Beta/route math untouched.

Menu availability is not a performed test, patient evidence, symptom, diagnosis
or a likelihood. Values/hidden Source records are never feature inputs.
"""
from copy import deepcopy
from contextlib import contextmanager
from unittest.mock import patch
import re
from app.core.routing_utility import observed_tokens as legacy_tokens
from app.core.agent_registry import AgentRegistry as LegacyRegistry
from app.locdi.llm_setup import digest
from app.locdi.run_simulation import _SOCKET_LOCK

VERSION='PUBLIC_OBSERVATION_AND_DECLARED_MENU.v1'
FIELDS={'kind','code','unit'}

def public_menu(rows):
    return sorted([{'kind':r['kind'],'code':r['code'],'unit':r['unit']} for r in rows],key=digest)

def tokens(observation,menu):
    if not isinstance(menu,list):raise ValueError('PUBLIC_MENU_LIST_REQUIRED')
    out=set(legacy_tokens(observation))
    for r in menu:
        if set(r)!=FIELDS or r['kind'] not in ('Test','Ask') or not isinstance(r['code'],str) or not r['code']:
            raise ValueError('PUBLIC_MENU_CODE_ONLY_NO_VALUES_OR_REFERENCE')
        if r['unit'] is not None and not isinstance(r['unit'],str):raise ValueError('PUBLIC_MENU_UNIT')
        # Exact Code/Unit tokens. No clinical synonyms, value conversion,
        # inferred diagnosis, target ID, patient ID, or unperformed result.
        out.add('menu_kind_'+r['kind'].lower())
        out.add('menu_code_'+r['code'].casefold())
        out.add('menu_unit_'+(r['unit'] or 'unspecified').casefold())
    return sorted(out)

def fit_observation(observation,menu):
    out=deepcopy(observation);out['_routing_public_menu']=deepcopy(menu);return out

@contextmanager
def fitting_feature_scope():
    def extract(observation):
        if '_routing_public_menu' not in observation:raise ValueError('EXPLICIT_FEATURE_CONTRACT_REQUIRED')
        return tokens(observation,observation['_routing_public_menu'])
    with _SOCKET_LOCK,patch('app.core.routing_utility.observed_tokens',new=extract):yield

def registry_type(menu):
    frozen=deepcopy(menu)
    class MenuRegistry(LegacyRegistry):
        def route(self,observation,departments,*,mode='llm',seed=0):
            if mode!='adaptive':return super().route(observation,departments,mode=mode,seed=seed)
            with _SOCKET_LOCK,patch('app.core.routing_utility.observed_tokens',new=lambda _:tokens(observation,frozen)):
                result=super().route(observation,departments,mode=mode,seed=seed)
            result['feature_contract']={'version':VERSION,'public_menu_fingerprint':digest(frozen),
                'token_count':len(tokens(observation,frozen)),'menu_as_patient_evidence':False}
            return result
    return MenuRegistry

def backend(cases,*,admitted_evidence,admitter,**kwargs):
    """Scoped Native transfer; old admission functions/files are unchanged."""
    from app.locdi.augmentation_experiment_backend import make_paired_backend
    from app.locdi.terminal_only_recovery import terminal_only_scope
    kwargs['routing_evidence']=admitted_evidence
    execute=make_paired_backend(cases,**kwargs)
    bindings=kwargs['bindings']
    def run(case,features):
        menu=public_menu(bindings[case['case_ref']]['rows'])
        with _SOCKET_LOCK,patch('app.locdi.common_menu_backend.admit_development_proxy',new=admitter),\
                patch('app.locdi.common_menu_backend.AgentRegistry',new=registry_type(menu)),terminal_only_scope():
            return execute(case,features)
    return run
