"""External exact-ID availability policy. Source lookup and values stay unchanged."""
from copy import deepcopy

VERSION='agentclinic.state_aware_actions.v1'
KINDS=('ASK','EXAMINE','TEST')


class ActionAvailability:
    def __init__(self,scenario_ref,catalog,*,enabled):
        if type(enabled) is not bool or not isinstance(scenario_ref,str):raise ValueError('POLICY_SCOPE')
        if set(catalog)!=set(KINDS):raise ValueError('POLICY_CATALOG')
        self.scenario_ref=scenario_ref;self.catalog=deepcopy(catalog);self.enabled=enabled
        self.ids={kind:{t['target'] for t in catalog[kind]} for kind in KINDS}
        self.consumed=set()

    def snapshot(self):return [{'action':a,'target':t} for a,t in sorted(self.consumed)]

    def available_catalog(self):
        return {kind:[deepcopy(t) for t in self.catalog[kind] if not self.enabled or (kind,t['target']) not in self.consumed] for kind in KINDS}

    def classify(self,action):
        kind,target=action.get('action'),action.get('target')
        if kind=='DIAGNOSE':return True,None
        if kind not in KINDS or target not in self.ids.get(kind,set()):return False,'UNAVAILABLE'
        if self.enabled and (kind,target) in self.consumed:return False,'ALREADY_OBSERVED'
        return True,None

    def observe(self,event):
        kind,target=event.get('action'),event.get('target')
        if (kind in KINDS and target in self.ids[kind] and event.get('status')=='EXECUTED'
                and event.get('validation')=='VALID' and event.get('observation_ref') and event.get('observation') is not None):
            self.consumed.add((kind,target))
