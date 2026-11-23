"""Private Source capability for C4-31; no diagnosis/scoring access in policy API."""
from copy import deepcopy
import math
from app.locdi.voi_information_improvement import SourceOutcomePanel


class SourceReplayEnvironment:
    def __init__(self, model, values, budget):
        if type(budget) is not int or not 0 <= budget <= 5:
            raise ValueError('BUDGET_RANGE')
        modeled_values = {k:v for k,v in values.items() if k in model['provenance']['action_key_mapping']}
        self._panel = SourceOutcomePanel(model, {'values': modeled_values})
        self._budget = budget
        self.clock = 0
        self.releases = []

    def acquire(self, selected):
        if selected not in self._panel.menu():
            raise ValueError('INVALID_OR_REPEATED_ACTION')
        key = next(k for k, a in self._panel.mapping.items() if a == selected)
        value = self._panel.values.get(key)
        if value is None:
            raise ValueError('MISSING_SOURCE_OUTCOME')
        minute = value.get('available_minute')
        if type(minute) not in (int, float) or not math.isfinite(minute) or minute < 0:
            raise ValueError('INVALID_RELEASE_TIME')
        if value.get('availability_basis') != 'RECORDED_TIME_PROXY':
            raise ValueError('UNVERIFIED_RELEASE_POLICY')
        before = self.clock
        result = self._panel.acquire(selected, selected=selected,
                                     budget_remaining=self._budget-len(self.releases))
        self.clock = max(self.clock, minute)
        self.releases.append({**result, 'selected_at_minute': before,
                              'released_at_minute': self.clock})
        # A policy receives only this selected encoded observation, after clock advance.
        return {'action_id': selected, 'outcome_id': result['outcome_id'],
                'released_at_minute': self.clock}

    def private_trace(self):
        return deepcopy(self.releases)
