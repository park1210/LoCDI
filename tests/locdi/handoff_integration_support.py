"""TEST_ONLY: existing deterministic Client/Utility/VoI fixtures, no new rates."""
from dataclasses import replace
import json
from runtime_integration_support import options_for
from test_sequential_handoff import HandoffClient


class IntegrationClient(HandoffClient):
    def __init__(self,mode=None):
        super().__init__(mode);self.specialty_calls=0

    def chat_json(self,system_prompt,user_prompt,model=None,*,schema=None):
        p=json.loads(user_prompt)
        if p['handoff_stage']=='SPECIALTY_ASSESSMENT' and self.mode in {'semantic_repair','semantic_failure'}:
            self.specialty_calls+=1;self.calls.append({'payload':p,'system':system_prompt,'model':model})
            invalid=self.specialty_calls==1 or self.mode=='semantic_failure'
            return json.dumps({'assessment':'TEST_ONLY unverified opinion','certainty':'supported' if invalid else 'unresolved',
                'evidence_indices':[0] if invalid else [],'completed':True})
        if self.mode=='later_failure' and p['handoff_stage']=='ER_REASSESSMENT' and len(self.calls)>=3:
            raise TimeoutError('TEST_ONLY private error')
        return super().chat_json(system_prompt,user_prompt,model,schema=schema)


def integrated_options(case,*,routing=False,voi=False,abstain=False,fallback=False,mode=None):
    options=options_for(case,routing=routing,voi=voi,abstain=abstain,fallback=fallback)
    # Routing is done at the outer logical Handoff only. Core stays ER-owned
    # so its existing policy does not replace ER with an in-place specialty.
    features=dict(options.runner_options,routing_mode='er_only',primary_router='rule_similarity')
    return replace(options,client=IntegrationClient(mode),runner_options=features)
