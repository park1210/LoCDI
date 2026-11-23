"""TEST_ONLY toys reuse existing .9/.1 finite outcomes, never clinical rates.

Real LoCDI event text is read unchanged to bind an observed category. Alternative
outcomes are TEST_ONLY symbols, never delivered or saved as patient evidence.
"""
from copy import deepcopy
from test_simulation import SourceActionClient, smoke_options
from test_routing_voi_readiness import test_only_registry, test_only_routing, test_only_voi


def options_for(case, *, routing=False, voi=False, abstain=False, fallback=False, failure=None):
    registry=test_only_registry(); utility=test_only_routing(registry)
    for row in utility['samples']:
        row['tokens']=['pain']  # Explicit toy support at the real case's initial observation.
    if abstain:
        utility['min_samples']=999
    model=test_only_voi(); model['actions']=[]
    for i,event in enumerate(case['environment']['current_encounter_events']):
        if event['action'] not in {'Ask','Test'}:
            continue
        ask=event['action']=='Ask'; text=event['evidence'] if ask else event['target']+': '+event['evidence']
        model['actions'].append({'id':'TEST_ONLY_ACTION_%02d'%i,'type':'ask_patient' if ask else 'request_test','target':event['target'],
            'outcomes':[{'id':'a','likelihood':{'TEST_ONLY_A':.9,'TEST_ONLY_B':.1},
                'observation':{'source':'history' if ask else 'objective_findings','text':text},'supports_diagnoses':[] if ask else ['TEST_ONLY_A']},
                {'id':'b','likelihood':{'TEST_ONLY_A':.1,'TEST_ONLY_B':.9},
                'observation':{'source':'history' if ask else 'objective_findings','text':'TEST_ONLY_OTHER_%02d'%i},'supports_diagnoses':[] if ask else ['TEST_ONLY_B']}]})
    if fallback:
        model['actions']=model['actions'][:1]
    features={'memory_mode':'selective','top_k':5,'routing_mode':'adaptive' if routing else 'er_only',
              'primary_router':'adaptive' if routing else 'rule_similarity','voi_enabled':voi,
              'action_selector':'bayesian_voi' if voi else 'specialist_heuristic'}
    return smoke_options(client=SourceActionClient(failure),registry_manifest=registry,runner_options=features,
        routing_utility=utility if routing else None,voi_model=model if voi else None,
        diagnosis_catalog=list(model['diagnoses']) if voi else None,diagnosis_departments=model['diagnoses'] if voi else None)
