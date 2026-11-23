"""Opt-in empty-object grammar refinement; identical v1 accepted values.

No illegal model response is repaired. Core's array schema and semantic
validator still apply after the unchanged v1 decoder.
"""
from app.locdi.evidence_selection_wire import encode as _v1_encode, decode
from app.locdi.memory_citation_wire import _decision_nodes

VERSION = 'evidence_index_set.strict_empty.v2'


def encode(payload, schema):
    p, s = _v1_encode(payload, schema)
    for node in _decision_nodes(s):
        selection = node['properties']['evidence_selection']
        if selection.get('maxProperties') == 0:
            # v1 declared an index property despite requiring zero properties.
            # Explicit empty properties + enum also constrain grammar backends
            # that do not enforce maxProperties while generating a response.
            selection['properties'] = {}
            selection['enum'] = [{}]
    p['instruction'] += (
        ' Evidence grammar evidence_index_set.strict_empty.v2: for request_test, '
        'ask_question and every non-diagnose action, evidence_selection MUST '
        'be exactly {}; never attach a previous result index to a test request. '
        'Existing diagnosis citation validation is unchanged.'
    )
    return p, s
