"""Pinned original-file loader and copy-isolated external OSCE projection."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from app.locdi.agentclinic_scenario_schema import (
    REPOSITORY, REVISION, SOURCE_SHA256, GROUP, POLICY_VERSION, Scenario,
    Target, ScoringKey, ScenarioBundle, ScenarioContractError, validate_record)
from app.locdi.reference_acquisition import strict_json
from app.locdi.llm_setup import digest

ASK_FIELDS = {'demographics': ('Demographics',), 'history': ('History',),
              'symptoms': ('Symptoms',), 'primary_symptom': ('Symptoms', 'Primary_Symptom'),
              'secondary_symptoms': ('Symptoms', 'Secondary_Symptoms'),
              'past_medical_history': ('Past_Medical_History',),
              'social_history': ('Social_History',), 'review_of_systems': ('Review_of_Systems',),
              'current_medications': ('Current_Medications',), 'medications': ('Medications',),
              'drug_history': ('Drug_History',)}


def _targets(store, prefix):
    result = []
    def visit(value, path):
        # Each nonempty container is an explicitly requestable group; descendants
        # are separate exact targets. Source order is retained, aliases are absent.
        if path:
            result.append(Target(prefix + '/%04d' % (len(result) + 1),
                                 ' / '.join(path), path, deepcopy(value),
                                 'group' if isinstance(value, (dict, list)) else 'leaf'))
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, path + (k,))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(v, path + (str(i),))
    visit(store, ())
    return tuple(result)


def _adapt(row, ordinal, source):
    if type(ordinal) is not int or ordinal < 1:
        raise ScenarioContractError('ORDINAL')
    c = validate_record(row)
    fingerprint = digest(row)
    sid = 'AC%03d_%s' % (ordinal, fingerprint[:16])
    source = dict(source, ordinal=ordinal, record_fingerprint=fingerprint,
                  evaluation_group=GROUP, policy_version=POLICY_VERSION)
    actor = deepcopy(c['Patient_Actor'])
    asks = []
    for name, path in ASK_FIELDS.items():
        value = actor
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            asks.append(Target(name, ' / '.join(path), path, deepcopy(value),
                               'group' if isinstance(value, (dict, list)) else 'leaf'))
    exams, tests = _targets(c['Physical_Examination_Findings'], 'exam'), _targets(c['Test_Results'], 'test')
    label = c['Correct_Diagnosis'].strip()
    # A source target label must not expose the private answer before selection.
    def literal(text):
        return re.search(r'(?<!\w)' + re.escape(label) + r'(?!\w)', text, re.I) is not None
    if literal(c['Objective_for_Doctor']) or any(literal(t.label) or literal(t.label.replace('_', ' '))
                                               for t in (*asks, *exams, *tests)):
        raise ScenarioContractError('INITIAL_OR_CATALOG_REFERENCE_EXPOSURE')
    scenario = Scenario(sid, source, c['Objective_for_Doctor'], actor,
                        deepcopy(c['Physical_Examination_Findings']), deepcopy(c['Test_Results']),
                        tuple(asks), exams, tests)
    return scenario, ScoringKey(sid, fingerprint, c['Correct_Diagnosis'])


def adapt_test_record(row, ordinal=1):
    """Explicit nonclinical fixture path, never source authentication."""
    return _adapt(deepcopy(row), ordinal, {'repository': 'TEST_ONLY', 'revision': 'TEST_ONLY',
                                          'original_sha256': None, 'test_only': True})


def load_scenarios(path, *, revision=REVISION):
    if revision != REVISION:
        raise ScenarioContractError('SOURCE_REVISION_MISMATCH')
    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ScenarioContractError('SOURCE_SIZE_LIMIT')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ScenarioContractError('SOURCE_HASH_MISMATCH')
    try:
        rows = [strict_json(line) for line in data.decode('utf8').splitlines()]
    except (ValueError, UnicodeError):
        raise ScenarioContractError('SOURCE_JSONL_INVALID') from None
    if len(rows) != 107:
        raise ScenarioContractError('SOURCE_RECORD_COUNT')
    source = {'repository': REPOSITORY, 'revision': REVISION, 'file': 'agentclinic_medqa.jsonl',
              'original_sha256': SOURCE_SHA256, 'test_only': False}
    pairs = [_adapt(r, i, source) for i, r in enumerate(rows, 1)]
    return ScenarioBundle(tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))


def action_catalog(scenario):
    return {kind: [{'target': t.target_id, 'label': t.label, 'node_kind': t.node_kind}
                   for t in targets]
            for kind, targets in (('ASK', scenario.ask_targets), ('EXAMINE', scenario.exam_targets),
                                  ('TEST', scenario.test_targets))}
