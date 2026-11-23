"""Private routing collection contracts. Never generates, judges or fits.

Public validation reports contain codes/counts only. Conversion returns PRIVATE
Core fit input rows, never a Utility. Review records are caller-supplied evidence;
checking their linkage cannot independently establish clinical truth.
"""
from copy import deepcopy
from datetime import datetime
import re

from app.core.agent_registry import AgentRegistry
from app.core.routing_utility import observed_tokens
from app.locdi.llm_setup import digest
from validation.artifact_readiness import observation_scope_valid

SPLITS = {'train', 'validation', 'test'}
ACTIONS = {'ask_patient', 'request_test', 'consult_specialty', 'diagnose', 'retrieve_memory', 'wait'}
EVALUATION_METHODS = {'independent_reference_match', 'blinded_clinical_review', 'independent_benchmark_scorer'}
DECISION_FIELDS = {'patient_id', 'visit_id', 'decision_id', 'minute', 'observation', 'source_ref',
                   'split', 'registry_fingerprint', 'tokens', 'generation', 'allowed_actions'}


def _private_input(value):
    forbidden = {'gold', 'gold_label', 'evaluation', 'provenance', 'task_annotations', 'source_event_store',
                 'current_encounter_events', 'hidden_test_results', 'data_replay', 'future_results',
                 'expected_diagnoses', 'accepted_diagnoses', 'accepted_departments'}
    if isinstance(value, dict):
        return any(k.casefold() in forbidden or _private_input(v) for k, v in value.items())
    return isinstance(value, list) and any(_private_input(v) for v in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('TIMEZONE_REQUIRED')
    return parsed


def _review(value):
    try:
        return (isinstance(value, dict) and value.get('status') == 'REVIEWED'
                and all(_text(value.get(k)) for k in ('reviewer_ref', 'procedure_version', 'reviewed_at'))
                and _time(value['reviewed_at']) is not None)
    except (ValueError, TypeError, AttributeError):
        return False


def _leaves(value, prefix=''):
    """JSON pointer -> value, including empty collections, for release coverage."""
    if isinstance(value, dict) and value:
        result = {}
        for key, item in value.items():
            result.update(_leaves(item, prefix + '/' + key.replace('~', '~0').replace('/', '~1')))
        return result
    if isinstance(value, list) and value:
        result = {}
        for index, item in enumerate(value):
            result.update(_leaves(item, prefix + '/' + str(index)))
        return result
    return {prefix: value}


def validate_split_manifest(manifest, *, holdout_patient_ids, dataset_fingerprint):
    """Validate explicit assignments; never invent or shuffle a split."""
    codes, counts = [], {s: 0 for s in sorted(SPLITS)}
    try:
        if (not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'dataset_fingerprint', 'holdout_fingerprint', 'patients', 'decisions'}
                or manifest['schema_version'] != 'locdi.routing_split.v1'
                or not _sha(dataset_fingerprint) or manifest['dataset_fingerprint'] != dataset_fingerprint):
            raise ValueError()
        holdout = list(holdout_patient_ids)
        if not holdout or any(not _text(x) for x in holdout) or len(set(holdout)) != len(holdout):
            raise ValueError()
        if manifest['holdout_fingerprint'] != digest(sorted(holdout)):
            codes.append('HOLDOUT_FINGERPRINT_MISMATCH')
        patients = manifest['patients']
        if not isinstance(patients, dict) or set(patients) != SPLITS:
            raise ValueError()
        owner = {}
        for split, values in patients.items():
            if not isinstance(values, list) or any(not _text(x) for x in values):
                raise ValueError()
            counts[split] = len(values)
            for patient in values:
                if patient in owner:
                    codes.append('PATIENT_SPLIT_OVERLAP_OR_DUPLICATE')
                owner[patient] = split
        if any(owner.get(p) != 'test' for p in holdout):
            codes.append('FINAL_HOLDOUT_NOT_RESERVED')
        seen = set()
        if not isinstance(manifest['decisions'], list):
            raise ValueError()
        for decision in manifest['decisions']:
            if not isinstance(decision, dict) or set(decision) != {'patient_id', 'decision_id', 'split'}:
                raise ValueError()
            if not _text(decision['decision_id']) or decision['decision_id'] in seen:
                codes.append('DECISION_ID_DUPLICATE_OR_INVALID')
            seen.add(decision['decision_id'])
            if decision['split'] not in SPLITS or owner.get(decision['patient_id']) != decision['split']:
                codes.append('DECISION_SPLIT_MISMATCH')
    except (ValueError, TypeError, KeyError, AttributeError):
        codes.append('SPLIT_MANIFEST_INVALID')
    return {'status': 'VALID' if not codes else 'BLOCKED', 'blocker_codes': sorted(set(codes)),
            'train_patient_count': counts['train'], 'validation_patient_count': counts['validation'],
            'evaluation_patient_count': counts['test']}


def _source_valid(decision, source):
    try:
        for key in ('patient_id', 'visit_id', 'decision_id', 'minute', 'observation'):
            if decision[key] != source[key]:
                return False
        if (not _text(decision['patient_id']) or not _text(decision['visit_id']) or not _text(decision['decision_id'])
                or type(decision['minute']) is not int or decision['minute'] < 0
                or not _sha(source['source_fingerprint']) or not _review(source['review'])
                or not observation_scope_valid(decision['observation']) or _private_input(decision['observation'])):
            return False
        # Every leaf, including empty containers, needs a reviewed release bound.
        leaves, visibility = _leaves(source['observation']), source['visibility']
        if not isinstance(visibility, dict) or set(visibility) != set(leaves):
            return False
        if any(type(t) is not int or not 0 <= t <= decision['minute'] for t in visibility.values()):
            return False
        if source['observation'].get('elapsed_minutes') != decision['minute']:
            return False
        memory = source['observation'].get('longitudinal_memory', {}).get('records', [])
        if memory:
            cutoff = _time(source['visit_started_at'])
            for record in memory:
                if (record['patient_id'] != decision['patient_id'] or record['visit_id'] == decision['visit_id']
                        or _time(record['available_at']) > cutoff or _time(record['visit_started_at']) >= cutoff):
                    return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def prepare_collection(decision, registry_manifest, source):
    """Return a PRIVATE task plan, not responses. No client is accepted.

    All specialties receive the same observation, Memory, actions and decoding.
    Their canonical role/dept and prompt fingerprint remain explicitly bound.
    """
    try:
        registry = AgentRegistry(registry_manifest)
        agents = [a for a in registry.snapshot()['agents'] if a['role'] == 'specialty']
        generation = decision['generation']
        if (not isinstance(decision, dict) or set(decision) != DECISION_FIELDS
                or not agents or not _source_valid(decision, source)
                or decision['registry_fingerprint'] != digest(registry.snapshot())
                or decision['tokens'] != observed_tokens(decision['observation'])
                or not isinstance(decision['allowed_actions'], list) or not decision['allowed_actions']
                or len(set(decision['allowed_actions'])) != len(decision['allowed_actions'])
                or not set(decision['allowed_actions']) <= ACTIONS
                or set(generation) != {'model_identifier', 'temperature', 'seed', 'max_output_tokens', 'prompt_fingerprints'}
                or not isinstance(generation['model_identifier'], str)
                or re.fullmatch('sha256:[0-9a-f]{64}', generation['model_identifier']) is None
                or type(generation['temperature']) not in (int, float) or not 0 <= generation['temperature'] <= 2
                or type(generation['seed']) is not int
                or type(generation['max_output_tokens']) is not int or not 1 <= generation['max_output_tokens'] <= 32768
                or set(generation['prompt_fingerprints']) != {a['agent_id'] for a in agents}
                or not all(_sha(v) for v in generation['prompt_fingerprints'].values())):
            raise ValueError()
        private = deepcopy(decision)
        fingerprint = digest(private)
        return {'status': 'PLANNED_NOT_EXECUTED', 'configuration_fingerprint': fingerprint,
                'tasks': [{'agent_ref': 'Agent%03d' % (i + 1), 'binding': deepcopy(agent),
                           'shared_input': deepcopy(private), 'configuration_fingerprint': fingerprint}
                          for i, agent in enumerate(agents)]}
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ValueError('COLLECTION_INPUT_CONTRACT_INVALID') from None


def _index(rows, key):
    if not isinstance(rows, list):
        raise ValueError()
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not _text(row.get(key)) or row[key] in result:
            raise ValueError()
        result[row[key]] = row
    return result


def validate_training_data(data, *, registry_manifest, source_catalog, label_catalog,
                           holdout_patient_ids, dataset_fingerprint, allow_test_only=False):
    """No verdict inference. Caller supplies separately reviewed private catalogs.

    These checks establish transport/linkage, not reviewer authenticity or label
    medical validity. Research readiness is always false. Missing is not false.
    """
    report = {'registry_status': 'BLOCKED', 'source_data_status': 'MISSING', 'split_status': 'MISSING',
              'correctness_status': 'MISSING', 'complete_group_count': 0, 'train_patient_count': 0,
              'evaluation_patient_count': len(set(holdout_patient_ids)), 'training_readiness': 'BLOCKED',
              'research_readiness': False, 'blocker_codes': [], 'generation_requests': 0, 'utility_fit_count': 0}
    codes = report['blocker_codes']
    try:
        registry = AgentRegistry(registry_manifest)
        agents = {a['agent_id']: a for a in registry.snapshot()['agents'] if a['role'] == 'specialty'}
        if not agents:
            raise ValueError()
        report['registry_status'] = 'VALID'
    except (ValueError, TypeError, KeyError, AttributeError):
        codes.append('REGISTRY_INVALID')
        return report
    if data is None:
        codes.extend(['COLLECTION_DATA_MISSING', 'SPLIT_MANIFEST_MISSING', 'REVIEWED_CORRECTNESS_MISSING'])
        return report
    try:
        if data['schema_version'] != 'locdi.routing_collection.v1' or data['registry'] != registry.snapshot():
            codes.append('REGISTRY_SNAPSHOT_OR_DATA_SCHEMA_MISMATCH')
        purpose = data['purpose']
        if purpose not in {'routing_training', 'TEST_ONLY'} or purpose == 'TEST_ONLY' and allow_test_only is not True:
            codes.append('TEST_ONLY_OR_PURPOSE_NOT_AUTHORIZED')
        split = validate_split_manifest(data['split_manifest'], holdout_patient_ids=holdout_patient_ids,
                                        dataset_fingerprint=dataset_fingerprint)
        report.update({k: split[k] for k in ('train_patient_count', 'evaluation_patient_count')})
        report['split_status'] = split['status']
        codes.extend(split['blocker_codes'])
        sources = _index(source_catalog, 'source_ref')
        labels = _index(label_catalog, 'label_ref')
        decisions = _index(data['decisions'], 'decision_id')
        if len({digest(d['generation']) for d in decisions.values()}) > 1:
            codes.append('GENERATION_CONFIGURATION_NOT_FIXED')
        outcomes = _index(data['outcomes'], 'response_ref')
        judgments = _index(data['judgments'], 'response_ref')
        assignments = {d['decision_id']: d for d in data['split_manifest']['decisions']}
        if set(decisions) != set(assignments):
            codes.append('DECISION_SCOPE_MISMATCH')
        group_ok = set()
        for did, decision in decisions.items():
            group_codes = []
            source = sources.get(decision.get('source_ref'))
            if not source or source.get('purpose') != purpose:
                group_codes.append('SOURCE_LINK_OR_PURPOSE_MISMATCH')
            try:
                plan = prepare_collection(decision, registry.snapshot(), source)
            except ValueError:
                plan = None
                group_codes.append('OBSERVATION_VISIBILITY_OR_COLLECTION_CONTRACT_INVALID')
            assigned = assignments.get(did, {})
            if decision['patient_id'] != assigned.get('patient_id') or decision['split'] != assigned.get('split'):
                group_codes.append('DECISION_SPLIT_MISMATCH')
            rows = [r for r in outcomes.values() if r.get('decision_id') == did]
            if len(rows) != len(agents) or {r.get('agent_id') for r in rows} != set(agents):
                group_codes.append('SPECIALTY_GROUP_INCOMPLETE_OR_DUPLICATE')
            for row in rows:
                agent = agents.get(row.get('agent_id'))
                if not agent or row.get('model') != agent['model'] or row.get('version') != agent['version']:
                    group_codes.append('MODEL_VERSION_BINDING_MISMATCH')
                if (not plan or row.get('configuration_fingerprint') != plan['configuration_fingerprint']
                        or row.get('patient_id') != decision['patient_id'] or row.get('purpose') != purpose
                        or row.get('source_ref') != decision.get('source_ref')
                        or row.get('observation_fingerprint') != digest(decision['observation'])
                        or row.get('tokens') != decision['tokens'] or row.get('generation') != decision['generation']):
                    group_codes.append('SHARED_INPUT_OR_SOURCE_BINDING_MISMATCH')
                if (row.get('execution_status') != 'SUCCEEDED' or row.get('response_contract_status') != 'VALID'
                        or not _sha(row.get('response_fingerprint'))
                        or row.get('observed_model_identifier') != decision['generation']['model_identifier']):
                    group_codes.append('RESPONSE_NOT_EVALUABLE')
                judgment = judgments.get(row['response_ref'], {})
                label = labels.get(judgment.get('label_ref'), {})
                if (judgment.get('status') != 'REVIEWED' or type(judgment.get('correct')) is not bool
                        or not _review(judgment.get('review'))):
                    group_codes.append('CORRECTNESS_NOT_REVIEWED')
                if (not label or not _review(label.get('review')) or label.get('clinical_review_status') != 'REVIEWED'
                        or label.get('evaluation_method') not in EVALUATION_METHODS or not _text(label.get('source_ref'))
                        or label.get('independent_reference') is not True or not _sha(label.get('source_fingerprint'))
                        or label.get('purpose') != purpose or not _sha(label.get('label_fingerprint'))
                        or label.get('patient_id') != decision['patient_id'] or label.get('decision_id') != did
                        or judgment.get('patient_id') != decision['patient_id'] or judgment.get('decision_id') != did
                        or judgment.get('agent_id') != row.get('agent_id')
                        or judgment.get('response_fingerprint') != row.get('response_fingerprint')
                        or judgment.get('label_fingerprint') != label.get('label_fingerprint')
                        or judgment.get('evaluation_method') != label.get('evaluation_method')
                        or judgment.get('purpose') != purpose):
                    group_codes.append('INDEPENDENT_LABEL_OR_RESPONSE_LINK_INVALID')
                try:
                    if (_time(judgment['review']['reviewed_at']) < _time(row['recorded_at'])
                            or judgment['review']['reviewer_ref'] == row.get('agent_id')):
                        group_codes.append('JUDGMENT_REVIEW_TIMING_OR_INDEPENDENCE_INVALID')
                except (ValueError, TypeError, KeyError, AttributeError):
                    group_codes.append('JUDGMENT_REVIEW_TIMING_OR_INDEPENDENCE_INVALID')
            if not group_codes:
                group_ok.add(did)
            codes.extend(group_codes)
        if set(judgments) != set(outcomes) or any(r.get('decision_id') not in decisions for r in outcomes.values()):
            codes.append('ORPHAN_OUTCOME_OR_JUDGMENT')
        report['complete_group_count'] = len(group_ok)
        report['source_data_status'] = 'VALID' if decisions and not any('SOURCE' in c or 'OBSERVATION' in c for c in codes) else 'BLOCKED'
        report['correctness_status'] = 'REVIEWED_LINKS_VALID' if decisions and not any('CORRECTNESS' in c or 'LABEL' in c or 'RESPONSE' in c for c in codes) else 'BLOCKED'
        if not any(d['split'] == 'train' and did in group_ok for did, d in decisions.items()):
            codes.append('NO_COMPLETE_TRAIN_GROUP')
        if not codes:
            report['training_readiness'] = 'TEST_ONLY_CONTRACT_VALID' if purpose == 'TEST_ONLY' else 'FIT_INPUT_CONTRACT_READY'
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        codes.append('COLLECTION_OR_CATALOG_SCHEMA_INVALID')
    report['blocker_codes'] = sorted(set(codes))
    return report


def to_fit_rows(data, **validation_options):
    """Gate PRIVATE rows for fit_utility(rows, registry, *, weights=None,
    min_samples=2). Never call it; no partial-group salvage or false imputation.
    """
    report = validate_training_data(data, **validation_options)
    if report['training_readiness'] not in {'FIT_INPUT_CONTRACT_READY', 'TEST_ONLY_CONTRACT_VALID'}:
        return {'report': report, 'rows': None}
    judgments = {r['response_ref']: r for r in data['judgments']}
    decisions = {d['decision_id']: d for d in data['decisions']}
    rows = []
    for outcome in data['outcomes']:
        decision = decisions[outcome['decision_id']]
        if decision['split'] != 'train':
            continue
        rows.append({'patient_id': decision['patient_id'], 'decision_id': decision['decision_id'],
                     'agent_id': outcome['agent_id'], 'model': outcome['model'], 'version': outcome['version'],
                     'split': 'train', 'observation': deepcopy(decision['observation']),
                     'correct': judgments[outcome['response_ref']]['correct']})
    return {'report': report, 'rows': rows}
