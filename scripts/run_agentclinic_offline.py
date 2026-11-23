"""Pinned AgentClinic development execution. Offline Fake Client ONLY, no live switch."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.locdi.agentclinic_scenario_adapter import load_scenarios, action_catalog
from app.locdi.agentclinic_scenario_schema import REVISION, SOURCE_SHA256, LOCAL_SHA256, GROUP
from app.locdi.agentclinic_turn_runtime import OfflineDoctorClient, TurnOptions, run_scenario
from app.locdi.agentclinic_episode_exporter import export_episode
from app.locdi.episode_exporter import _write
from app.locdi.llm_setup import digest
from app.locdi.dataset_freeze import sha

DEFAULT_SOURCE = ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'
LICENSE = DEFAULT_SOURCE.parent/'LICENSE.txt'


class SourceLookupFakeDoctor(OfflineDoctorClient):
    """Predetermined actions; no scoring-key input or clinical inference."""
    def __init__(self, actions):
        self.actions = deepcopy(actions)

    def choose_action(self, observation):
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def actions_for(scenario):
    catalog = action_catalog(scenario)
    return [{'action': 'ASK', 'target': 'history'},
            {'action': 'EXAMINE', 'target': catalog['EXAMINE'][0]['target']},
            {'action': 'TEST', 'target': catalog['TEST'][0]['target'] if catalog['TEST'] else 'missing'},
            {'action': 'DIAGNOSE', 'predicted_diagnosis': 'TEST_ONLY_FINAL_NOT_A_CLINICAL_JUDGMENT'}]


def execute_batch(public_root, private_root, *, source=DEFAULT_SOURCE):
    """Uses new files only; callers may precreate audit namespace directories."""
    bundle = load_scenarios(source)
    if sha(LICENSE) != '1585a9d786b35544fd18c9632582ff9258691fb7a8b317071c8f7691740cdec4':
        raise ValueError('PINNED_MIT_NOTICE_HASH')
    with (Path(public_root)/'MIT_NOTICE.txt').open('x', encoding='utf8') as f:
        f.write(LICENSE.read_text(encoding='utf8'))
    manifest = {'schema_version': 'agentclinic.external_manifest.v1',
                'evaluation_group': GROUP, 'repository': 'SamuelSchmidgall/AgentClinic',
                'revision': REVISION, 'original_sha256': SOURCE_SHA256, 'local_sha256': LOCAL_SHA256,
                'source_records': 107, 'license': 'MIT_AUTHOR_DECLARED_AND_REPOSITORY_VERIFIED',
                'license_notice': 'MIT_NOTICE.txt', 'internal_offline_execution': 'AUTHORIZED_C3_11_TASK',
                'live_evaluation_approved': False, 'independent_clinical_reference_approved': False,
                'development_exposed_ordinals': list(range(1, 108)),
                'independent_test_split': 'NOT_ESTABLISHED', 'model_pretraining_contamination': 'UNKNOWN',
                'mode': 'OFFLINE_FAKE_CLIENT', 'fixture_policy': 'FIRST_SOURCE_BRANCH_E2E_V1',
                'cross_case_memory': False, 'clinical_time_and_resources': 'NOT_SUPPORTED'}
    _write(manifest, 'reports/source_manifest.json', public_root, 'reports')
    _write({'keys': [dict(scenario_ref=k.scenario_id, record_fingerprint=k.source_record_fingerprint,
                         correct_diagnosis=k.correct_diagnosis) for k in bundle.scoring_keys],
            'scoring_performed': False}, 'reports/scoring_keys.json', private_root, 'reports')
    rows = []
    for ordinal, scenario in enumerate(bundle.scenarios, 1):
        episode = run_scenario(scenario, SourceLookupFakeDoctor(actions_for(scenario)))
        public = export_episode(episode, public_root=public_root, private_root=private_root,
                                name='Scenario%03d' % ordinal)
        rows.append({'ordinal': ordinal, 'scenario_ref': scenario.scenario_id, **public['result']})
    # Diagnostic executions supplement, never replace, the fixed 107-case run.
    s = bundle.scenarios[0]
    probes = [
        ('Unresolved', [{'action': 'DIAGNOSE', 'predicted_diagnosis': '', 'unresolved_reason': 'TEST_ONLY_ABSTAIN'}], TurnOptions()),
        ('Incomplete', [{'action': 'ASK', 'target': 'history'}]*3, TurnOptions(3, 1)),
        ('Failed', [{'action': 'ASK', 'target': 'history'}, TimeoutError('TEST_ONLY_FAILURE')], TurnOptions())]
    probe_rows = []
    for name, plan, options in probes:
        episode = run_scenario(s, SourceLookupFakeDoctor(plan), options=options)
        public = export_episode(episode, public_root=public_root, private_root=private_root,
                                name='Diagnostic'+name)
        probe_rows.append({'diagnostic': name, **public['result']})
    summary = {'schema_version': 'agentclinic.external_summary.v1',
               'evaluation_group': GROUP, 'mode': 'OFFLINE_FAKE_CLIENT',
               'adapter_scenarios': len(bundle.scenarios), 'main_case_count': len(rows),
               'main_status_counts': dict(Counter(r['status'] for r in rows)),
               'test_request_executed_cases': sum(r['invalid_action_count'] == 0 for r in rows),
               'test_unavailable_case_ordinals': [r['ordinal'] for r in rows if r['invalid_action_count']],
               'rows': rows, 'diagnostic_executions': probe_rows,
               'diagnostic_status_counts': dict(Counter(r['status'] for r in probe_rows)),
               'clinical_metrics': 'NOT_EVALUATED', 'grading_calls': 0, 'real_generation_calls': 0,
               'research_evaluation_ready': False, 'source_manifest_fingerprint': digest(manifest)}
    _write(summary, 'metrics/summary.json', public_root, 'metrics')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', required=True, help='NEW immutable offline output namespace')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', args.namespace):
        parser.error('INVALID_NAMESPACE')
    public = ROOT/'outputs/research_evaluation'/args.namespace
    private = ROOT/'outputs/private/research_evaluation'/args.namespace
    if public.exists() or private.exists():
        parser.error('NAMESPACE_ALREADY_EXISTS')
    public.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    def offline(event, values):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind'):
            raise RuntimeError('AGENTCLINIC_OFFLINE')
    sys.addaudithook(offline)
    result = execute_batch(public, private, source=args.source)
    print(result['main_status_counts'])


if __name__ == '__main__':
    main()
