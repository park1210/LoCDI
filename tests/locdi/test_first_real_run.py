"""Read-only consistency checks on the first saved real run; no replay."""
import json
from pathlib import Path
import unittest
from app.locdi.dataset_loader import load_cases
from app.locdi.llm_setup import digest

ROOT = Path(__file__).resolve().parents[2]
RUN = 'fa86db9c632343ccb0382ac23f5dc7ad'


def keys(value):
    if isinstance(value, dict):return set(value) | set().union(*(keys(v) for v in value.values()), set())
    if isinstance(value, list):return set().union(*(keys(v) for v in value), set())
    return set()


def strings(value):
    if isinstance(value, str):return [value]
    if isinstance(value, dict):return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):return [s for v in value for s in strings(v)]
    return []


class FirstRealRunContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = json.loads((ROOT/'outputs/episodes'/f'{RUN}_real_case001.json').read_text(encoding='utf-8'))
        cls.report = json.loads((ROOT/'outputs/reports'/f'{RUN}_real_run.json').read_text(encoding='utf-8'))
        lines = (ROOT/'outputs/metrics'/f'{RUN}_real_summary.jsonl').read_text(encoding='utf-8').splitlines()
        if len(lines) != 1:raise AssertionError('Expected one summary record')
        cls.summary = json.loads(lines[0])

    def test_fingerprints_and_three_file_fields_match(self):
        ep, su, report = self.ep, self.summary, self.report
        self.assertEqual(digest(ep['configuration']), ep['configuration_fingerprint'])
        fields = ['case_ref','patient_ref','visit_ref','configuration_fingerprint','workflow','memory','capabilities','result','llm_execution']
        self.assertEqual(digest({k:ep[k] for k in fields}), ep['normalized_trace_fingerprint'])
        self.assertEqual(su['episode_count'], 1)
        row = su['episodes'][0]
        for key in ('status','diagnosis_unresolved','elapsed_simulation_minutes','action_counts'):
            self.assertEqual(row[key], ep['result'][key])
        for key in ('configuration_fingerprint','normalized_trace_fingerprint','llm_execution','run_id','case_ref'):
            self.assertEqual(row[key], ep[key])
        self.assertEqual(report['llm_execution'], ep['llm_execution'])
        self.assertEqual(report['status'], ep['result']['status'])
        self.assertEqual(ep['configuration']['llm_settings_fingerprint'], ep['llm_execution']['settings_fingerprint'])
        self.assertEqual(su['status_counts'], {'completed':1,'incomplete':0,'failed':0})
        for value in (ep, su, report):
            self.assertFalse(value['mock_execution'] or value['research_evaluation_eligible'] or value['simulation_ready'])
            self.assertEqual(value['clinical_metrics'], 'NOT_EVALUATED')

    def test_action_links_order_and_retrieval_records(self):
        ep = self.ep
        events = {e['sequence']:e for e in ep['workflow']['events']}
        self.assertEqual(list(events), list(range(1, len(events)+1)))
        self.assertEqual([e['minute'] for e in events.values()], sorted(e['minute'] for e in events.values()))
        decisions = ep['workflow']['decisions']
        self.assertEqual([d['selected_action'] for d in decisions], ['request_test','diagnose'])
        self.assertEqual([d['minute'] for d in decisions], [3,13])
        for decision, action in zip(decisions, ('order_test','er_finalize')):
            self.assertTrue(decision['execution_success'])
            self.assertEqual(decision['call_status'], 'success')
            for seq in decision['execution_event_sequences']:
                self.assertEqual(events[seq]['executed_action'], action)
                self.assertEqual(events[seq]['minute'], decision['minute'])
        ordered = events[decisions[0]['execution_event_sequences'][0]]
        received = next(e for e in events.values() if e['event_type']=='result_received')
        self.assertEqual(ordered['target_ref'], decisions[0]['target_ref'])
        self.assertEqual(received['target_ref'], ordered['target_ref'])
        self.assertEqual(received['minute'], 9)
        retrievals = ep['memory']['retrievals']
        self.assertEqual([r['minute'] for r in retrievals], [3,13])
        self.assertTrue(all(r['cutoff_compliant'] and len(r['memory_refs'])==5 for r in retrievals))
        self.assertEqual(retrievals[0]['memory_refs'], retrievals[1]['memory_refs'])

    def test_public_keys_identifiers_and_evidence_absent_without_payload_claim(self):
        documents = [self.ep, self.summary, self.report]
        forbidden = {'gold','evaluation','provenance','source_event_store','data_replay','patient_id','visit_id','case_id','task_annotations','current_encounter_events'}
        self.assertFalse(set().union(*(keys(v) for v in documents)) & forbidden)
        case = load_cases(ROOT/'datasets/raw/longitudinal_benchmark_50.jsonl')[0]
        needles = {case['case_id'],case['patient_id'],case['agent_visible']['index_encounter_id']}
        for past in case['agent_visible']['past_visits']:
            needles.add(past['visit_id']); needles.update(r['memory_id'] for r in past['records'])
        for e in case['environment']['current_encounter_events']:needles.update([e['event_id'], e['evidence']])
        needles.update(e['event_id'] for e in case['environment']['source_event_store'])
        leaves = [s for doc in documents for s in strings(doc)]
        self.assertFalse(any(needle and needle in text for needle in needles for text in leaves))
        # These files provide no raw payload/response; no reconstruction assertion.
        self.assertFalse({'user_prompt','system_prompt','raw_response'} & set().union(*(keys(v) for v in documents)))
