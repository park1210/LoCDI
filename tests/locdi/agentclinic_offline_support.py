"""TEST_ONLY nonclinical scripted doctor, never a real-model fallback."""
from copy import deepcopy
from functools import lru_cache
from app.locdi.llm_setup import ROOT
from app.locdi.agentclinic_scenario_adapter import load_scenarios, action_catalog, adapt_test_record
from app.locdi.agentclinic_turn_runtime import OfflineDoctorClient

SOURCE = ROOT/'outputs/private/research_evaluation/c3_10_v1/upstream_snapshot/agentclinic_medqa.jsonl'


@lru_cache(maxsize=1)
def bundle():
    return load_scenarios(SOURCE)


def fixture():
    return {'OSCE_Examination': {
        'Objective_for_Doctor': 'TEST_ONLY_OBJECTIVE',
        'Patient_Actor': {'Demographics': 'TEST_ONLY_ACTOR', 'History': 'TEST_ONLY_HISTORY',
                          'Symptoms': {'Primary_Symptom': 'TEST_ONLY_SYMPTOM', 'Secondary_Symptoms': []},
                          'Past_Medical_History': ['TEST_ONLY_PAST'], 'Social_History': {'value': 'TEST_ONLY_SOCIAL'},
                          'Review_of_Systems': 'TEST_ONLY_REVIEW'},
        'Physical_Examination_Findings': {'Exam': {'value': 'TEST_ONLY_EXAM'}},
        'Test_Results': {'Group': {'A': 'TEST_ONLY_A', 'B': 'TEST_ONLY_B'}},
        'Correct_Diagnosis': 'TEST_ONLY_PRIVATE_KEY'}}


class ScriptedDoctor(OfflineDoctorClient):
    def __init__(self, actions):
        self.actions = deepcopy(actions)
        self.seen = []

    def choose_action(self, observation):
        self.seen.append(deepcopy(observation))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def e2e_actions(scenario):
    catalog = action_catalog(scenario)
    return [{'action': 'ASK', 'target': 'history'},
            {'action': 'EXAMINE', 'target': catalog['EXAMINE'][0]['target']},
            {'action': 'TEST', 'target': catalog['TEST'][0]['target'] if catalog['TEST'] else 'missing'},
            {'action': 'DIAGNOSE', 'predicted_diagnosis': 'TEST_ONLY_FINAL_NOT_A_CLINICAL_JUDGMENT'}]
