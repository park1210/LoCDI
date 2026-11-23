"""TEST_ONLY numerical contracts; no clinical rates, fitting or model calls."""
from copy import deepcopy
import itertools
import math
from types import SimpleNamespace
import unittest

from app.core.diagnostic_belief import validation_utility
from app.core.information_value import InformationPolicy, select_action
from app.core.voi_planning import plan_voi
from scripts.probe_voi_stop_tolerance import probe
from test_routing_voi_readiness import test_only_voi


BELIEF = {"TEST_ONLY_A": .5, "TEST_ONLY_B": .5}
EPSILON = 1e-12  # Existing planner policy; not a new algorithm parameter.


def channel(name="TEST_ONLY_T", probability=.9):
    return {"id": name, "type": "request_test", "target": name, "outcomes": [
        {"id": "positive", "likelihood": {"TEST_ONLY_A": probability, "TEST_ONLY_B": 1-probability}},
        {"id": "negative", "likelihood": {"TEST_ONLY_A": 1-probability, "TEST_ONLY_B": probability}}]}


def interface(actions, diagnose=True):
    rows = [{"action_id": a["id"], "type": "Test", "action": "request_test",
             "target": a["target"], "outcome_model": a["id"]} for a in actions]
    if diagnose:
        rows.append({"action_id": "diagnose", "type": "Diagnose", "action": "diagnose",
                     "target": None, "outcome_model": None})
    return rows


class StopConsistencyTests(unittest.TestCase):
    def compare(self, actions, utility=None, horizon=1, diagnose=True):
        utility = utility if utility is not None else validation_utility(BELIEF)
        before = deepcopy((BELIEF, actions, utility))
        plan = plan_voi(BELIEF, actions, utility, horizon=horizon)
        selection = select_action(interface(actions, diagnose), selector="bayesian_voi", belief=BELIEF,
                                  artifact={"actions": actions, "terminal_utility": utility, "horizon": horizon})
        self.assertEqual(before, (BELIEF, actions, utility))
        selected = selection["selected_action"]
        if plan["selected_action_id"] is None:
            self.assertEqual(selected["type"], "Diagnose") if diagnose else self.assertIsNone(selected)
        else:
            self.assertEqual(selected["action_id"], plan["selected_action_id"])
            self.assertEqual(selected["q_value"], plan["action_values"][0]["q_value"])
        # Selector must retain the planner's unrounded Q, VoI and posterior audit values.
        for row in selection["values"]:
            if row["type"] != "Diagnose":
                expected = next(r for r in plan["action_values"] if r["action_id"] == row["action_id"])
                for field in ("q_value", "voi", "outcomes", "information_gain_nats"):
                    self.assertEqual(row[field], expected[field])
        return plan, selection

    def boundary(self, value):
        # Perfect information and a zero baseline make Q=value exactly, including epsilon.
        utility = {"TEST_ONLY_A": {"TEST_ONLY_A": 0., "TEST_ONLY_B": 0.},
                   "TEST_ONLY_B": {"TEST_ONLY_A": -2*value, "TEST_ONLY_B": 2*value}}
        plan, selection = self.compare([channel(probability=1.)], utility)
        self.assertEqual(plan["baseline_terminal_utility"], 0.)
        self.assertEqual(plan["action_values"][0]["voi"], value)
        return plan, selection

    def test_c4_01_actual_input_is_permanent_regression(self):
        result = probe()
        self.assertEqual(result["voi"], 1.000310945187266e-13)
        self.assertIsNone(result["planner_selected_action_id"])
        self.assertEqual(result["selector_selected_type"], "Diagnose")
        self.assertTrue(result["same_stop_decision"])

    def test_negative_floating_point_voi_stops(self):
        # Valid categorical channels sum to 1 within IEEE rounding, not a clinical model.
        a = channel()
        a["outcomes"] = [{"id": str(i), "likelihood": {d: p for d in BELIEF}}
                         for i, p in enumerate((.6, .3, .1))]
        utility = {"TEST_ONLY_A": {d: 1. for d in BELIEF}}
        plan, _ = self.compare([a], utility)
        self.assertLess(plan["action_values"][0]["voi"], 0.)
        self.assertIsNone(plan["selected_action_id"])

    def test_exact_zero_stops(self):
        plan, _ = self.boundary(0.)
        self.assertIsNone(plan["selected_action_id"])

    def test_positive_below_epsilon_stops(self):
        plan, _ = self.boundary(EPSILON / 2)
        self.assertIsNone(plan["selected_action_id"])

    def test_exact_epsilon_stops(self):
        plan, _ = self.boundary(EPSILON)
        self.assertIsNone(plan["selected_action_id"])

    def test_one_float_below_epsilon_stops(self):
        plan, _ = self.boundary(math.nextafter(EPSILON, 0.))
        self.assertIsNone(plan["selected_action_id"])

    def test_one_float_above_epsilon_continues(self):
        plan, _ = self.boundary(math.nextafter(EPSILON, math.inf))
        self.assertEqual(plan["selected_action_id"], "TEST_ONLY_T")

    def test_above_epsilon_continues(self):
        plan, _ = self.boundary(2*EPSILON)
        self.assertEqual(plan["selected_action_id"], "TEST_ONLY_T")

    def test_original_positive_boundary_all_horizons(self):
        # Use the exact original complementary literals rather than 1-p rounding.
        a = channel(probability=.5000000000001)
        a["outcomes"][0]["likelihood"]["TEST_ONLY_B"] = .4999999999999
        a["outcomes"][1]["likelihood"]["TEST_ONLY_A"] = .4999999999999
        for h in (1, 2, 3):
            with self.subTest(horizon=h):
                plan, _ = self.compare([a], horizon=h)
                self.assertEqual(plan["action_values"][0]["voi"], 1.000310945187266e-13)
                self.assertIsNone(plan["selected_action_id"])

    def test_exact_q_tie_uses_lexical_id_regardless_of_candidate_order(self):
        actions = [channel("TEST_ONLY_Z"), channel("TEST_ONLY_A")]
        for ordering in itertools.permutations(actions):
            plan, _ = self.compare(list(ordering))
            self.assertEqual(plan["selected_action_id"], "TEST_ONLY_A")

    def test_near_q_tie_uses_actual_greater_q_not_a_new_tie_tolerance(self):
        actions = [channel("TEST_ONLY_A", .9), channel("TEST_ONLY_Z", .90000000000001)]
        for ordering in itertools.permutations(actions):
            plan, _ = self.compare(list(ordering))
            self.assertEqual(plan["selected_action_id"], "TEST_ONLY_Z")
            gap = plan["action_values"][0]["q_value"] - plan["action_values"][1]["q_value"]
            self.assertGreater(gap, 0.)
            self.assertLess(gap, EPSILON)

    def test_no_candidates_does_not_synthesize_an_action(self):
        plan, selection = self.compare([], diagnose=False)
        self.assertIsNone(plan["selected_action_id"])
        self.assertIsNone(selection["selected_action"])

    def test_diagnose_only(self):
        plan, selection = self.compare([])
        self.assertEqual(plan["action_values"], [])
        self.assertEqual(selection["selected_action"]["type"], "Diagnose")

    def test_stop_without_diagnose_candidate_never_forces_information(self):
        self.compare([channel(probability=.5)], diagnose=False)

    def test_continue_without_diagnose_candidate(self):
        plan, _ = self.compare([channel()], diagnose=False)
        self.assertEqual(plan["selected_action_id"], "TEST_ONLY_T")

    def test_three_channel_bellman_horizons_1_2_3(self):
        actions = [channel("TEST_ONLY_{}".format(i)) for i in range(3)]
        for h, q in ((1, .9), (2, .9), (3, .972)):
            with self.subTest(horizon=h):
                plan, _ = self.compare(actions, horizon=h)
                self.assertAlmostEqual(plan["optimal_value"], q, places=14)
                self.assertEqual(plan["selected_action_id"], "TEST_ONLY_0")

    def test_raw_q_ranking_is_not_rewritten_to_hide_positive_voi(self):
        _, selection = self.boundary(EPSILON / 2)
        self.assertEqual(selection["values"][0]["type"], "Test")
        self.assertGreater(selection["values"][0]["voi"], 0.)
        self.assertEqual(selection["selected_action"]["type"], "Diagnose")

    def test_information_gain_keeps_distinct_objective(self):
        actions = [channel()]
        utility = {"TEST_ONLY_A": {d: 1. for d in BELIEF}}
        plan, _ = self.compare(actions, utility)
        self.assertIsNone(plan["selected_action_id"])
        selection = select_action(interface(actions), selector="information_gain", belief=BELIEF,
                                  artifact={"actions": actions, "terminal_utility": utility, "horizon": 3})
        self.assertEqual(selection["selected_action"]["action_id"], "TEST_ONLY_T")
        self.assertGreater(selection["selected_action"]["information_gain_nats"], 0.)
        self.assertEqual(selection["horizon"], 1)

    def test_information_gain_rank_and_ties_remain_original_contract(self):
        for p in (.5, .5000000000001, .9, 1.):
            actions = [channel("TEST_ONLY_Z", p), channel("TEST_ONLY_A", p)]
            selection = select_action(interface(actions), selector="information_gain", belief=BELIEF,
                                      artifact={"actions": actions})
            expected = sorted(selection["values"], key=lambda r: (
                -r["information_gain_nats"], r["type"] != "Diagnose", r["action_id"]))
            self.assertEqual(selection["values"], expected)
            self.assertEqual(selection["selected_action"], expected[0])

    def test_information_policy_propagates_tiny_voi_stop(self):
        model = test_only_voi()
        model["horizon"] = 1
        outcomes = model["actions"][0]["outcomes"]
        outcomes[0]["likelihood"] = {"TEST_ONLY_A": .5000000000001, "TEST_ONLY_B": .4999999999999}
        outcomes[1]["likelihood"] = {"TEST_ONLY_A": .4999999999999, "TEST_ONLY_B": .5000000000001}
        runtime = SimpleNamespace(case_data={}, question_history=[], voi_state={}, current_time_min=0,
                                  voi_memory_pool={"mode": "none"})
        result = InformationPolicy(model).propose(runtime, {}, actions=["request_test", "diagnose"],
            tests=["TEST_ONLY_T"], departments=[], evidence_records=[], allow_unresolved=True)
        self.assertEqual(result["action"], "diagnose")
        self.assertEqual(result["certainty"], "unresolved")
        self.assertEqual(runtime.voi_state["selector_status"]["selected_action"]["type"], "Diagnose")


if __name__ == "__main__":
    unittest.main()
