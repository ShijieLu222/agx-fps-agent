"""Regression fixtures for observed connectors and bounded route preferences.

The closed-loop wall fixture is kinematic, not a replica of AGX collision or
movement physics. It checks that feedback can escape a failed initial leg.
"""
import copy
import math
import unittest

import agent
from test_agent import observation
from test_v3_navigation import fork_map


def connector_map(kind="walk"):
    positions = {"safe": [0., 0., 0.], "blocked": [10., 0., 0.],
                 "goal": [20., 0., 0.], "around": [0., 8., 0.],
                 "far": [20., 8., 0.], "isolated": [8., .05, 0.]}
    links = [("safe", "blocked", kind), ("blocked", "goal", "walk"),
             ("safe", "around", "walk"), ("around", "far", "walk"),
             ("far", "goal", "walk")]
    return {"sites": [{"center": positions["goal"], "radius": 4.}],
            "nav": {"nodes": [{"id": key, "position": pos}
                              for key, pos in positions.items()],
                    "edges": [{"from": a, "to": b, "kind": kind}
                              for a, b, kind in links]}}


class InitialConnectorTests(unittest.TestCase):
    def prepared(self, pos=(8., .1, 0.), kind="walk"):
        p = agent.Player()
        p.load_map(connector_map(kind))
        p.path_nodes = ["blocked", "goal", None]
        p.path = [p.nodes["blocked"], p.nodes["goal"], p.nodes["goal"]]
        p.last_position = list(pos)
        p.nav_tick = 600
        self.assertIsNone(p.nav_from)
        return p

    def test_failed_initial_connector_returns_on_declared_walk_corridor(self):
        p = self.prepared()
        p.penalize_leg()
        self.assertEqual(p.retry_node, "safe")
        self.assertIn(("safe", "blocked"), p.failed_edges)
        path = p.route([8., .1, 0.], [20., 0., 0.])
        self.assertEqual(path[0], [0., 0., 0.])
        self.assertNotIn([10., 0., 0.], path)
        self.assertNotIn([8., .05, 0.], path)
        self.assertIn([0., 8., 0.], path)

    def test_unconnected_nearby_point_and_wrong_height_cannot_authorize_recovery(self):
        for pos in ((8., .75, 0.), (8., 0., .75), (12., 1., 0.)):
            with self.subTest(pos=pos):
                p = self.prepared(pos)
                p.penalize_leg()
                self.assertIsNone(p.retry_node)
                self.assertFalse(p.failed_edges)

    def test_initial_drop_is_not_reversed_as_a_walk(self):
        p = self.prepared(kind="drop")
        p.penalize_leg()
        self.assertIsNone(p.retry_node)
        self.assertFalse(p.failed_edges)

    def test_failed_initial_connector_can_escape_in_closed_loop(self):
        mp = connector_map()
        # The isolated node is only useful for the negative snapping test above.
        mp["nav"]["nodes"] = [n for n in mp["nav"]["nodes"]
                                if n["id"] != "isolated"]
        p = agent.Player()
        p.load_map(mp)
        row = observation(side="defend")
        m = row["me"]
        m.update(x=8., y=0., z=0., yaw=0.)
        contacts = 0
        for tick in range(600, 4800, 3):
            row["match_tick"] = tick
            action = p.move(row, [20., 0., 0.], {})
            cosine, sine = math.cos(m["yaw"]), math.sin(m["yaw"])
            forward, right = action["move_forward"], action["move_right"]
            x = m["x"] + .3 * (forward * cosine + right * sine)
            y = m["y"] + .3 * (-forward * sine + right * cosine)
            if 8.8 < x < 11.2 and -1.5 < y < 1.5:
                contacts += 1
            else:
                m["x"], m["y"] = x, y
            m["yaw"] += action["yaw_rate"] * .15
            if math.hypot(m["x"] - 20., m["y"]) < 1.:
                break
        self.assertGreater(contacts, 0)
        self.assertGreater(p.escapes, 0)
        self.assertLess(math.hypot(m["x"] - 20., m["y"]), 1.)


class RouteThreatTests(unittest.TestCase):
    def setUp(self):
        self.p = agent.Player()
        self.p.load_map(fork_map())
        self.row = observation(side="defend", round_index=0)
        self.goal = [20., 0., 0.]

    def add_threat(self, repetitions=1):
        for _ in range(repetitions):
            self.p.nav_from = 0
            self.p.path_nodes = ["short", "goal", None]
            self.p.remember_threat(self.row)

    def route(self, side="defend", round_index=0):
        self.p.nav_side, self.p.nav_round = side, round_index
        return self.p.route([0., 0., 0.], self.goal)

    def test_observed_threat_prefers_a_reasonable_connected_alternative(self):
        self.assertEqual(self.route()[1], [10., 0., 0.])
        self.add_threat()
        self.assertEqual(self.route()[1], [0., 8., 0.])
        self.assertIn(("short", 10.), self.p.edges[0])

    def test_preference_survives_navigation_reset_but_not_a_side_change_or_expiry(self):
        self.add_threat()
        self.p.reset_navigation()
        self.assertEqual(self.route(round_index=1)[1], [0., 8., 0.])
        self.assertEqual(self.route(side="attack", round_index=1)[1], [10., 0., 0.])
        self.assertEqual(self.route(round_index=5)[1], [10., 0., 0.])
        self.assertFalse(self.p.route_threats)

    def test_repeated_damage_does_not_force_arbitrarily_long_detours(self):
        self.add_threat(100)
        self.assertLessEqual(max(v[1] for v in self.p.route_threats.values()), 120.)
        # The recorded Nevaio case has a 78m direct vs 258m alternate route.
        # An expensive alternative should remain optional, even after repeats.
        for key in self.p.edges:
            self.p.edges[key] = [(dst, cost + (500 if key == "around" else 0))
                                 for dst, cost in self.p.edges[key]]
        self.assertEqual(self.route()[1], [10., 0., 0.])

    def test_new_map_discards_old_preferences(self):
        self.add_threat()
        self.p.load_map(copy.deepcopy(fork_map()))
        self.assertFalse(self.p.route_threats)
        self.assertEqual(self.route()[1], [10., 0., 0.])

    def test_missing_or_nonwalk_leg_does_not_invent_a_threat_edge(self):
        self.p.path_nodes = ["short"]
        self.p.nav_from = None
        self.p.remember_threat(self.row)
        self.p.nav_from = "goal"  # No goal->0 edge in the public graph.
        self.p.path_nodes = [0]
        self.p.remember_threat(self.row)
        self.assertFalse(self.p.route_threats)


if __name__ == "__main__":
    unittest.main()
