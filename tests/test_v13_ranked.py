"""Game-observation regressions for the ranked v13 changes."""
import math
import unittest

import agent
from test_agent import observation
from test_v2 import advance, contact, enemy


def route_row():
    row = observation(map_id="mirage-cs2-v54", match_tick=1002, round_tick=1002,
                      map={"spawns": {"T": [{"position": [0, 0, 0]}]},
                           "sites": [{"center": [22., 0., 0.], "radius": 3.}]},
                      bomb={"state": "held", "carrier": 0})
    row["me"].update(active_slot=0, carrying_bomb=True, armor=100, helmet=True, cash=1200,
                     throwable_kinds=[], throw_ready=0)
    row["me"]["weapons"][0] = {"id": "ak47", "ammo": 30, "reserve": 90}
    return row


class RankedTests(unittest.TestCase):
    def test_departure_keeps_normal_walk_after_window_and_after_contact(self):
        row, p = route_row(), agent.Player()
        row["me"]["carrying_bomb"] = False
        p.load_map(row["map"])
        p.path = [[3., 0., 0.], [3., 8., 0.]]
        for rt, seen, expect_walk in [(600, None, False), (1100, None, True), (600, 590, True)]:
            row["round_tick"], p.last_visible_tick = rt, seen
            a = p.prepare_approach(row, {"move_forward": .75, "walk": False})
            self.assertEqual(a["walk"], expect_walk)

    def test_departure_full_input_is_limited_to_level_unobstructed_legs(self):
        row, p = route_row(), agent.Player()
        p.load_map(row["map"])
        row["me"]["carrying_bomb"] = False
        row["round_tick"] = 660
        for waypoint, stuck, magnitude in [([8, 0, 0], 0, 1.), ([3, 0, 0], 0, .75),
                                            ([8, 0, 1], 0, .75), ([8, 0, 0], 4, .75)]:
            p.path, p.stuck = [waypoint], stuck
            a = p.prepare_approach(row, {"move_forward": .45, "move_right": .6})
            self.assertAlmostEqual(math.hypot(a["move_forward"], a["move_right"]), magnitude)
            self.assertAlmostEqual(a["move_forward"]/a["move_right"], .75)

    def test_carrier_moves_upright_after_losing_visible_target(self):
        row, p = route_row(), agent.Player()
        row["visible"] = [enemy(x=1, y=8)]
        self.assertEqual(p.act(row)["move_forward"], 0.)
        row.pop("map")
        row["visible"] = []
        advance(row)
        a = p.act(row)
        self.assertGreater(a["move_forward"], .5)
        self.assertFalse(a.get("crouch"))
        self.assertAlmostEqual(a["yaw_rate"], 0.)

    def test_mirage_buy_prime_release_once_with_inventory_feedback(self):
        row, p = route_row(), agent.Player()
        row["phase"] = "buy"
        self.assertEqual(p.act(row), {"buy": "smokegrenade"})
        row.pop("map")
        advance(row)
        self.assertEqual(p.act(row), {})
        row["me"]["throwable_kinds"] = ["smokegrenade"]
        row["phase"] = "live"
        for state in ["prime", "release", None]:
            advance(row)
            a = p.act(row)
            self.assertEqual(a.get("throw_state"), state)
            self.assertGreater(a["move_forward"], .5)

    def test_pending_utility_releases_priority_to_contact_or_plant(self):
        for interrupt in ("seen", "plant"):
            row, p = route_row(), agent.Player()
            row["me"]["throwable_kinds"] = ["smokegrenade"]
            self.assertEqual(p.act(row).get("throw_state"), "prime")
            row.pop("map")
            advance(row)
            if interrupt == "plant":
                row["me"]["x"] = 22.
            else:
                row["visible"] = [enemy()]
            a = p.act(row)
            self.assertEqual(a.get("throw_state"), "cancel")
            if interrupt == "plant":
                self.assertEqual(a.get("use"), "plant")
            else:
                self.assertIn("fire", a)

    def test_interrupted_smoke_can_be_retried_after_contact_clears(self):
        row, p = route_row(), agent.Player()
        row["me"]["throwable_kinds"] = ["smokegrenade"]
        self.assertEqual(p.act(row).get("throw_state"), "prime")
        row.pop("map")
        row["visible"] = [enemy()]
        self.assertEqual(p.act(advance(row)).get("throw_state"), "cancel")
        row["visible"] = []
        self.assertNotIn("throw_state", p.act(advance(row)))
        self.assertEqual(p.act(advance(row, 183)).get("throw_state"), "prime")
        self.assertEqual(p.act(advance(row)).get("throw_state"), "release")
        self.assertNotIn("throw_state", p.act(advance(row, 60)))

    def test_late_arrival_can_use_smoke_but_last_seconds_preserve_objective(self):
        row, p = route_row(), agent.Player()
        row.update(round_tick=4800, match_tick=4800, phase_remaining_ticks=1600)
        row["me"]["throwable_kinds"] = ["smokegrenade"]
        self.assertEqual(p.act(row).get("throw_state"), "prime")
        row.pop("map")
        row["me"]["throwable_kinds"] = []
        advance(row)
        self.assertEqual(p.act(row).get("throw_state"), "release")
        row, p = route_row(), agent.Player()
        row.update(phase_remaining_ticks=450)
        row["me"]["throwable_kinds"] = ["smokegrenade"]
        self.assertNotIn("throw_state", p.act(row))

    def test_smoke_limits_and_essential_purchases(self):
        for side, level, cash, armor, expected in [
                ("attack", "mirage-cs2-v54", 900, 0, "armor"),
                ("attack", "mirage-cs2-v54", 299, 100, None),
                ("attack", "mirage-cs2-v54", 300, 100, "smokegrenade"),
                ("defend", "mirage-cs2-v54", 1500, 100, "defusekit"),
                ("attack", "other", 1500, 100, None)]:
            row = route_row()
            row.update(phase="buy", side=side, map_id=level)
            row["me"].update(cash=cash, armor=armor)
            self.assertEqual(agent.Player().act(row).get("buy"), expected)

    def test_followup_hitbox_requires_observed_shot_on_same_track(self):
        for slot, dist in [(0, 36.), (1, 18.)]:
            row, p = contact(slot=slot), agent.Player()
            target = row["visible"][0]
            target.update(x=dist, parts=["head", "chest"])
            m = row["me"]
            m["pitch"] = math.atan2(1.18-1.62, dist)
            gun = m["weapons"][slot]
            for _ in range(6):
                a = p.act(row)
                self.assertNotIn(target["seat"], p.head_targets)
                if _ == 0:
                    self.assertTrue(a["fire"])
                row.pop("map", None)
                advance(row)
            gun["last_shot"] = row["match_tick"]-2
            p.act(row)
            self.assertIn(target["seat"], p.head_targets)
            advance(row)
            target["seat"] += 1
            p.act(row)
            self.assertNotIn(target["seat"], p.head_targets)
