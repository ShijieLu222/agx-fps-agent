"""Contract smoke tests, not a simulator or a measure of playing strength."""
import copy
import gc
import json
import math
import unittest

import agent


MAP = {
    "sites": [
        {"center": [10.0, 0.0, 0.0], "radius": 4.0},
        {"center": [0.0, 10.0, 0.0], "radius": 4.0},
    ],
    "nav": {
        "nodes": [
            {"id": 0, "position": [0.0, 0.0, 0.0]},
            {"id": 1, "position": [10.0, 0.0, 0.0]},
            {"id": 2, "position": [0.0, 10.0, 0.0]},
        ],
        "edges": [
            {"from": 0, "to": 1, "kind": "walk"},
            {"from": 0, "to": 2, "kind": "walk"},
        ],
    },
}


def observation(**changes):
    row = {
        "map_id": "test-map", "static_schema_hash": "test-v1",
        "seat": 0, "team": 0, "side": "attack", "phase": "live",
        "match_tick": 600, "round_tick": 600, "round_index": 0,
        "visible": [], "sounds": [],
        "bomb": {"state": "unknown", "fuse_end_tick": None},
        "me": {
            "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0,
            "alive": True, "armor": 0, "cash": 800, "has_kit": False,
            "grounded": True, "crouch": False, "carrying_bomb": False,
            "blind_ticks": 0,
            "weapons": [None, {"id": "pistol", "ammo": 20,
                               "reserve": 60, "reload_until": 0}, None],
        },
    }
    row.update(changes)
    return row


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.gc_enabled = gc.isenabled()
        agent.init({})

    def tearDown(self):
        if self.gc_enabled:
            gc.enable()

    def run_action(self, row):
        action = agent.act(row)
        # JSON wire safety and the subset of the current action schema we emit.
        json.dumps(action, allow_nan=False)
        self.assertIsInstance(action, dict)
        analog = {"move_forward", "move_right", "yaw_rate", "pitch_rate"}
        boolean = {"jump", "crouch", "walk", "fire", "reload"}
        allowed = analog | boolean | {"weapon_slot", "buy", "use", "pickup_id"}
        self.assertFalse(set(action) - allowed)
        for key, value in action.items():
            if key in analog:
                self.assertIn(type(value), (int, float))
                self.assertTrue(math.isfinite(value))
                self.assertLessEqual(abs(value), 1.0)
            elif key in boolean:
                self.assertIs(type(value), bool)
            elif key == "weapon_slot":
                self.assertIs(type(value), int)
                self.assertIn(value, (0, 1, 2))
            elif key == "buy":
                self.assertIn(value, ("ak47", "m4a1", "armor", "defusekit"))
            elif key == "use":
                self.assertIn(value, ("plant", "defuse", "pickup"))
            elif key == "pickup_id":
                self.assertIsInstance(value, str)
                self.assertTrue(0 < len(value) <= 128)
        return action

    def test_first_map_is_cached_when_later_observations_omit_it(self):
        first = observation(phase="buy", match_tick=0, round_tick=0, map=copy.deepcopy(MAP))
        self.run_action(first)
        self.assertEqual(agent._player.nodes, {})  # Graph work is deferred.
        self.run_action(observation(phase="buy", match_tick=3, round_tick=3))
        self.assertEqual(len(agent._player.nodes), 3)
        self.assertIn("move_forward", self.run_action(observation()))

    def test_missing_and_empty_static_data_do_not_crash(self):
        for fields in ({}, {"map": {}}, {"map": {"sites": [], "nav": {}}}):
            with self.subTest(fields=fields):
                agent.init({})
                self.assertEqual(self.run_action(observation(**fields)), {"weapon_slot": 1})

    def test_changed_map_identity_discards_stale_routes(self):
        self.run_action(observation(map=copy.deepcopy(MAP)))
        agent._player.escape_until = 9999
        self.run_action(observation(static_schema_hash="test-v2"))
        self.assertEqual(agent._player.nodes, {})
        self.assertEqual(agent._player.path, [])
        self.assertEqual(agent._player.escape_until, 0)

    def test_init_discards_all_previous_match_state(self):
        self.run_action(observation(map=copy.deepcopy(MAP)))
        agent.init({})
        self.assertIsNone(agent._player.map_identity)
        self.assertEqual(agent._player.sites, [])
        self.assertEqual(self.run_action(observation()), {"weapon_slot": 1})

    def test_buy_uses_side_and_current_inventory(self):
        for side, cash, kit, armor, expected in (
            ("attack", 2700, False, 0, "ak47"),
            ("defend", 400, False, 0, "defusekit"),
            ("defend", 2900, True, 0, "m4a1"),
            ("attack", 800, False, 0, "armor"),
            ("attack", 0, False, 100, None),
        ):
            with self.subTest(side=side, cash=cash):
                row = observation(phase="buy", side=side)
                row["me"].update(cash=cash, has_kit=kit, armor=armor)
                action = self.run_action(row)
                self.assertEqual(action, {"buy": expected} if expected else {})

    def test_null_weapons_and_visible_enemy_never_fire(self):
        row = observation(map=copy.deepcopy(MAP), visible=[{
            "x": 8.0, "y": -2.0, "z": 0.0,
            "crouch": False, "parts": ["chest"],
        }])
        row["me"]["weapons"] = [None, None, None]
        self.assertIs(self.run_action(row)["fire"], False)

    def test_empty_magazine_requests_reload(self):
        row = observation(map=copy.deepcopy(MAP))
        row["me"]["weapons"][1]["ammo"] = 0
        self.assertIs(self.run_action(row)["reload"], True)

    def test_plant_and_defuse_do_not_request_conflicting_reload(self):
        for side, use in (("attack", "plant"), ("defend", "defuse")):
            with self.subTest(side=side):
                row = observation(side=side, map=copy.deepcopy(MAP))
                row["me"].update(x=10.0, carrying_bomb=(side == "attack"))
                row["me"]["weapons"][1]["ammo"] = 0
                if side == "defend":
                    row["bomb"] = {"state": "planted", "position": [10.0, 0.0, 0.0],
                                   "fuse_end_tick": 3000}
                action = self.run_action(row)
                self.assertEqual(action["use"], use)
                self.assertNotIn("reload", action)
                self.assertNotIn("fire", action)
                self.assertNotIn("move_forward", action)

    def test_hidden_bomb_position_is_not_required(self):
        row = observation(side="defend", map=copy.deepcopy(MAP),
                          bomb={"state": "planted", "fuse_end_tick": 3000})
        self.assertNotIn("use", self.run_action(row))

    def test_dead_post_and_finished_are_neutral(self):
        for phase, alive in (("post", True), ("finished", True), ("live", False)):
            with self.subTest(phase=phase, alive=alive):
                row = observation(phase=phase)
                row["me"]["alive"] = alive
                self.assertEqual(self.run_action(row), {})

    def test_round_change_clears_escape_without_discarding_map(self):
        self.run_action(observation(map=copy.deepcopy(MAP)))
        agent._player.escape_until = 99999
        agent._player.escapes = 5
        self.run_action(observation(phase="buy", round_index=12, round_tick=0,
                                   match_tick=10000, side="defend"))
        self.assertEqual(agent._player.escape_until, 0)
        self.assertEqual(agent._player.escapes, 0)
        self.assertEqual(len(agent._player.sites), 2)


if __name__ == "__main__":
    unittest.main()
