"""Regressions for failures observed in the first v2 match, not win estimates."""
import copy
import gc
import math
import unittest

import agent
from test_agent import MAP, observation
from test_v2 import advance, enemy


class V3TacticsTests(unittest.TestCase):
    def setUp(self):
        self.gc_enabled = gc.isenabled()
        agent.init({})

    def tearDown(self):
        if self.gc_enabled:
            gc.enable()

    def row(self, enemy_distance=115.27):
        row = observation(map=copy.deepcopy(MAP), seat=5, side="attack",
                          visible=[enemy(0, enemy_distance, 0.0)])
        row["me"].update(active_slot=1, vx=0., vy=0., carrying_bomb=True,
                         pitch=math.atan2(1.18-1.62, enemy_distance))
        return row

    def assert_moving_without_fire(self, action):
        self.assertFalse(action.get("fire", False))
        self.assertGreater(math.hypot(action.get("move_forward", 0),
                                      action.get("move_right", 0)), .1)

    def test_carrier_does_not_stop_for_visible_enemy_beyond_pistol_range(self):
        row = self.row()
        for _ in range(12):
            self.assert_moving_without_fire(agent.act(row))
            advance(row)

    def test_exhausted_carrier_keeps_moving_even_with_near_visible_enemy(self):
        row = self.row(15.)
        row["me"]["weapons"][1].update(ammo=0, reserve=0)
        for _ in range(20):
            self.assert_moving_without_fire(agent.act(row))
            advance(row)

    def test_empty_reloadable_gun_reloads_while_advancing(self):
        row = self.row(15.)
        row["me"]["weapons"][1].update(ammo=0, reserve=20)
        action = agent.act(row)
        self.assertTrue(action.get("reload"))
        self.assert_moving_without_fire(action)

    def test_same_distant_target_is_valid_for_a_rifle(self):
        row = self.row()
        row["me"].update(active_slot=0, carrying_bomb=False)
        row["me"]["weapons"][0] = {"id": "ak47", "ammo": 30, "reserve": 90}
        self.assertTrue(agent.act(row)["fire"])

    def test_out_of_range_held_target_cannot_override_in_range_alternative(self):
        row = self.row(50.)
        row["me"]["carrying_bomb"] = False
        agent.act(row)
        advance(row)
        row["visible"] = [enemy(0, 115., 0.), enemy(1, 100., 1.)]
        agent.act(row)
        self.assertEqual(agent._player.target_seat, 1)

    def test_carrier_distant_duel_has_bounded_dwell_and_committed_advance(self):
        row = self.row(80.)
        self.assertTrue(agent.act(row)["fire"])
        self.assert_moving_without_fire(agent.act(advance(row, 120)))
        self.assert_moving_without_fire(agent.act(advance(row, 60)))
        # A close threat can interrupt the planned advance for self-defence.
        row["visible"] = [enemy(0, 10., 0.)]
        row["me"]["pitch"] = math.atan2(1.18-1.62, 10.)
        self.assertTrue(agent.act(advance(row))["fire"])

    def test_deadline_preempts_distant_duel_without_waiting_two_seconds(self):
        row = self.row(80.)
        row["phase_remaining_ticks"] = 600
        self.assert_moving_without_fire(agent.act(row))

    def test_defender_searches_nearest_public_site_then_other_site(self):
        row = self.row()
        row.update(side="defend", visible=[], bomb={"state": "planted",
                   "fuse_end_tick": 3000})
        row["me"].update(x=0., y=9., carrying_bomb=False)
        agent.act(row)
        self.assertEqual(agent._player.goal, [0., 10., 0.])
        agent.act(advance(row))  # Reach the search site; no location revealed.
        agent.act(advance(row, 90))
        self.assertEqual(agent._player.goal, [10., 0., 0.])
        self.assertIsNone(agent._player.known_bomb)

    def test_legal_bomb_position_overrides_search_and_preserves_defuse(self):
        row = self.row()
        row.update(side="defend", visible=[], bomb={"state": "planted",
                   "fuse_end_tick": 3000})
        row["me"]["carrying_bomb"] = False
        agent.act(row)
        row["bomb"]["position"] = [1., 0., 0.]
        self.assertEqual(agent.act(advance(row)).get("use"), "defuse")
        row["bomb"].pop("position")
        self.assertEqual(agent.act(advance(row)).get("use"), "defuse")

    def test_unknown_weapon_does_not_invent_a_projectile_range(self):
        row = self.row(10.)
        row["me"]["weapons"][1]["id"] = "future-weapon"
        self.assert_moving_without_fire(agent.act(row))


if __name__ == "__main__":
    unittest.main()
