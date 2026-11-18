"""Closed-loop AGX fixtures for the two recent Mirage losses."""
import math
import unittest

import agent
from test_agent import observation


def ascent_map(rise=.8, kind="source-adjacency"):
    points = [[0., 0., 0.], [4., 0., rise], [9., 0., rise], [0., 5., 0.]]
    return {"sites": [{"center": points[2], "radius": 1.}],
            "nav": {"nodes": [{"id": i, "position": pos} for i, pos in enumerate(points)],
                    "edges": [{"from": a, "to": b, "kind": kind}
                              for a, b in [(0, 1), (1, 2), (0, 3), (3, 2)]]}}


class BlockedAscentTests(unittest.TestCase):
    def test_carrier_crosses_early_ledge_and_starts_plant(self):
        row = observation(map=ascent_map())
        m = row["me"]
        m["carrying_bomb"] = True
        p = agent.Player()
        for tick in range(600, 1200, 3):
            row["match_tick"] = tick
            action = p.act(row)
            row.pop("map", None)
            if action.get("use") == "plant":
                break
            x = m["x"] + .25 * action.get("move_forward", 0.)
            if m["x"] < 1.1 <= x:
                if not action.get("jump"):
                    continue
                m["z"] = .8
            m["x"] = x
        self.assertEqual(action.get("use"), "plant")
        self.assertEqual(p.jump_attempts, {1: 1})
        self.assertEqual(p.escapes, 0)

    def test_no_early_jump_on_flat_high_or_uncertified_leg(self):
        for rise, kind in [(0., "source-adjacency"), (2., "source-adjacency"), (.8, "drop")]:
            with self.subTest(rise=rise, kind=kind):
                p = agent.Player()
                p.load_map(ascent_map(rise, kind))
                row = observation()
                for tick in range(600, 657, 3):
                    row["match_tick"] = tick
                    self.assertFalse(p.move(row, p.nodes[1], {}).get("jump"))

    def test_longer_range_jump_is_limited_to_one_attempt_per_node(self):
        p, row = agent.Player(), observation()
        p.load_map(ascent_map())
        jumps = 0
        for tick in range(600, 900, 3):
            row["match_tick"] = tick
            jumps += bool(p.move(row, p.nodes[1], {}).get("jump"))
        self.assertEqual(jumps, 1)


class BombSearchTests(unittest.TestCase):
    def row(self):
        sites = [{"id": key, "center": [x, 0., 0.], "radius": 5.}
                 for key, x in [("A", 0.), ("B", 70.)]]
        return observation(side="defend", map={"sites": sites},
                           bomb={"state": "planted", "fuse_end_tick": 3000})

    def test_far_beep_redirects_search_before_reaching_wrong_site(self):
        p, row = agent.Player(), self.row()
        p.act(row)
        self.assertEqual(p.search_site, 0)
        row.pop("map")
        row["me"]["x"] = 7.
        row["match_tick"] += 3
        row["sounds"] = [{"type": "bomb_beep", "distance_band": "far"}]
        p.act(row)
        self.assertEqual(p.goal, [70., 0., 0.])
        self.assertIsNone(p.known_bomb)
        row.update(sounds=[], match_tick=row["match_tick"] + 3)
        p.act(row)
        self.assertEqual(p.search_site, 1)
        row["bomb"].update(site="A", position=[7.1, 0., 0.])
        self.assertEqual(p.act(row).get("use"), "defuse")

    def test_sound_bounds_and_conflicting_or_irrelevant_cues(self):
        p, row = agent.Player(), self.row()
        p.load_map(row["map"])
        def cue(band, kind="bomb_beep_fast"):
            return {"type": kind, "distance_band": band}
        for sounds, x, expected in [([cue("near")], 70., 1), ([cue("mid")], 15., 0),
                                    ([cue("far")], 13., None), ([cue("near")], 35., None),
                                    ([cue("far", "footstep")], 0., None),
                                    ([cue("unknown")], 0., None),
                                    ([cue("near"), cue("far")], 0., None)]:
            with self.subTest(sounds=sounds, x=x):
                self.assertEqual(p.audible_bomb_site(sounds, [x, 0., 0.]), expected)

    def test_far_beep_does_not_choose_disconnected_site(self):
        p, row = agent.Player(), self.row()
        a, b = row["map"]["sites"]
        row["map"]["nav"] = {"nodes": [{"id": s["id"], "position": s["center"]} for s in (a, b)]}
        row["sounds"] = [{"type": "bomb_beep", "distance_band": "far"}]
        p.act(row)
        self.assertEqual(p.search_site, 0)
        self.assertTrue(math.isinf(p.site_travel_costs(row, a["center"])[1]))
