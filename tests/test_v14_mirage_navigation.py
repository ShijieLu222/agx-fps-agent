"""Directed game-map traversal and short riser regressions."""
import copy
import math
import unittest

import agent
from test_agent import observation


GEOMETRY = "ef9384945046620f3f61ac579ae91f0beac8d5541c871e1996faf7a36f2e39db"


def corridor_map():
    anchors = {"2126": [59.05, 60.45, 13.35], "2089": [59.23, 60.73, 13.38],
               "1720": [89.8, 51., 15.95], "1724": [89.8, 51.1, 15.95]}
    return {"id": "mirage-cs2-v54", "geometry_hash": GEOMETRY,
            "nav": {"nodes": [{"id": k, "position": v} for k, v in anchors.items()],
                    "edges": []}}


class MirageNavigationTests(unittest.TestCase):
    def test_corridor_is_directed_and_requires_matching_geometry(self):
        arena = corridor_map()
        p = agent.Player()
        p.load_map(arena)
        p.route(p.nodes["2126"], p.nodes["1720"])
        self.assertEqual([n for n in p.path_nodes if type(n) is int], list(range(-1, -18, -1)))
        self.assertEqual(p.corridor_jumps, {(-5, -6), (-7, -8), (-10, -11)})
        self.assertFalse(any(n == -16 for n, _ in p.edges[-17]))
        for field in ("id", "geometry_hash"):
            other = copy.deepcopy(arena)
            other[field] = "new-map"
            p.load_map(other)
            self.assertNotIn(-1, p.nodes)
            self.assertFalse(p.corridor_jumps)

    def test_witnessed_jump_triggers_only_near_launch_and_is_bounded(self):
        row, p = observation(), agent.Player()
        p.load_map(corridor_map())
        a, b = p.nodes[-5], p.nodes[-6]
        row["me"].update(dict(zip(("x", "y", "z"), a)))
        p.goal, p.path, p.path_nodes, p.nav_from = list(b), [list(b)], [-6], -5
        first = p.move(row, b, {}, stop_at_goal=False)
        self.assertTrue(first["jump"])
        row["match_tick"] += 3
        row["me"]["grounded"] = False
        self.assertFalse(p.move(row, b, {}, stop_at_goal=False).get("jump"))
        self.assertEqual(p.jump_attempts[-6], 1)

    def test_nearest_witnessed_point_cannot_force_opposite_direction(self):
        arena = corridor_map()
        public = {"crossing": [70.55, 60.6, 13.7], "A": [61., 67., 13.7]}
        arena["nav"]["nodes"].extend({"id": k, "position": v} for k, v in public.items())
        arena["nav"]["edges"].append({"from": "crossing", "to": "A", "kind": "walk"})
        p = agent.Player()
        p.load_map(arena)
        start = p.nodes[-5]
        p.route(start, public["A"])
        self.assertEqual(p.path_nodes, ["crossing", "A", None])
        p.route(public["A"], start)
        self.assertEqual(p.path_nodes, ["A", "crossing", None])

    def test_stalled_mirage_riser_reaches_next_floor(self):
        row, p = observation(), agent.Player()
        m, arena = row["me"], corridor_map()
        # The sampled target rises just .28 m, measured at the repeated A
        # approach. The fixture stops grounded movement at a .20 m riser.
        arena["nav"]["nodes"] += [{"id": "begin", "position": [0., 0., 0.]},
                                  {"id": "end", "position": [3.2, 0., .28]}]
        arena["nav"]["edges"].append({"from": "begin", "to": "end", "kind": "source-adjacency"})
        p.load_map(arena)
        jumped = False
        for tick in range(600, 1200, 3):
            row["match_tick"] = tick
            a = p.move(row, p.nodes["end"], {})
            x = m["x"] + .18 * a.get("move_forward", 0)
            if m["x"] < 1.5 <= x:
                if not a.get("jump"):
                    continue
                m["z"], jumped = .20, True
            m["x"] = x
            if math.dist([x, 0., m["z"]], p.nodes["end"]) < .65:
                break
        self.assertTrue(jumped)
        self.assertLess(math.dist([m["x"], 0., m["z"]], p.nodes["end"]), .65)
