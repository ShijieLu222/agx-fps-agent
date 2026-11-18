import unittest

import agent
from test_agent import observation


def search_row():
    return observation(side="defend", bomb={"state": "planted", "fuse_end_tick": 4000},
                       map={"sites": [{"id": "A", "center": [0., 0., 0.], "radius": 6.},
                                      {"id": "B", "center": [70., 0., 0.], "radius": 6.}]})


class RetakeTests(unittest.TestCase):
    def test_far_cues_can_redirect_before_visiting_the_wrong_center(self):
        p, row = agent.Player(), search_row()
        row["me"]["x"] = 27.
        p.act(row)
        self.assertEqual(p.search_site, 0)
        row.pop("map")
        row["sounds"] = [{"type": "bomb_beep", "distance_band": "far", "direction": "unknown"}]
        for x in (23., 20., 17.):
            row["me"]["x"] = x
            row["match_tick"] += 60
            a = p.act(row)
        self.assertEqual(p.search_site, 1)
        self.assertIsNone(p.known_bomb)
        self.assertNotIn("use", a)

    def test_repeat_at_same_position_does_not_strengthen_evidence(self):
        p, row = agent.Player(), search_row()
        p.load_map(row["map"])
        sounds = [{"type": "bomb_beep", "distance_band": "far"}]
        self.assertIsNone(p.compare_beep_areas(sounds, [24., 0., 0.]))
        masks = list(p.beep_masks)
        for _ in range(20):
            self.assertIsNone(p.compare_beep_areas(sounds, [24., 0., 0.]))
        self.assertEqual(p.beep_masks, masks)
        self.assertIsNone(p.compare_beep_areas(
            sounds+[{"type": "bomb_beep_fast", "distance_band": "near"}], [0., 0., 0.]))
        self.assertEqual(p.beep_masks, masks)

    def test_partial_band_needs_corroborating_recent_teammate_loss(self):
        row, p = search_row(), agent.Player()
        p.load_map(row["map"])
        sounds = [{"type": "bomb_beep", "distance_band": "far"}]
        pos = [20., 0., 0.]
        self.assertIsNone(p.compare_beep_areas(sounds, pos, 2300))
        p.fallen_zones = {2: ("B", 2200)}
        self.assertEqual(p.compare_beep_areas(sounds, pos, 2300), 1)
        self.assertIsNone(p.compare_beep_areas(sounds, pos, 3600))
        p.fallen_zones[3] = ("A", 2290)
        self.assertIsNone(p.compare_beep_areas(sounds, pos, 2300))

    def test_only_observed_enemy_plants_contribute_to_later_searches(self):
        row, p = search_row(), agent.Player()
        row["round_index"] = 12
        row["me"]["x"] = 32.
        p.act(row)
        self.assertEqual(p.confirmed_enemy_plants, {})
        row.pop("map")
        row["bomb"]["site"] = "B"
        row["match_tick"] += 3
        p.act(row)
        self.assertEqual(p.confirmed_enemy_plants, {1: 12})
        row["round_index"] = 13
        row["bomb"].pop("site")
        row["match_tick"] += 60
        p.act(row)
        self.assertEqual(p.search_site, 1)
        self.assertIsNone(p.known_bomb)
        row["round_index"] = 20
        row["match_tick"] += 60
        p.act(row)
        self.assertEqual(p.search_site, 0)

    def test_objective_interactions_use_published_distances_with_margin(self):
        for side, state, distance, expected in [
                ("defend", "planted", 2.55, "defuse"), ("defend", "planted", 2.85, None),
                ("attack", "world", 2.05, "pickup"), ("attack", "world", 2.35, None)]:
            row = search_row()
            row.update(side=side, bomb={"state": state, "position": [distance, 0., 0.], "id": "b"})
            action = agent.Player().act(row)
            self.assertEqual(action.get("use"), expected)
