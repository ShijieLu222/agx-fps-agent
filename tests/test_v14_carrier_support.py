"""AGX attacking support decisions with a visible allied objective."""
import math
import unittest

import agent
from test_agent import observation
from test_v2 import advance, enemy


def attack_row():
    return observation(map={"spawns": {"T": [{"position": [0., 0., 0.]}]},
                            "sites": [{"id": "A", "center": [0., 75., 0.], "radius": 6.},
                                      {"id": "B", "center": [80., 0., 0.], "radius": 6.}]},
                       bomb={"state": "held", "carrier": 2, "position": [0., 0., 0.]})


class CarrierSupportTests(unittest.TestCase):
    def test_observed_progress_changes_site_before_entering_v13_radius(self):
        row, p = attack_row(), agent.Player()
        p.act(row)
        self.assertIsNone(p.attack_site)
        row.pop("map")
        row["bomb"]["position"] = [35., 0., 0.]
        a = p.act(advance(row, 180))
        self.assertEqual(p.attack_site, 1)
        self.assertEqual(p.goal, [80., 0., 0.])
        self.assertGreater(a["move_forward"], .5)
        self.assertLess(abs(a["move_right"]), .01)

    def test_regroup_goal_holds_through_small_movements_and_eases_on_reunion(self):
        row, p = attack_row(), agent.Player()
        p.act(row)
        row.pop("map")
        row["me"].update(x=48., y=-38.)
        row["bomb"]["position"] = [43., 24., 0.]
        p.act(advance(row, 180))
        for dx in range(1, 16):
            row["bomb"]["position"][0] = 43.+dx
            p.act(advance(row, 30))
            self.assertEqual(p.goal, [43., 24., 0.])
        row["bomb"]["position"] = [52., 39., 0.]
        p.act(advance(row))
        self.assertEqual(p.goal, [52., 39., 0.])
        row["me"].update(x=52., y=29.)
        p.act(advance(row))
        self.assertEqual(p.goal, [80., 0., 0.])
        row["me"]["y"] = 23.
        p.act(advance(row))
        self.assertIsNone(p.escort_goal)

    def test_carrier_behind_at_spawn_does_not_pull_advanced_player_back(self):
        row, p = attack_row(), agent.Player()
        row["me"]["y"] = 40.
        for _ in range(3):
            p.act(row)
            self.assertIsNone(p.escort_goal)
            self.assertEqual(p.goal, [0., 75., 0.])
            advance(row)

    def test_handoff_midpoint_noise_cannot_switch_sites_repeatedly(self):
        row, p = attack_row(), agent.Player()
        row["map"]["sites"][1]["center"] = [32., 75., 0.]
        row["bomb"]["position"] = [1., 70., 0.]
        p.act(row)
        self.assertEqual(p.attack_site, 0)
        row.pop("map")
        for x in (15., 17., 15.5, 16.5):
            row["bomb"]["position"] = [x, 70., 0.]
            p.act(advance(row))
            self.assertEqual(p.attack_site, 0)
        row["bomb"].update(carrier=1, position=[31., 73., 0.])
        p.act(advance(row))
        self.assertEqual(p.attack_site, 1)
        self.assertIsNone(p.escort_goal)

    def test_lost_carrier_position_and_objective_interactions_override_escort(self):
        row, p = attack_row(), agent.Player()
        p.act(row)
        row.pop("map")
        row["me"].update(x=48., y=-38.)
        row["bomb"]["position"] = [43., 24., 0.]
        p.act(advance(row, 180))
        self.assertIsNotNone(p.escort_goal)
        row["bomb"].pop("position")
        p.act(advance(row))
        self.assertIsNone(p.escort_goal)
        row["me"].update(x=0., y=0.)
        row["bomb"].update(state="world", id="r0:bomb", position=[0., .1, 0.])
        self.assertEqual(p.act(advance(row)).get("use"), "pickup")
        row["bomb"].update(state="held", carrier=0, position=[80., 0., 0.])
        row["me"].update(x=80., carrying_bomb=True)
        self.assertEqual(p.act(advance(row)).get("use"), "plant")

    def test_current_visible_combat_still_preempts_regroup(self):
        row, p = attack_row(), agent.Player()
        p.act(row)
        row.pop("map")
        row["bomb"]["position"] = [35., 0., 0.]
        row["visible"] = [enemy()]
        a = p.act(advance(row, 180))
        self.assertEqual(p.target_seat, row["visible"][0]["seat"])
        self.assertEqual(a.get("move_forward"), 0.)

    def test_escort_reaches_carriers_site_in_open_floor_fixture(self):
        # Feed the requested direction back into a simple open floor, with a
        # carrier walking across the initially assigned site route.
        for direction in (-1., 1.):
            row, p = attack_row(), agent.Player()
            row.update(match_tick=360, round_tick=360)
            row["map"]["sites"][1]["center"][0] *= direction
            m = row["me"]
            for frame in range(661):
                row["bomb"]["position"] = [min(80., frame*.2)*direction, 0., 0.]
                a = p.act(row)
                fwd, right = a.get("move_forward", 0.), a.get("move_right", 0.)
                c, s = math.cos(m["yaw"]), math.sin(m["yaw"])
                m["vx"], m["vy"] = 5*(c*fwd+s*right), 5*(-s*fwd+c*right)
                m["x"] += m["vx"]*.05
                m["y"] += m["vy"]*.05
                m["yaw"] += .15*a.get("yaw_rate", 0.)
                if math.hypot(m["x"]-80*direction, m["y"]) <= 6:
                    break
                advance(row)
            else:
                self.fail("support player did not reach the carrier's site within 33 seconds")
