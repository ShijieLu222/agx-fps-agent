"""View-control regressions. Kinematics/FOV here are surrogates, not AGX physics."""
import copy
import math
import unittest

import agent
from test_agent import MAP, observation
from test_v2 import enemy


def arena():
    row = observation(map=copy.deepcopy(MAP), round_tick=360, match_tick=360)
    row['map']['spawns'] = {'CT': [{'position': [0., 30., 0.]}],
                            'T': [{'position': [-30., 0., 0.]}]}
    row['me'].update(hp=100., vx=0., vy=0., active_slot=1)
    return row


class WatchTests(unittest.TestCase):
    def test_public_spawn_only_turns_view_never_fires_or_changes_movement(self):
        row = arena()
        p = agent.Player()
        watched = p.act(row)
        baseline_map = copy.deepcopy(row)
        del baseline_map['map']['spawns']
        normal = agent.Player().act(baseline_map)
        for key in ('move_forward', 'move_right', 'weapon_slot', 'jump', 'fire', 'reload', 'use'):
            self.assertEqual(watched.get(key), normal.get(key))
        self.assertLess(watched['yaw_rate'], -.9)
        self.assertFalse(watched.get('fire', False))

    def test_navigation_camera_returns_to_route_within_two_seconds(self):
        row = arena()
        p = agent.Player()
        p.act(row)
        row.pop('map')
        row['match_tick'] += 75
        row['round_tick'] += 75
        action = p.act(row)
        self.assertAlmostEqual(action['yaw_rate'], 0.)

    def test_rotating_view_preserves_requested_world_movement(self):
        row = arena()
        p = agent.Player()
        p.act(row)
        row.pop('map')
        for yaw in (-2., -1., 0., 1., 2.):
            with self.subTest(yaw=yaw):
                row['me']['yaw'] = yaw
                row['match_tick'] += 3
                row['round_tick'] += 3
                action = p.act(row)
                forward, right = action['move_forward'], action['move_right']
                vx = math.cos(yaw)*forward + math.sin(yaw)*right
                vy = -math.sin(yaw)*forward + math.cos(yaw)*right
                self.assertGreater(vx, .7)
                self.assertAlmostEqual(vy, 0., places=7)

    def test_surrogate_side_sighting_while_advancing(self):
        # An unobstructed target is outside a synthetic 90-degree FOV initially.
        # This verifies a view sweep, not that a real hidden target becomes seen.
        row = arena()
        p = agent.Player()
        seen_at = None
        for i in range(40):
            m = row['me']
            bearing = math.atan2(-(30.-m['y']), -m['x'])
            if abs(agent.angle(bearing-m['yaw'])) < math.pi/4:
                seen_at = i
                row['visible'] = [enemy(5, 0., 30.)]
                action = p.act(row)
                self.assertEqual(action['move_forward'], 0.)
                self.assertIsNotNone(p.enemy_memory)
                break
            action = p.act(row)
            self.assertFalse(action.get('fire', False))
            yaw = m['yaw']
            f, r = action.get('move_forward', 0), action.get('move_right', 0)
            m['x'] += .2*(math.cos(yaw)*f+math.sin(yaw)*r)
            m['y'] += .2*(-math.sin(yaw)*f+math.cos(yaw)*r)
            m['yaw'] = agent.angle(yaw+.15*action.get('yaw_rate', 0))
            m['pitch'] += .1*action.get('pitch_rate', 0)
            row.pop('map', None)
            row['round_tick'] += 3
            row['match_tick'] += 3
        self.assertIsNotNone(seen_at)
        self.assertLess(seen_at, 20)
        self.assertGreater(row['me']['x'], .3)

    def test_three_substep_turning_surrogate_still_reaches_route_goal(self):
        # Rotate before movement in each of three 60 Hz substeps to expose
        # drift caused by holding body-relative inputs during a gaze change.
        row, p = arena(), agent.Player()
        lateral = 0.
        for _ in range(120):
            a = p.act(row)
            m = row['me']
            for _ in range(3):
                m['yaw'] = agent.angle(m['yaw'] + .05*a.get('yaw_rate', 0))
                m['pitch'] += a.get('pitch_rate', 0)/30
                f, r = a.get('move_forward', 0), a.get('move_right', 0)
                m['vx'] = 4*(math.cos(m['yaw'])*f + math.sin(m['yaw'])*r)
                m['vy'] = 4*(-math.sin(m['yaw'])*f + math.cos(m['yaw'])*r)
                m['x'] += m['vx']/60
                m['y'] += m['vy']/60
                lateral = max(lateral, abs(m['y']))
            row.pop('map', None)
            row['match_tick'] += 3
            row['round_tick'] += 3
            if math.hypot(m['x']-10, m['y']) < 1.2:
                break  # v9 continues clearing the next lane after entering.
        self.assertLess(math.hypot(m['x']-10, m['y']), 1.2)
        self.assertLess(lateral, .8)

    def test_damage_search_changes_sector_without_restarting_on_each_hit(self):
        row = arena()
        p = agent.Player()
        p.act(row)
        row.pop('map')
        for i in range(1, 15):
            row['match_tick'] += 3
            row['round_tick'] += 3
            row['me']['hp'] -= 1
            action = p.act(row)
            if i == 1:
                started, first = p.damage_watch_start, action['yaw_rate']
            self.assertEqual(p.damage_watch_start, started)
            self.assertFalse(action.get('fire', False))
        self.assertLess(first, 0)
        self.assertGreater(action['yaw_rate'], 0)
        row['match_tick'] = started + 108
        row['round_tick'] = 435
        action = p.act(row)
        self.assertAlmostEqual(action['yaw_rate'], 0.)

    def test_objective_and_combat_take_priority(self):
        row = arena()
        row['me'].update(x=10., carrying_bomb=True)
        p = agent.Player()
        self.assertEqual(p.act(row).get('use'), 'plant')
        row['me']['carrying_bomb'] = False
        row['visible'] = [enemy(5, 20., 0.)]
        p.damage_watch_start = row['match_tick']
        action = p.act(row)
        self.assertEqual(action['move_forward'], 0.)
        self.assertAlmostEqual(action['yaw_rate'], 0.)
        self.assertIsNone(p.damage_watch_start)

    def test_blind_airborne_recovery_and_defense_do_not_use_spawn_gaze(self):
        for condition in ('blind', 'airborne', 'escape', 'defense'):
            row = arena()
            p = agent.Player()
            p.act(row)
            row.pop('map')
            if condition == 'blind': row['me']['blind_ticks'] = 10
            if condition == 'airborne': row['me']['grounded'] = False
            if condition == 'escape': p.escape_until = 900
            if condition == 'defense': row['side'] = 'defend'
            controls = {'move_forward': .75, 'yaw_rate': .2}
            self.assertEqual(p.watch_navigation(row, controls), controls)

    def test_map_and_round_changes_clear_search_state(self):
        row = arena()
        p = agent.Player()
        p.act(row)
        row.pop('map')
        p.damage_watch_start = 360
        row['round_index'] += 1
        p.act(row)
        self.assertIsNone(p.damage_watch_start)
        row['map_id'] = 'different'
        p.act(row)
        self.assertEqual(p.spawn_views, {})


if __name__ == '__main__':
    unittest.main()
