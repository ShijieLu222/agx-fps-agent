"""Own-observation safety regressions; no hidden enemy or game simulation."""
import copy
import json
import math
import unittest

import agent
from test_agent import MAP, observation
from test_v2 import enemy


class UnseenDamageTests(unittest.TestCase):
    def setUp(self):
        self.p = agent.Player()
        self.row = observation(map=copy.deepcopy(MAP), side='defend', seat=4)
        self.row['me'].update(hp=100., active_slot=1, vx=0., vy=0.)
        self.threats = []
        self.p.remember_threat = lambda o: self.threats.append(o['match_tick'])

    def step(self, x, y=0., z=0., hp=None, ticks=3, **changes):
        self.row['match_tick'] += ticks
        self.row['round_tick'] += ticks
        self.row.update(changes)
        self.row['me'].update(x=x, y=y, z=z)
        if hp is not None:
            self.row['me']['hp'] = hp
        a = self.p.act(self.row)
        self.row.pop('map', None)
        json.dumps(a, allow_nan=False)
        for key in ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate'):
            self.assertLessEqual(abs(a.get(key, 0.)), 1.)
        return a

    def walk(self):
        for x in (0., .2, .4, .6, .8, 1.):
            self.step(x)

    def test_unseen_damage_reverses_observed_walk_with_no_blind_shot(self):
        self.walk()
        self.p.enemy_memory = ([5., 5., 0.], self.row['match_tick'])
        a = self.step(1.2, hp=74.)
        self.assertLess(a['move_forward'], -.8)
        self.assertAlmostEqual(a['move_right'], 0.)
        self.assertEqual(a['yaw_rate'], 0.)
        self.assertFalse(a['fire'])
        self.assertFalse(a['crouch'])
        self.assertFalse(a['walk'])
        self.assertIsNone(self.p.enemy_memory)
        self.assertEqual(len(self.threats), 1)

    def test_damage_response_is_bounded_and_does_not_restart_on_each_hit(self):
        self.walk()
        self.step(1.2, hp=74.)
        until = self.p.retreat_until
        for i in range(1, 8):
            self.step(1.2, hp=74.-i)
        self.assertEqual(self.p.retreat_until, until)
        self.assertEqual(len(self.threats), 1)
        self.row['match_tick'] = until
        self.row['round_tick'] += 54
        self.p.act(self.row)
        self.assertEqual(self.p.retreat_path, [])
        self.assertEqual(len(self.threats), 1)

    def test_visible_enemy_interrupts_retreat_and_uses_normal_combat(self):
        self.walk()
        self.step(1.2, hp=74.)
        self.row['me']['pitch'] = math.atan2(1.18-1.62, 10.-1.2)
        a = self.step(1.2, visible=[enemy()])
        self.assertEqual(self.p.retreat_path, [])
        self.assertTrue(a['fire'])
        self.assertEqual(len(self.threats), 1)

    def test_no_injury_does_not_trigger_evasion_from_unattributed_sounds(self):
        self.walk()
        a = self.step(1.2, sounds=[{'kind': 'footstep', 'direction': 'center', 'band': 'near'}])
        self.assertGreater(a['move_forward'], 0.)
        self.assertEqual(self.threats, [])

    def test_no_traversed_path_does_not_invent_retreat_route(self):
        self.step(1.)
        self.step(1., hp=50.)
        self.assertEqual(self.p.retreat_path, [])
        self.assertEqual(self.threats, [])

    def test_drop_and_airborne_motion_invalidate_reverse_traversal(self):
        self.walk()
        self.row['me']['grounded'] = False
        self.step(1.2, z=-.1, hp=80.)
        self.assertEqual(self.p.retreat_path, [])
        self.row['me']['grounded'] = True
        self.step(1.2, z=-2., hp=60.)
        self.assertEqual(self.p.retreat_path, [])
        self.assertEqual(self.threats, [])

    def test_observed_corner_is_retraced_without_diagonal_shortcut(self):
        for x, y in ((0., 0.), (.2, 0.), (.4, 0.), (.6, 0.), (.6, .2), (.6, .4)):
            self.step(x, y)
        a = self.step(.6, .6, hp=75.)
        self.assertAlmostEqual(a['move_forward'], 0., places=6)
        self.assertLess(a['move_right'], -.8)
        self.assertEqual(self.p.retreat_path[0], [.6, .4, 0.])

    def test_objective_in_range_takes_precedence_over_unseen_damage(self):
        for side, use in (('attack', 'plant'), ('defend', 'defuse')):
            with self.subTest(side=side):
                self.setUp()
                self.row['side'] = side
                for x in (7.4, 7.6, 7.8, 8.):
                    self.step(x)
                self.row['me']['carrying_bomb'] = side == 'attack'
                if side == 'defend':
                    self.row['bomb'] = {'state': 'planted', 'position': [10., 0., 0.], 'fuse_end_tick': 3000}
                a = self.step(8.2, hp=70.)
                self.assertEqual(a.get('use'), use)
                self.assertEqual(self.threats, [])

    def test_instant_unseen_death_remembers_threat_only_once(self):
        self.walk()
        self.row['me']['alive'] = False
        self.step(1.2, hp=0.)
        self.step(1.2, hp=0.)
        self.assertEqual(len(self.threats), 1)
        self.assertEqual(self.p.breadcrumbs, [])

    def test_known_enemy_death_does_not_masquerade_as_unseen_threat(self):
        self.walk()
        self.step(1.2, visible=[enemy()])
        self.row['me']['alive'] = False
        self.step(1.2, hp=0., visible=[])
        self.assertEqual(self.threats, [])

    def test_round_and_map_changes_clear_transient_safety_state(self):
        for change in ({'round_index': 1}, {'map_id': 'different'}):
            with self.subTest(change=change):
                self.setUp()
                self.walk()
                self.step(1.2, hp=74.)
                self.step(0., hp=100., **change)
                self.assertEqual(self.p.retreat_path, [])
                self.assertEqual(self.p.damage_cooldown, 0)
                self.assertEqual(len(self.p.breadcrumbs), 1)
                self.assertEqual(len(self.threats), 1)

    def test_all_team_seats_start_moving_at_live_start(self):
        for seat in range(10):
            with self.subTest(seat=seat):
                p = agent.Player()
                row = observation(map=copy.deepcopy(MAP), seat=seat,
                                  round_tick=360, match_tick=360)
                a = p.act(row)
                self.assertGreater(math.hypot(a.get('move_forward', 0.), a.get('move_right', 0.)), .5)


if __name__ == '__main__':
    unittest.main()
