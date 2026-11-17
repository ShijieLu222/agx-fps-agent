"""Objective route continuity and bounded acoustic caution, not win estimates."""
import unittest

import agent
from test_agent import observation
from test_v2 import advance, contact


def corridor():
    return {
        'sites': [{'id': 'A', 'center': [40., 20., 0.], 'radius': 4.}],
        'nav': {
            'nodes': [{'id': i, 'position': p} for i, p in enumerate(
                [[0., 0., 0.], [40., 0., 0.], [40., 20., 0.], [0., 20., 0.]])],
            'edges': [{'from': a, 'to': b, 'kind': 'walk'}
                      for a, b in [(0, 1), (1, 2), (0, 3)]],
        },
    }


class RouteContinuityTests(unittest.TestCase):
    def setup_route(self):
        p = agent.Player()
        p.load_map(corridor())
        r = observation(side='defend')
        r['me'].update(vx=0., vy=0.)
        goal = [40., 20., 0.]
        p.move(r, goal, {})
        r['me']['x'] = 12.
        p.move(advance(r), goal, {})
        self.assertEqual(p.path_nodes[0], 1)
        return p, r

    def test_new_bomb_coordinate_keeps_forward_corridor_and_corner(self):
        p, r = self.setup_route()
        new_goal = [39., 19., 0.]
        a = p.move(advance(r), new_goal, {})
        self.assertGreater(a['move_forward'], .7)
        self.assertEqual(p.path[:2], [[40., 0., 0.], [40., 20., 0.]])
        self.assertEqual(p.path[-1], new_goal)
        self.assertEqual(p.nav_from, 0)

    def test_repeated_coordinate_updates_do_not_reset_progress_clock(self):
        p, r = self.setup_route()
        before = p.leg_progress_tick
        for i in range(8):
            p.move(advance(r), [39. + i * .01, 19., 0.], {})
        self.assertEqual(p.leg_progress_tick, before)
        self.assertEqual(p.path[0], [40., 0., 0.])

    def test_different_graph_destination_replans(self):
        p, r = self.setup_route()
        p.move(advance(r), [0., 20., 0.], {})
        self.assertEqual(p.path[-1], [0., 20., 0.])
        self.assertNotIn([40., 20., 0.], p.path)

    def test_failed_leg_recovery_is_not_bypassed_by_bomb_update(self):
        p, r = self.setup_route()
        p.penalize_leg()
        p.path = []
        p.move(advance(r), [39., 19., 0.], {})
        self.assertEqual(p.path[0], [0., 0., 0.])
        self.assertIsNotNone(p.retry_node)


class AcousticPaceTests(unittest.TestCase):
    def row(self):
        r = observation(side='defend', sounds=[
            {'type': 'footstep', 'direction': 'left', 'distance_band': 'near'}])
        r['me'].update(vx=0., vy=0.)
        return r

    def test_repeated_anonymous_sounds_allow_only_short_slow_window(self):
        p, r = agent.Player(), self.row()
        p.path = [[40., 0., 0.]]
        slowed = 0
        for _ in range(200):
            p.observe_sounds(r)
            a = p.prepare_approach(r, {'move_forward': .75, 'walk': False})
            slowed += bool(a.get('walk'))
            self.assertIsNotNone(p.sound_yaw)
            self.assertFalse(a.get('fire', False))
            advance(r)
        self.assertGreater(slowed, 0)
        self.assertLessEqual(slowed, 12)
        self.assertEqual(a['move_forward'], .75)

    def test_quiet_interval_rearms_caution_but_one_missing_cue_does_not(self):
        p, r = agent.Player(), self.row()
        p.observe_sounds(r)
        advance(r, 45)
        p.observe_sounds(r)
        self.assertFalse(p.prepare_approach(r, {'move_forward': .75}).get('walk'))
        r['sounds'] = []
        p.observe_sounds(advance(r, 181))
        r['sounds'] = self.row()['sounds']
        p.observe_sounds(advance(r))
        self.assertTrue(p.prepare_approach(r, {'move_forward': .75})['walk'])

    def test_carrier_and_postplant_travel_keep_speed_and_local_corner_control(self):
        for carrying, state in [(True, 'held'), (False, 'planted')]:
            p, r = agent.Player(), self.row()
            r['side'] = 'attack'
            r['me']['carrying_bomb'] = carrying
            r['bomb']['state'] = state
            p.observe_sounds(r)
            p.path = [[4., 0., 0.], [4., 8., 0.]]
            a = p.prepare_approach(r, {'move_forward': .75})
            self.assertEqual(a['move_forward'], .75)
            self.assertFalse(a['walk'])
            p.path[0] = [1.5, 0., 0.]
            a = p.prepare_approach(r, {'move_forward': .75})
            self.assertAlmostEqual(a['move_forward'], .6)
            self.assertFalse(a['walk'])

    def test_visible_threat_and_interaction_keep_existing_priority(self):
        for state in ('held', 'planted'):
            r = contact()
            r['side'] = 'attack'
            r['map'] = corridor()
            r['bomb'] = {'state': state, 'position': [40., 20., 0.]}
            r['me']['carrying_bomb'] = state == 'held'
            p = agent.Player()
            a = p.act(r)
            self.assertEqual(a.get('move_forward'), 0.)
            self.assertEqual(p.target_seat, r['visible'][0]['seat'])
            self.assertTrue(a.get('fire'))
        r = observation(side='attack')
        r['me']['carrying_bomb'] = True
        p = agent.Player()
        a = {'use': 'plant', 'move_forward': 0., 'fire': False}
        self.assertEqual(p.prepare_approach(r, a), a)

    def test_round_reset_restores_first_cue_caution(self):
        p, r = agent.Player(), self.row()
        p.observe_sounds(r)
        advance(r, 90)
        p.observe_sounds(r)
        self.assertFalse(p.prepare_approach(r, {}).get('walk'))
        p.reset_tactics()
        p.observe_sounds(r)
        self.assertTrue(p.prepare_approach(r, {'move_forward': .75})['walk'])


if __name__ == '__main__':
    unittest.main()
