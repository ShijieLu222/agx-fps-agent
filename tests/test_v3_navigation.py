"""Navigation recovery behavior on small, explicitly connected public graphs."""
import copy
import gc
import math
import unittest

import agent
from test_agent import observation


def fork_map():
    positions = {0: [0., 0., 0.], 'short': [10., 0., 0.],
                 'goal': [20., 0., 0.], 'around': [0., 8., 0.],
                 4: [20., 8., 0.]}
    links = [(0, 'short'), ('short', 'goal'), (0, 'around'),
             ('around', 4), (4, 'goal')]
    return {'sites': [{'center': positions['goal'], 'radius': 4.}],
            'nav': {'nodes': [{'id': k, 'position': v}
                              for k, v in positions.items()],
                    'edges': [{'from': a, 'to': b, 'kind': 'walk'}
                              for a, b in links]}}


class NavigationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.gc_enabled = gc.isenabled()
        self.p = agent.Player()
        self.p.load_map(fork_map())
        self.row = observation()
        self.goal = [20., 0., 0.]

    def tearDown(self):
        if self.gc_enabled:
            gc.enable()

    def move(self, tick, x=0., y=0.):
        self.row['match_tick'] = tick
        self.row['me'].update(x=x, y=y)
        action = self.p.move(self.row, self.goal, {'weapon_slot': 1})
        for key in ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate'):
            self.assertTrue(math.isfinite(action[key]))
            self.assertLessEqual(abs(action[key]), 1.)
        return action

    def block_short_route(self):
        for tick in range(600, 667, 3):
            self.move(tick, x=.2)

    def test_observed_blockage_selects_a_connected_alternative(self):
        self.move(597)
        self.assertEqual(self.p.path[0], [10., 0., 0.])
        self.block_short_route()
        self.assertIn((0, 'short'), self.p.failed_edges)
        for tick in range(669, 730, 3):
            self.move(tick, x=.2)
        self.assertEqual(self.p.path[0], [0., 8., 0.])
        self.assertEqual(self.p.path_nodes[:3], ['around', 4, 'goal'])

    def test_closed_loop_recovery_reaches_goal_around_a_blocked_shortcut(self):
        # A small kinematic fixture, not an AGX physics replica: a wall blocks
        # the advertised short edge while the three-edge detour stays open.
        m = self.row['me']
        wall_contacts = 0
        for tick in range(600, 4200, 3):
            action = self.move(tick, m['x'], m['y'])
            cosine, sine = math.cos(m['yaw']), math.sin(m['yaw'])
            forward, right = action['move_forward'], action['move_right']
            x = m['x'] + .3 * (forward * cosine + right * sine)
            y = m['y'] + .3 * (-forward * sine + right * cosine)
            if 3.8 < x < 6.2 and -2. < y < 2.:
                wall_contacts += 1
            else:
                m['x'], m['y'] = x, y
            m['yaw'] += action['yaw_rate'] * .15
            if math.hypot(m['x'] - 20., m['y']) < 1.:
                break
        self.assertGreater(wall_contacts, 0)
        self.assertGreaterEqual(self.p.escapes, 1)
        self.assertLess(math.hypot(m['x'] - 20., m['y']), 1.)

    def test_sideways_motion_cannot_hide_lack_of_route_progress(self):
        # The old displacement-only detector resets every lateral oscillation.
        self.move(600)
        for i, tick in enumerate(range(603, 861, 3)):
            self.move(tick, x=.1, y=1.0 if (i // 5) % 2 else -1.0)
        self.assertIn((0, 'short'), self.p.failed_edges)
        self.assertGreaterEqual(self.p.escapes, 1)

    def test_combat_pause_is_not_treated_as_a_blocked_route(self):
        self.move(600)
        self.move(1200)
        self.assertEqual(self.p.failed_edges, {})
        self.assertEqual(self.p.escapes, 0)

    def test_edge_failure_expires_and_short_route_can_be_retried(self):
        self.block_short_route()
        self.p.nav_tick = 3000
        path = self.p.route([0., 0., 0.], self.goal)
        self.assertEqual(path[1], [10., 0., 0.])
        self.assertEqual(self.p.failed_edges, {})

    def test_failed_edge_retains_backtrack_anchor_before_replanning(self):
        self.move(600)
        # At x=6 nearest-node snapping would jump over the failed segment.
        self.p.nav_tick = 700
        self.p.penalize_leg()
        path = self.p.route([6., 0., 0.], self.goal)
        self.assertEqual(path[:2], [[0., 0., 0.], [0., 8., 0.]])

    def drop_fixture(self):
        p = agent.Player()
        p.load_map({'nav': {
            'nodes': [{'id': 'high', 'position': [0., 0., 5.]},
                      {'id': 'low', 'position': [3., 0., 0.]},
                      {'id': 'goal', 'position': [9., 0., 0.]},
                      {'id': 'isolated', 'position': [2., .1, 0.]}],
            'edges': [{'from': 'high', 'to': 'low', 'kind': 'drop'},
                      {'from': 'low', 'to': 'goal', 'kind': 'walk'}]}})
        row = observation()
        row['me'].update(x=0., y=0., z=5.)
        p.move(row, [9., 0., 0.], {})
        row['me'].update(x=2., z=0.)
        for tick in range(603, 670, 3):
            row['match_tick'] = tick
            p.move(row, [9., 0., 0.], {})
        self.assertGreater(p.escapes, 0)
        return p, row

    def test_descent_recovery_continues_to_drop_destination_without_reversing(self):
        p, row = self.drop_fixture()
        path = p.route([2., 0., 0.], [9., 0., 0.])
        self.assertEqual(path[0], [3., 0., 0.])
        self.assertNotIn([0., 0., 5.], path)
        self.assertNotIn([2., .1, 0.], path)  # Nearby does not mean connected.

    def test_descent_recovery_does_not_direct_connect_to_unreachable_upper_goal(self):
        p, row = self.drop_fixture()
        path = p.route([2., 0., 0.], [0., 0., 5.])
        self.assertEqual(path, [[3., 0., 0.]])

    def test_walk_after_a_drop_can_backtrack_on_the_reached_lower_floor(self):
        p, row = self.drop_fixture()
        row['match_tick'] = p.escape_until + 3
        row['me'].update(x=3., z=0.)
        p.move(row, [9., 0., 0.], {})
        self.assertEqual(p.nav_from, 'low')
        p.penalize_leg()
        self.assertEqual(p.route([5., 0., 0.], [9., 0., 0.])[0], [3., 0., 0.])

    def test_loading_a_new_map_discards_failure_memory(self):
        self.block_short_route()
        self.assertTrue(self.p.failed_edges)
        self.p.load_map(fork_map())
        self.assertEqual(self.p.failed_edges, {})
        self.assertIsNone(self.p.retry_node)
        self.assertEqual(self.p.path_nodes, [])

    def test_map_and_round_changes_clear_failure_memory(self):
        for changed in ('map', 'round'):
            with self.subTest(changed=changed):
                p = agent.Player()
                row = observation(map=copy.deepcopy(fork_map()))
                p.act(row)
                p.failed_edges[(0, 'short')] = (9999, 100.)
                row.pop('map')
                row['match_tick'] += 3
                if changed == 'map':
                    row['static_schema_hash'] = 'another-map'
                else:
                    row['round_index'] += 1
                p.act(row)
                self.assertEqual(p.failed_edges, {})

    def test_no_navigation_graph_remains_a_finite_direct_fallback(self):
        p = agent.Player()
        action = p.move(observation(), [5., 2., 0.], {})
        self.assertTrue(math.isfinite(action['move_forward']))
        self.assertEqual(p.path_nodes, [None])


if __name__ == '__main__':
    unittest.main()
