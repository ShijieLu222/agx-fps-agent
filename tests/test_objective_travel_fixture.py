"""Regression tests for the closed-loop evaluator's local objective referee."""
import types
import unittest

from scripts.evaluate_objective_travel import Terrain, rollout
from test_agent import observation


class ObjectiveRefereeTests(unittest.TestCase):
    def simulate(self, distance, action, limit=10):
        site = {'id': 'A', 'center': [0., 0., 0.], 'radius': 4.}
        mapping = {'id': 'test', 'geometry_hash': 'test', 'sites': [site]}
        terrain = Terrain({'walkable_surface_candidates': {
            'origin_xy': [0, 0], 'cell_size_m': 1, 'cells': [[0, 0, 0.]]},
            'wall_cover_contours': []})
        module = types.SimpleNamespace(init=lambda _: None, act=action)
        return rollout(module, mapping, terrain, {'position': [distance, 0., 0.], 'yaw': 0.},
                       site, 'defend', observation(), limit)

    def test_continuous_in_range_use_completes_at_full_no_kit_duration(self):
        always = lambda _: {'use': 'defuse'}
        self.assertEqual(self.simulate(2.65, always)['completion_s'], 10.)
        self.assertIsNone(self.simulate(2.65, always, limit=9)['completion_s'])
        self.assertIsNone(self.simulate(2.8, always)['completion_s'])

    def test_one_dropped_request_resets_progress(self):
        def interrupted(row):
            return {} if row['round_tick'] == 510 else {'use': 'defuse'}

        self.assertIsNone(self.simulate(0., interrupted)['completion_s'])
