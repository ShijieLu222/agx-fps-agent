"""Map contract/startup regressions; fixtures do not simulate a real match."""
import unittest
from unittest.mock import patch

import agent
from test_agent import observation
from test_v9_defense_combat import station_map
from scripts.replay_check import validate_action


class StartupTests(unittest.TestCase):
    def opening(self, row):
        p = agent.Player()
        first = observation(map=row, phase='buy', match_tick=0, round_tick=0)
        self.assertEqual(p.act(first), {'buy': 'armor'})
        self.assertTrue(p.nodes)
        self.assertTrue(p.guard_plans)
        with patch.object(p, 'load_map', side_effect=AssertionError('Map must be cached')):
            self.assertFalse(validate_action(p.act(observation(phase='buy', match_tick=3, round_tick=3))))
            self.assertFalse(validate_action(p.act(observation(phase='live', match_tick=360, round_tick=360))))
        return p

    def test_null_hints_still_builds_navigable_stations(self):
        row = station_map(); row['bot_waypoints'] = None
        p = self.opening(row)
        self.assertTrue(p.pressure_points)

    def test_missing_empty_and_null_hints_have_equivalent_fallback(self):
        plans = []
        for value in ('missing', {}, None):
            row = station_map(); row.pop('bot_waypoints')
            if value != 'missing': row['bot_waypoints'] = value
            p = agent.Player(); p.load_map(row); plans.append((p.guard_plans, p.pressure_points))
        self.assertEqual(plans[0], plans[1]); self.assertEqual(plans[1], plans[2])
        self.assertTrue(plans[0][0])

    def test_null_individual_hint_lists_fall_back_to_graph(self):
        row = station_map(); row['bot_waypoints'] = {'holdB': None, 'watchB_CT': None}
        self.opening(row)

    def test_null_boxes_do_not_disable_guard_or_pressure(self):
        row = station_map(); row['boxes'] = None
        self.assertTrue(self.opening(row).pressure_points)

    def test_null_landmarks_do_not_block_navigation(self):
        row = station_map(); row['route_landmarks'] = None
        self.assertEqual(self.opening(row).zone_points, {})

    def test_null_spawn_group_is_optional(self):
        row = station_map(); row['spawns']['CT'] = None
        self.opening(row)

    def test_map_work_occurs_in_first_callback_only(self):
        row = station_map(); p = agent.Player()
        with patch.object(p, 'build_guard_plans', wraps=p.build_guard_plans) as build:
            p.act(observation(map=row, phase='buy', match_tick=0, round_tick=0))
            self.assertEqual(build.call_count, 1)
            for tick, phase, rnd in [(3, 'buy', 0), (360, 'live', 0), (8000, 'buy', 1)]:
                self.assertFalse(validate_action(p.act(observation(phase=phase,match_tick=tick,round_tick=0,round_index=rnd))))
            self.assertEqual(build.call_count, 1)

    def test_map_change_rebuilds_without_stale_plans(self):
        p = self.opening(station_map())
        row = station_map(); row['bot_waypoints'] = None
        for n in row['nav']['nodes']: n['position'][0] += 100
        row['sites'][0]['center'][0] += 100
        for pads in row['spawns'].values():
            for pad in pads: pad['position'][0] += 100
        p.act(observation(map=row, map_id='another', phase='buy', match_tick=0, round_tick=0))
        self.assertTrue(p.guard_plans)
        self.assertTrue(all(s[0] >= 100 for s in p.guard_plans[0]['stations']))


if __name__ == '__main__': unittest.main()
