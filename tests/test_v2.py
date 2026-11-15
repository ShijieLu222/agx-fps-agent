"""Independent behavior regressions for the selective friend-policy merge.

Synthetic observations validate decisions and contract safety, not match wins.
"""
import copy
import gc
import json
import math
import unittest

import agent
from test_agent import MAP, observation


def enemy(seat=5, x=10.0, y=0.0):
    return {'seat': seat, 'x': x, 'y': y, 'z': 0.0,
            'crouch': False, 'parts': ['chest']}


def contact(slot=0, **changes):
    row = observation(map=copy.deepcopy(MAP), visible=[enemy()], **changes)
    row['me'].update(active_slot=slot, vx=0.0, vy=0.0, vz=0.0,
                     pitch=math.atan2(1.18 - 1.62, 10.0))
    if slot == 0:
        row['me']['weapons'][0] = {'id': 'ak47', 'ammo': 30, 'reserve': 90,
                                  'reload_until': 0, 'shot_ready': 0}
    return row


def advance(row, ticks=3):
    row.pop('map', None)
    row['match_tick'] += ticks
    row['round_tick'] += ticks
    return row


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.gc_enabled = gc.isenabled()
        agent.init({})

    def tearDown(self):
        if self.gc_enabled:
            gc.enable()

    def act(self, row):
        action = agent.act(row)
        json.dumps(action, allow_nan=False)
        allowed = {'weapon_slot', 'buy', 'use', 'pickup_id', 'move_forward',
                   'move_right', 'yaw_rate', 'pitch_rate', 'crouch', 'jump',
                   'walk', 'fire', 'reload'}
        self.assertFalse(set(action) - allowed)
        for key in ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate'):
            if key in action:
                self.assertIn(type(action[key]), (int, float))
                self.assertTrue(math.isfinite(action[key]))
                self.assertLessEqual(abs(action[key]), 1)
        for key in ('crouch', 'jump', 'walk', 'fire', 'reload'):
            if key in action:
                self.assertIs(type(action[key]), bool)
        if 'weapon_slot' in action:
            self.assertIs(type(action['weapon_slot']), int)
            self.assertIn(action['weapon_slot'], (0, 1, 2))
        return action

    def test_countermovement_opposes_observed_world_velocity(self):
        row = contact()
        row['me'].update(vx=2.0, vy=-1.0)
        action = self.act(row)
        self.assertLess(action['move_forward'], 0)
        self.assertGreater(action['move_right'], 0)
        self.assertFalse(action['fire'])
        agent.init({})
        row['me'].update(yaw=math.pi / 2, vx=1.0, vy=2.0)
        action = self.act(row)
        self.assertGreater(action['move_forward'], 0)
        self.assertLess(action['move_right'], 0)

    def test_firing_requires_grounded_and_slow_but_not_arbitrary_global_tick(self):
        for grounded, velocity, expected in ((False, 0., False), (True, 2., False),
                                             (True, 0., True)):
            with self.subTest(grounded=grounded, velocity=velocity):
                agent.init({})
                row = contact(slot=1)
                row['match_tick'] = 603  # v1's global pistol pulse would miss it.
                row['me'].update(grounded=grounded, vx=velocity)
                self.assertIs(self.act(row)['fire'], expected)

    def test_pistol_releases_trigger_between_ready_requests(self):
        row = contact(slot=1)
        self.assertTrue(self.act(row)['fire'])
        self.assertFalse(self.act(advance(row))['fire'])
        self.assertTrue(self.act(advance(row))['fire'])

    def test_rifle_burst_recovers_before_next_requested_burst(self):
        row = contact()
        self.assertTrue(self.act(row)['fire'])
        advance(row, 21)
        self.assertTrue(self.act(row)['fire'])
        advance(row, 3)
        self.assertFalse(self.act(row)['fire'])
        advance(row, 15)
        self.assertFalse(self.act(row)['fire'])
        advance(row, 3)
        self.assertTrue(self.act(row)['fire'])

    def test_false_readiness_is_never_treated_as_elapsed_numeric_clock(self):
        row = contact()
        row['me']['weapons'][0]['shot_ready'] = False
        self.assertFalse(self.act(row)['fire'])
        row['me']['weapons'][0]['shot_ready'] = True
        row['me']['shot_ready'] = False
        self.assertFalse(self.act(advance(row))['fire'])

    def test_visible_target_is_stable_then_changes_when_target_disappears(self):
        row = contact()
        row['visible'] = [enemy(5, 10., 0.), enemy(6, 12., 1.)]
        self.assertAlmostEqual(self.act(row)['yaw_rate'], 0)
        advance(row)
        row['visible'] = [enemy(6, 10., 1.), enemy(5, 10.1, 0.)]
        self.assertAlmostEqual(self.act(row)['yaw_rate'], 0)
        advance(row)
        row['visible'] = [enemy(6, 10., 1.)]
        self.assertLess(self.act(row)['yaw_rate'], -.1)

    def test_recent_contact_stance_is_held_then_expires(self):
        row = contact()
        self.act(row)
        row['visible'] = []
        self.assertTrue(self.act(advance(row))['crouch'])
        self.assertFalse(self.act(advance(row, 91)).get('crouch', False))

    def test_much_closer_visible_target_can_replace_held_target(self):
        row = contact()
        row['visible'] = [enemy(5, 10., 0.), enemy(6, 12., 1.)]
        self.assertAlmostEqual(self.act(row)['yaw_rate'], 0)
        row['visible'] = [enemy(5, 10., 0.), enemy(6, 0., -5.)]
        self.assertGreater(self.act(advance(row))['yaw_rate'], .5)

    def test_blindness_releases_trigger_without_using_old_target_to_fire(self):
        row = contact(slot=1)
        self.assertTrue(self.act(row)['fire'])
        row['me']['blind_ticks'] = 15
        row['visible'] = []
        self.assertFalse(self.act(advance(row)).get('fire', False))
        row['me']['blind_ticks'] = 0
        row['visible'] = [enemy()]
        self.assertTrue(self.act(advance(row))['fire'])

    def test_last_seen_preaim_never_fires_through_lost_visibility(self):
        row = contact()
        row['visible'] = [enemy(5, 0., -10.)]
        self.act(row)
        row['visible'] = []
        action = self.act(advance(row))
        self.assertGreater(action.get('yaw_rate', 0), .5)
        self.assertFalse(action.get('fire', False))
        # Make progress so the navigation escape timer is not the yaw source.
        row['me']['x'] = 1.0
        action = self.act(advance(row, 181))
        self.assertFalse(action.get('fire', False))
        self.assertAlmostEqual(action.get('yaw_rate', 0), 0)

    def test_loaded_secondary_used_for_empty_reloadable_primary_during_contact(self):
        row = contact()
        row['me']['weapons'][0]['ammo'] = 0
        action = self.act(row)
        self.assertEqual(action['weapon_slot'], 1)
        self.assertFalse(action.get('fire', False))  # Still holding primary.
        self.assertNotIn('reload', action)
        row['visible'] = []
        row['me']['active_slot'] = 1
        self.assertEqual(self.act(advance(row))['weapon_slot'], 1)

    def test_running_reload_is_not_requested_again_or_interrupted_by_switch(self):
        row = contact()
        row['visible'] = []
        row['me']['weapons'][0].update(ammo=0, reload_until=900)
        action = self.act(row)
        self.assertEqual(action['weapon_slot'], 0)
        self.assertNotIn('reload', action)
        self.assertFalse(action.get('fire', False))

    def test_contact_preserves_active_reload_then_allows_fallback_after_deadline(self):
        for deadline in (900, True):
            with self.subTest(deadline=deadline):
                agent.init({})
                row = contact()
                row['me']['weapons'][0].update(ammo=0, reload_until=deadline)
                action = self.act(row)
                self.assertEqual(action['weapon_slot'], 0)
                self.assertNotIn('reload', action)
                self.assertFalse(action.get('fire', False))
                row['me']['weapons'][0]['reload_until'] = row['match_tick']
                action = self.act(advance(row))
                self.assertEqual(action['weapon_slot'], 1)
                self.assertNotIn('reload', action)
                self.assertFalse(action.get('fire', False))

    def test_objective_brakes_before_continuous_use_without_switching(self):
        for side, use in (('attack', 'plant'), ('defend', 'defuse')):
            with self.subTest(side=side):
                agent.init({})
                row = observation(map=copy.deepcopy(MAP), side=side)
                row['me'].update(x=10., vx=2., vy=0., active_slot=2,
                                 carrying_bomb=side == 'attack')
                if side == 'defend':
                    row['bomb'] = {'state': 'planted', 'position': [10., 0., 0.],
                                   'fuse_end_tick': 3000}
                action = self.act(row)
                self.assertEqual(action['weapon_slot'], 2)
                self.assertLess(action['move_forward'], 0)
                self.assertNotIn('use', action)
                self.assertNotIn('reload', action)
                row['me']['vx'] = 0
                self.assertEqual(self.act(advance(row)), {'weapon_slot': 2, 'use': use})
                self.assertEqual(self.act(advance(row)), {'weapon_slot': 2, 'use': use})

    def test_known_planted_position_is_remembered_without_inventing_hidden_position(self):
        row = observation(side='defend', bomb={'state': 'planted',
                          'position': [1.5, 0., 0.], 'fuse_end_tick': 3000})
        row['me']['active_slot'] = 1
        self.assertEqual(self.act(row).get('use'), 'defuse')
        row['bomb'].pop('position')
        self.assertEqual(self.act(advance(row)).get('use'), 'defuse')
        row['bomb']['state'] = 'defused'
        self.assertNotIn('use', self.act(advance(row)))
        agent.init({})
        row['bomb']['state'] = 'planted'
        self.assertNotIn('use', self.act(row))

    def test_round_and_map_reset_discard_contact_and_bomb_memory(self):
        for reset in ('round', 'map'):
            with self.subTest(reset=reset):
                agent.init({})
                row = contact(side='defend')
                row['bomb'] = {'state': 'planted', 'position': [1.5, 0., 0.],
                               'fuse_end_tick': 3000}
                self.act(row)
                row['visible'] = []
                row['bomb'].pop('position')
                advance(row)
                if reset == 'round':
                    row['round_index'] += 1
                else:
                    row['static_schema_hash'] = 'new-static-version'
                action = self.act(row)
                self.assertNotIn('use', action)
                self.assertFalse(action.get('crouch', False))

    def test_bomb_memory_is_discarded_after_terminal_states(self):
        for state in ('defused', 'exploded'):
            with self.subTest(state=state):
                agent.init({})
                row = observation(side='defend', bomb={'state': 'planted',
                                  'position': [1.5, 0., 0.], 'fuse_end_tick': 3000})
                self.assertEqual(self.act(row).get('use'), 'defuse')
                row['bomb'] = {'state': state, 'fuse_end_tick': None}
                self.assertNotIn('use', self.act(advance(row)))
                row['bomb'] = {'state': 'planted', 'fuse_end_tick': 6000}
                self.assertNotIn('use', self.act(advance(row)))

    def test_unknown_bomb_state_suspends_use_but_preserves_legal_location(self):
        row = observation(side='defend', bomb={'state': 'planted',
                          'position': [1.5, 0., 0.], 'fuse_end_tick': 3000})
        self.assertEqual(self.act(row).get('use'), 'defuse')
        row['bomb'] = {'state': 'unknown', 'fuse_end_tick': None}
        self.assertNotIn('use', self.act(advance(row)))
        row['bomb'] = {'state': 'planted', 'fuse_end_tick': 3000}
        self.assertEqual(self.act(advance(row)).get('use'), 'defuse')

    def test_attack_seats_share_goal_defenders_spread_across_sites(self):
        def first_direction(side, seat):
            agent.init({})
            row = observation(map=copy.deepcopy(MAP), side=side, seat=seat)
            action = self.act(row)
            return action.get('move_forward', 0), action.get('move_right', 0)
        self.assertEqual(first_direction('attack', 0), first_direction('attack', 1))
        self.assertNotEqual(first_direction('defend', 0), first_direction('defend', 1))


if __name__ == '__main__':
    unittest.main()
