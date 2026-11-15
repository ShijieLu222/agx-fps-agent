"""Stopping regressions, including a bounded *approximate* motion fixture.

The fixture is not AGX physics: it omits collisions, slopes, crouch transitions,
and weapon effects. Its friction/acceleration sweep checks controller stability,
not server completion time or winning strength. The R15 velocity values below
are measurements from the owner's legal observation, not model parameters.
"""
import copy
import math
import unittest

import agent
from test_agent import MAP, observation
from test_v2 import contact


def motion_step(me, action, friction=5.0, stop_speed=2.0, acceleration=30.0):
    """Three 60 Hz steps of a deliberately simple ground-motion surrogate."""
    yaw = me['yaw']
    forward = action.get('move_forward', 0.0)
    right = action.get('move_right', 0.0)
    desired = (4.5 * (forward * math.cos(yaw) + right * math.sin(yaw)),
               4.5 * (-forward * math.sin(yaw) + right * math.cos(yaw)))
    vx, vy = me.get('vx', 0.0), me.get('vy', 0.0)
    for _ in range(3):
        speed = math.hypot(vx, vy)
        if speed:
            after_drag = max(0.0, speed - max(speed, stop_speed) * friction / 60)
            vx, vy = vx * after_drag / speed, vy * after_drag / speed
        if forward or right:
            dx, dy = desired[0] - vx, desired[1] - vy
            delta = math.hypot(dx, dy)
            scale = min(1.0, acceleration / 60 / delta) if delta else 0.0
            vx, vy = vx + dx * scale, vy + dy * scale
    me.update(vx=vx, vy=vy)


class V4StoppingTests(unittest.TestCase):
    def test_r15_observed_velocities_release_instead_of_reverse(self):
        player = agent.Player()
        # Consecutive R15 observations, followed by the repeated limit cycle.
        velocities = [(4.0927825663, -0.1558811338),
                      (1.7368006388, -0.0661497960),
                      (-0.4238078058, 0.0161416338),
                      (0.8569052762, -0.0326370845),
                      (-0.7804841078, 0.0297264196),
                      (0.7804841078, -0.0297264196)]
        for vx, vy in velocities:
            with self.subTest(vx=vx, vy=vy):
                me = dict(vx=vx, vy=vy, yaw=0.0, grounded=True, active_slot=0)
                action = player.objective_action(me, 1, 'plant')
                self.assertEqual(action['weapon_slot'], 0)
                self.assertEqual(action['move_forward'], 0.0)
                self.assertEqual(action['move_right'], 0.0)
                self.assertNotIn('use', action)
                if math.hypot(vx, vy) <= 1.0:
                    self.assertEqual(agent.counterstop(me),
                                     {'move_forward': 0.0, 'move_right': 0.0})

    def test_closed_loop_objective_stops_and_sustains_interaction(self):
        for use, hold_decisions in (('plant', 60), ('defuse', 200)):
            for speed in (0.78105, 4.09575, 6.0):
                for friction in (3.0, 5.0, 8.0):
                    for yaw in (0.0, math.pi / 2, -2.0):
                        with self.subTest(use=use, speed=speed,
                                          friction=friction, yaw=yaw):
                            player = agent.Player()
                            me = dict(vx=speed * .6, vy=-speed * .8,
                                      yaw=yaw, grounded=True, active_slot=2)
                            started, consecutive = None, 0
                            previous = speed
                            for decision in range(260):
                                action = player.objective_action(me, 0, use)
                                self.assertEqual(action['weapon_slot'], 2)
                                self.assertFalse(action.get('fire', False))
                                self.assertFalse(action.get('reload', False))
                                if action.get('use') == use:
                                    if started is None:
                                        started = decision
                                    consecutive += 1
                                elif started is not None:
                                    self.fail('Objective request interrupted after stopping')
                                motion_step(me, action, friction=friction, stop_speed=1.0)
                                speed_now = math.hypot(me['vx'], me['vy'])
                                self.assertLessEqual(speed_now, previous + 1e-12)
                                previous = speed_now
                                if consecutive >= hold_decisions:
                                    break
                            self.assertIsNotNone(started)
                            self.assertLess(started, 30)  # Surrogate bound only.
                            self.assertEqual(consecutive, hold_decisions)

    def test_combat_stopping_converges_across_motion_surrogates(self):
        for speed in (.78105, 2.0, 4.5, 6.0):
            for friction in (3.0, 5.0, 8.0):
                for acceleration in (10.0, 30.0, 50.0):
                    with self.subTest(speed=speed, friction=friction,
                                      acceleration=acceleration):
                        me = dict(vx=-.8 * speed, vy=.6 * speed,
                                  yaw=1.1, grounded=True)
                        for _ in range(60):
                            motion_step(me, agent.counterstop(me), friction=friction,
                                        acceleration=acceleration)
                        self.assertLessEqual(math.hypot(me['vx'], me['vy']), .1)

    def test_airborne_objective_waits_then_resumes_on_landing(self):
        player = agent.Player()
        for use in ('plant', 'defuse'):
            me = dict(vx=0.0, vy=0.0, yaw=0.0, grounded=False, active_slot=1)
            action = player.objective_action(me, 0, use)
            self.assertNotIn('use', action)
            self.assertEqual(action['weapon_slot'], 1)
            me['grounded'] = True
            self.assertEqual(player.objective_action(me, 0, use),
                             {'weapon_slot': 1, 'use': use})

    def test_releasing_brake_does_not_relax_fire_speed_requirement(self):
        row = contact()
        row['me']['vx'] = .78105
        player = agent.Player()
        action = player.act(row)
        self.assertFalse(action['fire'])
        self.assertEqual(action['move_forward'], 0.0)
        self.assertEqual(action['move_right'], 0.0)

    def test_near_objective_takes_priority_without_reloading_or_switching(self):
        for side, use in (('attack', 'plant'), ('defend', 'defuse')):
            row = observation(map=copy.deepcopy(MAP), side=side)
            row['me'].update(x=10.0, vx=.78105, vy=0.0, active_slot=2,
                             carrying_bomb=side == 'attack')
            if side == 'defend':
                row['bomb'] = {'state': 'planted', 'position': [10.0, 0.0, 0.0],
                               'fuse_end_tick': 3000}
            row['me']['weapons'][1]['ammo'] = 0
            player = agent.Player()
            first = player.act(row)
            self.assertEqual(first['weapon_slot'], 2)
            self.assertEqual(first['move_forward'], 0.0)
            self.assertNotIn('reload', first)
            row.pop('map')
            row['me']['vx'] = 0.0
            row['match_tick'] += 3
            row['round_tick'] += 3
            self.assertEqual(player.act(row), {'weapon_slot': 2, 'use': use})


if __name__ == '__main__':
    unittest.main()
