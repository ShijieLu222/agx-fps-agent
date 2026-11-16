"""Behavioral regressions from v8 Dust2. Fixtures are not the AGX simulator."""
import copy
import math
import unittest

import agent
from test_agent import MAP, observation
from test_v2 import advance, contact, enemy


def station_map():
    return {
        'sites': [{'id': 'B', 'center': [10., 10., 0.], 'radius': 4.}],
        'spawns': {'CT': [{'position': [0., 0., 0.]}],
                   'T': [{'position': [20., 0., 0.]}]},
        'bot_waypoints': {'holdB': [[10., 7., 0.], [7., 10., 0.]],
                          'watchB_CT': [[20., 7., 1.18]]},
        'nav': {'nodes': [{'id': i, 'position': p} for i, p in enumerate(
            [[0., 0., 0.], [10., 0., 0.], [10., 7., 0.], [10., 10., 0.], [7., 10., 0.]])],
            'edges': [{'from': a, 'to': b, 'kind': 'walk'} for a, b in [(0,1),(1,2),(2,3),(3,4)]]}}


class GuardTests(unittest.TestCase):
    def row(self):
        r = observation(map=station_map(), side='defend', seat=6, team=1)
        r['me'].update(hp=100., vx=0., vy=0., active_slot=1)
        return r

    def test_guard_goal_is_separate_from_bomb_center_and_different_seats_spread(self):
        goals = []
        for seat in (5,7):
            r, p = self.row(), agent.Player()
            r['seat'] = seat
            p.act(r)
            goals.append(p.goal)
            self.assertGreater(math.dist(p.goal, [10.,10.,0.]), 2.)
        self.assertNotEqual(goals[0], goals[1])

    def test_closed_loop_reaches_station_checks_entrance_then_pushes(self):
        r, p = self.row(), agent.Player()
        for _ in range(300):
            a = p.act(r)
            m = r['me']
            if a.get('crouch') and not a.get('move_forward') and not a.get('move_right'):
                break
            # Three rotation/motion substeps, no collisions; station traversal
            # remains a surrogate and is not claimed as a platform victory.
            for _ in range(3):
                m['yaw'] = agent.angle(m['yaw']+.05*a.get('yaw_rate',0))
                m['pitch'] += a.get('pitch_rate',0)/30
                f, side = a.get('move_forward',0), a.get('move_right',0)
                m['vx'] = 4*(math.cos(m['yaw'])*f+math.sin(m['yaw'])*side)
                m['vy'] = 4*(-math.sin(m['yaw'])*f+math.cos(m['yaw'])*side)
                m['x'] += m['vx']/60; m['y'] += m['vy']/60
            advance(r)
        pos = [r['me'][k] for k in ('x','y','z')]
        self.assertLess(math.dist(pos,p.guard_plans[0]['stations'][p.guard_choice]),1.3)
        for _ in range(29):
            a = p.act(r)
            self.assertFalse(a.get('fire',False))
            self.assertEqual(a.get('move_forward'),0.)
            m['yaw'] = agent.angle(m['yaw']+.15*a['yaw_rate'])
            advance(r)
        self.assertLess(abs(agent.angle(r['me']['yaw']-math.atan2(-(7-r['me']['y']),20-r['me']['x']))),.01)
        a=p.act(advance(r,6))
        self.assertGreater(math.hypot(a.get('move_forward',0),a.get('move_right',0)),.1)
        self.assertGreaterEqual(p.pressure_index,0)

    def test_same_xy_watch_point_never_makes_guard_look_straight_up(self):
        r,p=self.row(),agent.Player();p.act(r)
        r['me'].update(x=10.,y=7.)
        plan={'watch': [[10.,7.,1.18],[20.,7.,1.18]]}
        a=p.guard_action(r,{'weapon_slot':1},plan)
        self.assertAlmostEqual(a['yaw_rate'],0.)
        self.assertLess(abs(a['pitch_rate']),.6)

    def test_disconnected_and_inside_box_stations_are_excluded(self):
        r,p=self.row(),agent.Player();m=r['map']
        m['nav']['nodes'] += [{'id':9,'position':[10.,7.2,0.]},{'id':10,'position':[10.,7.5,0.]}]
        m['nav']['edges'].append({'from':9,'to':10,'kind':'walk'})
        m['boxes']=[{'x':7.,'y':10.,'z':0.,'w':2.,'d':2.,'h':3.}]
        p.load_map(m)
        self.assertEqual(p.guard_plans[0]['stations'],[[10.,7.,0.]])

    def test_failed_station_switches_to_alternative_not_same_point_forever(self):
        r,p=self.row(),agent.Player();p.act(r)
        first=list(p.goal)
        for _ in range(70):p.act(advance(r))
        self.assertNotEqual(p.goal,first)
        self.assertTrue(p.guard_bad)

    def test_guard_never_preempts_defuse_or_current_visible_combat(self):
        r,p=self.row(),agent.Player();p.act(r)
        r['me'].update(x=10.,y=7.,pitch=math.atan2(1.18-1.62,10.))
        r['visible']=[enemy(0,20.,7.)]
        a=p.act(advance(r));self.assertTrue(a['fire'])
        r['bomb']={'state':'planted','position':[10.,7.,0.],'fuse_end_tick':3000}
        a=p.act(advance(r));self.assertNotIn('use',a)
        self.assertEqual(a.get('move_forward'),0.)  # Pistol release frame.
        self.assertEqual(p.target_seat,0)
        r['visible']=[]
        a=p.act(advance(r));self.assertEqual(a.get('use'),'defuse')

    def test_guard_station_and_retreat_memory_reset_on_map_round_change(self):
        for change in ({'round_index':1},{'map_id':'new'}):
            r,p=self.row(),agent.Player();p.act(r)
            p.guard_bad.add(0);p.head_targets.add(0)
            r.update(change);p.act(advance(r))
            self.assertFalse(p.guard_bad)
            self.assertFalse(p.head_targets)


class InitiativeTests(unittest.TestCase):
    def support_row(self):
        m=copy.deepcopy(MAP)
        for s,name in zip(m['sites'],('A','B')):s['id']=name
        m['route_landmarks']={'A':[10.,0.,0.],'B':[0.,10.,0.]}
        return observation(map=m,side='defend',seat=0,
                           teammates=[{'seat':1,'alive':True,'zone':'B'}])

    def test_teammate_death_support_is_new_bounded_and_not_hidden_enemy_track(self):
        r,p=self.support_row(),agent.Player();p.act(r)
        self.assertEqual(p.goal,[10.,0.,0.])
        r['teammates'][0]['alive']=False
        p.act(advance(r));self.assertEqual(p.goal,[0.,10.,0.])
        until=p.support_until
        for _ in range(10):p.act(advance(r))
        self.assertEqual(p.support_until,until)
        self.assertIsNone(p.enemy_memory)
        p.act(advance(r,720));self.assertEqual(p.goal,[10.,0.,0.])

    def test_initial_dead_unknown_zone_and_round_reset_do_not_invent_support(self):
        for zone in ('B','secret'):
            r,p=self.support_row(),agent.Player()
            r['teammates'][0].update(alive=False,zone=zone)
            p.act(r);self.assertIsNone(p.support_zone)
        r,p=self.support_row(),agent.Player();p.act(r)
        r['teammates'][0]['alive']=False;p.act(advance(r))
        r['round_index']+=1;p.act(advance(r));self.assertIsNone(p.support_zone)

    def test_planted_bomb_immediately_overrides_support_then_fight_then_defuse(self):
        r,p=self.support_row(),agent.Player();r['me'].update(active_slot=1,vx=0.,vy=0.)
        p.act(r);r['teammates'][0]['alive']=False;p.act(advance(r))
        r['bomb']={'state':'planted','position':[10.,0.,0.],'fuse_end_tick':700}
        p.act(advance(r));self.assertEqual(p.goal,[10.,0.,0.])
        r['visible']=[enemy(5,80.,0.)]
        r['me']['pitch']=math.atan2(1.18-1.62,80.)
        a=p.act(advance(r));self.assertTrue(a['fire'])
        self.assertEqual(a.get('move_forward'),0.)
        # Even an expired planning deadline cannot select the old blind rush.
        a=p.act(advance(r,150));self.assertEqual(a.get('move_forward'),0.)
        self.assertNotIn('use',a)
        r['visible']=[];r['me']['x']=10.
        a=p.act(advance(r));self.assertEqual(a.get('use'),'defuse')

    def test_visible_enemy_prevents_empty_gun_defuse_and_requests_reload(self):
        r=self.support_row();r['me'].update(x=10.,active_slot=1,vx=0.,vy=0.)
        r['me']['weapons'][1]['ammo']=0
        r['visible']=[enemy(5,15.,0.)]
        r['bomb']={'state':'planted','position':[10.,0.,0.]}
        a=agent.Player().act(r)
        self.assertNotIn('use',a);self.assertTrue(a.get('reload'))

    def test_assault_enters_site_then_continues_clearing_and_stops_for_contact(self):
        r=observation(map=station_map(),side='attack')
        r['me'].update(x=10.,y=10.,vx=0.,vy=0.,active_slot=1)
        p=agent.Player();p.act(r)
        a=p.act(advance(r,48));self.assertGreaterEqual(p.pressure_index,0)
        self.assertNotEqual(p.goal,[10.,10.,0.])
        self.assertGreater(math.hypot(a.get('move_forward',0),a.get('move_right',0)),.1)
        r['visible']=[enemy(5,20.,10.)];r['me']['pitch']=math.atan2(1.18-1.62,10.)
        a=p.act(advance(r));self.assertTrue(a['fire'])
        self.assertEqual(a.get('move_forward'),0.)
        r['visible']=[]
        for _ in range(15):a=p.act(advance(r))
        self.assertGreater(math.hypot(a.get('move_forward',0),a.get('move_right',0)),.1)

    def test_guard_not_counted_stuck_while_intentionally_forward(self):
        r=GuardTests().row();p=agent.Player();p.act(r)
        p.pressure_index=0;p.guard_started=0;p.leg_progress_tick=0
        p.last_move_tick=r['match_tick']
        p.guard_goal(r,0,p.guard_plans[0])
        self.assertFalse(p.guard_bad)

    def test_forward_failure_retires_target_and_changes_goal(self):
        r=GuardTests().row();p=agent.Player();p.act(r)
        p.pressure_index=0;p.pressure_started=0;p.leg_progress_tick=0
        p.last_move_tick=r['match_tick']
        old=p.pressure_points[('defend',0)][0]
        g=p.pressure_goal(r,0,[10.,7.,0.])
        self.assertIn(0,p.pressure_bad);self.assertNotEqual(g,old)

    def test_postplant_watches_retaking_approach_instead_of_continuous_spin(self):
        r=observation(map=station_map(),side='attack',bomb={'state':'planted','position':[10.,10.,0.]})
        r['me'].update(x=10.,y=10.,vx=0.,vy=0.,yaw=3*math.pi/4)
        p=agent.Player()
        for _ in range(100):
            a=p.act(r)
            r['me']['yaw']=agent.angle(r['me']['yaw']+.15*a['yaw_rate'])
            r['me']['pitch']+=.1*a['pitch_rate']
            advance(r)
        self.assertLess(abs(a['yaw_rate']),.001)
        self.assertEqual(a['move_forward'],0.)


class PersistentRetreatTests(unittest.TestCase):
    def setUp(self):
        self.p=agent.Player();self.r=observation(map=copy.deepcopy(MAP),side='defend')
        self.r['me'].update(hp=100.,vx=0.,vy=0.,active_slot=1)

    def put(self,x,y,**changes):
        self.r['me'].update(x=x,y=y,**changes)
        a=self.p.act(self.r);advance(self.r);return a

    def test_corner_path_survives_one_minute_idle_then_immediately_reverses(self):
        for i in range(11):self.put(i*.2,0.)
        for i in range(1,11):self.put(2.,i*.2)
        count=len(self.p.breadcrumbs)
        for _ in range(1200):self.put(2.,2.)
        self.assertEqual(len(self.p.breadcrumbs),count)
        a=self.put(2.,2.,hp=70.)
        self.assertFalse(a.get('fire',False))
        self.assertAlmostEqual(a['move_forward'],0.)
        self.assertLess(a['move_right'],-.8)
        self.assertEqual(self.p.retreat_path[0],[2.,1.8,0.])
        # Following samples takes the actual corner, not a diagonal shortcut.
        for y in (1.8,1.6,1.4,1.2,1.,.8,.6,.4,.2,0.):
            a=self.put(2.,y)
        self.assertLess(a['move_forward'],-.8)
        self.assertAlmostEqual(a['move_right'],0.)

    def test_old_trace_does_not_survive_jump_or_teleport(self):
        for discontinuity in ('jump','teleport'):
            with self.subTest(discontinuity=discontinuity):
                self.setUp()
                for i in range(10):self.put(i*.2,0.)
                for _ in range(300):self.put(1.8,0.)
                if discontinuity=='jump':
                    self.put(1.8,0.,grounded=False)
                    a=self.put(1.8,0.,hp=50.,grounded=True)
                else:a=self.put(10.,10.,hp=50.)
                self.assertEqual(self.p.retreat_path,[])


class ClockAndAimTests(unittest.TestCase):
    def test_head_aim_acquires_and_fires_in_turning_crouch_substep_loop(self):
        r=contact();r['visible'][0]['parts']=['head','chest','abdomen']
        r['me']['weapons'][0].update(id='m4a1',inaccuracy=.005)
        r['me']['yaw']=.3
        p=agent.Player();head_shots=[];ready=0
        for _ in range(30):
            r['me']['shot_ready']=ready
            a=p.act(r)
            if a.get('fire') and p.head_targets:head_shots.append(r['match_tick'])
            for t in range(r['match_tick']+1,r['match_tick']+4):
                r['me']['yaw']=agent.angle(r['me']['yaw']+.05*a.get('yaw_rate',0))
                r['me']['pitch']+=a.get('pitch_rate',0)/30
                r['me']['crouch']=a.get('crouch',False)
                if a.get('fire') and t>=ready:ready=t+6
            advance(r)
        self.assertGreater(len(head_shots),5)
        self.assertLess(head_shots[0],624)  # Acquire within 0.4 s in this fixture.

    def test_substep_weapon_model_emits_six_tick_rifle_intervals(self):
        # Observed contract: observation at t; action executes on t+1..t+3.
        # Weapon fixture intentionally omits ballistics/damage, but schedules
        # shot readiness independently of the policy under test.
        for gunid,interval in [('m4a1',6),('ak47',6)]:
            r=contact();r['me']['weapons'][0]['id']=gunid
            r['me']['crouch']=True;r['me']['pitch']=math.atan2(1.18-.95,10.)
            p=agent.Player();shots=[];ready=0
            for _ in range(20):
                r['me']['shot_ready']=ready;r['me']['weapons'][0]['shot_ready']=ready
                a=p.act(r)
                for t in range(r['match_tick']+1,r['match_tick']+4):
                    if a.get('fire') and t>=ready:
                        shots.append(t);ready=t+interval
                advance(r)
            self.assertGreaterEqual(len(shots),10)
            self.assertEqual(set(b-a for a,b in zip(shots,shots[1:])),{6})

    def test_execution_boundary_never_relaxes_reload_switch_bool_or_ammo(self):
        for condition in ('reload','switch','bool','empty','airborne','fast','unseen','unconfirmed_head'):
            with self.subTest(condition=condition):
                r=contact();r['me']['weapons'][0]['shot_ready']=r['match_tick']+1
                if condition=='reload':r['me']['weapons'][0]['reload_until']=r['match_tick']+1
                if condition=='switch':r['me']['switch_until']=r['match_tick']+1
                if condition=='bool':r['me']['shot_ready']=False
                if condition=='empty':r['me']['weapons'][0]['ammo']=0
                if condition=='airborne':r['me']['grounded']=False
                if condition=='fast':r['me']['vx']=2.
                if condition=='unseen':r['visible']=[]
                if condition=='unconfirmed_head':r['visible'][0]['parts']=['head']
                self.assertFalse(agent.Player().act(r).get('fire',False))

    def test_real_round6_cooldown_boundary_is_no_longer_skipped(self):
        r=contact();r['match_tick']=19545
        r['me']['weapons'][0].update(id='m4a1',shot_ready=19546)
        r['me']['shot_ready']=19546
        self.assertTrue(agent.Player().act(r)['fire'])
        r['me']['weapons'][0]['shot_ready']=19547
        self.assertFalse(agent.Player().act(r)['fire'])

    def test_stable_close_head_is_selected_but_far_or_unsteady_stays_body(self):
        for dist,speed,expected_z in [(20.,0.,1.64),(100.,0.,1.18),(20.,2.,1.18)]:
            r=contact();r['visible']=[enemy(5,dist,0.)];r['visible'][0]['parts']=['head','chest']
            r['me']['vx']=speed;p=agent.Player()
            for _ in range(4):p.act(r);advance(r)
            p.observe_targets(r)
            target=p.aim_point(r['visible'][0],r['me'],r['me']['weapons'][0],r['match_tick'])
            self.assertAlmostEqual(target[2],expected_z)
            r['visible'][0]['parts']=['chest'];p.observe_targets(r)
            self.assertEqual(p.aim_point(r['visible'][0],r['me'],r['me']['weapons'][0],r['match_tick'])[2],1.18)
            self.assertNotIn(5,p.head_targets)

    def test_distant_target_requires_smaller_angular_error(self):
        r=contact();r['visible']=[enemy(5,100.,0.)]
        r['me'].update(yaw=.01,pitch=math.atan2(1.18-1.62,100.))
        self.assertFalse(agent.Player().act(r)['fire'])
        r['me']['yaw']=0.
        self.assertTrue(agent.Player().act(r)['fire'])


class InformationAndEconomyTests(unittest.TestCase):
    def test_right_sound_prioritizes_right_view_and_slows_without_firing(self):
        r=observation(map=copy.deepcopy(MAP))
        r['sounds']=[{'type':'footstep','direction':'right','distance_band':'mid'}]
        p=agent.Player();a=p.act(r)
        self.assertLess(a['yaw_rate'],0.)
        self.assertLess(a['move_forward'],.5)
        self.assertFalse(a.get('fire',False))
        yaw=p.sound_yaw
        r['me']['yaw']=-.3;p.act(advance(r))
        self.assertEqual(p.sound_yaw,yaw)
        r['sounds']=[];p.act(advance(r,31));self.assertIsNone(p.sound_yaw)

    def test_unknown_self_far_sounds_cannot_create_fire_or_false_target(self):
        for direction,band in [('unknown','mid'),('self','self'),('left','far')]:
            r=observation(map=copy.deepcopy(MAP),sounds=[{'type':'footstep','direction':direction,'distance_band':band}])
            p=agent.Player();a=p.act(r)
            self.assertIsNone(p.sound_yaw);self.assertFalse(a.get('fire',False))
            self.assertGreater(a['move_forward'],.7)

    def test_slowdown_is_local_to_corner_and_never_stops_forward_progress(self):
        p=agent.Player();p.path=[[3.,0.,0.],[3.,10.,0.]]
        r=observation();r['me'].update(vx=4.,vy=0.)
        a=p.prepare_approach(r,{'move_forward':.75})
        self.assertGreater(a['move_forward'],0.)
        self.assertLess(a['move_forward'],.5)
        p.path=[[12.,0.,0.],[12.,10.,0.]]
        self.assertEqual(p.prepare_approach(r,{'move_forward':.75}),{'move_forward':.75})

    def test_helmet_and_bounded_force_buy_follow_own_budget(self):
        r=observation(phase='buy');r['me'].update(armor=100,cash=1000)
        r['me']['weapons'][0]={'id':'ak47','ammo':30,'reserve':90}
        self.assertEqual(agent.Player().act(r),{'buy':'helmet'})
        r['me']['helmet']=True;self.assertEqual(agent.Player().act(r),{})
        for side,cash,gun in [('attack',2450,'galilar'),('defend',1900,'mp9')]:
            r=observation(phase='buy',side=side,loss_level=2);r['me']['cash']=cash
            self.assertEqual(agent.Player().act(r),{'buy':gun})
            r['loss_level']=0;self.assertNotEqual(agent.Player().act(r).get('buy'),gun)

    def test_legal_carrier_site_commitment_without_hidden_coordinate_inference(self):
        r=observation(map=copy.deepcopy(MAP),bomb={'state':'held','carrier':1,'position':[0.,9.,0.]})
        p=agent.Player();p.act(r);self.assertEqual(p.goal,[0.,10.,0.])
        r['bomb']={'state':'unknown'};p=agent.Player();p.act(r)
        self.assertEqual(p.goal,[10.,0.,0.])


if __name__=='__main__':unittest.main()
