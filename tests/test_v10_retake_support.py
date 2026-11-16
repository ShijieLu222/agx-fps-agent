"""Behavioral regressions for the first v9 match; no official physics claims."""
import copy
import math
import unittest

import agent
from test_agent import observation
from test_v2 import advance, enemy, contact
import test_v9_defense_combat as v9fixtures


def route_map(disconnected=False):
    # A is near in Euclidean distance but requires a long published detour.
    return {'sites':[{'id':'A','center':[5.,0.,0.],'radius':2.},
                     {'id':'B','center':[0.,15.,0.],'radius':2.}],
            'route_landmarks':{'A':[5.,0.,0.],'B':[0.,15.,0.]},
            'nav':{'nodes':[{'id':i,'position':p}for i,p in enumerate(
                [[0.,0.,0.],[5.,0.,0.],[0.,15.,0.],[0.,50.,0.]])],
                   'edges':[{'from':0,'to':2,'kind':'walk'},
                            {'from':0,'to':3,'kind':'walk'}]+([] if disconnected else
                            [{'from':3,'to':1,'kind':'walk'}])}}


class SupportTests(unittest.TestCase):
    def test_long_support_walk_completes_and_keeps_region_after_local_check(self):
        m={'sites':[{'id':'A','center':[0.,0.,0.],'radius':4.},
                    {'id':'B','center':[100.,0.,0.],'radius':4.}],
           'route_landmarks':{'A':[0.,0.,0.],'B':[100.,0.,0.]},
           'nav':{'nodes':[{'id':0,'position':[0.,0.,0.]},{'id':1,'position':[100.,0.,0.]}],
                  'edges':[{'from':0,'to':1,'kind':'walk'}]}}
        r=observation(map=m,side='defend',teammates=[{'seat':1,'alive':True,'zone':'B'}])
        r['me'].update(vx=0.,vy=0.,hp=100.,active_slot=1)
        p=agent.Player();p.act(r);r['teammates'][0]['alive']=False
        reached=False
        for i in range(1100):
            a=p.act(advance(r));me=r['me']
            if i==250:self.assertEqual(p.defend_site,1)  # Still travelling after 12 s.
            for _ in range(3):
                me['yaw']=agent.angle(me['yaw']+.05*a.get('yaw_rate',0))
                f,s=a.get('move_forward',0),a.get('move_right',0)
                me['vx']=4*(math.cos(me['yaw'])*f+math.sin(me['yaw'])*s)
                me['vy']=4*(-math.sin(me['yaw'])*f+math.cos(me['yaw'])*s)
                me['x']+=me['vx']/60;me['y']+=me['vy']/60
            reached |= math.dist([me['x'],me['y'],0.],[100.,0.,0.])<1.2
            self.assertEqual(p.defend_site,1)
        self.assertTrue(reached)
        self.assertIsNone(p.support_zone)
        self.assertGreater(r['me']['x'],98.)
        self.assertEqual(p.goal,[100.,0.,0.])

    def test_second_remote_death_cannot_immediately_reverse_committed_support(self):
        r=v9fixtures.InitiativeTests().support_row();r['teammates'].append({'seat':2,'alive':True,'zone':'A'})
        p=agent.Player();p.act(r);r['teammates'][0]['alive']=False;p.act(advance(r))
        r['teammates'][1]['alive']=False;p.act(advance(r))
        self.assertEqual(p.support_zone,'B');self.assertEqual(p.defend_site,1)
        p.act(advance(r,360))
        self.assertEqual(p.defend_site,1)  # Equal-distance stale second event is insufficient.

    def test_new_alert_after_completed_check_can_change_responsibility(self):
        r=v9fixtures.InitiativeTests().support_row();r['teammates'].append({'seat':2,'alive':True,'zone':'A'})
        p=agent.Player();p.act(r);r['teammates'][0]['alive']=False;p.act(advance(r))
        r['me']['y']=10.;p.act(advance(r));p.act(advance(r,363))
        self.assertIsNone(p.support_zone);self.assertEqual(p.defend_site,1)
        r['teammates'][1]['alive']=False;p.act(advance(r))
        self.assertEqual(p.defend_site,0)

    def test_no_support_from_unknown_region_or_attack_side_and_reset_at_round(self):
        for side,zone in [('attack','B'),('defend','unknown')]:
            r=v9fixtures.InitiativeTests().support_row();r['side']=side;r['teammates'][0]['zone']=zone
            p=agent.Player();p.act(r);r['teammates'][0]['alive']=False;p.act(advance(r))
            self.assertIsNone(p.defend_site)
        r=v9fixtures.InitiativeTests().support_row();p=agent.Player();p.act(r)
        r['teammates'][0]['alive']=False;p.act(advance(r));r['round_index']+=1;p.act(advance(r))
        self.assertIsNone(p.defend_site);self.assertFalse(p.support_alerts)


class SearchTests(unittest.TestCase):
    def row(self,**bomb):
        return observation(map=route_map(),side='defend',bomb={'state':'planted','fuse_end_tick':3000,**bomb})

    def test_unknown_site_uses_route_cost_not_euclidean_distance(self):
        r,p=self.row(),agent.Player();p.act(r)
        self.assertEqual(p.search_site,1);self.assertEqual(p.goal,[0.,15.,0.])
        costs=p.site_travel_costs(r,[0.,0.,0.]);self.assertGreater(costs[0],100.);self.assertEqual(costs[1],15.)

    def test_search_estimate_does_not_mutate_navigation_or_use_disconnected_target(self):
        r,p=self.row(),agent.Player();r['map']=route_map(True);p.act(r)
        saved=copy.deepcopy((p.path,p.path_nodes,p.nav_from,p.retry_node,p.failed_edges))
        costs=p.site_travel_costs(r,[0.,0.,0.])
        self.assertTrue(math.isinf(costs[0]));self.assertEqual(costs[1],15.)
        self.assertEqual((p.path,p.path_nodes,p.nav_from,p.retry_node,p.failed_edges),saved)

    def test_announced_site_and_known_bomb_override_inference(self):
        r,p=self.row(site='A'),agent.Player();p.act(r);self.assertEqual(p.goal,[5.,0.,0.])
        r['bomb']['position']=[0.,14.,0.];p.act(advance(r));self.assertEqual(p.goal,[0.,14.,0.])

    def test_stale_hint_cannot_override_route_or_become_exact_bomb_location(self):
        r,p=self.row(),agent.Player();p.act(r);p.search_site=None
        p.bomb_hint=(0,r['match_tick']-1201)
        self.assertEqual(p.unknown_bomb_goal(r,[0.,0.,0.]),[0.,15.,0.]);self.assertIsNone(p.known_bomb)

    def test_fuse_budget_overrides_soft_search_hint(self):
        r,p=self.row(fuse_end_tick=960),agent.Player();r['me']['has_kit']=True
        p.act(r);p.search_site=None;p.bomb_hint=(0,r['match_tick'])
        # Neither target is comfortably feasible here; still take the shortest route.
        self.assertEqual(p.unknown_bomb_goal(r,[0.,0.,0.]),[0.,15.,0.])

    def test_retake_discounts_danger_but_preserves_physical_failure_penalty(self):
        p=agent.Player();p.route_threats[('defend',0,1)]=(2,120.)
        self.assertEqual(p.edge_cost(0,1,10.,'defend',100,1),130.)
        self.assertLessEqual(p.edge_cost(0,1,10.,'defend',100,1,True),18.)
        p.failed_edges[(0,1)]=(500,70.)
        self.assertGreaterEqual(p.edge_cost(0,1,10.,'defend',100,1,True),80.)
        self.assertEqual(p.edge_cost(0,1,10.,'defend',600,3,True),10.)


class RetakePaceTests(unittest.TestCase):
    def test_repeated_flicker_cannot_refresh_slowdown_and_visible_still_fights(self):
        r=contact();r['side']='defend';r['map']=route_map()
        r['bomb']={'state':'planted','position':[0.,15.,0.],'fuse_end_tick':3000}
        p=agent.Player();moving_slow=0;visible_fights=0
        for i in range(240):
            r['visible']=[enemy(5,10.,0.)]if i%31==0 else []
            a=p.act(r)
            if r['visible']:
                self.assertEqual(a.get('move_forward'),0.);visible_fights+=1
            elif a.get('crouch'):moving_slow+=1
            self.assertNotIn('use',a)
            advance(r)
        self.assertEqual(moving_slow,6)
        self.assertGreater(visible_fights,5)

    def test_retake_sound_keeps_gaze_without_double_speed_penalty(self):
        r=observation(side='defend',map=route_map(),bomb={'state':'planted','position':[0.,15.,0.]},
                      sounds=[{'type':'footstep','direction':'right','distance_band':'near'}])
        p=agent.Player();a=p.act(r)
        self.assertFalse(a['walk']);self.assertFalse(a.get('crouch'))
        self.assertGreater(math.hypot(a['move_forward'],a['move_right']),.7)
        self.assertLess(a['yaw_rate'],0.)

    def test_retake_only_slows_close_corner_and_never_overrides_use_or_escape(self):
        r=observation(side='defend');p=agent.Player();p.retake=True
        a={'move_forward':.75,'move_right':0.,'walk':False}
        p.path=[[4.,0.,0.],[4.,8.,0.]]
        self.assertEqual(p.prepare_approach(r,a)['move_forward'],.75)
        p.path=[[1.5,0.,0.],[1.5,8.,0.]]
        self.assertAlmostEqual(p.prepare_approach(r,a)['move_forward'],.6)
        self.assertFalse(p.prepare_approach(r,a)['walk'])
        self.assertEqual(p.prepare_approach(r,dict(a,use='defuse')),dict(a,use='defuse'))
        p.escape_until=1000;self.assertEqual(p.prepare_approach(r,a),a)

    def test_long_clear_interval_allows_new_short_contact_budget(self):
        r=contact();r['side']='defend';r['map']=route_map()
        r['bomb']={'state':'planted','position':[0.,15.,0.]};p=agent.Player();p.act(r)
        r['visible']=[]
        for _ in range(8):p.act(advance(r))
        self.assertEqual(p.retake_slow_used,18)
        r['visible']=[enemy(5,10.,0.)];p.act(advance(r,303))
        r['visible']=[];a=p.act(advance(r));self.assertTrue(a['crouch'])


if __name__=='__main__':unittest.main()
