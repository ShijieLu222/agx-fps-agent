"""Reproduce v9 decisions and report actual objective/mobility evidence.

Public events are retrospective only. Detailed own samples remain private.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import action_changed,verify_export
from scripts.compare_players import events_for


def analyze(prefix,seat,source,bundle):
    root=ROOT/'data/private'
    _,meta,footer,_=verify_export(root/f'{prefix}-training.jsonl',seat,bundle)
    desc,events=events_for(prefix)
    assert desc['job_id']==meta['job_id']
    mod,sha=load_policy(source);p=mod._player
    mode={}
    for name in ('move','combat','guard_action','objective_action','unseen_damage_escape','lost_contact_hold'):
        original=getattr(p,name)
        def wrapper(*args,_name=name,_orig=original,**kw):
            out=_orig(*args,**kw)
            if out is not None:
                mode['method']=_name
                if _name=='move':mode['goal']=list(args[1])
                if _name=='combat':mode['selected_target']=p.target_seat
            return out
        setattr(p,name,wrapper)
    rounds={};rows=[];prev=None;count=mismatches=faults=0;plans=None
    for line in (root/f'{prefix}-training.jsonl').open():
        raw=json.loads(line);o=raw.get('observation')
        if not o:continue
        mode.clear();old_support=(p.support_zone,p.support_until)
        a=mod.act(o);count+=1;mismatches+=bool(action_changed(a,raw['action']));faults+=bool(raw.get('fault'))
        if p.nodes and plans is None:plans={'guard':p.guard_plans,'pressure':{str(k):v for k,v in p.pressure_points.items()},'sites':p.sites}
        if o['phase']!='live' or not o['me']['alive']:continue
        m=o['me'];tick=o['match_tick'];rn=o['round_index']+1;pos=[m[k] for k in ('x','y','z')]
        r=rounds.setdefault(rn,{'round':rn,'side':o['side'],'counts':Counter(),'start':tick,'support_events':[],'first':{},'distance_walked_m':0.,'stationary_max_seconds':0.,'stationary_since':tick})
        c=r['counts'];c['alive_calls']+=1;c[mode.get('method','other')]+=1
        c['visible_calls']+=bool(o['visible']);c['fire_requests']+=bool(a.get('fire'));c['carrying_calls']+=bool(m['carrying_bomb'])
        c['planted_calls']+=o['bomb']['state']=='planted';c['pressure_calls']+=p.pressure_index>=0 and o['bomb']['state']!='planted'
        c['support_calls']+=p.support_zone is not None and tick<p.support_until and o['bomb']['state']!='planted'
        for label,flag in [('visible',bool(o['visible'])),('fire',bool(a.get('fire'))),('planted_seen',o['bomb']['state']=='planted'),('defuse_request',a.get('use')=='defuse'),('plant_request',a.get('use')=='plant')]:
            if flag:r['first'].setdefault(label,tick)
        if old_support!=(p.support_zone,p.support_until) and p.support_zone is not None:
            r['support_events'].append({'tick':tick,'zone':p.support_zone,'position':pos,'until':p.support_until,'bomb_state':o['bomb']['state']})
        reqmove=math.hypot(a.get('move_forward',0),a.get('move_right',0))
        if prev and prev['round']==rn and tick-prev['tick']<=6:
            delta=math.dist(pos,prev['pos']);r['distance_walked_m']+=delta
            c['requested_move_calls']+=reqmove>.1
            c['requested_move_but_delta_under_1cm']+=reqmove>.1 and delta<.01
            if delta>=.01:r['stationary_since']=tick
            r['stationary_max_seconds']=max(r['stationary_max_seconds'],(tick-r['stationary_since'])/60)
            if m['hp']<prev['hp']:r['first'].setdefault('received',tick)
        gun=m['weapons'][m['active_slot']]
        sample={'round':rn,'side':o['side'],'tick':tick,'round_tick':o['round_tick'],'pos':pos,'yaw':m['yaw'],'pitch':m['pitch'],'hp':m['hp'],'speed':math.hypot(m['vx'],m['vy']),'grounded':m['grounded'],'carrying':m['carrying_bomb'],'kit':m['has_kit'],'bomb':o['bomb'],'alive_by_team':o.get('alive_by_team'),'visible':o['visible'],'sounds':o.get('sounds'),'action':a,'mode':dict(mode),'goal':p.goal,'guard_site':p.guard_site,'guard_choice':p.guard_choice,'guard_bad':sorted(p.guard_bad),'pressure_index':p.pressure_index,'pressure_bad':sorted(p.pressure_bad),'support_zone':p.support_zone,'support_until':p.support_until,'path':p.path[:3],'nav_from':p.nav_from,'path_nodes':p.path_nodes[:3],'retry':p.retry_node,'escapes':p.escapes,'recovery_len':len(p.recovery_trace),'retreat_len':len(p.retreat_path),'enemy_memory':p.enemy_memory,'gun':gun,'teammates':o.get('teammates')}
        rows.append(sample);prev=sample;r['last']=sample
    assert not mismatches
    for rn,r in rounds.items():
        es=[e for e in events if e['round_index']+1==rn]
        end=next(e for e in es if e['type']=='round_end')
        r['end_tick']=end['match_tick'];r['won']=end['round_winner']==seat//5;r['end_reason']=end['reason']
        r['kills']=sum(e['type']=='kill_reward' and e.get('killer')==seat for e in es)
        r['deaths']=sum(e['type']=='kill' and e.get('target')==seat for e in es)
        r['damage']=sum(e['hp_damage'] for e in es if e['type']=='hit' and e.get('player')==seat)
        r['shots']=sum(e['type']=='fire' and e.get('player')==seat for e in es)
        r['last_alive_seconds_before_end']=(end['match_tick']-r['last']['tick'])/60
        r['last']={k:r['last'][k] for k in ('tick','pos','hp','bomb','carrying','alive_by_team','action','mode','goal','pressure_index','guard_bad','escapes')}
        del r['stationary_since']
    (root/f'{prefix}-v9-samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    return {'format':'agx-fps-v9-postmatch.v1','job_id':meta['job_id'],'seat':seat,'source_sha256':sha,'bundle_sha256':bundle,'verified_training':{k:footer[k] for k in ('rows','raw_bytes','sha256')},'verified_events':desc['streams']['public_events'],'decisions':count,'mismatches':mismatches,'faults':faults,'plans':plans,'rounds':list(rounds.values()),'limitations':'Exact recorded-policy reproduction and observed movement, not counterfactual new outcomes. Teammate and enemy positions are limited to own observation; public events only label retrospective results.'}

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--prefix',required=True);a.add_argument('--seat',type=int,required=True);a.add_argument('--source',required=True);a.add_argument('--bundle',required=True);a.add_argument('--output',required=True);v=a.parse_args()
    x=analyze(v.prefix,v.seat,v.source,v.bundle);Path(v.output).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    print('decisions',x['decisions'],'mismatches',x['mismatches'],'faults',x['faults'])
    for r in x['rounds']:print(r['round'],r['side'],'W' if r['won'] else 'L',r['end_reason'],f"{r['kills']}/{r['deaths']}",round(r['damage']),r['shots'],'still',round(r['stationary_max_seconds'],1),'last alive',round(r['last_alive_seconds_before_end'],1),'lastpos',[round(v,1) for v in r['last']['pos']],'bomb',r['last']['bomb'],'goal',r['last']['goal'])
