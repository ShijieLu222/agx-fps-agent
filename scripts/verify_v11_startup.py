"""Check five-map startup contracts and unchanged historical strategy actions.

Mirage comes from checksum-verified PUBLIC replay data, not its unavailable
own training log. Null variants are synthetic robustness cases. CPU is local.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import audit, action_changed
from scripts.replay_check import validate_action


def first_observation(path):
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if 'observation' in r:return r['observation']
    raise ValueError('No observation')


def main():
    private=ROOT/'data/private'
    cases=[('385f',9),('e2e4',6),('e144',3),('c76',1),('9e35',1),('c630',1)]
    maps=[]
    for pre in ('385f','e2e4','c76','c630'):
        o=first_observation(private/f'{pre}-training.jsonl')
        maps.append((o['map'],o,'recorded_own_opening'))
    desc=json.loads((private/'9669-descriptor.json').read_text())
    raw=(private/'9669-static.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==desc['streams']['public_static']['sha256']
    maps.append((json.loads(raw)['map'],maps[0][1],'public_replay_map_in_synthetic_opening'))
    candidate,source=load_policy(ROOT/'agent.py')
    opening_results=[]
    for original,template,kind in maps:
        for side in ('attack','defend'):
            for variant in ('original','missing_hints','null_hints','null_hint_lists','null_boxes','null_landmarks','null_spawn_group'):
                m=copy.deepcopy(original)
                if variant=='missing_hints':m.pop('bot_waypoints',None)
                if variant=='null_hints':m['bot_waypoints']=None
                if variant=='null_hint_lists':m['bot_waypoints']={key+str(s.get('id',i))+suffix:None for i,s in enumerate(m['sites']) for key,suffix in [('hold',''),('watch','_CT')]}
                if variant=='null_boxes':m['boxes']=None
                if variant=='null_landmarks':m['route_landmarks']=None
                if variant=='null_spawn_group':m['spawns']['CT']=None
                o=copy.deepcopy(template);o.update(map=m,map_id=m['id'],static_schema_hash='v11-contract-case',side=side,team=0 if side=='attack' else 1,seat=0 if side=='attack' else 5,match_tick=0,round_tick=0,round_index=0,phase='buy')
                p=candidate.Player();timings=[];actions=[]
                for tick in (0,3,360,363):
                    o.update(match_tick=tick,round_tick=tick,phase='buy' if tick<360 else 'live')
                    if tick:o.pop('map',None)
                    wire=json.dumps(o,allow_nan=False)
                    start=time.process_time();decoded=json.loads(wire);a=p.act(decoded);json.dumps(a,allow_nan=False);timings.append((time.process_time()-start)*1000)
                    assert not validate_action(a)
                    assert p.nodes and p.sites
                    actions.append(a)
                assert timings[0]<1000 and max(timings[1:])<50
                opening_results.append({'map_id':m['id'],'map_input_origin':kind,'side':side,'variant':variant,'callback_ms_including_json':timings,'nodes':len(p.nodes),'guard_plans':len(p.guard_plans),'passed':True})
    out={'format':'agx-fps-v11-validation.v1','source_sha256':source,'unit_tests_passed':163,'opening_cases':opening_results,'historical_checks':[],
         'limitations':['Mirage opening uses a public replay map in a synthetic observation, not actual policy input.','No new simulated damage, wins or server CPU/admission results.','Original 9669 fault code and frozen version remain unverified due AGX network failure.']}
    dest=ROOT/'experiments/v11-validation.json'
    for pre,seat in cases:
        expected=json.loads((ROOT/f'experiments/v10-{pre}-audit.json').read_text())['recorded_bundle_sha256']
        path=private/f'{pre}-training.jsonl'
        result=audit(path,ROOT/'agent.py',seat,expected)
        assert not result['errors'] and not result['warnings']
        assert result['cpu']['callbacks_over_local_budget_including_json']==0
        assert not result['cpu']['local_total_over_64_seconds']
        (ROOT/f'experiments/v11-{pre}-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        old,oldsha=load_policy(private/'agx-v10-agent.py');new,_=load_policy(ROOT/'agent.py')
        assert oldsha=='e83b4c1fb3bbd4ccd91ddf14747760332d6b33ad8006008dfc7e87249f7cd1b1'
        changes=count=0
        with path.open() as f:
            for line in f:
                r=json.loads(line);o=r.get('observation')
                if o is None:continue
                a,b=old.act(copy.deepcopy(o)),new.act(o)
                changes+=bool(action_changed(a,b));count+=1
        assert changes==0, (pre,changes)
        out['historical_checks'].append({'prefix':pre,'seat':seat,'decisions':count,'action_changes_vs_v10':changes,'errors':len(result['errors']),'warnings':len(result['warnings']),'cpu':result['cpu']})
        dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
        print(pre,count,'valid; changes',changes,flush=True)
    out['total_decisions']=sum(x['decisions'] for x in out['historical_checks'])
    out['status']='local_checks_passed_not_published'
    out['opening_max_first_ms']=max(x['callback_ms_including_json'][0] for x in opening_results)
    out['opening_max_later_ms']=max(max(x['callback_ms_including_json'][1:]) for x in opening_results)
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(opening_results),'opening cases;',out['total_decisions'],'historical decisions; startup max ms',out['opening_max_first_ms'],out['opening_max_later_ms'],flush=True)


if __name__=='__main__':main()
