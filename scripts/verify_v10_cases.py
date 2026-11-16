"""Verify v10 behavior on the checksum-verified v9 first match.

Historical observations do not react to new actions. These are issue and
interface regressions, never counterfactual wins, movement or damage.
"""
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import action_changed, verify_export
from scripts.replay_check import validate_action


def main():
    training=ROOT/'data/private/385f-training.jsonl'
    bundle='1abc777347a92c602519c68c1b4942e77a64f346ea57f180cbf6e6274c8792db'
    _,meta,footer,_=verify_export(training,9,bundle)
    old,oldsha=load_policy(ROOT/'data/private/agx-v9-agent.py')
    new,newsha=load_policy(ROOT/'agent.py')
    cases={};count=0;changed=0;oldslow=newslow=0;attack_changes=0
    selected={15711,38961,47166,12285,12378,12849}
    for line in training.open():
        row=json.loads(line);o=row.get('observation')
        if o is None:continue
        a,b=old.act(o),new.act(o);count+=1
        assert not action_changed(a,row['action'])
        assert not validate_action(b)
        json.dumps(b,allow_nan=False)
        diff=action_changed(a,b);changed+=diff
        if o['side']=='attack':attack_changes+=diff
        tick=o['match_tick']
        if o['round_index']==2 and 12285<=tick<=12933 and o['me']['alive']:
            oldslow+=bool(a.get('crouch')) and math.hypot(a.get('move_forward',0),a.get('move_right',0))>.1
            newslow+=bool(b.get('crouch')) and math.hypot(b.get('move_forward',0),b.get('move_right',0))>.1
        if tick in selected:
            p=new._player
            cases[tick]={'round':o['round_index']+1,'position':[o['me'][k]for k in ('x','y','z')],
                'legal_bomb':o['bomb'],'v9':a,'v10':b,'v9_goal':old._player.goal,'v10_goal':p.goal,
                'v10_defend_site':p.defend_site,'v10_support_until':p.support_until,'v10_search_site':p.search_site}
            if tick==47166:cases[tick]['v10_site_costs']=p.site_travel_costs(o,cases[tick]['position'])
    assert cases[15711]['v10_defend_site']==1
    assert cases[38961]['v10_defend_site']==1
    assert cases[47166]['v9_goal']==[112.,90.,2.4]
    assert cases[47166]['v10_search_site']==1
    assert oldslow==203 and newslow<=6
    assert attack_changes==0, 'This iteration must preserve v9 attack actions on the recorded inputs'
    out={'format':'agx-fps-v10-issue-regressions.v1','job_id':meta['job_id'],
         'baseline_source_sha256':oldsha,'candidate_source_sha256':newsha,'decisions':count,
         'verified_training':{k:footer[k]for k in ('rows','raw_bytes','sha256')},
         'changed_actions':changed,'attack_changed_actions':attack_changes,
         'round3_moving_crouch_callbacks':{'v9':oldslow,'v10':newslow},'cases':cases,
         'assertions_passed':['All recorded v9 actions reproduced','All v10 actions JSON/schema valid',
           'Round 4 and 9 support expiry does not revert responsibility to A',
           'Round 10 unknown-bomb search selects B by current graph cost',
           'Round 3 repeated flicker no longer maintains crouch travel',
           'Recorded attack-side actions unchanged'],
         'limitations':'Fixed historical inputs and simplified unit-test fixtures, not new match outcomes or full collision/physics validation.'}
    (ROOT/'experiments/v10-issue-regressions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k]for k in ('decisions','changed_actions','attack_changed_actions','round3_moving_crouch_callbacks','assertions_passed')},ensure_ascii=False))


if __name__=='__main__':main()
