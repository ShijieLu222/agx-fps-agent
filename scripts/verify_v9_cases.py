"""Verify v9 issue regressions on authenticated, checksum-verified v8 inputs.

The changed strategy does not advance physics or generate new observations.
Shot timing is separately exercised by the 60 Hz fixture in the unit tests.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import action_changed, verify_export
from scripts.replay_check import validate_action


def verify(training, baseline, candidate):
    bundle = 'd948b68705b72dc4c3931d1e4fd36b3bffeb683ebd6b9f85abf75dfbe5fc5fa0'
    _, meta, footer, _ = verify_export(training, 6, bundle)
    old, old_hash = load_policy(baseline)
    new, new_hash = load_policy(candidate)
    selected = {5184, 8931, 12999, 13005, 15945, 19545, 22302, 25473, 28125}
    cases, count, plans = [], 0, None
    for line in Path(training).open():
        row = json.loads(line)
        o = row.get('observation')
        if o is None:
            continue
        a, b = old.act(o), new.act(o)
        if action_changed(a, row['action']):
            raise ValueError('v8 baseline differs from recorded action')
        if validate_action(b):
            raise ValueError('invalid v9 action')
        json.dumps(b, allow_nan=False)
        count += 1
        if plans is None and new._player.guard_plans:
            plans = new._player.guard_plans
        if o['match_tick'] in selected:
            m=o['me']
            cases.append({'tick':o['match_tick'], 'round':o['round_index']+1,
                'hp':m['hp'], 'visible_count':len(o['visible']),
                'sounds':o['sounds'], 'v8':a, 'v9':b,
                'retained_points':len(new._player.breadcrumbs),
                'retreat_points':len(new._player.retreat_path),
                'guard_choice':new._player.guard_choice,
                'head_selected':sorted(new._player.head_targets)})
    by_tick={c['tick']:c for c in cases}
    retreat = by_tick[5184]
    assert math.hypot(retreat['v8'].get('move_forward',0),retreat['v8'].get('move_right',0)) == 0
    assert math.hypot(retreat['v9'].get('move_forward',0),retreat['v9'].get('move_right',0)) > .5
    assert retreat['retreat_points'] and not retreat['v9'].get('fire',False)
    assert by_tick[12999]['v8']['yaw_rate'] > 0 and by_tick[12999]['v9']['yaw_rate'] < 0
    return {'format':'agx-fps-v9-issue-regressions.v1', 'job_id':meta['job_id'],
        'baseline_source_sha256':old_hash,'candidate_source_sha256':new_hash,
        'decisions':count,'verified_training':{k:footer[k] for k in ('rows','raw_bytes','sha256')},
        'guard_plans':plans,'cases':cases,
        'assertions_passed':['All recorded v8 actions reproduced',
         'All v9 actions JSON/schema valid','Round 2 first unseen hit requests witnessed retreat after long stop',
         'Round 4 right-side footsteps turn the view right before the first hit'],
        'limitations':'Fixed historical observations are not new damage, survival or match outcomes. '
         'New navigation and head aiming change subsequent observations in a real match.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,default=ROOT/'data/private/e2e4-training.jsonl')
    p.add_argument('--baseline',type=Path,default=ROOT/'data/private/agx-v8-agent.py')
    p.add_argument('--candidate',type=Path,default=ROOT/'agent.py')
    p.add_argument('--output',type=Path,default=ROOT/'experiments/v9-issue-regressions.json')
    a=p.parse_args();result=verify(a.training,a.baseline,a.candidate)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'decisions':result['decisions'],'passed':result['assertions_passed']}))

if __name__=='__main__':main()
