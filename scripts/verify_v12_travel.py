"""Verify recorded v11 actions and audit v12 on fixed legal observations.

No game physics is advanced. Requested movement changes are not saved time,
new survival, damage or wins. Raw observations stay in ignored data/private.
"""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import action_changed, audit, verify_export
from scripts.compare_players import events_for
from scripts.replay_check import validate_action

V11_BUNDLE = 'adcccc500476286c4fda1ab6187dcb7d35b878ef330ec7bb5bd09beeb08657de'
V11_SOURCE = 'e044b26e3de850493d231a753df71d4435277030468ebb43b00fa8c0d97f7163'
RECENT = [('11dd', 7), ('5538', 2), ('eb98', 7)]


def compare(prefix, seat):
    path = ROOT / f'data/private/{prefix}-training.jsonl'
    _, metadata, footer, _ = verify_export(path, seat, V11_BUNDLE)
    desc, _ = events_for(prefix)
    assert desc['job_id'] == metadata['job_id']
    old, old_sha = load_policy(ROOT / 'data/private/agx-v11-agent.py')
    new, new_sha = load_policy(ROOT / 'agent.py')
    assert old_sha == V11_SOURCE
    totals, fields = Counter(), Counter()
    movement = {}
    retargets = []
    current = {}
    original_move = new._player.move

    def move(o, goal, action, stop_at_goal=True):
        p = new._player
        before = copy.deepcopy((p.goal, p.path, p.path_nodes, p.nav_from))
        result = original_move(o, goal, action, stop_at_goal)
        # Report only target changes that actually preserve the active prefix.
        if (before[0] != goal and before[1] and len(before[2]) >= 2
                and before[2][-1] is None and before[2][-2] in p.nodes
                and before[2][-2] == min(p.nodes,
                    key=lambda n: math.dist(goal, p.nodes[n]))
                and p.path and before[1][0] == p.path[0]
                and before[3] == p.nav_from):
            retargets.append({'round': o['round_index'] + 1,
                             'tick': o['match_tick'], 'side': o['side']})
        current['new_moved'] = True
        return result

    new._player.move = move
    old_move = old._player.move

    def baseline_move(*args, **kwargs):
        current['old_moved'] = True
        return old_move(*args, **kwargs)

    old._player.move = baseline_move
    for line in path.open():
        row = json.loads(line)
        o = row.get('observation')
        if o is None:
            continue
        current.clear()
        a, b = old.act(o), new.act(o)
        totals['decisions'] += 1
        totals['recorded_faults'] += bool(row.get('fault'))
        totals['baseline_mismatches'] += bool(action_changed(a, row['action']))
        totals['changed_actions'] += bool(action_changed(a, b))
        assert not validate_action(b)
        json.dumps(b, allow_nan=False)
        for key in a.keys() | b.keys():
            fields[key] += bool(action_changed({key: a.get(key)}, {key: b.get(key)}))
        if current.get('old_moved') or current.get('new_moved'):
            m = o['me']
            label = ('carrier' if m['carrying_bomb'] else
                     'postplant_attack' if o['side'] == 'attack' and o['bomb']['state'] == 'planted'
                     else o['side'])
            c = movement.setdefault(label, Counter())
            for tag, action in [('old', a), ('new', b)]:
                if current.get(tag + '_moved'):
                    c[tag + '_move_calls'] += 1
                    c[tag + '_walk_calls'] += bool(action.get('walk'))
            c['both_move_calls'] += bool(current.get('old_moved') and current.get('new_moved'))
    assert not totals['baseline_mismatches'] and not totals['recorded_faults']
    return {'prefix': prefix, 'job_id': metadata['job_id'], 'seat': seat,
            'baseline_source_sha256': old_sha, 'candidate_source_sha256': new_sha,
            'recorded_bundle_sha256': V11_BUNDLE,
            'verified_training': {k: footer[k] for k in ('rows', 'raw_bytes', 'sha256')},
            'totals': dict(totals), 'changed_fields': {k: v for k, v in fields.items() if v},
            'movement_requests': {k: dict(v) for k, v in movement.items()},
            'preserved_retargets': retargets}


def startup():
    private = ROOT / 'data/private'
    with (private / '11dd-training.jsonl').open() as source:
        template = next(json.loads(s)['observation'] for s in source if '"observation"' in s)
    module, _ = load_policy(ROOT / 'agent.py')
    checks = []
    for prefix in ('11dd', '5538', '8342', 'eb98', 'dde6'):
        desc = json.loads((private / f'{prefix}-descriptor.json').read_text())
        raw = (private / f'{prefix}-static.jsonl').read_bytes()
        expected = desc['streams']['public_static']
        assert len(raw) == expected['raw_bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['sha256']
        public_map = json.loads(raw)['map']
        for side in ('attack', 'defend'):
            for null_hints in (False, True):
                o = copy.deepcopy(template)
                m = copy.deepcopy(public_map)
                if null_hints:
                    m['bot_waypoints'] = None
                o.update(map=m, map_id=m['id'], static_schema_hash='v12-synthetic-startup',
                         side=side, seat=0 if side == 'attack' else 5,
                         team=0 if side == 'attack' else 1, round_index=0)
                spawn = m['spawns']['T' if side == 'attack' else 'CT'][0]['position']
                o['me'].update(dict(zip(('x', 'y', 'z'), spawn)))
                p = module.Player()
                timings = []
                for tick in (0, 3, 360, 363):
                    o.update(match_tick=tick, round_tick=tick, phase='buy' if tick < 360 else 'live')
                    if tick:
                        o.pop('map', None)
                    wire = json.dumps(o)
                    start = time.process_time()
                    a = p.act(json.loads(wire))
                    json.dumps(a, allow_nan=False)
                    timings.append((time.process_time() - start) * 1000)
                    assert not validate_action(a)
                assert timings[0] < 1000 and max(timings[1:]) < 50
                checks.append({'map': m['id'], 'side': side, 'null_hints': null_hints,
                               'callback_ms_including_json': timings})
    return checks


def main():
    result = {'format': 'agx-fps-v12-validation.v1',
              'source_sha256': hashlib.sha256((ROOT / 'agent.py').read_bytes()).hexdigest(),
              'status': 'in_progress', 'recent_comparisons': [], 'historical_audits': [],
              'startup': startup(),
              'limitations': [
                  'Fixed historical observations do not measure future match results or saved travel time.',
                  'Startup cases combine verified public maps with synthetic own observations.',
                  'Local CPU checks are not server admission or server execution timings.',
                  '8342, f490 and dde6 own training exports returned training_trace_unavailable (503).']}
    dest = ROOT / 'experiments/v12-validation.json'
    for prefix, seat in RECENT:
        c = compare(prefix, seat)
        result['recent_comparisons'].append(c)
        print(prefix, c['totals'], 'route continuations', len(c['preserved_retargets']), flush=True)
    cases = [(p, s, V11_BUNDLE) for p, s in RECENT]
    for prefix, seat in [('385f', 9), ('e2e4', 6), ('e144', 3), ('c76', 1), ('9e35', 1), ('c630', 1)]:
        identity = json.loads((ROOT / f'experiments/v11-{prefix}-audit.json').read_text())
        cases.append((prefix, seat, identity['recorded_bundle_sha256']))
    for prefix, seat, bundle in cases:
        report = audit(ROOT / f'data/private/{prefix}-training.jsonl', ROOT / 'agent.py', seat, bundle)
        assert not report['errors'] and not report['warnings'], (prefix, report['errors'], report['warnings'])
        cpu = report['cpu']
        assert not (cpu['callbacks_over_local_budget_including_json']
                    or cpu['local_total_over_64_seconds'] or cpu['import_and_init_over_2_seconds'])
        (ROOT / f'experiments/v12-{prefix}-audit.json').write_text(json.dumps(report, indent=2) + '\n')
        result['historical_audits'].append({'prefix': prefix, 'seat': seat,
                                           'decisions_checked': report['decisions_checked'], 'cpu': cpu})
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(prefix, 'audit passed; local CPU seconds', round(cpu['local_total_seconds'], 3), flush=True)
    result['status'] = 'local_checks_passed_not_published'
    result['total_historical_decisions'] = sum(r['decisions_checked']
                                              for r in result['historical_audits'])
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
