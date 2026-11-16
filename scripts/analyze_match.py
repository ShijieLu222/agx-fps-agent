"""Verified own-seat diagnosis and fixed-observation policy comparison.

Public events are used for reporting only, never passed to either policy.
Candidate actions do not advance physics or establish new match results.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_training import action_changed, verify_export
from scripts.replay_check import validate_action


def load_policy(path):
    source = Path(path).read_bytes()
    module = types.ModuleType('match_audit_policy')
    exec(compile(source, str(path), 'exec'), module.__dict__)
    module.init({})
    return module, hashlib.sha256(source).hexdigest()


def analyze(training, descriptor, events_path, baseline, candidate, seat, bundle):
    seat, metadata, footer, recorded_hash = verify_export(training, seat, bundle)
    descriptor = json.loads(Path(descriptor).read_text())
    if (descriptor.get('job_id') != metadata['job_id']
            or descriptor.get('rules_version') != metadata['rules_version']
            or descriptor.get('complete') is not True):
        raise ValueError('descriptor identity/completion mismatch')
    raw = Path(events_path).read_bytes()
    expected = descriptor['streams']['public_events']
    digest = hashlib.sha256(raw).hexdigest()
    events = [json.loads(line) for line in raw.splitlines()]
    if (digest != expected['sha256'] or len(raw) != expected['raw_bytes']
            or len(events) != expected['rows']):
        raise ValueError('public event checksum/size/count mismatch')
    fires = {(e['match_tick'], e['player']): e for e in events if e['type'] == 'fire'}
    first_hits = {}
    for event in events:
        if event['type'] == 'hit' and event.get('target') == seat:
            first_hits.setdefault(event['round_index'], event)
    pending_hits = sorted(first_hits.values(), key=lambda e: e['match_tick'])
    hit_cursor, public_angles = 0, {}
    old, old_hash = load_policy(baseline)
    new, new_hash = load_policy(candidate)
    rounds = defaultdict(Counter)
    sides, first_damage, death_states = {}, {}, {}
    counts, changes = Counter(), Counter()
    previous = None
    first = None
    for line in Path(training).open():
        row = json.loads(line)
        o = row.get('observation')
        if o is None or row['seat'] != seat:
            continue
        if first is None:
            first = o
        m, r = o['me'], o['round_index'] + 1
        while previous and hit_cursor < len(pending_hits) and pending_hits[hit_cursor]['match_tick'] < o['match_tick']:
            hit = pending_hits[hit_cursor]
            fire = fires.get((hit['match_tick'], hit.get('player')))
            if fire and previous['match_tick'] <= hit['match_tick']:
                pm = previous['me']
                dx, dy = fire['from'][0] - pm['x'], fire['from'][1] - pm['y']
                offset = (math.atan2(-dy, dx) - pm['yaw'] + math.pi) % (2*math.pi) - math.pi
                public_angles[hit['round_index'] + 1] = {
                    'tick': hit['match_tick'], 'bearing_offset_deg': round(math.degrees(offset), 2),
                    'horizontal_distance_m': round(math.hypot(dx, dy), 2),
                    'own_visible_count_before_hit': len(previous['visible'])}
            hit_cursor += 1
        sides[r] = o['side']
        c = rounds[r]
        recorded, original, proposed = row['action'], old.act(o), new.act(o)
        counts['decisions'] += 1
        counts['recorded_faults'] += bool(row.get('fault'))
        counts['baseline_action_mismatches'] += action_changed(original, recorded)
        warnings = validate_action(proposed)
        if warnings:
            raise ValueError('candidate action invalid: ' + str(warnings))
        json.dumps(proposed, allow_nan=False)
        changed = action_changed(original, proposed)
        counts['changed_actions'] += changed
        for key in set(original) | set(proposed):
            if action_changed({key: original.get(key)}, {key: proposed.get(key)}):
                changes[key] += 1
        if (previous and previous['round_index'] == o['round_index']
                and previous['me']['alive']):
            if not m['alive']:
                death_states[r] = {'never_saw_enemy': c['visible_calls'] == 0,
                                   'last_call_no_sight': not previous['visible']}
            elif m['hp'] < previous['me']['hp'] and r not in first_damage:
                first_damage[r] = {'tick': o['match_tick'], 'visible': len(o['visible']),
                                   'baseline_yaw_rate': original.get('yaw_rate', 0),
                                   'candidate_yaw_rate': proposed.get('yaw_rate', 0)}
        if m['alive'] and o['phase'] == 'live':
            c['alive_live_calls'] += 1
            c['visible_calls'] += bool(o['visible'])
            c['carrying_calls'] += bool(m['carrying_bomb'])
            c['recorded_fire_requests'] += bool(recorded.get('fire'))
            c['recorded_plant_requests'] += recorded.get('use') == 'plant'
            c['recorded_defuse_requests'] += recorded.get('use') == 'defuse'
            if changed:
                c['changed_view_calls'] += 1
            slot = m.get('active_slot', 1)
            gun = m['weapons'][slot]
            reachable = old._player.reachable_enemies(o, gun)
            c['reachable_calls'] += bool(reachable)
            c['reachable_speed_blocked_calls'] += bool(reachable) and math.hypot(m['vx'], m['vy']) >= .25
        previous = o
    if counts['baseline_action_mismatches']:
        raise ValueError('baseline did not exactly reproduce recorded actions')
    own_team = seat // 5
    for event in events:
        r, kind = event['round_index'] + 1, event['type']
        c = rounds[r]
        if kind == 'round_end':
            c['team_won'] = int(event['round_winner'] == own_team)
            c['ended'] = 1
        if kind == 'kill_reward' and event.get('killer') == seat:
            c['kills'] += 1
        if kind == 'kill' and event.get('target') == seat:
            c['deaths'] += 1
        if kind == 'fire' and event.get('player') == seat:
            c['shot_events'] += 1
        if kind == 'hit' and event.get('player') == seat:
            c['hit_events'] += 1
            c['hp_damage'] += event['hp_damage']
        if kind == 'plant_complete_detail' and event.get('player') == seat:
            c['plants_completed'] += 1
        if kind == 'defuse_complete_detail' and event.get('player') == seat:
            c['defuses_completed'] += 1
    by_side = defaultdict(Counter)
    for r, c in rounds.items():
        c['deaths_without_any_sighting'] = int(death_states.get(r, {}).get('never_saw_enemy', False))
        c['deaths_without_final_sighting'] = int(death_states.get(r, {}).get('last_call_no_sight', False))
        by_side[sides[r]].update(c)
    return {
        'format': 'agx-fps-match-diagnosis.v1', 'job_id': metadata['job_id'],
        'map_id': first['map_id'], 'seat': seat, 'rules_version': metadata['rules_version'],
        'recorded_bundle_sha256': recorded_hash,
        'baseline_source_sha256': old_hash, 'candidate_source_sha256': new_hash,
        'verified_training': {k: footer[k] for k in ('rows', 'raw_bytes', 'sha256')},
        'verified_events': expected, 'result': descriptor['result']['round_wins'],
        'counts': dict(counts), 'changed_fields': dict(changes),
        'by_side': {s: dict(c) for s, c in by_side.items()},
        'rounds': [{**dict(c), 'round': r, 'side': sides[r],
                    'first_damage': first_damage.get(r),
                    'retrospective_first_hit_geometry': public_angles.get(r)}
                   for r, c in sorted(rounds.items())],
        'limitations': 'Fixed historical observations; no counterfactual physics or win-rate estimate. '
                       'Public events are retrospective reporting only. Shot/hit events can differ '
                       'from requested fire; penetration can create multiple hit events per shot.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'descriptor', 'events', 'baseline', 'candidate', 'bundle', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seat', type=int, required=True)
    args = parser.parse_args()
    result = analyze(args.training, args.descriptor, args.events, args.baseline,
                     args.candidate, args.seat, args.bundle)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'job_id': result['job_id'], 'counts': result['counts'],
                      'changed_fields': result['changed_fields']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
