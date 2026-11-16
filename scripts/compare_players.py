"""Retrospective, checksum-verified FPS player comparison; never policy inputs.

Use --matches with the saved authorized match-list JSON. Raw training/frames
stay in ignored data/private. Opponent behavior is observable, source is not.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import action_changed, verify_export


def events_for(prefix):
    root = ROOT / 'data/private'
    desc = json.loads((root / f'{prefix}-descriptor.json').read_text())
    raw = (root / f'{prefix}-events.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = desc['streams']['public_events']
    assert desc['complete'] and len(raw) == expected['raw_bytes']
    assert len(rows) == expected['rows']
    assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    assert len({e['event_id'] for e in rows}) == len(rows)
    return desc, rows


def public_comparison(prefix, match):
    desc, events = events_for(prefix)
    assert desc['job_id'] == match['id']
    starts = {e['round_index']: e for e in events if e['type'] == 'round_start'}
    ends = {e['round_index']: e for e in events if e['type'] == 'round_end'}
    players = []
    for p in match['participants']:
        seat = p['seat']
        fire = [e for e in events if e['type'] == 'fire' and e.get('player') == seat]
        hits = [e for e in events if e['type'] == 'hit' and e.get('player') == seat]
        deaths = [e for e in events if e['type'] == 'kill' and e.get('target') == seat]
        kills = [e for e in events if e['type'] == 'kill_reward' and e.get('killer') == seat]
        first = defaultdict(dict)
        for e in events:
            if e['type'] == 'fire' and e.get('player') == seat:
                first[e['round_index']].setdefault('shot', e['match_tick'])
            if e['type'] == 'hit' and e.get('target') == seat:
                first[e['round_index']].setdefault('received', e['match_tick'])
        per_side = {}
        for side in ('attack', 'defend'):
            rounds = [r for r, start in starts.items()
                      if (start['attack_team'] == seat // 5) == (side == 'attack')]
            hs = [e for e in hits if e['round_index'] in rounds]
            fs = [e for e in fire if e['round_index'] in rounds]
            per_side[side] = {'rounds': len(rounds), 'damage': sum(e['hp_damage'] for e in hs),
                'shots': len(fs), 'hit_events': len(hs),
                'kills': sum(e['round_index'] in rounds for e in kills),
                'deaths': sum(e['round_index'] in rounds for e in deaths)}
        gaps = Counter((a['weapon'], b['match_tick'] - a['match_tick'])
                       for a, b in zip(fire, fire[1:])
                       if a['weapon'] == b['weapon'] and a['round_index'] == b['round_index']
                       and b['match_tick'] - a['match_tick'] <= 42)
        # One bullet may penetrate multiple victims. Shot-level rate uses keys,
        # not hit-event count; shotgun pellets remain separate fire events.
        bullet_hits = {(e['match_tick'], e['player']) for e in hits if e['kind'] == 'bullet'}
        hit_fire_events = sum((e['match_tick'], seat) in bullet_hits for e in fire)
        alive_ticks = 0
        for r, start in starts.items():
            dead = next((e['match_tick'] for e in deaths if e['round_index'] == r), ends[r]['match_tick'])
            alive_ticks += max(0, dead - start['buy_freeze_end_tick'])
        players.append({'seat': seat, 'name': p['display_name'], 'is_mine': p['is_mine'],
            'team': seat // 5, 'damage': sum(e['hp_damage'] for e in hits),
            'kills': len(kills), 'deaths': len(deaths), 'fire_events': len(fire),
            'hit_events': len(hits), 'fire_events_with_bullet_hit': hit_fire_events,
            'head_hit_events': sum(e['part'] == 'head' for e in hits),
            'damage_by_part': {part: sum(e['hp_damage'] for e in hits if e['part'] == part)
                               for part in sorted({e['part'] for e in hits})},
            'weapons_fired': dict(Counter(e['weapon'] for e in fire)),
            'purchases': dict(Counter(e['kind'] for e in events if e['type'] == 'buy' and e.get('seat') == seat)),
            'short_same_weapon_shot_gaps': {f'{gun}:{gap}': n for (gun, gap), n in sorted(gaps.items())},
            'rounds_fired': len({e['round_index'] for e in fire}),
            'rounds_damaged': len({e['round_index'] for e in hits}),
            'death_rounds_without_shot': sum('shot' not in first[e['round_index']] for e in deaths),
            'rounds_received_before_first_shot': sum('received' in v and v['received'] < v.get('shot', math.inf) for v in first.values()),
            'alive_live_seconds': alive_ticks / 60, 'by_side': per_side})
    return {'prefix': prefix, 'job_id': desc['job_id'], 'created_at': match['created_at'],
            'result': desc['result']['round_wins'], 'verified_events': desc['streams']['public_events'],
            'players': sorted(players, key=lambda p: -p['damage'])}


def own_diagnosis(prefix, policy_path, expected_bundle, seat):
    path = ROOT / 'data/private' / f'{prefix}-training.jsonl'
    _, meta, footer, bundle = verify_export(path, seat, expected_bundle)
    mod, source_hash = load_policy(policy_path)
    state = mod._player
    mode = {}
    orig_move, orig_combat, orig_watch = state.move, state.combat, state.watch_navigation
    def move(o, goal, action, stop_at_goal=True):
        result = orig_move(o, goal, action, stop_at_goal)
        mode['move'] = 'arrived' if state.arrival_goal else ('escape' if o['match_tick'] < state.escape_until else 'route')
        return result
    def combat(o, action, gun, enemies=None):
        result = orig_combat(o, action, gun, enemies)
        enemy = next(e for e in (enemies if enemies is not None else o['visible']) if e['seat'] == state.target_seat)
        yaw, pitch = state.aim_error(o['me'], state.aim_point(enemy), gun)
        mode['combat'] = {'yaw_error_deg': math.degrees(yaw), 'pitch_error_deg': math.degrees(pitch)}
        tick = o['match_tick']
        # Merely diagnose the observed boundary; this does not change an action.
        clocks = [v for v in [gun.get('shot_ready'), o['me'].get('shot_ready')] if type(v) in (int, float)]
        mode['one_tick_cooldown'] = any(v == tick + 1 for v in clocks)
        return result
    def watch(o, action, recovering=False):
        result = orig_watch(o, action, recovering)
        if action_changed(action, result):
            mode['watch'] = 'damage_search' if state.damage_watch_start is not None and 0 <= o['match_tick']-state.damage_watch_start < 108 else 'attack_approach'
        return result
    state.move, state.combat, state.watch_navigation = move, combat, watch
    rounds, samples, previous, count, mismatches, faults = {}, defaultdict(list), None, 0, 0, 0
    for line in path.open():
        row = json.loads(line)
        o = row.get('observation')
        if o is None:
            continue
        mode.clear()
        a = mod.act(o)
        count += 1
        mismatches += action_changed(a, row['action'])
        faults += bool(row.get('fault'))
        if o['phase'] != 'live' or not o['me']['alive']:
            continue
        rnd, m, tick = o['round_index'] + 1, o['me'], o['match_tick']
        r = rounds.setdefault(rnd, {'round': rnd, 'side': o['side'], 'counts': Counter(), 'first': {}, 'death': None})
        c = r['counts'];c['alive_live'] += 1
        c['visible'] += bool(o['visible'])
        c['fire_request'] += bool(a.get('fire'))
        if 'move' in mode:c['move_'+mode['move']] += 1
        if 'watch' in mode:c['watch_'+mode['watch']] += 1
        if 'combat' in mode:
            c['combat'] += 1
            c['combat_speed_blocked'] += math.hypot(m['vx'], m['vy']) >= .25
            c['combat_aim_blocked'] += abs(mode['combat']['yaw_error_deg']) >= math.degrees(.025) or abs(mode['combat']['pitch_error_deg']) >= math.degrees(.025)
            c['combat_cooldown_one_tick'] += mode['one_tick_cooldown']
        for label, flag in [('visible', bool(o['visible'])), ('fire', bool(a.get('fire'))),
                            ('arrival', mode.get('move') == 'arrived')]:
            if flag:r['first'].setdefault(label, tick)
        sounds = [s for s in o.get('sounds', []) if s.get('direction') not in ('self', 'unknown')]
        c['directional_sound_calls'] += bool(sounds)
        if previous and previous['round_index'] == o['round_index'] and m['hp'] < previous['me']['hp'] and 'damage' not in r['first']:
            r['first']['damage'] = tick
            r['first_damage_hp'] = m['hp']
        samples[rnd].append({'tick': tick, 'round_tick': o['round_tick'],
            'position': [m[k] for k in ('x','y','z')], 'hp': m['hp'], 'yaw': m['yaw'],
            'visible': [e['seat'] for e in o['visible']], 'sounds': sounds,
            'speed': math.hypot(m['vx'],m['vy']), 'fire': bool(a.get('fire')),
            'yaw_rate': a.get('yaw_rate', 0), 'move_mode': mode.get('move'),
            'gun': m['weapons'][m['active_slot']]['id'] if m['weapons'][m['active_slot']] else None,
            'helmet': m['helmet'], 'armor': m['armor'], 'cash': m['cash']})
        previous = o
    assert not mismatches, 'Instrumented policy diverged from recorded actions'
    _, events = events_for(prefix)
    deaths = {e['round_index']+1:e for e in events if e['type']=='kill' and e.get('target')==seat}
    for rnd, r in rounds.items():
        ss = samples[rnd]
        incoming = [e for e in events if e['type']=='hit' and e.get('target')==seat and e['round_index']+1==rnd]
        if rnd in deaths:r['death'] = deaths[rnd]['match_tick']
        first_tick = incoming[0]['match_tick'] if incoming else None
        r['first_public_hit_tick'] = first_tick
        r['first_hit_to_death_seconds'] = (r['death']-first_tick)/60 if first_tick is not None and r['death'] is not None else None
        r['first_sight_hp'] = next((s['hp'] for s in ss if s['visible']), None)
        r['directional_sounds_before_hit_2s'] = [s for s in ss if first_tick is not None and first_tick-120 <= s['tick'] < first_tick and s['sounds']]
        r['last_position'] = ss[-1]['position']
        r['end_loadout'] = {k:ss[-1][k] for k in ('gun','armor','helmet','cash')}
        r['first_shot_hp'] = next((s['hp'] for s in ss if s['fire']),None)
        r['round_damage'] = sum(e['hp_damage'] for e in events if e['type']=='hit' and e.get('player')==seat and e['round_index']+1==rnd)
    totals = Counter()
    for r in rounds.values():totals.update(r['counts'])
    return {'job_id': meta['job_id'], 'seat': seat, 'bundle': bundle, 'source_sha256': source_hash,
            'verified_training': {k:footer[k] for k in ('rows','raw_bytes','sha256')},
            'decisions': count, 'action_mismatches': mismatches, 'faults': faults,
            'counts': dict(totals), 'rounds': list(rounds.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matches', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--policy', required=True, help='Verified archived v8 source, not the current candidate')
    a = p.parse_args()
    matches = json.loads(Path(a.matches).read_text())['items']
    prefixes = ['e2e4','e144','c76','9e35','c630']
    comparisons = [public_comparison(pre, next(m for m in matches if m['id'].startswith(pre))) for pre in prefixes]
    own = own_diagnosis('e2e4', Path(a.policy), 'd948b68705b72dc4c3931d1e4fd36b3bffeb683ebd6b9f85abf75dfbe5fc5fa0', 6)
    result = {'format':'agx-fps-player-comparison.v1', 'matches':comparisons, 'latest_own_policy':own,
        'limitations':['Retrospective public events and verified own inputs; no opponent private source or input.',
        'Different maps, sides and opponents: no controlled v7/v8 win-rate estimate.',
        'Hit-event proportion and head-hit proportion are not target-selection intent.',
        'Counterfactual damage cannot be recovered from fixed recorded observations.']}
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for c in comparisons:
        top=c['players'][0];me=next(p for p in c['players'] if p['is_mine'])
        print(c['prefix'],'top',top['name'],round(top['damage'],1),'own',round(me['damage'],1))
    print('Own counts',dict(own['counts']))

if __name__ == '__main__':main()
