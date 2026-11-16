"""Verify public evidence for the 9669 zero-stat match.

Public replay maps are NOT necessarily the original policy input. Local
reproductions distinguish null and absent fields and never assert a server
fault code or submitted bundle that has not been obtained from own logs.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_players import events_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/private/agx-v10-agent.py")
    args = parser.parse_args()
    private = ROOT / 'data/private'
    desc, events = events_for('9669')
    raw = (private / '9669-static.jsonl').read_bytes()
    expected = desc['streams']['public_static']
    assert len(raw) == expected['raw_bytes']
    assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    replay_map = json.loads(raw)['map']
    chunk = (private / '9669-frame-0.jsonl').read_bytes()
    assert hashlib.sha256(chunk).hexdigest() == desc['static']['streams']['public_frames'][0]
    frames = [json.loads(line) for line in chunk.splitlines()[:-1]]
    source = args.source.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    assert source_hash == 'e83b4c1fb3bbd4ccd91ddf14747760332d6b33ad8006008dfc7e87249f7cd1b1'
    policy = types.ModuleType('verified_v10')
    exec(compile(source, 'verified-v10-agent.py', 'exec'), policy.__dict__)
    reproductions = []
    for variant in ('public_replay_as_is', 'same_map_bot_waypoints_omitted', 'same_map_bot_waypoints_empty_object'):
        row = copy.deepcopy(replay_map)
        if variant.endswith('_omitted'):
            row.pop('bot_waypoints', None)
        elif variant.endswith('_empty_object'):
            row['bot_waypoints'] = {}
        player = policy.Player()
        start = time.process_time()
        error = None
        try:
            player.load_map(row)
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        reproductions.append({'variant': variant, 'error': error,
                               'local_cpu_ms': (time.process_time()-start)*1000,
                               'guard_plan_count': len(player.guard_plans)})
    contracts = []
    for prefix in ('385f', 'e2e4', 'e144', 'c76', '9e35', 'c630'):
        with (private / f'{prefix}-training.jsonl').open() as f:
            for line in f:
                record = json.loads(line)
                if 'observation' not in record:
                    continue
                m = record['observation']['map']
                contracts.append({'prefix': prefix, 'map_id': m['id'],
                                  'bot_waypoints_present': 'bot_waypoints' in m,
                                  'bot_waypoints_type': type(m.get('bot_waypoints')).__name__})
                break
    seat = 6
    own = {
        'kills': sum(e['type']=='kill_reward' and e.get('killer')==seat for e in events),
        'deaths': sum(e['type']=='kill' and e.get('target')==seat for e in events),
        'shots': sum(e['type']=='fire' and e.get('player')==seat for e in events),
        'damage_dealt': sum(e['hp_damage'] for e in events if e['type']=='hit' and e.get('player')==seat),
        'damage_received': sum(e['hp_damage'] for e in events if e['type']=='hit' and e.get('target')==seat),
        'buys': [{k:e[k] for k in ('match_tick','kind','price')} for e in events if e['type']=='buy' and e.get('seat')==seat],
        'fault_events': [{k:e[k] for k in ('match_tick','round_index','seat')} for e in events if e['type']=='fault' and e['seat']==seat],
    }
    opening = []
    for frame in frames[:6]:
        p = next(p for p in frame['players'] if p['seat']==seat)
        opening.append({'tick': frame['match_tick'], 'phase':frame['phase'],
                        **{k:p[k] for k in ('x','y','z','alive','hp','armor')}})
    halves = []
    for lo, hi, side in ((0,12,'attack'),(12,24,'defend')):
        ends = [e for e in events if e['type']=='round_end' and lo<=e['round_index']<hi]
        halves.append({'side':side,'wins':sum(e['round_winner']==1 for e in ends),
                       'losses':sum(e['round_winner']==0 for e in ends),
                       'end_reasons':dict(Counter(e['reason'] for e in ends))})
    out = {'format':'agx-fps-zero-stat-analysis.v1','job_id':desc['job_id'],
           'map_id':replay_map['id'],'seat':seat,'team':1,'round_wins':desc['result']['round_wins'],
           'verified_public_events':desc['streams']['public_events'],
           'verified_public_static':expected,'opening_chunk_sha256':desc['static']['streams']['public_frames'][0],
           'own_public_stats':own,'opening_frames':opening,'halves':halves,
           'local_source_sha256':source_hash,'public_map_initialization_reproductions':reproductions,
           'previous_test_map_contracts':contracts,
           'confirmed':['Public fault at match tick 3, before live phase starts at tick 360.',
                        'Exactly one armor purchase; zero shots, hits dealt/received, kills and deaths in complete public events.'],
           'unconfirmed':['Frozen bundle/version in this specific match: own metadata currently unavailable.',
                          'Exact runtime fault code/traceback: own training export currently unavailable.',
                          'Whether bot_waypoints was null in the actual policy input: replay maps may normalize absent fields to null.',
                          'Whether the fault was a map exception, callback CPU timeout or another runner failure.'],
           'limitations':['Public replay data are retrospective only and cannot substitute for own recorded policy input.',
                          'Removing the field is a controlled local hypothesis test, not a replay of the actual match.',
                          'Local CPU timing is not a measurement of server CPU.',
                          'No source change or new publication was made during this analysis.']}
    target = ROOT / 'experiments/v10-9669-zero-analysis.json'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'own':own,'reproductions':reproductions,'halves':halves},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
