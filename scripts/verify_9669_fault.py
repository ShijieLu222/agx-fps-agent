"""Reproduce the archived V10 startup fault using the owner's verified export.

Only two callbacks were recorded. This verifies the V11 fix for that input,
not a complete new match or the server's CPU budget.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_match import load_policy
from scripts.audit_training import complete_records, verify_export
from scripts.replay_check import validate_action


def main():
    path = ROOT / 'data/private/9669-training.jsonl'
    expected = '4add5462a81b39cded1fd19aa2a07e398b51d91f360cde6fc6398fb5608d4333'
    seat, metadata, footer, bundle = verify_export(path, 6, expected)
    rows = [r for r in complete_records(path) if 'observation' in r]
    first_map = rows[0]['observation']['map']
    assert len(rows) == 2 and 'bot_waypoints' in first_map
    assert first_map['bot_waypoints'] is None
    assert rows[0]['fault'] is False and rows[1]['fault'] is True
    results = {}
    for version, source in [('v10', ROOT / 'data/private/agx-v10-agent.py'),
                            ('v11', ROOT / 'data/private/agx-v11-agent.py')]:
        module, digest = load_policy(source)
        if version == 'v10':
            assert digest == 'e83b4c1fb3bbd4ccd91ddf14747760332d6b33ad8006008dfc7e87249f7cd1b1'
        if version == 'v11':
            assert digest == 'e044b26e3de850493d231a753df71d4435277030468ebb43b00fa8c0d97f7163'
        callbacks = []
        for row in rows:
            wire = json.dumps(copy.deepcopy(row['observation']), allow_nan=False)
            start = time.process_time()
            result = {'tick': row['observation']['match_tick']}
            try:
                action = module.act(json.loads(wire))
                json.dumps(action, allow_nan=False)
                result.update(action=action, validation_errors=validate_action(action))
            except Exception as error:
                result.update(error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
            result['local_callback_ms_including_json'] = (time.process_time()-start)*1000
            callbacks.append(result)
        results[version] = {'source_sha256': digest, 'callbacks': callbacks}
    old, new = results['v10']['callbacks'], results['v11']['callbacks']
    assert old[0]['action'] == rows[0]['action']
    assert old[1]['error_type'] == 'AttributeError'
    assert 'hints.get' in old[1]['traceback']
    assert all('error' not in r and not r['validation_errors'] for r in new)
    output = {
        'match_id': metadata['job_id'], 'seat': seat,
        'recorded_bundle_sha256': bundle,
        'recorded_version_id': metadata['participants'][0]['version_id'],
        'export_complete': footer,
        'export_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'actual_policy_input_bot_waypoints_is_null': True,
        'recorded_fault_tick': 3, 'results': results,
        'conclusion': 'V10 raises AttributeError on the actual second callback; published V11 accepts both original callbacks.',
        'limitations': ['Export records fault=true but no server exception text or fault subtype.',
                        'Only two original callbacks exist; this is not a full V11 match replay.',
                        'CPU timing is local, not server CPU.'],
    }
    dest = ROOT / 'experiments/v11-9669-fault-reproduction.json'
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
