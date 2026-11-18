"""Compare two packages' planted-site searches on verified AGX observations."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import types
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_training import action_changed, complete_records
from scripts.replay_check import validate_action


def policy(bundle):
    module = types.ModuleType("ranked_policy")
    with ZipFile(bundle) as archive:
        source = archive.read("agent.py")
    exec(compile(source, str(bundle), "exec"), module.__dict__)
    module.init({})
    return module


def compare(training, descriptor, chunks, baseline, candidate):
    desc = json.loads(descriptor.read_bytes())
    events = b"".join(path.read_bytes() for path in chunks)
    expected = desc["streams"]["public_events"]
    if (hashlib.sha256(events).hexdigest() != expected["sha256"]
            or len(events) != expected["raw_bytes"] or len(events.splitlines()) != expected["rows"]):
        raise ValueError("Public events failed integrity checks")
    plants = {}
    for line in events.splitlines():
        event = json.loads(line)
        if event["type"] == "plant_complete_detail":
            plants[event["round_index"]] = (event["site"], event["match_tick"])
    policies = [(name, policy(path)) for name, path in [("v13", baseline), ("v14", candidate)]]
    baseline_digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
    candidate_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    rounds, totals = defaultdict(Counter), Counter()
    for row in complete_records(training):
        if row.get("type") == "metadata":
            if row["job_id"] != desc["job_id"] or len(row["participants"]) != 1:
                raise ValueError("Expected a matching single-seat export")
            seat = row["participants"][0]["seat"]
            recorded_bundle = row["participants"][0]["source_hash"]
        obs = row.get("observation")
        if obs is None:
            continue
        if row["seat"] != seat:
            raise ValueError("Mismatched policy seat")
        totals["decisions"] += 1
        totals["recorded_faults"] += bool(row["fault"])
        rnd = obs["round_index"]
        for name, module in policies:
            # Public events are labels for scoring after the policy returns.
            action = module.act(obs)
            if validate_action(action):
                raise ValueError("Unexpected action schema warning")
            if name == "v13" and recorded_bundle == baseline_digest:
                totals["recorded_v13_action_mismatches"] += bool(action_changed(action, row["action"]))
            p = module._player
            if (rnd not in plants or obs["phase"] != "live" or obs["side"] != "defend"
                    or not obs["me"]["alive"] or obs["bomb"]["state"] != "planted"
                    or obs["bomb"].get("site") is not None or p.known_bomb is not None
                    or p.search_site is None):
                continue
            site, start = plants[rnd]
            wrong = p.sites[p.search_site]["id"] != site
            stats = rounds[rnd+1]
            totals[name+"_unknown"] += 1
            totals[name+"_wrong"] += wrong
            stats[name+"_wrong"] += wrong
            if obs["match_tick"]-start <= 3:
                stats[name+"_initial_wrong"] = int(wrong)
            if wrong:
                stats[name+"_last_wrong_s"] = (obs["match_tick"]-start)/60
    return dict(match=desc["job_id"], seat=seat, recorded_bundle=recorded_bundle,
                baseline_sha256=baseline_digest, candidate_sha256=candidate_digest,
                totals=totals, rounds=rounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("events", nargs="+", type=Path, help="Public event chunks in order")
    parser.add_argument("--baseline", type=Path, default=ROOT/"dist/agx-fps-agent-v13.zip")
    parser.add_argument("--candidate", type=Path, default=ROOT/"dist/agx-fps-agent-v14.zip")
    args = parser.parse_args()
    print(json.dumps(compare(args.training, args.descriptor, args.events, args.baseline, args.candidate),
                     indent=2, ensure_ascii=False))
