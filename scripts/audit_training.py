"""Strict FPS export verification and historical-observation policy checks.

This advances no game physics. A new action cannot change the next recorded
observation, so movement requests, damage opportunities and CPU measurements
are not new match results or proof of server admission. Only run trusted code.

Example:
  python3 scripts/audit_training.py data/private/training.jsonl --seat 5 \
    --expected-recorded-bundle-sha256 <the accepted version's bundle hash> \
    --output experiments/replay-v3.json

The aggregate report omits account identifiers, seeds and raw observations.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.replay_check import validate_action

# Frozen rsi-fps-5.5 public weapon catalogue, not policy internals.
# https://agx.cool/docs/agent/fps.zh.md (2026-11-15)
RANGES = {
    **dict.fromkeys("ak47 m4a1 awp scar20 m4a4 ssg08 galilar sg553".split(), 208.0768),
    **dict.fromkeys("pistol usp elite p250 fiveseven deagle tec9".split(), 104.0384),
    "nova": 76.2, "mag7": 35.559999999999995,
    "sawedoff": 35.559999999999995, "xm1014": 76.2,
    "mp9": 91.44, "mp7": 91.44, "bizon": 91.44, "mac10": 91.44,
}


def valid_seat(value):
    return type(value) is int and 0 <= value <= 9


def reject_nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)


def complete_records(path):
    """Validate the export while yielding its records; exhaust before accepting.

    SHA-256 covers original decision bytes, including their newline. Metadata
    identifies the recorded ZIP, not the candidate source file or match seed.
    """
    metadata, outcome, complete = None, False, False
    participants, ticks, steps, maps = {}, {}, {}, {}
    count, byte_count = 0, 0
    seats, digest = Counter(), hashlib.sha256()
    with Path(path).open("rb") as source:
        for number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            if complete:
                raise ValueError("data after export_complete")
            try:
                record = json.loads(raw, parse_constant=reject_nonfinite)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("invalid JSON at line " + str(number)) from error
            if not isinstance(record, dict):
                raise ValueError("record must be an object")
            kind = record.get("type")
            if kind == "metadata":
                if metadata is not None or count or outcome:
                    raise ValueError("metadata must occur once at the beginning")
                if (record.get("game_id") != "fps"
                        or record.get("schema") != "agent-tournament.training-export.v2"
                        or record.get("decision_encoding") != "original-jsonl-bytes"
                        or record.get("observation_semantics") != "recorded_policy_input"
                        or record.get("completion_record") != "export_complete"
                        or not isinstance(record.get("rules_version"), str)):
                    raise ValueError("unsupported FPS export identity or encoding")
                for participant in record.get("participants") or []:
                    if not isinstance(participant, dict):
                        raise ValueError("participant must be an object")
                    seat = participant.get("seat")
                    if (not valid_seat(seat) or seat in participants
                            or type(participant.get("team")) is not int
                            or type(participant.get("role")) is not int
                            or participant.get("team") != seat // 5
                            or participant.get("role") != 0
                            or not isinstance(participant.get("source_hash"), str)
                            or not re.fullmatch(r"[0-9a-f]{64}", participant["source_hash"])):
                        raise ValueError("invalid or duplicate participant identity")
                    participants[seat] = participant
                if not participants:
                    raise ValueError("metadata has no participants")
                metadata = record
            elif "observation" in record:
                if metadata is None or outcome:
                    raise ValueError("decision outside metadata/outcome boundaries")
                seat, observation = record.get("seat"), record["observation"]
                if not valid_seat(seat) or seat not in participants or not isinstance(observation, dict):
                    raise ValueError("unknown decision seat or invalid observation")
                participant = participants[seat]
                if (type(observation.get("seat")) is not int or observation["seat"] != seat
                        or type(observation.get("team")) is not int
                        or type(observation.get("role")) is not int
                        or observation.get("team") != participant["team"]
                        or observation.get("role") != participant["role"]
                        or observation.get("rules_version") != metadata["rules_version"]):
                    raise ValueError("observation identity differs from metadata")
                tick, step = observation.get("match_tick"), record.get("step")
                if (type(tick) is not int or tick < 0 or tick <= ticks.get(seat, -1)
                        or type(step) is not int or step < 0 or step <= steps.get(seat, -1)):
                    raise ValueError("decision clocks must increase for each seat")
                identity = (observation.get("map_id"), observation.get("static_schema_hash"))
                if not all(isinstance(value, str) and value for value in identity):
                    raise ValueError("missing map identity")
                if seat not in ticks:
                    if tick != 0 or not isinstance(observation.get("map"), dict):
                        raise ValueError("first decision must include tick zero and initial map")
                    maps[seat] = identity
                elif maps[seat] != identity:
                    raise ValueError("map identity changed within one match export")
                ticks[seat], steps[seat] = tick, step
                count += 1
                byte_count += len(raw)
                seats[str(seat)] += 1
                digest.update(raw)
            elif kind == "outcome":
                if metadata is None or outcome or not count:
                    raise ValueError("unexpected outcome")
                outcome = True
            elif kind == "export_complete":
                if metadata is None or not outcome or record.get("complete") is not True:
                    raise ValueError("missing metadata/outcome or incomplete export")
                if (type(record.get("rows")) is not int or record["rows"] != count
                        or type(record.get("raw_bytes")) is not int or record["raw_bytes"] != byte_count
                        or record.get("sha256") != digest.hexdigest()
                        or record.get("seat_calls") != dict(seats)
                        or any(type(value) is not int for value in record["seat_calls"].values())
                        or set(map(int, seats)) != set(participants)):
                    raise ValueError("export checksum, byte count or participant counts do not match")
                complete = True
            else:
                raise ValueError("unexpected record at line " + str(number))
            yield record
    if not complete:
        raise ValueError("missing export_complete: log may be truncated")


def verify_export(path, requested=None, expected_recorded_bundle=None):
    metadata, footer = None, None
    for record in complete_records(path):
        if record.get("type") == "metadata":
            metadata = record
        elif record.get("type") == "export_complete":
            footer = record
    participants = {p["seat"]: p for p in metadata["participants"]}
    if requested is None:
        if len(participants) != 1:
            raise ValueError("multiple seats require --seat")
        requested = next(iter(participants))
    if not valid_seat(requested) or requested not in participants:
        raise ValueError("requested seat is not in export")
    source_hash = participants[requested]["source_hash"]
    if expected_recorded_bundle is not None and source_hash != expected_recorded_bundle:
        raise ValueError("recorded bundle SHA-256 does not match expected accepted version")
    return requested, metadata, footer, source_hash


def action_changed(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return left != right
    for key in set(left) | set(right):
        a, b = left.get(key), right.get(key)
        if type(a) in (int, float) and type(b) in (int, float):
            if abs(a - b) > 1e-10:
                return True
        elif a != b:
            return True
    return False


def summarize_behavior(observation, action, counts, recorded=None):
    """Count requested controls, never successful movement or inferred hits."""
    if observation.get("phase") != "live" or not observation["me"].get("alive"):
        return
    counts["alive_live_calls"] += 1
    me = observation["me"]
    moving = math.hypot(action.get("move_forward", 0), action.get("move_right", 0)) > 0.01
    counts["movement_requests"] += moving
    counts["fire_requests"] += bool(action.get("fire"))
    if isinstance(recorded, dict) and recorded.get("fire"):
        counts["recorded_fire_requests"] += 1
        counts["recorded_fire_replaced_by_movement_without_fire"] += moving and not action.get("fire", False)
    counts["plant_requests"] += action.get("use") == "plant"
    counts["defuse_requests"] += action.get("use") == "defuse"
    visible = observation.get("visible") or []
    if not visible or me.get("blind_ticks", 0):
        return
    counts["visible_calls"] += 1
    guns = (me.get("weapons") or [])[:2]
    empty = not any(gun and (gun.get("ammo", 0) > 0 or gun.get("reserve", 0) > 0) for gun in guns)
    if empty:
        counts["visible_all_firearms_exhausted_calls"] += 1
        counts["visible_exhausted_movement_requests"] += moving
    slot = action.get("weapon_slot", me.get("active_slot", 1))
    gun = guns[slot] if 0 <= slot < len(guns) else None
    if action.get("fire") and gun and gun.get("id") in RANGES:
        nearest = min(math.sqrt(sum((enemy[k] - me[k]) ** 2 for k in ("x", "y", "z"))) for enemy in visible)
        counts["fire_requests_with_every_visible_enemy_beyond_catalogue_range"] += nearest > RANGES[gun["id"]]


def audit(path, agent_path=ROOT / "agent.py", seat=None, expected_recorded_bundle=None):
    seat, metadata, footer, recorded_hash = verify_export(path, seat, expected_recorded_bundle)
    source = Path(agent_path).read_bytes()
    module = types.ModuleType("agx_training_audit_candidate")
    module.__file__ = str(agent_path)
    started = time.process_time()
    exec(compile(source, str(agent_path), "exec"), module.__dict__)
    if not callable(getattr(module, "act", None)):
        raise ValueError("candidate has no callable act")
    if callable(getattr(module, "init", None)):
        module.init({"seed": 0, "seat": seat, "role": 0,
                     "weights_dir": str(Path(agent_path).resolve().parent)})
    init_cpu = time.process_time() - started
    timings, wire_timings, errors, warnings = [], [], [], set()
    counts, by_round = Counter(), defaultdict(Counter)
    decode_cpu = encode_cpu = 0.0
    # Export integrity was verified before any candidate code was executed.
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            started = time.process_time()
            row = json.loads(raw, parse_constant=reject_nonfinite)
            decoded = time.process_time() - started
            if "observation" not in row or row.get("seat") != seat:
                continue
            decode_cpu += decoded
            observation = row["observation"]
            counts["recorded_faults"] += bool(row.get("fault"))
            try:
                started = time.process_time()
                action = module.act(observation)
                elapsed = time.process_time() - started
                timings.append(elapsed)
                started = time.process_time()
                json.dumps(action, allow_nan=False)
                encoded = time.process_time() - started
                encode_cpu += encoded
                wire_timings.append(decoded + elapsed + encoded)
                warnings.update(validate_action(action))
                counts["changed_actions_vs_recorded"] += action_changed(action, row.get("action"))
                summarize_behavior(observation, action, counts, row.get("action"))
                summarize_behavior(observation, action, by_round[str(observation["round_index"] + 1)], row.get("action"))
            except Exception as error:
                errors.append({"line": number, "error": type(error).__name__ + ": " + str(error)})
                break  # A fault may have corrupted persistent candidate state.
    sorted_times = sorted(timings)
    budget_exceeded = sum(t > (1.0 if i == 0 else 0.05) for i, t in enumerate(wire_timings))
    local_cpu = init_cpu + decode_cpu + sum(timings) + encode_cpu
    return {
        "format": "agx-fps-historical-observation-audit.v1",
        "python_version": sys.version.split()[0], "rules_version": metadata["rules_version"],
        "recorded_bundle_sha256": recorded_hash,
        "expected_recorded_bundle_verified": expected_recorded_bundle is not None,
        "candidate_source_sha256": hashlib.sha256(source).hexdigest(),
        "export_integrity": {"complete": True, "rows": footer["rows"], "raw_bytes": footer["raw_bytes"],
                             "decision_sha256": footer["sha256"], "rows_bytes_sha256_seats_verified": True},
        "decisions_checked": len(timings), "errors": errors,
        "cpu": {"import_and_init_seconds": init_cpu, "act_seconds": sum(timings),
                "json_row_decode_seconds": decode_cpu, "action_encode_seconds": encode_cpu,
                "local_total_seconds": local_cpu,
                "local_callback_max_ms_including_json": max(wire_timings, default=0) * 1000,
                "act_max_ms": max(timings, default=0) * 1000,
                "act_p99_ms": sorted_times[max(0, math.ceil(len(sorted_times) * .99) - 1)] * 1000 if sorted_times else 0,
                "act_median_ms": statistics.median(timings) * 1000 if timings else 0,
                "callbacks_over_local_budget_including_json": budget_exceeded,
                "local_total_over_64_seconds": local_cpu > 64,
                "import_and_init_over_2_seconds": init_cpu > 2},
        "requested_behavior": dict(counts),
        "by_historical_round": dict(by_round), "warnings": sorted(warnings),
        "limitations": "Historical legal observations only; no physics or future observation changes. "
                       "Controls requested are not successful movement, new damage, wins or ranking. "
                       "CPU includes local row decoding, policy and action encoding, but excludes server IPC, "
                       "validation and scheduling; this is not platform budget/admission proof. "
                       "Init uses public policy seed 0, never the metadata's hidden match seed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("--agent", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--seat", type=int, choices=range(10))
    parser.add_argument("--expected-recorded-bundle-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.training, args.agent, args.seat, args.expected_recorded_bundle_sha256)
    except Exception as error:
        parser.exit(2, "Audit failed: " + type(error).__name__ + ": " + str(error) + "\n")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    cpu = report["cpu"]
    return int(bool(report["errors"] or cpu["callbacks_over_local_budget_including_json"]
                    or cpu["local_total_over_64_seconds"] or cpu["import_and_init_over_2_seconds"]))


if __name__ == "__main__":
    sys.exit(main())
