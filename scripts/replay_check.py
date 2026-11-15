"""Check a policy against YOUR exported legal observations, without simulation.

This does not estimate win rate: subsequent observations belong to the old
policy's trajectory. Timings exclude JSON I/O and platform overhead and are
not admission evidence. Only run code you trust; this is not a sandbox.
Contract reference: https://agx.cool/docs/agent/fps.zh.md (2026-11-15).
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ANALOG = {"move_forward", "move_right", "yaw_rate", "pitch_rate", "climb"}
BOOLEAN = {"jump", "crouch", "duck", "walk", "fire", "fire_secondary", "reload", "sprint", "ads"}
INTEGER = {"weapon_slot": 2, "zoom_level": 2, "signal": 15}
ENUMS = {
    "ladder": {"none", "attach", "detach"},
    "use": {"none", "plant", "defuse", "pickup"},
    "throw": {"none", "hegrenade", "flashbang", "smokegrenade", "molotov", "incgrenade", "decoy"},
    "throw_state": {"idle", "prime", "release", "cancel"},
    "throw_mode": {"full", "lob", "drop"},
    "buy": set("none ak47 m4a1 awp pistol usp elite p250 fiveseven deagle nova mag7 mp9 mp7 bizon scar20 m4a4 ssg08 tec9 xm1014 sawedoff mac10 galilar sg553 knife armor helmet defusekit hegrenade flashbang smokegrenade molotov incgrenade decoy".split()),
}


def validate_action(action):
    """Validate documented basic controls; advanced message checks stay explicit."""
    if not isinstance(action, dict):
        raise ValueError("act must return an object")
    warnings = []
    for key, value in action.items():
        if key in ANALOG:
            valid = type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1
        elif key in BOOLEAN:
            valid = type(value) is bool
        elif key in INTEGER:
            valid = type(value) is int and 0 <= value <= INTEGER[key]
        elif key in ENUMS:
            valid = isinstance(value, str) and value in ENUMS[key]
        elif key == "pickup_id":
            valid = value is None or isinstance(value, str) and 1 <= len(value) <= 128
        elif key == "refund_item_id":
            valid = value is None or isinstance(value, str)
        elif key == "drop":
            valid = isinstance(value, str) and len(value) <= 128
            warnings.append("drop request semantics require platform validation")
        elif key == "message":
            valid = (isinstance(value, dict) and set(value) == {"kind", "zone", "value"}
                     and isinstance(value["kind"], str) and isinstance(value["zone"], str)
                     and type(value["value"]) is int and 0 <= value["value"] <= 15)
            warnings.append("message.kind and map zone membership are not checked locally")
        else:
            raise ValueError(f"Unknown action field: {key}")
        if not valid:
            raise ValueError(f"Invalid action field: {key}")
    encoded = json.dumps(action, allow_nan=False).encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise ValueError("Action alone exceeds 64 KiB; platform limit includes its response envelope")
    return warnings


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON at line {number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {number}")
            yield number, row


def decision_seat(row):
    if not isinstance(row.get("observation"), dict):
        return None
    seat = row.get("seat")
    if type(seat) is not int or not 0 <= seat <= 9:
        raise ValueError("Decision row requires integer seat in 0..9")
    observed_seat = row["observation"].get("seat", seat)
    if type(observed_seat) is not int or observed_seat != seat:
        raise ValueError("Decision seat and observation.seat disagree")
    return seat


def select_seat(path, requested=None):
    seats, last, ended = set(), None, False
    for number, row in rows(path):
        if ended:
            raise ValueError("Data appears after export_complete")
        last = row
        ended = row.get("type") == "export_complete"
        seat = decision_seat(row)
        if seat is not None:
            seats.add(seat)
    if last is None or not ended:
        raise ValueError("Incomplete export: final export_complete row is missing")
    if not seats:
        raise ValueError("No decision rows with object observations")
    if requested is None and len(seats) != 1:
        raise ValueError(f"Multiple seats {sorted(seats)}: choose your seat with --seat")
    if requested is not None and requested not in seats:
        raise ValueError(f"Seat {requested} has no observations")
    return requested if requested is not None else next(iter(seats))


def check(path, agent_path=ROOT / "agent.py", seat=None):
    seat = select_seat(path, seat)
    spec = importlib.util.spec_from_file_location("agx_checked_agent", agent_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load agent file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "act", None)):
        raise ValueError("Agent has no callable act(observation)")
    start = time.process_time()
    if callable(getattr(module, "init", None)):
        # A fresh public policy seed. Never pass a hidden match seed from metadata.
        module.init({"seed": 0, "seat": seat, "role": 0, "weights_dir": str(Path(agent_path).resolve().parent)})
    init_seconds = time.process_time() - start
    timings, warnings, errors, previous_tick = [], set(), [], None
    for number, row in rows(path):
        if decision_seat(row) != seat:
            continue
        observation = row["observation"]
        tick = observation.get("match_tick")
        if type(tick) is not int or tick < 0:
            raise ValueError(f"Invalid match_tick at line {number}")
        if previous_tick is not None and tick <= previous_tick:
            raise ValueError("Selected observations are not in increasing match_tick order; use one match per file")
        if previous_tick is None and (tick != 0 or "map" not in observation):
            warnings.add("First selected observation lacks match_tick=0/map; stateful checks may be incomplete")
        previous_tick = tick
        try:
            start = time.process_time()
            action = module.act(observation)
            elapsed = time.process_time() - start
            timings.append(elapsed)
            warnings.update(validate_action(action))
        except Exception as error:
            errors.append({"line": number, "match_tick": tick, "error": f"{type(error).__name__}: {error}"})
            # A fault can corrupt persistent policy state; do not pretend later checks are valid.
            break
    slow = sum(t > (1.0 if i == 0 else 0.05) for i, t in enumerate(timings))
    if slow or init_seconds > 2 or init_seconds + sum(timings) > 64:
        warnings.add("Local policy CPU exceeds a published budget; platform accounting also includes overhead")
    return {
        "seat": seat, "decisions_checked": len(timings), "errors": errors,
        "init_cpu_seconds": init_seconds, "act_cpu_seconds": sum(timings),
        "max_act_cpu_seconds": max(timings, default=0), "callbacks_over_local_budget": slow,
        "warnings": sorted(warnings), "export_complete_marker": True,
        "limitations": "Offline observation replay only: not a simulator, win-rate estimate, sandbox, or platform admission. Export marker presence is checked; server checksum/count fields are not verified. Timings exclude JSON and platform overhead. Init uses public policy seed 0, not the original runtime context.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path)
    parser.add_argument("--agent", type=Path, default=ROOT / "agent.py")
    parser.add_argument("--seat", type=int, choices=range(10))
    args = parser.parse_args()
    try:
        report = check(args.training, args.agent, args.seat)
    except Exception as error:
        parser.exit(2, f"Check failed: {type(error).__name__}: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
