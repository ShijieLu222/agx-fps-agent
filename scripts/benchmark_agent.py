"""Measure synthetic map loading/A* CPU, not game performance or admission.

Usage: python3 scripts/benchmark_agent.py --size 64 --runs 7
The grid is artificial. JSON decoding, real map shapes, physics and total-match
CPU are not modeled; a fast result cannot establish platform budget compliance.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Player


def grid(size):
    nodes = [{"id": y * size + x, "position": [float(x), float(y), 0.0]}
             for y in range(size) for x in range(size)]
    edges = []
    for y in range(size):
        for x in range(size):
            node = y * size + x
            for neighbor in ([node + 1] if x + 1 < size else []) + (
                    [node + size] if y + 1 < size else []):
                edges.append({"from": node, "to": neighbor, "kind": "walk"})
    return {"sites": [], "nav": {"nodes": nodes, "edges": edges}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()
    if args.size < 2 or args.runs < 1:
        parser.error("size must be at least 2 and runs at least 1")
    row = grid(args.size)
    player = Player()
    started = time.process_time()
    player.load_map(row)
    load_ms = 1000 * (time.process_time() - started)
    durations = []
    for _ in range(args.runs):
        started = time.process_time()
        path = player.route([0., 0., 0.], [args.size - 1., args.size - 1., 0.])
        durations.append(1000 * (time.process_time() - started))
    print(json.dumps({
        "synthetic_grid": [args.size, args.size],
        "nodes": len(row["nav"]["nodes"]), "edges": len(row["nav"]["edges"]),
        "runs": args.runs, "map_load_cpu_ms": load_ms,
        "route_median_cpu_ms": statistics.median(durations),
        "route_max_cpu_ms": max(durations), "path_points": len(path),
        "route_runs_above_50ms": sum(t > 50 for t in durations),
        "limitation": "Synthetic local CPU only; excludes platform/JSON overhead and is not a win-rate or full-match budget test.",
    }, indent=2))


if __name__ == "__main__":
    main()
