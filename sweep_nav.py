"""One-factor-at-a-time parameter sweep for the separation between the active
perception planner and the baselines, on a TUNING episode set.

Every configuration runs on the same seeded tuning episodes (--seed, default 1,
disjoint from the main evaluation's seed 0). Configurations that only change the
active planner (perception weights) rerun only that planner and reuse the
baselines from the base configuration, which sees identical episodes. The
summary ranks configurations by the paired success gap active - fixed camera.

Pick a configuration here, then CONFIRM it on fresh episodes (a new seed) with
evaluate_nav.py: the best of many tuning runs overstates the true gap.

Example:
    python sweep_nav.py --episodes 50 --workers 12
    python sweep_nav.py --analyze-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

BASE = dict(fov=100, range=6.0, delta_unseen=0.01)
ALL = ["dmpc", "supermpc", "supermpc-scan", "supermpc-active"]
ROBUST = ["supermpc", "supermpc-scan", "supermpc-active"]  # D-MPC ignores delta_u

# name, settings that differ from BASE, planners to run (others come from `reuse`)
CONFIGS = [
    ("base", {}, ALL, None),
    ("du0.002", {"delta_unseen": 0.002}, ROBUST, "base"),
    ("du0.005", {"delta_unseen": 0.005}, ROBUST, "base"),
    ("du0.02", {"delta_unseen": 0.02}, ROBUST, "base"),
    ("du0.05", {"delta_unseen": 0.05}, ROBUST, "base"),
    ("range4", {"range": 4.0}, ALL, None),
    ("range8", {"range": 8.0}, ALL, None),
    ("fov130", {"fov": 130}, ALL, None),
    ("wu0.5", {"w_perception_unseen": 0.5}, ["supermpc-active"], "base"),
    ("wu5", {"w_perception_unseen": 5.0}, ["supermpc-active"], "base"),
    ("wp0.5", {"w_perception": 0.5}, ["supermpc-active"], "base"),
    ("wp5", {"w_perception": 5.0}, ["supermpc-active"], "base"),
]
FLAGS = dict(fov="--fov", range="--range", delta_unseen="--delta-unseen",
             w_perception="--w-perception", w_perception_unseen="--w-perception-unseen")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=50, help="tuning episodes per scene")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--dir", default="results/sweep")
    p.add_argument("--only", nargs="+", default=None, help="run only these configuration names")
    p.add_argument("--analyze-only", action="store_true")
    return p.parse_args()


def run(args):
    os.makedirs(args.dir, exist_ok=True)
    for name, diff, planners, _ in CONFIGS:
        if args.only and name not in args.only:
            continue
        settings = {**BASE, **diff}
        cmd = [sys.executable, "evaluate_nav.py", "--episodes", str(args.episodes), "--seed", str(args.seed),
               "--workers", str(args.workers), "--out", os.path.join(args.dir, f"{name}.csv"),
               "--planners", *planners]
        for k, v in settings.items():
            cmd += [FLAGS[k], str(v)]
        print(f"\n### {name}: {settings}, planners {planners}", flush=True)
        out = subprocess.run(cmd, capture_output=True, text=True)
        lines = [l for l in out.stdout.splitlines() if "jobs to run" in l or "FAILED" in l]
        print("\n".join(lines) or out.stderr[-2000:], flush=True)


def analyze(args):
    from scipy.stats import binomtest

    from evaluate_nav import load

    def rows_for(name):
        path = os.path.join(args.dir, f"{name}.csv")
        return load(path) if os.path.exists(path) else []

    print(f"\nTuning set: seed {args.seed}. Gap = paired success difference vs SupeR-MPC active; "
          f"a:b = episodes only active / only the other succeeded.")
    print(f"{'config':10s} {'n':>4s} {'active':>7s} {'fixed':>7s} {'scan':>7s} {'dmpc':>7s} "
          f"{'gap vs fixed':>13s} {'a:b':>7s} {'p':>6s} {'gap vs best other':>18s} "
          f"{'unseen coll-steps fixed/active':>31s}")
    summary = []
    for name, diff, planners, reuse in CONFIGS:
        rows = rows_for(name)
        if reuse:
            rows = [r for r in rows_for(reuse) if r["planner"] not in planners] + rows
        by = {(r["scene"], r["episode"], r["planner"]): r for r in rows}
        eps = sorted({(r["scene"], r["episode"]) for r in rows})
        eps = [e for e in eps if all((e[0], e[1], p) in by for p in ALL)]
        if not eps:
            continue
        rate = {p: np.mean([by[(s, e, p)]["success"] for s, e in eps]) for p in ALL}
        act = [by[(s, e, "supermpc-active")] for s, e in eps]
        fix = [by[(s, e, "supermpc")] for s, e in eps]
        a = int(sum(bool(x["success"]) and not y["success"] for x, y in zip(act, fix)))
        b = int(sum(bool(y["success"]) and not x["success"] for x, y in zip(act, fix)))
        p = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
        best_other = max(rate["supermpc"], rate["supermpc-scan"], rate["dmpc"])
        unseen = (sum(r["coll_untracked"] for r in fix), sum(r["coll_untracked"] for r in act))
        summary.append((rate["supermpc-active"] - rate["supermpc"], name))
        print(f"{name:10s} {len(eps):4d} {rate['supermpc-active']:7.2f} {rate['supermpc']:7.2f} "
              f"{rate['supermpc-scan']:7.2f} {rate['dmpc']:7.2f} {rate['supermpc-active'] - rate['supermpc']:+13.2f} "
              f"{f'{a}:{b}':>7s} {p:6.3f} {rate['supermpc-active'] - best_other:+18.2f} "
              f"{f'{unseen[0]}/{unseen[1]}':>31s}")
    if summary:
        print("\nRanked by gap vs fixed: " + ", ".join(f"{n} ({g:+.2f})" for g, n in sorted(summary, reverse=True)))
        print(f"{len(summary)} configurations compared on one tuning set: confirm the chosen one on fresh episodes.")


def main():
    args = parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
