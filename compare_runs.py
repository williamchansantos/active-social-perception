"""Compare several evaluate_nav.py runs (one CSV per configuration) on the same episodes.

Prints success per planner and configuration (pooled and per scene), reached goal,
and the mean over planners. With --paired REF, also paired success counts and exact
McNemar p-values of every planner against REF within each configuration.

    python compare_runs.py results/stale/*.csv --planners supermpc supermpc-active
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evaluate_nav import load, wilson


def main():
    from scipy.stats import binomtest

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--planners", nargs="+", default=None)
    p.add_argument("--paired", default=None, help="reference planner for paired tests")
    p.add_argument("--per-scene", action="store_true")
    args = p.parse_args()
    runs = {os.path.splitext(os.path.basename(f))[0]: load(f) for f in args.csvs}
    planners = args.planners or list(dict.fromkeys(r["planner"] for rows in runs.values() for r in rows))
    scenes = sorted({r["scene"] for rows in runs.values() for r in rows})
    width = max(len(k) for k in runs) + 2
    print(f"{'config':{width}s}" + "".join(f"{pl.replace('supermpc-', ''):>12s}" for pl in planners) + f"{'mean':>8s}")
    for name, rows in runs.items():
        by = {(r["scene"], r["episode"], r["planner"]): r for r in rows}
        eps = sorted({(r["scene"], r["episode"]) for r in rows})
        eps = [e for e in eps if all((*e, pl) in by for pl in planners if any(r["planner"] == pl for r in rows))]
        succ = {}
        for pl in planners:
            rs = [by[(*e, pl)] for e in eps if (*e, pl) in by]
            succ[pl] = np.mean([r["success"] for r in rs]) if rs else np.nan
        print(f"{name:{width}s}" + "".join(f"{succ[pl]:12.3f}" for pl in planners)
              + f"{np.nanmean(list(succ.values())):8.3f}   (n={len(eps)})")
        if args.per_scene:
            for sc in scenes:
                vals = []
                for pl in planners:
                    rs = [by[(*e, pl)] for e in eps if e[0] == sc and (*e, pl) in by]
                    vals.append(np.mean([r["success"] for r in rs]) if rs else np.nan)
                print(f"{'  ' + sc:{width}s}" + "".join(f"{v:12.3f}" for v in vals))
            reach = []
            for pl in planners:
                rs = [by[(*e, pl)] for e in eps if (*e, pl) in by]
                reach.append(np.mean([r["reached"] for r in rs]) if rs else np.nan)
            print(f"{'  reached':{width}s}" + "".join(f"{v:12.3f}" for v in reach))
        if args.paired and any(r["planner"] == args.paired for r in rows):
            out = []
            for pl in planners:
                if pl == args.paired or not any(r["planner"] == pl for r in rows):
                    continue
                a = sum(bool(by[(*e, args.paired)]["success"]) and not by[(*e, pl)]["success"] for e in eps)
                b = sum(bool(by[(*e, pl)]["success"]) and not by[(*e, args.paired)]["success"] for e in eps)
                pv = binomtest(int(a), int(a + b)).pvalue if a + b else 1.0
                out.append(f"{pl.replace('supermpc-', '')} {int(a)}:{int(b)} p={pv:.3f}")
            print(f"{'  vs ' + args.paired.replace('supermpc-', ''):{width}s}  " + " | ".join(out))


if __name__ == "__main__":
    main()
