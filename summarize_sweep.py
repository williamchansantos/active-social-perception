"""Summarize a one-parameter sweep of evaluate_nav.py runs (one CSV per value).

For each value: success per planner (95% Wilson CI), the paired success gap
active - fixed camera (exact McNemar), collision-steps with tracked and
never-detected pedestrians, and steps to goal. Saves a two-panel figure.

Example:
    python summarize_sweep.py --pattern "results/range/range_{}.csv" --values 3 4 5 6 8 10 12 \
        --label "camera range (m)" --out results/range_sweep.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evaluate_nav import load, wilson

PLANNERS = ["dmpc", "supermpc", "supermpc-scan", "supermpc-active"]
NAMES = {"dmpc": "D-MPC", "supermpc": "SupeR-MPC, fixed", "supermpc-scan": "SupeR-MPC, scanning",
         "supermpc-active": "SupeR-MPC, active"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", required=True, help="CSV path with {} for the value")
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--label", default="value")
    p.add_argument("--out", default=None, help="figure path")
    return p.parse_args()


def main():
    from scipy.stats import binomtest

    args = parse_args()
    stats = {}
    print(f"{args.label:>16s} " + " ".join(f"{NAMES[p]:>20s}" for p in PLANNERS)
          + f" {'gap act-fix':>12s} {'a:b':>7s} {'p':>6s} {'steps fix/act':>14s} "
          f"{'coll trk fix/act':>17s} {'coll unseen fix/act':>20s}")
    for v in args.values:
        path = args.pattern.format(v)
        if not os.path.exists(path):
            continue
        rows = load(path)
        by = {(r["scene"], r["episode"], r["planner"]): r for r in rows}
        eps = sorted({(r["scene"], r["episode"]) for r in rows})
        eps = [e for e in eps if all((e[0], e[1], p) in by for p in PLANNERS)]
        rs = {p: [by[(s, e, p)] for s, e in eps] for p in PLANNERS}
        succ = {p: np.mean([r["success"] for r in rs[p]]) for p in PLANNERS}
        act, fix = rs["supermpc-active"], rs["supermpc"]
        a = int(sum(bool(x["success"]) and not y["success"] for x, y in zip(act, fix)))
        b = int(sum(bool(y["success"]) and not x["success"] for x, y in zip(act, fix)))
        pval = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
        steps = {p: np.mean([r["steps"] for r in rs[p] if r["reached"]]) for p in PLANNERS}
        ctrk = {p: int(sum(r["coll_tracked"] for r in rs[p])) for p in PLANNERS}
        cun = {p: int(sum(r["coll_untracked"] for r in rs[p])) for p in PLANNERS}
        stats[v] = dict(n=len(eps), succ=succ, k={p: int(sum(r["success"] for r in rs[p])) for p in PLANNERS},
                        gap=succ["supermpc-active"] - succ["supermpc"], p=pval, ctrk=ctrk, cun=cun, steps=steps)
        ab = f"{a}:{b}"
        st = "%.1f/%.1f" % (steps["supermpc"], steps["supermpc-active"])
        tk = "%d/%d" % (ctrk["supermpc"], ctrk["supermpc-active"])
        un = "%d/%d" % (cun["supermpc"], cun["supermpc-active"])
        print(f"{v:>16s} " + " ".join(f"{succ[p]:20.2f}" for p in PLANNERS)
              + f" {stats[v]['gap']:+12.2f} {ab:>7s} {pval:6.3f} {st:>14s} {tk:>17s} {un:>20s}")
    if not stats or not args.out:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [float(v) for v in stats]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = {"dmpc": ("0.7", "--"), "supermpc": ("0.35", "-"), "supermpc-scan": ("0.65", "-"),
              "supermpc-active": ("tab:blue", "-")}
    for p in PLANNERS:
        c, ls = styles[p]
        y = [stats[v]["succ"][p] for v in stats]
        ci = [wilson(stats[v]["k"][p], stats[v]["n"]) for v in stats]
        ax[0].errorbar(xs, y, yerr=[[yy - lo for yy, (lo, _) in zip(y, ci)], [hi - yy for yy, (_, hi) in zip(y, ci)]],
                       color=c, ls=ls, marker="o", capsize=3, lw=2 if p == "supermpc-active" else 1.2, label=NAMES[p])
        tot = [stats[v]["ctrk"][p] + stats[v]["cun"][p] for v in stats]
        ax[1].plot(xs, tot, color=c, ls=ls, marker="o", lw=2 if p == "supermpc-active" else 1.2, label=NAMES[p])
    ax[0].set(xlabel=args.label, ylabel="success rate (95% Wilson CI)", title="Success", xticks=xs)
    ax[1].set(xlabel=args.label, ylabel="collision-steps (tracked + never detected)", title="Collisions", xticks=xs)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
