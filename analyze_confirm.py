"""Held-out confirmation table: the proposed method against every baseline and oracle.

Combines the confirmation runs (results/confirm23/{obs,open,prune25}.csv, same
episodes) and labels each planner by its configuration. For every entry: success
with a 95% Wilson CI, per-scene success, reached goal, collision-free, steps, and
the paired comparison with the proposed method (episodes only the method / only
the other succeeded, exact McNemar, Holm-corrected over the realistic baselines).

    python analyze_confirm.py --dir results/confirm23
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from evaluate_nav import load, wilson

METHOD = "obs:supermpc-active"
NAMES = {
    "obs:supermpc-active": "Proposed: risk-aware gaze + observation-aware constraint",
    "obs:supermpc": "Fixed camera + observation-aware constraint",
    "obs:supermpc-infogain": "Information gain + observation-aware constraint",
    "obs:supermpc-360": "360 camera + observation-aware constraint (oracle)",
    "open:dmpc": "D-MPC (fixed camera)",
    "open:supermpc": "SupeR-MPC, fixed camera",
    "open:supermpc-scan": "SupeR-MPC, scanning camera",
    "open:supermpc-nearest": "SupeR-MPC, look at nearest",
    "open:supermpc-infogain": "SupeR-MPC, information gain",
    "open:supermpc-active": "SupeR-MPC, risk-aware gaze (previous method)",
    "open:supermpc-gazeoracle": "Gaze oracle: same camera, ground-truth pointing",
    "open:supermpc-360": "360 camera (oracle)",
    "open:supermpc-oracle": "Perfect perception (oracle)",
    "prune25:supermpc": "Fixed camera, old stale-track threshold (2.5 m)",
    "prune25:supermpc-infogain": "Information gain, old stale-track threshold",
    "prune25:supermpc-active": "Risk-aware gaze, old stale-track threshold",
}
ORACLES = {"obs:supermpc-360", "open:supermpc-gazeoracle", "open:supermpc-360", "open:supermpc-oracle"}


def main():
    from scipy.stats import binomtest

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/confirm23")
    args = p.parse_args()
    runs = {}
    for cfg in ("obs", "open", "prune25"):
        path = os.path.join(args.dir, f"{cfg}.csv")
        if os.path.exists(path):
            for r in load(path):
                runs.setdefault(f"{cfg}:{r['planner']}", {})[(r["scene"], int(r["episode"]))] = r
    eps = sorted(set.intersection(*[set(v) for v in runs.values()]))
    scenes = sorted({e[0] for e in eps})
    print(f"{len(eps)} paired episodes ({', '.join(scenes)})\n")
    m = runs[METHOD]
    rows, pvals = [], {}
    for key in [k for k in NAMES if k in runs]:
        rs = [runs[key][e] for e in eps]
        k = int(sum(r["success"] for r in rs))
        lo, hi = wilson(k, len(rs))
        a = b = 0
        if key != METHOD:
            a = int(sum(bool(m[e]["success"]) and not runs[key][e]["success"] for e in eps))
            b = int(sum(bool(runs[key][e]["success"]) and not m[e]["success"] for e in eps))
            pvals[key] = binomtest(a, a + b).pvalue if a + b else 1.0
        rows.append(dict(key=key, succ=k / len(rs), lo=lo, hi=hi, a=a, b=b,
                         scene=[np.mean([runs[key][e]["success"] for e in eps if e[0] == s]) for s in scenes],
                         reach=np.mean([r["reached"] for r in rs]), cfree=np.mean([r["collisions"] == 0 for r in rs]),
                         steps=np.mean([r["steps"] for r in rs if r["reached"]])))
    # Holm correction over the realistic (non-oracle) comparisons
    real = sorted([k for k in pvals if k not in ORACLES], key=lambda k: pvals[k])
    holm, running = {}, 0.0
    for i, k in enumerate(real):
        running = max(running, min(1.0, (len(real) - i) * pvals[k]))
        holm[k] = running
    print(f"{'planner':58s} {'success [95% CI]':>20s} " + " ".join(f"{s:>7s}" for s in scenes)
          + f" {'reached':>8s} {'coll-free':>9s} {'steps':>6s} {'method:other':>13s} {'p':>7s} {'p_holm':>7s}")
    for r in rows:
        ab = f"{r['a']}:{r['b']}" if r["key"] != METHOD else "-"
        pv = f"{pvals[r['key']]:.4f}" if r["key"] in pvals else "-"
        ph = f"{holm[r['key']]:.4f}" if r["key"] in holm else "-"
        print(f"{NAMES[r['key']]:58s} {r['succ']:.3f} [{r['lo']:.2f},{r['hi']:.2f}] "
              + " ".join(f"{v:7.3f}" for v in r["scene"])
              + f" {r['reach']:8.3f} {r['cfree']:9.3f} {r['steps']:6.1f} {ab:>13s} {pv:>7s} {ph:>7s}")
    gz, fx, o360 = (runs.get(k) for k in ("open:supermpc-gazeoracle", "open:supermpc", "open:supermpc-360"))
    if gz and fx and o360:
        s = {k: np.mean([runs[k][e]["success"] for e in eps]) for k in (METHOD, "open:supermpc", "open:supermpc-360",
                                                                          "open:supermpc-gazeoracle")}
        print(f"\nGap closed from fixed camera to 360 camera: "
              f"{(s[METHOD] - s['open:supermpc']) / (s['open:supermpc-360'] - s['open:supermpc']):.0%}; "
              f"to the gaze oracle: {(s[METHOD] - s['open:supermpc']) / (s['open:supermpc-gazeoracle'] - s['open:supermpc']):.0%}")


if __name__ == "__main__":
    main()
