"""Analysis of the pre-registered 60 deg ETH/Hotel replication (see PLAN.md in this folder)."""
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load, wilson  # noqa: E402

P = {"proposed": "supermpc-active", "infogain": "supermpc-infogain", "nearest": "supermpc-nearest",
     "fixed": "supermpc", "gaze oracle": "supermpc-gazeoracle"}


def idx(f, p):
    return {(r["scene"], int(r["episode"])): r for r in load(f) if r["planner"] == p}


def mcnemar(A, B, keys, fn):
    a = int(sum(bool(fn(A[k])) and not fn(B[k]) for k in keys))
    b = int(sum(bool(fn(B[k])) and not fn(A[k]) for k in keys))
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)


runs = {m: idx(f"{HERE}/{'obs' if m == 'proposed' else 'open'}.csv", p) for m, p in P.items()}
o360 = idx("results/eth23/open.csv", "supermpc-360")
keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
print(f"{len(keys)} paired episodes (ETH + Hotel, seed 23, 60 deg); 360 camera: "
      f"{np.mean([o360[k]['success'] for k in keys]):.3f}\n")
print(f"{'method':12s} {'success [95% CI]':>19s} {'eth':>6s} {'hotel':>6s} {'reached':>8s} {'never-seen':>11s} "
      f"{'tracked':>8s} | proposed won:lost (success)  never-seen collisions")
primary = {}
for m, X in runs.items():
    rs = [X[k] for k in keys]
    k_s = int(sum(r["success"] for r in rs))
    lo, hi = wilson(k_s, len(rs))
    z = [np.mean([r["success"] for r in rs if r["scene"] == s]) for s in ("eth", "hotel")]
    line = (f"{m:12s} {k_s / len(rs):.3f} [{lo:.2f},{hi:.2f}] {z[0]:6.3f} {z[1]:6.3f} "
            f"{np.mean([r['reached'] for r in rs]):8.3f} {sum(r['coll_untracked'] > 0 for r in rs):11d} "
            f"{sum(r['coll_tracked'] > 0 for r in rs):8d}")
    if m != "proposed":
        a, b, p = mcnemar(runs["proposed"], X, keys, lambda r: r["success"])
        an, bn, pn = mcnemar(runs["proposed"], X, keys, lambda r: r["coll_untracked"] > 0)
        line += f" | {a:3d}:{b:<3d} p={p:.4f}        {an:3d}:{bn:<3d} p={pn:.4f}"
        if m in ("infogain", "nearest"):
            primary[m] = (a, b, p)
        for s in ("eth", "hotel"):
            ks = [k for k in keys if k[0] == s]
            a2, b2, p2 = mcnemar(runs["proposed"], X, ks, lambda r: r["success"])
            line += f" | {s} {a2}:{b2} p={p2:.3f}"
    print(line)
print("\nPrimary hypotheses (pooled success, Holm over the two):")
running = 0.0
for i, (m, (a, b, p)) in enumerate(sorted(primary.items(), key=lambda kv: kv[1][2])):
    running = max(running, min(1.0, (len(primary) - i) * p))
    print(f"  proposed vs {m:9s}: {a}:{b}  p={p:.4f}  p_holm={running:.4f}  -> "
          f"{'CONFIRMED' if running < 0.05 and a > b else 'not confirmed'}")
print("\n60 vs 100 deg on the same episodes (success):")
r100 = {"proposed": idx("results/eth23/obs.csv", "supermpc-active"), "infogain": idx("results/eth23/open.csv", "supermpc-infogain"),
        "nearest": idx("results/eth23/open.csv", "supermpc-nearest"), "fixed": idx("results/eth23/open.csv", "supermpc"),
        "gaze oracle": idx("results/eth23/open.csv", "supermpc-gazeoracle")}
for m in P:
    print(f"  {m:12s} 100 deg {np.mean([r100[m][k]['success'] for k in keys]):.3f} -> 60 deg "
          f"{np.mean([runs[m][k]['success'] for k in keys]):.3f}")
