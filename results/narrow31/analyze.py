"""Analysis of the pre-registered narrow-FOV confirmation (see PLAN.md in this folder)."""
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


o360 = idx(f"{HERE}/open_360.csv", "supermpc-360")
print(f"360 camera: success {np.mean([r['success'] for r in o360.values()]):.3f} (n={len(o360)})")
primary = {}
for fov in (45, 60, 70):
    runs = {m: idx(f"{HERE}/{'obs' if m == 'proposed' else 'open'}_fov{fov}.csv", p) for m, p in P.items()}
    keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
    print(f"\n=== {fov} deg ({len(keys)} paired episodes){'  [PRIMARY]' if fov == 60 else ''}")
    print(f"{'method':12s} {'success [95% CI]':>20s} {'zara01':>7s} {'zara02':>7s} {'reached':>8s} {'never-seen eps':>15s} "
          f"{'tracked eps':>12s} | proposed won:lost (success)   never-seen collisions")
    for m, X in runs.items():
        rs = [X[k] for k in keys]
        k_s = int(sum(r["success"] for r in rs))
        lo, hi = wilson(k_s, len(rs))
        z = [np.mean([r["success"] for r in rs if r["scene"] == s]) for s in ("zara01", "zara02")]
        line = (f"{m:12s} {k_s / len(rs):.3f} [{lo:.2f},{hi:.2f}] {z[0]:7.3f} {z[1]:7.3f} "
                f"{np.mean([r['reached'] for r in rs]):8.3f} {sum(r['coll_untracked'] > 0 for r in rs):15d} "
                f"{sum(r['coll_tracked'] > 0 for r in rs):12d}")
        if m != "proposed":
            a, b, p = mcnemar(runs["proposed"], X, keys, lambda r: r["success"])
            an, bn, pn = mcnemar(runs["proposed"], X, keys, lambda r: r["coll_untracked"] > 0)
            line += f" | {a:3d}:{b:<3d} p={p:.4f}          {an:3d}:{bn:<3d} p={pn:.4f}"
            if fov == 60 and m in ("infogain", "nearest"):
                primary[m] = (a, b, p)
        print(line)
print("\nPrimary hypotheses at 60 deg (success; Holm over the two; one-sided direction checked):")
ps = sorted(primary.items(), key=lambda kv: kv[1][2])
running = 0.0
for i, (m, (a, b, p)) in enumerate(ps):
    running = max(running, min(1.0, (len(ps) - i) * p))
    verdict = "CONFIRMED" if running < 0.05 and a > b else "not confirmed"
    print(f"  proposed vs {m:9s}: {a}:{b}  p={p:.4f}  p_holm={running:.4f}  -> {verdict}")
