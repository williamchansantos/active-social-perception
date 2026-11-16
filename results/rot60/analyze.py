"""Analysis of the 60 deg comparisons with rotation-augmented predictors (see PLAN.md)."""
import os, sys
import numpy as np
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load, wilson  # noqa: E402

P = {"proposed": "supermpc-active", "infogain": "supermpc-infogain", "nearest": "supermpc-nearest",
     "fixed": "supermpc", "gaze oracle": "supermpc-gazeoracle", "360": "supermpc-360"}
OLD = {"zara": ("results/narrow31/obs_fov60.csv", "results/narrow31/open_fov60.csv"),
       "eth": ("results/eth60/obs.csv", "results/eth60/open.csv")}
SC = {"zara": ("zara01", "zara02"), "eth": ("eth", "hotel")}


def idx(f, p):
    return {(r["scene"], int(r["episode"])): r for r in load(f) if r["planner"] == p} if os.path.exists(f) else {}


def mc(A, B, keys, fn):
    a = int(sum(bool(fn(A[k])) and not fn(B[k]) for k in keys)); b = int(sum(bool(fn(B[k])) and not fn(A[k]) for k in keys))
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)


for tag in ("zara", "eth"):
    runs = {m: idx(f"{HERE}/{tag}_{'obs' if m == 'proposed' else 'open'}.csv", p) for m, p in P.items()}
    old = {m: idx(OLD[tag][0] if m == "proposed" else OLD[tag][1], p) for m, p in P.items()}
    if tag == "zara":
        old["360"] = idx("results/narrow31/open_360.csv", "supermpc-360")
    else:
        old["360"] = idx("results/eth23/open.csv", "supermpc-360")
    keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
    s1, s2 = SC[tag]
    print(f"\n=== {tag.upper()} ({len(keys)} paired episodes, 60 deg, rotation-augmented predictors)")
    print(f"{'method':12s} {'success [95% CI]':>19s} {s1:>7s} {s2:>7s} {'reached':>8s} {'never-seen':>11s} {'tracked':>8s} "
          f"{'old pred.':>10s} | proposed won:lost  p        never-seen won:lost  p")
    prim = {}
    for m, X in runs.items():
        rs = [X[k] for k in keys]
        k_s = int(sum(r["success"] for r in rs)); lo, hi = wilson(k_s, len(rs))
        z = [np.mean([r["success"] for r in rs if r["scene"] == s]) for s in (s1, s2)]
        o = old.get(m, {})
        old_s = np.mean([o[k]["success"] for k in keys if k in o]) if o else float("nan")
        line = (f"{m:12s} {k_s / len(rs):.3f} [{lo:.2f},{hi:.2f}] {z[0]:7.3f} {z[1]:7.3f} {np.mean([r['reached'] for r in rs]):8.3f} "
                f"{sum(r['coll_untracked'] > 0 for r in rs):11d} {sum(r['coll_tracked'] > 0 for r in rs):8d} {old_s:10.3f}")
        if m != "proposed":
            a, b, p = mc(runs["proposed"], X, keys, lambda r: r["success"])
            an, bn, pn = mc(runs["proposed"], X, keys, lambda r: r["coll_untracked"] > 0)
            line += f" | {a:3d}:{b:<3d} {p:.4f}   {an:3d}:{bn:<3d} {pn:.4f}"
            if m in ("infogain", "nearest"):
                prim[m] = (a, b, p)
        print(line)
    running = 0.0
    for i, (m, (a, b, p)) in enumerate(sorted(prim.items(), key=lambda kv: kv[1][2])):
        running = max(running, min(1.0, (len(prim) - i) * p))
        print(f"  PRIMARY proposed vs {m:9s}: {a}:{b} p={p:.4f} p_holm={running:.4f} -> "
              f"{'CONFIRMED' if running < 0.05 and a > b else 'not confirmed'}")
