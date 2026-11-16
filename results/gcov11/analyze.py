"""Tuning-set comparison: predictor-sensitivity vs CV state covariance (60 deg, rotation-augmented predictors)."""
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load, wilson  # noqa: E402


def idx(f, p):
    return {(r["scene"], int(r["episode"])): r for r in load(f) if r["planner"] == p} if os.path.exists(f) else {}


def mc(A, B, keys, fn=lambda r: r["success"]):
    a = int(sum(bool(fn(A[k])) and not fn(B[k]) for k in keys))
    b = int(sum(bool(fn(B[k])) and not fn(A[k]) for k in keys))
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)


for tag, sc in (("zara", ("zara01", "zara02")), ("eth", ("eth", "hotel"))):
    R = {}
    for mode in ("predictor", "cv"):
        R[("proposed", mode)] = idx(f"{HERE}/{tag}_{mode}_obs.csv", "supermpc-active")
        for lab, p in (("infogain", "supermpc-infogain"), ("nearest", "supermpc-nearest"), ("fixed", "supermpc")):
            R[(lab, mode)] = idx(f"{HERE}/{tag}_{mode}_open.csv", p)
    R = {k: v for k, v in R.items() if v}
    if not R:
        continue
    keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    print(f"\n=== {tag.upper()} (seed 11, {len(keys)} paired episodes, 60 deg)")
    print(f"{'method':10s} {'cov model':10s} {'success [95% CI]':>19s} {sc[0]:>7s} {sc[1]:>7s} {'reached':>8s} {'steps':>6s} "
          f"{'never-seen':>11s} {'tracked':>8s}")
    for (lab, mode), X in R.items():
        rs = [X[k] for k in keys]
        k_s = int(sum(r["success"] for r in rs))
        lo, hi = wilson(k_s, len(rs))
        z = [np.mean([r["success"] for r in rs if r["scene"] == s]) for s in sc]
        print(f"{lab:10s} {mode:10s} {k_s / len(rs):.3f} [{lo:.2f},{hi:.2f}] {z[0]:7.3f} {z[1]:7.3f} "
              f"{np.mean([r['reached'] for r in rs]):8.3f} {np.mean([r['steps'] for r in rs if r['reached']]):6.1f} "
              f"{sum(r['coll_untracked'] > 0 for r in rs):11d} {sum(r['coll_tracked'] > 0 for r in rs):8d}")
    print("  paired success (a:b = only the first / only the second succeeded):")
    for A, B in ((("proposed", "predictor"), ("proposed", "cv")), (("infogain", "predictor"), ("infogain", "cv")),
                 (("proposed", "predictor"), ("infogain", "predictor")), (("proposed", "cv"), ("infogain", "cv")),
                 (("proposed", "predictor"), ("nearest", "predictor")), (("proposed", "predictor"), ("fixed", "predictor"))):
        if A in R and B in R:
            a, b, p = mc(R[A], R[B], keys)
            print(f"    {A[0]}/{A[1]} vs {B[0]}/{B[1]}: {a}:{b} p={p:.3f}")
