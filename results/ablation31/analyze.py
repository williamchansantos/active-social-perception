"""Ablation table (see PLAN.md): each ablation paired with the full method at 60 deg, seed 31."""
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load, wilson  # noqa: E402

ABL = [("open_constraint", "supermpc-active", "open-loop unseen constraint (previous method)"),
       ("no_unseen_constraint", "supermpc-active", "no unseen-pedestrian constraint"),
       ("no_unseen_term", "active-tracked-only", "no unseen-pedestrian perception term"),
       ("no_tracked_term", "active-unseen-only", "no tracked-pedestrian perception term"),
       ("fixed_camera", "supermpc", "fixed camera (no active gaze)"),
       ("kf_tracker", "active-kf", "plain Kalman tracker (no PMB)"),
       ("prune25", "supermpc-active", "stale-track fix removed (prune 2.5 m)"),
       ("no_existence", "supermpc-active", "no existence weighting"),
       ("no_robot_sens", "active-no-sens", "no robot sensitivity"),
       ("no_state_cov", "active-no-statecov", "no predicted-state covariance")]


def idx(f, p):
    return {(r["scene"], int(r["episode"])): r for r in load(f) if r["planner"] == p}


full = idx("results/narrow31/obs_fov60.csv", "supermpc-active")
rows = [("FULL METHOD", full)]
for name, planner, label in ABL:
    path = f"{HERE}/{name}.csv"
    if os.path.exists(path):
        rows.append((label, idx(path, planner)))
keys = sorted(set.intersection(*[set(r) for _, r in rows]))
print(f"{len(keys)} paired episodes (Zara01 + Zara02, seed 31, 60 deg FOV)\n")
print(f"{'variant':48s} {'success [95% CI]':>19s} {'reached':>8s} {'steps':>6s} {'never-seen eps':>15s} {'tracked eps':>12s} "
      f"| full won:lost   p        never-seen full:abl  p")
pvals = []
for label, R in rows:
    rs = [R[k] for k in keys]
    k_s = int(sum(r["success"] for r in rs))
    lo, hi = wilson(k_s, len(rs))
    line = (f"{label:48s} {k_s / len(rs):.3f} [{lo:.2f},{hi:.2f}] {np.mean([r['reached'] for r in rs]):8.3f} "
            f"{np.mean([r['steps'] for r in rs if r['reached']]):6.1f} {sum(r['coll_untracked'] > 0 for r in rs):15d} "
            f"{sum(r['coll_tracked'] > 0 for r in rs):12d}")
    if R is not full:
        a = int(sum(bool(full[k]["success"]) and not R[k]["success"] for k in keys))
        b = int(sum(bool(R[k]["success"]) and not full[k]["success"] for k in keys))
        p = binomtest(a, a + b).pvalue if a + b else 1.0
        an = int(sum(full[k]["coll_untracked"] > 0 and not R[k]["coll_untracked"] > 0 for k in keys))
        bn = int(sum(R[k]["coll_untracked"] > 0 and not full[k]["coll_untracked"] > 0 for k in keys))
        pn = binomtest(an, an + bn).pvalue if an + bn else 1.0
        pvals.append((label, p))
        line += f" | {a:3d}:{b:<3d} {p:7.4f}   {an:3d}:{bn:<3d} {pn:7.4f}"
    print(line)
ps = sorted(pvals, key=lambda kv: kv[1])
running, holm = 0.0, {}
for i, (label, p) in enumerate(ps):
    running = max(running, min(1.0, (len(ps) - i) * p))
    holm[label] = running
print("\nHolm-corrected p (success, over all ablations): " + "; ".join(f"{k}: {v:.3f}" for k, v in holm.items()))
print("(full won:lost = episodes only the full method / only the ablation succeeded; never-seen full:abl = episodes "
      "with a never-seen collision only in the full method / only in the ablation)")
