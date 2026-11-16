"""Analysis of results/socnav/ablation_vis/PLAN.md: ablations with RGB-D vision vs the full method (confirmation runs).

    python -m socnav_bridge.analyze_ablation_vis > results/socnav/ablation_vis/summary.txt
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binomtest

from socnav_bridge.episodes import EPISODES

ROOT = "results/socnav"
SEEDS = (1, 2, 3)
SPAWN_S = 2.0
ABLATIONS = [
    ("open_constraint", "observation-aware coupling (open-loop unseen constraint)"),
    ("no_unseen_constraint", "unseen-pedestrian chance constraint"),
    ("no_unseen_term", "unseen term of the gaze cost"),
    ("no_tracked_term", "tracked term of the gaze cost"),
    ("fixed_camera", "active gaze (fixed camera)"),
    ("kf_tracker", "PMB tracker (plain Kalman tracker)"),
    ("prune25", "stale-track pruning (prune at 2.5 m)"),
    ("no_existence", "existence weighting"),
    ("no_robot_sens", "robot uncertainty propagation"),
    ("no_state_cov", "track state covariance in predictions"),
]


def load(prefix):
    out = {}
    for s in SEEDS:
        d = os.path.join(ROOT, f"{prefix}_s{s}")
        for e in EPISODES:
            lg, cj = os.path.join(d, f"{e}.episode_log.txt"), os.path.join(d, f"{e}.collisions.json")
            if not (os.path.exists(lg) and os.path.exists(cj)):
                continue
            reached = "termination cause: Success" in open(lg).read()
            c = json.load(open(cj))["coll"]
            out[(e, s)] = dict(reached=reached, ncoll=len(c), nspawn=sum(x[2] < SPAWN_S for x in c),
                               s_all=reached and not c, s=reached and not [x for x in c if x[2] >= SPAWN_S])
    return out


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def main():
    F = load("conf_vis_active")
    rows, ps = [], {}
    for name, label in ABLATIONS:
        A = load(f"abl_vis_{name}")
        keys = sorted(set(F) & set(A))
        if not keys:
            continue
        w = sum(F[k]["s"] and not A[k]["s"] for k in keys)
        l = sum(A[k]["s"] and not F[k]["s"] for k in keys)
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        ps[name] = p
        per_f = {e: np.mean([F[(e, s)]["s"] for s in SEEDS if (e, s) in F and (e, s) in A]) for e in EPISODES}
        per_a = {e: np.mean([A[(e, s)]["s"] for s in SEEDS if (e, s) in F and (e, s) in A]) for e in EPISODES}
        ek = [e for e in EPISODES if not np.isnan(per_f[e])]
        ew, el = sum(per_f[e] > per_a[e] for e in ek), sum(per_f[e] < per_a[e] for e in ek)
        rows.append((name, label, len(keys), np.mean([F[k]["s"] for k in keys]), np.mean([A[k]["s"] for k in keys]),
                     np.mean([A[k]["s_all"] for k in keys]), np.mean([A[k]["reached"] for k in keys]),
                     sum(A[k]["ncoll"] for k in keys), sum(A[k]["nspawn"] for k in keys), w, l, p, ew, el))
    hp = holm(ps)
    print("=== ABLATIONS WITH RGB-D VISION (SocNavBench, 30 episodes x seeds 1-3, paired with the full method)")
    print("success = excluding spawn collisions (primary); all = all collisions")
    print(f"{'removed component':58s} {'n':>3s} {'full':>5s} {'abl.':>5s} {'all':>5s} {'reach':>5s} "
          f"{'coll(sp)':>9s} {'full won:lost':>13s} {'p':>7s} {'p_holm':>7s} {'per-ep':>7s}")
    for (name, label, n, sf, sa, sall, rch, nc, ns, w, l, p, ew, el) in rows:
        print(f"- {label:56s} {n:3d} {sf:5.2f} {sa:5.2f} {sall:5.2f} {rch:5.2f} {nc:5d}({ns:2d}) {w:6d}:{l:<6d} "
              f"{p:7.4f} {hp[name]:7.4f} {ew:3d}:{el:<3d}")
    if F:
        print(f"\nfull method: success {np.mean([x['s'] for x in F.values()]):.3f} (all collisions "
              f"{np.mean([x['s_all'] for x in F.values()]):.3f}), collisions {sum(x['ncoll'] for x in F.values())} "
              f"({sum(x['nspawn'] for x in F.values())} spawn), n = {len(F)}")


if __name__ == "__main__":
    main()
