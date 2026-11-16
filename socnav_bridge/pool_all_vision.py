"""POST-HOC combined analysis: final method (no observation-aware coupling) with RGB-D vision on SocNavBench,
all seeds run with identical settings (walls, 60 deg / 4 m, vision): seeds 1-3 (proposed from the vision ablation
abl_vis_open_constraint, baselines from the confirmation run conf_vis_*), 4-6 (fin_vis_*), 7-16 (f61_vis_*).
Same tests as results/final61/PLAN.md part B: episode-level permutation test, cluster bootstrap CI, pooled McNemar.
    python -m socnav_bridge.pool_all_vision > results/final61/socnav_pooled_s1-16.txt
"""
import json
import os

import numpy as np
from scipy.stats import binomtest

from socnav_bridge.episodes import EPISODES

SOURCES = {"proposed": [("abl_vis_open_constraint", range(1, 4)), ("fin_vis_active", range(4, 7)), ("f61_vis_active", range(7, 17))],
           "fixed": [("conf_vis_fixed", range(1, 4)), ("fin_vis_fixed", range(4, 7)), ("f61_vis_fixed", range(7, 17))],
           "information gain": [("conf_vis_infogain", range(1, 4)), ("fin_vis_infogain", range(4, 7)), ("f61_vis_infogain", range(7, 17))],
           "decoupled": [("f61_vis_decoupled", range(7, 17))]}


def load(method):
    out = {}
    for prefix, seeds in SOURCES[method]:
        for s in seeds:
            for e in EPISODES:
                d = f"results/socnav/{prefix}_s{s}"
                lg, cj = f"{d}/{e}.episode_log.txt", f"{d}/{e}.collisions.json"
                if os.path.exists(lg) and os.path.exists(cj):
                    reached = "termination cause: Success" in open(lg).read()
                    c = json.load(open(cj))["coll"]
                    out[(e, s)] = dict(s=reached and not [x for x in c if x[2] >= 2.0], s_all=reached and not c, n=len(c))
    return out


rng = np.random.default_rng(0)
P = load("proposed")
print("POST-HOC: final method with RGB-D vision, SocNavBench, all seeds with identical settings combined")
for f, label in (("s", "SUCCESS EXCLUDING SPAWN (primary metric)"), ("s_all", "SUCCESS, ALL COLLISIONS")):
    print(f"\n=== {label}")
    for m in ("fixed", "information gain", "decoupled"):
        B = load(m)
        keys = sorted(set(P) & set(B))
        seeds = sorted({s for _, s in keys})
        per = {}
        for (e, s) in keys:
            per.setdefault(e, []).append(int(P[(e, s)][f]) - int(B[(e, s)][f]))
        d = np.array([np.mean(v) for v in per.values()])
        flips = rng.choice([-1.0, 1.0], size=(100000, len(d)))
        p_perm = float(np.mean((flips * d).mean(1) >= d.mean()))
        lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(20000)], [2.5, 97.5])
        w = sum(P[k][f] and not B[k][f] for k in keys)
        l = sum(B[k][f] and not P[k][f] for k in keys)
        print(f"proposed {np.mean([P[k][f] for k in keys]):.3f} vs {m:16s} {np.mean([B[k][f] for k in keys]):.3f}  "
              f"diff {d.mean():+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]  p_perm = {p_perm:.4f}  pooled {w}:{l} "
              f"p = {binomtest(w, w + l, alternative='greater').pvalue:.4f}  episodes better:worse "
              f"{int((d > 0).sum())}:{int((d < 0).sum())}  (n = {len(keys)}, seeds {seeds[0]}-{seeds[-1]})")
