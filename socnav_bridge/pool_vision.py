"""Combined analysis of the final method with RGB-D vision over SocNavBench seeds 1-6 (decided after the seed 4-6
confirmation; not pre-registered). Final method = uncoupled unseen constraint: seeds 1-3 from the vision ablation
run abl_vis_open_constraint (same settings as the confirmation), seeds 4-6 from fin_vis_active. Baselines:
conf_vis_* (seeds 1-3) and fin_vis_* (seeds 4-6). Seeds of one episode are correlated, so besides the pooled
McNemar test we report an episode-level sign test (mean over the 6 seeds) and a cluster bootstrap over episodes.
    python -m socnav_bridge.pool_vision > results/final53/socnav_pooled.txt
"""
import numpy as np
from scipy.stats import binomtest

from socnav_bridge.analyze_ablation_vis import load as load3
from socnav_bridge.episodes import EPISODES


def load(prefix, seeds):
    import socnav_bridge.analyze_ablation_vis as a
    old = a.SEEDS
    a.SEEDS = seeds
    try:
        return load3(prefix)
    finally:
        a.SEEDS = old


PROP = {**load("abl_vis_open_constraint", (1, 2, 3)), **load("fin_vis_active", (4, 5, 6))}
BASE = {"fixed": {**load("conf_vis_fixed", (1, 2, 3)), **load("fin_vis_fixed", (4, 5, 6))},
        "infogain": {**load("conf_vis_infogain", (1, 2, 3)), **load("fin_vis_infogain", (4, 5, 6))}}
rng = np.random.default_rng(0)
print("Final method with RGB-D vision, SocNavBench seeds 1-6 combined (post hoc)")
for f, label in (("s", "success excluding spawn"), ("s_all", "success, all collisions")):
    print(f"\n=== {label}")
    print(f"proposed {np.mean([x[f] for x in PROP.values()]):.3f} (n = {len(PROP)})")
    for m, B in BASE.items():
        keys = sorted(set(PROP) & set(B))
        w = sum(PROP[k][f] and not B[k][f] for k in keys)
        l = sum(B[k][f] and not PROP[k][f] for k in keys)
        p = binomtest(w, w + l, alternative="greater").pvalue
        # episode level: mean over seeds per episode
        dep = {}
        for (e, s) in keys:
            dep.setdefault(e, []).append(int(PROP[(e, s)][f]) - int(B[(e, s)][f]))
        d = np.array([np.mean(v) for v in dep.values()])
        ew, el = int((d > 0).sum()), int((d < 0).sum())
        pe = binomtest(ew, ew + el, alternative="greater").pvalue
        boots = [rng.choice(d, len(d)).mean() for _ in range(20000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"vs {m:8s} {np.mean([B[k][f] for k in keys]):.3f}: pooled won:lost {w}:{l} (n = {len(keys)}) one-sided "
              f"p = {p:.4f} | episode level better:worse {ew}:{el} p = {pe:.4f} | difference {d.mean():+.3f}, "
              f"95% cluster-bootstrap CI [{lo:+.3f}, {hi:+.3f}]")
    for s0, s1 in ((1, 3), (4, 6)):
        sub = [k for k in PROP if s0 <= k[1] <= s1]
        print(f"  seeds {s0}-{s1}: proposed {np.mean([PROP[k][f] for k in sub]):.3f}, fixed "
              f"{np.mean([BASE['fixed'][k][f] for k in sub]):.3f}, infogain {np.mean([BASE['infogain'][k][f] for k in sub]):.3f}")
