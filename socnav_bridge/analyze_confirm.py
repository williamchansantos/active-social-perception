"""Analysis of results/socnav/confirm/PLAN.md (SocNavBench confirmation run, seeds 1-3).

    python -m socnav_bridge.analyze_confirm > results/socnav/confirm/summary.txt
    python -m socnav_bridge.analyze_confirm fin 4 5 6 > results/final53/socnav_summary.txt   # final method
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

from socnav_bridge.episodes import EPISODES

ROOT = "results/socnav"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "conf"
SEEDS = tuple(int(x) for x in sys.argv[2:]) or (1, 2, 3)
METHODS = {"proposed": "active", "infogain": "infogain", "fixed": "fixed"}
SPAWN_S = 2.0


def load(mode, method, seed):
    d = os.path.join(ROOT, f"{PREFIX}_{mode}_{METHODS[method]}_s{seed}")
    out = {}
    for e in EPISODES:
        lg, cj = os.path.join(d, f"{e}.episode_log.txt"), os.path.join(d, f"{e}.collisions.json")
        if not (os.path.exists(lg) and os.path.exists(cj)):
            continue
        reached = "termination cause: Success" in open(lg).read()
        c = json.load(open(cj))["coll"]
        out[(e, seed)] = dict(reached=reached, ncoll=len(c), nspawn=sum(x[2] < SPAWN_S for x in c),
                              s_all=reached and not c, s=reached and not [x for x in c if x[2] >= SPAWN_S])
    return out


def onesided_mcnemar(A, B, keys, f):
    w = sum(A[k][f] and not B[k][f] for k in keys)
    l = sum(B[k][f] and not A[k][f] for k in keys)
    return w, l, (binomtest(w, w + l, alternative="greater").pvalue if w + l else 1.0)


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def main():
    R = {}
    for mode in ("gt", "vis"):
        for m in METHODS:
            R[(mode, m)] = {}
            for s in SEEDS:
                R[(mode, m)].update(load(mode, m, s))
    for f, label in (("s", "SUCCESS EXCLUDING SPAWN (primary)"), ("s_all", "SUCCESS, ALL COLLISIONS (secondary)")):
        print(f"\n=== {label}")
        print(f"{'method':10s} {'gt n':>5s} {'gt':>6s} {'vis n':>6s} {'vision':>7s}   collisions gt / vision (spawn)")
        for m in METHODS:
            G, V = R[("gt", m)], R[("vis", m)]
            print(f"{m:10s} {len(G):5d} {np.mean([x[f] for x in G.values()] or [np.nan]):6.3f} {len(V):6d} "
                  f"{np.mean([x[f] for x in V.values()] or [np.nan]):7.3f}   "
                  f"{sum(x['ncoll'] for x in G.values())} ({sum(x['nspawn'] for x in G.values())}) / "
                  f"{sum(x['ncoll'] for x in V.values())} ({sum(x['nspawn'] for x in V.values())})")
        P = R[("vis", "proposed")]
        ps, txt = {}, {}
        for h, m in (("H1", "fixed"), ("H2", "infogain")):
            B = R[("vis", m)]
            keys = sorted(set(P) & set(B))
            w, l, p = onesided_mcnemar(P, B, keys, f)
            ps[h], txt[h] = p, f"vision: proposed > {m:8s} won:lost {w}:{l} (n = {len(keys)})"
        for h, m in (("H3", "fixed"), ("H4", "infogain")):
            Gp, Vp, Gb, Vb = R[("gt", "proposed")], R[("vis", "proposed")], R[("gt", m)], R[("vis", m)]
            keys = sorted(set(Gp) & set(Vp) & set(Gb) & set(Vb))
            cp = {k: int(Gp[k][f]) - int(Vp[k][f]) for k in keys}
            cb = {k: int(Gb[k][f]) - int(Vb[k][f]) for k in keys}
            less = sum(cp[k] < cb[k] for k in keys)
            more = sum(cp[k] > cb[k] for k in keys)
            p = binomtest(less, less + more, alternative="greater").pvalue if less + more else 1.0
            ps[h] = p
            txt[h] = (f"perception cost proposed < {m:8s} (gt - vision: proposed {np.mean(list(cp.values())):+.3f}, "
                      f"{m} {np.mean(list(cb.values())):+.3f}; smaller:larger {less}:{more}, n = {len(keys)})")
        hp = holm(ps)
        for h in ("H1", "H2", "H3", "H4"):
            print(f"  {h} {txt[h]}  p = {ps[h]:.4f}  p_holm = {hp[h]:.4f}" + ("   PRIMARY" if f == "s" else ""))
        # per-episode check: majority over the 3 seeds
        maj = {}
        for key in (("vis", "proposed"), ("vis", "fixed"), ("vis", "infogain")):
            maj[key] = {e: np.mean([R[key][(e, s)][f] for s in SEEDS if (e, s) in R[key]]) for e in EPISODES
                        if all((e, s) in R[key] for s in SEEDS)}
        for m in ("fixed", "infogain"):
            A, B = maj[("vis", "proposed")], maj[("vis", m)]
            keys = sorted(set(A) & set(B))
            w, l = sum(A[e] > B[e] for e in keys), sum(A[e] < B[e] for e in keys)
            p = binomtest(w, w + l, alternative="greater").pvalue if w + l else 1.0
            print(f"  per episode (mean over seeds), vision: proposed better:worse than {m:8s} {w}:{l}  p = {p:.3f}")
        print("  per dataset, vision:", "  ".join(
            f"{ds}: " + "/".join(f"{np.mean([x[f] for (e, s), x in R[('vis', m)].items() if ds in e.lower()] or [np.nan]):.2f}"
                                for m in METHODS) for ds in ("eth", "hotel", "univ", "zara1", "zara2")),
            "(proposed/infogain/fixed)")


if __name__ == "__main__":
    main()
