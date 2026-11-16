"""Final result tables, all from raw results, with SUCCESS COUNTING ALL COLLISIONS as the primary metric
(reached the goal in time with no robot-pedestrian collision at all); success excluding spawn collisions is
reported as a secondary column. Final method = no observation-aware coupling.

  A. 2D replays, Zara01+Zara02, seed 53 (results/final53/zara.csv; pre-registered, primary metric = all collisions)
  B. SocNavBench, RGB-D vision, final method, seeds 1-6 pooled (post hoc pooling; seeds 1-3 of the final method
     from the vision ablation run abl_vis_open_constraint, seeds 4-6 from the final confirmation)
  C. SocNavBench, simulated camera on true positions, final method, seeds 4-6
  D. SocNavBench vision ablations (full method with coupling, seeds 1-3) with all collisions

    python make_final_tables.py > results/final_tables.txt
    python make_final_tables.py --exclude-spawn > results/final_tables_nospawn.txt   # spawn-excluded as primary
"""
import sys

import numpy as np
from scipy.stats import binomtest

from evaluate_nav import load as load_csv
from socnav_bridge.analyze_ablation_vis import ABLATIONS
from socnav_bridge.analyze_confirm import load as load_conf


def mcnemar(A, B, keys, f, alt):
    w = sum(bool(A[k][f]) and not B[k][f] for k in keys)
    l = sum(bool(B[k][f]) and not A[k][f] for k in keys)
    return w, l, (binomtest(w, w + l, alternative=alt).pvalue if w + l else 1.0)


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


PRIM, SEC = ("s", "s_all") if "--exclude-spawn" in sys.argv else ("s_all", "s")
LABEL = "success excl. spawn" if PRIM == "s" else "success (all collisions)"
SECLABEL = "all coll." if PRIM == "s" else "excl. spawn"


def flag(x):
    return x if isinstance(x, bool) else str(x) in ("True", "1", "1.0")


# ---------------------------------------------------------------- A. 2D Zara seed 53
P2 = {"proposed": "supermpc-active", "information gain": "supermpc-infogain", "look-at-nearest": "supermpc-nearest",
      "fixed": "supermpc", "gaze oracle": "supermpc-gazeoracle", "360 camera": "supermpc-360"}
rows = load_csv("results/final53/zara.csv")
R = {m: {(r["scene"], int(r["episode"])): dict(s_all=flag(r["success"]), s=flag(r["success_nospawn"]),
                                               unseen=int(float(r["coll_untracked"])), tracked=int(float(r["coll_tracked"])))
         for r in rows if r["planner"] == p} for m, p in P2.items()}
keys = sorted(set.intersection(*[set(v) for v in R.values()]))
print("=" * 100)
print(f"A. 2D REPLAYS, Zara01+Zara02, seed 53, {len(keys)} paired episodes (60 deg FOV, 4 m) -- pre-registered")
print("=" * 100)
print(f"PRIMARY METRIC: {LABEL}; secondary: {SECLABEL}")
print(f"{'gaze':18s} {'success':>8s} {SECLABEL:>12s} {'never-seen coll.':>17s} {'tracked coll.':>14s}")
for m in P2:
    r = R[m]
    print(f"{m:18s} {np.mean([r[k][PRIM] for k in keys]):8.3f} {np.mean([r[k][SEC] for k in keys]):12.3f} "
          f"{sum(r[k]['unseen'] for k in keys):17d} {sum(r[k]['tracked'] for k in keys):14d}")
pa = {}
for o, alt in (("look-at-nearest", "greater"), ("information gain", "two-sided")):
    pa[o] = mcnemar(R["proposed"], R[o], keys, PRIM, alt)
hp = holm({o: v[2] for o, v in pa.items()})
for o, (w, l, p) in pa.items():
    print(f"  PRIMARY proposed vs {o:16s} won:lost {w}:{l}  p = {p:.4f}  p_Holm = {hp[o]:.4f}")
for o, alt in (("fixed", "greater"), ("gaze oracle", "two-sided"), ("360 camera", "two-sided")):
    w, l, p = mcnemar(R["proposed"], R[o], keys, PRIM, alt)
    print(f"  secondary proposed vs {o:14s} won:lost {w}:{l}  p = {p:.4g}")

# ---------------------------------------------------------------- B. SocNavBench vision pooled seeds 1-6
import socnav_bridge.analyze_ablation_vis as av  # noqa: E402


def load_abl(prefix, seeds):
    old, av.SEEDS = av.SEEDS, seeds
    try:
        return av.load(prefix)
    finally:
        av.SEEDS = old


PROP = {**load_abl("abl_vis_open_constraint", (1, 2, 3)), **load_abl("fin_vis_active", (4, 5, 6))}
BASE = {"information gain": {**load_abl("conf_vis_infogain", (1, 2, 3)), **load_abl("fin_vis_infogain", (4, 5, 6))},
        "fixed": {**load_abl("conf_vis_fixed", (1, 2, 3)), **load_abl("fin_vis_fixed", (4, 5, 6))}}
rng = np.random.default_rng(0)
print("\n" + "=" * 100)
print(f"B. SOCNAVBENCH, RGB-D VISION, final method, seeds 1-6 pooled ({len(PROP)} paired episode-seeds) -- post hoc")
print("=" * 100)
print(f"{'gaze':18s} {'success':>8s} {SECLABEL:>12s} {'collisions':>11s}")
for m, D in [("proposed", PROP)] + list(BASE.items()):
    print(f"{m:18s} {np.mean([x[PRIM] for x in D.values()]):8.3f} {np.mean([x[SEC] for x in D.values()]):12.3f} "
          f"{sum(x['ncoll'] for x in D.values()):11d}")
for m, B in BASE.items():
    k = sorted(set(PROP) & set(B))
    w, l, p = mcnemar(PROP, B, k, PRIM, "greater")
    dep = {}
    for (e, s) in k:
        dep.setdefault(e, []).append(int(PROP[(e, s)][PRIM]) - int(B[(e, s)][PRIM]))
    d = np.array([np.mean(v) for v in dep.values()])
    ew, el = int((d > 0).sum()), int((d < 0).sum())
    pe = binomtest(ew, ew + el, alternative="greater").pvalue
    lo, hi = np.percentile([rng.choice(d, len(d)).mean() for _ in range(20000)], [2.5, 97.5])
    print(f"  proposed vs {m:16s} {d.mean():+.3f} [95% cluster-bootstrap CI {lo:+.3f}, {hi:+.3f}]; pooled won:lost {w}:{l} "
          f"p = {p:.4f}; episodes better:worse {ew}:{el} p = {pe:.4f}")
for s0, s1, src in ((1, 3, "pre-registered H1/H2 were run with the coupled variant"), (4, 6, "pre-registered, final method")):
    sub = [k for k in PROP if s0 <= k[1] <= s1]
    parts = [f"proposed {np.mean([PROP[k][PRIM] for k in sub]):.3f}"] + \
            [f"{m} {np.mean([B[k][PRIM] for k in sub]):.3f}" for m, B in BASE.items()]
    tests = []
    for m, B in BASE.items():
        w, l, p = mcnemar(PROP, B, sub, PRIM, "greater")
        tests.append(f"vs {m} {w}:{l} p={p:.3f}")
    print(f"  seeds {s0}-{s1}: {', '.join(parts)}; {'; '.join(tests)}  ({src})")

# ---------------------------------------------------------------- C. SocNavBench simulated camera, seeds 4-6
import socnav_bridge.analyze_confirm as ac  # noqa: E402

ac.PREFIX, ac.SEEDS = "fin", (4, 5, 6)
G = {m: {} for m in ac.METHODS}
for m in ac.METHODS:
    for s in ac.SEEDS:
        G[m].update(load_conf("gt", m, s))
print("\n" + "=" * 100)
print(f"C. SOCNAVBENCH, SIMULATED CAMERA ON TRUE POSITIONS, final method, seeds 4-6 ({len(G['proposed'])} paired)")
print("=" * 100)
print(f"{'gaze':18s} {'success':>8s} {SECLABEL:>12s} {'collisions':>11s}")
for m, label in (("proposed", "proposed"), ("infogain", "information gain"), ("fixed", "fixed")):
    D = G[m]
    print(f"{label:18s} {np.mean([x[PRIM] for x in D.values()]):8.3f} {np.mean([x[SEC] for x in D.values()]):12.3f} "
          f"{sum(x['ncoll'] for x in D.values()):11d}")
for m, label in (("infogain", "information gain"), ("fixed", "fixed")):
    k = sorted(set(G["proposed"]) & set(G[m]))
    w, l, p = mcnemar(G["proposed"], G[m], k, PRIM, "two-sided")
    print(f"  proposed vs {label:16s} won:lost {w}:{l}  p = {p:.4f} (two-sided)")

# ---------------------------------------------------------------- D. vision ablations, all collisions
F = av.load("conf_vis_active")
print("\n" + "=" * 100)
print(f"D. SOCNAVBENCH VISION ABLATIONS (full method incl. coupling, seeds 1-3, {len(F)} paired), {LABEL}")
print("=" * 100)
print(f"full method: {LABEL} {np.mean([x[PRIM] for x in F.values()]):.3f} "
      f"({SECLABEL} {np.mean([x[SEC] for x in F.values()]):.3f})")
res, ps = [], {}
for name, label in ABLATIONS:
    A = av.load(f"abl_vis_{name}")
    k = sorted(set(F) & set(A))
    w, l, p = mcnemar(F, A, k, PRIM, "two-sided")
    ps[name] = p
    res.append((name, label, np.mean([A[x][PRIM] for x in k]), np.mean([A[x][SEC] for x in k]), w, l, p))
hp = holm(ps)
print(f"{'removed component':58s} {'success':>8s} {SECLABEL[:8]:>8s} {'full won:lost':>14s} {'p':>7s} {'p_Holm':>7s}")
for name, label, sa, ss, w, l, p in sorted(res, key=lambda r: r[2]):
    print(f"- {label:56s} {sa:8.3f} {ss:8.3f} {w:7d}:{l:<6d} {p:7.4f} {hp[name]:7.4f}")
