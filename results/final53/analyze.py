"""Analysis of part A of results/final53/PLAN.md (2D replays, seed 53, final method).

    python results/final53/analyze.py > results/final53/summary.txt
"""
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load  # noqa: E402

P = {"proposed": "supermpc-active", "infogain": "supermpc-infogain", "nearest": "supermpc-nearest",
     "fixed": "supermpc", "gaze oracle": "supermpc-gazeoracle", "360": "supermpc-360"}
GROUPS = {"zara": ("Zara01+Zara02", ("zara01", "zara02")), "eth": ("ETH+Hotel", ("eth", "hotel"))}


def idx(rows, p):
    return {(r["scene"], int(r["episode"])): r for r in rows if r["planner"] == p}


def counts(A, B, keys, fn):
    w = sum(bool(fn(A[k])) and not fn(B[k]) for k in keys)
    l = sum(bool(fn(B[k])) and not fn(A[k]) for k in keys)
    return w, l


def coll(r, kind):
    v = r.get(f"coll_{kind}")
    return int(float(v)) if v not in (None, "") else 0


for tag, (label, scenes) in GROUPS.items():
    f = f"{HERE}/{tag}.csv"
    if not os.path.exists(f):
        continue
    rows = load(f)
    R = {m: idx(rows, p) for m, p in P.items()}
    keys = sorted(set.intersection(*[set(v) for v in R.values() if v]))
    print(f"\n=== {label} (seed 53, {len(keys)} paired episodes, 60 deg, final method: no observation-aware coupling)")
    cols = [k for k in rows[0].keys() if k.startswith("coll_")]
    print(f"{'method':12s} {'success':>8s} " + " ".join(f"{s:>7s}" for s in scenes) + f" {'reached':>8s}  collisions by kind")
    for m in P:
        r = R[m]
        sc = [np.mean([r[k]["success"] for k in keys if k[0] == s]) for s in scenes]
        kinds = {c[5:]: sum(coll(r[k], c[5:]) for k in keys) for c in cols}
        print(f"{m:12s} {np.mean([r[k]['success'] for k in keys]):8.3f} " + " ".join(f"{x:7.3f}" for x in sc)
              + f" {np.mean([r[k]['reached'] for k in keys]):8.3f}  {kinds}")
    succ = lambda r: r["success"]
    w1, l1 = counts(R["proposed"], R["nearest"], keys, succ)
    p1 = binomtest(w1, w1 + l1, alternative="greater").pvalue if w1 + l1 else 1.0
    w2, l2 = counts(R["proposed"], R["infogain"], keys, succ)
    p2 = binomtest(w2, w2 + l2).pvalue if w2 + l2 else 1.0
    ph = sorted([("nearest", p1), ("infogain", p2)], key=lambda x: x[1])
    holm = {ph[0][0]: min(1.0, 2 * ph[0][1])}
    holm[ph[1][0]] = max(holm[ph[0][0]], min(1.0, ph[1][1]))
    print(f"  PRIMARY proposed > nearest  (one-sided): {w1}:{l1}  p = {p1:.4f}  p_holm = {holm['nearest']:.4f}")
    print(f"  PRIMARY proposed vs infogain (two-sided): {w2}:{l2}  p = {p2:.4f}  p_holm = {holm['infogain']:.4f}")
    w3, l3 = counts(R["proposed"], R["fixed"], keys, succ)
    print(f"  secondary proposed > fixed (one-sided): {w3}:{l3}  p = "
          f"{binomtest(w3, w3 + l3, alternative='greater').pvalue if w3 + l3 else 1.0:.4f}")
    for o in ("gaze oracle", "360"):
        w, l = counts(R["proposed"], R[o], keys, succ)
        print(f"  oracle: proposed vs {o:11s} {w}:{l}  p = {binomtest(w, w + l).pvalue if w + l else 1.0:.4f}")
