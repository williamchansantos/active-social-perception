"""Analysis of results/socnav/val30/PLAN.md: SocNavBench curated episodes, ground-truth observation vs RGB-D vision.

    python -m socnav_bridge.analyze_val30 > results/socnav/val30/summary.txt
    python -m socnav_bridge.analyze_val30 val30w > results/socnav/val30w/summary.txt   # with walls
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import binomtest

from socnav_bridge.episodes import EPISODES

ROOT = "results/socnav"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "val30"
METHODS = {"proposed": "active", "infogain": "infogain", "nearest": "nearest", "fixed": "fixed", "360": "360"}


def load(mode, method):
    d = os.path.join(ROOT, f"{PREFIX}_{mode}_{METHODS[method]}")
    if not os.path.isdir(d):
        return {}
    out = {}
    for e in EPISODES:
        js = glob.glob(os.path.join(d, f"{e}_*.json"))
        lg = os.path.join(d, f"{e}.episode_log.txt")
        if not js or not os.path.exists(lg):
            continue
        J = json.load(open(js[0]))
        txt = open(lg).read()
        cause = re.search(r"Robot termination cause: (\w+)", txt).group(1)
        coll = int(re.search(r"Robot collided with (\d+) agent", txt).group(1))
        L = J["log"]
        n_det = [len(x) for x in L.get("detected", [])]
        out[e] = dict(reached=cause == "Success", cause=cause, collisions=coll, success=cause == "Success" and coll == 0,
                      steps=J["steps"], min_clear=J["min_clearance"], mean_det=float(np.mean(n_det)) if n_det else 0.0)
    return out


def mcnemar(A, B, keys, f="success"):
    a = sum(bool(A[k][f]) and not B[k][f] for k in keys)
    b = sum(bool(B[k][f]) and not A[k][f] for k in keys)
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)


def main():
    R = {(m, k): load(m, k) for m in ("gt", "vis") for k in METHODS}
    for mode, label in (("gt", "GROUND-TRUTH OBSERVATION (simulated camera on true positions)"),
                        ("vis", "RGB-D VISION (rendered images, detector, depth, association)")):
        print(f"\n=== {label}")
        print(f"{'method':10s} {'n':>3s} {'success':>8s} {'reached':>8s} {'coll-free':>9s} {'collisions':>10s} "
              f"{'steps(succ)':>11s} {'min clr':>8s}   per dataset success")
        for k in METHODS:
            r = R[(mode, k)]
            if not r:
                continue
            v = list(r.values())
            per = {}
            for e, x in r.items():
                ds = e.split("_")[1].lower().rstrip("12") if not e.startswith("t_ETH") else "eth"
                ds = {"zara": "zara1" if "zara1" in e else "zara2"}.get(ds, ds)
                per.setdefault(ds, []).append(x["success"])
            ps = "  ".join(f"{d} {np.mean(s):.2f}" for d, s in sorted(per.items()))
            print(f"{k:10s} {len(v):3d} {np.mean([x['success'] for x in v]):8.2f} {np.mean([x['reached'] for x in v]):8.2f} "
                  f"{np.mean([x['collisions'] == 0 for x in v]):9.2f} {sum(x['collisions'] for x in v):10d} "
                  f"{np.mean([x['steps'] for x in v if x['success']] or [np.nan]):11.1f} "
                  f"{np.mean([x['min_clear'] for x in v]):8.2f}   {ps}")
        P = R[(mode, "proposed")]
        for k in ("infogain", "nearest", "fixed"):
            B = R[(mode, k)]
            keys = sorted(set(P) & set(B))
            if keys:
                a, b, p = mcnemar(P, B, keys)
                print(f"  proposed vs {k:9s} success won:lost {a}:{b}  p = {p:.3f}  (n = {len(keys)})")
    print("\n=== COST OF REAL PERCEPTION (gt -> vision, same method and episode)")
    for k in ("proposed", "infogain", "nearest", "fixed"):
        G, V = R[("gt", k)], R[("vis", k)]
        keys = sorted(set(G) & set(V))
        if keys:
            a, b, p = mcnemar(G, V, keys)
            print(f"  {k:9s} success gt {np.mean([G[e]['success'] for e in keys]):.2f} -> vision "
                  f"{np.mean([V[e]['success'] for e in keys]):.2f}  (gt better:vision better {a}:{b}, p = {p:.3f}); "
                  f"collisions {sum(G[e]['collisions'] for e in keys)} -> {sum(V[e]['collisions'] for e in keys)}; "
                  f"detections/step {np.mean([G[e]['mean_det'] for e in keys]):.2f} -> {np.mean([V[e]['mean_det'] for e in keys]):.2f}")
    print("\n=== PER EPISODE (S = success, R = reached with collisions, T = timeout; digit = collisions)")
    cols = [(m, k) for m in ("gt", "vis") for k in METHODS if R[(m, k)]]
    print(f"{'episode':28s} " + " ".join(f"{m[0]}-{k[:4]:4s}" for m, k in cols))
    for e in EPISODES:
        cells = []
        for c in cols:
            x = R[c].get(e)
            if x is None:
                cells.append("   -  ")
            else:
                tag = "S" if x["success"] else ("R" if x["reached"] else ("T" if x["cause"] == "Timeout" else x["cause"][0]))
                cells.append(f"  {tag}{x['collisions'] if x['collisions'] else ' '}  ")
        print(f"{e:28s} " + " ".join(cells))


if __name__ == "__main__":
    main()
