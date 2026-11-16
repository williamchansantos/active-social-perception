"""Analysis of the network-coasting rerun (see PLAN.md): core comparison and ablations."""
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load, wilson  # noqa: E402

P = {"proposed": "supermpc-active", "infogain": "supermpc-infogain", "nearest": "supermpc-nearest",
     "fixed": "supermpc", "gaze oracle": "supermpc-gazeoracle", "360": "supermpc-360"}
SC = {"zara": ("zara01", "zara02"), "eth": ("eth", "hotel")}


def idx(f, p):
    return {(r["scene"], int(r["episode"])): r for r in load(f) if r["planner"] == p} if os.path.exists(f) else {}


def mc(A, B, keys, fn=lambda r: r["success"]):
    a = int(sum(bool(fn(A[k])) and not fn(B[k]) for k in keys))
    b = int(sum(bool(fn(B[k])) and not fn(A[k]) for k in keys))
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)


def holm(items):
    out, running = {}, 0.0
    for i, (k, p) in enumerate(sorted(items.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


full_zara = None
for tag in ("zara", "eth"):
    runs = {m: idx(f"{HERE}/{tag}_{'obs' if m == 'proposed' else 'open'}.csv", p) for m, p in P.items()}
    old = {m: idx(f"results/rot60/{tag}_{'obs' if m == 'proposed' else 'open'}.csv", p) for m, p in P.items()}
    if not all(runs.values()):
        print(f"\n{tag}: incomplete")
        continue
    keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
    if tag == "zara":
        full_zara = runs["proposed"]
    s1, s2 = SC[tag]
    print(f"\n=== {tag.upper()} ({len(keys)} paired episodes, 60 deg, network coasting + CV propagation)")
    print(f"{'method':12s} {'success [95% CI]':>19s} {s1:>7s} {s2:>7s} {'reached':>8s} {'steps':>6s} {'never-seen':>11s} "
          f"{'tracked':>8s} {'CV coast':>9s} | proposed won:lost  p")
    prim = {}
    for m, X in runs.items():
        rs = [X[k] for k in keys]
        k_s = int(sum(r["success"] for r in rs))
        lo, hi = wilson(k_s, len(rs))
        z = [np.mean([r["success"] for r in rs if r["scene"] == s]) for s in (s1, s2)]
        o = old[m]
        old_s = np.mean([o[k]["success"] for k in keys if k in o]) if o else float("nan")
        line = (f"{m:12s} {k_s / len(rs):.3f} [{lo:.2f},{hi:.2f}] {z[0]:7.3f} {z[1]:7.3f} "
                f"{np.mean([r['reached'] for r in rs]):8.3f} {np.mean([r['steps'] for r in rs if r['reached']]):6.1f} "
                f"{sum(r['coll_untracked'] > 0 for r in rs):11d} {sum(r['coll_tracked'] > 0 for r in rs):8d} {old_s:9.3f}")
        if m != "proposed":
            a, b, p = mc(runs["proposed"], X, keys)
            line += f" | {a:3d}:{b:<3d} {p:.4f}"
            if m in ("infogain", "nearest"):
                prim[m] = p
        print(line)
    for m, ph in holm(prim).items():
        print(f"  PRIMARY proposed vs {m}: p_holm = {ph:.4f}")
    for m in ("proposed", "infogain"):
        a, b, p = mc(runs[m], old[m], keys)
        print(f"  {m}: network coasting vs CV coasting {a}:{b} p={p:.3f}")

ABL = [("open_constraint", "supermpc-active", "observation-aware coupling (open-loop constraint)"),
       ("no_unseen_constraint", "supermpc-active", "unseen-pedestrian constraint"),
       ("no_unseen_term", "active-tracked-only", "unseen-pedestrian perception term"),
       ("no_tracked_term", "active-unseen-only", "tracked-pedestrian perception term"),
       ("fixed_camera", "supermpc", "active gaze (fixed camera)"),
       ("kf_tracker", "active-kf", "PMB tracker (plain Kalman)"),
       ("prune25", "supermpc-active", "stale-track pruning (2.5 m)"),
       ("no_existence", "supermpc-active", "existence weighting"),
       ("no_robot_sens", "active-no-sens", "robot sensitivity"),
       ("no_state_cov", "active-no-statecov", "state covariance")]
if full_zara:
    rows = [(lab, idx(f"{HERE}/abl_{n}.csv", p)) for n, p, lab in ABL]
    rows = [(lab, R) for lab, R in rows if R]
    if rows:
        keys = sorted(set(full_zara).intersection(*[set(R) for _, R in rows]))
        print(f"\n=== ABLATIONS (Zara seed 31, {len(keys)} paired episodes; removed component)")
        rs = [full_zara[k] for k in keys]
        print(f"{'FULL METHOD':52s} {np.mean([r['success'] for r in rs]):.3f}  reached {np.mean([r['reached'] for r in rs]):.3f}  "
              f"never-seen {sum(r['coll_untracked'] > 0 for r in rs):3d}  tracked {sum(r['coll_tracked'] > 0 for r in rs):3d}")
        ps = {}
        lines = []
        for lab, R in rows:
            a, b, p = mc(full_zara, R, keys)
            ps[lab] = p
            rr = [R[k] for k in keys]
            lines.append((lab, f"{np.mean([r['success'] for r in rr]):.3f}  reached {np.mean([r['reached'] for r in rr]):.3f}  "
                               f"never-seen {sum(r['coll_untracked'] > 0 for r in rr):3d}  tracked {sum(r['coll_tracked'] > 0 for r in rr):3d}"
                               f"  full won:lost {a}:{b} p={p:.4f}"))
        h = holm(ps)
        for lab, txt in lines:
            print(f"{'- ' + lab:52s} {txt}  p_holm={h[lab]:.3f}")
