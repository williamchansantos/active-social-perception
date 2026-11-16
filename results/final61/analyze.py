"""Pre-registered analysis of results/final61/PLAN.md (written before any results).

    python results/final61/analyze.py A   # part A (2D); writes results/final61/socnav_methods.txt (top 3)
    python results/final61/analyze.py B   # part B (SocNavBench vision, seeds 7-16)
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ".")

PRIM, SEC = "s", "s_all"  # success excl. spawn (primary), all collisions (secondary)


def holm(ps):
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def mcnemar(A, B, keys, f, alt):
    w = sum(bool(A[k][f]) and not B[k][f] for k in keys)
    l = sum(bool(B[k][f]) and not A[k][f] for k in keys)
    return w, l, (binomtest(w, w + l, alternative=alt).pvalue if w + l else 1.0)


def flag(x):
    return x if isinstance(x, bool) else str(x) in ("True", "1", "1.0")


# ------------------------------------------------------------------------------------------- part A
METHODS_A = {"proposed": "supermpc-active", "information gain": "supermpc-infogain", "TTC heuristic": "supermpc-ttc",
             "look-at-nearest": "supermpc-nearest", "decoupled": "supermpc-decoupled", "scan": "supermpc-scan",
             "fixed": "supermpc", "gaze oracle": "supermpc-gazeoracle", "360 camera": "supermpc-360"}
ABLATIONS_A = {"no unseen constraint": ("abl_no_unseen_constraint", "supermpc-active"),
               "no unseen gaze term": ("abl_no_unseen_term", "active-tracked-only"),
               "no tracked gaze term": ("abl_no_tracked_term", "active-unseen-only"),
               "plain Kalman tracker": ("abl_kf_tracker", "active-kf"),
               "no stale-track pruning": ("abl_prune25", "supermpc-active"),
               "no existence weighting": ("abl_no_existence", "supermpc-active"),
               "no robot-uncertainty propagation": ("abl_no_robot_sens", "active-no-sens"),
               "no track state covariance": ("abl_no_state_cov", "active-no-statecov")}
CANDIDATES_B = ["information gain", "TTC heuristic", "look-at-nearest", "decoupled", "scan", "fixed"]
SOCNAV_CAMERA = {"information gain": "infogain", "TTC heuristic": "ttc", "look-at-nearest": "nearest",
                 "decoupled": "decoupled", "scan": "scan", "fixed": "fixed"}


def load_a(fname, planner):
    from evaluate_nav import load
    out = {}
    for r in load(f"{HERE}/{fname}.csv"):
        if r["planner"] == planner:
            out[(r["scene"], int(r["episode"]))] = dict(s=flag(r["success_nospawn"]), s_all=flag(r["success"]),
                                                       unseen=int(float(r["coll_untracked"])),
                                                       tracked=int(float(r["coll_tracked"])), spawn=int(float(r["coll_spawn"])))
    return out


def part_a():
    R = {m: load_a("main", p) for m, p in METHODS_A.items()}
    keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    print(f"=== PART A: 2D replays, seed 61, {len(keys)} paired episodes (Zara01, Zara02, ETH, Hotel)")
    print(f"{'method':18s} {'success':>8s} {'all coll.':>10s} {'Zara':>6s} {'ETH+Hot':>8s} {'never-seen':>11s} {'tracked':>8s} {'spawn':>6s}")
    for m in METHODS_A:
        r = R[m]
        z = [k for k in keys if k[0].startswith("zara")]
        e = [k for k in keys if not k[0].startswith("zara")]
        print(f"{m:18s} {np.mean([r[k][PRIM] for k in keys]):8.3f} {np.mean([r[k][SEC] for k in keys]):10.3f} "
              f"{np.mean([r[k][PRIM] for k in z]):6.3f} {np.mean([r[k][PRIM] for k in e]):8.3f} "
              f"{sum(r[k]['unseen'] for k in keys):11d} {sum(r[k]['tracked'] for k in keys):8d} {sum(r[k]['spawn'] for k in keys):6d}")
    prim = {}
    for o in ["TTC heuristic", "look-at-nearest", "decoupled", "scan", "fixed"]:
        prim[o] = mcnemar(R["proposed"], R[o], keys, PRIM, "greater")
    prim["information gain"] = mcnemar(R["proposed"], R["information gain"], keys, PRIM, "two-sided")
    hp = holm({o: v[2] for o, v in prim.items()})
    print("\nPRIMARY (success excl. spawn, Holm over six)")
    for o, (w, l, p) in prim.items():
        side = "two-sided" if o == "information gain" else "one-sided"
        print(f"  proposed vs {o:16s} won:lost {w:3d}:{l:<3d} p = {p:.4g} ({side})  p_Holm = {hp[o]:.4g}")
    print("\nsecondary: per scene group and all collisions")
    for grp, sel in (("Zara", lambda k: k[0].startswith("zara")), ("ETH+Hotel", lambda k: not k[0].startswith("zara"))):
        kk = [k for k in keys if sel(k)]
        txt = []
        for o in ["information gain", "TTC heuristic", "look-at-nearest", "decoupled", "scan", "fixed"]:
            w, l, p = mcnemar(R["proposed"], R[o], kk, PRIM, "two-sided")
            txt.append(f"{o} {w}:{l} p={p:.3g}")
        print(f"  {grp}: " + "; ".join(txt))
    txt = []
    for o in ["information gain", "TTC heuristic", "look-at-nearest", "decoupled", "scan", "fixed", "gaze oracle", "360 camera"]:
        w, l, p = mcnemar(R["proposed"], R[o], keys, SEC, "two-sided")
        txt.append(f"{o} {w}:{l} p={p:.3g}")
    print("  all collisions: " + "; ".join(txt))
    # ablations
    print("\nABLATION FAMILY (two-sided, Holm over eight), full method = proposed above")
    res, ps = {}, {}
    for name, (fname, planner) in ABLATIONS_A.items():
        if not os.path.exists(f"{HERE}/{fname}.csv"):
            continue
        A = load_a(fname, planner)
        kk = sorted(set(R["proposed"]) & set(A))
        res[name] = (np.mean([A[k][PRIM] for k in kk]), np.mean([A[k][SEC] for k in kk]),
                     *mcnemar(R["proposed"], A, kk, PRIM, "two-sided"), len(kk))
        ps[name] = res[name][4]
    hp = holm(ps)
    full = np.mean([R["proposed"][k][PRIM] for k in keys])
    print(f"  full method {full:.3f}")
    for name, (sa, sall, w, l, p, n) in sorted(res.items(), key=lambda kv: kv[1][0]):
        print(f"  - {name:34s} {sa:.3f} (all coll. {sall:.3f})  full won:lost {w}:{l}  p = {p:.4g}  p_Holm = {hp[name]:.4g}  n = {n}")
    # selection of the three SocNavBench alternatives (pre-registered rule)
    ranked = sorted(CANDIDATES_B, key=lambda m: -np.mean([R[m][k][PRIM] for k in keys]))
    top3 = ranked[:3]
    print("\nSocNavBench alternatives (top 3 by pooled 2D success excl. spawn): " + ", ".join(top3))
    with open(f"{HERE}/socnav_methods.txt", "w") as f:
        f.write(" ".join(SOCNAV_CAMERA[m] for m in top3) + "\n")


# ------------------------------------------------------------------------------------------- part B
def load_b(cam, seeds):
    out = {}
    for s in seeds:
        d = f"results/socnav/f61_vis_{cam}_s{s}"
        from socnav_bridge.episodes import EPISODES
        for e in EPISODES:
            lg, cj = f"{d}/{e}.episode_log.txt", f"{d}/{e}.collisions.json"
            if not (os.path.exists(lg) and os.path.exists(cj)):
                continue
            reached = "termination cause: Success" in open(lg).read()
            c = json.load(open(cj))["coll"]
            out[(e, s)] = dict(s=reached and not [x for x in c if x[2] >= 2.0], s_all=reached and not c,
                               ncoll=len(c))
    return out


def part_b():
    seeds = range(7, 17)
    others = open(f"{HERE}/socnav_methods.txt").read().split()
    P = load_b("active", seeds)
    rng = np.random.default_rng(0)
    print(f"=== PART B: SocNavBench, RGB-D vision, seeds 7-16 ({len(P)} runs of the proposed method)")
    print(f"proposed: success {np.mean([x[PRIM] for x in P.values()]):.3f} (all coll. {np.mean([x[SEC] for x in P.values()]):.3f}),"
          f" collisions {sum(x['ncoll'] for x in P.values())}")
    ps, rows = {}, {}
    for cam in others:
        B = load_b(cam, seeds)
        keys = sorted(set(P) & set(B))
        per = {}
        for (e, s) in keys:
            per.setdefault(e, []).append(int(P[(e, s)][PRIM]) - int(B[(e, s)][PRIM]))
        d = np.array([np.mean(v) for v in per.values()])
        flips = rng.choice([-1.0, 1.0], size=(100000, len(d)))
        p_perm = float(np.mean((flips * d).mean(1) >= d.mean()))
        boots = [rng.choice(d, len(d)).mean() for _ in range(20000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        w, l, p_mc = mcnemar(P, B, keys, PRIM, "greater")
        w2, l2, p2 = mcnemar(P, B, keys, SEC, "greater")
        ps[cam] = p_perm
        rows[cam] = (np.mean([B[k][PRIM] for k in keys]), np.mean([B[k][SEC] for k in keys]), d.mean(), lo, hi,
                     p_perm, w, l, p_mc, w2, l2, p2, len(keys), sum(B[k]["ncoll"] for k in keys))
    hp = holm(ps)
    print("\nPRIMARY: proposed > alternative (one-sided episode-level permutation test, Holm over three)")
    for cam, (sb, sball, dm, lo, hi, pp, w, l, pm, w2, l2, p2, n, nc) in rows.items():
        print(f"  vs {cam:10s} success {sb:.3f} (all coll. {sball:.3f}, collisions {nc}); difference {dm:+.3f} "
              f"[95% cluster-bootstrap CI {lo:+.3f}, {hi:+.3f}]  p_perm = {pp:.4g}  p_Holm = {hp[cam]:.4g}  (n = {n})")
        print(f"      secondary: pooled McNemar {w}:{l} p = {pm:.4g}; all collisions {w2}:{l2} p = {p2:.4g}")


if __name__ == "__main__":
    part_a() if (sys.argv[1:] or ["A"])[0] == "A" else part_b()
