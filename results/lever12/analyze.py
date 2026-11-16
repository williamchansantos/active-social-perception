"""results/lever12/PLAN.md: each variant vs base, paired, on the tuning seed."""
import os, sys
import numpy as np
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(HERE, "..", "..")); sys.path.insert(0, ".")
from evaluate_nav import load
flag = lambda x: x if isinstance(x, bool) else str(x) in ("True", "1", "1.0")
def idx(v, planner):
    return {(r["scene"], int(r["episode"])): dict(s=flag(r["success_nospawn"]), s_all=flag(r["success"]),
            unseen=int(float(r["coll_untracked"])), tracked=int(float(r["coll_tracked"])), steps=float(r["steps"]))
            for r in load(f"{HERE}/{v}.csv") if r["planner"] == planner}
for planner, label in (("supermpc-active", "PROPOSED"), ("supermpc", "FIXED CAMERA")):
    print(f"=== {label} (seed 11, 1200 paired episodes)")
    print(f"{'variant':6s} {'success':>8s} {'all coll':>9s} {'Zara':>6s} {'ETH+Hot':>8s} {'never-seen':>11s} {'tracked':>8s} {'steps':>6s} {'won:lost vs base':>17s} {'p':>8s}")
    B = idx("base", planner)
    for v in ("base", "du01", "gcov", "both"):
        if not os.path.exists(f"{HERE}/{v}.csv"): continue
        A = idx(v, planner); k = sorted(set(A) & set(B))
        z = [x for x in k if x[0].startswith("zara")]; e = [x for x in k if not x[0].startswith("zara")]
        w = sum(A[x]["s"] and not B[x]["s"] for x in k); l = sum(B[x]["s"] and not A[x]["s"] for x in k)
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        print(f"{v:6s} {np.mean([A[x]['s'] for x in k]):8.3f} {np.mean([A[x]['s_all'] for x in k]):9.3f} "
              f"{np.mean([A[x]['s'] for x in z]):6.3f} {np.mean([A[x]['s'] for x in e]):8.3f} "
              f"{sum(A[x]['unseen'] for x in k):11d} {sum(A[x]['tracked'] for x in k):8d} {np.mean([A[x]['steps'] for x in k]):6.1f} "
              f"{w:8d}:{l:<8d} {p:8.4f}")
    print()
