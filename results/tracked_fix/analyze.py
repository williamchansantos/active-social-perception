"""Compare variants of the proposed method with the base proposed method and baselines on the tuning set
(seed 11, 600 episodes): success, and episodes with collisions against never-seen / tracked pedestrians."""
import glob, os, sys
import numpy as np
from scipy.stats import binomtest
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ".")
from evaluate_nav import load

def idx(path, planner):
    return {(r["scene"], int(r["episode"])): r for r in load(path) if r["planner"] == planner}

runs = {"proposed (base)": idx("results/tune600/obs_du02.csv", "supermpc-active"),
        "information gain": idx("results/tune600/base.csv", "supermpc-infogain"),
        "look at nearest": idx("results/tune600/oracles.csv", "supermpc-nearest")}
for f in sorted(glob.glob("results/tracked_fix/*.csv")):
    rows = idx(f, "supermpc-active")
    if len(rows) == 600:
        runs[os.path.basename(f)[:-4]] = rows
keys = sorted(set.intersection(*[set(v) for v in runs.values()]))
base = runs["proposed (base)"]
def pair(A, B, fn):
    a = int(sum(bool(fn(A[k])) and not fn(B[k]) for k in keys)); b = int(sum(bool(fn(B[k])) and not fn(A[k]) for k in keys))
    return a, b, (binomtest(a, a + b).pvalue if a + b else 1.0)
print(f"{len(keys)} paired episodes\n")
print(f"{'variant':20s} {'success':>8s} {'reached':>8s} {'never-seen coll eps':>20s} {'tracked coll eps':>17s} | vs base: success  /  never-seen  /  tracked")
for name, R in runs.items():
    rs = [R[k] for k in keys]
    s = f"{name:20s} {np.mean([r['success'] for r in rs]):8.3f} {np.mean([r['reached'] for r in rs]):8.3f} " \
        f"{sum(r['coll_untracked'] > 0 for r in rs):20d} {sum(r['coll_tracked'] > 0 for r in rs):17d}"
    if name != "proposed (base)":
        out = []
        for fn, good in ((lambda r: r["success"], True), (lambda r: r["coll_untracked"] > 0, False), (lambda r: r["coll_tracked"] > 0, False)):
            a, b, p = pair(R, base, fn)
            out.append(f"{a}:{b} p={p:.3f}")
        s += " | " + "  /  ".join(out)
    print(s)
print("\n(a:b = episodes where only the variant / only the base has it)")
