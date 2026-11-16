"""Success excluding spawn collisions (secondary metric) for the val30 / val30w SocNavBench runs.

success_nospawn = SocNavBench "Success" (goal reached) and no collision (centre distance < 0.44 m) with a pedestrian
who had existed for at least SPAWN_S = 2 s; collisions with pedestrians who had just appeared (often on top of the
robot) are left out. Also success_all with the same distance-based collision count, to check it against
SocNavBench's own count. Per-episode collision data are cached in results/socnav/<prefix>/collisions.json.
    python -m socnav_bridge.spawn_excluded val30w > results/socnav/val30w/summary_nospawn.txt
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import binomtest

from socnav_bridge.check_spawns import SPAWN_S, TAGS, episode
from socnav_bridge.episodes import EPISODES

NAMES = {"active": "proposed", "infogain": "infogain", "nearest": "nearest", "fixed": "fixed", "360": "360"}


def collect(prefix):
    cache = f"results/socnav/{prefix}/collisions.json"
    data = json.load(open(cache)) if os.path.exists(cache) else {}
    for tag in TAGS:
        for ep in EPISODES:
            key = f"{tag}/{ep}"
            if key in data:
                continue
            log = f"results/socnav/{prefix}_{tag}/{ep}.episode_log.txt"
            out = episode(tag, ep, prefix)
            if out is None or not os.path.exists(log):
                continue
            _, coll = out
            reached = "termination cause: Success" in open(log).read()
            data[key] = dict(reached=reached, coll=[[n, t, age, d0] for n, (t, age, d0) in coll.items()],
                             snb_coll=int(re.search(r"collided with (\d+)", open(log).read()).group(1)))
        json.dump(data, open(cache, "w"))
    return data


def mc(a, b, keys):
    w = sum(a[k] and not b[k] for k in keys)
    l = sum(b[k] and not a[k] for k in keys)
    return w, l, (binomtest(w, w + l).pvalue if w + l else 1.0)


def main(prefix):
    data = collect(prefix)
    S, SN = {}, {}
    for tag in TAGS:
        for ep in EPISODES:
            x = data.get(f"{tag}/{ep}")
            if x is None:
                continue
            nonspawn = [c for c in x["coll"] if c[2] >= SPAWN_S]
            S.setdefault(tag, {})[ep] = x["reached"] and not x["coll"]
            SN.setdefault(tag, {})[ep] = x["reached"] and not nonspawn
    print(f"=== {prefix}: success excluding collisions with pedestrians younger than {SPAWN_S:.0f} s")
    for mode, label in (("gt", "simulated camera on ground truth"), ("vis", "RGB-D vision")):
        print(f"\n--- {label}")
        print(f"{'method':10s} {'n':>3s} {'success(all coll)':>17s} {'success excl. spawn':>19s}")
        for m in ("active", "infogain", "nearest", "fixed", "360"):
            t = f"{mode}_{m}"
            if t in S:
                print(f"{NAMES[m]:10s} {len(S[t]):3d} {np.mean(list(S[t].values())):17.2f} {np.mean(list(SN[t].values())):19.2f}")
        P = SN.get(f"{mode}_active", {})
        for m in ("infogain", "nearest", "fixed"):
            B = SN.get(f"{mode}_{m}", {})
            keys = sorted(set(P) & set(B))
            if keys:
                w, l, p = mc(P, B, keys)
                print(f"  excl. spawn: proposed vs {NAMES[m]:9s} won:lost {w}:{l}  p = {p:.3f}")
    print("\n--- cost of real perception, excl. spawn (gt -> vision)")
    for m in ("active", "infogain", "nearest", "fixed"):
        G, V = SN.get(f"gt_{m}", {}), SN.get(f"vis_{m}", {})
        keys = sorted(set(G) & set(V))
        if keys:
            w, l, p = mc(G, V, keys)
            print(f"  {NAMES[m]:9s} {np.mean([G[k] for k in keys]):.2f} -> {np.mean([V[k] for k in keys]):.2f}  "
                  f"(gt better:vision better {w}:{l}, p = {p:.3f})")
    agree = [(len(x["coll"]) > 0) == (x["snb_coll"] > 0) for x in data.values()]
    print(f"\nchecks: distance-based collisions agree with SocNavBench's count (any vs none) in "
          f"{np.mean(agree):.1%} of {len(agree)} episodes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val30w")
