"""How many pedestrians appear mid-scene, and how many robot collisions involve a pedestrian who had just appeared?

Uses SocNavBench's saved simulator states (every 0.05 s) of the val30w batches. A collision is the first time the
robot-pedestrian centre distance drops below 0.44 m (robot 0.24 + pedestrian 0.2), as SocNavBench counts it;
"spawn" collisions are those within SPAWN_S seconds of the pedestrian's first appearance.
    python -m socnav_bridge.check_spawns [prefix=val30w]
"""
import glob
import json
import os
import re
import sys

import numpy as np

from socnav_bridge.episodes import EPISODES

SNB = "C:/Users/James/Desktop/SocNavBench/tests/socnav"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "val30w"
TAGS = ["gt_active", "gt_infogain", "gt_nearest", "gt_fixed", "gt_360",
        "vis_active", "vis_infogain", "vis_nearest", "vis_fixed"]
SPAWN_S = 2.0


def episode(tag, ep, prefix=None):
    files = sorted(glob.glob(f"{SNB}/test_supermpc_{prefix or PREFIX}_{tag}/{ep}/sim_state_data/sim_state_*.json"),
                   key=lambda f: float(os.path.basename(f)[10:-5]))
    if not files:
        return None
    first, coll = {}, {}
    for f in files:
        s = json.load(open(f))
        t = float(s["sim_t"])
        if not s["robots"]:
            continue
        r = np.array(list(s["robots"].values())[0]["current_config"][:2])
        for n, a in s["pedestrians"].items():
            q = np.array(a["current_config"][:2])
            d = float(np.linalg.norm(q - r))
            if n not in first:
                first[n] = (t, d)
            if d < 0.44 and n not in coll:
                coll[n] = (t, t - first[n][0], first[n][1])
    return first, coll


def main():
    print(f"{PREFIX}: pedestrians appearing after the episode start, and collisions (centre distance < 0.44 m)")
    print(f"{'batch':14s} {'appear>0s':>9s} {'<4m':>5s} {'<2m':>5s} | {'coll':>4s} {'spawn<2s':>8s} "
          f"{'appeared<4m':>11s}   spawn collisions: episode, pedestrian (age s, distance at appearance m)")
    for tag in TAGS:
        n_app = n4 = n2 = n_coll = n_sp = n_near = 0
        ex = []
        for ep in EPISODES:
            out = episode(tag, ep)
            if out is None:
                continue
            first, coll = out
            for n, (t, d) in first.items():
                if t > 0.06:
                    n_app += 1
                    n4 += d < 4.0
                    n2 += d < 2.0
            for n, (t, age, d0) in coll.items():
                n_coll += 1
                if age < SPAWN_S:
                    n_sp += 1
                    ex.append(f"{ep[2:]}:{n[-4:]}({age:.1f}s,{d0:.1f}m)")
                n_near += first[n][0] > 0.06 and d0 < 4.0
        print(f"{tag:14s} {n_app:9d} {n4:5d} {n2:5d} | {n_coll:4d} {n_sp:8d} {n_near:11d}   {' '.join(ex[:6])}")


if __name__ == "__main__":
    main()


def summarize_dir(sim_dir, out_json):
    """Per-episode collision / appearance summary from one episode's sim_state_data (used by run_batch.sh)."""
    files = sorted(glob.glob(os.path.join(sim_dir, "sim_state_*.json")), key=lambda f: float(os.path.basename(f)[10:-5]))
    first, coll = {}, {}
    for f in files:
        s = json.load(open(f))
        if not s["robots"]:
            continue
        t = float(s["sim_t"])
        r = np.array(list(s["robots"].values())[0]["current_config"][:2])
        for n, a in s["pedestrians"].items():
            d = float(np.linalg.norm(np.array(a["current_config"][:2]) - r))
            if n not in first:
                first[n] = (t, d)
            if d < 0.44 and n not in coll:
                coll[n] = (t, t - first[n][0], first[n][1])
    json.dump(dict(n_states=len(files), coll=[[n, t, age, d0] for n, (t, age, d0) in coll.items()],
                   appear=[[n, t, d] for n, (t, d) in first.items()]), open(out_json, "w"))
