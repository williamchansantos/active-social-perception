"""How late are pedestrians detected, and why?

Reruns evaluation episodes (identical seeds) with a per-step hook. A close
encounter is a pedestrian who comes within --close m of the robot. For each, at
the step before the encounter's most recent (re)detection it records: distance at
detection, bearing relative to the robot's heading and to the camera, whether the
pedestrian had been tracked before and was dropped (re-detection) or never seen,
and the undetected-pedestrian density the PMB held within 1 m of where the
pedestrian actually was (did the tracker expect someone there?).

    python diagnose_late.py --planners supermpc-active supermpc-360 --seed 11 --episodes 100
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from evaluate_nav import SCENES, _WORKER, _init_worker, _predictor, checkpoint_for, make_episodes, scene_frames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planners", nargs="+", default=["supermpc-active", "supermpc-360"])
    p.add_argument("--scenes", nargs="+", default=["zara01", "zara02"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--range", type=float, default=4.0)
    p.add_argument("--delta-unseen", type=float, default=0.02)
    p.add_argument("--prune", type=float, default=1.0)
    p.add_argument("--close", type=float, default=1.5, help="m, close-encounter distance")
    p.add_argument("--workers", type=int, default=13)
    return p.parse_args()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _job(ep, planner, opts):
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    sc = ep["scene"]
    if sc not in _WORKER["priors"]:
        first, _, _, split = scene_frames(sc)
        _WORKER["priors"][sc] = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=opts["delta_unseen"],
                           fov=opts["fov"], cam_range=opts["range"], mpc_overrides=dict(unseen_joint=False),
                           pmb_overrides=dict(prune_pos_std=opts["prune"]))
    recs = []

    def hook(step, x, peds, tracks, tracker, u, **_):
        dens = tracker.undetected_density()  # (nx, ny) expected never-detected pedestrians per cell
        g = tracker.grid
        near_mass = {}
        for pid, q in peds.items():
            if pid in tracks:
                continue
            d2 = np.sum((tracker.centers - q) ** 2, -1)
            near_mass[pid] = float(dens[d2 < 1.0].sum())
        recs.append(dict(step=step, x=x.copy(), u=np.array(u), peds={k: v.copy() for k, v in peds.items()},
                         tracked=set(tracks), near_mass=near_mass))

    seed = opts["seed"] + 1000 * list(SCENES).index(sc) + ep["episode"]
    res, _ = run_episode(planner, PedestrianScene(SCENES[sc]["path"], ep["frame"]), ep["start"], ep["goal"],
                         _predictor(checkpoint_for(sc, "clean", None)), args, seed, _WORKER["priors"][sc], hook=hook)
    events = []
    first_close = {}
    for i, rec in enumerate(recs):
        for pid, q in rec["peds"].items():
            if pid not in first_close and np.linalg.norm(q - rec["x"][:2]) < opts["close"]:
                first_close[pid] = i
    for pid, ic in first_close.items():
        # most recent (re)detection at or before the close encounter
        idet = None
        for i in range(ic, -1, -1):
            if pid in recs[i]["tracked"] and (i == 0 or pid not in recs[i - 1]["tracked"]):
                idet = i
                break
        if idet is None and pid not in recs[ic]["tracked"]:
            events.append(dict(kind="untracked at encounter", det_dist=0.0))
            continue
        if idet is None or idet == 0:
            continue  # tracked from the start: not informative
        prev = recs[idet - 1]
        q, x = prev["peds"].get(pid), prev["x"]
        if q is None:
            events.append(dict(kind="entered scene", det_dist=float(np.linalg.norm(recs[idet]["peds"][pid] - recs[idet]["x"][:2]))))
            continue
        d = q - x[:2]
        bear_body = np.degrees(wrap(np.arctan2(d[1], d[0]) - x[2]))
        bear_cam = np.degrees(wrap(np.arctan2(d[1], d[0]) - x[2] - x[3]))
        seen_before = any(pid in r["tracked"] for r in recs[: idet - 1])
        dist = float(np.linalg.norm(d))
        why = ("beyond range" if dist > opts["range"] else "outside FOV" if abs(bear_cam) > opts["fov"] / 2
               else "in view (missed/occluded)")
        events.append(dict(kind="re-detected" if seen_before else "first detection", det_dist=float(
            np.linalg.norm(recs[idet]["peds"][pid] - recs[idet]["x"][:2])), bear_body=bear_body, bear_cam=bear_cam,
            why=why, mass=prev["near_mass"].get(pid, np.nan), lead=ic - idet, speed=float(prev["u"][0])))
    return planner, res, events


def main():
    args = parse_args()
    opts = dict(seed=args.seed, delta_unseen=args.delta_unseen, fov=args.fov, range=args.range, prune=args.prune,
                close=args.close)
    jobs = [(ep, pl) for sc in args.scenes for ep in make_episodes(sc, args.episodes, args.seed, "clean", 90)
            for pl in args.planners]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=("checkpoints/loo_zara01.pt",)) as pool:
        outs = list(pool.map(_job, [e for e, _ in jobs], [p for _, p in jobs], [opts] * len(jobs)))
    for pl in args.planners:
        evs = [e for p, _, es in outs if p == pl for e in es]
        print(f"\n=== {pl}: {len(evs)} close encounters (< {args.close} m) with pedestrians not tracked from the start")
        print("  " + ", ".join(f"{k} {v}" for k, v in Counter(e["kind"] for e in evs).most_common()))
        det = [e for e in evs if "why" in e]
        dd = np.array([e["det_dist"] for e in det])
        print(f"  detection distance: median {np.median(dd):.2f} m; < 2.5 m in {np.mean(dd < 2.5):.0%}; "
              f"lead time to encounter median {np.median([e['lead'] for e in det]):.0f} steps")
        late = [e for e in det if e["det_dist"] < 2.5]
        if late:
            print(f"  late (< 2.5 m, n={len(late)}): " + ", ".join(f"{k} {v}" for k, v in Counter(e['why'] for e in late).most_common())
                  + " | " + ", ".join(f"{k} {v}" for k, v in Counter(e['kind'] for e in late).most_common()))
            bb = np.abs([e["bear_body"] for e in late])
            print(f"    bearing from robot heading: <30 deg {np.mean(bb < 30):.0%}, 30-90 {np.mean((bb >= 30) & (bb < 90)):.0%}, "
                  f">=90 {np.mean(bb >= 90):.0%}; robot speed then median {np.median([e['speed'] for e in late]):.2f} m/s")
            m = np.array([e["mass"] for e in late if e["kind"] == "first detection"])
            if len(m):
                print(f"    never-seen ones: undetected density within 1 m of them (expected count): median {np.median(m):.3f}, "
                      f"> 0.05 in {np.mean(m > 0.05):.0%}")


if __name__ == "__main__":
    main()
