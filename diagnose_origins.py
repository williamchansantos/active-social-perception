"""Where do the pedestrians a robot collides with come from?

Reruns clean-protocol episodes (identical seeds) for several methods and, for every
first contact with a pedestrian (distance < robot + pedestrian radius), records:
  - sector 2 s before contact, relative to the robot's heading: front (<30 deg),
    oblique (30-60), side (60-120), behind (>120), and left/right; and distance
  - relative motion at contact: head-on, crossing, same direction, standing
  - whether the camera covered the pedestrian 2 s before contact (FOV, range, occlusion)
  - detection: never detected / first detected how many steps before contact and how far
  - spawned: appeared in the scene within 3 m of the robot

    python diagnose_origins.py --episodes 150 --seed 11
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

METHODS = {  # label: (planner, MPC overrides) -- as in results/tune600 and results/confirm23
    "proposed": ("supermpc-active", dict(unseen_joint=False, unseen_obs_aware=True)),
    "infogain": ("supermpc-infogain", dict(unseen_joint=False)),
    "nearest": ("supermpc-nearest", dict(unseen_joint=False)),
    "fixed": ("supermpc", dict(unseen_joint=False)),
    "360": ("supermpc-360", dict(unseen_joint=False)),
}
LEAD = 5  # steps (2 s) before contact


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=150, help="per Zara scene")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--workers", type=int, default=13)
    p.add_argument("--scenes", nargs="+", default=["zara01", "zara02"])
    return p.parse_args()


def wrap_deg(a):
    return (np.degrees(a) + 180) % 360 - 180


def _job(job):
    sc, ep_i, seed, methods = job
    from evaluate_nav import SCENES, _init_worker, _predictor, make_episodes, scene_frames
    from robot_nav.camera import CameraConfig, visible
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene
    from run_nav import PLANNERS, run_episode

    _init_worker(f"checkpoints/loo_{sc}.pt")
    first, _, _, split = scene_frames(sc)
    prior = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
    ep = make_episodes(sc, 300, seed, "clean", 90)[ep_i]
    events = []
    for name in methods:
        planner, ov = METHODS[name]
        cam = CameraConfig(fov_deg=100.0, max_range=4.0)
        for key, value in PLANNERS[planner].get("camera", {}).items():
            setattr(cam, key, value)
        recs = []
        nav = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=100, cam_range=4,
                              mpc_overrides=ov, pmb_overrides=dict(prune_pos_std=1.0))
        run_episode(planner, PedestrianScene(SCENES[sc]["path"], ep["frame"]), ep["start"], ep["goal"],
                    _predictor(f"checkpoints/loo_{sc}.pt"), nav, seed + 1000 * list(SCENES).index(sc) + ep["episode"],
                    prior, hook=lambda step, x, peds, tracks, **_: recs.append(
                        (x.copy(), {k: v.copy() for k, v in peds.items()}, set(tracks))))
        r_col = 0.6
        in_contact = set()
        for i, (x, peds, tracked) in enumerate(recs):
            contact = {pid for pid, q in peds.items() if np.linalg.norm(q - x[:2]) < r_col}
            for pid in sorted(contact - in_contact):
                ev = dict(method=name, scene=sc, episode=ep["episode"])
                # first appearance and first tracking
                appear = next(j for j in range(i + 1) if pid in recs[j][1])
                xa, pa, _ = recs[appear]
                ev["spawned"] = appear > 0 and np.linalg.norm(pa[pid] - xa[:2]) < 3.0
                det = next((j for j in range(i + 1) if pid in recs[j][2]), None)
                if det is None:
                    ev["detect"] = "never"
                else:
                    ev["lead"] = i - det
                    ev["det_dist"] = float(np.linalg.norm(recs[det][1][pid] - recs[det][0][:2]))
                    ev["detect"] = "late (<=2 steps)" if i - det <= 2 else ("3-5 steps" if i - det <= 5 else ">5 steps")
                # position LEAD steps before contact (or first appearance, if later)
                j = max(i - LEAD, appear)
                xj, pj, _ = recs[j]
                d = pj[pid] - xj[:2]
                ev["dist_2s"] = float(np.linalg.norm(d))
                b = wrap_deg(np.arctan2(d[1], d[0]) - xj[2])
                ab = abs(b)
                ev["sector"] = "front" if ab < 30 else "oblique" if ab < 60 else "side" if ab < 120 else "behind"
                ev["side"] = "left" if b > 0 else "right"
                ev["visible_2s"] = bool(visible(xj[:2], xj[2] + xj[3], pj, cam)[pid]) if j < i else False
                ev["at_appear"] = j == appear and appear > i - LEAD
                # relative motion at contact (pedestrian velocity in the robot's body frame)
                if i > 0 and pid in recs[i - 1][1]:
                    v = (peds[pid] - recs[i - 1][1][pid]) / 0.4
                    c, s = np.cos(-x[2]), np.sin(-x[2])
                    vb = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
                    speed = np.linalg.norm(vb)
                    if speed < 0.2:
                        ev["motion"] = "standing"
                    else:
                        ang = abs(wrap_deg(np.arctan2(vb[1], vb[0])))
                        ev["motion"] = "same direction" if ang < 45 else "head-on" if ang > 135 else "crossing"
                else:
                    ev["motion"] = "unknown"
                events.append(ev)
            in_contact = contact
    return events


def main():
    args = parse_args()
    jobs = [(sc, i, args.seed, args.methods) for sc in args.scenes for i in range(args.episodes)]
    with ProcessPoolExecutor(args.workers) as pool:
        events = [e for es in pool.map(_job, jobs) for e in es]
    n_ep = len(jobs)

    def table(title, key, cats):
        print(f"\n{title}")
        print(f"{'method':10s} {'contacts':>9s} " + " ".join(f"{c:>15s}" for c in cats))
        for m in args.methods:
            ev = [e for e in events if e["method"] == m]
            cnt = Counter(e[key] for e in ev)
            print(f"{m:10s} {len(ev):9d} " + " ".join(f"{cnt[c]:6d} ({cnt[c] / max(len(ev), 1):4.0%})" for c in cats))

    print(f"{n_ep} episodes ({' + '.join(args.scenes)}, seed {args.seed}); first contacts per method")
    table("Where they were 2 s before contact (relative to the robot's heading)", "sector",
          ["front", "oblique", "side", "behind"])
    table("Left or right of the robot 2 s before contact", "side", ["left", "right"])
    table("Relative motion at contact (pedestrian velocity in the robot frame)", "motion",
          ["head-on", "crossing", "same direction", "standing"])
    table("Detection", "detect", ["never", "late (<=2 steps)", "3-5 steps", ">5 steps"])
    print("\nCamera coverage 2 s before contact, spawning, and distance")
    for m in args.methods:
        ev = [e for e in events if e["method"] == m]
        if not ev:
            continue
        dd = [e["det_dist"] for e in ev if "det_dist" in e]
        print(f"  {m:10s} in camera view 2 s before: {np.mean([e['visible_2s'] for e in ev]):4.0%} | spawned within 3 m: "
              f"{np.mean([e['spawned'] for e in ev]):4.0%} | distance 2 s before: median {np.median([e['dist_2s'] for e in ev]):.1f} m"
              f" | first-detection distance: median {np.median(dd) if dd else float('nan'):.1f} m")
    print("\nSector x detection (contacts): never detected / late (<=2 steps) / earlier")
    for m in args.methods:
        ev = [e for e in events if e["method"] == m]
        parts = []
        for s in ("front", "oblique", "side", "behind"):
            es = [e for e in ev if e["sector"] == s]
            parts.append(f"{s} {sum(e['detect'] == 'never' for e in es)}/{sum(e['detect'] == 'late (<=2 steps)' for e in es)}/"
                         f"{sum(e['detect'] in ('3-5 steps', '>5 steps') for e in es)}")
        print(f"  {m:10s} " + "  |  ".join(parts))


if __name__ == "__main__":
    main()
