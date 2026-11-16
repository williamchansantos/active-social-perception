"""Animate evaluation episodes: what the robot sees, tracks, predicts and expects.

Reruns an episode from a clean-protocol evaluation (identical seeds, so it matches
the CSV row) and draws, per step:
  - background: the PMB's expected density of never-detected pedestrians
  - true pedestrians: green = detected this step, blue = tracked (not in view),
    hollow gray = unknown to the robot
  - tracked estimates: predicted mean path and 2-sigma ellipses (LSTM + state
    covariance) at 0.4 s, 1.6 s and 3.2 s, faded by existence probability
  - camera field of view, robot, planned path with its 2-sigma uncertainty, trail
Several planners are drawn side by side on the same episode.

    python make_video.py --scene zara02 --episode 192 --seed 23
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np

from evaluate_nav import SCENES, _WORKER, _init_worker, _predictor, checkpoint_for, make_episodes, scene_frames

CONFIGS = {  # label: (planner, MPC overrides, PMB overrides) -- as in results/confirm23
    "proposed": ("supermpc-active", dict(unseen_joint=False, unseen_obs_aware=True), dict(prune_pos_std=1.0)),
    "fixed": ("supermpc", dict(unseen_joint=False), dict(prune_pos_std=1.0)),
    "infogain": ("supermpc-infogain", dict(unseen_joint=False), dict(prune_pos_std=1.0)),
    "360": ("supermpc-360", dict(unseen_joint=False), dict(prune_pos_std=1.0)),
}
TITLES = {"proposed": "Proposed: risk-aware gaze + observation-aware constraint", "fixed": "Fixed camera",
          "infogain": "Information-gain gaze", "360": "360° camera (oracle)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="zara02")
    p.add_argument("--episode", type=int, default=192)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--n-episodes", type=int, default=300, help="episodes in the evaluation (for the episode list)")
    p.add_argument("--configs", nargs="+", default=["proposed"], choices=list(CONFIGS))
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--range", type=float, default=4.0)
    p.add_argument("--delta-unseen", type=float, default=0.02)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def run(ep, label, args):
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    planner, mpc_ov, pmb_ov = CONFIGS[label]
    sc = ep["scene"]
    first, _, _, split = scene_frames(sc)
    prior = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
    nav = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=args.delta_unseen, fov=args.fov,
                          cam_range=args.range, mpc_overrides=mpc_ov, pmb_overrides=pmb_ov)
    frames = []

    def hook(step, x, peds, tracks, pred, info, tracker, **_):
        frames.append(dict(x=x.copy(), peds={k: v.copy() for k, v in peds.items()},
                           tracks={k: (t.x.copy(), getattr(t, "r", 1.0)) for k, t in tracks.items()},
                           ids=list(pred.ids), mean=pred.mean.copy(), cov=pred.cov.copy(), r=pred.existence.copy(),
                           plan=info["X"].copy(), robot_cov=np.array(info["robot_cov"]), sel=set(info["selected"]),
                           density=tracker.undetected_density().copy(), grid=tracker.grid))

    seed = args.seed + 1000 * list(SCENES).index(sc) + ep["episode"]
    res, log = run_episode(planner, PedestrianScene(SCENES[sc]["path"], ep["frame"]), ep["start"], ep["goal"],
                           _predictor(checkpoint_for(sc, "clean", None)), nav, seed, prior, hook=hook)
    for f, seen in zip(frames, log["seen"]):
        f["seen"] = set(seen)
    from robot_nav.camera import CameraConfig

    cam = CameraConfig(fov_deg=args.fov, max_range=args.range)
    from run_nav import PLANNERS

    for key, value in PLANNERS[planner].get("camera", {}).items():
        setattr(cam, key, value)
    return res, frames, cam


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch, Wedge

    from run_nav import cov_ellipse

    args = parse_args()
    _init_worker(checkpoint_for(args.scene, "clean", None))
    ep = next(e for e in make_episodes(args.scene, args.n_episodes, args.seed, "clean", 90)
              if e["episode"] == args.episode)
    runs = {lab: run(ep, lab, args) for lab in args.configs}
    for lab, (res, frames, _) in runs.items():
        print(f"{lab}: success={res['success']} reached={res['reached']} collisions={res['collisions']} "
              f"steps={res['steps']}")
    start, goal = ep["start"], ep["goal"]
    T = max(len(f) for _, f, _ in runs.values())
    a = SCENES[args.scene]["axis"]
    lo, hi = min(start[a], goal[a]) - 1.5, max(start[a], goal[a]) + 1.5
    c = start[1 - a]
    xlim, ylim = ((lo, hi), (c - 5.5, c + 5.5)) if a == 0 else ((c - 5.5, c + 5.5), (lo, hi))
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 8.5 * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]) + 1.9),
                             squeeze=False)
    axes = axes[0]
    vmax = max(np.percentile(np.concatenate([fr["density"].ravel() for _, fs, _ in runs.values() for fr in fs]), 99.5),
               1e-6)

    def draw(i):
        for ax, (lab, (res, frames, cam)) in zip(axes, runs.items()):
            ax.clear()
            f = frames[min(i, len(frames) - 1)]
            done = i >= len(frames)
            g = f["grid"]
            ax.imshow(f["density"].T, origin="lower", cmap="Purples", vmin=0, vmax=vmax, alpha=0.75,
                      extent=(g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res), interpolation="bilinear")
            x = f["x"]
            head = np.degrees(x[2] + x[3])
            fov = min(cam.fov_deg, 359.9)
            ax.add_patch(Wedge(x[:2], cam.max_range, head - fov / 2, head + fov / 2, color="gold", alpha=0.25, lw=0))
            trail = np.array([fr["x"][:2] for fr in frames[: min(i, len(frames) - 1) + 1]])
            ax.plot(trail[:, 0], trail[:, 1], "-", color="tab:red", lw=1, alpha=0.5)
            # tracked estimates and predictions
            for j, pid in enumerate(f["ids"]):
                alpha = float(np.clip(f["r"][j], 0.15, 1.0))
                est = f["tracks"].get(pid)
                if pid in f["sel"]:  # pedestrians the planner constrains: prediction + uncertainty
                    ax.plot(f["mean"][:, j, 0], f["mean"][:, j, 1], "-", color="tab:blue", lw=0.8, alpha=0.6 * alpha)
                    ax.add_patch(cov_ellipse(f["mean"][-1, j], f["cov"][-1, j], color="tab:blue", alpha=0.07 * alpha,
                                             lw=0))
                    ax.add_patch(cov_ellipse(f["mean"][-1, j], f["cov"][-1, j], fill=False, color="tab:blue",
                                             alpha=0.35 * alpha, lw=0.6, ls="--"))
                    ax.add_patch(cov_ellipse(f["mean"][0, j], f["cov"][0, j], fill=False, color="tab:blue",
                                             alpha=0.9 * alpha, lw=1.2))
                if est is not None:
                    ax.plot(*est[0][:2], "+", color="tab:blue", ms=6, alpha=alpha)
            # true pedestrians
            for pid, q in f["peds"].items():
                if pid in f["seen"]:
                    ax.add_patch(Circle(q, 0.3, color="tab:green", alpha=0.9, lw=0, zorder=4))
                elif pid in f["tracks"]:
                    ax.add_patch(Circle(q, 0.3, color="tab:blue", alpha=0.6, lw=0, zorder=4))
                else:
                    ax.add_patch(Circle(q, 0.3, fill=False, color="0.25", lw=1.3, ls="--", zorder=4))
            # robot, plan and its uncertainty
            plan = f["plan"]
            ax.plot(plan[0], plan[1], ".-", color="tab:red", lw=1.5, ms=3, zorder=5)
            for k in (len(f["robot_cov"]) // 2, len(f["robot_cov"]) - 1):
                rc = f["robot_cov"][k]
                S = np.array([[rc[0], rc[1]], [rc[1], rc[2]]])
                ax.add_patch(cov_ellipse(plan[:2, k + 1], S, fill=False, color="tab:red", lw=0.8, alpha=0.8))
            hit = any(np.linalg.norm(q - x[:2]) < 0.6 for q in f["peds"].values())
            ax.add_patch(Circle(x[:2], 0.3, color="black" if hit else "tab:red", zorder=6))
            ax.plot(*start, "ks", ms=8)
            ax.plot(*goal, "k*", ms=16)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            status = ""
            if done:
                status = "  —  GOAL" if res["reached"] else "  —  timed out"
            outcome = "success" if res["success"] else ("collision" if res["collisions"] else "timeout")
            ax.set_title(f"{TITLES[lab]}\nt = {min(i, len(frames) - 1) * 0.4:4.1f} s{status}   "
                         f"(episode outcome: {outcome})", fontsize=10)
    handles = [Patch(color="tab:green", alpha=0.85, label="pedestrian, detected now"),
               Patch(color="tab:blue", alpha=0.55, label="pedestrian, tracked (not detected now)"),
               Line2D([], [], color="0.35", ls="--", label="pedestrian, unknown to robot"),
               Line2D([], [], color="tab:blue", lw=1.2, label="track estimate (+), 2σ at 0.4 s"),
               Patch(facecolor=(0.12, 0.47, 0.71, 0.15), edgecolor="tab:blue", ls="--",
                     label="predicted path, 2σ at 3.2 s"),
               Patch(color="gold", alpha=0.4, label="camera field of view"),
               Line2D([], [], color="tab:red", marker=".", label="robot plan (2σ)"),
               Patch(color="indigo", alpha=0.35, label="expected never-seen pedestrians (PMB)")]
    fig.legend(handles=handles, loc="lower center", ncol=3 if n == 1 else 4, fontsize=8.5, frameon=False,
               columnspacing=1.5, handlelength=1.6)
    fig.suptitle(f"{args.scene.capitalize()}, held-out episode {args.episode}: replayed crowd, "
                 f"{int(args.fov)}° camera, {args.range:.0f} m range  (2× real time)", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.15, wspace=0.03)
    out = args.out or f"plots/video_{args.scene}_ep{args.episode}_{'_'.join(args.configs)}.gif"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=T + 6, interval=1000 / args.fps)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
