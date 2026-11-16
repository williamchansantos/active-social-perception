"""Paper figure: the Poisson (never-detected pedestrian) part of the PMB tracker in one real episode.

(a) the scene prior the robot starts with, (b) the intensity just after the camera has looked, (c) a few steps
later, as the cleared cone refills and drifts with the pedestrian flows, (d) the forecast the planner uses: the
intensity predicted along the planned path without new measurements, with P(>=1 never-detected pedestrian) in
each 0.6 m collision disk. Ground truth is overlaid: tracked pedestrians and pedestrians with no track.

    LOO_PREFIX=loorot python make_ppp_figure.py --scene zara01 --episode auto --out plots/fig_ppp.pdf
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge

from evaluate_nav import SCENES, checkpoint_for, make_episodes, scene_frames

BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = "#eb6834"  # categorical slot 2: tracked pedestrians
INK, MUTED = "#1a1a19", "#6b6a66"
R_DISK, H = 0.6, 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="zara01")
    p.add_argument("--episode", default="auto", help="episode index, or auto (most pedestrians near the path)")
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--t1", type=int, default=None, help="step of panel (b); default: chosen automatically")
    p.add_argument("--gap", type=int, default=5, help="steps between panels (b) and (c)")
    p.add_argument("--out", default="plots/fig_ppp.pdf")
    return p.parse_args()


def record(sc, ep, seed):
    from robot_nav.mpc import MPCConfig
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.prediction import TrajectoryPredictor
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    first, _, _, split = scene_frames(sc)
    prior = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
    predictor = TrajectoryPredictor(checkpoint_for(sc, "clean", None), MPCConfig().N)
    scene = PedestrianScene(SCENES[sc]["path"], ep["frame"])
    steps = []

    def hook(step, x, tracker, info, peds, **_):
        steps.append(dict(step=step, x=x.copy(), D=tracker.D.copy(), lam=tracker.rate_mean, a=tracker.rate_a,
                          b=tracker.rate_b, tracked={p: tr.x[:2].copy() for p, tr in tracker.tracks.items()},
                          peds={p: q.copy() for p, q in peds.items()}, X=np.asarray(info["X"]).copy()))

    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=60.0, cam_range=4.0,
                           mpc_overrides={"unseen_joint": False}, pmb_overrides={"prune_pos_std": 1.0},
                           robot_overrides=None)
    run_episode("supermpc-active", scene, ep["start"], ep["goal"], predictor, args,
                seed + 1000 * list(SCENES).index(sc) + ep["episode"], prior, hook=hook)
    return prior, steps


def main(args):
    from robot_nav.pmb import channel_mixing, ppp_predict

    eps = make_episodes(args.scene, 450, args.seed, protocol="clean")[:60]
    ep = max(eps, key=lambda e: e["path_peds"]) if args.episode == "auto" else eps[int(args.episode)]
    prior, steps = record(args.scene, ep, args.seed)
    g = prior.grid
    cell = g.res ** 2
    extent = [g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res]

    def untracked(s):
        return [q for p, q in s["peds"].items() if p not in s["tracked"]]

    if args.t1 is None:  # a step with people both tracked and untracked near the robot, and room for (c)
        cand = range(4, len(steps) - args.gap - 1)
        args.t1 = max(cand, key=lambda i: min(sum(np.linalg.norm(q - steps[i]["x"][:2]) < 6 for q in untracked(steps[i])),
                                              sum(np.linalg.norm(q - steps[i]["x"][:2]) < 6 for q in steps[i]["tracked"].values())))
    t1, t2 = args.t1, args.t1 + args.gap
    s0, s1, s2 = steps[0], steps[t1], steps[t2]
    # forecast from step t2 along its plan
    D = s2["D"].copy()
    mix = channel_mixing(0.02)
    centers = g.centers
    disks = []
    for k in range(1, H + 1):
        D = ppp_predict(D, prior, mix)
        c = s2["X"][:2, k]
        M = float(D.sum(0)[np.linalg.norm(centers - c, axis=-1) <= R_DISK].sum())
        disks.append((c, 1 - (s2["b"] / (s2["b"] + M)) ** s2["a"]))
    Dk = D

    dens = [s0["lam"] * prior.occupancy.sum(0) / cell, s1["lam"] * s1["D"].sum(0) / cell,
            s2["lam"] * s2["D"].sum(0) / cell, s2["lam"] * Dk.sum(0) / cell]
    vmax = np.percentile(dens[0][dens[0] > 0], 99)
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE)

    # crop to the region the robot traverses
    xs = np.concatenate([[s["x"][0] for s in steps], [ep["goal"][0]]])
    ys = np.concatenate([[s["x"][1] for s in steps], [ep["goal"][1]]])
    pad = 4.5
    xlim = (max(extent[0], xs.min() - pad), min(extent[1], xs.max() + pad))
    ylim = (max(extent[2], ys.min() - pad), min(extent[3], ys.max() + pad))

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), sharex=True, sharey=True, constrained_layout=True)
    titles = ["(a) Scene prior, $t=0$", f"(b) After looking, $t={t1 * 0.4:.1f}$ s",
              f"(c) Refilling, $t={t2 * 0.4:.1f}$ s", f"(d) Forecast, +{H * 0.4:.1f} s"]
    snaps = [s0, s1, s2, s2]
    for i, (ax, d, s) in enumerate(zip(axes, dens, snaps)):
        im = ax.imshow(d.T, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=vmax, interpolation="bilinear")
        x, y, th, phi = s["x"]
        if i in (1, 2):
            ax.add_patch(Wedge((x, y), 4.0, np.degrees(th + phi) - 30, np.degrees(th + phi) + 30, fill=False,
                               ec=INK, lw=0.8, ls=(0, (3, 2))))
        if i == 3:  # truth H steps later for the pedestrians untracked now: what the forecast predicts
            fut = steps[min(t2 + H, len(steps) - 1)]["peds"]
            for pid, q in fut.items():
                if pid not in s["tracked"]:
                    ax.add_patch(Circle(q, 0.22, fill=False, ec=INK, lw=0.8))
        if 0 < i < 3:
            for q in untracked(s):
                ax.add_patch(Circle(q, 0.22, fill=False, ec=INK, lw=0.8))
            for q in s["tracked"].values():
                ax.add_patch(Circle(q, 0.22, color=ORANGE, lw=0))
        if i > 0:
            ax.plot([x, x + 0.7 * np.cos(th)], [y, y + 0.7 * np.sin(th)], color=INK, lw=1.2)
            ax.add_patch(Circle((x, y), 0.3, color=INK, lw=0))
        if i == 3:
            X = s["X"]
            ax.plot(X[0], X[1], color=INK, lw=1.0)
            pmax = max(p for _, p in disks)
            for c, p in disks:
                ax.add_patch(Circle(c, R_DISK, fc=ORANGE, alpha=0.12 + 0.5 * p / max(pmax, 1e-9), ec="none"))
                ax.add_patch(Circle(c, R_DISK, fill=False, ec=ORANGE, lw=0.6))
            ax.text(0.97, 0.96, f"P(never-detected pedestrian\nin disk), 0.4-3.2 s:\n{disks[0][1]:.3f} to {pmax:.3f}",
                    transform=ax.transAxes, fontsize=6, color=INK, ha="right", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.plot(*ep["goal"][:2], marker="*", color=INK, ms=7, mew=0)
        ax.set_title(titles[i], fontsize=8, color=INK)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6, length=2)
    axes[0].set_ylabel("y (m)", fontsize=7)
    for ax in axes:
        ax.set_xlabel("x (m)", fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, fraction=0.03)
    cb.set_label("never-detected pedestrians / m$^2$", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=ORANGE, ms=5, label="tracked pedestrian"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=5, label="pedestrian without a track (truth; in (d): {:.1f} s later)".format(H * 0.4)),
               Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 2)), label="camera field of view"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=ORANGE, ms=7, label="collision disk on the plan")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.45, -0.1))
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", bbox_inches="tight", dpi=200)
    print(f"saved {args.out}; episode {ep['episode']} ({ep['path_peds']} pedestrians near the path), t1={t1}, t2={t2}, "
          f"{len(steps)} steps")


if __name__ == "__main__":
    main(parse_args())
