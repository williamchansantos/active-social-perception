"""Animate a SocNavBench episode run through the SupeR-MPC bridge.

Combines SocNavBench's saved simulator states (every pedestrian, ground truth) with the server's per-step log
(robot pose and camera pan, detections, PMB tracks, predictions, planned path) and draws, per 0.4 s step:
  - the Univ traversible map (gray = obstacle)
  - true pedestrians: green = detected this step, hollow gray = not detected
  - PMB tracks: 2-sigma position ellipses, faded by existence probability
  - predicted paths of tracked pedestrians with 2-sigma ellipses at 0.4 s, 1.6 s and 3.2 s
  - camera field of view, robot (SocNavBench radius), planned path, waypoint, goal and trail

    python -m socnav_bridge.make_video --log results/socnav/test_socnav_univ_active.json
    python -m socnav_bridge.make_video --log results/socnav/X/t_univ1_active_vision.json         --sim-dir .../tests/socnav/test_supermpc_X/t_univ1/sim_state_data --frames results/socnav/X/frames/t_univ1
With --frames (server --vision --save-frames), the robot's camera image with the detector's boxes and
depth ranges is drawn next to the map.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np

SOCNAV = os.environ.get("SOCNAVBENCH_DIR", r"C:/Users/James/Desktop/SocNavBench")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", default="results/socnav/test_socnav_univ_active.json")
    p.add_argument("--sim-dir", default=f"{SOCNAV}/tests/socnav/test_supermpc/test_socnav_univ/sim_state_data")
    p.add_argument("--map", default=None, help="default: the episode's map (curated episodes) or Univ")
    p.add_argument("--frames", default=None, help="saved camera frames (server --save-frames)")
    p.add_argument("--fov", type=float, default=None, help="default: from the log")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--out", default=None)
    p.add_argument("--still", type=int, default=None, help="save only this step as a figure (--out .pdf/.png)")
    p.add_argument("--compact", action="store_true", help="no title or legend box (for paper figures)")
    p.add_argument("--view", type=float, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1"),
                   help="crop the map (with --still)")
    return p.parse_args()


def ellipse(ax, mu, c, color, alpha, lw=1.0, ls="-"):
    from matplotlib.patches import Ellipse
    C = np.array([[c[0], c[1]], [c[1], c[2]]])
    w, V = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mu, 4 * np.sqrt(max(w[1], 1e-8)), 4 * np.sqrt(max(w[0], 1e-8)), angle=ang, fill=False,
                         ec=color, alpha=alpha, lw=lw, ls=ls))


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle, Wedge

    d = json.load(open(args.log))
    L = d["log"]
    fov = args.fov or d["fov"]
    rng = d["range"]
    if args.map is None:
        from socnav_bridge.episodes import EPISODES, spec
        args.map = spec(d["name"])["map"] if d["name"] in EPISODES else "Univ"
    m = pickle.load(open(f"{SOCNAV}/sd3dis/stanford_building_parser_dataset/traversibles/{args.map}/data.pkl", "rb"),
                    encoding="latin1")
    trav, res = np.asarray(m["traversible"]), m["resolution"] / 100.0
    files = sorted(glob.glob(os.path.join(args.sim_dir, "sim_state_*.json")),
                   key=lambda f: float(os.path.basename(f)[10:-5]))
    times = np.array([float(os.path.basename(f)[10:-5]) for f in files])

    # Pedestrian ids exactly as the server assigned them (order of first appearance in the state messages).
    ids, frames = {}, []
    for t in L["t"]:
        s = json.load(open(files[int(np.argmin(np.abs(times - t)))]))
        peds = {}
        for name, a in s["pedestrians"].items():
            ids.setdefault(name, len(ids) + 1)
            peds[ids[name]] = np.array(a["current_config"][:2])
        frames.append(peds)

    goal = np.array(d["goal"]) if d.get("goal") else np.array([15.5, 8.0])
    R = np.array(L["robot"])
    if args.frames:
        fig, (ax, axc) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.07, wspace=0.06)
    else:
        fig, ax = plt.subplots(figsize=(9.5, 7.2))
        axc = None
    H, W = trav.shape

    def draw(k):
        ax.clear()
        ax.imshow(~trav.astype(bool), cmap="Greys", vmin=0, vmax=2.2, origin="lower",
                  extent=[0, W * res, 0, H * res], interpolation="nearest")
        x, y, th, phi = R[k]
        ax.add_patch(Wedge((x, y), rng, np.degrees(th + phi) - fov / 2, np.degrees(th + phi) + fov / 2,
                           color="gold", alpha=0.25, lw=0))
        det = set(L["detected"][k])
        dxy = np.array(L.get("det_xy", [[]] * len(R))[k]).reshape(-1, 2)
        for pid, q in frames[k].items():
            # vision detections carry track ids, not pedestrian ids: match by position instead
            seen = pid in det if not d.get("vision") else bool(len(dxy)) and np.min(np.linalg.norm(dxy - q, axis=1)) < 0.7
            if seen:
                ax.add_patch(Circle(q, 0.2, color="tab:green", alpha=0.9))
            else:
                ax.add_patch(Circle(q, 0.2, fill=False, ec="gray", lw=1.0))
        for tr in L["tracks"][k]:
            ellipse(ax, tr[:2], tr[2:5], "tab:blue", 0.25 + 0.75 * tr[5], lw=1.2)
        for mean, cov, r in zip(L["pred"][k], L["pred_cov"][k], L["pred_r"][k]):
            mean = np.array(mean)
            ax.plot(mean[:, 0], mean[:, 1], "-", color="tab:purple", alpha=0.3 + 0.6 * r, lw=1)
            for j in (0, 3, 7):
                if j < len(mean):
                    ellipse(ax, mean[j], cov[j], "tab:purple", (0.15 + 0.5 * r) * (1 - j / 12), lw=0.8, ls="--")
        if d.get("vision") and len(dxy):
            ax.plot(dxy[:, 0], dxy[:, 1], "+", color="darkgreen", ms=9, mew=2)
        ax.plot(R[: k + 1, 0], R[: k + 1, 1], "-", color="tab:red", alpha=0.5, lw=1.5)
        plan = np.array(L["plan"][k])
        ax.plot(plan[:, 0], plan[:, 1], ".-", color="tab:red", lw=2, ms=4)
        ax.add_patch(Circle((x, y), 0.24, color="tab:red"))
        ax.plot([x, x + 0.45 * np.cos(th)], [y, y + 0.45 * np.sin(th)], "k-", lw=2)
        ax.plot(*L["waypoint"][k], "x", color="tab:orange", ms=9, mew=2)
        ax.plot(*goal, "*", color="gold", mec="k", ms=18)
        ax.set_xlim(0, W * res)
        ax.set_ylim(0, H * res)
        ax.set_aspect("equal")
        status = d["termination_cause"] if k == len(R) - 1 else ""
        if axc is not None:
            draw_camera(axc, k)
        mode = "RGB-D vision" if d.get("vision") else "simulated camera on ground truth"
        ax.set_title(f"SocNavBench {d['name']} ({args.map}): SupeR-MPC, {d['camera']} {fov:.0f}° camera, {mode}\n"
                     f"t = {L['t'][k]:4.1f} s   detected {len(det)}, tracks {len(L['tracks'][k])}   {status}",
                     fontsize=9.5)
        ax.text(0.01, 0.01, ("green: detected" + ("  (+: detection from RGB-D)" if d.get("vision") else "") +
                "   gray ring: not detected   blue: PMB track (2σ, faded by existence)\n"
                "purple: predicted paths, 2σ at 0.4/1.6/3.2 s   red: robot, plan, trail   x: waypoint   ★: goal"),
                transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(fc="white", alpha=0.85, lw=0))

    def draw_camera(axc, k):
        axc.clear()
        f = os.path.join(args.frames, f"{k:03d}.npz")
        axc.set_xticks([])
        axc.set_yticks([])
        if not os.path.exists(f):
            return
        c = np.load(f)
        axc.imshow(c["rgb"])
        for (x0, y0, x1, y1), sc, rg in zip(c["boxes"], c["scores"], c["ranges"]):
            axc.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5))
            lab = f"{sc:.2f}" + ("" if np.isnan(rg) else f"  {rg:.1f} m")
            axc.text(x0, y0 - 2, lab, color="lime", fontsize=7, va="bottom")
        axc.set_title(f"robot camera (pan {np.degrees(R[k, 3]):+.0f}°): person detections, range from depth",
                      fontsize=9)

    out = args.out or os.path.splitext(args.log)[0] + ".gif"
    if args.still is not None:
        draw(args.still)
        if args.compact:
            for t in ax.texts[-1:]:
                t.remove()
            ax.set_title(f"t = {L['t'][args.still]:.1f} s", fontsize=9)
        if args.view:
            ax.set_xlim(*args.view[:2])
            ax.set_ylim(*args.view[2:])
            if axc is not None:
                axc.set_title("robot camera: detections, range from depth", fontsize=9)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print("saved", out)
        return
    anim = FuncAnimation(fig, draw, frames=len(R))
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main(parse_args())
