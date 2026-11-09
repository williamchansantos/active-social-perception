"""Births relative to a synthetic robot's field of view (roadmap step: FOV births).

For every scene:
    1. Place a stationary robot (see `fov.place_robot`; override with --robot).
    2. Replay the annotation and record each pedestrian's first frame inside the
       cone, labelled "scene_entry" (entered the scene inside the cone) or
       "walked_in" (was in the scene first, then moved into the cone).
    3. Print a comparison with the scene-level births of step 1.
    4. Fit and score uniform / grid / KDE birth models on the FOV births with the
       same 80/20 time split as `evaluate_birth` (bounds = the visible area).

Figures under --plot-dir:
    <scene>_fov_births.png       cone, births by origin, cumulative arrivals vs scene-level
    <scene>_fov_birth_model.png  the usual four-panel model comparison on FOV births

Example:
    python -m social_lstm.birth.analyze_fov
    python -m social_lstm.birth.analyze_fov --placement edge --fov-deg 90 --range 6
    python -m social_lstm.birth.analyze_fov --data data/ucy/zara01 --robot 0 12 -90
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from social_lstm.birth.appearance import DEFAULT_DT, DEFAULT_SCENES, SceneAppearances, extract_appearances
from social_lstm.birth.evaluate_birth import BLUE, ORANGE, SURFACE, TEXT, TRAJ_GRAY, _style, evaluate_scene, format_table, plot_scene
from social_lstm.birth.fov import RobotView, extract_fov_appearances, origin_counts, place_robot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=list(DEFAULT_SCENES), help="scene directories or obsmat.txt files")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--placement", choices=["center", "edge"], default="center", help="automatic robot pose")
    p.add_argument("--robot", type=float, nargs=3, metavar=("X", "Y", "HEADING_DEG"),
                   help="explicit robot pose (meters, degrees); applies to every scene given")
    p.add_argument("--fov-deg", type=float, default=60.0, help="horizontal field of view [deg]")
    p.add_argument("--range", type=float, default=8.0, help="maximum sensing range [m]")
    # model settings, same meaning as in evaluate_birth
    p.add_argument("--cell", type=float, default=0.5)
    p.add_argument("--bandwidth", type=float, default=0.5)
    p.add_argument("--region", type=float, default=2.0)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--uniform-mix", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-dir", default="plots/birth")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def plot_fov_births(scene: SceneAppearances, fov: SceneAppearances, view: RobotView, path: str, dpi: int) -> None:
    """Left: scene with the cone and FOV births coloured by origin. Right: cumulative
    arrivals over time, scene-level vs. field-of-view."""
    allp = fov.extras["scene_positions"]
    ev = fov.arrivals()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.3, 1]})

    _style(ax1, equal=True)
    ax1.scatter(allp[:, 0], allp[:, 1], s=2, color=TRAJ_GRAY, alpha=0.25, linewidths=0, label="all annotated positions")
    poly = view.outline()
    ax1.fill(poly[:, 0], poly[:, 1], color=BLUE, alpha=0.08, zorder=1)
    ax1.plot(poly[:, 0], poly[:, 1], color=BLUE, linewidth=1.2, zorder=2, label="field of view")
    ax1.plot(*view.position, marker="s", color=TEXT, markersize=7, zorder=5, label="robot")
    for origin, color, label in (("scene_entry", BLUE, "entered scene inside cone"),
                                 ("walked_in", ORANGE, "already in scene, walked into cone")):
        pts = np.array([e.position for e in ev if e.origin == origin]).reshape(-1, 2)
        vel = np.array([e.velocity for e in ev if e.origin == origin]).reshape(-1, 2)
        ok = ~np.isnan(vel).any(axis=1)
        ax1.quiver(pts[ok, 0], pts[ok, 1], vel[ok, 0], vel[ok, 1], angles="xy", scale_units="xy", scale=1.0,
                   color=color, alpha=0.5, width=0.003, headwidth=4)
        ax1.scatter(pts[:, 0], pts[:, 1], s=20, color=color, edgecolors="white", linewidths=0.6, zorder=4,
                    label=f"{label} ({len(pts)})")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_title(view.describe(), fontsize=9.5, color=TEXT)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7.5, frameon=False, labelcolor=TEXT)

    _style(ax2)
    for s, color, label in ((scene, TRAJ_GRAY, "scene-level births (step 1)"), (fov, BLUE, "field-of-view births")):
        t = s.times()
        ax2.step(t, np.arange(1, len(t) + 1), where="post", color=color, linewidth=2,
                 label=f"{label}: {len(t)}, {60 * s.arrival_rate():.1f}/min")
    ax2.set_xlabel("time since first annotated frame [s]")
    ax2.set_ylabel("cumulative arrivals")
    ax2.set_title("Arrivals over time", fontsize=10, color=TEXT)
    ax2.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT)

    fig.suptitle(f"{scene.name}: births as seen by a stationary robot", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    summary: List[dict] = []
    model_rows: List[dict] = []

    for path in args.data:
        scene = extract_appearances(path, dt=args.dt)
        if args.robot is not None:
            x, y, hdg = args.robot
            view = RobotView(np.array([x, y]), math.radians(hdg), args.fov_deg, args.range)
        else:
            view = place_robot(scene.all_positions, args.placement, args.fov_deg, args.range)
        fov = extract_fov_appearances(path, view, dt=args.dt)
        oc = origin_counts(fov)
        n_fov = len(fov.arrivals())
        summary.append({
            "scene": scene.name,
            "robot": f"({view.position[0]:.1f}, {view.position[1]:.1f}) {math.degrees(view.heading):.0f}deg",
            "scene_births": len(scene.arrivals()),
            "fov_births": n_fov,
            "scene_entry": oc["scene_entry"],
            "walked_in": oc["walked_in"],
            "frac_walked_in": oc["walked_in"] / max(n_fov, 1),
            "in_cone_at_start": sum(e.present_at_start for e in fov.events),
            "reentries": int(fov.extras["reentries"]),
            "scene_rate_per_min": 60 * scene.arrival_rate(),
            "fov_rate_per_min": 60 * fov.arrival_rate(),
            "cone_area_m2": view.area,
        })
        rows, extras = evaluate_scene(fov, args)
        model_rows.extend(rows)
        if not args.no_plots:
            plot_fov_births(scene, fov, view, os.path.join(args.plot_dir, f"{scene.name}_fov_births.png"), args.dpi)
            plot_scene(fov, rows, extras, args)

    print(f"placement={'explicit' if args.robot else args.placement}, FOV {args.fov_deg:g} deg, range {args.range:g} m\n")
    head = f"{'scene':<11s} {'robot pose':<22s} {'scene':>6s} {'fov':>5s} {'entry':>6s} {'walked':>7s} {'frac_w':>6s} {'at_t0':>5s} {'reent':>5s} {'scene/min':>9s} {'fov/min':>7s}"
    print(head)
    print("-" * len(head))
    for r in summary:
        print(f"{r['scene']:<11s} {r['robot']:<22s} {r['scene_births']:6d} {r['fov_births']:5d} {r['scene_entry']:6d} "
              f"{r['walked_in']:7d} {r['frac_walked_in']:6.2f} {r['in_cone_at_start']:5d} {r['reentries']:5d} "
              f"{r['scene_rate_per_min']:9.1f} {r['fov_rate_per_min']:7.1f}")
    print("\nscene = step-1 births (first annotated anywhere); fov = first frame inside the cone;"
          "\nentry = entered the scene inside the cone; walked = was in the scene, then walked into the cone;"
          "\nat_t0 = inside the cone in the first frame (left-censored, not counted); reent = extra cone entries.\n")
    print("Birth models fitted on the field-of-view births (bounds = visible area):\n")
    print(format_table(model_rows))

    with open(os.path.join(args.plot_dir, "fov_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    if not args.no_plots:
        print(f"\nSaved {2 * len(summary)} figures and fov_summary.csv to {args.plot_dir}/")


if __name__ == "__main__":
    main()
