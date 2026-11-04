"""Step 1 of the Poisson birth baseline: empirical appearance analysis.

For every scene, find where and when each pedestrian first appears, print a
per-scene summary table (pedestrian count, duration, arrival rate, ...) and save
two figures per scene under --plot-dir:

    <scene>_appearance_map.png   appearance positions (+ initial velocity arrows)
                                 and a 2D histogram of them, both drawn over all
                                 annotated trajectory points
    <scene>_arrivals_time.png    cumulative arrivals vs. time against the
                                 constant-rate line, and the inter-arrival-time
                                 histogram against the exponential density

Example:
    python -m social_lstm.birth.analyze_appearances
    python -m social_lstm.birth.analyze_appearances --data data/ucy/zara01 --cell 0.5
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from social_lstm.birth.appearance import (
    DEFAULT_DT,
    DEFAULT_SCENES,
    SceneAppearances,
    format_summary_table,
    load_scenes,
    summary_row,
)

# Plot styling: one accent hue for appearances, a second for destinations,
# neutral grays for context and text.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TRAJ_GRAY = "#b8b7b2"
TEXT = "#52514e"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues_seq", ["#dbe8f8", "#2a78d6", "#0d3a73"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=list(DEFAULT_SCENES), help="scene directories or obsmat.txt files")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="seconds per annotated frame step")
    p.add_argument("--cell", type=float, default=1.0, help="histogram cell size in meters")
    p.add_argument(
        "--include-initial",
        action="store_true",
        help="count pedestrians present in the first frame as arrivals (default: report them separately)",
    )
    p.add_argument("--edge-margin", type=float, default=1.0, help="meters from the data bounding box counted as 'edge'")
    p.add_argument("--plot-dir", default="plots/birth")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--no-plots", action="store_true", help="only print the summary table")
    return p.parse_args()


def _style_axes(ax: plt.Axes, equal: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TRAJ_GRAY)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=TRAJ_GRAY, alpha=0.35, linewidth=0.6)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def plot_appearance_map(scene: SceneAppearances, cell: float, include_initial: bool, path: str, dpi: int) -> None:
    """Left: every appearance position with its initial-velocity arrow. Right: 2D
    histogram of appearance positions with `cell`-meter bins. Both are drawn over
    all annotated positions so the walkable area is visible."""
    pos = scene.positions(include_initial)
    vel = scene.velocities(include_initial)
    initial = np.array([e.position for e in scene.events if e.present_at_start]).reshape(-1, 2)
    allp = scene.all_positions

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style_axes(ax, equal=True)
        ax.scatter(allp[:, 0], allp[:, 1], s=2, color=TRAJ_GRAY, alpha=0.25, linewidths=0, label="all annotated positions")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    # Arrow = displacement over the next 1 s (length in meters equals speed in m/s).
    ok = ~np.isnan(vel).any(axis=1)
    ax1.quiver(
        pos[ok, 0], pos[ok, 1], vel[ok, 0], vel[ok, 1],
        angles="xy", scale_units="xy", scale=1.0, color=BLUE, alpha=0.6, width=0.003, headwidth=4,
    )
    ax1.scatter(pos[:, 0], pos[:, 1], s=22, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3,
                label=f"appearance ({len(pos)}), arrow = velocity x 1 s")
    if len(initial) and not include_initial:
        ax1.scatter(initial[:, 0], initial[:, 1], s=30, facecolors="none", edgecolors=BLUE, linewidths=1.0,
                    zorder=3, label=f"present in first frame ({len(initial)}, not counted)")
    if scene.destinations is not None:
        for ax in (ax1, ax2):
            ax.scatter(scene.destinations[:, 0], scene.destinations[:, 1], marker="x", s=70, color=ORANGE,
                       linewidths=1.8, zorder=4, label="destinations.txt (ETH)")
    ax1.set_title("Where pedestrians first appear", color=TEXT, fontsize=11)
    ax1.legend(loc="best", fontsize=7.5, frameon=False, labelcolor=TEXT)

    lo, hi = allp.min(axis=0) - cell, allp.max(axis=0) + cell
    xedges = np.arange(lo[0], hi[0] + cell, cell)
    yedges = np.arange(lo[1], hi[1] + cell, cell)
    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[xedges, yedges])
    Hm = np.ma.masked_where(H.T == 0, H.T)  # transparent empty cells
    mesh = ax2.pcolormesh(xedges, yedges, Hm, cmap=BLUES, vmin=1, vmax=max(H.max(), 2), alpha=0.9, zorder=2)
    cb = fig.colorbar(mesh, ax=ax2, fraction=0.04, pad=0.02)
    cb.set_label(f"appearances per {cell:g} m cell", color=TEXT, fontsize=8)
    cb.ax.tick_params(colors=TEXT, labelsize=8)
    cb.outline.set_visible(False)
    ax2.set_title(f"Appearance histogram ({cell:g} m cells)", color=TEXT, fontsize=11)
    if scene.destinations is not None:
        ax2.legend(loc="best", fontsize=7.5, frameon=False, labelcolor=TEXT)

    rate = scene.arrival_rate(include_initial)
    fig.suptitle(
        f"{scene.name}: {len(pos)} arrivals in {scene.duration_s:.0f} s  "
        f"({60 * rate:.1f} per minute, mean gap {1 / rate:.1f} s)",
        color=TEXT, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_arrivals_over_time(scene: SceneAppearances, include_initial: bool, path: str, dpi: int) -> None:
    """Left: cumulative arrival count vs. time; a homogeneous Poisson process
    follows the dashed straight line on average. Right: inter-arrival times vs.
    the exponential density with the same mean (the Poisson assumption)."""
    t = scene.times(include_initial)
    rate = scene.arrival_rate(include_initial)
    gaps = np.diff(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    _style_axes(ax1)
    _style_axes(ax2)

    ax1.step(t, np.arange(1, len(t) + 1), where="post", color=BLUE, linewidth=2, label="observed arrivals")
    ax1.plot([0, scene.duration_s], [0, rate * scene.duration_s], "--", color=TEXT, linewidth=1.2,
             label=f"constant rate {rate:.3f}/s")
    ax1.set_xlabel("time since first annotated frame [s]")
    ax1.set_ylabel("cumulative arrivals")
    ax1.set_title("Arrivals over time", color=TEXT, fontsize=11)
    ax1.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT)

    if len(gaps):
        bin_w = scene.dt
        edges = np.arange(0, gaps.max() + 2 * bin_w, bin_w) - bin_w / 2
        ax2.hist(gaps, bins=edges, density=True, color=BLUE, alpha=0.8, label="inter-arrival times")
        xs = np.linspace(0, edges[-1], 300)
        ax2.plot(xs, rate * np.exp(-rate * xs), color=TEXT, linewidth=1.5, label="exponential, same mean")
        ax2.text(0.98, 0.9, f"{np.mean(gaps == 0):.0%} of gaps are 0 s\n(same-frame arrivals, e.g. groups)",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=TEXT)
    ax2.set_xlabel("inter-arrival time [s]")
    ax2.set_ylabel("density")
    ax2.set_title("Inter-arrival times", color=TEXT, fontsize=11)
    ax2.legend(loc="center right", fontsize=8, frameon=False, labelcolor=TEXT)

    fig.suptitle(scene.name, color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main():
    args = parse_args()
    scenes = load_scenes(args.data, dt=args.dt)
    rows: List[dict] = [summary_row(s, args.include_initial, args.edge_margin) for s in scenes]

    print(format_summary_table(rows))
    print(
        "\nrate_per_s excludes pedestrians present in the first frame unless --include-initial;"
        f"\nfrac_edge<1m = fraction of arrivals within {args.edge_margin:g} m of the annotated-area bounding box;"
        "\ndt_check_s   = time step implied by the obsmat velocity columns (should equal --dt)."
    )

    os.makedirs(args.plot_dir, exist_ok=True)
    with open(os.path.join(args.plot_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if args.no_plots:
        return
    for s in scenes:
        plot_appearance_map(s, args.cell, args.include_initial,
                            os.path.join(args.plot_dir, f"{s.name}_appearance_map.png"), args.dpi)
        plot_arrivals_over_time(s, args.include_initial,
                                os.path.join(args.plot_dir, f"{s.name}_arrivals_time.png"), args.dpi)
    print(f"\nSaved {2 * len(scenes)} figures and summary.csv to {args.plot_dir}/")


if __name__ == "__main__":
    main()
