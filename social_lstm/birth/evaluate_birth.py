"""Step 2 of the Poisson birth baseline: fit and evaluate on held-out time.

For every scene:
    1. Split appearance events by time: the first --train-frac of the recording
       is used to fit, the rest (default: last 20 %) is held out.
    2. Fit a `GridBirthModel` (scalar rate + --cell m spatial grid) on the
       training part, and a uniform model with the same scalar rate as baseline.
    3. Score both on the held-out part:
         - predicted vs. actual total number of arrivals,
         - predicted vs. actual arrivals per --region m square (RMSE and the
           Poisson log-probability of the actual counts),
         - Poisson point-process log-likelihood of the held-out events, and the
           per-event spatial log-density (how well the model says *where*,
           separately from *how many*).
    4. Save one figure per scene to --plot-dir (<scene>_birth_model.png):
       fitted intensity with held-out events, predicted vs. actual per region,
       and one sampled realisation of births next to what actually happened.

Example:
    python -m social_lstm.birth.evaluate_birth
    python -m social_lstm.birth.evaluate_birth --data data/ucy/zara01 --cell 1.0 --region 4.0
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

from social_lstm.birth.appearance import DEFAULT_DT, DEFAULT_SCENES, SceneAppearances, load_scenes
from social_lstm.birth.poisson_birth import Bounds, GridBirthModel, poisson_log_pmf

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TRAJ_GRAY = "#b8b7b2"
TEXT = "#52514e"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues_seq", ["#ffffff", "#dbe8f8", "#2a78d6", "#0d3a73"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=list(DEFAULT_SCENES), help="scene directories or obsmat.txt files")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="seconds per annotated frame step")
    p.add_argument("--cell", type=float, default=0.5, help="birth-intensity grid cell size [m]")
    p.add_argument("--region", type=float, default=2.0, help="square size [m] for the per-region count comparison")
    p.add_argument("--train-frac", type=float, default=0.8, help="fraction of the recording (by time) used to fit")
    p.add_argument("--uniform-mix", type=float, default=0.05, help="fraction of the rate spread uniformly (smoothing)")
    p.add_argument("--seed", type=int, default=0, help="seed for the sampled realisation in the plots")
    p.add_argument("--plot-dir", default="plots/birth")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def scene_bounds(scene: SceneAppearances, margin: float) -> Bounds:
    """Bounding box of every annotated position, padded by `margin` meters. This is
    scene geometry (where people can walk), not appearance data, so using the whole
    recording for it does not leak held-out events into the fit."""
    lo, hi = scene.all_positions.min(axis=0) - margin, scene.all_positions.max(axis=0) + margin
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def split_by_time(scene: SceneAppearances, train_frac: float):
    """Appearance positions before / after the split time, and the two durations."""
    t_split = train_frac * scene.duration_s
    ev = scene.arrivals()
    train = np.array([e.position for e in ev if e.time_s < t_split]).reshape(-1, 2)
    test = np.array([e.position for e in ev if e.time_s >= t_split]).reshape(-1, 2)
    return train, test, t_split, scene.duration_s - t_split


def evaluate_scene(scene: SceneAppearances, args) -> dict:
    train, test, t_train, t_test = split_by_time(scene, args.train_frac)
    bounds = scene_bounds(scene, margin=args.cell)
    grid = GridBirthModel.fit(train, t_train, bounds, cell=args.cell, uniform_mix=args.uniform_mix)
    unif = GridBirthModel.uniform(grid.total_rate, bounds, cell=args.cell)

    # Per-region counts: model prediction vs. what happened in the held-out time.
    rx, ry, exp_grid = grid.region_counts(t_test, args.region)
    _, _, exp_unif = unif.region_counts(t_test, args.region)
    actual, _, _ = np.histogram2d(test[:, 0], test[:, 1], bins=[rx, ry])

    def spatial_per_event(m: GridBirthModel) -> float:
        # log of the normalised spatial density  lambda(x) / lam  averaged over events
        return float(np.mean(np.log(m.intensity(test) / m.total_rate))) if len(test) else float("nan")

    row = {
        "scene": scene.name,
        "train_n": len(train),
        "test_n": len(test),
        "pred_n": grid.expected_count(t_test),
        "rate_per_s": grid.total_rate,
        "ll_grid": grid.log_likelihood(test, t_test),
        "ll_unif": unif.log_likelihood(test, t_test),
        "spatial_grid": spatial_per_event(grid),
        "spatial_unif": spatial_per_event(unif),
        "region_rmse_grid": float(np.sqrt(np.mean((exp_grid - actual) ** 2))),
        "region_rmse_unif": float(np.sqrt(np.mean((exp_unif - actual) ** 2))),
        "region_logp_grid": float(poisson_log_pmf(actual, exp_grid).sum()),
        "region_logp_unif": float(poisson_log_pmf(actual, exp_unif).sum()),
        "n_regions": int(actual.size),
    }
    extras = dict(grid=grid, unif=unif, train=train, test=test, t_test=t_test,
                  rx=rx, ry=ry, exp_grid=exp_grid, exp_unif=exp_unif, actual=actual)
    return row, extras


def _style(ax: plt.Axes, equal: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=TRAJ_GRAY, alpha=0.35, linewidth=0.6)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def plot_scene(scene: SceneAppearances, row: dict, ex: dict, args) -> None:
    grid: GridBirthModel = ex["grid"]
    test, t_test = ex["test"], ex["t_test"]
    allp = scene.all_positions
    sample = grid.sample(t_test, np.random.default_rng(args.seed))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=SURFACE,
                                        gridspec_kw={"width_ratios": [1.25, 1, 1.25]})

    # (1) fitted intensity + held-out events
    _style(ax1, equal=True)
    mesh = ax1.pcolormesh(grid.x_edges, grid.y_edges, grid.rate.T / grid.cell_area, cmap=BLUES, shading="flat")
    cb = fig.colorbar(mesh, ax=ax1, fraction=0.04, pad=0.02)
    cb.set_label("fitted intensity [arrivals / s / m²]", color=TEXT, fontsize=8)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.outline.set_visible(False)
    ax1.scatter(test[:, 0], test[:, 1], s=16, color=ORANGE, edgecolors="white", linewidths=0.5, zorder=3,
                label=f"held-out arrivals ({len(test)})")
    ax1.set_title(f"Fit on first {100 * args.train_frac:.0f} % ({row['train_n']} arrivals), {args.cell:g} m cells",
                  fontsize=10, color=TEXT)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.5, frameon=False, labelcolor=TEXT)

    # (2) predicted vs actual per region
    _style(ax2)
    a, eg, eu = ex["actual"].ravel(), ex["exp_grid"].ravel(), ex["exp_unif"].ravel()
    top = max(a.max(), eg.max(), 1.0) * 1.08
    ax2.plot([0, top], [0, top], "--", color=TEXT, linewidth=1, label="perfect")
    ax2.scatter(eu, a, s=22, color=TRAJ_GRAY, label=f"uniform (RMSE {row['region_rmse_unif']:.2f})", zorder=2)
    ax2.scatter(eg, a, s=22, color=BLUE, edgecolors="white", linewidths=0.5,
                label=f"grid (RMSE {row['region_rmse_grid']:.2f})", zorder=3)
    ax2.set_xlabel(f"predicted arrivals per {args.region:g} m region")
    ax2.set_ylabel("actual arrivals")
    ax2.set_title(f"Held-out {t_test:.0f} s: {row['n_regions']} regions", fontsize=10, color=TEXT)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=TEXT)

    # (3) one sampled realisation vs reality
    _style(ax3, equal=True)
    ax3.scatter(allp[:, 0], allp[:, 1], s=2, color=TRAJ_GRAY, alpha=0.2, linewidths=0)
    ax3.scatter(test[:, 0], test[:, 1], s=16, color=ORANGE, edgecolors="white", linewidths=0.5, zorder=3,
                label=f"actual held-out ({len(test)})")
    ax3.scatter(sample.positions[:, 0], sample.positions[:, 1], s=16, color=BLUE, edgecolors="white",
                linewidths=0.5, zorder=3, label=f"one sample from model ({len(sample.times)})")
    ax3.set_title(f"Sampled births for {t_test:.0f} s (expected {row['pred_n']:.1f})", fontsize=10, color=TEXT)
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5, frameon=False, labelcolor=TEXT)

    fig.suptitle(
        f"{scene.name}: held-out log-likelihood grid {row['ll_grid']:.1f} vs uniform {row['ll_unif']:.1f}",
        color=TEXT, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, f"{scene.name}_birth_model.png"), dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)


def format_table(rows: List[dict]) -> str:
    cols = [
        ("scene", "scene", "{:<11s}"), ("train_n", "train_n", "{:7d}"), ("test_n", "test_n", "{:6d}"),
        ("pred_n", "pred_n", "{:6.1f}"), ("ll_grid", "ll_grid", "{:8.1f}"), ("ll_unif", "ll_unif", "{:8.1f}"),
        ("spatial_grid", "sp/evt_grid", "{:11.2f}"), ("spatial_unif", "sp/evt_unif", "{:11.2f}"),
        ("region_rmse_grid", "rmse_grid", "{:9.2f}"), ("region_rmse_unif", "rmse_unif", "{:9.2f}"),
        ("region_logp_grid", "rlogp_grid", "{:10.1f}"), ("region_logp_unif", "rlogp_unif", "{:10.1f}"),
    ]
    head = " ".join(h if i == 0 else h.rjust(len(fmt.format(0.0 if "f" in fmt else 0))) for i, (_, h, fmt) in enumerate(cols))
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(" ".join(fmt.format(r[k]) for k, _, fmt in cols))
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    rows: List[dict] = []
    for scene in load_scenes(args.data, dt=args.dt):
        row, extras = evaluate_scene(scene, args)
        rows.append(row)
        if not args.no_plots:
            plot_scene(scene, row, extras, args)

    print(f"cell {args.cell:g} m, regions {args.region:g} m, train fraction {args.train_frac:g}, "
          f"uniform mix {args.uniform_mix:g}\n")
    print(format_table(rows))
    print(
        "\npred_n       = rate fitted on the training part x held-out duration (compare with test_n)"
        "\nll_*         = Poisson point-process log-likelihood of held-out events (higher is better)"
        "\nsp/evt_*     = mean log spatial density of held-out events (where, independent of how many)"
        "\nrmse_*       = RMSE of predicted vs actual counts per region"
        "\nrlogp_*      = sum of log Poisson probabilities of the actual region counts (higher is better)"
    )
    with open(os.path.join(args.plot_dir, "birth_eval_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if not args.no_plots:
        print(f"\nSaved {len(rows)} figures and birth_eval_summary.csv to {args.plot_dir}/")


if __name__ == "__main__":
    main()
