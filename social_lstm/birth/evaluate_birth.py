"""Step 2 of the Poisson birth baseline: fit and evaluate on held-out time.

For every scene:
    1. Split appearance events by time: the first --train-frac of the recording
       is used to fit, the rest (default: last 20 %) is held out.
    2. Fit on the training part: a `GridBirthModel` (scalar rate + --cell m grid),
       a `KDEBirthModel` (same rate, Gaussian kernels of --bandwidth m), and a
       uniform model with the same scalar rate as the baseline.
    3. Score all three on the held-out part:
         - predicted vs. actual total number of arrivals,
         - predicted vs. actual arrivals per --region m square (RMSE and the
           Poisson log-probability of the actual counts),
         - Poisson point-process log-likelihood of the held-out events, and the
           per-event spatial log-density (how well the model says *where*,
           separately from *how many*).
    4. Save one figure per scene to --plot-dir (<scene>_birth_model.png): grid and
       KDE intensities with held-out events, predicted vs. actual per region, and
       one sampled realisation of KDE births next to what actually happened.

Example:
    python -m social_lstm.birth.evaluate_birth
    python -m social_lstm.birth.evaluate_birth --data data/ucy/zara01 --cell 1.0 --bandwidth 0.5
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
from social_lstm.birth.poisson_birth import Bounds, GridBirthModel, KDEBirthModel, poisson_log_pmf

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
    p.add_argument("--bandwidth", type=float, default=0.5, help="KDE Gaussian kernel width [m]")
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


def evaluate_scene(scene: SceneAppearances, args):
    """Fit the three models on the first part of the scene and score them on the rest.
    Returns one summary row per model plus the fitted objects for plotting."""
    train, test, t_train, t_test = split_by_time(scene, args.train_frac)
    bounds = scene_bounds(scene, margin=args.cell)
    grid = GridBirthModel.fit(train, t_train, bounds, cell=args.cell, uniform_mix=args.uniform_mix)
    models = {
        "uniform": GridBirthModel.uniform(grid.total_rate, bounds, cell=args.cell),
        "grid": grid,
        "kde": KDEBirthModel.fit(train, t_train, bounds, bandwidth=args.bandwidth, uniform_mix=args.uniform_mix),
    }

    rows, expected = [], {}
    actual = None
    for name, m in models.items():
        rx, ry, exp = m.region_counts(t_test, args.region)
        if actual is None:
            actual, _, _ = np.histogram2d(test[:, 0], test[:, 1], bins=[rx, ry])
        expected[name] = exp
        rows.append({
            "scene": scene.name,
            "model": name,
            "train_n": len(train),
            "test_n": len(test),
            "pred_n": m.expected_count(t_test),
            "ll": m.log_likelihood(test, t_test),
            # log of the normalised spatial density lambda(x) / lam, averaged over held-out events
            "spatial_per_event": float(np.mean(np.log(m.intensity(test) / m.total_rate))) if len(test) else float("nan"),
            "region_rmse": float(np.sqrt(np.mean((exp - actual) ** 2))),
            "region_logp": float(poisson_log_pmf(actual, exp).sum()),
            "n_regions": int(actual.size),
        })
    extras = dict(models=models, train=train, test=test, t_test=t_test, rx=rx, ry=ry,
                  expected=expected, actual=actual)
    return rows, extras


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


def _intensity_panel(fig, ax, model, test, title: str) -> None:
    """Heatmap of a model's intensity on a fine raster with held-out arrivals on top."""
    _style(ax, equal=True)
    x_min, x_max, y_min, y_max = model.bounds
    xs = np.linspace(x_min, x_max, 161)
    ys = np.linspace(y_min, y_max, int(161 * (y_max - y_min) / (x_max - x_min)) + 2)
    X, Y = np.meshgrid(xs, ys)
    Z = model.intensity(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    mesh = ax.pcolormesh(xs, ys, Z, cmap=BLUES, shading="auto")
    cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("intensity [arrivals / s / m²]", color=TEXT, fontsize=8)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.outline.set_visible(False)
    ax.scatter(test[:, 0], test[:, 1], s=14, color=ORANGE, edgecolors="white", linewidths=0.5, zorder=3,
               label=f"held-out arrivals ({len(test)})")
    ax.set_title(title, fontsize=10, color=TEXT)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.5, frameon=False, labelcolor=TEXT)


def plot_scene(scene: SceneAppearances, rows: List[dict], ex: dict, args) -> None:
    by_model = {r["model"]: r for r in rows}
    models = ex["models"]
    test, t_test = ex["test"], ex["t_test"]
    allp = scene.all_positions
    sample = models["kde"].sample(t_test, np.random.default_rng(args.seed))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5.4), facecolor=SURFACE,
                                             gridspec_kw={"width_ratios": [1.2, 1.2, 0.9, 1.1]})
    _intensity_panel(fig, ax1, models["grid"], test,
                     f"Grid, {args.cell:g} m cells (LL {by_model['grid']['ll']:.1f})")
    _intensity_panel(fig, ax2, models["kde"], test,
                     f"KDE, bandwidth {args.bandwidth:g} m (LL {by_model['kde']['ll']:.1f})")

    # predicted vs actual per region
    _style(ax3)
    a = ex["actual"].ravel()
    top = max(a.max(), max(e.max() for e in ex["expected"].values()), 1.0) * 1.08
    ax3.plot([0, top], [0, top], "--", color=TEXT, linewidth=1, label="perfect")
    for name, color, z in (("uniform", TRAJ_GRAY, 2), ("grid", BLUE, 3), ("kde", ORANGE, 4)):
        ax3.scatter(ex["expected"][name].ravel(), a, s=20, color=color, edgecolors="white", linewidths=0.5,
                    zorder=z, label=f"{name} (RMSE {by_model[name]['region_rmse']:.2f})")
    ax3.set_xlabel(f"predicted arrivals per {args.region:g} m region")
    ax3.set_ylabel("actual arrivals")
    ax3.set_title(f"Held-out {t_test:.0f} s: {rows[0]['n_regions']} regions", fontsize=10, color=TEXT)
    ax3.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=TEXT)

    # one sampled realisation (from the KDE) vs reality
    _style(ax4, equal=True)
    ax4.scatter(allp[:, 0], allp[:, 1], s=2, color=TRAJ_GRAY, alpha=0.2, linewidths=0)
    ax4.scatter(test[:, 0], test[:, 1], s=14, color=ORANGE, edgecolors="white", linewidths=0.5, zorder=3,
                label=f"actual held-out ({len(test)})")
    ax4.scatter(sample.positions[:, 0], sample.positions[:, 1], s=14, color=BLUE, edgecolors="white",
                linewidths=0.5, zorder=3, label=f"one KDE sample ({len(sample.times)})")
    ax4.set_title(f"Sampled births for {t_test:.0f} s (expected {rows[0]['pred_n']:.1f})", fontsize=10, color=TEXT)
    ax4.set_xlabel("x [m]")
    ax4.set_ylabel("y [m]")
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5, frameon=False, labelcolor=TEXT)

    fig.suptitle(
        f"{scene.name}: held-out log-likelihood uniform {by_model['uniform']['ll']:.1f}, "
        f"grid {by_model['grid']['ll']:.1f}, KDE {by_model['kde']['ll']:.1f}",
        color=TEXT, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, f"{scene.name}_birth_model.png"), dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)


def format_table(rows: List[dict]) -> str:
    cols = [
        ("scene", "scene", "{:<14s}"), ("model", "model", "{:<8s}"), ("train_n", "train_n", "{:7d}"),
        ("test_n", "test_n", "{:6d}"), ("pred_n", "pred_n", "{:6.1f}"), ("ll", "ll", "{:8.1f}"),
        ("spatial_per_event", "sp/evt", "{:7.2f}"), ("region_rmse", "rmse", "{:6.2f}"),
        ("region_logp", "rlogp", "{:8.1f}"),
    ]
    head = " ".join(h.ljust(len(fmt.format("" if "s" in fmt else 0))) if "s" in fmt
                    else h.rjust(len(fmt.format(0.0 if "f" in fmt else 0))) for _, h, fmt in cols)
    lines = [head, "-" * len(head)]
    prev = None
    for r in rows:
        if prev is not None and r["scene"] != prev:
            lines.append("")
        prev = r["scene"]
        lines.append(" ".join(fmt.format(r[k]) for k, _, fmt in cols))
    return "\n".join(lines)


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    rows: List[dict] = []
    for scene in load_scenes(args.data, dt=args.dt):
        scene_rows, extras = evaluate_scene(scene, args)
        rows.extend(scene_rows)
        if not args.no_plots:
            plot_scene(scene, scene_rows, extras, args)

    print(f"grid cell {args.cell:g} m, KDE bandwidth {args.bandwidth:g} m, regions {args.region:g} m, "
          f"train fraction {args.train_frac:g}, uniform mix {args.uniform_mix:g}\n")
    print(format_table(rows))
    print(
        "\npred_n = rate fitted on the training part x held-out duration (compare with test_n)"
        "\nll     = Poisson point-process log-likelihood of held-out events (higher is better)"
        "\nsp/evt = mean log spatial density of held-out events (where, independent of how many)"
        "\nrmse   = RMSE of predicted vs actual counts per region"
        "\nrlogp  = sum of log Poisson probabilities of the actual region counts (higher is better)"
    )
    with open(os.path.join(args.plot_dir, "birth_eval_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if not args.no_plots:
        print(f"\nSaved {len(rows) // 3} figures and birth_eval_summary.csv to {args.plot_dir}/")


if __name__ == "__main__":
    main()
