"""Hands-on exercise: does a Poisson process describe pedestrian arrivals in time?

The Poisson model says: arrivals happen at a constant rate `lam` (per second),
independently of each other. Its one testable prediction for counting is

    N(window of length T) ~ Poisson(mean = lam * T)

and a Poisson distribution has variance == mean. So the exercise is:

    1. Fit lam = (number of arrivals) / (recording duration).
    2. Chop the recording into windows of T seconds and count arrivals in each.
    3. Compare the observed counts with Poisson(lam * T):
         - histogram of counts vs. the Poisson probabilities,
         - variance / mean ("dispersion index": 1 for Poisson, > 1 means bursty).

Nothing here is used by the model code; it is for building intuition.

Example:
    python -m social_lstm.birth.exercise_counts --data data/ucy/zara01 --window 10
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from social_lstm.birth.appearance import DEFAULT_DT, DEFAULT_SCENES, load_scenes

BLUE = "#2a78d6"
TEXT = "#52514e"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=list(DEFAULT_SCENES), help="scene directories")
    p.add_argument("--window", type=float, default=10, help="window length T in seconds")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--plot-dir", default="plots/birth")
    return p.parse_args()


def poisson_pmf(k: np.ndarray, mean: float) -> np.ndarray:
    """P(N = k) = mean^k * exp(-mean) / k!   (computed in log space to avoid overflow)."""
    k = np.asarray(k)
    log_p = k * math.log(mean) - mean - np.array([math.lgamma(ki + 1) for ki in k])
    return np.exp(log_p)


def count_in_windows(times: np.ndarray, duration: float, window: float) -> np.ndarray:
    """Number of arrivals in each consecutive window [0,T), [T,2T), ... that fits in `duration`."""
    n_windows = int(duration // window)  # drop the partial last window
    edges = np.arange(n_windows + 1) * window
    counts, _ = np.histogram(times, bins=edges)
    return counts


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    scenes = load_scenes(args.data, dt=args.dt)

    # One row per scene. Left: the counts laid out in time. Right: their tally.
    fig, axes = plt.subplots(len(scenes), 2, figsize=(12, 3.2 * len(scenes)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1]})
    print(f"window T = {args.window:g} s")
    print(f"{'scene':<11s} {'lam/s':>7s} {'lam*T':>7s} {'windows':>8s} {'mean':>6s} {'var':>6s} {'var/mean':>9s} {'P(0)':>6s} {'P(>=5)':>7s}")
    for (ax_time, ax_hist), scene in zip(axes, scenes):
        # Step 1: the only parameter of the model.
        lam = scene.arrival_rate()  # arrivals / duration, excluding people present at frame 0
        expected = lam * args.window

        # Step 2: observed counts per window, in time order.
        counts = count_in_windows(scene.times(), scene.duration_s, args.window)
        mean, var = counts.mean(), counts.var(ddof=1)

        # Step 3: tally the counts and compare to Poisson(expected), scaled to the
        # number of windows so both are "number of windows with k arrivals".
        ks = np.arange(0, counts.max() + 3)
        observed = np.bincount(counts, minlength=len(ks))[: len(ks)]
        predicted = poisson_pmf(ks, expected) * len(counts)

        print(f"{scene.name:<11s} {lam:7.3f} {expected:7.2f} {len(counts):8d} {mean:6.2f} {var:6.2f} {var / mean:9.2f} "
              f"{poisson_pmf(np.array([0]), expected)[0]:6.3f} {poisson_pmf(ks, expected)[5:].sum():7.3f}")

        starts = np.arange(len(counts)) * args.window
        ax_time.bar(starts, counts, width=args.window * 0.9, align="edge", color=BLUE, alpha=0.8,
                    label="arrivals in window")
        ax_time.axhline(expected, color=TEXT, linestyle="--", linewidth=1.2, label=f"lam*T = {expected:.1f}")
        ax_time.set_title(f"{scene.name}: {len(counts)} windows of {args.window:g} s", fontsize=10, color=TEXT)
        ax_time.set_xlabel("time since first annotated frame [s]")
        ax_time.set_ylabel("arrivals per window")
        ax_time.legend(fontsize=7, frameon=False, loc="upper right")

        ax_hist.bar(ks, observed, color=BLUE, alpha=0.8, label="observed")
        ax_hist.plot(ks, predicted, "o-", color=TEXT, markersize=4, linewidth=1.2,
                     label=f"Poisson(mean={expected:.1f})")
        ax_hist.set_title(f"tally: var/mean = {var / mean:.2f}", fontsize=10, color=TEXT)
        ax_hist.set_xlabel(f"arrivals per {args.window:g} s window")
        ax_hist.set_ylabel("number of windows")
        ax_hist.legend(fontsize=7, frameon=False)
        for ax in (ax_time, ax_hist):
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    print("\nvar/mean = 1 is exactly Poisson; > 1 means arrivals are burstier than independent arrivals.")

    fig.tight_layout()
    out = os.path.join(args.plot_dir, f"exercise_counts_T{args.window:g}.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
