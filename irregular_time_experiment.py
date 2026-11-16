"""Predict from observations that are not equally spaced in time, by letting a
Kalman filter resample them onto the LSTM's fixed frame grid.

Pipeline per pedestrian:
  1. Noisy position measurements arrive at arbitrary timestamps (frame units,
     1 frame = 0.4 s in Zara).
  2. A constant-velocity Kalman filter with continuous white-noise acceleration
     (consistent for any step size) runs over the measurements at their true
     times and reports its estimate at each grid time 0..obs_len-1. Grid times
     with no nearby measurement are predict-only, so P grows through gaps and
     with staleness at the last grid time ("now").
  3. The uniform filtered history goes to the unchanged LSTM; the final P is
     propagated with the CV term and added to the LSTM covariance (validated in
     state_uncertainty_experiment.py).

Timing scenarios:
  regular  every grid frame measured (reference)
  dropout  each grid frame independently missing with probability --drop-prob
  async    continuous timestamps: random phase per pedestrian, intervals
           ~ U(1 - j, 1 + j) frames with j = --async-jitter
  both     async timestamps, then dropout on top

Ground truth at off-grid times is linearly interpolated between annotations.
Baseline ("interp"): linearly interpolate the raw noisy measurements onto the
grid, no filter, LSTM covariance only.

Example:
    python irregular_time_experiment.py --checkpoint checkpoints/social_lstm_zara_varobs.pt
"""
from __future__ import annotations

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import crop, split_sequences
from social_lstm.dataset import load_sequences
from social_lstm.model import SocialLSTM
from state_uncertainty_experiment import (
    cv_propagated_cov,
    delta_esv,
    estimate_accel_std,
    fmt_esv,
    nll,
    run_rollout,
    sq_mahalanobis,
)

SCENARIOS = ["regular", "dropout", "async", "both"]
VARIANTS = ["interp", "kf", "kf+cv"]
STALENESS_BINS = [(0.0, 0.5), (0.5, 1.5), (1.5, np.inf)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/social_lstm_zara_varobs.pt")
    p.add_argument("--data", nargs="+", default=["data/ucy/zara01", "data/ucy/zara02"])
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--noise", type=float, nargs="+", default=[0.1, 0.2], help="measurement std (m)")
    p.add_argument("--drop-prob", type=float, default=0.4)
    p.add_argument("--async-jitter", type=float, default=0.5)
    p.add_argument("--realizations", type=int, default=20, help="noise + timing draws per window")
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0, help="must match main.py's seed to reproduce its split")
    p.add_argument("--plot-dir", default="plots/irregular_time")
    return p.parse_args()


# ----------------------------------------------------------------------------- Kalman filter

def cwna_matrices(dt: float, qc: float):
    """CV transition and continuous white-noise-acceleration process noise over a
    step of `dt` frames, state [px, py, vx, vy]. Unlike the piecewise-constant
    form, two steps of dt/2 give exactly the covariance of one step of dt."""
    I = np.eye(2)
    F = np.block([[I, dt * I], [np.zeros((2, 2)), I]])
    Q = qc * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I], [dt ** 2 / 2 * I, dt * I]])
    return F, Q


def irregular_kalman(times, z, grid, sigma, qc, v0_std=0.6):
    """Filter measurements z (M, 2) taken at sorted `times` (M,) and report the
    estimate at each `grid` time. Grid times before the first measurement are
    back-extrapolated from the earliest grid estimate with the final velocity.

    Returns grid positions (G, 2), final state (4,) and covariance (4, 4) at grid[-1].
    """
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = sigma ** 2 * np.eye(2)
    x = np.concatenate([z[0], [0.0, 0.0]])
    P = np.diag([sigma ** 2, sigma ** 2, v0_std ** 2, v0_std ** 2])
    t = times[0]

    def predict(x, P, dt):
        F, Q = cwna_matrices(dt, qc)
        return F @ x, F @ P @ F.T + Q

    out = np.full((len(grid), 2), np.nan)
    i = 1
    for g, tg in enumerate(grid):
        while i < len(times) and times[i] <= tg:
            x, P = predict(x, P, times[i] - t)
            t = times[i]
            K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
            x = x + K @ (z[i] - H @ x)
            P = (np.eye(4) - K @ H) @ P
            i += 1
        if tg < times[0]:
            continue
        x, P = predict(x, P, tg - t)
        t = tg
        out[g] = x[:2]

    missing = np.isnan(out[:, 0])
    if missing.any():
        g0 = np.argmax(~missing)
        out[missing] = out[g0] + (grid[missing] - grid[g0])[:, None] * x[2:]
    return out, x, P


# ----------------------------------------------------------------------------- timing

def sample_times(scenario, t_now, drop_prob, jitter, rng):
    """Measurement timestamps in [0, t_now] for one pedestrian (at least two)."""
    while True:
        if scenario in ("regular", "dropout"):
            times = np.arange(0.0, t_now + 1e-9)
        else:
            t, ts = rng.uniform(0.0, 1.0), []
            while t <= t_now:
                ts.append(t)
                t += rng.uniform(1.0 - jitter, 1.0 + jitter)
            times = np.array(ts)
        if scenario in ("dropout", "both"):
            times = times[rng.uniform(size=len(times)) >= drop_prob]
        if len(times) >= 2:
            return times


def interp_track(track, times):
    """Positions of a (T, 2) grid track at arbitrary `times` (linear interpolation)."""
    grid = np.arange(len(track))
    return np.stack([np.interp(times, grid, track[:, d]) for d in range(2)], -1)


# ----------------------------------------------------------------------------- experiment

def experiment(model, seqs, scenario, obs_len, max_obs_len, pred_len, sigma, qc, args, rng):
    t_now = obs_len - 1
    grid = np.arange(obs_len, dtype=np.float64)
    d2 = {v: [] for v in VARIANTS}
    nlls = {v: [] for v in VARIANTS}
    disp = {v: [] for v in ("interp", "kf")}
    stale, n_meas, nees = [], [], []
    for seq in seqs:
        clean = crop(seq.positions, obs_len, max_obs_len).astype(np.float64)
        target = clean[obs_len:]
        num_peds = clean.shape[1]
        for _ in range(args.realizations):
            kf_hist = np.zeros((obs_len, num_peds, 2))
            lin_hist = np.zeros((obs_len, num_peds, 2))
            P_all = np.zeros((num_peds, 4, 4))
            for p in range(num_peds):
                times = sample_times(scenario, t_now, args.drop_prob, args.async_jitter, rng)
                z = interp_track(clean[:obs_len, p], times) + sigma * rng.standard_normal((len(times), 2))
                kf_hist[:, p], x, P_all[p] = irregular_kalman(times, z, grid, sigma, qc)
                lin_hist[:, p] = np.stack([np.interp(grid, times, z[:, d]) for d in range(2)], -1)
                err_now = clean[t_now, p] - x[:2]
                nees.append(err_now @ np.linalg.inv(P_all[p][:2, :2]) @ err_now)
                stale.append(t_now - times[-1])
                n_meas.append(len(times))

            outs = {}
            for name, hist in (("interp", lin_hist), ("kf", kf_hist)):
                inp = clean.copy()
                inp[:obs_len] = hist
                outs[name] = run_rollout(model, inp, obs_len, pred_len)
            covs = {
                "interp": outs["interp"],
                "kf": outs["kf"],
                "kf+cv": (outs["kf"][0], outs["kf"][1] + cv_propagated_cov(P_all, pred_len)),
            }
            for v, (mu, cov) in covs.items():
                err = target - mu
                d2[v].append(sq_mahalanobis(err, cov))
                nlls[v].append(nll(err, cov))
            for v in disp:
                disp[v].append(np.linalg.norm(target - outs[v][0], axis=-1))

    cat = lambda xs: np.concatenate(xs, axis=1)
    return {
        "d2": {v: cat(x) for v, x in d2.items()},
        "nll": {v: cat(x) for v, x in nlls.items()},
        "disp": {v: cat(x) for v, x in disp.items()},
        "stale": np.array(stale),
        "n_meas": float(np.mean(n_meas)),
        "nees": float(np.mean(nees)),
    }


def report(scenario, r):
    print(f"\n[{scenario}]  meas/ped={r['n_meas']:.1f}  mean staleness={r['stale'].mean():.2f} frames  "
          f"KF NEES pos@now={r['nees']:.2f} (ideal 2)")
    for v in ("interp", "kf"):
        e = r["disp"][v]
        print(f"  {v:6s} ADE={e.mean():.3f}  FDE={e[-1].mean():.3f}")
    for v in VARIANTS:
        print(f"  {v:6s} dESV {fmt_esv(r['d2'][v])}   NLL={r['nll'][v].mean():.3f}")
    # Staleness is per (realization, pedestrian), matching the pedestrian axis of the metrics.
    parts = []
    for lo, hi in STALENESS_BINS:
        m = (r["stale"] >= lo) & (r["stale"] < hi)
        if m.any():
            d2 = r["d2"]["kf+cv"][:, m]
            parts.append(f"[{lo:g},{hi:g}) n={m.sum()}: NLL={r['nll']['kf+cv'][:, m].mean():.2f} "
                         f"dESV2={delta_esv(d2)[1]:+.3f}")
    print("  kf+cv by staleness (frames): " + " | ".join(parts))


def plot(results, sigmas, pred_len, plot_dir, obs_len):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 1 + len(sigmas), figsize=(5 * (1 + len(sigmas)), 4.5))
    width = 0.8 / len(VARIANTS)
    xs = np.arange(len(SCENARIOS))
    ax = axes[0]
    for j, v in enumerate(VARIANTS):
        for i_s, sigma in enumerate(sigmas):
            vals = [results[(s, sigma)]["nll"][v].mean() for s in SCENARIOS]
            ax.bar(xs + j * width + i_s * width / len(sigmas), vals, width / len(sigmas),
                   color=plt.cm.tab10(j), alpha=1.0 - 0.4 * i_s,
                   label=f"{v} σ={sigma}")
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels(SCENARIOS)
    ax.set(title="mean NLL (lower is better)", ylabel="NLL")
    ax.legend(fontsize=7)
    ks = np.arange(1, pred_len + 1)
    for ax, sigma in zip(axes[1:], sigmas):
        for i, s in enumerate(SCENARIOS):
            for lvl, ls in ((0, "-"), (1, "--")):
                esv = [delta_esv(results[(s, sigma)]["d2"]["kf+cv"][k])[lvl] for k in range(pred_len)]
                ax.plot(ks, esv, ls, color=plt.cm.tab10(i), label=f"{s} ΔESV{lvl + 1}")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(title=f"kf+cv calibration per horizon, σ={sigma} m", xlabel="horizon step k")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(plot_dir, f"obs{obs_len}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")


def main():
    args = parse_args()
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    max_obs_len = ckpt.get("max_obs_len", ckpt["obs_len"])
    pred_len = ckpt["pred_len"]
    model = SocialLSTM(**ckpt["hyperparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sequences = load_sequences(args.data, obs_len=max_obs_len, pred_len=pred_len, skip=args.skip)
    random.shuffle(sequences)
    train_seqs, val_seqs, test_seqs = split_sequences(sequences, args.val_frac, args.test_frac)
    held_out = test_seqs + val_seqs
    # Var of second differences under CWNA sampled at dt = 1 is 2/3 qc.
    qc = 1.5 * estimate_accel_std(train_seqs) ** 2
    print(f"{len(held_out)} held-out windows, obs_len={args.obs_len}, qc={qc:.2e} m^2/frame^3, "
          f"drop_prob={args.drop_prob}, async intervals U({1 - args.async_jitter:g},{1 + args.async_jitter:g})")

    clean_err = []
    for seq in held_out:
        clean = crop(seq.positions, args.obs_len, max_obs_len).astype(np.float64)
        mu, _ = run_rollout(model, clean, args.obs_len, pred_len)
        clean_err.append(np.linalg.norm(clean[args.obs_len:] - mu, axis=-1))
    clean_err = np.concatenate(clean_err, axis=1)
    print(f"clean regular inputs (reference): ADE={clean_err.mean():.3f}  FDE={clean_err[-1].mean():.3f}")

    results = {}
    for sigma in args.noise:
        print(f"\n===== sigma={sigma} m =====   dESV (1sig 2sig 3sig |mean|)")
        for scenario in SCENARIOS:
            r = experiment(model, held_out, scenario, args.obs_len, max_obs_len, pred_len, sigma, qc, args, rng)
            results[(scenario, sigma)] = r
            report(scenario, r)
    plot(results, args.noise, pred_len, args.plot_dir, args.obs_len)


if __name__ == "__main__":
    main()
