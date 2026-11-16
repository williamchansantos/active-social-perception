"""Test whether initial-state (estimation) uncertainty can be handled by adding a
constant-velocity-propagated covariance to the Social LSTM's predicted covariance.

For each held-out Zara window and each measurement-noise level sigma:
  1. Corrupt every observed position (all pedestrians) with N(0, sigma^2 I).
  2. Run a constant-velocity Kalman filter per pedestrian over the noisy
     measurements -> filtered position history and final state covariance P
     over [px, py, vx, vy] (per-frame velocity units).
  3. Feed the filtered history to the trained LSTM and roll out pred_len steps.
  4. Score several total-covariance hypotheses against the clean ground truth:
       model    Sigma_model(k)
       +const   Sigma_model(k) + P_pp
       +cv      Sigma_model(k) + J_k P J_k^T,  J_k = [I, k I]
  5. Propagation check: across R noise realizations of the same window, compare
     E_r[(mu_k - mu_k^clean)(mu_k - mu_k^clean)^T], how far the LSTM's mean moves
     from its clean-input prediction, with the CV prediction J_k P J_k^T.

Example:
    python state_uncertainty_experiment.py --checkpoint checkpoints/social_lstm_zara_varobs.pt
"""
from __future__ import annotations

import argparse
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import crop, split_sequences
from social_lstm.dataset import load_sequences
from social_lstm.engine import predict_sequence_with_gaussian
from social_lstm.model import SocialLSTM

VARIANTS = ["model", "+const", "+cv"]
SIGMA_LEVELS = (1, 2, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/social_lstm_zara_varobs.pt")
    p.add_argument("--data", nargs="+", default=["data/ucy/zara01", "data/ucy/zara02"])
    p.add_argument("--obs-lens", type=int, nargs="+", default=[8])
    p.add_argument("--noise", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3], help="measurement std (m)")
    p.add_argument("--realizations", type=int, default=10, help="noise draws per window")
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0, help="must match main.py's seed to reproduce its split")
    p.add_argument("--plot-dir", default="plots/state_uncertainty")
    return p.parse_args()


# ----------------------------------------------------------------------------- Kalman filter

def cv_matrices(q: float):
    """Per-frame (dt = 1) CV transition and white-acceleration process noise, state [px, py, vx, vy]."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = 1.0
    G = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]])
    return F, q ** 2 * G @ G.T


def kalman_filter(z: np.ndarray, sigma: float, q: float, v0_std: float = 0.6):
    """Filter noisy positions z (T, N, 2). Returns filtered positions (T, N, 2),
    final state mean (N, 4) and covariance (N, 4, 4)."""
    T, N, _ = z.shape
    F, Q = cv_matrices(q)
    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    R = sigma ** 2 * np.eye(2)

    x = np.zeros((N, 4))
    x[:, :2] = z[0]
    P = np.tile(np.diag([sigma ** 2, sigma ** 2, v0_std ** 2, v0_std ** 2]), (N, 1, 1))
    filtered = [x[:, :2].copy()]
    for t in range(1, T):
        x = x @ F.T
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + np.einsum("nij,nj->ni", K, z[t] - x[:, :2])
        P = (np.eye(4) - K @ H) @ P
        filtered.append(x[:, :2].copy())
    return np.stack(filtered), x, P


def cv_propagated_cov(P: np.ndarray, pred_len: int) -> np.ndarray:
    """J_k P J_k^T for k = 1..pred_len, J_k = [I, k I]. Returns (pred_len, N, 2, 2)."""
    out = []
    for k in range(1, pred_len + 1):
        J = np.hstack([np.eye(2), k * np.eye(2)])
        out.append(J @ P @ J.T)
    return np.stack(out)


# ----------------------------------------------------------------------------- metrics

def model_cov(sx, sy, rho) -> np.ndarray:
    cxy = rho * sx * sy
    return np.stack([np.stack([sx ** 2, cxy], -1), np.stack([cxy, sy ** 2], -1)], -2)


def sq_mahalanobis(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return np.einsum("...i,...ij,...j->...", err, np.linalg.inv(cov), err)


def nll(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return 0.5 * sq_mahalanobis(err, cov) + 0.5 * np.log(np.linalg.det(cov)) + math.log(2 * math.pi)


def delta_esv(d2: np.ndarray) -> list:
    return [float((d2 <= k ** 2).mean() - (1 - math.exp(-0.5 * k ** 2))) for k in SIGMA_LEVELS]


def estimate_accel_std(sequences) -> float:
    """Per-axis std of second differences of ground-truth positions (m / frame^2)."""
    acc = np.concatenate([np.diff(s.positions, n=2, axis=0).reshape(-1) for s in sequences])
    return float(acc.std())


# ----------------------------------------------------------------------------- experiment

@torch.no_grad()
def run_rollout(model, positions_np, obs_len, pred_len):
    pos = torch.from_numpy(positions_np.astype(np.float32))
    mu, sx, sy, rho = predict_sequence_with_gaussian(model, pos, obs_len, pred_len)
    return mu.numpy().astype(np.float64), model_cov(sx.numpy(), sy.numpy(), rho.numpy()).astype(np.float64)


def experiment(model, seqs, obs_len, max_obs_len, pred_len, sigma, q, n_real, rng):
    d2 = {v: [] for v in VARIANTS}
    nlls = {v: [] for v in VARIANTS}
    spread_tr, cv_tr, model_tr, nees_p, nees_v = [], [], [], [], []
    for seq in seqs:
        clean = crop(seq.positions, obs_len, max_obs_len).astype(np.float64)
        target = clean[obs_len:]
        true_v = clean[obs_len - 1] - clean[obs_len - 2]
        mu_clean, _ = run_rollout(model, clean, obs_len, pred_len)
        mus, cv_covs = [], []
        for _ in range(n_real):
            z = clean[:obs_len] + sigma * rng.standard_normal(clean[:obs_len].shape)
            filt, x, P = kalman_filter(z, sigma, q)
            inp = clean.copy()
            inp[:obs_len] = filt
            mu, cov_m = run_rollout(model, inp, obs_len, pred_len)
            cov_cv = cv_propagated_cov(P, pred_len)
            covs = {"model": cov_m, "+const": cov_m + P[None, :, :2, :2], "+cv": cov_m + cov_cv}
            err = target - mu
            for v, c in covs.items():
                d2[v].append(sq_mahalanobis(err, c))
                nlls[v].append(nll(err, c))
            ep, ev = clean[obs_len - 1] - x[:, :2], true_v - x[:, 2:]
            nees_p.append(sq_mahalanobis(ep, P[:, :2, :2]))
            nees_v.append(sq_mahalanobis(ev, P[:, 2:, 2:]))
            model_tr.append(np.trace(cov_m, axis1=-2, axis2=-1))
            mus.append(mu)
            cv_covs.append(cov_cv)
        mus = np.stack(mus)  # (R, pred_len, N, 2)
        # Second moment about the clean-input prediction, so filter bias (e.g. lag
        # on turns) counts as propagated error, as it does in P.
        dev = mus - mu_clean
        spread = np.einsum("rkni,rknj->knij", dev, dev) / n_real
        spread_tr.append(np.trace(spread, axis1=-2, axis2=-1))
        cv_tr.append(np.trace(np.mean(cv_covs, 0), axis1=-2, axis2=-1))

    cat = lambda xs: np.concatenate(xs, axis=1)  # stack along pedestrians, keep horizon axis
    return {
        "d2": {v: cat(x) for v, x in d2.items()},
        "nll": {v: cat(x) for v, x in nlls.items()},
        "spread": cat(spread_tr),
        "cv": cat(cv_tr),
        "model_tr": cat(model_tr),
        "nees_p": float(np.concatenate(nees_p).mean()),
        "nees_v": float(np.concatenate(nees_v).mean()),
    }


@torch.no_grad()
def clean_baseline(model, seqs, obs_len, max_obs_len, pred_len):
    d2 = []
    for seq in seqs:
        clean = crop(seq.positions, obs_len, max_obs_len).astype(np.float64)
        mu, cov = run_rollout(model, clean, obs_len, pred_len)
        d2.append(sq_mahalanobis(clean[obs_len:] - mu, cov))
    return np.concatenate(d2, axis=1)


def fmt_esv(d2):
    e = delta_esv(d2)
    return " ".join(f"{x:+.3f}" for x in e) + f"  |{np.mean(np.abs(e)):.3f}|"


def plot(results, obs_len, pred_len, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    ks = np.arange(1, pred_len + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for (sigma, r), color in zip(results.items(), cmap):
        for v, ls in zip(VARIANTS, [":", "--", "-"]):
            esv1 = [delta_esv(r["d2"][v][k])[0] for k in range(pred_len)]
            axes[0].plot(ks, esv1, ls, color=color, label=f"σ={sigma} {v}")
        ratio = r["spread"].mean(1) / r["cv"].mean(1)
        axes[1].plot(ks, ratio, "-o", color=color, markersize=3, label=f"σ={sigma}")
        axes[2].plot(ks, np.sqrt(r["cv"].mean(1)), "-", color=color, label=f"σ={sigma} CV term")
        axes[2].plot(ks, np.sqrt(r["spread"].mean(1)), "--", color=color, label=f"σ={sigma} LSTM spread")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set(title="ΔESV₁ per horizon (<0 over-confident)", xlabel="horizon step k", ylabel="ΔESV₁")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].axhline(1, color="gray", lw=0.8)
    axes[1].set(title="tr E[(μ_k-μ_k^clean)(·)ᵀ] / tr(J_k P J_kᵀ)  (1 = CV-like)", xlabel="horizon step k")
    axes[1].legend(fontsize=7)
    axes[2].set(title="√trace: CV-predicted vs actual LSTM mean spread (m)", xlabel="horizon step k")
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out = os.path.join(plot_dir, f"obs{obs_len}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


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

    # Reproduce main.py's split so only sequences unseen in training are used.
    sequences = load_sequences(args.data, obs_len=max_obs_len, pred_len=pred_len, skip=args.skip)
    random.shuffle(sequences)
    train_seqs, val_seqs, test_seqs = split_sequences(sequences, args.val_frac, args.test_frac)
    held_out = test_seqs + val_seqs
    q = estimate_accel_std(train_seqs)
    print(f"{len(held_out)} held-out windows, KF accel std q={q:.4f} m/frame^2, {args.realizations} realizations")

    for obs_len in args.obs_lens:
        base = clean_baseline(model, held_out, obs_len, max_obs_len, pred_len)
        print(f"\n=== obs_len={obs_len} ===   dESV (1sig 2sig 3sig |mean|)")
        print(f"clean inputs, model cov:            {fmt_esv(base)}")
        results = {}
        for sigma in args.noise:
            r = experiment(model, held_out, obs_len, max_obs_len, pred_len, sigma, q, args.realizations, rng)
            results[sigma] = r
            print(f"\nsigma={sigma:.2f} m   KF NEES pos={r['nees_p']:.2f} (ideal 2)  vel={r['nees_v']:.2f} (ideal 2)")
            for v in VARIANTS:
                print(f"  {v:7s} dESV {fmt_esv(r['d2'][v])}   NLL={r['nll'][v].mean():.3f}")
            ratio = r["spread"].mean(1) / r["cv"].mean(1)
            share = r["cv"].mean(1) / (r["cv"].mean(1) + r["model_tr"].mean(1))
            print("  LSTM-spread / CV-term, k=1,3,6,9,12: "
                  + " ".join(f"{ratio[k - 1]:.2f}" for k in (1, 3, 6, 9, 12)))
            print("  CV-term share of total trace,  same k: "
                  + " ".join(f"{share[k - 1]:.2f}" for k in (1, 3, 6, 9, 12)))
        plot(results, obs_len, pred_len, args.plot_dir)


if __name__ == "__main__":
    main()
