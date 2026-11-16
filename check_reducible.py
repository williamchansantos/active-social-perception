"""How much prediction uncertainty can the camera actually remove, and how much does the planner assume?

On each scene's held-out half (rotation-augmented leave-one-scene-out predictors by default), every window is
predicted twice: from the true (clean) history, and from the history our tracker would produce -- noisy
measurements (sigma = --meas-std), a constant-velocity Kalman filter (as in the PMB tracker), and predict-only
steps where the camera missed the pedestrian. The extra squared error of the tracked prediction is the part
of the uncertainty that better observation removes ("actual reducible"). The planner models that part as the
CV-propagated tracker covariance tr(J_k P J_k^T) ("modelled reducible") and treats the predictor's own spread
tr(Sigma_LSTM) as irreducible.

    python check_reducible.py --prefix loorot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from evaluate_nav import SCENES, scene_frames  # noqa: E402
from robot_nav.prediction import TrajectoryPredictor  # noqa: E402
from robot_nav.tracking import Track, TrackerConfig, cwna  # noqa: E402
from social_lstm.dataset import load_obsmat  # noqa: E402
from social_lstm.engine import predict_sequence_with_gaussian  # noqa: E402

H = 8
PATTERNS = {  # name: (history length, detected-flag function over history steps)
    "fresh": (8, lambda rng: np.ones(8, bool)),
    "coasted 2": (8, lambda rng: np.r_[np.ones(6, bool), np.zeros(2, bool)]),
    "coasted 4": (8, lambda rng: np.r_[np.ones(4, bool), np.zeros(4, bool)]),
    "sparse 50%": (8, lambda rng: np.r_[True, rng.uniform(size=7) < 0.5]),
    "new (3 steps)": (3, lambda rng: np.ones(3, bool)),
}


def windows(sc, obs):
    f2p = load_obsmat(SCENES[sc]["path"])
    _, step, _, split = scene_frames(sc)
    fr = [f for f in sorted(f2p) if f >= split]
    out = []
    for j in range(len(fr) - obs - H):
        frames = fr[j: j + obs + H]
        if any(b - a != step for a, b in zip(frames[:-1], frames[1:])):
            continue
        ids = [p for p in f2p[frames[0]] if all(p in f2p[f] for f in frames)]
        if ids:
            out.append(np.array([[f2p[f][p] for p in ids] for f in frames], dtype=np.float32))
    return out


def tracked_history(true_hist, detected, meas_std, rng, tcfg, F, Q):
    """KF estimates at each history step (as the PMB tracker records them) and the final covariance."""
    R = meas_std ** 2 * np.eye(2)
    z0 = true_hist[0] + meas_std * rng.standard_normal(2)
    tr = Track(z0, R, tcfg.v0_std)
    hist = [tr.x[:2].copy()]
    for t in range(1, len(true_hist)):
        tr.predict(F, Q)
        if detected[t]:
            tr.update(true_hist[t] + meas_std * rng.standard_normal(2), R)
        hist.append(tr.x[:2].copy())
    return np.array(hist, dtype=np.float32), tr.P.copy()


def predict(model, hist):
    with torch.no_grad():
        mu, sx, sy, rho = predict_sequence_with_gaussian(model, torch.from_numpy(hist), hist.shape[0], H)
    sx, sy, rho = sx.numpy(), sy.numpy(), rho.numpy()
    cov = np.stack([np.stack([sx ** 2, rho * sx * sy], -1), np.stack([rho * sx * sy, sy ** 2], -1)], -2)
    return mu.numpy(), cov  # mean (H, n, 2), covariance (H, n, 2, 2)


def inside95(err, cov):
    d2 = np.einsum("kni,knij,knj->kn", err, np.linalg.inv(cov), err)
    return d2 < 5.991


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="loorot")
    p.add_argument("--scenes", nargs="+", default=["zara01", "zara02", "eth", "hotel"])
    p.add_argument("--meas-std", type=float, default=0.12)
    p.add_argument("--max-windows", type=int, default=600, help="per scene and pattern")
    args = p.parse_args()
    rng = np.random.default_rng(0)
    tcfg = TrackerConfig()
    F, Q = cwna(1.0, tcfg.qc)
    J = [np.hstack([np.eye(2), (k + 1) * np.eye(2)]) for k in range(H)]
    ks = (0, 3, 7)
    print(f"Squared-error budget per pedestrian (m^2, sum over x and y), held-out halves, {args.prefix} predictors, "
          f"measurement std {args.meas_std} m")
    print("  actual reducible = E|e_tracked|^2 - E|e_clean|^2 ;  modelled reducible = E tr(J P J^T) ;  "
          "LSTM spread = E tr(Sigma_LSTM)")
    print(f"{'scene':7s} {'history':14s} | {'clean err^2':>17s} | {'actual reducible':>17s} | {'modelled reducible':>18s} | "
          f"{'LSTM spread':>17s} | {'modelled / actual':>17s}")
    print(f"{'':7s} {'':14s} | {'0.4  1.6  3.2 s':>17s} | {'0.4  1.6  3.2 s':>17s} | {'0.4  1.6  3.2 s':>18s} | "
          f"{'0.4  1.6  3.2 s':>17s} | {'0.4  1.6  3.2 s':>17s}")
    pooled = {}
    for sc in args.scenes:
        predictor = TrajectoryPredictor(f"checkpoints/{args.prefix}_{sc}.pt", H)
        model = predictor.model
        for name, (obs, pat) in PATTERNS.items():
            wins = windows(sc, obs)
            if len(wins) > args.max_windows:
                wins = [wins[i] for i in rng.choice(len(wins), args.max_windows, replace=False)]
            ec, et, mod, spread, modG, in_cv, in_g = [], [], [], [], [], [], []
            for w in wins:
                n = w.shape[1]
                det = pat(rng)
                hist = np.zeros((obs, n, 2), np.float32)
                Ps = []
                for j in range(n):
                    hist[:, j], P = tracked_history(w[:obs, j], det, args.meas_std, rng, tcfg, F, Q)
                    Ps.append(P)
                mu_c, _ = predict(model, w[:obs])
                mu_t, cov_l = predict(model, hist)
                G = predictor.state_jacobian(torch.from_numpy(hist))
                Pa = np.stack(Ps)
                cov_cv = np.stack([np.einsum("ab,jbc,dc->jad", J[k], Pa, J[k]) for k in range(H)])
                cov_g = np.einsum("kjab,jbc,kjdc->kjad", G, Pa, G)
                err = w[obs:] - mu_t
                ec.append(np.sum((w[obs:] - mu_c) ** 2, -1))
                et.append(np.sum(err ** 2, -1))
                spread.append(np.trace(cov_l, axis1=2, axis2=3))
                mod.append(np.trace(cov_cv, axis1=2, axis2=3))
                modG.append(np.trace(cov_g, axis1=2, axis2=3))
                in_cv.append(inside95(err, cov_l + cov_cv))
                in_g.append(inside95(err, cov_l + cov_g))
            ec, et, mod, spread, modG, in_cv, in_g = (np.concatenate(a, 1) for a in (ec, et, mod, spread, modG, in_cv, in_g))
            act = et.mean(1) - ec.mean(1)
            for key, arr in (("ec", ec.mean(1)), ("act", act), ("mod", mod.mean(1)), ("spr", spread.mean(1)),
                             ("modG", modG.mean(1)), ("in_cv", in_cv.mean(1)), ("in_g", in_g.mean(1))):
                pooled.setdefault((name, key), []).append(arr)
            f = lambda a: " ".join(f"{a[k]:5.2f}" for k in ks)
            ratio = " ".join(f"{mod.mean(1)[k] / act[k]:5.1f}" if act[k] > 1e-3 else "    -" for k in ks)
            print(f"{sc:7s} {name:14s} | {f(ec.mean(1)):>17s} | {f(act):>17s} | {f(mod.mean(1)):>18s} | "
                  f"{f(spread.mean(1)):>17s} | {ratio:>17s}")
    print("\nPooled over scenes (mean of per-scene values):")
    for name in PATTERNS:
        a = {key: np.mean(pooled[(name, key)], 0) for key in ("ec", "act", "mod", "spr", "modG", "in_cv", "in_g")}
        f = lambda v: " ".join(f"{v[k]:5.2f}" for k in ks)
        ratio = " ".join(f"{a['mod'][k] / a['act'][k]:5.1f}" if a["act"][k] > 1e-3 else "    -" for k in ks)
        share = " ".join(f"{a['act'][k] / (a['ec'][k] + a['act'][k]):5.0%}" for k in ks)
        ratio_g = " ".join(f"{a['modG'][k] / a['act'][k]:5.1f}" if a["act"][k] > 1e-3 else "    -" for k in ks)
        print(f"  {name:14s} actual reducible {f(a['act'])} | CV model {f(a['mod'])} (x{ratio}) | "
              f"predictor-sensitivity model {f(a['modG'])} (x{ratio_g}) | reducible share {share}")
        print(f"  {'':14s} total covariance, true position inside 95%: CV "
              + " ".join(f"{100 * a['in_cv'][k]:4.0f}%" for k in ks) + " | predictor-sensitivity "
              + " ".join(f"{100 * a['in_g'][k]:4.0f}%" for k in ks))


if __name__ == "__main__":
    main()
