"""Coasting a pedestrian who left the camera's view: constant velocity vs the network's own prediction.

On each scene's held-out half, a pedestrian is detected (noisy, Kalman-filtered) for 8 - m steps and then goes
dark for the last m steps. From the current time, the next 8 steps (3.2 s) are predicted in two ways:

  A  constant-velocity coasting (current tracker): predict-only Kalman steps, the coasted positions appended
     to the history, the network run on that history; uncertainty = network spread + J_k P_now J_k^T
  B  network coasting: the network run on the last real history (8 - m points) for m + 8 steps, the last 8 kept;
     uncertainty = network spread at those steps + J_(m+k) P_det J_(m+k)^T (P at the last detection)

Reference: fully observed (detected every step). Reported per dark length m: error at 3.2 s, average error over
the horizon (ADE), and the share of true positions inside the total 95% ellipse at 3.2 s.

    python check_coasting.py --prefix loorot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from check_reducible import tracked_history, windows  # noqa: E402
from robot_nav.prediction import TrajectoryPredictor  # noqa: E402
from robot_nav.tracking import TrackerConfig, cwna  # noqa: E402
from social_lstm.engine import predict_sequence_with_gaussian  # noqa: E402

H, OBS = 8, 8


def rollout(model, hist, steps):
    with torch.no_grad():
        mu, sx, sy, rho = predict_sequence_with_gaussian(model, torch.from_numpy(hist), hist.shape[0], steps)
    sx, sy, rho = sx.numpy(), sy.numpy(), rho.numpy()
    cov = np.stack([np.stack([sx ** 2, rho * sx * sy], -1), np.stack([rho * sx * sy, sy ** 2], -1)], -2)
    return mu.numpy(), cov


def cv_cov(P, k):
    J = np.hstack([np.eye(2), k * np.eye(2)])
    return J @ P @ J.T


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="loorot")
    p.add_argument("--scenes", nargs="+", default=["zara01", "zara02", "eth", "hotel"])
    p.add_argument("--meas-std", type=float, default=0.12)
    p.add_argument("--max-windows", type=int, default=600)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    tcfg = TrackerConfig()
    F, Q = cwna(1.0, tcfg.qc)
    print(f"{args.prefix} predictors, measurement std {args.meas_std} m; error in m; 'in95' = true position inside "
          f"the total 95% ellipse at 3.2 s")
    print(f"{'scene':7s} {'dark':>5s} | {'observed: err3.2 ADE in95':>27s} | {'A CV coast: err3.2 ADE in95':>29s} | "
          f"{'B network coast: err3.2 ADE in95':>34s}")
    pooled = {}
    for sc in args.scenes:
        model = TrajectoryPredictor(f"checkpoints/{args.prefix}_{sc}.pt", H).model
        wins = windows(sc, OBS)
        if len(wins) > args.max_windows:
            wins = [wins[i] for i in rng.choice(len(wins), args.max_windows, replace=False)]
        for m in (0, 1, 2, 4, 6):
            res = {s: dict(e8=[], ade=[], inn=[]) for s in ("obs", "A", "B")}
            for w in wins:
                n = w.shape[1]
                truth = w[OBS:]  # (H, n, 2)
                det = np.r_[np.ones(OBS - m, bool), np.zeros(m, bool)]
                histA = np.zeros((OBS, n, 2), np.float32)
                PA, Pdet = [], []
                for j in range(n):
                    histA[:, j], P = tracked_history(w[:OBS, j], det, args.meas_std, rng, tcfg, F, Q)
                    PA.append(P)
                    # covariance at the last detection = the same filter run on the detected part only
                    _, Pd = tracked_history(w[:OBS - m, j], det[:OBS - m], args.meas_std, rng, tcfg, F, Q)
                    Pdet.append(Pd)
                # A: CV coasting (m = 0 is the fully observed reference)
                muA, covA = rollout(model, histA, H)
                totA = covA + np.stack([np.stack([cv_cov(PA[j], k + 1) for j in range(n)]) for k in range(H)])
                key = "obs" if m == 0 else "A"
                for s, mu, tot in ((key, muA, totA),):
                    err = truth - mu
                    d = np.linalg.norm(err, axis=-1)
                    d2 = np.einsum("kni,knij,knj->kn", err, np.linalg.inv(tot), err)
                    res[s]["e8"].append(d[-1]); res[s]["ade"].append(d.mean(0)); res[s]["inn"].append(d2[-1] < 5.991)
                if m == 0:
                    continue
                # B: network coasting from the last real history (same noisy filtered points up to the detection)
                histB = histA[: OBS - m]
                muB, covB = rollout(model, histB, m + H)
                muB, covB = muB[m:], covB[m:]
                totB = covB + np.stack([np.stack([cv_cov(Pdet[j], m + k + 1) for j in range(n)]) for k in range(H)])
                err = truth - muB
                d = np.linalg.norm(err, axis=-1)
                d2 = np.einsum("kni,knij,knj->kn", err, np.linalg.inv(totB), err)
                res["B"]["e8"].append(d[-1]); res["B"]["ade"].append(d.mean(0)); res["B"]["inn"].append(d2[-1] < 5.991)
            cell = {}
            for s in res:
                if res[s]["e8"]:
                    e8, ade, inn = (np.concatenate(res[s][x]) for x in ("e8", "ade", "inn"))
                    cell[s] = (e8.mean(), ade.mean(), inn.mean())
                    pooled.setdefault((m, s), []).append(cell[s])
            f = lambda t: f"{t[0]:6.2f} {t[1]:5.2f} {100 * t[2]:5.0f}%" if t else ""
            if m == 0:
                print(f"{sc:7s} {m:5d} | {f(cell['obs']):>27s} |")
            else:
                print(f"{sc:7s} {m:5d} | {'':>27s} | {f(cell['A']):>29s} | {f(cell['B']):>34s}")
    print("\nPooled over scenes (mean of per-scene values): error at 3.2 s / ADE / inside 95%")
    for m in (0, 1, 2, 4, 6):
        for s in (("obs",) if m == 0 else ("A", "B")):
            v = np.mean(pooled[(m, s)], 0)
            name = {"obs": "fully observed", "A": "A constant-velocity coasting", "B": "B network coasting"}[s]
            print(f"  dark {m} steps ({0.4 * m:.1f} s)  {name:30s} {v[0]:.2f} m / {v[1]:.2f} m / {100 * v[2]:.0f}%")


if __name__ == "__main__":
    main()
