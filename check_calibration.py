"""Calibration of the leave-one-scene-out predictors on each scene's held-out half.

For every window (OBS observed -> 8 predicted steps) the squared Mahalanobis distance d2 of the true position
under the predicted Gaussian is computed. A calibrated 2D Gaussian puts 39.3% / 86.5% / 98.9% of points inside
1 / 2 / 3 sigma (d2 <= 1, 4, 9). Reported: observed coverage minus that ideal (dESV; > 0 = too wide /
conservative, < 0 = too narrow / overconfident), and the mean negative log-likelihood.

    python check_calibration.py loo loorot
"""
import os
import sys

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from evaluate_nav import SCENES, scene_frames  # noqa: E402
from robot_nav.prediction import TrajectoryPredictor  # noqa: E402
from social_lstm.dataset import load_obsmat  # noqa: E402
from social_lstm.engine import predict_sequence_with_gaussian  # noqa: E402

H = 8
IDEAL = {1: 1 - np.exp(-0.5), 2: 1 - np.exp(-2), 3: 1 - np.exp(-4.5)}


def windows(sc, obs):
    f2p = load_obsmat(SCENES[sc]["path"])
    _, step, last, split = scene_frames(sc)
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


def stats(model, wins, obs):
    d2s, nlls = [], []
    for tr in wins:
        with torch.no_grad():
            mu, sx, sy, rho = predict_sequence_with_gaussian(model, torch.from_numpy(tr[:obs]), obs, H)
        mu, sx, sy, rho = mu.numpy(), sx.numpy(), sy.numpy(), rho.numpy()
        e = tr[obs:] - mu
        cxy = rho * sx * sy
        det = sx ** 2 * sy ** 2 - cxy ** 2
        d2 = (sy ** 2 * e[..., 0] ** 2 - 2 * cxy * e[..., 0] * e[..., 1] + sx ** 2 * e[..., 1] ** 2) / det
        d2s.append(d2)
        nlls.append(np.log(2 * np.pi) + 0.5 * np.log(det) + 0.5 * d2)
    return np.concatenate(d2s, 1), np.concatenate(nlls, 1)  # (H, n)


def main():
    sets = sys.argv[1:] or ["loo", "loorot"]
    for obs in (8, 3):
        print(f"\n=== {obs} observed steps; dESV = observed coverage - ideal (pp); NLL mean")
        print(f"{'scene':7s} {'set':7s} {'n':>6s} | {'0.4 s: 1s 2s 3s':>18s} | {'1.6 s: 1s 2s 3s':>18s} | "
              f"{'3.2 s: 1s 2s 3s':>18s} | {'NLL 0.4/1.6/3.2':>16s}")
        for sc in SCENES:
            wins = windows(sc, obs)
            for pre in sets:
                model = TrajectoryPredictor(f"checkpoints/{pre}_{sc}.pt", H).model
                d2, nll = stats(model, wins, obs)
                cells = []
                for k in (0, 3, 7):
                    cells.append(" ".join(f"{100 * (np.mean(d2[k] <= s * s) - IDEAL[s]):+5.0f}" for s in (1, 2, 3)))
                print(f"{sc:7s} {pre:7s} {d2.shape[1]:6d} | {cells[0]:>18s} | {cells[1]:>18s} | {cells[2]:>18s} | "
                      f"{nll[0].mean():5.2f}/{nll[3].mean():5.2f}/{nll[7].mean():5.2f}")


if __name__ == "__main__":
    main()
