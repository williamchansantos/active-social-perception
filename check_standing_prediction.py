"""Check 1: does the Social LSTM over-inflate uncertainty for standing pedestrians?

On the held-out half (clean protocol: frames after the split, leave-one-scene-out
checkpoint), every pedestrian with 8 observed and 8 future positions is predicted
jointly with its neighbours. Windows are grouped by observed speed over the last
two observed steps (standing < 0.2 m/s, slow 0.2-0.6, walking > 0.6). Per group and
horizon: predicted spread sqrt(sx*sy), actual error, and the fraction of true
positions inside the predicted 95% ellipse (0.95 = calibrated, more = too wide).

    python check_standing_prediction.py --scenes zara01 zara02
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from evaluate_nav import SCENES, checkpoint_for, scene_frames
from robot_nav.prediction import TrajectoryPredictor
from social_lstm.dataset import load_obsmat
from social_lstm.engine import predict_sequence_with_gaussian

OBS, H = 8, 8
CHI2_95 = 5.991
GROUPS = [("standing", 0.0, 0.2), ("slow", 0.2, 0.6), ("walking", 0.6, 99.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["zara01", "zara02"])
    args = p.parse_args()
    for sc in args.scenes:
        model = TrajectoryPredictor(checkpoint_for(sc, "clean", None), H).model
        f2p = load_obsmat(SCENES[sc]["path"])
        _, step, last, split = scene_frames(sc)
        res = {g[0]: dict(sd=[], err=[], d2=[], cv_err=[]) for g in GROUPS}
        for f0 in range(split, last - (OBS + H) * step, step):
            frames = [f0 + i * step for i in range(OBS + H)]
            if any(f not in f2p for f in frames):
                continue
            ids = [pid for pid in f2p[frames[0]] if all(pid in f2p[f] for f in frames)]
            if not ids:
                continue
            traj = np.array([[f2p[f][pid] for pid in ids] for f in frames], dtype=np.float32)  # (16, M, 2)
            with torch.no_grad():
                mu, sx, sy, rho = predict_sequence_with_gaussian(model, torch.from_numpy(traj[:OBS]), OBS, H)
            mu, sx, sy, rho = mu.numpy(), sx.numpy(), sy.numpy(), rho.numpy()
            speed = np.linalg.norm(traj[OBS - 1] - traj[OBS - 3], axis=-1) / (2 * 0.4)
            cv = traj[OBS - 1] + (traj[OBS - 1] - traj[OBS - 2])[None] * np.arange(1, H + 1)[:, None, None]
            for j in range(len(ids)):
                g = next(n for n, lo, hi in GROUPS if lo <= speed[j] < hi)
                e = traj[OBS:, j] - mu[:, j]
                cxy = rho[:, j] * sx[:, j] * sy[:, j]
                det = sx[:, j] ** 2 * sy[:, j] ** 2 - cxy ** 2
                d2 = (sy[:, j] ** 2 * e[:, 0] ** 2 - 2 * cxy * e[:, 0] * e[:, 1] + sx[:, j] ** 2 * e[:, 1] ** 2) / det
                res[g]["sd"].append(np.sqrt(sx[:, j] * sy[:, j]))
                res[g]["err"].append(np.linalg.norm(e, axis=-1))
                res[g]["d2"].append(d2)
                res[g]["cv_err"].append(np.linalg.norm(traj[OBS:, j] - cv[:, j], axis=-1))
        print(f"\n== {sc} (held-out half, checkpoint {checkpoint_for(sc, 'clean', None)})")
        print(f"{'group':9s} {'n':>5s} | {'spread 1.2s':>11s} {'3.2s':>6s} | {'err 1.2s':>8s} {'3.2s':>6s} | "
              f"{'in95 1.2s':>9s} {'3.2s':>6s} | {'CV err 3.2s':>11s}")
        for g, _, _ in GROUPS:
            r = {k: np.array(v) for k, v in res[g].items()}
            if not len(r["sd"]):
                continue
            in95 = (r["d2"] < CHI2_95).mean(0)
            print(f"{g:9s} {len(r['sd']):5d} | {r['sd'][:, 2].mean():11.2f} {r['sd'][:, 7].mean():6.2f} | "
                  f"{np.median(r['err'][:, 2]):8.2f} {np.median(r['err'][:, 7]):6.2f} | "
                  f"{in95[2]:9.2f} {in95[7]:6.2f} | {np.median(r['cv_err'][:, 7]):11.2f}")


if __name__ == "__main__":
    main()
