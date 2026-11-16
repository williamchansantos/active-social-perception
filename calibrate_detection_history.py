"""Calibrate prediction uncertainty against the number of detections in a track's history.

Reruns tuning episodes (default seed 11, never the held-out seed) with several camera
strategies, and for every tracked pedestrian within --radius of the robot records, per
horizon step k: the squared prediction error against the recorded future position, and
the traces of the LSTM and the CV-propagated state covariance. Grouped by n_det, the
number of the (up to 8) history points that came from detections.

Fits, per horizon step k, the ratio s(n, k) = E[err^2] / E[tr(Sigma_model + Sigma_state)]
with a smooth form s(n, k) = s_inf(k) + (s0(k) - s_inf(k)) * exp(-n / tau), saved to
--out (npz) for MPCConfig.history_calibration.

    python calibrate_detection_history.py --episodes 26 --out checkpoints/history_calibration.npz
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

CFG = {"supermpc-active": dict(unseen_joint=False, unseen_obs_aware=True),
       "supermpc-infogain": dict(unseen_joint=False),
       "supermpc-nearest": dict(unseen_joint=False)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=26, help="per Zara scene")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=13)
    p.add_argument("--out", default="checkpoints/history_calibration.npz")
    return p.parse_args()


def _job(args_tuple):
    i, seed, radius = args_tuple
    from evaluate_nav import SCENES, _init_worker, _predictor, make_episodes, scene_frames
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    sc = "zara02" if i % 2 else "zara01"
    _init_worker(f"checkpoints/loo_{sc}.pt")
    first, _, _, split = scene_frames(sc)
    prior = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
    ep = make_episodes(sc, 300, seed, "clean", 90)[i // 2]
    rows = []
    for pl, ov in CFG.items():
        recs = []
        nav = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=100, cam_range=4,
                              mpc_overrides=ov, pmb_overrides=dict(prune_pos_std=1.0))

        def hook(step, x, peds, tracks, pred, **_):
            for j, pid in enumerate(pred.ids):
                t = tracks[pid]
                if np.linalg.norm(t.x[:2] - x[:2]) > radius:
                    continue
                recs.append((step, pid, int(sum(getattr(t, "det_flags", [1]))), len(t.history), pred.mean[:, j].copy(),
                             np.trace(pred.cov_model[:, j], axis1=1, axis2=2),
                             np.trace(pred.cov_state[:, j], axis1=1, axis2=2)))

        scene = PedestrianScene(SCENES[sc]["path"], ep["frame"])
        run_episode(pl, scene, ep["start"], ep["goal"], _predictor(f"checkpoints/loo_{sc}.pt"), nav,
                    seed + 1000 * list(SCENES).index(sc) + ep["episode"], prior, hook=hook)
        for step, pid, nd, age, mean, tm, ts in recs:
            K = len(mean)
            err2 = np.full(K, np.nan)
            for k in range(K):
                fut = scene.recorded(step + k + 1).get(pid)
                if fut is not None:
                    err2[k] = float(np.sum((fut - mean[k]) ** 2))
            rows.append((nd, age, err2, tm, ts))
    return rows


def main():
    args = parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        out = list(pool.map(_job, [(i, args.seed, args.radius) for i in range(2 * args.episodes)]))
    rows = [r for o in out for r in o]
    nd = np.array([r[0] for r in rows])
    err2 = np.array([r[2] for r in rows])
    tm = np.array([r[3] for r in rows])
    ts = np.array([r[4] for r in rows])
    K = err2.shape[1]
    print(f"{len(rows)} track-steps; detections in history n_det: "
          + ", ".join(f"{n}: {np.sum(nd == n)}" for n in range(9)))
    ratio = np.full((9, K), np.nan)
    ratio_model = np.full((9, K), np.nan)
    print("\nE[err^2] / E[tr(Sigma_model + Sigma_state)]  (1 = calibrated; rows n_det, columns horizon step)")
    print("n_det " + " ".join(f"{(k + 1) * 0.4:5.1f}s" for k in range(K)) + "   | err(3.2 s) rms  model-only ratio 3.2 s")
    for n in range(9):
        m = nd == n
        if m.sum() < 30:
            continue
        for k in range(K):
            ok = m & ~np.isnan(err2[:, k])
            ratio[n, k] = np.mean(err2[ok, k]) / np.mean(tm[ok, k] + ts[ok, k])
            ratio_model[n, k] = np.mean(err2[ok, k]) / np.mean(tm[ok, k])
        ok = m & ~np.isnan(err2[:, -1])
        print(f"{n:5d} " + " ".join(f"{ratio[n, k]:6.2f}" for k in range(K))
              + f"   | {np.sqrt(np.mean(err2[ok, -1])):6.2f} m       {ratio_model[n, -1]:6.2f}")
    # Smooth fit per step: s(n) = s_inf + (s0 - s_inf) exp(-n / tau), tau shared across steps.
    ns = np.arange(9)
    valid = ~np.isnan(ratio[:, 0])
    best = None
    for tau in np.linspace(0.5, 6.0, 23):
        params, sse = [], 0.0
        for k in range(K):
            y = ratio[valid, k]
            A = np.stack([np.ones(valid.sum()), np.exp(-ns[valid] / tau)], 1)  # s_inf, (s0 - s_inf)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse += float(np.sum((A @ coef - y) ** 2))
            params.append(coef)
        if best is None or sse < best[0]:
            best = (sse, tau, np.array(params))
    _, tau, params = best
    s_inf, amp = params[:, 0], params[:, 1]
    print(f"\nfit: tau = {tau:.2f} detections; s_inf(k) = {np.round(s_inf, 2)}; s0(k) = {np.round(s_inf + amp, 2)}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, tau=tau, s_inf=s_inf, amp=amp, ratio=ratio, ratio_model=ratio_model, seed=args.seed)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
