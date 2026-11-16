"""How accurate is the Poisson (undetected-pedestrian) part of the PMB tracker?

Runs the final setup (60 deg / 4 m camera, PMB pruned at 1 m, rotation-augmented leave-one-scene-out predictors,
clean protocol: prior from the first half of each recording, episodes from the second half) on the seed-61
episodes and, at every step, compares the tracker's undetected intensity with the ground truth. The tracker and
planner are not changed; this only observes.

Truth = pedestrians present with no Bernoulli track (never detected, or pruned back into the intensity).
  NOW: expected number lambda_hat * D(A) vs the true count, for A = whole scene, the camera's view cone, and the
       area within 4 m of the robot outside the view.
  FORECAST (what the planner's unseen-pedestrian constraint uses): for horizon k = 1..8, the intensity predicted
       k steps ahead without measurements, in the collision disk (radius 0.6 m) around the planned position k,
       vs the true number of pedestrians in that disk at step t+k that had no track at step t.
       Probability of >= 1 such pedestrian: negative binomial with the online crowd multiplier (as the planner),
       plain Poisson (lambda = 1), and the static prior (no sensing updates).

    LOO_PREFIX=loorot python check_ppp_accuracy.py --per-scene 60 --planners supermpc-active supermpc
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
from scipy.special import gammaln

from evaluate_nav import SCENES, checkpoint_for, make_episodes, scene_frames

SCENE_NAMES = ["zara01", "zara02", "eth", "hotel"]
H = 8
R_DISK = 0.6
NEAR = 4.0
BINS = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-scene", type=int, default=60)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--planners", nargs="+", default=["supermpc-active", "supermpc"])
    p.add_argument("--workers", type=int, default=13)
    p.add_argument("--out", default="results/ppp_accuracy")
    return p.parse_args()


def nb_logpmf(n, M, a, b):
    """log P(N = n) for N | lambda ~ Poisson(lambda M), lambda ~ Gamma(a, b)."""
    M = max(M, 1e-12)
    return gammaln(a + n) - gammaln(a) - gammaln(n + 1) + a * np.log(b / (b + M)) + n * np.log(M / (b + M))


def pois_logpmf(n, mu):
    mu = max(mu, 1e-12)
    return n * np.log(mu) - mu - gammaln(n + 1)


_W = {}


def run(job):
    from robot_nav.mpc import MPCConfig
    from robot_nav.pmb import channel_mixing, fit_scene_prior, ppp_predict
    from robot_nav.prediction import TrajectoryPredictor
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    ep, planner, seed0 = job
    sc = ep["scene"]
    if sc not in _W:
        first, _, _, split = scene_frames(sc)
        _W[sc] = (fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split)),
                  TrajectoryPredictor(checkpoint_for(sc, "clean", None), MPCConfig().N))
    prior, predictor = _W[sc]
    scene = PedestrianScene(SCENES[sc]["path"], ep["frame"])
    centers = prior.grid.centers
    mix = channel_mixing(0.02)
    rec = dict(now=[], fc=[])

    def hook(step, x, tracker, info, peds, **_):
        tracked = set(tracker.tracks)
        lam = tracker.rate_mean
        a, b = tracker.rate_a, tracker.rate_b
        Dsum = tracker.D.sum(0)
        untracked_now = [q for pid, q in peds.items() if pid not in tracked]
        # NOW: whole scene, inside the view cone (p_D > 0), near the robot but outside the view
        inview = tracker.last_pD > 0
        near = np.linalg.norm(centers - x[:2], axis=-1) <= NEAR
        def count(mask):
            g = prior.grid
            n = 0
            for q in untracked_now:
                ix, iy = int((q[0] - g.x0) / g.res), int((q[1] - g.y0) / g.res)
                n += 0 <= ix < g.nx and 0 <= iy < g.ny and bool(mask[ix, iy])
            return n
        full = np.ones_like(inview)
        rec["now"].append([lam * Dsum.sum(), count(full), lam * Dsum[inview].sum(), count(inview),
                           lam * Dsum[near & ~inview].sum(), count(near & ~inview),
                           prior.occupancy.sum(0)[near & ~inview].sum()])
        # FORECAST along the planned path
        X = np.asarray(info["X"])
        D, P0 = tracker.D.copy(), prior.occupancy.copy()
        for k in range(1, H + 1):
            D = ppp_predict(D, prior, mix)
            P0 = ppp_predict(P0, prior, mix)
            c = X[:2, k]
            disk = np.linalg.norm(centers - c, axis=-1) <= R_DISK
            M = float(D.sum(0)[disk].sum())
            M0 = float(P0.sum(0)[disk].sum())
            fut = scene.positions(step + k)
            n = sum(1 for pid, q in fut.items() if pid not in tracked and np.linalg.norm(q - c) <= R_DISK)
            rec["fc"].append([k, M, M0, a, b, n])

    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=60.0, cam_range=4.0,
                           mpc_overrides={"unseen_joint": False}, pmb_overrides={"prune_pos_std": 1.0},
                           robot_overrides=None)
    seed = seed0 + 1000 * list(SCENES).index(sc) + ep["episode"]
    run_episode(planner, scene, ep["start"], ep["goal"], predictor, args, seed, prior, hook=hook)
    return planner, sc, rec


def summarize(planner, recs, out):
    now = np.array([r for rec in recs for r in rec["now"]])
    fc = np.array([r for rec in recs for r in rec["fc"]])
    lines = [f"=== {planner}: {len(recs)} episodes, {len(now)} steps"]
    lines.append("NOW (expected vs true number of untracked pedestrians, mean per step):")
    for name, i in (("whole scene", 0), ("inside the view cone", 2), (f"within {NEAR:.0f} m, outside the view", 4)):
        e, t = now[:, i].mean(), now[:, i + 1].mean()
        lines.append(f"  {name:28s} expected {e:6.3f}  true {t:6.3f}  ratio {e / max(t, 1e-9):5.2f}")
    lines.append(f"  (static prior, within {NEAR:.0f} m outside view: expected {now[:, 6].mean():.3f})")
    lines.append("FORECAST in the 0.6 m collision disk at planned position k (no future measurements):")
    lines.append(f"  {'k':>2s} {'n':>6s} {'E[N] NB':>8s} {'true N':>7s} {'P>=1 NB':>8s} {'P>=1 Poi':>8s} {'P>=1 prior':>10s} "
                 f"{'freq':>6s} {'Brier NB':>9s} {'Poi':>7s} {'prior':>7s} {'NLL NB':>7s} {'Poi':>7s} {'prior':>7s}")
    all_p, all_y = [], []
    for k in range(1, H + 1):
        s = fc[fc[:, 0] == k]
        M, M0, a, b, n = s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
        lam = a / b
        p_nb = 1 - (b / (b + M)) ** a
        p_po = 1 - np.exp(-M)
        p_pr = 1 - np.exp(-M0)
        y = (n > 0).astype(float)
        all_p.append(p_nb), all_y.append(y)
        nll_nb = -np.mean([nb_logpmf(ni, Mi, ai, bi) for ni, Mi, ai, bi in zip(n, M, a, b)])
        nll_po = -np.mean([pois_logpmf(ni, Mi) for ni, Mi in zip(n, M)])
        nll_pr = -np.mean([pois_logpmf(ni, Mi) for ni, Mi in zip(n, M0)])
        lines.append(f"  {k:2d} {len(s):6d} {np.mean(lam * M):8.4f} {n.mean():7.4f} {p_nb.mean():8.4f} {p_po.mean():8.4f} "
                     f"{p_pr.mean():10.4f} {y.mean():6.4f} {np.mean((p_nb - y) ** 2):9.5f} {np.mean((p_po - y) ** 2):7.5f} "
                     f"{np.mean((p_pr - y) ** 2):7.5f} {nll_nb:7.4f} {nll_po:7.4f} {nll_pr:7.4f}")
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    lines.append("RELIABILITY of P(>=1 untracked pedestrian in the disk), negative binomial, all horizons:")
    lines.append(f"  {'predicted bin':>16s} {'n':>7s} {'mean predicted':>15s} {'observed freq':>14s}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            lines.append(f"  [{lo:5.3f}, {min(hi, 1):5.3f}) {int(m.sum()):7d} {p[m].mean():15.4f} {y[m].mean():14.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, f"summary_{planner}.txt"), "w") as f:
        f.write(txt + "\n")


def main(args):
    os.makedirs(args.out, exist_ok=True)
    jobs = []
    for sc in SCENE_NAMES:
        for ep in make_episodes(sc, 450, args.seed, protocol="clean")[: args.per_scene]:
            for pl in args.planners:
                jobs.append((ep, pl, args.seed))
    by = {pl: [] for pl in args.planners}
    with Pool(args.workers) as pool:
        for i, (pl, sc, rec) in enumerate(pool.imap_unordered(run, jobs)):
            by[pl].append(rec)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes", flush=True)
    for pl, recs in by.items():
        summarize(pl, recs, args.out)


if __name__ == "__main__":
    main(parse_args())
