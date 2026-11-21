"""Does our birth map make the tracker's undetected intensity more accurate?

Repeats the lab's accuracy check (`check_ppp_accuracy.py`: robot drives with the real
planner and camera through replayed crowds; the undetected intensity is compared
with the pedestrians that really had no track) for several birth-map variants on
exactly the same trajectories.

How the variants share a trajectory: the robot is driven by the lab's tracker and
prior, unchanged. For every other variant a "shadow" tracker with that variant's
prior receives the same detections and robot poses at every tracker step, but
nothing reads it. So all variants see identical camera views, and any difference
is due to the prior alone. No lab file is modified; the shadow trackers are attached
in the worker processes by wrapping `PMBTracker.__init__` and `PMBTracker.step`.

Variants:
    lab            the lab's prior, unchanged (reference; drives the robot)
    kde            our birth map: per-scene bandwidth, floor, first-frame pedestrians
                   excluded, survival recalibrated
    kde-labsurv    our birth map, the lab's survival map left untouched
    kde-keepinit   our bandwidth and floor, but first-frame pedestrians kept as births
                   (same arrival rate as the lab: isolates bandwidth + floor)

Protocol (the lab's): prior from the first half of each recording, episodes from the
second half, scenes zara01 / zara02 / eth / hotel, 60 deg / 4 m camera.

Reported per variant:
    NOW       expected vs true number of untracked pedestrians per step, in the whole
              scene, in the view cone, and within 4 m outside the view (ratio 1 = unbiased)
    near NLL  Poisson negative log-likelihood of the true count within 4 m outside the
              view (lower is better)
    FORECAST  for the 0.6 m collision disk at the planned position 1..8 steps ahead:
              mean predicted P(>=1 untracked pedestrian), observed frequency, Brier
              score and negative log-likelihood (lower is better). This is what the
              planner's unseen-pedestrian constraint uses.
    vs lab    difference to the lab variant with a 95 % interval from resampling
              episodes (negative = better than the lab)

Caveat: this measures the accuracy of the intensity only. Whether the robot plans
differently (and navigates better) with our map is a separate experiment.

Needs scipy, casadi and torch (the lab's planner and predictor). Run from the repo root:
    python3 -m social_lstm.birth.check_tracker_accuracy --per-scene 60
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("LOO_PREFIX", "loorot")  # the lab's final predictors; read by evaluate_nav at import
os.environ.setdefault("OMP_NUM_THREADS", "1")

VARIANTS: Dict[str, dict] = {
    "lab": {},
    "kde": dict(),
    "kde-labsurv": dict(recalibrate=False),
    "kde-keepinit": dict(exclude_initial=False),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-scene", type=int, default=60, help="episodes per scene")
    p.add_argument("--seed", type=int, default=61, help="episode seed (lab's final runs: 61)")
    p.add_argument("--planner", default="supermpc-active", help="planner name as in check_ppp_accuracy.py")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--workers", type=int, default=13)
    p.add_argument("--bootstrap", type=int, default=2000, help="episode resamples for the intervals")
    p.add_argument("--out", default="plots/birth/tracker_accuracy")
    return p.parse_args()


H, R_DISK, NEAR = 8, 0.6, 4.0  # as in check_ppp_accuracy.py
_W: dict = {}


def _init() -> None:
    """Worker start-up: give every driving PMBTracker one shadow tracker per variant
    prior (found on `prior._shadows`), stepped with the same detections and poses."""
    from robot_nav.pmb import PMBTracker

    orig_init, orig_step = PMBTracker.__init__, PMBTracker.step

    def init(self, prior, cam, *a, **kw):
        orig_init(self, prior, cam, *a, **kw)
        self.shadows = {v: PMBTracker(p, cam, *a, **kw) for v, p in getattr(prior, "_shadows", {}).items()}

    def step(self, detections, x_est, *a, **kw):
        orig_step(self, detections, x_est, *a, **kw)
        for sh in getattr(self, "shadows", {}).values():
            orig_step(sh, detections, x_est, *a, **kw)

    PMBTracker.__init__, PMBTracker.step = init, step


def _run(job) -> Tuple[str, int, Dict[str, dict]]:
    """One episode; returns the per-step records of every variant (same layout as the
    lab's script: now = [E_all, n_all, E_cone, n_cone, E_near, n_near, static prior],
    fc = [k, M, M0, a, b, n])."""
    from types import SimpleNamespace

    from evaluate_nav import SCENES, checkpoint_for, scene_frames
    from robot_nav.mpc import MPCConfig
    from robot_nav.pmb import channel_mixing, fit_scene_prior, ppp_predict
    from robot_nav.prediction import TrajectoryPredictor
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode
    from social_lstm.birth.tracker_birth import with_kde_birth

    ep, planner, seed0, variants = job
    sc = ep["scene"]
    if sc not in _W:
        first, _, _, split = scene_frames(sc)
        prior = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
        shadows = {v: with_kde_birth(prior, SCENES[sc]["path"], frame_range=(first, split), **VARIANTS[v])
                   for v in variants if v != "lab"}
        prior._shadows = shadows  # set after the copies were made, so shadows have no shadows
        _W[sc] = (prior, TrajectoryPredictor(checkpoint_for(sc, "clean", None), MPCConfig().N))
    prior, predictor = _W[sc]
    scene = PedestrianScene(SCENES[sc]["path"], ep["frame"])
    g, centers, mix = prior.grid, prior.grid.centers, channel_mixing(0.02)
    recs = {v: dict(now=[], fc=[]) for v in variants}

    def hook(step, x, tracker, info, peds, **_):
        X = np.asarray(info["X"])
        near = np.linalg.norm(centers - x[:2], axis=-1) <= NEAR
        for v in variants:
            trk = tracker if v == "lab" else tracker.shadows[v]
            pr = trk.prior
            tracked, lam, a, b = set(trk.tracks), trk.rate_mean, trk.rate_a, trk.rate_b
            Dsum = trk.D.sum(0)
            untracked = [q for pid, q in peds.items() if pid not in tracked]
            inview = trk.last_pD > 0

            def count(mask):
                n = 0
                for q in untracked:
                    ix, iy = int((q[0] - g.x0) / g.res), int((q[1] - g.y0) / g.res)
                    n += 0 <= ix < g.nx and 0 <= iy < g.ny and bool(mask[ix, iy])
                return n

            out = near & ~inview
            recs[v]["now"].append([lam * Dsum.sum(), count(np.ones_like(inview)),
                                   lam * Dsum[inview].sum(), count(inview), lam * Dsum[out].sum(), count(out),
                                   pr.occupancy.sum(0)[out].sum()])
            D, P0 = trk.D.copy(), pr.occupancy.copy()
            for k in range(1, H + 1):
                D, P0 = ppp_predict(D, pr, mix), ppp_predict(P0, pr, mix)
                c = X[:2, k]
                disk = np.linalg.norm(centers - c, axis=-1) <= R_DISK
                fut = scene.positions(step + k)
                n = sum(1 for pid, q in fut.items() if pid not in tracked and np.linalg.norm(q - c) <= R_DISK)
                recs[v]["fc"].append([k, float(D.sum(0)[disk].sum()), float(P0.sum(0)[disk].sum()), a, b, n])

    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=60.0, cam_range=4.0,
                           mpc_overrides={"unseen_joint": False}, pmb_overrides={"prune_pos_std": 1.0},
                           robot_overrides=None)  # the lab's final setup, as in check_ppp_accuracy.py
    seed = seed0 + 1000 * list(SCENES).index(sc) + ep["episode"]
    run_episode(planner, scene, ep["start"], ep["goal"], predictor, args, seed, prior, hook=hook)
    return sc, ep["episode"], recs


def episode_stats(rec: dict) -> np.ndarray:
    """Per-episode sums: [steps, E_all, n_all, E_cone, n_cone, E_near, n_near, near_nll,
    forecasts, sum p, sum y, sum Brier, sum NLL]."""
    from check_ppp_accuracy import nb_logpmf, pois_logpmf

    now, fc = np.array(rec["now"]).reshape(-1, 7), np.array(rec["fc"]).reshape(-1, 6)
    near_nll = -sum(pois_logpmf(n, e) for e, n in zip(now[:, 4], now[:, 5]))
    M, a, b, n = fc[:, 1], fc[:, 3], fc[:, 4], fc[:, 5]
    p, y = 1 - (b / (b + M)) ** a, (n > 0).astype(float)
    nll = -sum(nb_logpmf(ni, Mi, ai, bi) for ni, Mi, ai, bi in zip(n, M, a, b))
    return np.array([len(now), *now[:, :6].sum(0), near_nll, len(fc), p.sum(), y.sum(), ((p - y) ** 2).sum(), nll])


def metrics(S: np.ndarray) -> Dict[str, float]:
    """Pooled metrics from stacked per-episode sums (rows = episodes)."""
    t = S.sum(0)
    return {
        "exp_all": t[1] / t[0], "true_all": t[2] / t[0], "exp_cone": t[3] / t[0], "true_cone": t[4] / t[0],
        "exp_near": t[5] / t[0], "true_near": t[6] / t[0], "near_nll": t[7] / t[0],
        "fc_pred": t[9] / t[8], "fc_freq": t[10] / t[8], "fc_brier": t[11] / t[8], "fc_nll": t[12] / t[8],
    }


def paired_interval(A: np.ndarray, B: np.ndarray, key: str, n_boot: int, rng) -> Tuple[float, float, float]:
    """metric(A) - metric(B) with a 95 % interval, resampling the shared episodes."""
    point = metrics(A)[key] - metrics(B)[key]
    idx = rng.integers(0, len(A), size=(n_boot, len(A)))
    d = np.array([metrics(A[i])[key] - metrics(B[i])[key] for i in idx])
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(stats: Dict[str, Dict[Tuple[str, int], np.ndarray]], n_boot: int, title: str, keys: List[tuple]) -> str:
    rng = np.random.default_rng(0)
    lines = [title]
    lines.append(f"  {'variant':<13s} {'all exp/true':>13s} {'cone exp/true':>14s} {'near exp/true':>14s} {'near NLL':>9s} "
                 f"{'P>=1 pred':>9s} {'freq':>7s} {'Brier':>8s} {'fc NLL':>8s}")
    for v, st in stats.items():
        m = metrics(np.array([st[k] for k in keys]))
        lines.append(f"  {v:<13s} {m['exp_all']:6.2f}/{m['true_all']:<6.2f} {m['exp_cone']:7.3f}/{m['true_cone']:<6.3f} "
                     f"{m['exp_near']:7.3f}/{m['true_near']:<6.3f} {m['near_nll']:9.4f} {m['fc_pred']:9.4f} "
                     f"{m['fc_freq']:7.4f} {m['fc_brier']:8.5f} {m['fc_nll']:8.5f}")
    if "lab" in stats:
        B = np.array([stats["lab"][k] for k in keys])
        for v, st in stats.items():
            if v == "lab":
                continue
            A = np.array([st[k] for k in keys])
            parts = []
            for key, name in (("near_nll", "near NLL"), ("fc_brier", "Brier"), ("fc_nll", "fc NLL")):
                d, lo, hi = paired_interval(A, B, key, n_boot, rng)
                parts.append(f"{name} {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
            lines.append(f"  {v:<13s} vs lab: " + "; ".join(parts))
    return "\n".join(lines)


def main():
    args = parse_args()
    from check_ppp_accuracy import SCENE_NAMES
    from evaluate_nav import make_episodes

    os.makedirs(args.out, exist_ok=True)
    variants = ["lab"] + [v for v in args.variants if v != "lab"]
    jobs = [(ep, args.planner, args.seed, variants) for sc in SCENE_NAMES
            for ep in make_episodes(sc, 450, args.seed, protocol="clean")[: args.per_scene]]
    stats: Dict[str, Dict[Tuple[str, int], np.ndarray]] = {v: {} for v in variants}
    with Pool(args.workers, initializer=_init) as pool:
        for i, (sc, e, recs) in enumerate(pool.imap_unordered(_run, jobs)):
            for v, rec in recs.items():
                stats[v][(sc, e)] = episode_stats(rec)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes", flush=True)
    for v in variants:
        np.savez(os.path.join(args.out, f"episode_stats_{v}.npz"),
                 keys=np.array([f"{sc}:{e}" for sc, e in stats[v]]), stats=np.array(list(stats[v].values())))

    keys = sorted(set.intersection(*(set(s) for s in stats.values())))
    out = [f"planner {args.planner}, {len(keys)} episodes ({args.per_scene} per scene), seed {args.seed}\n",
           report(stats, args.bootstrap, "ALL SCENES", keys)]
    for sc in SCENE_NAMES:
        out.append(report(stats, args.bootstrap, sc.upper(), [k for k in keys if k[0] == sc]))
    out.append("\nexp/true: expected vs true untracked pedestrians per step. NLL, Brier: lower is better."
               "\nvs lab: variant minus lab with a 95 % interval over episodes; negative = better than the lab.")
    txt = "\n\n".join(out[:-1]) + "\n" + out[-1]
    print("\n" + txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nSaved {args.out}/summary.txt")


if __name__ == "__main__":
    main()
