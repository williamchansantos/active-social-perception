"""Calibrate the Poisson (never-detected pedestrian) part of the PMB tracker on data the evaluation never touches.

Data separation
  calibration: episodes in the SECOND QUARTER of each recording, with the scene prior fitted on the FIRST QUARTER
               (as in deployment: the prior comes from earlier data than the episodes)
  test:        the seed-61 episodes (second half of each recording, prior fitted on the first half), as in the paper
Each episode is run once with the final planner and the uncalibrated tracker; its detections, robot poses, plans and
the ground truth are recorded, and the tracker is then replayed offline with other parameters (the robot's path is
kept, so only the tracker's beliefs change).

Parameters (robot_nav.pmb.PMBConfig)
  p_detect_model  detection probability the filter assumes for a pedestrian in view. MEASURED, not fitted: the
                  fraction of untracked pedestrians inside the camera cone (4 m, 60 deg) that the camera detects in
                  that step, on the calibration episodes.
  relax           per-step relaxation of the undetected intensity toward the prior (refill by people who do not
                  follow the learned flows). FITTED by the negative-binomial likelihood of the untracked counts (near
                  the robot outside the view, and in the planned collision disks 1-8 steps ahead) on the calibration
                  episodes, with p_detect_model fixed. Never fitted to navigation success.

    LOO_PREFIX=loorot python calibrate_ppp.py record      # runs the episodes (about 30 min)
    LOO_PREFIX=loorot python calibrate_ppp.py fit         # measurement, fit, test-set evaluation (minutes)
"""
from __future__ import annotations

import os
import pickle
import sys
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
from scipy.special import gammaln

from evaluate_nav import SCENES, checkpoint_for, make_episodes, scene_frames

OUT = "results/ppp_calibration"
SCENE_NAMES = ["zara01", "zara02", "eth", "hotel"]
N_TRAIN, N_TEST, H, R_DISK, NEAR = 40, 60, 8, 0.6, 4.0
FOV, RANGE = 60.0, 4.0
RELAX_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
BINS = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]


def quarters(sc):
    first, step, _, split = scene_frames(sc)
    q1 = first + step * round((split - first) / 2 / step)
    return first, step, q1, split


def train_episodes(sc, n, seed=101, max_steps=90):
    """Episodes as make_episodes builds them, but starting in the second quarter of the recording."""
    from robot_nav.scene import PedestrianScene
    s = SCENES[sc]
    first, step, q1, split = quarters(sc)
    a, o = s["axis"], 1 - s["axis"]
    rng = np.random.default_rng([seed, SCENE_NAMES.index(sc)])
    eps = []
    for i in range(n):
        frame = int(rng.integers(q1, split - max_steps * step))
        c = rng.uniform(*s["line"])
        start, goal = np.zeros(2), np.zeros(2)
        start[a], start[o], goal[a], goal[o] = s["span"][0], c, s["span"][1], c
        if rng.uniform() < 0.5:
            start, goal = goal, start
        eps.append(dict(scene=sc, episode=i, frame=PedestrianScene(s["path"], frame).start_frame, start=start, goal=goal))
    return eps


def prior_for(sc, split_name):
    from robot_nav.pmb import fit_scene_prior
    first, step, q1, split = quarters(sc)
    rng_ = (first, q1) if split_name == "train" else (first, split)
    return fit_scene_prior(SCENES[sc]["path"], frame_range=rng_)


_W = {}


def record_one(job):
    from robot_nav.mpc import MPCConfig
    from robot_nav.prediction import TrajectoryPredictor
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    ep, split_name, seed0 = job
    sc = ep["scene"]
    key = (sc, split_name)
    if key not in _W:
        _W[key] = (prior_for(sc, split_name), TrajectoryPredictor(checkpoint_for(sc, "clean", None), MPCConfig().N))
    prior, predictor = _W[key]
    scene = PedestrianScene(SCENES[sc]["path"], ep["frame"])
    log = []

    def hook(step, x, x_est, det, peds, info, tracker, **_):
        log.append(dict(x=x.copy(), x_est=np.asarray(x_est).copy(), det={p: (np.asarray(z).copy(), np.asarray(R).copy())
                                                                          for p, (z, R) in det.items()},
                        peds={p: q.copy() for p, q in peds.items()}, X=np.asarray(info["X"])[:2].copy()))

    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=FOV, cam_range=RANGE,
                           mpc_overrides={"unseen_joint": False}, pmb_overrides={"prune_pos_std": 1.0}, robot_overrides=None)
    run_episode("supermpc-active", scene, ep["start"], ep["goal"], predictor, args,
                seed0 + 1000 * list(SCENES).index(sc) + ep["episode"], prior, hook=hook)
    # future positions beyond the episode's end, for the forecast truth
    n = len(log)
    future = [{p: q.copy() for p, q in scene.positions(n + k).items()} for k in range(H + 1)]
    return dict(scene=sc, split=split_name, log=log, future=future)


def record():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(ep, "train", 101) for sc in SCENE_NAMES for ep in train_episodes(sc, N_TRAIN)]
    jobs += [(ep, "test", 61) for sc in SCENE_NAMES for ep in make_episodes(sc, 450, 61, protocol="clean")[:N_TEST]]
    out = {"train": [], "test": []}
    with Pool(13) as pool:
        for i, rec in enumerate(pool.imap_unordered(record_one, jobs)):
            out[rec["split"]].append(rec)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    with open(f"{OUT}/logs.pkl", "wb") as f:
        pickle.dump(out, f)
    print({k: len(v) for k, v in out.items()})


# ------------------------------------------------------------------------------------------- measurement and replay
def in_cone(x, q):
    d = q - x[:2]
    r = np.linalg.norm(d)
    h = x[2] + x[3]
    return r <= RANGE and r > 1e-6 and np.dot(d / r, [np.cos(h), np.sin(h)]) >= np.cos(np.radians(FOV / 2)), r


def measure_detection(recs):
    """Detection rate of untracked pedestrians inside the camera cone, by range; and what the filter assumed."""
    from robot_nav.camera import CameraConfig
    from robot_nav.pmb import PMBConfig, PMBTracker
    bins = [0, 1, 2, 3, 4.0001]
    hit, n, assumed = np.zeros(4), np.zeros(4), np.zeros(4)
    for rec in recs:
        tr = PMBTracker(_prior(rec), CameraConfig(fov_deg=FOV, max_range=RANGE), PMBConfig(prune_pos_std=1.0))
        prev = set()
        for s in rec["log"]:
            occl = np.array([z for z, _ in s["det"].values()]).reshape(-1, 2)
            for pid, q in s["peds"].items():
                inside, r = in_cone(s["x"], q)
                if pid in prev or not inside:
                    continue
                b = int(np.searchsorted(bins, r, side="right") - 1)
                n[b] += 1
                hit[b] += pid in s["det"]
                assumed[b] += float(tr.detection_prob(q, s["x_est"], occl))
            tr.step(s["det"], s["x_est"])
            prev = set(tr.tracks)
    return bins, hit, n, assumed


_PRIORS = {}


def _prior(rec):
    key = (rec["scene"], rec["split"])
    if key not in _PRIORS:
        _PRIORS[key] = prior_for(*key)
    return _PRIORS[key]


def nb_logpmf(n, M, a, b):
    M = max(M, 1e-12)
    return gammaln(a + n) - gammaln(a) - gammaln(n + 1) + a * np.log(b / (b + M)) + n * np.log(M / (b + M))


def replay(recs, p_detect_model=None, relax=0.0, prior_fn=None, **pmb_kw):
    """Replay the tracker over recorded episodes; return per-step 'now' records and per-disk forecast records."""
    from robot_nav.camera import CameraConfig
    from robot_nav.pmb import PMBConfig, PMBTracker, channel_mixing, ppp_predict
    now, fc = [], []
    for rec in recs:
        prior = _prior(rec) if prior_fn is None else prior_fn(rec)
        cfg = PMBConfig(prune_pos_std=1.0, p_detect_model=p_detect_model, relax=relax, **pmb_kw)
        tr = PMBTracker(prior, CameraConfig(fov_deg=FOV, max_range=RANGE), cfg)
        g, centers, mix = prior.grid, prior.grid.centers, channel_mixing(0.02)
        log, fut = rec["log"], rec["future"]
        n_steps = len(log)
        for t, s in enumerate(log):
            tr.step(s["det"], s["x_est"])
            tracked = set(tr.tracks)
            Dsum = tr.D.sum(0)
            near_out = (np.linalg.norm(centers - s["x"][:2], axis=-1) <= NEAR) & ~(tr.last_pD > 0)
            cnt = 0
            for pid, q in s["peds"].items():
                if pid in tracked:
                    continue
                ix, iy = int((q[0] - g.x0) / g.res), int((q[1] - g.y0) / g.res)
                cnt += 0 <= ix < g.nx and 0 <= iy < g.ny and bool(near_out[ix, iy])
            now.append((float(Dsum[near_out].sum()), cnt, tr.rate_a, tr.rate_b))
            D = tr.D.copy()
            for k in range(1, H + 1):
                D = ppp_predict(D, prior, mix)
                if relax > 0:
                    D = (1 - relax) * D + relax * prior.occupancy
                c = s["X"][:, k]
                M = float(D.sum(0)[np.linalg.norm(centers - c, axis=-1) <= R_DISK].sum())
                peds_k = log[t + k]["peds"] if t + k < n_steps else fut[t + k - n_steps]
                n = sum(1 for pid, q in peds_k.items() if pid not in tracked and np.linalg.norm(q - c) <= R_DISK)
                fc.append((k, M, tr.rate_a, tr.rate_b, n))
    return np.array(now), np.array(fc)


def score(now, fc):
    nll_now = -np.mean([nb_logpmf(n, M, a, b) for M, n, a, b in now])
    nll_fc = -np.mean([nb_logpmf(n, M, a, b) for k, M, a, b, n in fc])
    return nll_now, nll_fc


def report(name, now, fc):
    lam = now[:, 2] / now[:, 3]
    lines = [f"--- {name}",
             f"  near the robot outside the view: expected {np.mean(lam * now[:, 0]):.3f} untracked, true "
             f"{now[:, 1].mean():.3f} (ratio {np.mean(lam * now[:, 0]) / now[:, 1].mean():.2f}); NLL {score(now, fc)[0]:.4f}"]
    p = 1 - (fc[:, 3] / (fc[:, 3] + fc[:, 1])) ** fc[:, 2]
    y = (fc[:, 4] > 0).astype(float)
    lines.append(f"  collision-disk forecast: mean predicted P(>=1) {p.mean():.4f} vs observed {y.mean():.4f} "
                 f"(ratio {p.mean() / y.mean():.2f}); Brier {np.mean((p - y) ** 2):.5f}; NLL {score(now, fc)[1]:.4f}")
    lines.append(f"  {'predicted bin':>16s} {'n':>7s} {'predicted':>10s} {'observed':>9s}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            lines.append(f"  [{lo:5.3f}, {min(hi, 1):5.3f}) {int(m.sum()):7d} {p[m].mean():10.4f} {y[m].mean():9.4f}")
    return "\n".join(lines)


def fit():
    with open(f"{OUT}/logs.pkl", "rb") as f:
        logs = pickle.load(f)
    train, test = logs["train"], logs["test"]
    out = [f"Calibration of the PMB Poisson part: {len(train)} calibration episodes (2nd quarter, prior from the 1st), "
           f"{len(test)} test episodes (seed 61)"]
    bins, hit, n, assumed = measure_detection(train)
    p_meas = float(hit.sum() / n.sum())
    out.append("\n1. Detection of untracked pedestrians inside the camera cone (calibration episodes):")
    for i in range(4):
        out.append(f"  {bins[i]:.0f}-{min(bins[i + 1], 4):.0f} m: detected {hit[i] / max(n[i], 1):.3f} of {int(n[i])}; "
                   f"the filter assumed {assumed[i] / max(n[i], 1):.3f}")
    out.append(f"  pooled: measured {p_meas:.3f}, assumed {assumed.sum() / n.sum():.3f}  ->  p_detect_model = {p_meas:.3f}")
    out.append("\n2. Refill rate, fitted by NB likelihood on the calibration episodes (p_detect_model fixed):")
    best = None
    for rx in RELAX_GRID:
        a, b = score(*replay(train, p_meas, rx))
        out.append(f"  relax {rx:5.3f}: NLL near/outside view {a:.4f}, disk forecast {b:.4f}, sum {a + b:.4f}")
        if best is None or a + b < best[1]:
            best = (rx, a + b)
    relax = best[0]
    out.append(f"  -> relax = {relax}")
    out.append("\n3. Test episodes (seed 61; nothing fitted on them):")
    for name, pd_, rx in (("uncalibrated (as in the paper)", None, 0.0), (f"detection only (p = {p_meas:.3f})", p_meas, 0.0),
                          (f"refill only (relax = {relax})", None, relax),
                          (f"calibrated (p = {p_meas:.3f}, relax = {relax})", p_meas, relax)):
        out.append(report(name, *replay(test, pd_, rx)))
    txt = "\n".join(out)
    print(txt)
    with open(f"{OUT}/summary.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    record() if sys.argv[1:] == ["record"] else fit()
