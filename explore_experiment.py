"""Option 2: crowd-density exploration term in the gaze cost (MPCConfig.w_explore), with the weak crowd-multiplier
prior of option 1 (PMBConfig.rate_prior_strength = 1). Closed loop (the gaze changes what is seen), so episodes are
re-run per variant.

Calibration episodes only: 2nd quarter of each recording, 20 per scene, under
  A. no shift:          prior fitted on the 1st quarter
  B. controlled shift:  prior fitted on the 1st quarter with half the pedestrians removed
Variants: current tracker (fitted strength, no exploration); weak prior; weak prior + w_explore in {10, 50, 200}.
Rule (fixed before running): lowest summed negative-binomial NLL of the untracked counts (near the robot outside the
view + collision disks 1-8 steps ahead) over A and B; a variant whose success (excl. spawn) is more than 3 points
below the weak-prior variant's is excluded. Navigation success and camera pan are reported for every variant.

    LOO_PREFIX=loorot python explore_experiment.py
"""
import os
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np

import calibrate_ppp as c
from evaluate_nav import SCENES, checkpoint_for

N_PER_SCENE = 20
VARIANTS = {"current": dict(mpc={}, pmb={}),
            "weak prior": dict(mpc={}, pmb={"rate_prior_strength": 1.0}),
            "weak + explore 10": dict(mpc={"w_explore": 10.0}, pmb={"rate_prior_strength": 1.0}),
            "weak + explore 50": dict(mpc={"w_explore": 50.0}, pmb={"rate_prior_strength": 1.0}),
            "weak + explore 200": dict(mpc={"w_explore": 200.0}, pmb={"rate_prior_strength": 1.0})}
_W = {}


def run(job):
    from robot_nav.mpc import MPCConfig
    from robot_nav.pmb import channel_mixing, ppp_predict
    from robot_nav.prediction import TrajectoryPredictor
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode
    from weak_prior_experiment import thinned_prior

    ep, cond, vname = job
    sc = ep["scene"]
    key = (sc, cond)
    if key not in _W:
        prior = c.prior_for(sc, "train") if cond == "A" else thinned_prior(dict(scene=sc))
        _W[key] = (prior, TrajectoryPredictor(checkpoint_for(sc, "clean", None), MPCConfig().N))
    prior, predictor = _W[key]
    scene = PedestrianScene(SCENES[sc]["path"], ep["frame"])
    g, centers, mix = prior.grid, prior.grid.centers, channel_mixing(0.02)
    now, fc, pans = [], [], []

    def hook(step, x, tracker, info, peds, **_):
        tracked = set(tracker.tracks)
        near_out = (np.linalg.norm(centers - x[:2], axis=-1) <= c.NEAR) & ~(tracker.last_pD > 0)
        cnt = 0
        for pid, q in peds.items():
            if pid not in tracked:
                ix, iy = int((q[0] - g.x0) / g.res), int((q[1] - g.y0) / g.res)
                cnt += 0 <= ix < g.nx and 0 <= iy < g.ny and bool(near_out[ix, iy])
        now.append((float(tracker.D.sum(0)[near_out].sum()), cnt, tracker.rate_a, tracker.rate_b))
        D = tracker.D.copy()
        X = np.asarray(info["X"])
        for k in range(1, c.H + 1):
            D = ppp_predict(D, prior, mix)
            ck = X[:2, k]
            M = float(D.sum(0)[np.linalg.norm(centers - ck, axis=-1) <= c.R_DISK].sum())
            n = sum(1 for pid, q in scene.positions(step + k).items() if pid not in tracked and np.linalg.norm(q - ck) <= c.R_DISK)
            fc.append((k, M, tracker.rate_a, tracker.rate_b, n))
        pans.append(abs(x[3]))

    v = VARIANTS[vname]
    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=0.02, fov=c.FOV, cam_range=c.RANGE,
                           mpc_overrides=dict({"unseen_joint": False}, **v["mpc"]),
                           pmb_overrides=dict({"prune_pos_std": 1.0}, **v["pmb"]), robot_overrides=None)
    res, _ = run_episode("supermpc-active", scene, ep["start"], ep["goal"], predictor, args,
                         101 + 1000 * list(SCENES).index(sc) + ep["episode"], prior, hook=hook)
    return cond, vname, dict(now=now, fc=fc, pan=pans, success=bool(res["success_nospawn"]))


def main():
    jobs = [(ep, cond, vn) for sc in c.SCENE_NAMES for ep in c.train_episodes(sc, N_PER_SCENE)
            for cond in ("A", "B") for vn in VARIANTS]
    from weak_prior_experiment import thinned_prior
    for sc in c.SCENE_NAMES:  # write the thinned-prior files before the workers start
        thinned_prior(dict(scene=sc))
    res = {}
    with Pool(13) as pool:
        for i, (cond, vn, r) in enumerate(pool.imap_unordered(run, jobs)):
            res.setdefault((cond, vn), []).append(r)
            if (i + 1) % 80 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    out = ["Crowd-density exploration (option 2), calibration episodes, closed loop "
           f"({N_PER_SCENE} per scene per condition)",
           f"{'variant':20s} | {'A: NLL':>7s} {'ratio':>6s} {'succ':>5s} {'pan':>5s} | {'B: NLL':>7s} {'ratio':>6s} {'succ':>5s} {'pan':>5s} | {'A+B NLL':>8s}"]
    summary = {}
    for vn in VARIANTS:
        row = []
        for cond in ("A", "B"):
            rs = res[(cond, vn)]
            now = np.array([t for r in rs for t in r["now"]])
            fc = np.array([t for r in rs for t in r["fc"]])
            a, b = c.score(now, fc)
            lam = now[:, 2] / now[:, 3]
            row.append(dict(nll=a + b, ratio=np.mean(lam * now[:, 0]) / now[:, 1].mean(),
                            succ=np.mean([r["success"] for r in rs]), pan=np.degrees(np.mean([p for r in rs for p in r["pan"]]))))
        summary[vn] = row
        A, B = row
        out.append(f"{vn:20s} | {A['nll']:7.4f} {A['ratio']:6.2f} {A['succ']:5.2f} {A['pan']:5.1f} | {B['nll']:7.4f} {B['ratio']:6.2f} "
                   f"{B['succ']:5.2f} {B['pan']:5.1f} | {A['nll'] + B['nll']:8.4f}")
    ref = np.mean([summary["weak prior"][i]["succ"] for i in range(2)])
    ok = [vn for vn in VARIANTS if np.mean([summary[vn][i]["succ"] for i in range(2)]) >= ref - 0.03]
    best = min(ok, key=lambda vn: summary[vn][0]["nll"] + summary[vn][1]["nll"])
    out.append(f"-> chosen (lowest A+B NLL among variants within 3 points of the weak prior's success): {best}")
    txt = "\n".join(out)
    print(txt)
    with open(f"{c.OUT}/explore_summary.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
