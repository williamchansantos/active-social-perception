"""A robot crosses a replayed Zara crowd using SupeR-MPC, seeing pedestrians
only through a single pan camera with limited range and field of view.

Loop per 0.4 s step:
  ground-truth pedestrians -> camera detections (FOV, range, occlusion, noise,
  robot pose error) -> per-pedestrian Kalman tracks -> Social LSTM + CV state
  covariance -> SupeR-MPC (optionally with active perception) -> robot moves with
  true (uncertain) parameters and process noise.

Planners:
  dmpc             deterministic MPC, camera fixed forward
  supermpc         SupeR-MPC chance constraints, camera fixed forward
  supermpc-scan    SupeR-MPC, camera sweeping back and forth (naive active sensing)
  supermpc-active  SupeR-MPC with the risk-aware active perception cost

Example:
    python run_nav.py --planners dmpc supermpc supermpc-active --gif
    python run_nav.py --episodes 20 --planners supermpc supermpc-scan supermpc-active
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Wedge

from robot_nav.camera import CameraConfig, observe
from robot_nav.mpc import MPCConfig, SupeRMPC
from robot_nav.pmb import PMBConfig, PMBTracker, fit_scene_prior
from robot_nav.prediction import Prediction, TrajectoryPredictor
from robot_nav.robot import RobotConfig, dynamics
from robot_nav.scene import PedestrianScene
from robot_nav.tracking import Tracker, TrackerConfig

SPAWN_RADIUS = 3.0  # m

ACTIVE = dict(robust=True, camera_mode="active")
ORACLE_CAMERA = dict(fov_deg=360.0, max_range=1000.0, p_detect=1.0, occlusion=False, sigma0=0.01, sigma_per_m=0.0)

# Each planner: MPCConfig fields ("mpc"), optional tracker override ("tracker": "kf" | "pmb"),
# CameraConfig overrides ("camera"), and whether the predictor adds the CV state covariance.
PLANNERS = {
    # baselines
    "dmpc": dict(mpc=dict(robust=False, camera_mode="fixed")),
    "supermpc": dict(mpc=dict(robust=True, camera_mode="fixed")),
    "supermpc-scan": dict(mpc=dict(robust=True, camera_mode="scan")),
    # look at the closest tracked pedestrian (before 2026-10-13 this name meant the "ttc" behaviour below)
    "supermpc-nearest": dict(mpc=dict(robust=True, camera_mode="nearest")),
    "supermpc-infogain": dict(mpc=dict(ACTIVE, perception_objective="entropy")),
    "supermpc-ttc": dict(mpc=dict(robust=True, camera_mode="ttc")),  # most collision-relevant pedestrian (TTC proxy)
    "supermpc-decoupled": dict(mpc=dict(robust=True, camera_mode="decoupled")),  # motion first, then the camera
    # the method
    "supermpc-active": dict(mpc=dict(ACTIVE)),
    "supermpc-hybrid": dict(mpc=dict(ACTIVE, perception_objective="hybrid")),  # risk + information gain
    # oracles (upper bounds)
    "supermpc-360": dict(mpc=dict(robust=True, camera_mode="fixed"), camera=dict(fov_deg=360.0)),
    "supermpc-gazeoracle": dict(mpc=dict(robust=True, camera_mode="oracle_gaze")),  # same camera, privileged gaze
    "supermpc-oracle": dict(mpc=dict(robust=True, camera_mode="fixed"), camera=ORACLE_CAMERA),
    # ablations of the method
    "active-tracked-only": dict(mpc=dict(ACTIVE, w_perception_unseen=0.0)),
    "active-unseen-only": dict(mpc=dict(ACTIVE, w_perception=0.0)),
    "active-kf": dict(mpc=dict(ACTIVE), tracker="kf"),
    "active-no-sens": dict(mpc=dict(ACTIVE, robot_sensitivity=False)),
    "active-no-statecov": dict(mpc=dict(ACTIVE), state_cov=False),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="data/ucy/zara01/obsmat.txt")
    p.add_argument("--checkpoint", default="checkpoints/social_lstm_zara_varobs.pt")
    p.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=list(PLANNERS))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--start-frame", type=int, default=5281, help="first episode's start frame")
    p.add_argument("--frame-range", type=int, nargs=2, default=[200, 8500],
                   help="later episodes sample start frames uniformly from this range")
    p.add_argument("--start", type=float, nargs=2, default=[-7.0, 13.0])
    p.add_argument("--goal", type=float, nargs=2, default=[5.0, 13.0])
    p.add_argument("--y-jitter", type=float, default=3.0, help="later episodes shift the crossing line by U(-j, j)")
    p.add_argument("--max-steps", type=int, default=90)
    p.add_argument("--goal-tol", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-dir", default="plots/nav")
    p.add_argument("--gif", action="store_true", help="save an animation of each episode")
    p.add_argument("--fov", type=float, default=70.0, help="camera field of view (deg)")
    p.add_argument("--delta-unseen", type=float, default=None,
                   help="per-step risk budget for undetected pedestrians (default: MPCConfig)")
    p.add_argument("--tracker", choices=["kf", "pmb"], default="kf",
                   help="kf: Kalman tracks only; pmb: Poisson multi-Bernoulli (tracks + undetected intensity)")
    return p.parse_args()


def run_episode(name, scene, start, goal, predictor, args, seed, prior=None, hook=None):
    """hook(**state), if given, is called every step after planning (diagnostics)."""
    rng = np.random.default_rng(seed)
    spec = PLANNERS[name]
    rcfg, tcfg = RobotConfig(), TrackerConfig()
    for key, value in (getattr(args, "robot_overrides", None) or {}).items():  # e.g. camera pan-rate limit
        setattr(rcfg, key, value)
    cam = CameraConfig(fov_deg=getattr(args, "fov", 70.0), max_range=getattr(args, "cam_range", 6.0))
    for key, value in spec.get("camera", {}).items():
        setattr(cam, key, value)
    mcfg = MPCConfig(**spec["mpc"])
    if args.delta_unseen is not None:
        mcfg.delta_unseen = args.delta_unseen
    for key, value in (getattr(args, "mpc_overrides", None) or {}).items():  # e.g. perception weights
        setattr(mcfg, key, value)
    mpc = SupeRMPC(mcfg, rcfg, cam, tcfg)
    tracker_kind = spec.get("tracker") or args.tracker
    tracker = PMBTracker(prior, cam, PMBConfig(**(getattr(args, "pmb_overrides", None) or {}))) if tracker_kind == "pmb" else Tracker(tcfg)
    if tracker_kind == "pmb" and mcfg.w_explore > 0:  # crowd-density exploration term (option 2)
        mpc.explore_fn = lambda c: tracker.explore_masses(c, mcfg.explore_slots, mcfg.explore_radius)
    # Camera sub-steps: the camera turns and observes cam_substeps times per control step; the robot
    # input, pedestrian prediction and collision checks stay on the 0.4 s grid.
    n_sub = mcfg.cam_substeps
    assert n_sub == 1 or tracker_kind == "pmb", "camera sub-steps need the PMB tracker"
    L0, LQ = np.linalg.cholesky(mcfg.Sigma0), np.linalg.cholesky(mcfg.Sigma_Q)
    p_true = rcfg.p_nominal + np.linalg.cholesky(mcfg.Sigma_p) @ rng.standard_normal(2)
    d = goal - start
    x = np.array([*start, np.arctan2(d[1], d[0]), 0.0])
    r_col = rcfg.radius + mcfg.ped_radius

    log = dict(x=[], peds=[], seen=[], tracked=[], plan=[], pred=[], selected=[])
    collisions, clearances, solve_times, failures, coverage = 0, [], [], 0, []
    # Collision attribution: was the pedestrian tracked (with which existence) or never
    # seen, and was the robot moving or (nearly) stopped when it happened.
    coll_kinds = dict(tracked=0, untracked=0, stopped=0, spawn=0)
    # Pedestrians whose replay starts mid-episode within SPAWN_RADIUS of the robot (annotation
    # begins next to it): no sensor or planner could have seen them coming.
    present, spawned = set(scene.positions(0)), set()
    slacks, unseen_slacks, u = [], [], np.zeros(3)
    prev_peds = {}
    reached = False
    for step in range(args.max_steps):
        peds = scene.positions(step, robot=x[:2])  # a reactive crowd responds to the robot
        for pid in set(peds) - present:
            if np.linalg.norm(peds[pid] - x[:2]) < SPAWN_RADIUS:
                spawned.add(pid)
        present |= set(peds)
        if peds:
            clear = min(np.linalg.norm(x[:2] - q) for q in peds.values()) - r_col
            clearances.append(clear)
            collisions += clear < 0
            for pid, q in peds.items():
                if np.linalg.norm(x[:2] - q) < r_col:
                    coll_kinds["tracked" if pid in tracker.tracks else "untracked"] += 1
                    coll_kinds["stopped"] += abs(u[0]) < 0.1
                    coll_kinds["spawn"] += pid in spawned
        x_est = x + L0 @ rng.standard_normal(4)
        det = observe(x, x_est, mcfg.Sigma0, peds, cam, rng)
        if tracker_kind == "pmb":
            tracker.step(det, x_est) if n_sub == 1 else tracker.step(det, x_est, frac=1 / n_sub, record=True)
        else:
            tracker.step(det)
        near = [pid for pid, q in peds.items() if np.linalg.norm(q - x[:2]) < 4.0]
        if near:
            coverage.append(np.mean([pid in tracker.tracks for pid in near]))
        pred = predictor.predict(tracker.tracks, use_state_cov=spec.get("state_cov", True),
                                 state_cov_mode=mcfg.state_cov_mode, coast=mcfg.coast_mode)
        if pred.mean.shape[0] > mcfg.N:  # shorter planning horizon: the rollout's first N steps
            pred = Prediction(pred.ids, pred.mean[: mcfg.N], pred.cov_model[: mcfg.N], pred.cov_state[: mcfg.N],
                              pred.existence, pred.n_det, None if pred.state_jac is None else pred.state_jac[: mcfg.N])
        unseen_fn = None
        if tracker_kind == "pmb":
            unseen_fn = lambda path: tracker.unseen_forecast(path, mcfg.unseen_slots, mcfg.delta_unseen)
        if mcfg.camera_mode == "oracle_gaze":
            ids = [pid for pid in peds if pid in prev_peds]
            mpc.gt = (np.array([peds[i] for i in ids]).reshape(-1, 2),
                      np.array([(peds[i] - prev_peds[i]) / rcfg.dt for i in ids]).reshape(-1, 2))
        prev_peds = {k: v.copy() for k, v in peds.items()}
        u, info = mpc.solve(x_est, goal, pred, t=step * rcfg.dt, unseen_fn=unseen_fn)
        solve_times.append(info["solve_time"])
        failures += not info["ok"]
        slacks.append(info["max_slack"])
        unseen_slacks.append(info["unseen_slack"])

        log["x"].append(x.copy())
        log["peds"].append(peds)
        log["seen"].append(set(det))
        log["tracked"].append({pid: tr.x[:2].copy() for pid, tr in tracker.tracks.items()})
        log["plan"].append(info["X"])
        log["pred"].append(pred)
        log["selected"].append(info["selected"])
        if hook is not None:
            hook(step=step, x=x.copy(), x_est=x_est, peds=peds, tracks=tracker.tracks, pred=pred, info=info, u=u,
                 r_col=r_col, delta=mcfg.delta, tracker=tracker, det=det)

        if n_sub == 1:
            x = dynamics(x, u, p_true, rcfg.dt) + LQ @ rng.standard_normal(4)
        else:
            # Sub-steps: move with the s-th camera rate, observe the crowd interpolated to that time.
            peds_next = scene.positions(step + 1, robot=x[:2])
            for s in range(n_sub):
                x = dynamics(x, np.array([u[0], u[1], info["cam_sub"][s]]), p_true, rcfg.dt / n_sub)
                if s == n_sub - 1:
                    break
                f = (s + 1) / n_sub
                peds_sub = {pid: (1 - f) * q + f * peds_next[pid] for pid, q in peds.items() if pid in peds_next}
                x_sub = x + L0 @ rng.standard_normal(4)
                det_sub = observe(x, x_sub, mcfg.Sigma0, peds_sub, cam, rng)
                tracker.step(det_sub, x_sub, frac=1 / n_sub, record=False)
                log["seen"][-1] |= set(det_sub)
            x = x + LQ @ rng.standard_normal(4)
        if np.linalg.norm(x[:2] - goal) < args.goal_tol:
            reached = True
            break
    log["x"].append(x.copy())
    return dict(
        planner=name, reached=reached, collisions=int(collisions), steps=step + 1,
        success=reached and collisions == 0,
        min_clearance=float(min(clearances)) if clearances else np.inf,
        coll_tracked=coll_kinds["tracked"], coll_untracked=coll_kinds["untracked"],
        coll_stopped=coll_kinds["stopped"], coll_spawn=coll_kinds["spawn"],
        success_nospawn=reached and collisions - coll_kinds["spawn"] == 0,
        slack_steps=int(np.sum(np.array(slacks) > 1e-3)),
        unseen_slack_steps=int(np.sum(np.array(unseen_slacks) > 1e-3)),
        coverage=float(np.mean(coverage)) if coverage else np.nan,
        solve_ms=1e3 * float(np.nanmean(solve_times)), failures=failures,
    ), log


# ----------------------------------------------------------------------------- plotting

def fov_wedge(x, cam, **kw):
    heading = np.degrees(x[2] + x[3])
    return Wedge(x[:2], cam.max_range, heading - cam.fov_deg / 2, heading + cam.fov_deg / 2, **kw)


def cov_ellipse(mean, S, k=2.0, **kw):
    vals, vecs = np.linalg.eigh(S)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * k * np.sqrt(np.maximum(vals[::-1], 0))
    return Ellipse(mean, w, h, angle=angle, **kw)


def plot_episode(results, logs, start, goal, path):
    cam, rcfg = CameraConfig(), RobotConfig()
    fig, axes = plt.subplots(1, len(logs), figsize=(6 * len(logs), 6), squeeze=False)
    for ax, res, log in zip(axes[0], results, logs):
        traj = {}
        for peds in log["peds"]:
            for pid, q in peds.items():
                traj.setdefault(pid, []).append(q)
        for q in traj.values():
            q = np.array(q)
            ax.plot(q[:, 0], q[:, 1], "-", color="0.75", lw=1)
        xs = np.array(log["x"])
        for i in range(0, len(xs) - 1, 4):
            ax.add_patch(fov_wedge(xs[i], cam, color="tab:orange", alpha=0.06))
        sc = ax.scatter(xs[:, 0], xs[:, 1], c=np.arange(len(xs)), cmap="viridis", s=12, zorder=3)
        ax.plot(*start, "ks", ms=8)
        ax.plot(*goal, "k*", ms=14)
        for i, peds in enumerate(log["peds"]):
            for q in peds.values():
                if np.linalg.norm(q - xs[i, :2]) < rcfg.radius + 0.3:
                    ax.add_patch(Circle(xs[i, :2], rcfg.radius, color="red", alpha=0.5, zorder=4))
        ax.set_title(f"{res['planner']}: {'success' if res['success'] else 'FAIL'} "
                     f"({res['collisions']} coll., min clr {res['min_clearance']:.2f} m, {res['steps']} steps)")
        ax.set_aspect("equal")
        ax.set_xlim(min(start[0], goal[0]) - 2, max(start[0], goal[0]) + 2)
        ax.set_ylim(start[1] - 7, start[1] + 7)
        fig.colorbar(sc, ax=ax, fraction=0.03, label="step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def animate_episode(res, log, start, goal, path):
    cam, rcfg = CameraConfig(), RobotConfig()
    fig, ax = plt.subplots(figsize=(7, 7))

    def draw(i):
        ax.clear()
        x = log["x"][i]
        ax.add_patch(fov_wedge(x, cam, color="tab:orange", alpha=0.15))
        for pid, q in log["peds"][i].items():
            color = "tab:green" if pid in log["seen"][i] else ("tab:blue" if pid in log["tracked"][i] else "0.6")
            ax.add_patch(Circle(q, 0.3, color=color, alpha=0.6))
        pred = log["pred"][i]
        for pid in log["selected"][i]:
            j = pred.ids.index(pid)
            ax.plot(pred.mean[:, j, 0], pred.mean[:, j, 1], ":", color="tab:blue", lw=1)
            K = pred.mean.shape[0]
            for k in (1, K // 2, K - 1):
                ax.add_patch(cov_ellipse(pred.mean[k, j], pred.cov[k, j], fill=False, color="tab:blue",
                                         alpha=0.5, lw=0.8))
        plan = log["plan"][i]
        ax.plot(plan[0], plan[1], "-", color="tab:red", lw=1.5)
        ax.add_patch(Circle(x[:2], rcfg.radius, color="tab:red", alpha=0.8))
        ax.plot(*goal, "k*", ms=14)
        ax.set_xlim(min(start[0], goal[0]) - 2, max(start[0], goal[0]) + 2)
        ax.set_ylim(start[1] - 7, start[1] + 7)
        ax.set_aspect("equal")
        ax.set_title(f"{res['planner']}  step {i}   green=seen  blue=tracked  gray=unknown", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(log["plan"]), interval=400)
    anim.save(path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    predictor = TrajectoryPredictor(args.checkpoint, MPCConfig().N)
    prior = fit_scene_prior(args.scene) if args.tracker == "pmb" else None
    ep_rng = np.random.default_rng(args.seed)
    all_results = []
    for ep in range(args.episodes):
        start_frame = args.start_frame if ep == 0 else int(ep_rng.integers(*args.frame_range))
        dy = 0.0 if ep == 0 else ep_rng.uniform(-args.y_jitter, args.y_jitter)
        start, goal = np.array(args.start) + [0, dy], np.array(args.goal) + [0, dy]
        scene = PedestrianScene(args.scene, start_frame)
        results, logs = [], []
        for name in args.planners:
            t0 = time.time()
            res, log = run_episode(name, scene, start, goal, predictor, args, seed=args.seed + ep, prior=prior)
            res["episode"] = ep
            print(f"ep {ep:3d} frame {scene.start_frame:5d}  {name:16s} success={res['success']!s:5s} "
                  f"reached={res['reached']!s:5s} coll={res['collisions']} min_clr={res['min_clearance']:+.2f} "
                  f"steps={res['steps']:2d} coverage={res['coverage']:.2f} solve={res['solve_ms']:.0f}ms "
                  f"fail={res['failures']} coll[trk/untrk/stopped]={res['coll_tracked']}/"
                  f"{res['coll_untracked']}/{res['coll_stopped']} slack_steps[ped/unseen]="
                  f"{res['slack_steps']}/{res['unseen_slack_steps']} ({time.time() - t0:.0f}s)")
            results.append(res)
            logs.append(log)
            if args.gif:
                animate_episode(res, log, start, goal, os.path.join(args.plot_dir, f"ep{ep}_{name}.gif"))
        all_results.extend(results)
        if ep < 5:
            plot_episode(results, logs, start, goal, os.path.join(args.plot_dir, f"ep{ep}.png"))

    print("\nSummary over", args.episodes, "episodes")
    for name in args.planners:
        rs = [r for r in all_results if r["planner"] == name]
        print(f"  {name:16s} success={np.mean([r['success'] for r in rs]):.2f}  "
              f"collision-free={np.mean([r['collisions'] == 0 for r in rs]):.2f}  "
              f"reached={np.mean([r['reached'] for r in rs]):.2f}  "
              f"steps(reached)={np.mean([r['steps'] for r in rs if r['reached']] or [np.nan]):.1f}  "
              f"coverage={np.nanmean([r['coverage'] for r in rs]):.2f}  "
              f"solve={np.mean([r['solve_ms'] for r in rs]):.0f}ms")


if __name__ == "__main__":
    main()
