"""Large paired evaluation of the navigation planners on ETH/UCY crowds.

Protocols:
  legacy  Zara01/Zara02 only, episodes anywhere in the recording, one predictor
          checkpoint and a PMB prior fitted on the whole file (the original runs).
  clean   no test data in training: each scene uses a Social LSTM trained with that
          scene left out (checkpoints/loo_<scene>.pt), its PMB prior is fitted on the
          first half of the recording, and episodes start only in the second half.
Crowds: replay (recorded trajectories) or reactive (recorded paths + social-force
avoidance of the robot).

Every planner runs on the same episodes (scene, start frame, crossing line,
direction, noise seed), so comparisons are paired. Episodes run in parallel
worker processes and each result is appended to a CSV as soon as it finishes;
re-running skips finished (scene, episode, planner) jobs.

Analysis (also available alone with --analyze-only):
  success / collision-free / reached rates with 95% Wilson intervals
  paired McNemar exact tests vs the reference planner (success)
  paired Wilcoxon signed-rank test on steps to goal (episodes both reached)
  collision-steps split into tracked / never-detected pedestrians
  the same on the "interacting" subset (pedestrians cross near the robot's
  straight-line path) and with the Zara01 crowd-surge episodes broken out

Example:
    python evaluate_nav.py --episodes 100 --workers 12
    python evaluate_nav.py --analyze-only
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np

SCENES = {
    # obsmat path; the robot travels along `axis` from span[0] to span[1] (or back), on a crossing
    # line drawn from `line` on the other axis, perpendicular to the scene's main pedestrian flow.
    # `frames` = legacy start-frame range. Order matters: it seeds the episodes.
    "zara01": dict(path="data/ucy/zara01/obsmat.txt", axis=0, span=(-7.0, 5.0), line=(10.0, 16.0),
                   frames=(200, 8500)),
    "zara02": dict(path="data/ucy/zara02/obsmat.txt", axis=0, span=(-7.5, 5.0), line=(-8.0, 2.0),
                   frames=(200, 10000)),
    "eth": dict(path="data/eth/ewap_dataset/seq_eth/obsmat.txt", axis=1, span=(-1.0, 11.0), line=(2.0, 10.0)),
    "hotel": dict(path="data/eth/ewap_dataset/seq_hotel/obsmat.txt", axis=0, span=(-4.0, 6.0), line=(-7.0, 2.0)),
    "univ": dict(path="data/ucy/students03/obsmat.txt", axis=0, span=(-6.0, 6.0), line=(-4.0, 6.0)),
}
_FRAMES = {}


def scene_frames(scene_name):
    """(first frame, frame step, last frame, train/test split frame) of a recording."""
    if scene_name not in _FRAMES:
        f = np.unique(np.loadtxt(SCENES[scene_name]["path"])[:, 0])
        step = int(np.median(np.diff(f)))
        split = f.min() + step * round((f.max() - f.min()) / 2 / step)
        _FRAMES[scene_name] = (int(f.min()), step, int(f.max()), int(split))
    return _FRAMES[scene_name]


LOO_PREFIX = os.environ.get("LOO_PREFIX", "loo")  # "loorot": predictors trained with rotation augmentation


def checkpoint_for(scene_name, protocol, default):
    return f"checkpoints/{LOO_PREFIX}_{scene_name}.pt" if protocol == "clean" else default
SURGE = ("zara01", 5050, 5450)  # crowd surge (~3x mean density) found in the PMB calibration
PLANNERS = ["dmpc", "supermpc", "supermpc-scan", "supermpc-active"]
FIELDS = ["scene", "episode", "frame", "start_x", "start_y", "goal_x", "goal_y", "path_peds", "surge", "planner",
          "success", "reached", "collisions", "coll_tracked", "coll_untracked", "coll_stopped", "steps",
          "min_clearance", "coverage", "solve_ms", "failures", "slack_steps", "unseen_slack_steps",
          "coll_spawn", "success_nospawn"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    p.add_argument("--planners", nargs="+", default=PLANNERS)
    p.add_argument("--episodes", type=int, default=100, help="episodes per scene")
    p.add_argument("--tracker", choices=["kf", "pmb"], default="pmb")
    p.add_argument("--delta-unseen", type=float, default=0.01)
    p.add_argument("--fov", type=float, default=70.0, help="camera field of view (deg)")
    p.add_argument("--range", type=float, default=6.0, help="camera range (m)")
    p.add_argument("--w-perception", type=float, default=None, help="tracked-pedestrian perception weight")
    p.add_argument("--w-perception-unseen", type=float, default=None, help="unseen-pedestrian perception weight")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra MPCConfig overrides, e.g. --set unseen_joint=False")
    p.add_argument("--robot-set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="RobotConfig overrides, e.g. --robot-set omega_cam_bounds=(-5.0,5.0)")
    p.add_argument("--pmb-set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="PMBConfig overrides, e.g. --pmb-set prune_pos_std=1.0")
    p.add_argument("--protocol", choices=["legacy", "clean"], default="legacy")
    p.add_argument("--crowd", choices=["replay", "reactive"], default="replay")
    p.add_argument("--max-steps", type=int, default=90)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoints/social_lstm_zara_varobs.pt")
    p.add_argument("--out", default="results/nav_eval.csv")
    p.add_argument("--reference", default="supermpc-active", help="planner the paired tests compare against")
    p.add_argument("--analyze-only", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------- episodes

def make_episodes(scene_name, n, seed, protocol="legacy", max_steps=90):
    """Seeded episode list; also counts pedestrians passing near the straight-line path.
    The clean protocol starts episodes only in the second half of the recording."""
    from robot_nav.scene import PedestrianScene

    sc = SCENES[scene_name]
    a, o = sc["axis"], 1 - sc["axis"]
    if protocol == "clean":
        _, step_f, last, split = scene_frames(scene_name)
        frame_range = (split, last - max_steps * step_f)
    elif "frames" in sc:
        frame_range = sc["frames"]
    else:
        first, step_f, last, _ = scene_frames(scene_name)
        frame_range = (first, last - max_steps * step_f)
    rng = np.random.default_rng([seed, list(SCENES).index(scene_name)])
    episodes = []
    for ep in range(n):
        frame = int(rng.integers(*frame_range))
        c = rng.uniform(*sc["line"])
        start, goal = np.zeros(2), np.zeros(2)
        start[a], start[o], goal[a], goal[o] = sc["span"][0], c, sc["span"][1], c
        if rng.uniform() < 0.5:
            start, goal = goal, start
        scene = PedestrianScene(sc["path"], frame)
        near = set()
        lo, hi = min(sc["span"]), max(sc["span"])
        for step in range(40):  # ~16 s: the time a direct crossing takes, with margin
            for pid, q in scene.positions(step).items():
                if lo <= q[a] <= hi and abs(q[o] - c) < 1.5:
                    near.add(pid)
        surge = scene_name == SURGE[0] and SURGE[1] <= scene.start_frame <= SURGE[2]
        episodes.append(dict(scene=scene_name, episode=ep, frame=scene.start_frame, start=start, goal=goal,
                             path_peds=len(near), surge=surge))
    return episodes


# ----------------------------------------------------------------------------- workers

_WORKER = {}


def _init_worker(checkpoint):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from robot_nav.mpc import MPCConfig
    from robot_nav.prediction import TrajectoryPredictor

    _WORKER["predictor"] = TrajectoryPredictor(checkpoint, MPCConfig().N)
    _WORKER["predictors"] = {checkpoint: _WORKER["predictor"]}
    _WORKER["priors"] = {}


def _predictor(checkpoint):
    from robot_nav.mpc import MPCConfig
    from robot_nav.prediction import TrajectoryPredictor

    if checkpoint not in _WORKER["predictors"]:
        _WORKER["predictors"][checkpoint] = TrajectoryPredictor(checkpoint, MPCConfig().N)
    return _WORKER["predictors"][checkpoint]


def _run_job(job, opts):
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene, ReactiveScene
    from run_nav import run_episode

    ep, planner = job
    path = SCENES[ep["scene"]]["path"]
    key = (ep["scene"], opts["protocol"])
    if key not in _WORKER["priors"]:  # always built: some planners use the PMB tracker regardless
        first, _, _, split = scene_frames(ep["scene"])
        _WORKER["priors"][key] = fit_scene_prior(path, frame_range=(first, split) if opts["protocol"] == "clean"
                                                 else None)
    prior = _WORKER["priors"][key]
    args = SimpleNamespace(max_steps=opts["max_steps"], goal_tol=0.3, tracker=opts["tracker"],
                           delta_unseen=opts["delta_unseen"], fov=opts["fov"], cam_range=opts["range"],
                           mpc_overrides=opts["mpc_overrides"], pmb_overrides=opts.get("pmb_overrides"),
                           robot_overrides=opts.get("robot_overrides"))
    scene = (ReactiveScene if opts["crowd"] == "reactive" else PedestrianScene)(path, ep["frame"])
    seed = opts["seed"] + 1000 * list(SCENES).index(ep["scene"]) + ep["episode"]
    predictor = _predictor(checkpoint_for(ep["scene"], opts["protocol"], opts["checkpoint"]))
    res, _ = run_episode(planner, scene, ep["start"], ep["goal"], predictor, args, seed, prior)
    row = dict(scene=ep["scene"], episode=ep["episode"], frame=ep["frame"], start_x=ep["start"][0],
               start_y=ep["start"][1], goal_x=ep["goal"][0], goal_y=ep["goal"][1], path_peds=ep["path_peds"],
               surge=int(ep["surge"]), planner=planner)
    for k in FIELDS[10:]:
        v = res[k]
        row[k] = int(v) if isinstance(v, (bool, np.bool_)) else v
    return row


def run_all(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            done = {(r["scene"], int(r["episode"]), r["planner"]) for r in csv.DictReader(f)}
    jobs = []
    for scene in args.scenes:
        for ep in make_episodes(scene, args.episodes, args.seed, args.protocol, args.max_steps):
            jobs += [(ep, pl) for pl in args.planners if (scene, ep["episode"], pl) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already in {args.out}), {args.workers} workers")
    if not jobs:
        return
    opts = dict(robot_overrides={kv.split("=", 1)[0]: ast.literal_eval(kv.split("=", 1)[1]) for kv in args.robot_set},
                pmb_overrides={kv.split("=", 1)[0]: ast.literal_eval(kv.split("=", 1)[1]) for kv in args.pmb_set},
                tracker=args.tracker, delta_unseen=args.delta_unseen, max_steps=args.max_steps, seed=args.seed,
                protocol=args.protocol, crowd=args.crowd, checkpoint=args.checkpoint,
                fov=args.fov, range=args.range,
                mpc_overrides={**{k: v for k, v in (("w_perception", args.w_perception),
                                                    ("w_perception_unseen", args.w_perception_unseen)) if v is not None},
                               **{kv.split("=", 1)[0]: ast.literal_eval(kv.split("=", 1)[1]) for kv in args.set}})
    new_file = not os.path.exists(args.out)
    t0 = time.time()
    with open(args.out, "a", newline="") as f, \
            ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.checkpoint,)) as pool:
        writer = csv.DictWriter(f, FIELDS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(_run_job, job, opts): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            ep, planner = futures[fut]
            try:
                writer.writerow(fut.result())
                f.flush()
            except Exception as e:  # keep going; the job is retried on the next run
                print(f"  FAILED {ep['scene']} ep {ep['episode']} {planner}: {e!r}")
            if i % 20 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} done, {el / 60:.1f} min elapsed, ~{el / i * (len(jobs) - i) / 60:.1f} min left",
                      flush=True)


# ----------------------------------------------------------------------------- analysis

def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FIELDS:
            if k not in ("scene", "planner") and r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def table(rows, planners, title, reference):
    from scipy.stats import binomtest, wilcoxon

    by = {(r["scene"], r["episode"], r["planner"]): r for r in rows}
    eps = sorted({(r["scene"], r["episode"]) for r in rows})
    eps = [e for e in eps if all((e[0], e[1], p) in by for p in planners)]  # complete pairs only
    if not eps:
        return
    print(f"\n=== {title}: {len(eps)} episodes ===")
    print(f"{'planner':16s} {'success [95% CI]':24s} {'coll-free':>9s} {'reached':>8s} {'steps(reached)':>15s} "
          f"{'coll-steps trk/untrk':>21s} {'McNemar vs ref':>17s} {'steps vs ref (both reached)':>30s}"
          f"   {'excl. spawn: success, McNemar (spawn coll-steps)'}")
    ref = [by[(s, e, reference)] for s, e in eps] if reference in planners else None
    for pl in planners:
        rs = [by[(s, e, pl)] for s, e in eps]
        n, k = len(rs), int(sum(r["success"] for r in rs))
        lo, hi = wilson(k, n)
        cf = np.mean([r["collisions"] == 0 for r in rs])
        rc = np.mean([r["reached"] for r in rs])
        steps = [r["steps"] for r in rs if r["reached"]]
        trk, untrk = int(sum(r["coll_tracked"] for r in rs)), int(sum(r["coll_untracked"] for r in rs))
        mc, st = "", ""
        if ref is not None and pl != reference:
            a = sum(1 for x, y in zip(ref, rs) if x["success"] and not y["success"])  # ref wins
            b = sum(1 for x, y in zip(ref, rs) if y["success"] and not x["success"])  # this wins
            p = binomtest(a, a + b, 0.5).pvalue if a + b else 1.0
            mc = f"{a}:{b} p={p:.3f}"
            both = [(x["steps"], y["steps"]) for x, y in zip(ref, rs) if x["reached"] and y["reached"]]
            if len(both) > 5:
                d = np.array([y - x for x, y in both])
                pw = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                st = f"{np.median(d):+.0f} median, p={pw:.3f}"
        ns = ""
        if all(isinstance(r.get("success_nospawn"), float) for r in rs):
            ns = f"{np.mean([r['success_nospawn'] for r in rs]):.2f}"
            if ref is not None and pl != reference:
                a = sum(1 for x, y in zip(ref, rs) if x["success_nospawn"] and not y["success_nospawn"])
                b = sum(1 for x, y in zip(ref, rs) if y["success_nospawn"] and not x["success_nospawn"])
                ns += f", {a}:{b} p={binomtest(a, a + b, 0.5).pvalue if a + b else 1.0:.3f}"
            ns += f" ({int(sum(r['coll_spawn'] for r in rs))})"
        print(f"{pl:16s} {k / n:.2f} [{lo:.2f}, {hi:.2f}]        {cf:9.2f} {rc:8.2f} "
              f"{np.mean(steps) if steps else math.nan:15.1f} {trk:>10d}/{untrk:<10d} {mc:>17s} {st:>30s}   {ns}")


def plot(rows, planners, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("all", lambda r: True), ("interacting", lambda r: r["path_peds"] >= 3)] + \
             [(s, (lambda s: lambda r: r["scene"] == s)(s)) for s in SCENES]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.8 / len(planners)
    for i, pl in enumerate(planners):
        vals, errs = [], []
        for _, f in groups:
            rs = [r for r in rows if r["planner"] == pl and f(r)]
            k, n = sum(r["success"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            vals.append(k / n if n else np.nan)
            errs.append([vals[-1] - lo, hi - vals[-1]])
        axes[0].bar(np.arange(len(groups)) + i * w, vals, w, yerr=np.array(errs).T, capsize=3, label=pl)
    axes[0].set_xticks(np.arange(len(groups)) + 0.4 - w / 2)
    axes[0].set_xticklabels([g for g, _ in groups])
    axes[0].set(ylabel="success rate (95% Wilson CI)", ylim=(0, 1.05), title="Success")
    axes[0].legend(fontsize=8)
    data = [[r["steps"] for r in rows if r["planner"] == pl and r["reached"]] for pl in planners]
    axes[1].boxplot(data, labels=planners, showfliers=False)
    axes[1].set(ylabel="steps to goal (0.4 s)", title="Time to goal (episodes that reached)")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nSaved {path}")


def analyze(args):
    rows = load(args.out)
    planners = [p for p in args.planners if any(r["planner"] == p for r in rows)]
    print(f"\nReference planner for paired tests: {args.reference}  "
          f"(McNemar a:b = episodes only the reference succeeded : only this planner succeeded)")
    table(rows, planners, "All scenes", args.reference)
    for s in SCENES:
        table([r for r in rows if r["scene"] == s], planners, f"{s}", args.reference)
    table([r for r in rows if r["path_peds"] >= 3], planners, "Interacting (>=3 pedestrians near the path)",
          args.reference)
    table([r for r in rows if not r["surge"]], planners, "Excluding Zara01 surge episodes", args.reference)
    table([r for r in rows if r["surge"]], planners, "Zara01 surge episodes only", args.reference)
    plot(rows, planners, os.path.splitext(args.out)[0] + ".png")


def main():
    args = parse_args()
    if not args.analyze_only:
        run_all(args)
    analyze(args)


if __name__ == "__main__":
    main()
