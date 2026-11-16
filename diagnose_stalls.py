"""Check 2: what blocks the robot in episodes that time out?

Reruns the timed-out episodes of a clean-protocol evaluation (identical seeds,
so identical runs) with a per-step hook. A step is a stall when the commanded
speed is < 0.1 m/s and the robot is > 1 m from the goal. At each stall step the
blocker is the nearest constrained pedestrian whose predicted mean lies within
60 deg of the goal direction and 3 m; it is classed by its true speed (standing
< 0.2 m/s). For the blocker the script records the 2-sigma radius of the model
and state (tracker) covariance parts, and the existence r. Stalls with no such
pedestrian are "nothing tracked ahead" (unseen constraint or geometry).
Saves a snapshot of the longest stall of a few episodes.

    python diagnose_stalls.py --csv results/clean/replay.csv --scene zara02 --planners supermpc-active supermpc
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from evaluate_nav import SCENES, _WORKER, _init_worker, _predictor, checkpoint_for, load, make_episodes, scene_frames


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="results/clean/replay.csv")
    p.add_argument("--scene", default="zara02")
    p.add_argument("--planners", nargs="+", default=["supermpc-active", "supermpc"])
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--range", type=float, default=4.0)
    p.add_argument("--delta-unseen", type=float, default=0.02)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--figs", type=int, default=4, help="snapshot figures per planner")
    p.add_argument("--outdir", default="results/stalls")
    return p.parse_args()


def r2sigma(S):
    return 2 * np.sqrt(np.linalg.eigvalsh(S)[-1])


def _job(ep, planner, opts):
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    sc = ep["scene"]
    if sc not in _WORKER["priors"]:
        first, _, _, split = scene_frames(sc)
        _WORKER["priors"][sc] = fit_scene_prior(SCENES[sc]["path"], frame_range=(first, split))
    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=opts["delta_unseen"],
                           fov=opts["fov"], cam_range=opts["range"], mpc_overrides=dict(unseen_joint=False))
    recs, prev = [], {}

    def hook(step, x, x_est, peds, tracks, pred, info, u, r_col, delta, **_):
        nonlocal prev
        speeds = {pid: np.linalg.norm(q - prev[pid]) / 0.4 for pid, q in peds.items() if pid in prev}
        prev = {k: v.copy() for k, v in peds.items()}
        recs.append(dict(step=step, x=x, u=np.array(u), peds={k: v.copy() for k, v in peds.items()}, speeds=speeds,
                         ids=list(pred.ids), mean=pred.mean.copy(), cm=pred.cov_model.copy(), cs=pred.cov_state.copy(),
                         r=pred.existence.copy(), sel=list(info["selected"]), plan=info["X"].copy(),
                         unseen_slack=info["unseen_slack"], slack=float(info["max_slack"]),
                         ages={pid: len(t.history) for pid, t in tracks.items()}))

    seed = opts["seed"] + 1000 * list(SCENES).index(sc) + ep["episode"]
    res, _ = run_episode(planner, PedestrianScene(SCENES[sc]["path"], ep["frame"]), ep["start"], ep["goal"],
                         _predictor(checkpoint_for(sc, "clean", None)), args, seed, _WORKER["priors"][sc], hook=hook)
    goal = ep["goal"]
    stalls = []
    for rec in recs:
        x = rec["x"]
        if abs(rec["u"][0]) >= 0.1 or np.linalg.norm(goal - x[:2]) < 1.0:
            continue
        g = (goal - x[:2]) / np.linalg.norm(goal - x[:2])
        best = None
        for pid in rec["sel"]:
            j = rec["ids"].index(pid)
            d = rec["mean"][0, j] - x[:2]
            dist = np.linalg.norm(d)
            if dist < 3.0 and d @ g > dist * np.cos(np.radians(60)) and (best is None or dist < best[1]):
                best = (pid, dist, j)
        s = dict(ep=ep["episode"], step=rec["step"], unseen_slack=rec["unseen_slack"] > 1e-3,
                 n_standing_near=sum(1 for pid, v in rec["speeds"].items()
                                     if v < 0.2 and np.linalg.norm(rec["peds"][pid] - x[:2]) < 3.0))
        if best is None:
            s["kind"] = "nothing tracked ahead"
        else:
            pid, dist, j = best
            v = rec["speeds"].get(pid, np.nan)
            kind = "new" if np.isnan(v) else ("standing" if v < 0.2 else "moving")
            s.update(kind=kind, dist=dist, ped_speed=v,
                     model2s=r2sigma(rec["cm"][-1, j]), state2s=r2sigma(rec["cs"][-1, j]),
                     model2s_1=r2sigma(rec["cm"][0, j]), state2s_1=r2sigma(rec["cs"][0, j]), r=rec["r"][j],
                     age=rec["ages"].get(pid, 0))
        stalls.append(s)
    progress = 1 - np.linalg.norm(goal - recs[-1]["x"][:2]) / np.linalg.norm(goal - ep["start"])
    return dict(planner=planner, ep=ep, res=res, stalls=stalls, progress=progress, recs=recs)


def snapshot(out, path, cam_range, fov):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Wedge

    # middle of the longest run of consecutive stall steps
    steps = sorted(s["step"] for s in out["stalls"])
    runs, cur = [], [steps[0]]
    for a in steps[1:]:
        if a == cur[-1] + 1:
            cur.append(a)
        else:
            runs.append(cur)
            cur = [a]
    runs.append(cur)
    run = max(runs, key=len)
    rec = out["recs"][run[len(run) // 2]]
    x, ep = rec["x"], out["ep"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(*np.array([r["x"][:2] for r in out["recs"]]).T, color="0.6", lw=1, label="robot path")
    ax.plot(*ep["start"], "ks")
    ax.plot(*ep["goal"], "k*", ms=12)
    head = np.degrees(x[2] + x[3])
    ax.add_patch(Wedge(x[:2], cam_range, head - fov / 2, head + fov / 2, color="gold", alpha=0.2))
    ax.add_patch(Circle(x[:2], 0.3, color="tab:green"))
    ax.plot(*rec["plan"][:2], "g.-", lw=1)
    for pid, q in rec["peds"].items():
        v = rec["speeds"].get(pid, np.nan)
        ax.add_patch(Circle(q, 0.3, color="tab:red" if v < 0.2 else "tab:blue", alpha=0.8))
    for pid in rec["sel"]:
        j = rec["ids"].index(pid)
        for part, col in ((rec["cm"][-1, j], "tab:purple"), (rec["cm"][-1, j] + rec["cs"][-1, j], "k")):
            w, V = np.linalg.eigh(part)
            ax.add_patch(Ellipse(rec["mean"][-1, j], 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]),
                                 angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])), fill=False, color=col, lw=1))
    ax.set_title(f"{out['planner']}  {ep['scene']} ep {ep['episode']}  step {rec['step']} (stalled {len(run)} steps)\n"
                 "red = standing, blue = moving; purple = LSTM 2-sigma, black = LSTM+state 2-sigma at 3.2 s",
                 fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(x[0] - 6, x[0] + 6)
    ax.set_ylim(x[1] - 4.5, x[1] + 4.5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = [r for r in load(args.csv) if r["scene"] == args.scene]
    eps = {e["episode"]: e for e in make_episodes(args.scene, args.episodes, args.seed, "clean", 90)}
    jobs = [(eps[int(r["episode"])], r["planner"]) for r in rows if r["planner"] in args.planners and not r["reached"]]
    print(f"{len(jobs)} timed-out episodes to rerun")
    opts = dict(seed=args.seed, delta_unseen=args.delta_unseen, fov=args.fov, range=args.range)
    ckpt = checkpoint_for(args.scene, "clean", None)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(ckpt,)) as pool:
        outs = list(pool.map(_job, [e for e, _ in jobs], [p for _, p in jobs], [opts] * len(jobs)))
    for pl in args.planners:
        po = [o for o in outs if o["planner"] == pl]
        if not po:
            continue
        mism = sum(o["res"]["reached"] for o in po)
        st = [s for o in po for s in o["stalls"]]
        print(f"\n=== {pl}: {len(po)} timeouts (rerun reached goal in {mism}: should be 0); "
              f"median progress {np.median([o['progress'] for o in po]):.2f}; stall steps {len(st)} "
              f"(median {np.median([len(o['stalls']) for o in po]):.0f} of 90 per episode)")
        c = Counter(s["kind"] for s in st)
        print("  blocker: " + ", ".join(f"{k} {v} ({v / len(st):.0%})" for k, v in c.most_common()))
        per_ep = Counter(Counter(s["kind"] for s in o["stalls"]).most_common(1)[0][0] for o in po if o["stalls"])
        print("  dominant blocker per episode: " + ", ".join(f"{k} {v}" for k, v in per_ep.most_common()))
        print(f"  standing pedestrians within 3 m during stalls: median "
              f"{np.median([s['n_standing_near'] for s in st]):.0f}; unseen constraint needed slack in "
              f"{np.mean([s['unseen_slack'] for s in st]):.0%} of stall steps")
        for kind in ("standing", "moving", "new"):
            ss = [s for s in st if s["kind"] == kind]
            if ss:
                med = {k: np.median([s[k] for s in ss]) for k in
                       ("dist", "model2s_1", "state2s_1", "model2s", "state2s", "r", "age")}
                print(f"  {kind:8s} blocker: dist {med['dist']:.2f} m, 2-sigma radius at 0.4 s model "
                      f"{med['model2s_1']:.2f} / state {med['state2s_1']:.2f} m, at 3.2 s model {med['model2s']:.2f} / "
                      f"state {med['state2s']:.2f} m, existence r {med['r']:.2f}, track age {med['age']:.0f}")
        for o in sorted(po, key=lambda o: -len(o["stalls"]))[: args.figs]:
            if o["stalls"]:
                f = os.path.join(args.outdir, f"{args.scene}_{pl}_ep{o['ep']['episode']}.png")
                snapshot(o, f, args.range, args.fov)
                print(f"  saved {f}")


if __name__ == "__main__":
    main()
