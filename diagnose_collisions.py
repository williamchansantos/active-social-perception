"""Explain every collision with a TRACKED pedestrian in a set of evaluation episodes.

Reruns the episodes of an evaluate_nav.py run (same episode seed and noise
seeds, so the runs are identical) with a per-step hook, and for each first
contact with a pedestrian the robot was tracking, looks at the plan made one
step earlier and assigns the first matching cause:

  not-constrained    tracked, but not among the pedestrians the MPC constrained
                     (existence r <= delta, beyond consider_radius, or not in
                     the max_peds nearest)
  planned-violation  the plan already used this pedestrian's constraint slack at
                     the next step: the MPC saw no feasible way to keep clear
  prediction-miss    the plan kept clear, but the pedestrian ended outside the
                     95% ellipse of the one-step prediction
  execution-error    plan and prediction fine, but the robot ended > 0.15 m from
                     its planned position
  within-risk        none of the above: the accepted chance-constraint tail

Example:
    python diagnose_collisions.py --seed 2 --delta-unseen 0.02 --fov 100
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np

CHI2_95_2D = 5.991


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--fov", type=float, default=100.0)
    p.add_argument("--range", type=float, default=6.0)
    p.add_argument("--delta-unseen", type=float, default=0.02)
    p.add_argument("--planners", nargs="+", default=["supermpc", "supermpc-active"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--checkpoint", default="checkpoints/social_lstm_zara_varobs.pt")
    p.add_argument("--examples", type=int, default=6, help="example events to print per cause")
    p.add_argument("--w-slack", type=float, default=None, help="override MPCConfig.w_slack")
    p.add_argument("--protocol", choices=["legacy", "clean"], default="legacy")
    p.add_argument("--scenes", nargs="+", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="MPCConfig overrides")
    p.add_argument("--pmb-set", nargs="*", default=[], metavar="KEY=VALUE", help="PMBConfig overrides")
    return p.parse_args()


def snapshot(step, x, x_est, peds, tracks, pred, info, u, r_col, delta, **_):
    return dict(step=step, x=x, peds={k: v.copy() for k, v in peds.items()},
                tracks={k: (t.x.copy(), getattr(t, "r", 1.0), t.unseen) for k, t in tracks.items()},
                ids=list(pred.ids), mean=pred.mean.copy(), cov=pred.cov.copy(), sel=list(info["selected"]),
                slack=np.array(info["slack"]), kappa=np.atleast_1d(info["kappa"]), plan=info["X"].copy(),
                robot_cov=np.array(info["robot_cov"]), u=np.array(u), r_col=r_col, delta=delta)


def classify(prev, cur, pid, cfg):
    """Cause of a first contact with pid at step cur['step'], from the plan in prev."""
    ev = dict(step=cur["step"], pid=pid)
    if pid not in prev["tracks"]:
        ev["cause"] = "untracked"
        return ev
    tx, r, unseen = prev["tracks"][pid]
    ev.update(r=r, unseen=unseen, speed=float(prev["u"][0]))
    q_prev, q_cur, x_prev = prev["peds"].get(pid), cur["peds"][pid], prev["x"]
    rel = (q_prev if q_prev is not None else q_cur) - x_prev[:2]
    bearing = np.degrees(np.arctan2(rel[1], rel[0]) - x_prev[2])
    bearing = (bearing + 180) % 360 - 180
    ev["approach"] = "front" if abs(bearing) < 60 else ("behind" if abs(bearing) > 120 else "side")
    ev["ped_speed"] = float(np.linalg.norm(q_cur - q_prev) / 0.4) if q_prev is not None else np.nan
    if pid not in prev["sel"]:
        dist = np.linalg.norm(tx[:2] - x_prev[:2])
        why = ("r<=delta" if r <= prev["delta"] else
               "beyond radius" if dist >= cfg.consider_radius else "not in nearest max_peds")
        ev.update(cause="not-constrained", why=why)
        return ev
    slot, j = prev["sel"].index(pid), prev["ids"].index(pid)
    ev["kappa"] = float(prev["kappa"][slot]) if prev["kappa"].size > slot else 0.0
    ev["slack_next"] = float(prev["slack"][slot, 0])
    ev["slack_max"] = float(prev["slack"][slot].max())
    mu, S = prev["mean"][0, j], prev["cov"][0, j]
    err = q_cur - mu
    ev["pred_err"] = float(np.linalg.norm(err))
    ev["d2"] = float(err @ np.linalg.solve(S, err))
    planned = prev["plan"][:2, 1]
    ev["exec_err"] = float(np.linalg.norm(cur["x"][:2] - planned))
    ev["planned_clear"] = float(np.linalg.norm(planned - mu) - prev["r_col"])
    if ev["slack_next"] > 1e-3:
        ev["cause"] = "planned-violation"
    elif ev["d2"] > CHI2_95_2D:
        ev["cause"] = "prediction-miss"
    elif ev["exec_err"] > 0.15:
        ev["cause"] = "execution-error"
    else:
        ev["cause"] = "within-risk"
    return ev


_W = {}


def _init(checkpoint):
    from evaluate_nav import _init_worker

    _init_worker(checkpoint)
    _W.update(__import__("evaluate_nav")._WORKER)


def _job(ep, planner, opts):
    from evaluate_nav import SCENES, _predictor, checkpoint_for, scene_frames
    from robot_nav.mpc import MPCConfig
    from robot_nav.pmb import fit_scene_prior
    from robot_nav.scene import PedestrianScene
    from run_nav import run_episode

    path = SCENES[ep["scene"]]["path"]
    clean = opts["protocol"] == "clean"
    if ep["scene"] not in _W["priors"]:
        first, _, _, split = scene_frames(ep["scene"])
        _W["priors"][ep["scene"]] = fit_scene_prior(path, frame_range=(first, split) if clean else None)
    args = SimpleNamespace(max_steps=90, goal_tol=0.3, tracker="pmb", delta_unseen=opts["delta_unseen"],
                           fov=opts["fov"], cam_range=opts["range"], mpc_overrides=opts["overrides"],
                           pmb_overrides=opts["pmb"])
    recs = []
    seed = opts["seed"] + 1000 * list(SCENES).index(ep["scene"]) + ep["episode"]
    predictor = _predictor(checkpoint_for(ep["scene"], opts["protocol"], opts["checkpoint"]))
    res, _ = run_episode(planner, PedestrianScene(path, ep["frame"]), ep["start"], ep["goal"], predictor,
                         args, seed, _W["priors"][ep["scene"]], hook=lambda **kw: recs.append(snapshot(**kw)))
    cfg, events, in_contact = MPCConfig(), [], set()
    first_tracked = {}
    for i, rec in enumerate(recs):
        for pid in rec["tracks"]:
            first_tracked.setdefault(pid, i)
    for i, (prev, cur) in enumerate(zip(recs, recs[1:])):
        contact = {pid for pid, q in cur["peds"].items() if np.linalg.norm(cur["x"][:2] - q) < cur["r_col"]}
        for pid in sorted(contact - in_contact):
            ev = classify(prev, cur, pid, cfg)
            ev.update(scene=ep["scene"], episode=ep["episode"], planner=planner)
            if pid in first_tracked:
                ev["age_at_contact"] = i + 1 - first_tracked[pid]  # steps tracked before contact
            if ev["cause"] == "planned-violation":
                # Walk back while this pedestrian's constraints needed slack somewhere in the horizon.
                onset = i
                while onset > 0 and pid in recs[onset - 1]["sel"] and \
                        recs[onset - 1]["slack"][recs[onset - 1]["sel"].index(pid)].max() > 1e-3:
                    onset -= 1
                ev["slack_lead"] = i + 1 - onset  # steps between first slack and contact
                ev["age_at_onset"] = onset - first_tracked[pid]
                on = recs[onset]
                ev["dist_at_onset"] = float(np.linalg.norm(on["peds"].get(pid, on["tracks"][pid][0][:2]) - on["x"][:2]))
                ev["speed_at_onset"] = float(on["u"][0])
                # Why was this pedestrian not seen earlier? Status over the 6 steps before first tracking.
                from robot_nav.camera import CameraConfig, visible
                cam = CameraConfig(fov_deg=opts["fov"], max_range=opts["range"])
                hidden = []
                for rec in recs[max(0, first_tracked[pid] - 6): first_tracked[pid]]:
                    if pid not in rec["peds"]:
                        hidden.append("not in scene")
                        continue
                    d = rec["peds"][pid] - rec["x"][:2]
                    head = rec["x"][2] + rec["x"][3]
                    ang = abs((np.degrees(np.arctan2(d[1], d[0]) - head) + 180) % 360 - 180)
                    if np.linalg.norm(d) > cam.max_range:
                        hidden.append("beyond range")
                    elif ang > cam.fov_deg / 2:
                        hidden.append("outside FOV")
                    elif not visible(rec["x"][:2], head, rec["peds"], cam)[pid]:
                        hidden.append("occluded")
                    else:
                        hidden.append("missed (p_D)")
                ev["hidden"] = hidden
            events.append(ev)
        in_contact = contact
    return res, events


def main():
    args = parse_args()
    from evaluate_nav import SCENES, make_episodes

    import ast

    kv = lambda items: {x.split("=", 1)[0]: ast.literal_eval(x.split("=", 1)[1]) for x in items}
    overrides = {"w_slack": args.w_slack} if args.w_slack is not None else {}
    overrides.update(kv(args.set))
    opts = dict(seed=args.seed, delta_unseen=args.delta_unseen, fov=args.fov, range=args.range, overrides=overrides,
                pmb=kv(args.pmb_set), protocol=args.protocol, checkpoint=args.checkpoint)
    scenes = args.scenes or (list(SCENES) if args.protocol == "clean" else ["zara01", "zara02"])
    jobs = [(ep, pl) for s in scenes for ep in make_episodes(s, args.episodes, args.seed, args.protocol, 90)
            for pl in args.planners]
    events, results = [], []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.checkpoint,)) as pool:
        futs = [pool.submit(_job, ep, pl, opts) for ep, pl in jobs]
        for f in as_completed(futs):
            res, evs = f.result()
            results.append(res)
            events += evs
    tracked = [e for e in events if e["cause"] != "untracked"]
    print(f"{len(jobs)} episodes run; first contacts: {len(events)} "
          f"({len(tracked)} with tracked pedestrians, {len(events) - len(tracked)} untracked)")
    for pl in args.planners:
        evs = [e for e in tracked if e["planner"] == pl]
        rs = [r for r in results if r["planner"] == pl]
        print(f"\n=== {pl}: {len(evs)} tracked-pedestrian contacts | success {np.mean([r['success'] for r in rs]):.2f}  "
              f"collision-free {np.mean([r['collisions'] == 0 for r in rs]):.2f}  "
              f"reached {np.mean([r['reached'] for r in rs]):.2f}  "
              f"steps(reached) {np.mean([r['steps'] for r in rs if r['reached']]):.1f} ===")
        for cause, n in Counter(e["cause"] for e in evs).most_common():
            sub = [e for e in evs if e["cause"] == cause]
            extra = ""
            if cause == "not-constrained":
                extra = "  why: " + ", ".join(f"{w} {c}" for w, c in Counter(e["why"] for e in sub).most_common())
            print(f"  {cause:18s} {n:3d}{extra}")
        if evs:
            print("  approach: " + ", ".join(f"{a} {c}" for a, c in Counter(e["approach"] for e in evs).most_common())
                  + f"   robot speed at contact-1: median {np.median([e['speed'] for e in evs]):.2f} m/s, "
                  f"stopped(<0.1) {sum(e['speed'] < 0.1 for e in evs)}")
            con = [e for e in evs if "d2" in e]
            if con:
                print(f"  constrained contacts: median one-step prediction error {np.median([e['pred_err'] for e in con]):.2f} m, "
                      f"median d2 {np.median([e['d2'] for e in con]):.1f} (95% = 5.99), "
                      f"median kappa {np.median([e['kappa'] for e in con]):.2f}, "
                      f"slack used anywhere in horizon {sum(e['slack_max'] > 1e-3 for e in con)}/{len(con)}, "
                      f"median existence r {np.median([e['r'] for e in con]):.2f}")
        pv = [e for e in evs if e["cause"] == "planned-violation"]
        if pv:
            age = np.array([e["age_at_onset"] for e in pv])
            print(f"  planned violations: steps tracked when slack first appeared: median {np.median(age):.0f} "
                  f"(<=2 steps: {np.sum(age <= 2)}/{len(pv)}, >=5 steps: {np.sum(age >= 5)}/{len(pv)}); "
                  f"slack appeared {np.median([e['slack_lead'] for e in pv]):.0f} steps before contact (median); "
                  f"distance then {np.median([e['dist_at_onset'] for e in pv]):.1f} m; "
                  f"robot speed then {np.median([e['speed_at_onset'] for e in pv]):.2f} m/s; "
                  f"pedestrian speed {np.nanmedian([e['ped_speed'] for e in pv]):.2f} m/s")
            late = [e for e in pv if e["age_at_onset"] <= 2 and e.get("hidden")]
            last = Counter(e["hidden"][-1] for e in late)
            allsteps = Counter(h for e in late for h in e["hidden"])
            print(f"  late detections ({len(late)}): status the step before first detection: "
                  + ", ".join(f"{k} {v}" for k, v in last.most_common())
                  + "  |  over the 6 steps before: " + ", ".join(f"{k} {v}" for k, v in allsteps.most_common()))
    print("\nExamples:")
    for cause in ("planned-violation", "prediction-miss", "within-risk", "not-constrained", "execution-error"):
        for e in [e for e in tracked if e["cause"] == cause][: args.examples]:
            keys = ["scene", "episode", "planner", "step", "pid", "approach", "speed", "ped_speed", "r", "unseen", "kappa",
                    "slack_next", "slack_max", "pred_err", "d2", "planned_clear", "exec_err", "why"]
            print(f"  {cause:18s} " + " ".join(f"{k}={e[k]:.2f}" if isinstance(e.get(k), float) else f"{k}={e[k]}"
                                               for k in keys if k in e))


if __name__ == "__main__":
    main()
