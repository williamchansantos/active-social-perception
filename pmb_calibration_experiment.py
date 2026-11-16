"""Check the PMB tracker's calibration on replayed Zara crowds.

A robot crosses the scene on a straight line (no planner, so the test isolates
the tracker) with the camera either fixed forward or scanning. Each step it
compares the tracker's beliefs with ground truth:

  undetected intensity   expected number of never-detected pedestrians (whole
                         scene and within 5 m of the robot) vs the actual number
                         of present pedestrians with no Bernoulli track;
                         predictive NLL (Poisson, or negative-binomial with the
                         adaptive crowd rate) per region and jointly over 1 m
                         blocks vs the static prior (no sensing updates), and
                         broken down by how many pedestrians are tracked
  existence              Bernoulli r vs whether the pedestrian is still in the
                         scene (reliability bins and Brier score)
  first-sight velocity   velocity prior of new tracks (PPP channel mixture) vs
                         a zero-mean prior: error and NEES against true velocity

Example:
    python pmb_calibration_experiment.py --episodes 40
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from scipy.special import gammaln

from robot_nav.camera import CameraConfig, observe
from robot_nav.pmb import PMBConfig, PMBTracker, fit_scene_prior
from robot_nav.robot import RobotConfig, dynamics
from robot_nav.scene import PedestrianScene

NEAR_RADIUS = 5.0
BLOCK = 4  # cells per side of a 1 m evaluation block (0.25 m cells)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="data/ucy/zara01/obsmat.txt")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--speed", type=float, default=0.5, help="robot speed (m/s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-dir", default="plots/pmb")
    return p.parse_args()


def poisson_nll(counts, lam):
    lam = np.maximum(lam, 1e-9)
    return lam - counts * np.log(lam) + gammaln(counts + 1)


def blocks(a):
    nx, ny = (a.shape[0] // BLOCK) * BLOCK, (a.shape[1] // BLOCK) * BLOCK
    return a[:nx, :ny].reshape(nx // BLOCK, BLOCK, ny // BLOCK, BLOCK).sum((1, 3))


def true_velocity(scene, step, pid):
    nxt = scene.positions(step + 1)
    cur = scene.positions(step)
    if pid in nxt and pid in cur:
        return nxt[pid] - cur[pid]
    return None


def run_episode(prior, scene, camera_mode, adapt, start, goal, args, rng, snapshots=None):
    cam, rcfg = CameraConfig(), RobotConfig()
    tracker = PMBTracker(prior, cam, PMBConfig(adapt_rate=adapt))
    Sigma0 = np.diag([0.05 ** 2, 0.05 ** 2, 0.02 ** 2, 0.005 ** 2])
    L0 = np.linalg.cholesky(Sigma0)
    d = goal - start
    x = np.array([*start, np.arctan2(d[1], d[0]), 0.0])
    grid = prior.grid
    rec = dict(E_all=[], N_all=[], E_near=[], N_near=[], nll_all=[], nll_near=[], nll_pmb=[], nll_static=[],
               r=[], exists=[], n_tracked=[], rate=[], v_err_ppp=[], v_err_zero=[], nees_ppp=[], nees_zero=[])
    static_blocks = blocks(prior.occupancy.sum(0))
    for step in range(args.steps):
        peds = {pid: q for pid, q in scene.positions(step).items() if grid.contains(q)}
        x_est = x + L0 @ rng.standard_normal(4)
        det = observe(x, x_est, Sigma0, peds, cam, rng)
        before = set(tracker.tracks)
        # Existence check uses the prediction step's r, before this step's detections.
        tracker.step(det, x_est)

        for pid, tr in tracker.tracks.items():
            if pid in det:
                continue
            rec["r"].append(tr.r)
            rec["exists"].append(pid in peds)
        for pid in set(tracker.tracks) - before:
            v = true_velocity(scene, step, pid)
            if v is None:
                continue
            tr = tracker.tracks[pid]
            # The tracker's velocity right after creation is the PPP prior (a single
            # position fix carries no velocity information).
            for key, mean, cov in (("ppp", tr.x[2:], tr.P[2:, 2:]),
                                   ("zero", np.zeros(2), PMBConfig().v0_std ** 2 * np.eye(2))):
                e = v - mean
                rec[f"v_err_{key}"].append(np.linalg.norm(e))
                rec[f"nees_{key}"].append(e @ np.linalg.solve(cov, e))

        undetected = [q for pid, q in peds.items() if pid not in tracker.tracks]
        dens, base = tracker.undetected_density(), tracker.D.sum(0)
        near = np.linalg.norm(tracker.centers - x[:2], axis=-1) <= NEAR_RADIUS
        n_near = sum(np.linalg.norm(q - x[:2]) <= NEAR_RADIUS for q in undetected)
        rec["E_all"].append(dens.sum())
        rec["N_all"].append(len(undetected))
        rec["E_near"].append(dens[near].sum())
        rec["N_near"].append(n_near)
        rec["nll_all"].append(-tracker.count_logpmf(np.array([len(undetected)]), np.array([base.sum()])))
        rec["nll_near"].append(-tracker.count_logpmf(np.array([n_near]), np.array([base[near].sum()])))
        counts = blocks(grid.hist(np.array(undetected).reshape(-1, 2)))
        rec["nll_pmb"].append(-tracker.count_logpmf(counts, blocks(base)))
        rec["nll_static"].append(poisson_nll(counts, static_blocks).sum())
        rec["n_tracked"].append(len(tracker.tracks))
        rec["rate"].append(tracker.rate_mean)
        if snapshots is not None and step in (5, 20, 35):
            snapshots.append((step, dens.copy(), x.copy(), peds, set(tracker.tracks), dict(det)))

        if camera_mode == "scan":
            phi_next = np.sin(2 * np.pi * (step + 1) * rcfg.dt / 4.0)
            u = np.array([args.speed, 0.0, np.clip((phi_next - x[3]) / rcfg.dt, -2.5, 2.5)])
        else:
            u = np.array([args.speed, 0.0, -x[3] / rcfg.dt])
        x = dynamics(x, u, rcfg.p_nominal, rcfg.dt)
    return rec


def summarize(name, recs):
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    print(f"\n[{name}]")
    for region in ("all", "near"):
        E, N = cat[f"E_{region}"], cat[f"N_{region}"]
        label = "whole scene" if region == "all" else f"within {NEAR_RADIUS:g} m"
        print(f"  undetected {label:12s}: expected {E.mean():.2f}  actual {N.mean():.2f}  "
              f"corr {np.corrcoef(E, N)[0, 1]:.2f}  predictive NLL {cat[f'nll_{region}'].mean():.3f}")
    print(f"  1 m-block joint NLL per step: PMB {cat['nll_pmb'].mean():.2f}  vs static prior "
          f"{cat['nll_static'].mean():.2f}")
    nt, E, N = cat["n_tracked"], cat["E_all"], cat["N_all"]
    parts = []
    for lo, hi in [(0, 2), (2, 4), (4, 7), (7, 100)]:
        m = (nt >= lo) & (nt < hi)
        if m.sum():
            parts.append(f"[{lo},{hi if hi < 100 else 'inf'}): exp {E[m].mean():.2f} act {N[m].mean():.2f} "
                         f"(n={m.sum()})")
    print(f"  whole-scene undetected by #tracked (mean crowd multiplier {cat['rate'].mean():.2f}): "
          + " | ".join(parts))
    E, N = cat["E_near"], cat["N_near"]
    edges = [0, 0.25, 0.5, 1, 2, np.inf]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (E >= lo) & (E < hi)
        if m.sum() >= 5:
            parts.append(f"E in [{lo:g},{hi:g}): pred {E[m].mean():.2f} act {N[m].mean():.2f} (n={m.sum()})")
    print("  near-robot reliability: " + " | ".join(parts))
    r, ex = cat["r"], cat["exists"].astype(float)
    if len(r):
        bins = [0, 0.3, 0.6, 0.9, 0.999, 1.001]
        parts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (r >= lo) & (r < hi)
            if m.sum() >= 5:
                parts.append(f"r in [{lo:g},{min(hi, 1):g}{')' if hi < 1 else ']'}: mean r {r[m].mean():.2f} "
                             f"exists {ex[m].mean():.2f} (n={m.sum()})")
        print(f"  existence (missed/unseen tracks) Brier {np.mean((r - ex) ** 2):.3f}: " + " | ".join(parts))
    for key in ("ppp", "zero"):
        print(f"  first-sight velocity prior [{key:4s}]: mean |err| {cat[f'v_err_{key}'].mean():.3f} m/frame  "
              f"NEES {cat[f'nees_{key}'].mean():.2f} (ideal 2)  n={len(cat[f'v_err_{key}'])}")
    return cat


def plot(prior, snapshots, cats, path):
    cam, g = CameraConfig(), prior.grid
    extent = [g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res]
    fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(5 * (len(snapshots) + 1), 5))
    for ax, (step, dens, x, peds, tracked, det) in zip(axes, snapshots):
        ax.imshow(dens.T, origin="lower", extent=extent, cmap="magma_r", vmin=0, vmax=0.02)
        heading = np.degrees(x[2] + x[3])
        ax.add_patch(Wedge(x[:2], cam.max_range, heading - cam.fov_deg / 2, heading + cam.fov_deg / 2,
                           fill=False, color="tab:orange"))
        for pid, q in peds.items():
            c = "tab:green" if pid in det else ("tab:blue" if pid in tracked else "tab:red")
            ax.plot(*q, "o", color=c, ms=5)
        ax.plot(*x[:2], "s", color="tab:orange", ms=8)
        ax.set_title(f"step {step}: undetected intensity\ngreen=detected blue=tracked red=undetected", fontsize=9)
    ax = axes[-1]
    for name, cat in cats.items():
        E, N = cat["E_near"], cat["N_near"]
        edges = np.quantile(E, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(E, edges[1:-1]), 0, 7)
        ax.plot([E[idx == i].mean() for i in range(8)], [N[idx == i].mean() for i in range(8)], "-o", label=name)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set(xlabel=f"expected undetected within {NEAR_RADIUS:g} m", ylabel="actual (mean per bin)",
           title="undetected-count reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nSaved {path}")


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)
    prior = fit_scene_prior(args.scene)
    rng_ep = np.random.default_rng(args.seed)
    episodes = []
    for _ in range(args.episodes):
        y = rng_ep.uniform(9, 17)
        flip = rng_ep.uniform() < 0.5
        start, goal = np.array([-7.0, y]), np.array([5.0, y])
        if flip:
            start, goal = goal, start
        episodes.append((int(rng_ep.integers(100, 8500)), start, goal))
    print(f"{args.episodes} episodes x {args.steps} steps, prior occupancy {prior.occupancy.sum():.2f} peds, "
          f"births {prior.birth.sum():.3f}/frame")
    cats, snapshots = {}, []
    for mode in ("fixed", "scan"):
        for adapt in (False, True):
            name = f"{mode}, {'adaptive' if adapt else 'fixed'} rate"
            recs = []
            for i, (frame, start, goal) in enumerate(episodes):
                rng = np.random.default_rng(args.seed + 1000 + i)
                snap = snapshots if (mode == "scan" and adapt and i == 0) else None
                recs.append(run_episode(prior, PedestrianScene(args.scene, frame), mode, adapt, start, goal, args,
                                        rng, snap))
            cats[name] = summarize(f"camera {name}", recs)
    plot(prior, snapshots, cats, os.path.join(args.plot_dir, "calibration.png"))


if __name__ == "__main__":
    main()
