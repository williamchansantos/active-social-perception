"""Our KDE birth intensity in the format the lab's PMB tracker expects.

The tracker (`robot_nav/pmb.py`) reads births from `ScenePrior.birth`: an array of
shape (5, nx, ny) holding the expected number of new pedestrians per grid cell per
annotation frame, one layer per direction channel (+x, -x, +y, -y, static).
`fit_scene_prior` fills it with a smoothed histogram. This module builds the same
array from our KDE model instead, and `fit_scene_prior_kde` is a drop-in
replacement for `fit_scene_prior` (same arguments) that swaps only that array.

What is different from the lab's birth map:

  1. Bandwidth per scene. One Gaussian width for the scene, chosen by
     leave-one-out likelihood on the fitting events (or given with `bandwidth=`),
     instead of a fixed 2 cells = 0.5 m.
  2. Left-censoring. A pedestrian already present in the first frame of the
     recording did not arrive there, so it is not counted as a birth. (The lab's
     fit counts every pedestrian's first position in the fitted frames.)
  3. Floor. `uniform_mix` (default 5 %) of the birth rate is spread evenly over
     the walkable cells, so no walkable cell has a birth rate of exactly zero.

What is kept identical, so the comparison isolates the birth map:
  grid, occupancy, velocities, walkable mask, exit probabilities, crowd-multiplier
  parameters, direction channels (from the obsmat velocity at the first frame),
  and the per-frame normalisation (number of annotation steps fitted).

Assumptions to flag:
  * Each kernel's mass is integrated exactly over every cell (erf), not sampled
    at the cell centre, because the bandwidth can be smaller than a cell.
  * Kernel mass falling outside the grid is renormalised back inside (as in
    `KDEBirthModel`); mass on non-walkable cells is removed by the tracker's own
    walkable mask, exactly as for the lab's map.
  * The tracker's survival map is calibrated so that, with nobody observed, the
    undetected intensity settles at the observed occupancy. That calibration
    depends on the birth map, so it is repeated for ours (`recalibrate=True`).

Needs scipy (through robot_nav.pmb).

Example (prints a side-by-side summary of the two birth maps):
    python -m social_lstm.birth.tracker_birth --data data/ucy/zara01
"""
from __future__ import annotations

import argparse
import copy
from typing import Optional, Tuple

import numpy as np

from robot_nav.pmb import C, ScenePrior, channel_mixing, channel_of, fit_scene_prior, ppp_predict
from social_lstm.birth.appearance import DEFAULT_DT, DEFAULT_SCENES, resolve_obsmat, scene_name_from_path
from social_lstm.birth.compare_lab_birth import loo_bandwidth
from social_lstm.birth.poisson_birth import _gaussian_mass_1d

FrameRange = Optional[Tuple[float, float]]


def birth_events(obsmat_path: str, frame_range: FrameRange = None, exclude_range: FrameRange = None,
                 dt: float = DEFAULT_DT, exclude_initial: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Birth events inside the fitted frames, selected like `fit_scene_prior` selects frames.

    Returns (positions (N, 2), channel (N,), n_steps): the first position of every
    pedestrian whose first annotated frame is one of the fitted frames, its direction
    channel, and the number of annotation steps fitted (the lab's normalisation).
    With `exclude_initial`, pedestrians present in the recording's first frame are
    dropped (left-censoring). A pedestrian who is merely already walking when a
    fitted window starts is never a birth here, because their true first frame is
    outside the window.
    """
    d_all = np.loadtxt(obsmat_path)
    f_all = d_all[:, 0]
    keep = np.ones(len(d_all), bool)
    if frame_range is not None:
        keep &= (f_all >= frame_range[0]) & (f_all < frame_range[1])
    if exclude_range is not None:
        keep &= (f_all < exclude_range[0]) | (f_all >= exclude_range[1])

    # number of annotation steps fitted: same definition as fit_scene_prior
    frame_step = np.median(np.diff(np.unique(f_all)))
    kept_frames = np.unique(f_all[keep])
    n_steps = (kept_frames.max() - kept_frames.min()) / frame_step + 1
    if exclude_range is not None:
        grid_f = np.arange(f_all.min(), f_all.max() + 0.5 * frame_step, frame_step)
        n_steps = int(np.sum((grid_f < exclude_range[0]) | (grid_f >= exclude_range[1])))

    first_rows = np.array([rows[np.argmin(f_all[rows])] for rows in
                           (np.flatnonzero(d_all[:, 1] == pid) for pid in np.unique(d_all[:, 1]))])
    sel = first_rows[keep[first_rows]]
    if exclude_initial:
        sel = sel[f_all[sel] > f_all.min()]
    pos = d_all[sel][:, [2, 4]]
    ch = channel_of(d_all[sel][:, [5, 7]] * dt)  # m/s -> m/frame, as in fit_scene_prior
    return pos, np.asarray(ch), float(n_steps)


def kde_birth_array(positions: np.ndarray, channels: np.ndarray, n_steps: float, prior: ScenePrior,
                    bandwidth: float, uniform_mix: float = 0.05) -> np.ndarray:
    """(5, nx, ny) expected births per cell per frame from Gaussian kernels of width
    `bandwidth` [m] at `positions`, plus a uniform floor over the walkable cells.
    The total equals len(positions) / n_steps before the tracker's walkable mask."""
    g = prior.grid
    x_edges = g.x0 + g.res * np.arange(g.nx + 1)
    y_edges = g.y0 + g.res * np.arange(g.ny + 1)
    # mass of every kernel in every column / row of cells (the 2D Gaussian is separable)
    mx = _gaussian_mass_1d(x_edges[None, :-1], x_edges[None, 1:], positions[:, 0:1], bandwidth)  # (N, nx)
    my = _gaussian_mass_1d(y_edges[None, :-1], y_edges[None, 1:], positions[:, 1:2], bandwidth)  # (N, ny)
    mx = mx / mx.sum(axis=1, keepdims=True)  # renormalise mass that fell outside the grid
    my = my / my.sum(axis=1, keepdims=True)
    floor = prior.walkable / max(prior.walkable.sum(), 1)  # unit mass over walkable cells
    birth = np.zeros((C, g.nx, g.ny))
    for c in range(C):
        m = channels == c
        kde = mx[m].T @ my[m]  # sum of the channel's kernels, (nx, ny), mass = m.sum()
        birth[c] = ((1.0 - uniform_mix) * kde + uniform_mix * m.sum() * floor) / n_steps
    return birth


def recalibrate_survival(prior: ScenePrior, channel_switch: float = 0.02) -> float:
    """Repeat the calibration at the end of `fit_scene_prior` for the prior's current
    birth map: scale the exit probability so that the predict-only intensity stays at
    the observed occupancy. Sets `prior.survival` in place and returns the scale."""
    mix, target = channel_mixing(channel_switch), prior.occupancy.sum()
    lo, hi = 0.0, 1.0
    for _ in range(12):
        alpha = 0.5 * (lo + hi)
        prior.survival = 1 - alpha * prior.exit_prob
        D = prior.occupancy.copy()
        for _ in range(100):
            D = ppp_predict(D, prior, mix)
        lo, hi = (alpha, hi) if D.sum() > target else (lo, alpha)
    prior.survival = 1 - alpha * prior.exit_prob
    return alpha


def with_kde_birth(prior: ScenePrior, obsmat_path: str, frame_range: FrameRange = None,
                   exclude_range: FrameRange = None, dt: float = DEFAULT_DT, bandwidth: Optional[float] = None,
                   uniform_mix: float = 0.05, exclude_initial: bool = True, recalibrate: bool = True,
                   channel_switch: float = 0.02) -> ScenePrior:
    """Copy of `prior` whose birth map is our KDE (the input prior is not modified).
    `bandwidth=None` picks it by leave-one-out likelihood on the fitting events. The
    chosen value is stored on the result as `birth_bandwidth`."""
    pos, ch, n_steps = birth_events(obsmat_path, frame_range, exclude_range, dt, exclude_initial)
    g = prior.grid
    if bandwidth is None:
        bounds = (g.x0, g.x0 + g.nx * g.res, g.y0, g.y0 + g.ny * g.res)
        bandwidth, _ = loo_bandwidth(pos, n_steps * dt, bounds, uniform_mix)
    out = copy.copy(prior)
    out.birth = kde_birth_array(pos, ch, n_steps, prior, bandwidth, uniform_mix) * prior.walkable
    out.survival = prior.survival.copy()
    if recalibrate:
        recalibrate_survival(out, channel_switch)
    out.birth_bandwidth = float(bandwidth)
    return out


def fit_scene_prior_kde(obsmat_path: str, res: float = 0.25, margin: float = 1.0, smooth_cells: float = 2.0,
                        channel_switch: float = 0.02, frame_range: FrameRange = None, dt: float = DEFAULT_DT,
                        exclude_range: FrameRange = None, bandwidth: Optional[float] = None,
                        uniform_mix: float = 0.05, exclude_initial: bool = True,
                        recalibrate: bool = True) -> ScenePrior:
    """Drop-in replacement for `robot_nav.pmb.fit_scene_prior`: the lab's prior with
    only the birth map (and the survival calibration that depends on it) replaced."""
    prior = fit_scene_prior(obsmat_path, res=res, margin=margin, smooth_cells=smooth_cells,
                            channel_switch=channel_switch, frame_range=frame_range, dt=dt,
                            exclude_range=exclude_range)
    return with_kde_birth(prior, obsmat_path, frame_range, exclude_range, dt, bandwidth, uniform_mix,
                          exclude_initial, recalibrate, channel_switch)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=list(DEFAULT_SCENES), help="scene directories or obsmat.txt files")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--fit-frac", type=float, default=0.5, help="fit on this first fraction of the frames (lab protocol: 0.5)")
    p.add_argument("--bandwidth", type=float, default=None, help="KDE bandwidth [m]; default: leave-one-out per scene")
    p.add_argument("--uniform-mix", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    names = ["+x", "-x", "+y", "-y", "static"]
    for path in args.data:
        obsmat = resolve_obsmat(path)
        frames = np.unique(np.loadtxt(obsmat)[:, 0])
        fr = (frames[0], frames[int(args.fit_frac * len(frames))]) if args.fit_frac < 1 else None
        lab = fit_scene_prior(obsmat, frame_range=fr, dt=args.dt)
        ours = with_kde_birth(lab, obsmat, frame_range=fr, dt=args.dt, bandwidth=args.bandwidth,
                              uniform_mix=args.uniform_mix)
        mix = channel_mixing(0.02)
        steady = {}
        for tag, p in (("lab", lab), ("ours", ours)):
            D = p.occupancy.copy()
            for _ in range(300):
                D = ppp_predict(D, p, mix)
            steady[tag] = D.sum()
        walk = lab.walkable
        print(f"\n{scene_name_from_path(obsmat)}  grid {lab.grid.nx} x {lab.grid.ny}, bandwidth {ours.birth_bandwidth:g} m")
        print(f"  {'':28s} {'lab':>9s} {'ours':>9s}")
        print(f"  {'births per frame (total)':28s} {lab.birth.sum():9.4f} {ours.birth.sum():9.4f}")
        for c in range(C):
            print(f"  {'  channel ' + names[c]:28s} {lab.birth[c].sum():9.4f} {ours.birth[c].sum():9.4f}")
        print(f"  {'walkable cells with 0 birth':28s} {int(np.sum(lab.birth.sum(0)[walk] == 0)):9d} "
              f"{int(np.sum(ours.birth.sum(0)[walk] == 0)):9d}")
        print(f"  {'peak cell (births/frame)':28s} {lab.birth.sum(0).max():9.5f} {ours.birth.sum(0).max():9.5f}")
        print(f"  {'mean survival (walkable)':28s} {lab.survival[walk].mean():9.4f} {ours.survival[walk].mean():9.4f}")
        print(f"  {'unseen total, never looking':28s} {steady['lab']:9.3f} {steady['ours']:9.3f}   "
              f"(occupancy {lab.occupancy.sum():.3f})")


if __name__ == "__main__":
    main()
