"""Compare our birth intensities with the lab tracker's birth map on held-out time.

The lab's PMB tracker (`robot_nav/pmb.py`) already has a birth intensity:
`ScenePrior.birth`, a histogram of every pedestrian's first position on a 0.25 m
grid, smoothed with a Gaussian of 2 cells (0.5 m) and masked to the walkable
area. This script scores that map with the same held-out protocol as
`evaluate_birth.py`, next to our models, so the numbers are directly comparable.

For every scene (first --train-frac of the recording to fit, the rest held out):

    uniform    same scalar rate everywhere (baseline)
    grid       our 0.5 m histogram (unsmoothed, 5 % uniform floor)
    lab        the lab's map exactly as the tracker uses it
    lab+floor  the lab's map with the same 5 % uniform floor as our models
    kde0.5     our KDE with the default 0.5 m bandwidth
    kde-cv     our KDE with the bandwidth chosen per scene by leave-one-out
               likelihood on the TRAINING events only (no held-out data used)

All models share the lab grid's bounding box, so "per m^2" means the same thing
for each. Reported per model:

    pred_n   expected held-out arrivals (compare with test_n)
    sp/evt   mean log spatial density of the held-out events ("where", not "how
             many"); higher is better; exp(difference) is the likelihood ratio
    n_zero   held-out arrivals that the model says are impossible (intensity 0)
    ll       full Poisson point-process log-likelihood

Two differences from the lab's fit are visible here and are flagged as assumptions:
  * the lab counts pedestrians already present in the first training frame as
    births (left-censoring); our models drop them, so pred_n differs;
  * the lab's map is exactly zero outside the walkable mask and far from any
    training birth, so `lab` can score -inf; `lab+floor` removes that effect.

Needs scipy (imported by robot_nav.pmb); everything else is numpy only.

Example:
    python -m social_lstm.birth.compare_lab_birth
    python -m social_lstm.birth.compare_lab_birth --data data/ucy/zara01
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from robot_nav.pmb import fit_scene_prior
from social_lstm.birth.appearance import DEFAULT_DT, DEFAULT_SCENES, SceneAppearances, load_scenes
from social_lstm.birth.evaluate_birth import split_by_time
from social_lstm.birth.poisson_birth import Bounds, GridBirthModel, KDEBirthModel

BANDWIDTHS = (0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=list(DEFAULT_SCENES), help="scene directories or obsmat.txt files")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="seconds per annotated frame step")
    p.add_argument("--cell", type=float, default=0.5, help="cell size of our histogram model [m]")
    p.add_argument("--bandwidth", type=float, default=0.5, help="default KDE bandwidth [m]")
    p.add_argument("--train-frac", type=float, default=0.8, help="fraction of the recording (by time) used to fit")
    p.add_argument("--uniform-mix", type=float, default=0.05, help="fraction of the rate spread uniformly (floor)")
    p.add_argument("--lab-res", type=float, default=0.25, help="lab grid resolution [m] (tracker default)")
    p.add_argument("--lab-smooth-cells", type=float, default=2.0, help="lab Gaussian smoothing [cells] (tracker default)")
    p.add_argument("--out", default="plots/birth/lab_comparison.csv")
    return p.parse_args()


def lab_birth_model(scene: SceneAppearances, t_split: float, res: float, smooth_cells: float) -> GridBirthModel:
    """The lab's `ScenePrior.birth`, fitted on frames before `t_split` seconds, wrapped
    as a `GridBirthModel` (arrivals per second per cell, direction channels summed)."""
    split_frame = scene.first_frame + t_split / scene.dt * scene.frame_step
    prior = fit_scene_prior(scene.path, res=res, smooth_cells=smooth_cells,
                            frame_range=(scene.first_frame, split_frame), dt=scene.dt)
    g = prior.grid
    return GridBirthModel(
        x_edges=g.x0 + g.res * np.arange(g.nx + 1),
        y_edges=g.y0 + g.res * np.arange(g.ny + 1),
        rate=prior.birth.sum(axis=0) / scene.dt,  # per frame -> per second
    )


def with_floor(model: GridBirthModel, uniform_mix: float) -> GridBirthModel:
    """Same total rate, with `uniform_mix` of it spread evenly over all cells."""
    uniform = np.full_like(model.rate, model.total_rate / model.rate.size)
    return GridBirthModel(model.x_edges, model.y_edges, (1.0 - uniform_mix) * model.rate + uniform_mix * uniform)


def loo_bandwidth(train: np.ndarray, duration: float, bounds: Bounds, uniform_mix: float,
                  candidates: Sequence[float] = BANDWIDTHS) -> Tuple[float, Dict[float, float]]:
    """Pick the KDE bandwidth by leave-one-out likelihood on the training events:
    each event is scored by the density built from all the *other* events. Returns
    the best bandwidth and the mean leave-one-out log-density of every candidate."""
    n = len(train)
    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    d2 = ((train[:, None, :] - train[None, :, :]) ** 2).sum(axis=-1)
    scores = {}
    for h in candidates:
        mass = KDEBirthModel.fit(train, duration, bounds, bandwidth=h)._kernel_mass(bounds)  # truncation
        k = np.exp(-0.5 * d2 / h**2) / (2.0 * np.pi * h**2) / mass[None, :]
        np.fill_diagonal(k, 0.0)
        dens = (1.0 - uniform_mix) * k.sum(axis=1) / (n - 1) + uniform_mix / area
        scores[h] = float(np.mean(np.log(dens)))
    return max(scores, key=scores.get), scores


def evaluate_scene(scene: SceneAppearances, args) -> Tuple[List[dict], dict]:
    train, test, t_train, t_test = split_by_time(scene, args.train_frac)
    lab = lab_birth_model(scene, t_train, args.lab_res, args.lab_smooth_cells)
    bounds = lab.bounds  # shared by all models
    grid = GridBirthModel.fit(train, t_train, bounds, cell=args.cell, uniform_mix=args.uniform_mix)
    h_cv, cv_scores = loo_bandwidth(train, t_train, bounds, args.uniform_mix)
    models = {
        "uniform": GridBirthModel.uniform(grid.total_rate, bounds, cell=args.cell),
        "grid": grid,
        "lab": lab,
        "lab+floor": with_floor(lab, args.uniform_mix),
        f"kde{args.bandwidth:g}": KDEBirthModel.fit(train, t_train, bounds, args.bandwidth, args.uniform_mix),
        "kde-cv": KDEBirthModel.fit(train, t_train, bounds, h_cv, args.uniform_mix),
    }
    rows = []
    for name, m in models.items():
        with np.errstate(divide="ignore"):
            logd = np.log(m.intensity(test) / m.total_rate)
            ll = float(np.log(m.intensity(test)).sum() - t_test * m.total_rate)
        rows.append({
            "scene": scene.name, "model": name, "train_n": len(train), "test_n": len(test),
            "pred_n": m.expected_count(t_test), "spatial_per_event": float(logd.mean()),
            "n_zero": int(np.sum(~np.isfinite(logd))), "ll": ll,
            "bandwidth": h_cv if name == "kde-cv" else (args.bandwidth if name.startswith("kde") else float("nan")),
        })
    return rows, {"h_cv": h_cv, "cv_scores": cv_scores, "logd": {
        name: np.log(np.maximum(m.intensity(test) / m.total_rate, 1e-300)) for name, m in models.items()}}


def format_table(rows: List[dict]) -> str:
    head = f"{'scene':<12s} {'model':<10s} {'train_n':>7s} {'test_n':>6s} {'pred_n':>6s} {'sp/evt':>7s} {'n_zero':>6s} {'ll':>9s}"
    lines, prev = [head, "-" * len(head)], None
    for r in rows:
        if prev is not None and r["scene"] != prev:
            lines.append("")
        prev = r["scene"]
        lines.append(f"{r['scene']:<12s} {r['model']:<10s} {r['train_n']:7d} {r['test_n']:6d} {r['pred_n']:6.1f} "
                     f"{r['spatial_per_event']:7.2f} {r['n_zero']:6d} {r['ll']:9.1f}")
    return "\n".join(lines)


def main():
    args = parse_args()
    rows: List[dict] = []
    pooled: Dict[str, List[np.ndarray]] = {}
    chosen = []
    for scene in load_scenes(args.data, dt=args.dt):
        scene_rows, ex = evaluate_scene(scene, args)
        rows.extend(scene_rows)
        chosen.append((scene.name, ex["h_cv"]))
        for name, v in ex["logd"].items():
            pooled.setdefault(name, []).append(v)

    print(format_table(rows))
    print("\nleave-one-out bandwidth per scene: " + ", ".join(f"{n} {h:g} m" for n, h in chosen))

    n_total = sum(len(v) for v in next(iter(pooled.values())))
    print(f"\nPooled over {n_total} held-out arrivals (mean log spatial density; ratio = per-event likelihood ratio):")
    ref = np.concatenate(pooled["lab+floor"]).mean()
    for name, parts in pooled.items():
        v = np.concatenate(parts)
        zero = int(np.sum(v < -600))
        mean = "   -inf" if zero else f"{v.mean():7.3f}"
        ratio = "      -" if zero else f"{np.exp(v.mean() - ref):7.2f}"
        print(f"  {name:<10s} {mean}   ratio vs lab+floor {ratio}   impossible events {zero}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
