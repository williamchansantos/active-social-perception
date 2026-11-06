"""Poisson birth process with a grid-based spatial intensity.

The model has two parts, matching the PMBM "Poisson birth" component:

    * a scalar arrival rate  lam  [arrivals / s]  for the whole scene, and
    * a spatial distribution of where arrivals happen, stored as a rate per
      grid cell  rate[i, j]  [arrivals / s]  with  sum(rate) == lam.

Together they define an intensity  lambda(x, y) = rate[cell(x, y)] / cell_area
[arrivals / s / m^2]. The two things the planner needs follow directly:

    expected births in region R during T seconds  =  T * sum over cells of
                                                     rate * (overlap of cell with R)
    sampling births for T seconds                 =  N ~ Poisson(lam * T), then
                                                     N cells drawn with prob. rate / lam,
                                                     uniform position inside each cell

Fitting is counting: rate[i, j] = (arrivals that started in cell (i, j)) / duration.
To keep held-out events from having zero probability, a fraction `uniform_mix`
of the total rate is spread uniformly over all cells (see `GridBirthModel.fit`).

Held-out scoring uses the Poisson point-process log-likelihood of the observed
appearance events:

    log L = sum_i log lambda(x_i, y_i)  -  T * lam

The first term rewards intensity where arrivals actually happened; the second
term penalises intensity in total, so the model cannot win by being large
everywhere. A `GridBirthModel.uniform(...)` with the same lam is the baseline:
it has the same second term and only differs in where it puts the mass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Bounds = Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max


@dataclass
class BirthSample:
    """One sampled realisation of births over a time interval."""

    times: np.ndarray  # (N,) seconds from the start of the interval, sorted
    positions: np.ndarray  # (N, 2) x, y in meters


@dataclass
class GridBirthModel:
    """Homogeneous-in-time Poisson birth process with a piecewise-constant spatial intensity."""

    x_edges: np.ndarray  # (nx + 1,) cell boundaries in meters
    y_edges: np.ndarray  # (ny + 1,)
    rate: np.ndarray  # (nx, ny) arrivals per second per cell; rate.sum() == total_rate

    # ---- construction ---------------------------------------------------------------
    @staticmethod
    def make_edges(bounds: Bounds, cell: float) -> Tuple[np.ndarray, np.ndarray]:
        """Cell edges of size `cell` covering `bounds` (the last cell may overshoot)."""
        x_min, x_max, y_min, y_max = bounds
        nx = int(np.ceil((x_max - x_min) / cell))
        ny = int(np.ceil((y_max - y_min) / cell))
        return x_min + cell * np.arange(nx + 1), y_min + cell * np.arange(ny + 1)

    @classmethod
    def fit(
        cls,
        positions: np.ndarray,
        duration: float,
        bounds: Bounds,
        cell: float = 0.5,
        uniform_mix: float = 0.05,
    ) -> "GridBirthModel":
        """Maximum-likelihood fit: cell rate = arrivals in cell / duration.

        `uniform_mix` (0..1) is the fraction of the total rate re-distributed
        uniformly over all cells. With 0 the model is the pure histogram and
        any held-out arrival in an empty cell gets log-likelihood -inf; a small
        value (default 5 %) is the usual fix (additive smoothing in disguise).
        """
        x_edges, y_edges = cls.make_edges(bounds, cell)
        counts, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_edges, y_edges])
        total_rate = len(positions) / duration
        empirical = counts / duration
        uniform = np.full_like(empirical, total_rate / empirical.size)
        rate = (1.0 - uniform_mix) * empirical + uniform_mix * uniform
        return cls(x_edges=x_edges, y_edges=y_edges, rate=rate)

    @classmethod
    def uniform(cls, total_rate: float, bounds: Bounds, cell: float = 0.5) -> "GridBirthModel":
        """Baseline: same scalar rate, spread evenly over the scene."""
        x_edges, y_edges = cls.make_edges(bounds, cell)
        n_cells = (len(x_edges) - 1) * (len(y_edges) - 1)
        rate = np.full((len(x_edges) - 1, len(y_edges) - 1), total_rate / n_cells)
        return cls(x_edges=x_edges, y_edges=y_edges, rate=rate)

    # ---- basic quantities -----------------------------------------------------------
    @property
    def total_rate(self) -> float:
        """Scalar arrival rate lam [arrivals / s] for the whole scene."""
        return float(self.rate.sum())

    @property
    def cell_size(self) -> Tuple[float, float]:
        return float(self.x_edges[1] - self.x_edges[0]), float(self.y_edges[1] - self.y_edges[0])

    @property
    def cell_area(self) -> float:
        w, h = self.cell_size
        return w * h

    @property
    def bounds(self) -> Bounds:
        return float(self.x_edges[0]), float(self.x_edges[-1]), float(self.y_edges[0]), float(self.y_edges[-1])

    def cell_index(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(i, j) cell indices of each (x, y); points outside are clamped to the border cells."""
        i = np.searchsorted(self.x_edges, positions[:, 0], side="right") - 1
        j = np.searchsorted(self.y_edges, positions[:, 1], side="right") - 1
        return np.clip(i, 0, self.rate.shape[0] - 1), np.clip(j, 0, self.rate.shape[1] - 1)

    def intensity(self, positions: np.ndarray) -> np.ndarray:
        """lambda(x, y) in arrivals / s / m^2 at each query point."""
        i, j = self.cell_index(np.asarray(positions, dtype=float).reshape(-1, 2))
        return self.rate[i, j] / self.cell_area

    # ---- the two functions the planner needs --------------------------------------
    def expected_count(self, duration: float, region: Optional[Bounds] = None) -> float:
        """Expected number of new pedestrians in `region` (x_min, x_max, y_min, y_max)
        during `duration` seconds. Cells partially inside the region contribute in
        proportion to the overlapping area. `region=None` means the whole scene."""
        if region is None:
            return self.total_rate * duration
        x_min, x_max, y_min, y_max = region
        # Overlap length of each cell with [x_min, x_max] along x, likewise for y.
        wx = np.clip(np.minimum(self.x_edges[1:], x_max) - np.maximum(self.x_edges[:-1], x_min), 0.0, None)
        wy = np.clip(np.minimum(self.y_edges[1:], y_max) - np.maximum(self.y_edges[:-1], y_min), 0.0, None)
        frac = np.outer(wx, wy) / self.cell_area  # (nx, ny) fraction of each cell inside the region
        return float(duration * (self.rate * frac).sum())

    def sample(self, duration: float, rng: Optional[np.random.Generator] = None) -> BirthSample:
        """Draw one realisation of births over `duration` seconds."""
        rng = np.random.default_rng() if rng is None else rng
        n = rng.poisson(self.total_rate * duration)
        probs = (self.rate / self.total_rate).ravel()
        flat = rng.choice(probs.size, size=n, p=probs)
        i, j = np.unravel_index(flat, self.rate.shape)
        w, h = self.cell_size
        xs = self.x_edges[i] + rng.uniform(0.0, w, size=n)
        ys = self.y_edges[j] + rng.uniform(0.0, h, size=n)
        times = np.sort(rng.uniform(0.0, duration, size=n))
        return BirthSample(times=times, positions=np.column_stack([xs, ys]))

    # ---- scoring --------------------------------------------------------------------
    def log_likelihood(self, positions: np.ndarray, duration: float) -> float:
        """Poisson point-process log-likelihood of observed appearance positions
        over `duration` seconds:  sum_i log lambda(x_i) - duration * lam."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        return float(np.log(self.intensity(positions)).sum() - duration * self.total_rate)

    def region_counts(self, duration: float, region_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expected counts on a coarser grid of `region_size`-meter squares covering the
        scene. Returns (x_edges, y_edges, expected) with expected of shape (rx, ry)."""
        rx, ry = self.make_edges(self.bounds, region_size)
        expected = np.array(
            [[self.expected_count(duration, (rx[a], rx[a + 1], ry[b], ry[b + 1])) for b in range(len(ry) - 1)]
             for a in range(len(rx) - 1)]
        )
        return rx, ry, expected


def poisson_log_pmf(k: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """log P(N = k) for Poisson(mean), elementwise; mean is floored at a tiny value."""
    from math import lgamma

    k = np.asarray(k, dtype=float)
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-12)
    log_fact = np.vectorize(lambda v: lgamma(v + 1.0))(k)
    return k * np.log(mean) - mean - log_fact


# ------------------------------------------------------------------------------------
# Kernel density estimate (KDE) alternative to the grid
# ------------------------------------------------------------------------------------
_erf = np.vectorize(__import__("math").erf)


def _gaussian_mass_1d(lo: np.ndarray, hi: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    """Probability mass of N(mean, sigma^2) inside [lo, hi], elementwise (broadcasting)."""
    s = sigma * np.sqrt(2.0)
    return 0.5 * (_erf((hi - mean) / s) - _erf((lo - mean) / s))


@dataclass
class KDEBirthModel:
    """Poisson birth process whose spatial intensity is a Gaussian kernel density.

    Every training arrival at x_i contributes a 2D Gaussian bump of width
    `bandwidth` (same in x and y), truncated to `bounds` and renormalised so each
    bump has unit mass inside the scene. The intensity is

        lambda(x) = (1 - uniform_mix) * (1 / duration) * sum_i K_i(x)
                    + uniform_mix * total_rate / area(bounds)

    so it integrates to `total_rate` over the scene exactly like `GridBirthModel`,
    and the two models differ only in how they spread that rate in space.
    The bandwidth plays the role the cell size plays for the grid.
    """

    centers: np.ndarray  # (N, 2) training arrival positions
    bandwidth: float
    bounds: Bounds
    duration: float
    uniform_mix: float = 0.05

    @classmethod
    def fit(
        cls,
        positions: np.ndarray,
        duration: float,
        bounds: Bounds,
        bandwidth: float = 1.0,
        uniform_mix: float = 0.05,
    ) -> "KDEBirthModel":
        """'Fitting' a KDE is storing the points; the bandwidth is a user choice."""
        return cls(centers=np.asarray(positions, dtype=float).reshape(-1, 2), bandwidth=bandwidth,
                   bounds=bounds, duration=duration, uniform_mix=uniform_mix)

    # ---- basic quantities -----------------------------------------------------------
    @property
    def total_rate(self) -> float:
        return len(self.centers) / self.duration

    @property
    def area(self) -> float:
        x_min, x_max, y_min, y_max = self.bounds
        return (x_max - x_min) * (y_max - y_min)

    def _kernel_mass(self, region: Bounds) -> np.ndarray:
        """(N,) mass of each (untruncated) kernel inside `region`."""
        x_min, x_max, y_min, y_max = region
        mx = _gaussian_mass_1d(x_min, x_max, self.centers[:, 0], self.bandwidth)
        my = _gaussian_mass_1d(y_min, y_max, self.centers[:, 1], self.bandwidth)
        return mx * my

    def intensity(self, positions: np.ndarray) -> np.ndarray:
        """lambda(x, y) in arrivals / s / m^2 at each query point."""
        q = np.asarray(positions, dtype=float).reshape(-1, 2)
        d2 = ((q[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=-1)  # (M, N)
        kernel = np.exp(-0.5 * d2 / self.bandwidth**2) / (2.0 * np.pi * self.bandwidth**2)
        kernel = kernel / self._kernel_mass(self.bounds)[None, :]  # truncate + renormalise
        kde = kernel.sum(axis=1) / self.duration
        return (1.0 - self.uniform_mix) * kde + self.uniform_mix * self.total_rate / self.area

    # ---- the two functions the planner needs --------------------------------------
    def expected_count(self, duration: float, region: Optional[Bounds] = None) -> float:
        """Expected births in `region` during `duration` seconds (closed form via erf)."""
        if region is None:
            return self.total_rate * duration
        x_min, x_max, y_min, y_max = region
        bx0, bx1, by0, by1 = self.bounds
        clipped = (max(x_min, bx0), min(x_max, bx1), max(y_min, by0), min(y_max, by1))
        if clipped[0] >= clipped[1] or clipped[2] >= clipped[3]:
            return 0.0
        frac = self._kernel_mass(clipped) / self._kernel_mass(self.bounds)  # per-kernel fraction in region
        kde_part = (1.0 - self.uniform_mix) * frac.sum() / self.duration
        area_r = (clipped[1] - clipped[0]) * (clipped[3] - clipped[2])
        unif_part = self.uniform_mix * self.total_rate * area_r / self.area
        return float(duration * (kde_part + unif_part))

    def sample(self, duration: float, rng: Optional[np.random.Generator] = None) -> BirthSample:
        """N ~ Poisson(lam * T); each birth is either a uniform point (prob. uniform_mix)
        or a random training arrival plus Gaussian noise, re-drawn if it lands outside."""
        rng = np.random.default_rng() if rng is None else rng
        n = rng.poisson(self.total_rate * duration)
        x_min, x_max, y_min, y_max = self.bounds
        out = np.empty((n, 2))
        for k in range(n):
            if rng.uniform() < self.uniform_mix:
                out[k] = rng.uniform([x_min, y_min], [x_max, y_max])
                continue
            while True:
                p = self.centers[rng.integers(len(self.centers))] + rng.normal(0.0, self.bandwidth, size=2)
                if x_min <= p[0] < x_max and y_min <= p[1] < y_max:
                    out[k] = p
                    break
        return BirthSample(times=np.sort(rng.uniform(0.0, duration, size=n)), positions=out)

    # ---- scoring --------------------------------------------------------------------
    def log_likelihood(self, positions: np.ndarray, duration: float) -> float:
        """sum_i log lambda(x_i) - duration * lam, as for the grid model."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        return float(np.log(self.intensity(positions)).sum() - duration * self.total_rate)

    def region_counts(self, duration: float, region_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expected counts on a `region_size`-meter grid covering the scene (same layout as the grid model)."""
        rx, ry = GridBirthModel.make_edges(self.bounds, region_size)
        expected = np.array(
            [[self.expected_count(duration, (rx[a], rx[a + 1], ry[b], ry[b + 1])) for b in range(len(ry) - 1)]
             for a in range(len(rx) - 1)]
        )
        return rx, ry, expected
