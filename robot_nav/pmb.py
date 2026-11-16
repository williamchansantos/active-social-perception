"""Poisson multi-Bernoulli (PMB) pedestrian tracking for a limited-FOV camera.

Track-oriented PMB (the single-global-hypothesis special case of PMBM, exact
here because data association is given by the simulator):

  Undetected pedestrians: Poisson point process with gridded intensity D_u
  over (direction channel, cell); sum over a region = expected number of
  pedestrians there that the robot has never detected.
    predict: advect each channel with its mean velocity, diffuse, mix
             channels (turning), clip to walkable cells, add births
    update:  D_u *= 1 - p_D(cell)   (occlusion-aware detection probability)
  Crowd surges: the true undetected intensity is lambda * D_u with a scene-wide
  multiplier lambda ~ Gamma(a, b). First-time detections are Poisson with mean
  lambda * sum(p_D D_u) (each person counted once), giving the conjugate update
  a += n_new, b += sum(p_D D_u); a, b relax back to the prior over rate_tau
  frames (tuned on a different scene). The undetected count in a region is then
  negative-binomial:
    P(no undetected pedestrian in A) = (b / (b + D_u(A)))^a

  Detected pedestrians: Bernoulli components, existence r and Gaussian CV state.
    predict:      r <- p_S r, p_S fitted per cell from where people leave the scene
    detected:     Kalman update, r = 1 (no clutter in the simulator)
    missed:       r <- r (1 - p_D) / (1 - r p_D), p_D at the track mean
    first seen:   a new Bernoulli whose velocity prior is the PPP's local
                  direction-channel mixture (PPP x likelihood, projected)
    pruned:       low-r or very diffuse tracks are recycled into D_u

The scene prior (initial intensity = mean occupancy, birth intensity,
per-channel velocity statistics, walkable mask) is fitted from the scene's
annotations, i.e. it plays the role of a map of where people walk.
Units: meters and frames (1 frame = 0.4 s in Zara).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import ndimage
from scipy.special import gammaln

from .camera import CameraConfig
from .tracking import Track, TrackerConfig, cwna

CHANNELS = ("+x", "-x", "+y", "-y", "static")
C = len(CHANNELS)
STATIC_SPEED = 0.04  # m/frame (0.1 m/s)


def channel_of(v: np.ndarray) -> np.ndarray:
    """Direction channel index for velocities v (..., 2) in m/frame."""
    ang = np.arctan2(v[..., 1], v[..., 0])
    moving = np.select([np.abs(ang) < np.pi / 4, np.abs(ang) > 3 * np.pi / 4, ang > 0], [0, 1, 2], 3)
    return np.where(np.linalg.norm(v, axis=-1) < STATIC_SPEED, 4, moving)


def channel_mixing(switch: float) -> np.ndarray:
    mix = np.full((C, C), switch / (C - 1))
    np.fill_diagonal(mix, 1 - switch)
    return mix


def ppp_predict(D: np.ndarray, prior: "ScenePrior", mix: np.ndarray, frac: float = 1.0) -> np.ndarray:
    """Prediction of the undetected intensity (C, nx, ny) over `frac` frames (1 = one frame;
    fractions scale the advection, diffusion variance, survival exponent and births; `mix` must
    be the channel mixing for that fraction)."""
    res = prior.grid.res
    out = np.empty_like(D)
    for c in range(C):
        shifted = ndimage.shift(D[c], frac * prior.vel_mean[c] / res, order=1, mode="constant", cval=0.0)
        out[c] = ndimage.gaussian_filter(shifted, np.sqrt(frac * np.diag(prior.vel_cov[c])) / res, mode="constant")
    out = np.einsum("dc,cxy->dxy", mix.T, out)
    if frac == 1.0:
        return out * prior.survival * prior.walkable + prior.birth
    return out * prior.survival ** frac * prior.walkable + frac * prior.birth


# ----------------------------------------------------------------------------- grid and prior

@dataclass
class Grid:
    x0: float
    y0: float
    res: float
    nx: int
    ny: int

    @property
    def centers(self) -> np.ndarray:
        """(nx, ny, 2) cell centers."""
        xs = self.x0 + self.res * (np.arange(self.nx) + 0.5)
        ys = self.y0 + self.res * (np.arange(self.ny) + 0.5)
        return np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)

    def hist(self, pos: np.ndarray, weights=None) -> np.ndarray:
        h, _, _ = np.histogram2d(
            pos[:, 0], pos[:, 1], bins=[self.nx, self.ny], weights=weights,
            range=[[self.x0, self.x0 + self.nx * self.res], [self.y0, self.y0 + self.ny * self.res]],
        )
        return h

    def contains(self, pos) -> bool:
        return (self.x0 <= pos[0] < self.x0 + self.nx * self.res) and (self.y0 <= pos[1] < self.y0 + self.ny * self.res)


@dataclass
class ScenePrior:
    grid: Grid
    occupancy: np.ndarray  # (C, nx, ny) mean number of pedestrians per cell
    birth: np.ndarray  # (C, nx, ny) expected new pedestrians per cell per frame
    vel_mean: np.ndarray  # (C, 2) m/frame
    vel_cov: np.ndarray  # (C, 2, 2)
    walkable: np.ndarray  # (nx, ny) bool
    survival: np.ndarray  # (nx, ny) PPP per-frame survival (exit rate scaled for advection losses)
    exit_prob: np.ndarray  # (nx, ny) raw per-frame probability a pedestrian leaves the scene here
    rate_a0: float = 1e3  # Gamma shape (= rate) of the crowd multiplier prior; large = no surges
    rate_tau: float = 25.0  # frames over which crowd-rate evidence is forgotten

    def survival_at(self, pos) -> float:
        """Bernoulli per-frame survival at a position: raw exit rate on walkable
        cells (tracks have no advection loss), 0.5 off the walkable area."""
        g = self.grid
        ix, iy = int((pos[0] - g.x0) / g.res), int((pos[1] - g.y0) / g.res)
        if 0 <= ix < g.nx and 0 <= iy < g.ny and self.walkable[ix, iy]:
            return float(1 - self.exit_prob[ix, iy])
        return 0.5


def fit_scene_prior(obsmat_path: str, res: float = 0.25, margin: float = 1.0, smooth_cells: float = 2.0,
                    channel_switch: float = 0.02, frame_range=None, dt: float = 0.4,
                    exclude_range=None) -> ScenePrior:
    """Fit the undetected-pedestrian prior from an obsmat.txt file.

    frame_range = (lo, hi): fit only on annotation frames lo <= f < hi, so that
    episodes evaluated outside that range never inform the prior. The grid's
    bounding box still spans the whole file (scene geometry, not crowd data).
    dt = annotation step in seconds (0.4 s in ETH/UCY), for m/s -> m/frame.
    exclude_range = (lo, hi): leave out annotation frames lo <= f < hi instead (an episode's time window in the
    middle of a recording); the per-frame rates are then over the kept frames only.
    """
    d_all = np.loadtxt(obsmat_path)
    d = d_all if frame_range is None else d_all[(d_all[:, 0] >= frame_range[0]) & (d_all[:, 0] < frame_range[1])]
    if exclude_range is not None:
        d = d[(d[:, 0] < exclude_range[0]) | (d[:, 0] >= exclude_range[1])]
    frames = np.unique(d[:, 0])
    frame_step = np.median(np.diff(np.unique(d_all[:, 0])))
    n_steps = (frames.max() - frames.min()) / frame_step + 1
    if exclude_range is not None:  # annotation steps of the recording outside the window (empty frames count)
        grid_f = np.arange(d_all[:, 0].min(), d_all[:, 0].max() + 0.5 * frame_step, frame_step)
        kept_f = grid_f[(grid_f < exclude_range[0]) | (grid_f >= exclude_range[1])]
        n_steps = len(kept_f)
    pos = d[:, [2, 4]]
    vel = d[:, [5, 7]] * dt  # m/s -> m per annotation step
    lo, hi = d_all[:, [2, 4]].min(0) - margin, d_all[:, [2, 4]].max(0) + margin
    grid = Grid(lo[0], lo[1], res, int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res)))

    ch = channel_of(vel)
    first, last = np.zeros(len(d), bool), np.zeros(len(d), bool)
    for pid in np.unique(d[:, 1]):
        rows = np.flatnonzero(d[:, 1] == pid)
        first[rows[np.argmin(d[rows, 0])]] = True
        last[rows[np.argmax(d[rows, 0])]] = True

    occupancy, birth = np.zeros((C, grid.nx, grid.ny)), np.zeros((C, grid.nx, grid.ny))
    vel_mean, vel_cov = np.zeros((C, 2)), np.zeros((C, 2, 2))
    for c in range(C):
        m = ch == c
        occupancy[c] = ndimage.gaussian_filter(grid.hist(pos[m]), smooth_cells) / n_steps
        birth[c] = ndimage.gaussian_filter(grid.hist(pos[m & first]), smooth_cells) / n_steps
        vel_mean[c] = vel[m].mean(0)
        vel_cov[c] = np.cov(vel[m].T) + 1e-4 * np.eye(2)
    walkable = ndimage.binary_dilation(grid.hist(pos) > 0, iterations=int(round(1.0 / res)))
    # Per-frame exit probability = last appearances / pedestrian-frames, per cell.
    exits = ndimage.gaussian_filter(grid.hist(pos[last]), smooth_cells)
    visits = ndimage.gaussian_filter(grid.hist(pos), smooth_cells)
    exit_prob = np.clip(exits / np.maximum(visits, 1e-3), 0, 0.9)
    prior = ScenePrior(grid, occupancy * walkable, birth * walkable, vel_mean, vel_cov, walkable, 1 - exit_prob,
                       exit_prob)

    # Advection off the walkable area already removes some mass, so scale the exit
    # probability such that the predict-only intensity stays at the observed occupancy.
    mix, target = channel_mixing(channel_switch), prior.occupancy.sum()
    lo_a, hi_a = 0.0, 1.0
    for _ in range(12):
        alpha = 0.5 * (lo_a + hi_a)
        prior.survival = 1 - alpha * exit_prob
        D = prior.occupancy.copy()
        for _ in range(100):
            D = ppp_predict(D, prior, mix)
        lo_a, hi_a = (alpha, hi_a) if D.sum() > target else (lo_a, alpha)
    prior.survival = 1 - alpha * exit_prob

    # Crowd multiplier prior: frame counts are negative-binomial with var = m + m^2 / a0;
    # evidence is forgotten on the counts' autocorrelation (1/e) time.
    if exclude_range is None:
        idx = np.round((d[:, 0] - frames.min()) / frame_step).astype(int)
        counts = np.bincount(idx, minlength=int(n_steps)).astype(float)
    else:  # kept steps only (empty ones included), the two sides joined
        idx = np.round((d[:, 0] - d_all[:, 0].min()) / frame_step).astype(int)
        full = np.bincount(idx, minlength=len(grid_f)).astype(float)
        counts = full[(grid_f < exclude_range[0]) | (grid_f >= exclude_range[1])]
    m, v = counts.mean(), counts.var()
    prior.rate_a0 = float(1.0 / max((v - m) / m ** 2, 1e-3))
    c = counts - m
    acf = np.array([np.sum(c[:-lag] * c[lag:]) for lag in range(1, 200)]) / np.sum(c * c)
    below = np.flatnonzero(acf < np.exp(-1))
    prior.rate_tau = float(below[0] + 1) if len(below) else 200.0
    return prior


# ----------------------------------------------------------------------------- tracker

@dataclass
class PMBConfig(TrackerConfig):
    r_prune: float = 0.05  # Bernoullis below this existence are recycled into the PPP
    prune_pos_std: float = 2.5  # m; tracks more diffuse than this are recycled too
    cap_pos_std: Optional[float] = None  # m; if set, surviving tracks' position std is clamped to this
    channel_switch: float = 0.02  # per-frame probability an undetected pedestrian changes channel
    occluder_radius: float = 0.3
    adapt_rate: bool = True  # learn the crowd multiplier online (False: fixed lambda = 1, plain Poisson)
    rate_tau: Optional[float] = None  # forgetting time (frames); None = the prior's count-autocorrelation time
    # Calibration (calibrate_ppp.py; None / 0 = off, as in all pre-registered results):
    p_detect_model: Optional[float] = None  # detection probability the filter assumes in view (None: camera's p_detect)
    relax: float = 0.0  # per-frame relaxation of the undetected intensity toward the prior occupancy (refill)
    # Crowd multiplier prior strength (None: the prior's fitted a0). Small = "unsure how busy it is": lambda starts at
    # mean 1 with a large variance, adapts quickly to the first detections, and forgetting decays toward this weak
    # belief instead of the fitted one.
    rate_prior_strength: Optional[float] = None


class BernoulliTrack(Track):
    def __init__(self, z, R, vel_mean, vel_cov):
        super().__init__(z, R, v0_std=0.0)
        self.x[2:] = vel_mean
        self.P[2:, 2:] = vel_cov
        self.r = 1.0
        self.det_flags = [1]  # 1 where the history point came from a detection, 0 where it was coasted
        self.real_hist = list(self.history)  # history up to the last detection (network coasting)
        self.P_det = self.P.copy()  # covariance at the last detection


class PMBTracker:
    def __init__(self, prior: ScenePrior, cam: CameraConfig, cfg: PMBConfig = PMBConfig()):
        self.prior, self.cam, self.cfg = prior, cam, cfg
        self.grid = prior.grid
        self.centers = self.grid.centers
        self.D = prior.occupancy.copy()  # robot starts knowing nothing: stationary prior
        self.tracks: Dict[int, BernoulliTrack] = {}
        self.F, self.Q = cwna(1.0, cfg.qc)
        self.mix = channel_mixing(cfg.channel_switch)
        self.last_pD: Optional[np.ndarray] = None
        self.rate_a = self.rate_b = prior.rate_a0 if cfg.rate_prior_strength is None else cfg.rate_prior_strength

    # ------------------------------------------------------------------ detection model
    def detection_prob(self, pts: np.ndarray, x_est: np.ndarray, occluders: np.ndarray) -> np.ndarray:
        """p_D at points (..., 2) for a camera at the estimated robot pose, occluded
        by the pedestrians the robot currently detects."""
        d = pts - x_est[:2]
        dist = np.linalg.norm(d, axis=-1)
        heading = np.array([np.cos(x_est[2] + x_est[3]), np.sin(x_est[2] + x_est[3])])
        cosang = (d @ heading) / np.maximum(dist, 1e-9)
        inside = (dist <= self.cam.max_range) & (cosang >= np.cos(np.radians(self.cam.fov_deg) / 2))
        for o in (occluders if self.cam.occlusion else []):
            e = o - x_est[:2]
            e_dist = np.linalg.norm(e)
            # Points behind occluder o (farther along the same ray, within its angular radius).
            u = d / np.maximum(dist, 1e-9)[..., None]
            along = u @ e
            perp = np.linalg.norm(e - along[..., None] * u, axis=-1)
            inside &= ~((along > 0) & (perp < self.cfg.occluder_radius) & (dist > e_dist + self.cfg.occluder_radius))
        p = self.cam.p_detect if self.cfg.p_detect_model is None else self.cfg.p_detect_model
        return p * inside

    # ------------------------------------------------------------------ PPP
    def _predict_ppp(self, frac: float = 1.0):
        mix = self.mix if frac == 1.0 else channel_mixing(self.cfg.channel_switch * frac)
        self.D = ppp_predict(self.D, self.prior, mix, frac)

    def _local_velocity_prior(self, z, R):
        """Velocity marginal of PPP x likelihood at detection z, moment-matched."""
        lik = np.exp(-0.5 * np.einsum("xyi,ij,xyj->xy", self.centers - z, np.linalg.inv(R + 0.25 * np.eye(2)),
                                      self.centers - z))
        w = np.einsum("cxy,xy->c", self.D, lik)
        if w.sum() < 1e-9:
            return np.zeros(2), self.cfg.v0_std ** 2 * np.eye(2)
        w = w / w.sum()
        mean = w @ self.prior.vel_mean
        second = np.einsum("c,cij->ij", w, self.prior.vel_cov + np.einsum("ci,cj->cij", self.prior.vel_mean,
                                                                          self.prior.vel_mean))
        return mean, second - np.outer(mean, mean)

    def _recycle(self, tr: BernoulliTrack):
        """Return a pruned Bernoulli's existence mass to the PPP (in its velocity channel)."""
        S = tr.P[:2, :2] + (0.5 * self.grid.res) ** 2 * np.eye(2)
        dlt = self.centers - tr.x[:2]
        g = np.exp(-0.5 * np.einsum("xyi,ij,xyj->xy", dlt, np.linalg.inv(S), dlt)) * self.prior.walkable
        if g.sum() > 0:
            self.D[int(channel_of(tr.x[2:]))] += tr.r * g / g.sum() / self.rate_mean

    # ------------------------------------------------------------------ step
    def step(self, detections, x_est, frac: float = 1.0, record: bool = True):
        """detections: {ped_id: (z, R)}; x_est: estimated robot state [px, py, theta, phi].
        frac: time since the last step in frames (0.5 for a camera running at twice the frame
        rate); record: append the track positions to their histories (only on the frame grid
        the trajectory predictor was trained on)."""
        self._predict_ppp(frac)
        if self.cfg.relax > 0:  # people who do not follow the learned flows: refill toward the prior
            w = min(1.0, self.cfg.relax * frac)
            self.D = (1 - w) * self.D + w * self.prior.occupancy
        F, Q = (self.F, self.Q) if frac == 1.0 else cwna(frac, self.cfg.qc)
        occluders = np.array([z for z, _ in detections.values()]).reshape(-1, 2)
        self.last_pD = self.detection_prob(self.centers, x_est, occluders)
        if self.cfg.adapt_rate:
            tau = self.cfg.rate_tau if self.cfg.rate_tau is not None else self.prior.rate_tau
            g = np.exp(-frac / tau)
            a0 = self.prior.rate_a0 if self.cfg.rate_prior_strength is None else self.cfg.rate_prior_strength
            self.rate_a = g * self.rate_a + (1 - g) * a0 + sum(pid not in self.tracks for pid in detections)
            self.rate_b = g * self.rate_b + (1 - g) * a0 + float(np.sum(self.last_pD * self.D.sum(0)))

        for pid, tr in self.tracks.items():
            tr.r *= self.prior.survival_at(tr.x[:2]) ** frac
            tr.predict(F, Q)
            if pid in detections:
                tr.update(*detections[pid])
                tr.r = 1.0
                tr.unseen = 0
            else:
                pD = float(self.detection_prob(tr.x[:2], x_est, occluders))
                denom = 1 - tr.r * pD
                # pD = 1 and r = 1: a certain detection failed, so the pedestrian is gone.
                tr.r = tr.r * (1 - pD) / denom if denom > 1e-12 else 0.0
                tr.unseen += 1
            if record:
                tr.history.append(tr.x[:2].copy())
                del tr.history[: -self.cfg.max_history]
                tr.det_flags.append(int(tr.unseen == 0))
                del tr.det_flags[: -self.cfg.max_history]
                if tr.unseen == 0:
                    tr.real_hist = [h.copy() for h in tr.history]
                    tr.P_det = tr.P.copy()

        for pid, (z, R) in detections.items():
            if pid not in self.tracks:
                vm, vc = self._local_velocity_prior(z, R)
                self.tracks[pid] = BernoulliTrack(z, R, vm, vc)
                if not record:  # off the frame grid: the history starts at the next frame
                    self.tracks[pid].history = []
                    self.tracks[pid].det_flags = []

        self.D *= 1 - self.last_pD

        for pid in list(self.tracks):
            tr = self.tracks[pid]
            if tr.r < self.cfg.r_prune or np.sqrt(np.trace(tr.P[:2, :2]) / 2) > self.cfg.prune_pos_std:
                self._recycle(tr)
                del self.tracks[pid]
            elif self.cfg.cap_pos_std is not None:
                # Clamp a stale track's position spread: P <- D P D, D = diag(s, s, 1, 1), stays PSD
                # and keeps the velocity uncertainty (so the prediction still spreads over the horizon).
                std = np.sqrt(np.trace(tr.P[:2, :2]) / 2)
                if std > self.cfg.cap_pos_std:
                    D = np.diag([self.cfg.cap_pos_std / std] * 2 + [1.0, 1.0])
                    tr.P = D @ tr.P @ D

    # ------------------------------------------------------------------ queries
    @property
    def rate_mean(self) -> float:
        return self.rate_a / self.rate_b if self.cfg.adapt_rate else 1.0

    def undetected_density(self) -> np.ndarray:
        """(nx, ny) expected number of never-detected pedestrians per cell."""
        return self.rate_mean * self.D.sum(0)

    def count_logpmf(self, counts: np.ndarray, base_mass: np.ndarray) -> float:
        """Joint log-probability of undetected counts in disjoint regions with base
        intensity masses `base_mass` (D_u summed per region), integrating out lambda."""
        counts, base_mass = counts.ravel(), np.maximum(base_mass.ravel(), 1e-12)
        per_region = np.sum(counts * np.log(base_mass) - gammaln(counts + 1))
        if not self.cfg.adapt_rate:
            return float(per_region - base_mass.sum())
        a, b, N, M = self.rate_a, self.rate_b, counts.sum(), base_mass.sum()
        return float(per_region + gammaln(a + N) - gammaln(a) + a * np.log(b) - (a + N) * np.log(b + M))

    def prob_any_undetected(self, mask: np.ndarray) -> float:
        """P(at least one never-detected pedestrian in the cells of `mask`)."""
        M = float(self.D.sum(0)[mask].sum())
        if not self.cfg.adapt_rate:
            return 1 - np.exp(-M)
        return 1 - (self.rate_b / (self.rate_b + M)) ** self.rate_a

    def expected_undetected(self, center, radius) -> float:
        mask = np.linalg.norm(self.centers - np.asarray(center), axis=-1) <= radius
        return float(self.undetected_density()[mask].sum())

    def mass_limit(self, delta_u: float) -> float:
        """Largest base mass D_u(A) with P(>=1 undetected pedestrian in A) <= delta_u."""
        if not self.cfg.adapt_rate:
            return float(-np.log(1 - delta_u))
        return float(self.rate_b * ((1 - delta_u) ** (-1.0 / self.rate_a) - 1))

    def unseen_forecast(self, path: np.ndarray, n_slots: int, delta_u: float, coarsen: int = 2,
                        search_radius: float = 3.0):
        """Predict-only forecast of the undetected intensity along a planned path.

        For each horizon step k (path[k] = planned robot position at step k+1), the
        forecast intensity is coarsened to (coarsen x res) cells per direction channel
        and the n_slots nearest (cell, channel) point masses within search_radius are
        returned. Masses are in base units (lambda is integrated out by mass_limit).

        Returns pos (N, P, 2), mass (N, P), vel (N, P, 2), mass limit for delta_u, and the
        negative-binomial parameters (a, b) of the crowd multiplier (a = b = 1e6: plain Poisson).
        """
        N, P, g = len(path), n_slots, self.grid
        nxc, nyc = g.nx // coarsen, g.ny // coarsen
        centers = self.centers[: nxc * coarsen, : nyc * coarsen].reshape(nxc, coarsen, nyc, coarsen, 2).mean((1, 3))
        cpos = np.broadcast_to(centers, (C, nxc, nyc, 2)).reshape(-1, 2)
        cvel = np.broadcast_to(self.prior.vel_mean[:, None, None, :], (C, nxc, nyc, 2)).reshape(-1, 2)
        pos, mass, vel = np.full((N, P, 2), 1e3), np.zeros((N, P)), np.zeros((N, P, 2))
        D = self.D
        for k in range(N):
            D = ppp_predict(D, self.prior, self.mix)
            Dc = D[:, : nxc * coarsen, : nyc * coarsen].reshape(C, nxc, coarsen, nyc, coarsen).sum((2, 4)).ravel()
            dist = np.linalg.norm(cpos - path[k], axis=-1)
            cand = np.flatnonzero((dist < search_radius) & (Dc > 1e-6))
            cand = cand[np.argsort(dist[cand])][:P]
            pos[k, : len(cand)], mass[k, : len(cand)], vel[k, : len(cand)] = cpos[cand], Dc[cand], cvel[cand]
        a, b = (self.rate_a, self.rate_b) if self.cfg.adapt_rate else (1e6, 1e6)
        return pos, mass, vel, self.mass_limit(delta_u), a, b

    def explore_masses(self, center: np.ndarray, n_slots: int, radius: float, coarsen: int = 2):
        """The n_slots largest (coarse cell, channel) masses of the CURRENT undetected intensity within radius of
        center, for the crowd-density exploration term of the MPC (MPCConfig.w_explore). Returns pos (P, 2),
        velocity (P, 2) in m/frame (the channel's mean flow), base mass (P,) and the multiplier's b."""
        g = self.grid
        nxc, nyc = g.nx // coarsen, g.ny // coarsen
        centers = self.centers[: nxc * coarsen, : nyc * coarsen].reshape(nxc, coarsen, nyc, coarsen, 2).mean((1, 3))
        cpos = np.broadcast_to(centers, (C, nxc, nyc, 2)).reshape(-1, 2)
        cvel = np.broadcast_to(self.prior.vel_mean[:, None, None, :], (C, nxc, nyc, 2)).reshape(-1, 2)
        Dc = self.D[:, : nxc * coarsen, : nyc * coarsen].reshape(C, nxc, coarsen, nyc, coarsen).sum((2, 4)).ravel()
        cand = np.flatnonzero((np.linalg.norm(cpos - center, axis=-1) < radius) & (Dc > 1e-6))
        cand = cand[np.argsort(-Dc[cand])][:n_slots]
        pos, vel, mass = np.full((n_slots, 2), 1e3), np.zeros((n_slots, 2)), np.zeros(n_slots)
        pos[: len(cand)], vel[: len(cand)], mass[: len(cand)] = cpos[cand], cvel[cand], Dc[cand]
        b = self.rate_b if self.cfg.adapt_rate else 1e6
        return pos, vel, mass, b
