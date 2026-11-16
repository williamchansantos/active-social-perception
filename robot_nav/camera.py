"""Single pan camera with limited range and field of view.

A pedestrian is visible when it is within range, inside the FOV cone and not
occluded by another (closer) pedestrian, and a visible pedestrian is detected
with probability p_detect. Detections are relative positions
with range-dependent noise, mapped to the world frame with the robot's
*estimated* pose, so robot localization error leaks into pedestrian
measurements (correlated across pedestrians), as on a real robot. Data
association is assumed perfect (detections carry the true pedestrian id).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class CameraConfig:
    fov_deg: float = 70.0
    max_range: float = 6.0
    sigma0: float = 0.05  # measurement std at zero range (m)
    sigma_per_m: float = 0.02  # std growth per meter of range
    ped_radius: float = 0.3  # occluder radius
    occlusion: bool = True
    p_detect: float = 0.9  # chance a visible pedestrian is actually detected

    def sigma(self, dist):
        return self.sigma0 + self.sigma_per_m * dist


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def visible(cam_pos, cam_heading, peds: Dict[int, np.ndarray], cfg: CameraConfig) -> Dict[int, bool]:
    ids = list(peds)
    out = {}
    cos_half = np.cos(np.radians(cfg.fov_deg) / 2)
    heading = np.array([np.cos(cam_heading), np.sin(cam_heading)])
    for pid in ids:
        d = peds[pid] - cam_pos
        dist = np.linalg.norm(d)
        ok = 1e-6 < dist <= cfg.max_range and heading @ d / dist >= cos_half
        if ok and cfg.occlusion:
            u = d / dist
            for other in ids:
                if other == pid:
                    continue
                e = peds[other] - cam_pos
                along = e @ u
                if 0 < along < dist and np.linalg.norm(e - along * u) < cfg.ped_radius:
                    ok = False
                    break
        out[pid] = bool(ok)
    return out


def observe(x_true, x_est, pose_cov, peds, cfg: CameraConfig, rng) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Detections {ped_id: (world position estimate (2,), measurement covariance (2, 2))}.

    x_true / x_est: true and estimated robot state [px, py, theta, phi];
    pose_cov: (4, 4) covariance of the robot state estimate.
    """
    cam_heading = x_true[2] + x_true[3]
    vis = visible(x_true[:2], cam_heading, peds, cfg)
    out = {}
    for pid, seen in vis.items():
        if not seen or rng.uniform() >= cfg.p_detect:
            continue
        rel_body = rot(-x_true[2]) @ (peds[pid] - x_true[:2])
        dist = np.linalg.norm(rel_body)
        rel_body = rel_body + cfg.sigma(dist) * rng.standard_normal(2)
        z = x_est[:2] + rot(x_est[2]) @ rel_body
        # Sensor noise + robot position error + heading error rotating the detection.
        perp = rot(x_est[2]) @ np.array([-rel_body[1], rel_body[0]])
        R = cfg.sigma(dist) ** 2 * np.eye(2) + pose_cov[:2, :2] + pose_cov[2, 2] * np.outer(perp, perp)
        out[pid] = (z, R)
    return out
