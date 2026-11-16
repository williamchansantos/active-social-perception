"""RGB-D pedestrian perception: person detector + depth -> world positions -> association with the PMB tracks.

Replaces the simulated camera (robot_nav.camera.observe) when images are available (SocNavBench full render):
  1. PersonDetector: COCO-pretrained torchvision detector, class "person" above a score threshold.
  2. localize: per box, the bearing from the box centre column and the range from the depth pixels of the box's
     central part (a low percentile, so the background around the person does not count), plus half a body
     depth to get from the visible surface to the body centre.
  3. to_world: camera frame -> world with the estimated robot pose; covariance = calibrated range / bearing
     noise + robot pose uncertainty (as in observe()).
  4. associate: detections carry no identity, so they are matched to the PMB tracks' predicted positions
     (Hungarian on Mahalanobis distance with a chi-square gate); unmatched detections get new ids.
The output has observe()'s format, {id: (z, R)}, so PMBTracker.step is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class VisionConfig:
    fov_deg: float = 60.0
    width: int = 256
    score_thresh: float = 0.6
    model: str = "fasterrcnn_mobilenet_v3_large_fpn"  # or "fasterrcnn_resnet50_fpn_v2"
    min_box_h: int = 12  # px, smaller boxes are ignored (too far / too uncertain)
    max_range: float = 4.0  # m, detections beyond are dropped (= CameraConfig.max_range of the planner's model)
    depth_pct: float = 25.0  # percentile of the central-box depths taken as the person's surface
    body_half_depth: float = 0.17  # m, surface -> body centre (removes the -5 cm range bias of check_vision.py)
    # Measurement noise, calibrated on 74 rendered SocNavBench Univ frames (check_vision.py, MobileNet, 0.6):
    # range error sd 0.07-0.2 m, lateral error sd 0.1-0.2 m; people closer than 2 m are often cut by the image
    # border (box centre != body centre), lateral sd ~0.3 m.
    sigma_range: Tuple[float, float] = (0.10, 0.02)  # sigma_r = a + b * range
    sigma_lateral: float = 0.15  # m
    sigma_lateral_near: float = 0.30  # m, below near_range
    near_range: float = 2.0
    gate_chi2: float = 9.21  # 99% for 2 dof
    gate_dist: float = 1.5  # m, hard cap on association distance
    assoc_q: float = 0.15  # m^2 per step, extra position variance for the association prediction


class PersonDetector:
    def __init__(self, cfg: VisionConfig, device: str = "cpu"):
        import torch
        import torchvision.models.detection as tvd

        self.torch = torch
        self.cfg = cfg
        builders = {"fasterrcnn_mobilenet_v3_large_fpn": (tvd.fasterrcnn_mobilenet_v3_large_fpn,
                                                          tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
                    "fasterrcnn_resnet50_fpn_v2": (tvd.fasterrcnn_resnet50_fpn_v2,
                                                   tvd.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)}
        build, weights = builders[cfg.model]
        self.model = build(weights=weights, box_score_thresh=cfg.score_thresh).eval().to(device)
        self.device = device
        import os
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", torch.get_num_threads())))

    def __call__(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """rgb (H, W, 3) uint8 -> boxes (n, 4) [x0, y0, x1, y1] px, scores (n,) of persons."""
        t = self.torch.from_numpy(np.ascontiguousarray(rgb).copy()).permute(2, 0, 1).float().div(255.0).to(self.device)
        with self.torch.no_grad():
            out = self.model([t])[0]
        keep = (out["labels"] == 1) & (out["scores"] >= self.cfg.score_thresh)
        b = out["boxes"][keep].cpu().numpy()
        s = out["scores"][keep].cpu().numpy()
        ok = (b[:, 3] - b[:, 1]) >= self.cfg.min_box_h
        return b[ok], s[ok]


def focal(cfg: VisionConfig) -> float:
    return (cfg.width / 2.0) / np.tan(np.radians(cfg.fov_deg) / 2.0)


def localize(boxes: np.ndarray, depth: np.ndarray, cfg: VisionConfig) -> List[Tuple[float, float]]:
    """Per box: (range along the optical axis to the body centre [m], bearing [rad, left positive]) or None."""
    f = focal(cfg)
    cx = (depth.shape[1] - 1) / 2.0
    out = []
    for x0, y0, x1, y1 in boxes:
        w, h = x1 - x0, y1 - y0
        # central part of the box: middle 40% of the width, rows 15-65% (torso), where the person is
        r0, r1 = int(y0 + 0.15 * h), int(np.ceil(y0 + 0.65 * h))
        c0, c1 = int(x0 + 0.30 * w), int(np.ceil(x0 + 0.70 * w))
        patch = depth[max(r0, 0):max(r1, r0 + 1), max(c0, 0):max(c1, c0 + 1)].astype(np.float64)
        patch = patch[np.isfinite(patch)]
        if patch.size < 4:
            out.append(None)
            continue
        z = float(np.percentile(patch, cfg.depth_pct)) + cfg.body_half_depth
        u = 0.5 * (x0 + x1)
        out.append((z, float(np.arctan2(cx - u, f))))
    return out


def to_world(locs, x_est: np.ndarray, pose_cov: np.ndarray, cfg: VisionConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Camera-frame (z, bearing) -> world (position, covariance). x_est = [px, py, theta, phi]."""
    out = []
    head = x_est[2] + x_est[3]
    for loc in locs:
        if loc is None:
            continue
        z, b = loc
        rng = z / np.cos(b)
        if rng > cfg.max_range:
            continue
        a = head + b
        u = np.array([np.cos(a), np.sin(a)])
        n = np.array([-u[1], u[0]])
        p = x_est[:2] + rng * u
        sr = cfg.sigma_range[0] + cfg.sigma_range[1] * rng
        st = cfg.sigma_lateral_near if rng < cfg.near_range else cfg.sigma_lateral
        # sensor noise (range along u, bearing across) + robot position error + heading error rotating the point
        R = sr ** 2 * np.outer(u, u) + st ** 2 * np.outer(n, n) + pose_cov[:2, :2] \
            + (pose_cov[2, 2] + pose_cov[3, 3]) * rng ** 2 * np.outer(n, n)
        out.append((p, R))
    return out


@dataclass
class Associator:
    cfg: VisionConfig
    next_id: int = 1_000_000
    stats: Dict[str, int] = field(default_factory=lambda: dict(matched=0, new=0))

    def __call__(self, dets: List[Tuple[np.ndarray, np.ndarray]], tracks: dict, dt: float) -> Dict[int, tuple]:
        """Match detections to the tracks' predicted positions (one step of constant velocity)."""
        ids = list(tracks)
        out = {}
        if dets and ids:
            C = np.full((len(dets), len(ids)), 1e6)
            for j, pid in enumerate(ids):
                tr = tracks[pid]
                mu = tr.x[:2] + dt * tr.x[2:4]
                S0 = tr.P[:2, :2] + dt * (tr.P[:2, 2:4] + tr.P[2:4, :2]) + dt ** 2 * tr.P[2:4, 2:4] \
                    + self.cfg.assoc_q * np.eye(2)
                for i, (z, R) in enumerate(dets):
                    d = z - mu
                    if np.linalg.norm(d) > self.cfg.gate_dist:
                        continue
                    m2 = float(d @ np.linalg.solve(S0 + R, d))
                    if m2 < self.cfg.gate_chi2:
                        C[i, j] = m2 - 0.5 * np.log(max(getattr(tr, "r", 1.0), 1e-3))  # prefer likely tracks
            rows, cols = linear_sum_assignment(C)
            for i, j in zip(rows, cols):
                if C[i, j] < 1e5:
                    out[ids[j]] = dets[i]
            self.stats["matched"] += len(out)
        used = {id(v[0]) for v in out.values()}
        for z, R in dets:
            if id(z) not in used:
                out[self.next_id] = (z, R)
                self.next_id += 1
                self.stats["new"] += 1
        return out
