"""Pedestrian scenes from an ETH/UCY obsmat.txt file.

Simulation step i corresponds to annotation frame start_frame + i * frame_step
(0.4 s in ETH/UCY), the same step as the trajectory predictor and the MPC.

PedestrianScene replays the recorded trajectories (pedestrians ignore the robot).
ReactiveScene lets each pedestrian follow its recorded path but be pushed away
from the robot by a social-force repulsion, then pulled back to the path.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from social_lstm.dataset import load_obsmat


class PedestrianScene:
    def __init__(self, obsmat_path: str, start_frame: int):
        self.frame_to_peds = load_obsmat(obsmat_path)
        frames = np.array(sorted(self.frame_to_peds))
        self.frame_step = int(np.median(np.diff(frames)))
        # Snap the start onto the annotation grid.
        self.start_frame = int(frames[np.argmin(np.abs(frames - start_frame))])

    def recorded(self, step: int) -> Dict[int, np.ndarray]:
        peds = self.frame_to_peds.get(self.start_frame + step * self.frame_step, {})
        return {pid: np.array(pos) for pid, pos in peds.items()}

    def positions(self, step: int, robot: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        """Ground-truth {ped_id: (2,) position} at simulation step `step` (robot ignored)."""
        return self.recorded(step)


class ReactiveScene(PedestrianScene):
    """Recorded trajectories as references, plus avoidance of the robot.

    Each step a pedestrian moves with its recorded velocity, plus a correction
    `return_gain` of the way back towards its recorded position, plus a social
    force from the robot: an exponential repulsion A exp((r - d) / B) along the
    robot-to-pedestrian direction, weighted down for a robot behind the
    pedestrian (anisotropy lambda). Pedestrians appear and leave when their
    recording does; they do not react to each other (the recording already does).
    Steps must be queried in order (each step is computed from the previous one);
    repeated queries of a step return the cached positions.
    """

    def __init__(self, obsmat_path: str, start_frame: int, dt: float = 0.4, strength: float = 5.0,
                 decay: float = 0.4, radius: float = 0.6, anisotropy: float = 0.35, return_gain: float = 0.3,
                 max_speed: float = 2.0):
        super().__init__(obsmat_path, start_frame)
        self.dt, self.A, self.B, self.r = dt, strength, decay, radius
        self.lam, self.k, self.vmax = anisotropy, return_gain, max_speed
        self.cache: Dict[int, Dict[int, np.ndarray]] = {}

    def positions(self, step: int, robot: Optional[np.ndarray] = None) -> Dict[int, np.ndarray]:
        if step in self.cache:
            return self.cache[step]
        rec = self.recorded(step)
        if step == 0:
            self.cache[0] = rec
            return rec
        prev = self.positions(step - 1)  # cached in normal use (steps are queried in order)
        rec_prev = self.recorded(step - 1)
        out = {}
        for pid, q in rec.items():
            if pid not in prev or pid not in rec_prev:
                out[pid] = q  # appears where the recording starts
                continue
            p = prev[pid]
            v_rec = (q - rec_prev[pid]) / self.dt
            v = v_rec + self.k * (q - p) / self.dt
            if robot is not None:
                d = p - robot[:2]
                dist = max(np.linalg.norm(d), 1e-6)
                n = d / dist
                heading = v_rec / max(np.linalg.norm(v_rec), 1e-6)
                cos_phi = float(-n @ heading)  # 1 when the robot is straight ahead
                w = self.lam + (1 - self.lam) * (1 + cos_phi) / 2
                v = v + self.A * np.exp((self.r - dist) / self.B) * w * n * self.dt
            speed = np.linalg.norm(v)
            if speed > self.vmax:
                v = v * self.vmax / speed
            out[pid] = p + v * self.dt
        self.cache[step] = out
        return out
