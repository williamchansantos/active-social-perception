"""Per-pedestrian constant-velocity Kalman tracks from camera detections.

Runs once per simulation step (1 frame = 0.4 s): every track is predicted one
step, detected pedestrians are updated, and the posterior position is appended
to the track's grid history (the LSTM input). Unseen pedestrians keep being
predicted, so their covariance grows until they are seen again or dropped.
Process noise is continuous white-noise acceleration, as validated in
irregular_time_experiment.py. Units: meters and frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

H = np.hstack([np.eye(2), np.zeros((2, 2))])


def cwna(dt: float, qc: float):
    I = np.eye(2)
    F = np.block([[I, dt * I], [np.zeros((2, 2)), I]])
    Q = qc * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I], [dt ** 2 / 2 * I, dt * I]])
    return F, Q


@dataclass
class TrackerConfig:
    qc: float = 1.21e-3  # m^2/frame^3, from Zara training-set accelerations
    v0_std: float = 0.5  # prior velocity std for a new track (m/frame)
    max_history: int = 8  # LSTM observation length cap
    drop_after: int = 10  # steps unseen before a track is deleted


class Track:
    def __init__(self, z, R, v0_std):
        self.x = np.concatenate([z, [0.0, 0.0]])
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = R
        self.P[2:, 2:] = v0_std ** 2 * np.eye(2)
        self.history: List[np.ndarray] = [self.x[:2].copy()]
        self.unseen = 0

    def predict(self, F, Q):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R):
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P


class Tracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self.F, self.Q = cwna(1.0, cfg.qc)

    def step(self, detections):
        """detections: {ped_id: (z (2,), R (2, 2))} from the camera this step."""
        for pid, tr in self.tracks.items():
            tr.predict(self.F, self.Q)
            if pid in detections:
                tr.update(*detections[pid])
                tr.unseen = 0
            else:
                tr.unseen += 1
            tr.history.append(tr.x[:2].copy())
            del tr.history[: -self.cfg.max_history]
        for pid, (z, R) in detections.items():
            if pid not in self.tracks:
                self.tracks[pid] = Track(z, R, self.cfg.v0_std)
        for pid in [p for p, tr in self.tracks.items() if tr.unseen > self.cfg.drop_after]:
            del self.tracks[pid]
