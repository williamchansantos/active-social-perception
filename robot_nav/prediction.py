"""Gaussian pedestrian prediction for the planner: Social LSTM mean and
covariance on the tracked histories, plus the constant-velocity propagation of
each track's current state covariance (validated in
state_uncertainty_experiment.py / irregular_time_experiment.py).

The two covariance parts are kept separate because only the state part can be
reduced by looking at a pedestrian, which is what the active perception term
reasons about.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from social_lstm.engine import predict_sequence_with_gaussian
from social_lstm.model import SocialLSTM


@dataclass
class Prediction:
    ids: list
    mean: np.ndarray  # (N, M, 2) predicted means for steps 1..N
    cov_model: np.ndarray  # (N, M, 2, 2) LSTM covariance
    cov_state: np.ndarray  # (N, M, 2, 2) J_k P J_k^T, reducible by observation
    existence: np.ndarray = field(default=None)  # (M,) Bernoulli existence r (1 for plain Kalman tracks)
    n_det: np.ndarray = field(default=None)  # (M,) detections among each track's history points
    state_jac: np.ndarray = field(default=None)  # (N, M, 2, 4) d(mean_k)/d(track state [p, v]); None = CV [I, kI]

    def __post_init__(self):
        if self.existence is None:
            self.existence = np.ones(len(self.ids))
        if self.n_det is None:
            self.n_det = np.zeros(len(self.ids))

    @property
    def cov(self):
        return self.cov_model + self.cov_state


def cv_propagated_cov(P: np.ndarray, horizon: int) -> np.ndarray:
    """J_k P J_k^T for k = 1..horizon, J_k = [I, k I]. P: (M, 4, 4) -> (horizon, M, 2, 2)."""
    out = []
    for k in range(1, horizon + 1):
        J = np.hstack([np.eye(2), k * np.eye(2)])
        out.append(J @ P @ J.T)
    return np.stack(out)


class TrajectoryPredictor:
    def __init__(self, checkpoint: str, horizon: int):
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model = SocialLSTM(**ckpt["hyperparams"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.horizon = horizon

    @torch.no_grad()
    def state_jacobian(self, hist: torch.Tensor, steps: int = None) -> np.ndarray:
        """G_k = d(predicted mean at step k) / d(track state [p, v]) for every pedestrian, (N, M, 2, 4).

        A state error (dp, dv) is mapped onto the history the way short histories are padded: every history point
        at lag i moves by dp - i * dv (v in m per step). Forward-mode differentiation, one pass per state
        dimension with the same direction for every pedestrian at once (effects through neighbours' pooled hidden
        states are neglected; the pooling grid itself is piecewise constant)."""
        import torch.autograd.forward_ad as fwAD

        L, n = hist.shape[0], hist.shape[1]
        steps = steps or self.horizon
        lag = torch.arange(L - 1, -1, -1, dtype=hist.dtype)[:, None, None]
        G = np.zeros((steps, n, 2, 4))
        for d in range(4):
            tangent = torch.zeros(n, 4, dtype=hist.dtype)
            tangent[:, d] = 1.0
            with fwAD.dual_level():
                delta = fwAD.make_dual(torch.zeros(n, 4, dtype=hist.dtype), tangent)
                h = hist + delta[None, :, :2] - lag * delta[None, :, 2:]
                mu, *_ = predict_sequence_with_gaussian(self.model, h, L, steps)
                G[..., d] = fwAD.unpack_dual(mu).tangent.numpy()
        return G

    @torch.no_grad()
    def predict(self, tracks, use_state_cov: bool = True, state_cov_mode: str = "cv", coast: str = "cv") -> Prediction:
        """use_state_cov=False drops the propagated track covariance (ablation). state_cov_mode: "cv" propagates
        the track covariance with J_k = [I, kI]; "predictor" with the predictor's own sensitivity G_k.
        coast: "cv" predicts every track from its recorded history, which for a pedestrian out of view ends in
        constant-velocity coasted points; "network" predicts a track missed for m steps from its history up to
        the last detection, m + N steps ahead, keeping the last N, with the covariance at the last detection
        propagated over m + k steps (needs PMB tracks, which keep real_hist and P_det)."""
        ids = list(tracks)
        if not ids:
            z = np.zeros((self.horizon, 0, 2))
            return Prediction([], z, np.zeros(z.shape + (2,)), np.zeros(z.shape + (2,)))
        # All pedestrians share one LSTM call, so histories are padded to a common
        # length by back-extrapolating each track with its velocity estimate.
        L = max(2, max(len(tracks[i].history) for i in ids))
        hist = np.zeros((L, len(ids), 2))
        for j, pid in enumerate(ids):
            tr = tracks[pid]
            h = np.array(tr.history)
            pad = L - len(h)
            back = h[0] - np.arange(pad, 0, -1)[:, None] * tr.x[2:]
            hist[:, j] = np.concatenate([back, h]) if pad else h
        pos = torch.from_numpy(hist.astype(np.float32))
        mu, sx, sy, rho = predict_sequence_with_gaussian(self.model, pos, L, self.horizon)
        sx, sy, rho = sx.numpy(), sy.numpy(), rho.numpy()
        cxy = rho * sx * sy
        cov_model = np.stack([np.stack([sx ** 2, cxy], -1), np.stack([cxy, sy ** 2], -1)], -2)
        P = np.stack([tracks[i].P for i in ids])
        r = np.array([getattr(tracks[i], "r", 1.0) for i in ids])
        G = None
        if state_cov_mode == "predictor":
            G = self.state_jacobian(pos)
            cov_state = np.einsum("kjab,jbc,kjdc->kjad", G, P, G)
        else:
            cov_state = cv_propagated_cov(P, self.horizon)
        if not use_state_cov:
            cov_state = np.zeros_like(cov_state)
        n_det = np.array([sum(getattr(tracks[i], "det_flags", tracks[i].history)) if hasattr(tracks[i], "det_flags")
                          else len(tracks[i].history) for i in ids], dtype=float)
        mu = mu.numpy().astype(np.float64)
        cov_model = cov_model.astype(np.float64)
        if coast == "network":
            self._network_coast(tracks, ids, mu, cov_model, cov_state, state_cov_mode, use_state_cov)
        return Prediction(ids, mu, cov_model, cov_state, r, n_det, G)

    def _network_coast(self, tracks, ids, mu, cov_model, cov_state, state_cov_mode, use_state_cov):
        """Overwrite, in place, the prediction of every track missed for m >= 1 steps by the network's rollout
        from its last real history. Tracks sharing m share one rollout, aligned on the time of that detection;
        the other tracks join it with their histories cut at that time, so the social context is consistent."""
        N = self.horizon
        gaps = {}
        for j, pid in enumerate(ids):
            tr = tracks[pid]
            m = int(getattr(tr, "unseen", 0))
            if m >= 1 and getattr(tr, "real_hist", None):
                gaps.setdefault(m, []).append(j)
        for m, members in gaps.items():
            rows, hs = [], []
            for j, pid in enumerate(ids):
                tr = tracks[pid]
                if j in members:
                    h = np.array(tr.real_hist)
                elif len(tr.history) > m:
                    h = np.array(tr.history[: len(tr.history) - m])
                else:
                    continue
                rows.append(j)
                hs.append((h, tr.x[2:]))
            L = max(2, max(len(h) for h, _ in hs))
            hist = np.zeros((L, len(rows), 2))
            for c, (h, v) in enumerate(hs):
                pad = L - len(h)
                back = h[0] - np.arange(pad, 0, -1)[:, None] * v
                hist[:, c] = np.concatenate([back, h]) if pad else h
            pos = torch.from_numpy(hist.astype(np.float32))
            mu_g, sx, sy, rho = predict_sequence_with_gaussian(self.model, pos, L, m + N)
            mu_g, sx, sy, rho = mu_g.numpy()[m:], sx.numpy()[m:], sy.numpy()[m:], rho.numpy()[m:]
            cxy = rho * sx * sy
            cov_g = np.stack([np.stack([sx ** 2, cxy], -1), np.stack([cxy, sy ** 2], -1)], -2)
            G_g = self.state_jacobian(pos, m + N)[m:] if state_cov_mode == "predictor" else None
            for j in members:
                c = rows.index(j)
                P_det = tracks[ids[j]].P_det
                mu[:, j] = mu_g[:, c]
                cov_model[:, j] = cov_g[:, c]
                if not use_state_cov:
                    cov_state[:, j] = 0.0
                elif G_g is not None:
                    cov_state[:, j] = np.einsum("kab,bc,kdc->kad", G_g[:, c], P_det, G_g[:, c])
                else:
                    for k in range(N):
                        J = np.hstack([np.eye(2), (m + k + 1) * np.eye(2)])
                        cov_state[k, j] = J @ P_det @ J.T
