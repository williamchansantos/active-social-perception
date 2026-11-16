"""Unicycle robot with an independently actuated pan camera.

State x = [px, py, theta, phi]: position, body heading, camera pan relative to
the body (camera heading = theta + phi). Input u = [v, omega, omega_cam].
Uncertain parameters p = [k_v, k_omega]: multiplicative actuation gains
(e.g. wheel radius / track width calibration), nominally 1.
Discretized with forward Euler at the planning step dt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

NX, NU, NP = 4, 3, 2


@dataclass
class RobotConfig:
    dt: float = 0.4
    radius: float = 0.3
    v_bounds: tuple = (-0.2, 1.2)
    omega_bounds: tuple = (-1.5, 1.5)
    omega_cam_bounds: tuple = (-2.5, 2.5)
    phi_bounds: tuple = (-2.0, 2.0)  # camera pan limits relative to body (rad)
    p_nominal: np.ndarray = field(default_factory=lambda: np.ones(NP))


def dynamics(x, u, p, dt):
    """Works on numpy arrays and casadi symbols alike."""
    lib = ca if isinstance(x, (ca.MX, ca.SX)) else np
    px, py, th, ph = x[0], x[1], x[2], x[3]
    v, om, omc = u[0], u[1], u[2]
    dx = [p[0] * v * lib.cos(th), p[0] * v * lib.sin(th), p[1] * om, omc]
    if lib is ca:
        return x + dt * ca.vertcat(*dx)
    return x + dt * np.array(dx)


def jacobians(x, u, p, dt):
    """A = df/dx, B = df/du, Fp = df/dp at (x, u, p)."""
    th, v, om = x[2], u[0], u[1]
    c, s = np.cos(th), np.sin(th)
    A = np.eye(NX)
    A[0, 2] = -dt * p[0] * v * s
    A[1, 2] = dt * p[0] * v * c
    B = np.zeros((NX, NU))
    B[0, 0], B[1, 0] = dt * p[0] * c, dt * p[0] * s
    B[2, 1] = dt * p[1]
    B[3, 2] = dt
    Fp = np.zeros((NX, NP))
    Fp[0, 0], Fp[1, 0] = dt * v * c, dt * v * s
    Fp[2, 1] = dt * om
    return A, B, Fp
