"""SupeR-MPC (Zhu, Simeon, Cognetti, RA-L 2025) for a unicycle + pan camera
among predicted pedestrians, with a risk-aware active perception cost.

Uncertainty sources on the robot (Sec. IV of the paper):
  state estimate  Sigma0  -> Psi_k = A_{k-1} Psi_{k-1}                   (17)
  parameters      Sigma_p -> Pi_k = A Pi + B Theta + F_p, Theta = K Pi    (20)
  process noise   Sigma_Q -> k Sigma_Q
  Sigma^x_k = Psi Sigma0 Psi^T + Pi Sigma_p Pi^T + k Sigma_Q             (21)
  Sigma^u_k = Theta Sigma_p Theta^T                                      (22)
Pedestrian uncertainty comes from the predictor (LSTM covariance + CV-propagated
track covariance). Chance constraints:
  inputs     (25): bounds tightened by erfinv(1-2 delta) sqrt(2 Sigma^u_ii)
  collisions (27): a^T (c_r - c_j) - r >= kappa_j sqrt(2 a^T (Sigma_r + Sigma_j) a)
With a PMB tracker, pedestrian j exists with probability r_j, so
P(collide with j) = r_j P(collide | exists) <= delta gives the per-pedestrian
kappa_j = erfinv(1 - 2 min(delta / r_j, 1/2)); tracks with r_j <= delta need no
constraint. Undetected pedestrians (PPP, negative-binomial count) get their own
chance constraint at every horizon step:
  P(>=1 undetected pedestrian within r of c_r,k) <= delta_u
  <=>  sum_i m_ik softdisk(r - ||c_r,k - q_ik||) <= M_max(delta_u)
or, with unseen_joint=True, ONE joint constraint per plan (Boole's inequality over the horizon):
  sum_k P_k <= delta_u,  P_k = 1 - (b / (b + M_k))^a,  M_k = sum_i m_ik softdisk(...)
which bounds each plan's risk instead of each step's (8 per-step budgets of
delta_u let a plan accept up to ~8 delta_u, compounded by replanning).
with (q_ik, m_ik) the predict-only forecast of the undetected intensity near the
warm-start path (no future measurements are assumed, as for every constraint).
The feedback gains K (the paper's ComputeGains, obtained there from the
solver's primal-dual structure) are computed here as time-varying LQR gains
along the nominal trajectory.

Active perception (camera_mode="active"): constraints always use the current
information, so safety never relies on views that have not been taken yet. The
cost instead rewards plans whose camera views shrink the *future* constraint
back-off. With a smooth visibility model vis_jk (FOV cone x range) and
first-sighting probabilities pi_jm = vis_jm prod_{l<m} (1 - vis_jl):
  Sigma~_jk = Sigma_model_jk + (prod_{m<=k} (1 - vis_jm)) Sigma_state_jk
              + sum_{m<=k} pi_jm J_{k-m} P_ss J_{k-m}^T
  cost     += w_perc * rel_jk * erfinv(1-2 delta) sqrt(2 a_jk^T (Sigma_r,k + Sigma~_jk) a_jk)
P_ss is the steady-state track covariance under regular detections. Only the
perception_peds most collision-relevant pedestrians enter this term.

With a PMB tracker the cost also covers pedestrians nobody has seen yet. Each
forecast undetected mass m_ik at q_ik on the step-k path moves with its
channel velocity v_ik, so at an earlier planned view m it sits at
q_ik - (k - m) v_ik and is cleared with probability p_D vis_m:
  cost += w_unseen * sum_k sum_i w_ik m_ik prod_{m<=k} (1 - p_D vis_m(q_ik - (k-m) v_ik)) / M_max
i.e. the expected undetected mass left in the path's collision disks after the
planned views, in units of the unseen chance-constraint limit.

In both terms the collision-relevance weights (rel_jk, w_ik) and directions a_jk
are fixed from the warm-start plan, so the only way to lower the cost is to look
at the right place at the right time, not to move away from it.
"""
from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Optional

import casadi as ca
import numpy as np
from scipy.special import erfinv

from .camera import CameraConfig
from .prediction import Prediction, cv_propagated_cov
from .robot import NP, NU, NX, RobotConfig, dynamics, jacobians
from .tracking import H, TrackerConfig, cwna

FAR = 1e3  # position of unused pedestrian slots


@dataclass
class MPCConfig:
    N: int = 8  # 3.2 s, as in the paper's hardware experiments
    max_peds: int = 10
    consider_radius: float = 8.0
    robust: bool = True  # False: deterministic MPC (delta = 0.5, no sensitivities)
    delta: float = 0.05
    delta_unseen: float = 0.01  # risk budget for undetected pedestrians per horizon step
    unseen_slots: int = 30  # forecast point masses per horizon step
    perception_peds: int = 4  # most collision-relevant tracked pedestrians in the perception cost
    w_perception_unseen: float = 2.0  # weight of the undetected-pedestrian perception term
    perception_unseen_slots: int = 8  # highest-weight undetected masses per step in the perception cost
    unseen_kernel_width: float = 0.25  # m, softness of the collision disk for point masses
    unseen_joint: bool = False  # True: one joint risk budget per plan (ablation); False: delta_unseen per step
    ped_radius: float = 0.3
    w_goal: float = 1.0
    w_terminal: float = 10.0
    R: tuple = (0.1, 0.3, 0.05)  # input effort weights (v, omega, omega_cam)
    w_slack: float = 1e3
    camera_mode: str = "active"  # "fixed" | "scan" | "nearest" | "ttc" | "active" | "decoupled" | "oracle_gaze"
    # "ttc" (baseline): point at the most collision-relevant tracked pedestrian (largest relevance to the warm-start
    #   path summed over the horizon, a time-to-contact proxy). NOTE: results files before 2026-10-13 label this
    #   behaviour "nearest" / supermpc-nearest; it was the only "nearest" implementation until then.
    # "nearest" (baseline): point at the tracked pedestrian closest to the robot (distance now).
    # "decoupled" (baseline): plan the motion first without the camera (as "fixed"), then optimize only the
    #   camera with the active perception cost for that motion (two stages instead of one joint problem).
    perception_objective: str = "risk"  # "risk": chance-constraint back-off; "entropy": information gain;
    # "hybrid": w_perception * risk term + w_entropy * information gain on the tracked pedestrians, and
    # undetected masses weighted by unseen_floor + (1 - unseen_floor) * path relevance
    w_entropy: float = 1.0  # hybrid: weight of the information-gain term
    unseen_floor: float = 0.5  # hybrid: 0 = only masses near the path count (risk), 1 = all equally (entropy)
    risk_form: str = "sqrt"  # hybrid: "sqrt" back-off (risk), or "log" (scale-free: halving any back-off pays the same)
    relevance_floor: float = 0.0  # hybrid: added to the collision relevance of every tracked pedestrian
    # Relevance of undetected masses in the perception cost: "timed" = near the planned position at the same
    # step; "swept" = near the planned path at any step (pedestrians crossing it a little earlier or later count)
    unseen_relevance: str = "timed"
    unseen_swept_margin: float = 0.5  # m, added to the collision radius for "swept"
    # Observation-aware unseen constraint: an undetected mass counts only if the planned camera views
    # before the robot gets there would not have detected it (couples speed and gaze: do not drive
    # faster than you can see). False: open-loop constraint, as in the evaluations so far.
    unseen_obs_aware: bool = False
    unseen_constraint: bool = True  # False: no undetected-pedestrian chance constraint (ablation)
    existence_weighting: bool = True  # False: every track constrained as if it surely exists (ablation)
    # Camera sub-steps per control step: the camera rate and the observations run cam_substeps times
    # faster than the robot control and pedestrian prediction (0.4 s). Per step the camera gets
    # cam_substeps rates with mean U[2] (so the step dynamics are unchanged); a pedestrian counts as
    # seen in a step if any sub-step view sees it.
    cam_substeps: int = 1
    # Approaching-pedestrian term: keep tracked pedestrians who are closing in on the robot in view
    # (weight exp(-time to contact / approach_ttc) for those within approach_range), over the next
    # approach_steps steps. 0 disables it.
    # Detection-history calibration (calibrate_detection_history.py): in the perception cost, the expected
    # prediction covariance of a tracked pedestrian is scaled by the measured ratio s(n_eff, k) = E[err^2] /
    # E[tr Sigma] (piecewise linear in n_eff), n_eff = detections in its history + expected detections from
    # the planned views, so looking is credited for the better prediction a longer detected history gives.
    # None disables it.
    history_calibration: Optional[str] = None
    # How a track's state covariance P is propagated over the horizon: "cv" = J_k P J_k^T with J_k = [I, kI];
    # "predictor" = G_k P G_k^T with G_k the predictor's own sensitivity d(mean_k)/d(state) (Prediction.state_jac).
    # Set here; run_nav passes it to the predictor, and the perception cost's post-detection covariance follows it.
    state_cov_mode: str = "cv"
    # How a pedestrian out of view is predicted: "cv" = from its history ending in constant-velocity coasted points;
    # "network" = the network's rollout from the last detection, shifted to now (TrajectoryPredictor.predict).
    coast_mode: str = "cv"
    # Static obstacles (walls): per horizon step, the wall_points obstacle points nearest the warm-start position
    # (SupeRMPC.wall_fn -> points o and unit normals n into free space) give half-plane constraints
    # n^T (p - o) >= wall_clearance + kappa * robot position sigma along n (linear, so a plan cannot jump through
    # a wall the way it can with point-distance constraints); softened with a slack penalized by w_wall_slack.
    # 0 disables (as all 2D evaluations).
    wall_points: int = 0
    wall_clearance: float = 0.30
    w_wall_slack: float = 1e4
    w_approach: float = 0.0
    approach_ttc: float = 2.0  # s
    approach_range: float = 4.0  # m
    approach_steps: int = 3
    robot_sensitivity: bool = True  # False: ignore robot uncertainty propagation (ablation)
    w_perception: float = 2.0
    # Crowd-density exploration (option 2; 0 = off, as in all pre-registered results): reward views that reduce the
    # uncertainty of the PMB crowd multiplier lambda ~ Gamma(a, b). Covering expected undetected base mass E with the
    # planned views shrinks Var(lambda) by b / (b + E), so the term is -w_explore * log(1 + E / b): large while b is
    # small (little evidence, weak prior), fading as evidence accumulates. E sums, over the explore_slots largest
    # undetected masses within explore_radius (moving with their flow), the probability that the planned views see
    # each at least once.
    w_explore: float = 0.0
    explore_slots: int = 12
    explore_radius: float = 5.0
    relevance_length: float = 1.5  # m, decay of collision relevance with clearance
    w_cam_forward: float = 0.05  # weak prior to look ahead when nothing else matters
    vis_sharpness_angle: float = 30.0  # on cos(bearing) - cos(fov/2)
    vis_sharpness_range: float = 4.0  # 1/m
    fresh_range: float = 3.0  # range assumed for P_ss
    scan_amplitude: float = 1.0  # rad
    scan_period: float = 4.0  # s
    Sigma0: np.ndarray = field(default_factory=lambda: np.diag([0.05 ** 2, 0.05 ** 2, 0.02 ** 2, 0.005 ** 2]))
    Sigma_p: np.ndarray = field(default_factory=lambda: np.diag([0.1 ** 2, 0.1 ** 2]))
    Sigma_Q: np.ndarray = field(default_factory=lambda: np.diag([0.01 ** 2, 0.01 ** 2, 0.005 ** 2, 0.002 ** 2]))
    Q_lqr: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0, 1.0, 0.1]))
    R_lqr: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 1.0]))


def sigmoid(z):
    """Overflow-free logistic (unused pedestrian slots sit ~1 km away)."""
    return 0.5 * (1 + ca.tanh(0.5 * z))


def cov3(S):
    """(..., 2, 2) -> (..., 3) as [xx, xy, yy]."""
    return np.stack([S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]], -1)


def steady_state_track_cov(R: np.ndarray, tcfg: TrackerConfig, iters: int = 200) -> np.ndarray:
    F, Q = cwna(1.0, tcfg.qc)
    P = np.diag([R[0, 0], R[1, 1], tcfg.v0_std ** 2, tcfg.v0_std ** 2])
    for _ in range(iters):
        P = F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        P = (np.eye(4) - K @ H) @ P
    return P


# ----------------------------------------------------------------------------- sensitivities

def lqr_gains(As, Bs, Q, R):
    P = Q.copy()
    Ks = [None] * len(As)
    for k in reversed(range(len(As))):
        A, B = As[k], Bs[k]
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
        Ks[k] = K
    return Ks


def propagate_uncertainty(X, U, p_hat, cfg: MPCConfig, dt):
    """Sigma^x_k (k = 1..N) and Sigma^u_k (k = 0..N-1) along nominal (X, U), Eqs. 17-22."""
    N = U.shape[1]
    jac = [jacobians(X[:, k], U[:, k], p_hat, dt) for k in range(N)]
    Ks = lqr_gains([j[0] for j in jac], [j[1] for j in jac], cfg.Q_lqr, cfg.R_lqr)
    Psi, Pi = np.eye(NX), np.zeros((NX, NP))
    Sx, Su = [], []
    for k in range(N):
        A, B, Fp = jac[k]
        Theta = -Ks[k] @ Pi  # du_k/dp through feedback u = u~ - K (x^ - x~)
        Su.append(Theta @ cfg.Sigma_p @ Theta.T)
        Pi = A @ Pi + B @ Theta + Fp
        Psi = A @ Psi
        Sx.append(Psi @ cfg.Sigma0 @ Psi.T + Pi @ cfg.Sigma_p @ Pi.T + (k + 1) * cfg.Sigma_Q)
    return np.stack(Sx), np.stack(Su)


# ----------------------------------------------------------------------------- planner

class SupeRMPC:
    def __init__(self, cfg: MPCConfig, rcfg: RobotConfig, cam: CameraConfig, tcfg: TrackerConfig):
        self.cfg, self.rcfg, self.cam = cfg, rcfg, cam
        self.kappa = float(erfinv(1 - 2 * cfg.delta))
        Rfresh = cam.sigma(cfg.fresh_range) ** 2 * np.eye(2)
        P_ss = steady_state_track_cov(Rfresh, tcfg)
        self.P_ss = P_ss
        self.fresh = cov3(cv_propagated_cov(P_ss[None], cfg.N)[:, 0])  # J_l P_ss J_l^T, l = 1..N
        self.fresh = np.vstack([cov3(P_ss[:2, :2])[None], self.fresh])  # prepend l = 0
        self.hist_cal = None
        if cfg.history_calibration:
            c = np.load(cfg.history_calibration)
            self.hist_cal = dict(table=np.asarray(c["ratio"], float))  # (9, K): rows n_det = 0..8
        self._build()
        self._motion = None
        if cfg.camera_mode == "decoupled":  # stage-1 motion planner: same problem without the camera
            import dataclasses
            self._motion = SupeRMPC(dataclasses.replace(cfg, camera_mode="fixed"), rcfg, cam, tcfg)
            self._um, self._um_t = None, None
        self.X_ws = self.U_ws = None
        self.wall_fn = None  # path (N, 2) -> points, normals (N, K, 2) of nearby walls (cfg.wall_points > 0)
        self.explore_fn = None  # center (2,) -> pos, vel, mass, b of undetected masses (cfg.w_explore > 0)
        self._nearest = None  # camera target (camera_mode="nearest" / "oracle_gaze")
        self.gt = None  # (positions (n, 2), velocities (n, 2)) of ALL pedestrians, for camera_mode="oracle_gaze"
        self.Sx = self.Su = None
        self.step_count = 0

    # ------------------------------------------------------------------ NLP
    def _build(self):
        cfg, rc, N, M = self.cfg, self.rcfg, self.cfg.N, self.cfg.max_peds
        opti = ca.Opti()
        Pu = cfg.unseen_slots
        X, U, S = opti.variable(NX, N + 1), opti.variable(NU, N), opti.variable(M, N)
        Su = opti.variable(1, N)
        par = dict(
            x0=opti.parameter(NX), goal=opti.parameter(2), p=opti.parameter(NP),
            mu=opti.parameter(2 * M, N), cm=opti.parameter(3 * M, N), cs=opti.parameter(3 * M, N),
            rc=opti.parameter(3, N), ut=opti.parameter(NU, N), rel=opti.parameter(M, N),
            ahat=opti.parameter(2 * M, N), cam=opti.parameter(1, N), kappa=opti.parameter(M),
            kappa_p=opti.parameter(M), live=opti.parameter(M), app=opti.parameter(M), nd=opti.parameter(M),
            fr=opti.parameter(3 * M, N + 1), uq=opti.parameter(2 * Pu, N), uv=opti.parameter(2 * Pu, N), um=opti.parameter(Pu, N),
            ur=opti.parameter(1, N), umax=opti.parameter(), ua=opti.parameter(), ub=opti.parameter(),
            udelta=opti.parameter(),
            pq=opti.parameter(2 * cfg.perception_unseen_slots, N),
            pv=opti.parameter(2 * cfg.perception_unseen_slots, N),
            pm=opti.parameter(cfg.perception_unseen_slots, N),
        )
        if cfg.w_explore > 0:
            par["eq"] = opti.parameter(2 * cfg.explore_slots)
            par["ev"] = opti.parameter(2 * cfg.explore_slots)
            par["em"] = opti.parameter(cfg.explore_slots)
            par["eb"] = opti.parameter()
        if cfg.wall_points:
            par["wq"] = opti.parameter(2 * cfg.wall_points, N)
            par["wn"] = opti.parameter(2 * cfg.wall_points, N)
        r = rc.radius + cfg.ped_radius
        lo = [rc.v_bounds[0], rc.omega_bounds[0], rc.omega_cam_bounds[0]]
        hi = [rc.v_bounds[1], rc.omega_bounds[1], rc.omega_cam_bounds[1]]

        opti.subject_to(X[:, 0] == par["x0"])
        cost = 0
        for k in range(N):
            opti.subject_to(X[:, k + 1] == dynamics(X[:, k], U[:, k], par["p"], rc.dt))
            for i in range(NU):
                if cfg.camera_mode == "decoupled" and i < 2:
                    continue  # motion fixed to the stage-1 plan, which already satisfies these bounds
                opti.subject_to(opti.bounded(lo[i] + par["ut"][i, k], U[i, k], hi[i] - par["ut"][i, k]))
            opti.subject_to(opti.bounded(rc.phi_bounds[0], X[3, k + 1], rc.phi_bounds[1]))
            cost += cfg.w_goal * ca.sumsqr(X[:2, k + 1] - par["goal"])
            cost += sum(cfg.R[i] * U[i, k] ** 2 for i in range(NU))
            cost += cfg.w_cam_forward * (1 - ca.cos(X[3, k + 1]))
        cost += cfg.w_terminal * ca.sumsqr(X[:2, N] - par["goal"])
        active_cam = cfg.camera_mode in ("active", "decoupled")
        if not active_cam:
            opti.subject_to(U[2, :] == par["cam"])
        if cfg.camera_mode == "decoupled":  # stage 2: the motion is fixed to the stage-1 plan
            par["umot"] = opti.parameter(2, N)
            opti.subject_to(U[:2, :] == par["umot"])
        Sc = cfg.cam_substeps
        self.Dc = None
        self._sub = [[X[:, k + 1]] for k in range(N)]  # camera poses at the sub-step views of step k
        if Sc > 1:
            # Sub-step camera rates U[2, k] + Dc[s, k], s = 0..Sc-1, with sum_s Dc[s, k] = 0.
            Dc = opti.variable(Sc, N)
            self.Dc = Dc
            if not active_cam:
                opti.subject_to(ca.vec(Dc) == 0)
            for k in range(N):
                opti.subject_to(ca.sum1(Dc[:, k]) == 0)
                cost += cfg.R[2] * ca.sumsqr(Dc[:, k]) / Sc
                phi, subs = X[3, k], []
                for s in range(Sc):
                    w = U[2, k] + Dc[s, k]
                    opti.subject_to(opti.bounded(lo[2] + par["ut"][2, k], w, hi[2] - par["ut"][2, k]))
                    phi = phi + w * rc.dt / Sc
                    f = (s + 1) / Sc
                    if s < Sc - 1:
                        opti.subject_to(opti.bounded(rc.phi_bounds[0], phi, rc.phi_bounds[1]))
                        base = (1 - f) * X[:3, k] + f * X[:3, k + 1]  # position and heading, interpolated
                        subs.append(ca.vertcat(base, phi))
                    else:
                        subs.append(X[:, k + 1])
                self._sub[k] = subs

        # Collision chance constraints (27), softened with heavily penalized slacks so a
        # replayed (non-reactive) crowd walking into the robot cannot make the NLP infeasible.
        opti.subject_to(ca.vec(S) >= 0)
        cost += cfg.w_slack * ca.sum1(ca.sum2(S))
        for j in range(M):
            for k in range(N):
                mu = par["mu"][2 * j : 2 * j + 2, k]
                Sig = par["cm"][3 * j : 3 * j + 3, k] + par["cs"][3 * j : 3 * j + 3, k] + par["rc"][:, k]
                d = X[:2, k + 1] - mu
                n = ca.sqrt(ca.sumsqr(d) + 1e-6)
                a = d / n
                quad = a[0] ** 2 * Sig[0] + 2 * a[0] * a[1] * Sig[1] + a[1] ** 2 * Sig[2]
                opti.subject_to(n - r - par["kappa"][j] * ca.sqrt(2 * quad + 1e-9) + S[j, k] >= 0)

        # Walls: distance to each nearby obstacle point >= clearance + kappa * robot position sigma along it.
        self.S_wall = None
        if cfg.wall_points:
            Sw = opti.variable(1, N)
            self.S_wall = Sw
            opti.subject_to(Sw >= 0)
            cost += cfg.w_wall_slack * ca.sum2(Sw)
            for k in range(N):
                Sr = par["rc"][:, k]
                for j in range(cfg.wall_points):
                    o, a = par["wq"][2 * j : 2 * j + 2, k], par["wn"][2 * j : 2 * j + 2, k]
                    quad = a[0] ** 2 * Sr[0] + 2 * a[0] * a[1] * Sr[1] + a[1] ** 2 * Sr[2]
                    opti.subject_to(ca.dot(a, X[:2, k + 1] - o) - cfg.wall_clearance
                                    - self.kappa * ca.sqrt(2 * quad + 1e-9) + Sw[0, k] >= 0)

        # Undetected-pedestrian chance constraint: per step, or one joint budget per plan.
        opti.subject_to(ca.vec(Su) >= 0)
        cost += cfg.w_slack * ca.sum2(Su)
        risk = 0
        for k in range(N):
            mass = 0
            for i in range(Pu):
                q = par["uq"][2 * i : 2 * i + 2, k]
                d = ca.sqrt(ca.sumsqr(X[:2, k + 1] - q) + 1e-6)
                term = par["um"][i, k] * sigmoid((par["ur"][0, k] - d) / cfg.unseen_kernel_width)
                if cfg.unseen_obs_aware:
                    v = par["uv"][2 * i : 2 * i + 2, k]
                    for m in range(k):  # planned views at steps 1..k, before the robot reaches step k + 1
                        for vis in self._step_views(m, lambda f, q=q, v=v, m=m, k=k: q - (k - m) * v - (1 - f) * v):
                            term = term * (1 - self.cam.p_detect * vis)
                mass += term
            if cfg.unseen_joint:
                risk += 1 - (par["ub"] / (par["ub"] + mass)) ** par["ua"]
            elif cfg.unseen_constraint:
                opti.subject_to(mass - Su[0, k] <= par["umax"])
        if cfg.unseen_joint:
            opti.subject_to(risk - ca.sum2(Su) <= par["udelta"])

        if active_cam:
            cost += self._perception_cost(X, par)
            if cfg.w_approach > 0:
                for j in range(M):
                    for k in range(min(cfg.approach_steps, N)):
                        mu_k = par["mu"][2 * j : 2 * j + 2, k]
                        for vis in self._step_views(k, lambda f, b=mu_k: b):
                            cost += cfg.w_approach * par["app"][j] * (1 - vis) / len(self._sub[k])
            cost += cfg.w_perception_unseen * self._unseen_perception_cost(X, par)
            if cfg.w_explore > 0:
                cost -= cfg.w_explore * self._explore_gain(par)

        opti.minimize(cost)
        # expand: evaluate the MX graph as SX, ~10x faster for these many scalar terms.
        opti.solver("ipopt", {"expand": True, "print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
                              "ipopt.max_iter": 300, "ipopt.tol": 1e-4, "ipopt.warm_start_init_point": "yes"})
        self.opti, self.X, self.U, self.S, self.S_unseen, self.par = opti, X, U, S, Su, par

    def _fresh(self, par, j, lag):
        """Post-detection covariance [xx, xy, yy] of pedestrian j, lag steps after a view."""
        if self.cfg.state_cov_mode != "predictor":
            return self.fresh[lag]  # constant: identical to the original CV model
        return par["fr"][3 * j: 3 * j + 3, lag]

    def _step_views(self, k, target_at):
        """Visibilities of the sub-step views of step k (one per camera sub-step); target_at(f) is the
        target's position at fraction f of the step (f = 1: the end of the step, as the predictions)."""
        Sc = self.cfg.cam_substeps
        if Sc == 1:
            return [self._visibility(self._sub[k][0], target_at(1.0))]
        return [self._visibility(self._sub[k][s], target_at((s + 1) / Sc)) for s in range(Sc)]

    def _visibility(self, xk, target):
        """Smooth FOV x range visibility of a world point from planned state xk."""
        cfg = self.cfg
        heading = ca.vertcat(ca.cos(xk[2] + xk[3]), ca.sin(xk[2] + xk[3]))
        d = target - xk[:2]
        dist = ca.sqrt(ca.sumsqr(d) + 1e-6)
        cos_half = np.cos(np.radians(self.cam.fov_deg) / 2)
        vis = sigmoid(cfg.vis_sharpness_angle * (ca.dot(heading, d) / dist - cos_half))
        return vis * sigmoid(cfg.vis_sharpness_range * (self.cam.max_range - dist))

    def _explore_gain(self, par):
        """log(1 + E / b): information about the crowd multiplier from the planned views (see MPCConfig.w_explore)."""
        cfg, N = self.cfg, self.cfg.N
        E = 0
        for i in range(cfg.explore_slots):
            q, v = par["eq"][2 * i : 2 * i + 2], par["ev"][2 * i : 2 * i + 2]
            miss = 1
            for m in range(N):
                for vis in self._step_views(m, lambda f, q=q, v=v, m=m: q + (m + f) * v):
                    miss = miss * (1 - self.cam.p_detect * vis)
            E = E + par["em"][i] * (1 - miss)
        return ca.log(1 + E / par["eb"])

    def _unseen_perception_cost(self, X, par):
        """pm = w_ik m_ik (collision-weighted mass) for the highest-weight masses per step."""
        cfg, N = self.cfg, self.cfg.N
        cost = 0
        for k in range(N):
            for i in range(cfg.perception_unseen_slots):
                q, v = par["pq"][2 * i : 2 * i + 2, k], par["pv"][2 * i : 2 * i + 2, k]
                remain = 1
                for m in range(k + 1):  # planned views at steps 1..k+1 (X[:, m + 1])
                    for vis in self._step_views(m, lambda f, q=q, v=v, m=m, k=k: q - (k - m) * v - (1 - f) * v):
                        remain = remain * (1 - self.cam.p_detect * vis)
                cost += par["pm"][i, k] * remain
        return cost / par["umax"]

    def _perception_cost(self, X, par):
        """w_perception * (risk or entropy term) [+ w_entropy * entropy term for "hybrid"]."""
        cfg, N = self.cfg, self.cfg.N
        cost, entropy = 0, 0
        hc = self.hist_cal
        for j in range(min(cfg.perception_peds, cfg.max_peds)):
            miss = 1  # prod_{m<=k} (1 - vis_jm)
            first_seen = []  # pi_jm
            n_eff = par["nd"][j]  # detections in the history, plus expected detections from planned views
            for k in range(N):
                mu_k = par["mu"][2 * j : 2 * j + 2, k]
                mu_p = par["mu"][2 * j : 2 * j + 2, max(k - 1, 0)]
                views = self._step_views(k, lambda f, a=mu_p, b=mu_k: b if f == 1.0 else (1 - f) * a + f * b)
                if len(views) == 1:
                    vis = views[0]
                else:  # seen during the step if any sub-step view sees it
                    unseen_all = 1
                    for v_s in views:
                        unseen_all = unseen_all * (1 - v_s)
                    vis = 1 - unseen_all
                first_seen.append(miss * vis)
                miss = miss * (1 - vis)
                if hc is None:  # (summation order kept as in the original: runs are bit-reproducible)
                    Sig = par["cm"][3 * j : 3 * j + 3, k] + miss * par["cs"][3 * j : 3 * j + 3, k] + par["rc"][:, k]
                    for m in range(k + 1):
                        Sig = Sig + first_seen[m] * self._fresh(par, j, k - m)
                else:
                    Sig = par["cm"][3 * j : 3 * j + 3, k] + miss * par["cs"][3 * j : 3 * j + 3, k]
                    for m in range(k + 1):
                        Sig = Sig + first_seen[m] * self._fresh(par, j, k - m)
                    n_eff = n_eff + self.cam.p_detect * vis
                    tab = hc["table"][:, k]
                    scale = tab[0]  # piecewise-linear interpolation in n_eff, constant beyond n = 8
                    for i in range(len(tab) - 1):
                        scale = scale + (tab[i + 1] - tab[i]) * ca.fmin(ca.fmax(n_eff - i, 0), 1)
                    Sig = scale * Sig + par["rc"][:, k]
                if cfg.perception_objective == "entropy":
                    # Information gain: entropy of the expected covariance, no collision weighting.
                    cost += par["rel"][j, k] * 0.5 * ca.log(Sig[0] * Sig[2] - Sig[1] ** 2 + 1e-9)
                    continue
                a = par["ahat"][2 * j : 2 * j + 2, k]
                quad = a[0] ** 2 * Sig[0] + 2 * a[0] * a[1] * Sig[1] + a[1] ** 2 * Sig[2]
                if cfg.perception_objective == "hybrid":
                    # Risk where risk is known (relevance-weighted back-off) plus information gain
                    # everywhere (uniform entropy), so pedestrians become known before they matter.
                    back = (0.5 * ca.log(2 * quad + 1e-9) if cfg.risk_form == "log" else ca.sqrt(2 * quad + 1e-9))
                    cost += par["rel"][j, k] * par["kappa_p"][j] * back
                    entropy += par["live"][j] * 0.5 * ca.log(Sig[0] * Sig[2] - Sig[1] ** 2 + 1e-9)
                    continue
                cost += par["rel"][j, k] * par["kappa_p"][j] * ca.sqrt(2 * quad + 1e-9)
        return cfg.w_perception * cost + cfg.w_entropy * entropy

    # ------------------------------------------------------------------ online
    def _camera_command(self, x0, t, ut):
        cfg, dt, N = self.cfg, self.rcfg.dt, self.cfg.N
        lo, hi = self.rcfg.omega_cam_bounds[0] + ut[2], self.rcfg.omega_cam_bounds[1] - ut[2]
        if cfg.camera_mode == "scan":
            w = 2 * np.pi / cfg.scan_period
            target = cfg.scan_amplitude * np.sin(w * (t + dt * np.arange(N + 1)))
            cmd = np.diff(target) / dt
            cmd[0] = (target[1] - x0[3]) / dt
        elif cfg.camera_mode in ("nearest", "ttc", "oracle_gaze") and self._nearest is not None:
            # Heuristic: point the camera at the nearest constrained pedestrian.
            d = self._nearest - x0[:2]
            target = (np.arctan2(d[1], d[0]) - x0[2] + np.pi) % (2 * np.pi) - np.pi
            target = np.clip(target, *self.rcfg.phi_bounds)
            cmd = np.zeros(N)
            cmd[0] = (target - x0[3]) / dt
        else:
            cmd = np.zeros(N)
            cmd[0] = -x0[3] / dt
        return np.clip(cmd, lo, hi)[None, :]

    def _oracle_gaze_target(self, horizon_margin: float = 1.5, time_weight: float = 0.3):
        """Privileged gaze (upper bound for camera pointing with this FOV): the TRUE pedestrian,
        detected or not, whose constant-velocity extrapolation comes closest to the warm-start path
        soonest (score = closest approach + time_weight * seconds); None (look ahead) if nobody comes
        within horizon_margin."""
        if self.gt is None or len(self.gt[0]) == 0:
            return None
        q, v = self.gt
        dt, N = self.rcfg.dt, self.cfg.N
        k = np.arange(1, N + 1)
        traj = q[:, None, :] + v[:, None, :] * (k * dt)[None, :, None]  # (n, N, 2)
        d = np.linalg.norm(traj - self.X_ws[:2, 1:].T[None], axis=-1)  # (n, N)
        kmin = d.argmin(1)
        dmin = d[np.arange(len(q)), kmin]
        score = dmin + time_weight * (kmin + 1) * dt
        score[dmin > horizon_margin + self.rcfg.radius + self.cfg.ped_radius] = np.inf
        j = int(np.argmin(score))
        return q[j] if np.isfinite(score[j]) else None

    def _select_peds(self, x0, pred: Prediction):
        """Nearest pedestrians within range whose existence exceeds delta (others
        cannot violate their chance constraint)."""
        if not pred.ids:
            return []
        dist = np.linalg.norm(pred.mean[0] - x0[:2], axis=-1)
        order = [j for j in np.argsort(dist)
                 if dist[j] < self.cfg.consider_radius
                 and (pred.existence[j] > self.cfg.delta or not self.cfg.existence_weighting)]
        return order[: self.cfg.max_peds]

    def existence_kappa(self, r: np.ndarray) -> np.ndarray:
        """kappa_j = erfinv(1 - 2 min(delta / r_j, 1/2))."""
        if not self.cfg.existence_weighting:
            r = np.ones_like(r)
        return erfinv(1 - 2 * np.minimum(self.cfg.delta / np.maximum(r, 1e-9), 0.5))

    def solve(self, x0, goal, pred: Prediction, t: float, unseen_fn=None):
        """unseen_fn(path (N, 2)) -> (pos, mass, vel, mass_limit): forecast of the
        undetected intensity (PMB tracker); None disables the unseen constraint."""
        cfg, N, M = self.cfg, self.cfg.N, self.cfg.max_peds
        p_hat = self.rcfg.p_nominal
        if self.X_ws is None:
            self.X_ws = np.tile(x0[:, None], (1, N + 1))
            self.U_ws = np.zeros((NU, N))
        self.X_ws[:, 0] = x0
        if cfg.camera_mode == "decoupled" and self._um_t != t:
            # stage 1: motion without the camera; stage 2 (below) keeps it and optimizes the camera only
            self._motion.wall_fn, self._motion.gt = self.wall_fn, self.gt
            _, info_m = self._motion.solve(x0, goal, pred, t, unseen_fn)
            self._um, self._um_t = np.asarray(info_m["U"])[:2].copy(), t
            self.X_ws[:3] = np.asarray(info_m["X"])[:3]
            self.U_ws[:2] = self._um

        sel = self._select_peds(x0, pred)
        mu = np.full((N, M, 2), FAR)
        cm = np.tile(np.array([1e-4, 0.0, 1e-4]), (N, M, 1))
        cs = np.zeros((N, M, 3))
        if sel:
            mu[:, : len(sel)] = pred.mean[:, sel]
            cm[:, : len(sel)] = cov3(pred.cov_model[:, sel])
            cs[:, : len(sel)] = cov3(pred.cov_state[:, sel])
        diff = self.X_ws[:2, 1:].T[:, None, :] - mu  # (N, M, 2), from the warm-start plan
        dist = np.linalg.norm(diff, axis=-1)
        rel = np.exp(-np.maximum(dist - self.rcfg.radius - cfg.ped_radius, 0) / cfg.relevance_length)
        rel[:, len(sel):] = 0
        if cfg.perception_objective == "entropy":
            rel[:, : len(sel)] = 1.0  # information gain: every nearby pedestrian counts equally
        if cfg.perception_objective == "hybrid":
            rel[:, : len(sel)] += cfg.relevance_floor
        # Most collision-relevant pedestrians first: only the leading slots enter the perception cost.
        order = np.argsort(-rel[:, : len(sel)].sum(0))
        if cfg.perception_objective == "entropy":
            order = np.arange(len(sel))  # already nearest first
        elif cfg.perception_objective == "hybrid":
            # Interleave most-relevant and nearest, so the slots cover both risk and exploration.
            merged = []
            for pair in zip(order, np.arange(len(sel))):
                for i in pair:
                    if i not in merged:
                        merged.append(i)
            order = np.array(merged, dtype=int)
        perm = np.concatenate([order, np.arange(len(sel), M)]).astype(int)
        sel_by_dist = list(sel)  # nearest first (_select_peds)
        sel = [sel[i] for i in order]
        mu, cm, cs, diff, dist, rel = mu[:, perm], cm[:, perm], cs[:, perm], diff[:, perm], dist[:, perm], rel[:, perm]
        ahat = diff / np.maximum(dist, 1e-6)[..., None]

        self._nearest = None
        if sel and cfg.camera_mode == "ttc":  # most collision-relevant (relevance order)
            self._nearest = pred.mean[0, sel[0]]
        if sel and cfg.camera_mode == "nearest":  # closest to the robot
            self._nearest = pred.mean[0, sel_by_dist[0]]
        if cfg.camera_mode == "oracle_gaze":
            self._nearest = self._oracle_gaze_target()
        kappa_sel = np.zeros(M)
        if sel:
            kappa_sel[: len(sel)] = self.existence_kappa(pred.existence[sel])
        Pu = cfg.unseen_slots
        uq, um, uv, ur, umax = np.full((N, Pu, 2), FAR), np.zeros((N, Pu)), np.zeros((N, Pu, 2)), np.zeros(N), 1e6
        ua, ub, udelta = 1e6, 1e6, 1e6  # constraint inactive unless a forecast is given
        if not cfg.robust or self.Sx is None:
            # D-MPC, or Algorithm 1 line 2: deterministic solve to initialize the sensitivities.
            kappa, rc, ut = np.zeros(M), np.zeros((N, 3)), np.zeros((NU, N))
        else:
            kappa = kappa_sel
            rc = cov3(self.Sx[:, :2, :2])
            ut = self.kappa * np.sqrt(2 * np.diagonal(self.Su, axis1=1, axis2=2)).T
            if not cfg.robot_sensitivity:
                rc, ut = np.zeros_like(rc), np.zeros_like(ut)
            if unseen_fn is not None:
                uq, um, uv, umax, ua, ub = unseen_fn(self.X_ws[:2, 1:].T)
                udelta = cfg.delta_unseen
                # Robot position uncertainty inflates the collision disk (2 sigma, worst axis).
                ur = self.rcfg.radius + cfg.ped_radius + 2 * np.sqrt(np.linalg.eigvalsh(self.Sx[:, :2, :2])[:, -1])
                if not cfg.robot_sensitivity:
                    ur = np.full(N, self.rcfg.radius + cfg.ped_radius)

        o, P = self.opti, self.par
        o.set_value(P["x0"], x0)
        o.set_value(P["goal"], goal)
        o.set_value(P["p"], p_hat)
        o.set_value(P["mu"], mu.transpose(1, 2, 0).reshape(2 * M, N))
        o.set_value(P["cm"], cm.transpose(1, 2, 0).reshape(3 * M, N))
        o.set_value(P["cs"], cs.transpose(1, 2, 0).reshape(3 * M, N))
        o.set_value(P["rc"], rc.T)
        o.set_value(P["ut"], ut)
        o.set_value(P["rel"], rel.T)
        o.set_value(P["ahat"], ahat.transpose(1, 2, 0).reshape(2 * M, N))
        o.set_value(P["cam"], self._camera_command(x0, t, ut))
        if cfg.w_explore > 0:
            S = cfg.explore_slots
            if self.explore_fn is not None:
                eq, ev, em, eb = self.explore_fn(x0[:2])
            else:
                eq, ev, em, eb = np.full((S, 2), FAR), np.zeros((S, 2)), np.zeros(S), 1e6
            o.set_value(P["eq"], np.asarray(eq).ravel())
            o.set_value(P["ev"], np.asarray(ev).ravel())
            o.set_value(P["em"], np.asarray(em))
            o.set_value(P["eb"], float(eb))
        if cfg.camera_mode == "decoupled":
            o.set_value(P["umot"], self._um)
        o.set_value(P["kappa"], kappa)
        o.set_value(P["kappa_p"], kappa_sel)
        o.set_value(P["uq"], uq.transpose(1, 2, 0).reshape(2 * Pu, N))
        o.set_value(P["um"], um.T)
        o.set_value(P["uv"], uv.transpose(1, 2, 0).reshape(2 * Pu, N))
        o.set_value(P["ur"], ur[None, :])
        o.set_value(P["umax"], umax)
        o.set_value(P["ua"], ua)
        o.set_value(P["ub"], ub)
        o.set_value(P["udelta"], udelta)
        # Collision-relevance of each undetected mass: soft disk around the warm-start path;
        # only the highest-weight masses per step enter the perception cost.
        gap = ur[:, None] - np.linalg.norm(uq - self.X_ws[:2, 1:].T[:, None, :], axis=-1)
        if cfg.unseen_relevance == "swept":
            path = self.X_ws[:2, :].T  # (N + 1, 2), current position included
            d_path = np.linalg.norm(uq[:, :, None, :] - path[None, None], axis=-1).min(-1)  # (N, Pu)
            gap = ur[:, None] + cfg.unseen_swept_margin - d_path
        weighted = um / (1 + np.exp(-np.clip(gap / cfg.unseen_kernel_width, -50, 50)))
        if cfg.perception_objective == "entropy":
            weighted = um.copy()  # information gain: clear unseen mass anywhere nearby
        elif cfg.perception_objective == "hybrid":
            weighted = cfg.unseen_floor * um + (1 - cfg.unseen_floor) * weighted
        live = np.zeros(M)
        live[: len(sel)] = 1.0
        o.set_value(P["live"], live)
        fr = np.zeros((M, N + 1, 3))
        fr[:] = self.fresh[None]
        if cfg.state_cov_mode == "predictor" and pred.state_jac is not None and len(sel):
            Gs = pred.state_jac[:, sel]  # (N, n_sel, 2, 4)
            post = np.einsum("kjab,bc,kjdc->kjad", Gs, self.P_ss, Gs)  # lag 1..N
            fr[: len(sel), 1:] = cov3(post).transpose(1, 0, 2)
        o.set_value(P["fr"], fr.transpose(0, 2, 1).reshape(3 * M, N + 1))
        nd = np.zeros(M)
        if len(sel):
            nd[: len(sel)] = pred.n_det[sel]  # sel indexes the prediction
        o.set_value(P["nd"], nd)
        app = np.zeros(M)
        if cfg.w_approach > 0 and len(sel) and pred.mean.shape[0] > 1:
            dt = self.rcfg.dt
            v_rob = (self.X_ws[:2, 1] - self.X_ws[:2, 0]) / dt
            for j in range(len(sel)):
                q0 = mu[0, j] - (mu[1, j] - mu[0, j])  # current position (one step before the first prediction)
                v_ped = (mu[1, j] - mu[0, j]) / dt
                d = q0 - x0[:2]
                dist = np.linalg.norm(d)
                closing = -float((v_ped - v_rob) @ d) / max(dist, 1e-6)
                if dist < cfg.approach_range and closing > 0.1:
                    app[j] = np.exp(-(dist / closing) / cfg.approach_ttc)
        o.set_value(P["app"], app)
        top = np.argsort(-weighted, axis=1)[:, : cfg.perception_unseen_slots]
        rows = np.arange(N)[:, None]
        o.set_value(P["pq"], uq[rows, top].transpose(1, 2, 0).reshape(-1, N))
        o.set_value(P["pv"], uv[rows, top].transpose(1, 2, 0).reshape(-1, N))
        o.set_value(P["pm"], weighted[rows, top].T)
        o.set_initial(self.X, self.X_ws)
        o.set_initial(self.U, self.U_ws)
        o.set_initial(self.S, 0)
        o.set_initial(self.S_unseen, 0)
        if cfg.wall_points:
            # unused slots: a point far behind the robot with the normal towards it (never active)
            wq = np.tile(self.X_ws[:2, 1:].T[:, None, :] - [FAR, 0.0], (1, cfg.wall_points, 1))
            wn = np.tile(np.array([1.0, 0.0]), (N, cfg.wall_points, 1))
            if self.wall_fn is not None:
                pts, nrm = self.wall_fn(self.X_ws[:2, 1:].T)  # (N, K, 2) each; slots with nan stay unused
                ok = np.isfinite(pts[..., 0])
                wq[ok], wn[ok] = pts[ok], nrm[ok]
            o.set_value(P["wq"], wq.transpose(1, 2, 0).reshape(2 * cfg.wall_points, N))
            o.set_value(P["wn"], wn.transpose(1, 2, 0).reshape(2 * cfg.wall_points, N))
            o.set_initial(self.S_wall, 0)
        if self.Dc is not None:
            o.set_initial(self.Dc, 0)
        t0 = time.perf_counter()
        try:
            sol = o.solve()
            Xs, Us, Ss, Sus, ok = (sol.value(self.X), sol.value(self.U), sol.value(self.S),
                                   sol.value(self.S_unseen), True)
            Dcs = sol.value(self.Dc) if self.Dc is not None else None
        except RuntimeError:
            Xs, Us, Ss, Sus, ok = (o.debug.value(self.X), o.debug.value(self.U), o.debug.value(self.S),
                                   o.debug.value(self.S_unseen), False)
            Dcs = o.debug.value(self.Dc) if self.Dc is not None else None
        # Camera rate for each sub-step of the first control step.
        cam_sub = np.full(cfg.cam_substeps, float(Us[2, 0])) if Dcs is None else             float(Us[2, 0]) + np.atleast_2d(np.asarray(Dcs).reshape(cfg.cam_substeps, -1))[:, 0]
        solve_time = time.perf_counter() - t0
        stats = o.stats()

        if cfg.robust and self.Sx is None:
            self.X_ws, self.U_ws = Xs, Us
            self.Sx, self.Su = propagate_uncertainty(Xs, Us, p_hat, cfg, self.rcfg.dt)
            return self.solve(x0, goal, pred, t, unseen_fn)

        # Algorithm 1, lines 10-12: shift the solution and recompute sensitivities on it.
        x_end = dynamics(Xs[:, -1], Us[:, -1], p_hat, self.rcfg.dt)
        self.X_ws = np.hstack([Xs[:, 1:], x_end[:, None]])
        self.U_ws = np.hstack([Us[:, 1:], Us[:, -1:]])
        if cfg.robust:
            self.Sx, self.Su = propagate_uncertainty(self.X_ws, self.U_ws, p_hat, cfg, self.rcfg.dt)
        info = dict(ok=ok, X=Xs, U=Us, max_slack=float(np.max(Ss)) if Ss.size else 0.0,
                    unseen_slack=float(np.max(Sus)), unseen_limit=umax,
                    solve_time=solve_time, iters=stats.get("iter_count", -1),
                    selected=[pred.ids[j] for j in sel],
                    wall_slack=float(np.max(o.debug.value(self.S_wall))) if self.S_wall is not None else 0.0, robot_cov=rc, kappa=kappa, slack=np.atleast_2d(Ss),
                    cam_sub=cam_sub)
        return Us[:, 0], info
