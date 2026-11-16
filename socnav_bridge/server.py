"""SupeR-MPC planning server for SocNavBench (ground-truth bridge, stage 1).

SocNavBench's joystick (joystick_py/joystick_supermpc.py, inside the container) sends the episode map and,
every control step, the robot pose and all pedestrian poses. This server runs our stack on them -- simulated
pan camera (FOV, range, occlusion, missed detections, noise), PMB tracker, rotation-augmented Social LSTM,
SupeR-MPC with active perception -- and replies with 8 velocity commands (8 x 0.05 s = one 0.4 s step).

The MPC tracks a waypoint ~2.5 m ahead on the shortest path through SocNavBench's traversible map (obstacles
inflated by the robot radius plus a margin). With --walls K, the K nearest obstacle points of each horizon step are
MPC constraints (robot centre >= --wall-clearance from them) and the waypoint is the farthest path point within
the lookahead that is in line of sight; without, walls are only avoided through the waypoint (val30).

    LOO_PREFIX=loorot python -m socnav_bridge.server --dataset univ --port 2111
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_nav.camera import CameraConfig, observe  # noqa: E402
from robot_nav.mpc import MPCConfig, SupeRMPC  # noqa: E402
from robot_nav.pmb import PMBConfig, PMBTracker, fit_scene_prior  # noqa: E402
from robot_nav.prediction import TrajectoryPredictor  # noqa: E402
from robot_nav.robot import RobotConfig  # noqa: E402
from robot_nav.tracking import TrackerConfig  # noqa: E402
from socnav_bridge.datasets import obsmat_path  # noqa: E402

DT = 0.4
SCENE = {"univ": "univ", "zara01": "zara01", "zara02": "zara02", "eth": "eth", "hotel": "hotel"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="univ", choices=list(SCENE))
    p.add_argument("--port", type=int, default=2111)
    p.add_argument("--fov", type=float, default=60.0)
    p.add_argument("--range", type=float, default=4.0)
    p.add_argument("--camera", default="active", choices=["active", "fixed", "infogain", "nearest", "ttc", "decoupled", "scan"])
    p.add_argument("--waypoint-dist", type=float, default=2.5)
    p.add_argument("--walls", type=int, default=0, help="obstacle points per horizon step in the MPC (0: no walls)")
    p.add_argument("--wall-clearance", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="results/socnav")
    p.add_argument("--vision", action="store_true",
                   help="perceive pedestrians from the rendered RGB-D images (joystick with SUPERMPC_VISION=1) "
                        "instead of the simulated camera on ground-truth positions")
    p.add_argument("--detector", default="fasterrcnn_mobilenet_v3_large_fpn")
    p.add_argument("--save-frames", action="store_true", help="vision: save each step's RGB + detections + depth "
                   "to LOG_DIR/frames/EPISODE/ (for socnav_bridge.make_video --frames)")
    p.add_argument("--mpc-set", nargs="*", default=[], help="ablations: MPCConfig overrides key=value")
    p.add_argument("--pmb-set", nargs="*", default=[], help="ablations: PMBConfig overrides key=value")
    p.add_argument("--tracker", default="pmb", choices=["pmb", "kf"], help="kf: plain Kalman tracker (ablation)")
    p.add_argument("--no-state-cov", action="store_true", help="ablation: drop the track covariance from predictions")
    p.add_argument("--persistent", action="store_true", help="keep accepting new SocNavBench connections")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


class PathField:
    """Geodesic distance to the goal on the traversible map, obstacles inflated by `inflate` m."""

    def __init__(self, trav: np.ndarray, scale: float, goal: np.ndarray, inflate: float, res: float = 0.1):
        k = max(1, int(round(res / scale)))
        self.res = scale * k
        clear = ndimage.distance_transform_edt(trav) * scale
        free = clear[::k, ::k] > inflate
        self.free = free
        H, W = free.shape
        idx = -np.ones((H, W), int)
        cells = np.argwhere(free)
        idx[free] = np.arange(len(cells))
        rows, cols, w = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = cells
            b = cells + [di, dj]
            ok = (b[:, 0] >= 0) & (b[:, 0] < H) & (b[:, 1] >= 0) & (b[:, 1] < W)
            a, b = a[ok], b[ok]
            ok = free[b[:, 0], b[:, 1]]
            a, b = a[ok], b[ok]
            rows += [idx[a[:, 0], a[:, 1]]]
            cols += [idx[b[:, 0], b[:, 1]]]
            w += [np.full(len(a), np.hypot(di, dj) * self.res)]
        rows, cols, w = map(np.concatenate, (rows, cols, w))
        n = len(cells)
        G = csr_matrix((w, (rows, cols)), shape=(n, n))
        g = self.nearest_free(goal)
        dist = dijkstra(G, directed=False, indices=idx[g[0], g[1]])
        self.dist = np.full((H, W), np.inf)
        self.dist[free] = dist
        self.goal = goal
        self.line_of_sight = False

    def cell(self, xy):
        # traversible maps are indexed [row = y, col = x]
        return np.array([int(xy[1] / self.res), int(xy[0] / self.res)])

    def nearest_free(self, xy):
        c = self.cell(xy)
        cells = np.argwhere(self.free)
        return cells[np.argmin(np.sum((cells - c) ** 2, 1))]

    def waypoint(self, xy, lookahead: float) -> np.ndarray:
        if np.linalg.norm(np.asarray(xy) - self.goal) < lookahead:
            return self.goal.copy()
        c = self.cell(xy)
        if not (0 <= c[0] < self.free.shape[0] and 0 <= c[1] < self.free.shape[1]) or not np.isfinite(self.dist[c[0], c[1]]):
            c = self.nearest_free(xy)
        start, cells = c.copy(), [c.copy()]
        walked = 0.0
        while walked < lookahead:
            best, bd = None, self.dist[c[0], c[1]]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = c + [di, dj]
                    if 0 <= n[0] < self.free.shape[0] and 0 <= n[1] < self.free.shape[1] and self.dist[n[0], n[1]] < bd:
                        best, bd = n, self.dist[n[0], n[1]]
            if best is None:
                break
            walked += np.hypot(*(best - c)) * self.res
            c = best
            cells.append(c.copy())
        if self.line_of_sight:  # farthest path cell visible from the robot through (inflated) free space
            for c in reversed(cells):
                if self.visible(start, c):
                    break
        return np.array([(c[1] + 0.5) * self.res, (c[0] + 0.5) * self.res])

    def visible(self, a, b):
        n = int(np.ceil(np.hypot(*(b - a)))) + 1
        pts = np.round(a + np.linspace(0, 1, n + 1)[:, None] * (b - a)).astype(int)
        return bool(np.all(self.free[pts[:, 0], pts[:, 1]]))


def parse_sets(items):
    """['a=1', 'b=True', 'c="x"'] -> dict (values as Python literals)."""
    import ast
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


class WallPoints:
    """Obstacle boundary points of the traversible map (occupied cells next to free ones, on a 0.1 m grid).
    __call__(path (N, 2)) -> points, normals (N, K, 2): per path point the K nearest boundary points within
    max_dist (nan-filled) and the half-plane normals into free space: towards the path point if it is in free
    space, else along the free-space distance gradient at the boundary point."""

    def __init__(self, trav: np.ndarray, scale: float, k: int, max_dist: float = 2.0, res: float = 0.1):
        from scipy.spatial import cKDTree
        self.trav, self.scale = trav, scale
        edge = ~trav & ndimage.binary_dilation(trav, iterations=1)
        ij = np.argwhere(edge)
        xy = (ij[:, ::-1] + 0.5) * scale
        keep = np.unique(np.floor(xy / res).astype(int), axis=0, return_index=True)[1]
        self.pts = xy[keep]
        # gradient normals: direction of increasing free-space distance, smoothed over ~0.25 m
        edt = ndimage.gaussian_filter(ndimage.distance_transform_edt(trav), 0.25 / scale)
        gy, gx = np.gradient(edt)
        r, c = ij[keep, 0], ij[keep, 1]
        g = np.stack([gx[r, c], gy[r, c]], 1)
        self.grad = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        self.tree = cKDTree(self.pts)
        self.k, self.max_dist = k, max_dist

    def free(self, xy):
        i, j = int(xy[1] / self.scale), int(xy[0] / self.scale)
        return 0 <= i < self.trav.shape[0] and 0 <= j < self.trav.shape[1] and bool(self.trav[i, j])

    def __call__(self, path):
        path = np.asarray(path)
        d, idx = self.tree.query(path, k=self.k, distance_upper_bound=self.max_dist)
        d, idx = d.reshape(len(path), -1), idx.reshape(len(path), -1)
        P = np.full((len(path), self.k, 2), np.nan)
        Nrm = np.full((len(path), self.k, 2), np.nan)
        for n, q in enumerate(path):
            inside = not self.free(q)
            for m in range(self.k):
                if not np.isfinite(d[n, m]):
                    continue
                o = self.pts[idx[n, m]]
                v = q - o
                nv = np.linalg.norm(v)
                P[n, m] = o
                Nrm[n, m] = self.grad[idx[n, m]] if inside or nv < 1e-6 else v / nv
        return P, Nrm


class EpisodeRunner:
    def __init__(self, args, msg, prior, predictor):
        self.args = args
        self.name = msg["name"]
        self.rng = np.random.default_rng(args.seed)
        self.sim_dt = msg["sim_dt"]
        self.n_cmd = int(round(DT / self.sim_dt))
        trav = np.unpackbits(np.frombuffer(base64.b64decode(msg["trav_bits"]), np.uint8))
        trav = trav[: np.prod(msg["trav_shape"])].reshape(msg["trav_shape"]).astype(bool)
        self.goal = np.array(msg["robot_goal"][:2])
        self.rcfg = RobotConfig()
        self.rcfg.v_bounds = (0.0, 1.2)  # SocNavBench robot limits
        self.rcfg.omega_bounds = (-1.1, 1.1)
        self.rcfg.radius = 0.24  # SocNavBench robot base (user_params radius_cm=24)
        t0 = time.time()
        self.field = PathField(trav, msg["map_scale"], self.goal, inflate=0.24 + 0.15)
        overrides = dict(unseen_joint=False, delta_unseen=0.02, ped_radius=0.2,  # SocNavBench agent radius
                         wall_points=args.walls, wall_clearance=args.wall_clearance)
        if args.camera == "active":
            self.mcfg = MPCConfig(robust=True, camera_mode="active", unseen_obs_aware=True, **overrides)
        elif args.camera == "infogain":
            self.mcfg = MPCConfig(robust=True, camera_mode="active", perception_objective="entropy", **overrides)
        elif args.camera in ("nearest", "ttc", "decoupled", "scan"):
            # NOTE: SocNavBench runs before 2026-10-13 with --camera nearest used the behaviour now called "ttc"
            self.mcfg = MPCConfig(robust=True, camera_mode=args.camera, **overrides)
        else:
            self.mcfg = MPCConfig(robust=True, camera_mode="fixed", **overrides)
        if args.mpc_set:  # ablations: MPCConfig field overrides
            import dataclasses
            self.mcfg = dataclasses.replace(self.mcfg, **parse_sets(args.mpc_set))
        self.cam = CameraConfig(fov_deg=args.fov, max_range=args.range)
        self.mpc = SupeRMPC(self.mcfg, self.rcfg, self.cam, TrackerConfig())
        if args.walls:
            self.mpc.wall_fn = WallPoints(trav, msg["map_scale"], args.walls)
            self.field.line_of_sight = True
        if args.tracker == "kf":  # ablation: plain Kalman tracker (no existence, no undetected-pedestrian model)
            from robot_nav.tracking import Tracker
            self.tracker = Tracker(TrackerConfig())
        else:
            self.tracker = PMBTracker(prior, self.cam, PMBConfig(**dict(dict(prune_pos_std=1.0),
                                                                       **parse_sets(args.pmb_set))))
        self.predictor = predictor
        self.L0 = np.linalg.cholesky(self.mcfg.Sigma0)
        self.phi = 0.0
        self.step = 0
        if args.vision:
            from robot_nav.vision import Associator, VisionConfig
            self.vcfg = VisionConfig(fov_deg=args.fov, max_range=args.range, model=args.detector)
            self.detector = detector_cache(self.vcfg)
            self.assoc = Associator(self.vcfg)
        self.ids = {}
        self.log = dict(t=[], robot=[], waypoint=[], u=[], n_tracks=[], n_seen=[], min_clear=[], solve_ms=[], ok=[],
                        slack=[], wall_slack=[], unseen_slack=[], plan=[], peds=[], tracks=[], detected=[], pred=[], pred_cov=[], pred_r=[],
                        det_xy=[], n_boxes=[], vision_ms=[], render_s=[])
        print(f"[{self.name}] path field ready in {time.time() - t0:.1f} s; start {msg['robot_start'][:2]} "
              f"goal {msg['robot_goal'][:2]}")

    def pid(self, name):
        if name not in self.ids:
            self.ids[name] = len(self.ids) + 1
        return self.ids[name]

    def act(self, msg):
        r = msg["robot"]
        x = np.array([r[0], r[1], r[2], self.phi])
        peds = {self.pid(n): np.array(p) for n, p in msg["peds"].items()}
        x_est = x + self.L0 @ self.rng.standard_normal(4)
        if self.args.vision:
            det, n_boxes, vis_ms = self.perceive(msg, x_est)
        else:
            det, n_boxes, vis_ms = observe(x, x_est, self.mcfg.Sigma0, peds, self.cam, self.rng), None, 0.0
        mcfg = self.mcfg
        if self.args.tracker == "kf":
            self.tracker.step(det)
            unseen_fn = None
        else:
            self.tracker.step(det, x_est)
            unseen_fn = lambda path: self.tracker.unseen_forecast(path, mcfg.unseen_slots, mcfg.delta_unseen)
        pred = self.predictor.predict(self.tracker.tracks, use_state_cov=not self.args.no_state_cov,
                                      state_cov_mode=mcfg.state_cov_mode, coast=mcfg.coast_mode)
        wp = self.field.waypoint(x[:2], self.args.waypoint_dist)
        t0 = time.time()
        u, info = self.mpc.solve(x_est, wp, pred, t=self.step * DT, unseen_fn=unseen_fn)
        v = float(np.clip(u[0], 0.0, 1.2))
        w = float(np.clip(u[1], -1.1, 1.1))
        self.phi = float(np.clip(self.phi + u[2] * DT, *self.rcfg.phi_bounds))
        clear = min((np.linalg.norm(q - x[:2]) for q in peds.values()), default=np.inf)
        for k, val in (("t", msg["t"]), ("robot", list(map(float, x))), ("waypoint", list(map(float, wp))),
                       ("u", [v, w, float(u[2])]), ("n_tracks", len(self.tracker.tracks)), ("n_seen", len(det)),
                       ("min_clear", float(clear)), ("solve_ms", 1e3 * (time.time() - t0)), ("ok", bool(info["ok"])),
                       ("slack", info["max_slack"]), ("wall_slack", info.get("wall_slack", 0.0)), ("unseen_slack", info["unseen_slack"]),
                       ("plan", np.asarray(info["X"])[:2].T.round(3).tolist()),
                       ("peds", {int(i): list(map(float, q)) for i, q in peds.items() if np.linalg.norm(q - x[:2]) < 6}),
                       ("tracks", [[float(tr.x[0]), float(tr.x[1]), float(tr.P[0, 0]), float(tr.P[0, 1]), float(tr.P[1, 1]),
                                   float(getattr(tr, "r", 1.0))] for tr in self.tracker.tracks.values()]),
                       ("detected", sorted(int(i) for i in det)),
                       ("det_xy", [list(map(float, z)) for z, _ in det.values()]),
                       ("n_boxes", n_boxes), ("vision_ms", vis_ms), ("render_s", msg.get("render_s")),
                       ("pred", [np.asarray(m).round(3).tolist() for m in np.asarray(pred.mean).transpose(1, 0, 2)]
                        if np.size(pred.mean) else []),
                       ("pred_cov", [[[float(c[0, 0]), float(c[0, 1]), float(c[1, 1])] for c in C]
                                     for C in np.asarray(pred.cov_model + pred.cov_state).transpose(1, 0, 2, 3)]
                        if np.size(pred.mean) else []),
                       ("pred_r", [float(r) for r in pred.existence])):
            self.log[k].append(val)
        self.step += 1
        return [[v, w]] * self.n_cmd, self.phi

    def perceive(self, msg, x_est):
        """RGB-D images -> person boxes -> world positions -> association with the PMB tracks."""
        import io
        from PIL import Image
        from robot_nav.vision import localize, to_world
        t0 = time.time()
        assert abs(msg["phi"] - self.phi) < 1e-9, "joystick rendered with a different pan"
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(msg["rgb_png"]))).convert("RGB"))
        mm = np.asarray(Image.open(io.BytesIO(base64.b64decode(msg["depth_png"]))), dtype=np.float32)
        depth = np.where(mm > 0, mm / 1000.0, np.inf).astype(np.float32)
        boxes, scores = self.detector(rgb)
        locs = localize(boxes, depth, self.vcfg)
        dets = to_world(locs, x_est, self.mcfg.Sigma0, self.vcfg)
        if self.args.save_frames:
            d = os.path.join(self.args.log_dir, "frames", self.name)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{self.step:03d}.npz"), rgb=rgb, depth=depth.astype(np.float16),
                                boxes=boxes, scores=scores,
                                ranges=np.array([np.nan if l is None else l[0] / np.cos(l[1]) for l in locs]))
        det = self.assoc(dets, self.tracker.tracks, DT)
        return det, int(len(boxes)), 1e3 * (time.time() - t0)

    def finish(self, msg):
        os.makedirs(self.args.log_dir, exist_ok=True)
        out = dict(name=self.name, dataset=getattr(self, "dataset", self.args.dataset), goal=list(map(float, self.goal)), camera=self.args.camera, fov=self.args.fov,
                   vision=bool(self.args.vision),
                   range=self.args.range, termination_cause=msg.get("termination_cause"), steps=self.step,
                   min_clearance=float(min(self.log["min_clear"], default=np.inf)), log=self.log)
        path = os.path.join(self.args.log_dir, f"{self.name}_{self.args.camera}{'_vision' if self.args.vision else ''}.json")
        out["walls"] = self.args.walls
        out["ablation"] = dict(mpc_set=self.args.mpc_set, pmb_set=self.args.pmb_set, tracker=self.args.tracker,
                               no_state_cov=self.args.no_state_cov)
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"[{self.name}] finished: {msg.get('termination_cause')} after {self.step} steps "
              f"({self.step * DT:.1f} s), min clearance {out['min_clearance']:.2f} m, mean solve "
              f"{np.mean(self.log['solve_ms']):.0f} ms -> {path}")


_DETECTOR = {}


def detector_cache(vcfg):
    """One detector per model for the server's lifetime (loading takes a few seconds)."""
    from robot_nav.vision import PersonDetector
    if vcfg.model not in _DETECTOR:
        _DETECTOR[vcfg.model] = PersonDetector(vcfg)
    return _DETECTOR[vcfg.model]


def serve(args):
    priors, predictors = {}, {}

    def episode_models(name):
        """Prior and predictor of an episode. Curated episodes (socnav_bridge.episodes): the recording's prior
        fitted with the episode's time window left out, and the predictor trained without that scene. Other
        episodes (e.g. test_socnav_univ): --dataset, prior fitted on the whole recording."""
        from socnav_bridge.episodes import EPISODES, spec, window
        if name in EPISODES:
            ds, excl = spec(name)["dataset"], window(name)
        else:
            ds, excl = args.dataset, None
        if (ds, excl) not in priors:
            priors[(ds, excl)] = fit_scene_prior(obsmat_path(ds), exclude_range=excl)
        if ds not in predictors:
            prefix = os.environ.get("LOO_PREFIX", "loorot")
            predictors[ds] = TrajectoryPredictor(f"checkpoints/{prefix}_{SCENE[ds]}.pt", MPCConfig().N)
        print(f"[{name}] dataset {ds}, prior {'without frames %.0f-%.0f' % excl if excl else 'whole recording'}",
              flush=True)
        return priors[(ds, excl)], predictors[ds], ds
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"SupeR-MPC server ({args.camera} camera, {args.fov:.0f} deg, {args.range:.0f} m) on port {args.port}, "
          f"dataset {args.dataset}; waiting for SocNavBench...", flush=True)
    while True:  # one connection per SocNavBench run (run_batch.sh runs one episode per container)
        conn, addr = srv.accept()
        print(f"connected from {addr}", flush=True)
        try:
            handle(conn, args, episode_models)
        except (ConnectionError, OSError) as e:
            print(f"connection lost: {e!r}", flush=True)
        print("SocNavBench disconnected", flush=True)
        if not args.persistent:
            break
    print("server done.")


def handle(conn, args, episode_models):
    buf, runner = b"", None
    # a container killed at its time limit can leave the connection half-open (Docker port forwarding);
    # time out instead of blocking the persistent server forever
    conn.settimeout(600)
    with conn:
        while True:
            chunk = conn.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = json.loads(line)
                if args.verbose:
                    print(f"recv {msg['type']} ({len(line)} bytes)", flush=True)
                if msg["type"] == "episode":
                    prior, predictor, ds = episode_models(msg["name"])
                    runner = EpisodeRunner(args, msg, prior, predictor)
                    runner.dataset = ds
                    conn.sendall(b'{"type": "ready"}\n')
                elif msg["type"] == "state":
                    t0 = time.time()
                    cmds, phi = runner.act(msg)
                    conn.sendall((json.dumps({"cmds": cmds, "phi": phi}) + "\n").encode())
                    if args.verbose:
                        print(f"  t={msg['t']:.2f}: replied in {time.time() - t0:.2f} s", flush=True)
                elif msg["type"] == "end":
                    runner.finish(msg)


if __name__ == "__main__":
    serve(parse_args())
