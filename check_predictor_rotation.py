"""Held-out prediction quality per scene for two predictor sets (leave-one-scene-out), and the ETH rotation test.

    python check_predictor_rotation.py loo loorot
"""
import os, sys
import numpy as np, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")
from evaluate_nav import SCENES, scene_frames
from robot_nav.prediction import TrajectoryPredictor
from social_lstm.dataset import load_obsmat
from social_lstm.engine import predict_sequence_with_gaussian
OBS, H = 8, 8

def windows(sc, half="second"):
    f2p = load_obsmat(SCENES[sc]["path"]); first, step, last, split = scene_frames(sc)
    lo, hi = (split, last) if half == "second" else (first, split)
    out = []
    fr = [f for f in sorted(f2p) if lo <= f < hi]
    for j in range(len(fr) - OBS - H):
        frames = fr[j: j + OBS + H]
        if any(b - a != step for a, b in zip(frames[:-1], frames[1:])): continue
        ids = [p for p in f2p[frames[0]] if all(p in f2p[f] for f in frames)]
        if ids: out.append(np.array([[f2p[f][p] for p in ids] for f in frames], dtype=np.float32))
    return out, step

def motion_stats(sc):
    f2p = load_obsmat(SCENES[sc]["path"]); fr = sorted(f2p); step = int(np.median(np.diff(fr)))
    v = []
    for a in fr:
        b = a + step
        if b not in f2p: continue
        for p, q in f2p[a].items():
            if p in f2p[b]: v.append((np.array(f2p[b][p]) - np.array(q)) / 0.4)
    v = np.array(v); s = np.linalg.norm(v, axis=1); mv = v[s > 0.3]
    ang = np.degrees(np.arctan2(mv[:, 1], mv[:, 0]))
    horiz = np.mean((np.abs(ang) < 45) | (np.abs(ang) > 135))
    return np.median(s[s > 0.3]), np.percentile(s[s > 0.3], 90), horiz, np.mean(s < 0.2)

def evaluate(model, wins, rot=None):
    ade, fde, cv_ade, in95 = [], [], [], []
    for tr in wins:
        t = tr if rot is None else tr @ rot.T
        with torch.no_grad():
            mu, sx, sy, rho = predict_sequence_with_gaussian(model, torch.from_numpy(t[:OBS].astype(np.float32)), OBS, H)
        mu, sx, sy, rho = mu.numpy(), sx.numpy(), sy.numpy(), rho.numpy()
        e = t[OBS:] - mu; err = np.linalg.norm(e, axis=-1)
        ade.append(err.mean(0)); fde.append(err[-1])
        cv = t[OBS - 1] + (t[OBS - 1] - t[OBS - 2])[None] * np.arange(1, H + 1)[:, None, None]
        cv_ade.append(np.linalg.norm(t[OBS:] - cv, axis=-1).mean(0))
        cxy = rho * sx * sy; det = sx ** 2 * sy ** 2 - cxy ** 2
        d2 = (sy ** 2 * e[..., 0] ** 2 - 2 * cxy * e[..., 0] * e[..., 1] + sx ** 2 * e[..., 1] ** 2) / det
        in95.append((d2[-1] < 5.991))
    c = np.concatenate
    return c(ade).mean(), c(fde).mean(), c(cv_ade).mean(), c(in95).mean(), len(c(ade))

PREFIXES = sys.argv[1:] or ["loo", "loorot"]
print(f"held-out prediction, second half, 8 observed -> 8 predicted (3.2 s)")
print(f"{'scene':8s} {'set':8s} {'windows':>8s} {'ADE':>6s} {'FDE':>6s} {'CV ADE':>7s} {'in 95% at 3.2 s':>16s}")
for sc in SCENES:
    wins, _ = windows(sc)
    for pre in PREFIXES:
        model = TrajectoryPredictor(f"checkpoints/{pre}_{sc}.pt", H).model
        a, f, c, i, n = evaluate(model, wins)
        print(f"{sc:8s} {pre:8s} {n:8d} {a:6.2f} {f:6.2f} {c:7.2f} {i:16.0%}")
print()
print("ETH rotated 90 deg (a rotation-invariant model scores the same either way):")
wins, _ = windows("eth")
R = np.array([[0, -1], [1, 0]], dtype=np.float32)
for pre in PREFIXES:
    model = TrajectoryPredictor(f"checkpoints/{pre}_eth.pt", H).model
    a0, f0, _, i0, _ = evaluate(model, wins)
    a1, f1, _, i1, _ = evaluate(model, wins, R)
    print(f"  {pre:8s} as recorded ADE {a0:.2f} (in 95% {i0:.0%}) | rotated ADE {a1:.2f} (in 95% {i1:.0%})")
