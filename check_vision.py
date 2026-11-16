"""Evaluate the RGB-D pedestrian perception on rendered SocNavBench frames (SocNavBench/ours/render_replay.py).

Per frame: detect persons, localize with depth, and match to the true pedestrians (Hungarian, 1 m gate).
A true pedestrian counts as visible if enough ground-truth human pixels at its depth lie in its image column.
Reports precision, recall by range, the range / bearing errors (for VisionConfig.sigma_*), and timing.

    python check_vision.py --frames C:/Users/James/Desktop/SocNavBench/ours/replay
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from robot_nav.vision import PersonDetector, VisionConfig, focal, localize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="C:/Users/James/Desktop/SocNavBench/ours/replay")
    p.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_fpn")
    p.add_argument("--score", type=float, default=0.6)
    p.add_argument("--save", default=None, help="directory for annotated frames")
    return p.parse_args()


def visible_gt(d, cfg):
    """Indices of true pedestrians visible in the frame, with their (range, bearing)."""
    x, y, h = d["pose"]
    depth = d["depth"].astype(np.float32)
    mask = d["mask"]
    f, cx = focal(cfg), (depth.shape[1] - 1) / 2.0
    out = {}
    for i, q in enumerate(d["peds"][:, :2]):
        rel = q - [x, y]
        rng = float(np.hypot(*rel))
        b = float((np.arctan2(rel[1], rel[0]) - h + np.pi) % (2 * np.pi) - np.pi)
        if abs(b) > np.radians(cfg.fov_deg) / 2 + 0.05 or rng < 0.3:
            continue
        u = int(round(cx - f * np.tan(b)))
        c0, c1 = max(u - 3, 0), min(u + 4, depth.shape[1])
        if c0 >= c1:
            continue
        z = rng * np.cos(b)
        col = depth[:, c0:c1][mask[:, c0:c1]]
        if np.sum((col > z - 0.6) & (col < z + 0.3)) >= 15:
            out[i] = (rng, b)
    return out


def main(args):
    cfg = VisionConfig(model=args.model, score_thresh=args.score)
    det = PersonDetector(cfg)
    files = sorted(glob.glob(os.path.join(args.frames, "f*.npz")))
    rows, n_det, n_tp, n_fp_far, times = [], 0, 0, 0, []
    vis_by_bin = np.zeros(5)
    hit_by_bin = np.zeros(5)
    edges = [0, 2, 4, 6, 8, 99]
    for fpath in files:
        d = np.load(fpath)
        t0 = time.perf_counter()
        boxes, scores = det(d["rgb"])
        times.append(time.perf_counter() - t0)
        locs = localize(boxes, d["depth"].astype(np.float32), cfg)
        x, y, h = d["pose"]
        P = []
        for loc in locs:
            if loc is None:
                P.append(None)
                continue
            z, b = loc
            rng = z / np.cos(b)
            P.append(np.array([x, y]) + rng * np.array([np.cos(h + b), np.sin(h + b)]))
        vis = visible_gt(d, cfg)
        gt = d["peds"][:, :2]
        ok = [i for i, p in enumerate(P) if p is not None]
        n_det += len(ok)
        matched_gt = set()
        if ok:
            C = np.array([[np.linalg.norm(P[i] - g) for g in gt] for i in ok])
            r, c = linear_sum_assignment(C)
            for a, j in zip(r, c):
                if C[a, j] < 1.0:
                    n_tp += 1
                    matched_gt.add(j)
                    rel = gt[j] - [x, y]
                    rng_t = np.hypot(*rel)
                    b_t = (np.arctan2(rel[1], rel[0]) - h + np.pi) % (2 * np.pi) - np.pi
                    z, b = locs[ok[a]]
                    rows.append((rng_t, z / np.cos(b) - rng_t, (b - b_t + np.pi) % (2 * np.pi) - np.pi,
                                 j in vis))
        for j, (rng, _) in vis.items():
            k = np.searchsorted(edges, rng) - 1
            vis_by_bin[k] += 1
            hit_by_bin[k] += j in matched_gt
        if args.save:
            save_frame(args.save, fpath, d, boxes, scores, vis, matched_gt)
    rows = np.array(rows)
    print(f"{args.model}, score >= {args.score}: {len(files)} frames, {n_det} localized detections, "
          f"precision {n_tp / max(n_det, 1):.3f}, detector {1e3 * np.mean(times):.0f} ms/frame (CPU)")
    for k in range(5):
        print(f"  recall of visible pedestrians at {edges[k]}-{edges[k + 1]} m: "
              f"{hit_by_bin[k] / max(vis_by_bin[k], 1):.2f} ({int(hit_by_bin[k])}/{int(vis_by_bin[k])})")
    if len(rows):
        for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 99)):
            s = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi)]
            if len(s):
                print(f"  {lo}-{hi} m: n={len(s):3d}  range error mean {s[:, 1].mean():+.3f} sd {s[:, 1].std():.3f} m"
                      f"   bearing error mean {np.degrees(s[:, 2].mean()):+.2f} sd {np.degrees(s[:, 2].std()):.2f} deg")


def save_frame(out_dir, fpath, d, boxes, scores, vis, matched):
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    im = Image.fromarray(d["rgb"]).resize((512, 512))
    dr = ImageDraw.Draw(im)
    for (x0, y0, x1, y1), s in zip(boxes * 2, scores):
        dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
        dr.text((x0 + 2, y0 + 2), f"{s:.2f}", fill=(0, 255, 0))
    dr.text((4, 4), f"visible {len(vis)}  detected {len(boxes)}", fill=(255, 255, 0))
    im.save(os.path.join(out_dir, os.path.basename(fpath).replace(".npz", ".png")))


if __name__ == "__main__":
    main(parse_args())
