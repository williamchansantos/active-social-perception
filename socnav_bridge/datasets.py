"""SocNavBench pedestrian datasets in SocNavBench map coordinates, as obsmat files for our tracker prior.

SocNavBench stores each ETH/UCY recording as a 4-row CSV (frame, ped, "y", "x"), interpolated at the
recording's frame rate, and places it in its curated map with sign flips, an optional axis swap and an
offset (params/dataset_params.ini; agents/humans/recorded_human.py:gather_posn_data). This module applies
the same transform, subsamples to 0.4 s and writes an obsmat-format file (frame, ped, x, 0, y, vx, 0, vy)
that robot_nav.pmb.fit_scene_prior can read.
"""
from __future__ import annotations

import configparser
import os

import numpy as np

SOCNAV = os.environ.get("SOCNAVBENCH_DIR", r"C:/Users/James/Desktop/SocNavBench")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DT = 0.4


def dataset_params(name: str) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cp.read(os.path.join(SOCNAV, "params", "dataset_params.ini"))
    s = cp[name]
    return dict(file=os.path.join(SOCNAV, "agents", "humans", "datasets", s["file_name"]), fps=float(s["fps"]),
                offset=np.array(eval(s["offset"]), dtype=float), swapxy=s.getboolean("swapxy"),
                flipxn=s.getboolean("flipxn"), flipyn=s.getboolean("flipyn"))


def to_map_coords(name: str) -> np.ndarray:
    """(n, 4) array of frame, ped, x, y in SocNavBench map coordinates (every recorded frame)."""
    p = dataset_params(name)
    raw = np.loadtxt(p["file"], delimiter=",")  # rows: frame, ped, "y", "x"
    frame, ped, col_y, col_x = raw
    sx = -1.0 if p["flipxn"] else 1.0
    sy = -1.0 if p["flipyn"] else 1.0
    if p["swapxy"]:
        a, b = sy * col_y, sx * col_x
    else:
        a, b = sx * col_x, sy * col_y
    c, s = np.cos(p["offset"][2]), np.sin(p["offset"][2])
    x = a * c - b * s + p["offset"][0]
    y = a * s + b * c + p["offset"][1]
    return np.stack([frame, ped, x, y], 1)


def obsmat_path(name: str) -> str:
    """Write (once) and return the 0.4 s obsmat file of dataset `name` in map coordinates."""
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{name}_map.obsmat.txt")
    if os.path.exists(out):
        return out
    d = to_map_coords(name)
    fps = dataset_params(name)["fps"]
    step = int(round(DT * fps))
    f0 = d[:, 0].min()
    keep = ((d[:, 0] - f0) % step) == 0
    d = d[keep]
    rows = []
    for pid in np.unique(d[:, 1]):
        q = d[d[:, 1] == pid]
        q = q[np.argsort(q[:, 0])]
        v = np.zeros((len(q), 2))
        if len(q) > 1:
            v[1:] = np.diff(q[:, 2:4], axis=0) / DT
            v[0] = v[1]
        for (fr, _, x, y), (vx, vy) in zip(q, v):
            rows.append([fr, pid, x, 0.0, y, vx, 0.0, vy])
    rows = np.array(rows)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    np.savetxt(out, rows, fmt="%.4f")
    return out
