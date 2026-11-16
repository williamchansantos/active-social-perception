"""SocNavBench's curated validation episodes (params/episode_params_val.ini) for the SupeR-MPC bridge.

- EPISODES: the 30 single-recording episodes (ETH 6, Zara1 6, Zara2 5, Univ 8, Hotel 5). The 3 DoubleHotel ones,
  which mix two copies of the Hotel recording, are left out. The Hotel episodes name a "HotelSmall" map that is
  not in the public download; they are run in the "Hotel" map.
- write_ini(tests): our episode file (SocNavBench reads it via env SOCNAV_EPISODES).
- spec(name) / window(name): an episode's recording, and the recording frames it replays (for a leakage-free
  pedestrian prior: the prior is fitted with that window, plus a margin, left out).
"""
from __future__ import annotations

import ast
import configparser
import os

import numpy as np

from socnav_bridge.datasets import SOCNAV, dataset_params, to_map_coords

VAL = os.path.join(SOCNAV, "params", "episode_params_val.ini")
OURS = os.path.join(SOCNAV, "ours", "episodes.ini")
MAP_FIX = {"HotelSmall": "Hotel"}


def _cp():
    cp = configparser.ConfigParser(inline_comment_prefixes=(";",))
    cp.read(VAL)
    return cp


def _all():
    cp = _cp()
    out = []
    for s in cp.sections():
        if s == "episodes_params":
            continue
        if len(ast.literal_eval(cp[s]["pedestrian_datasets"])) == 1:
            out.append(s)
    return out


EPISODES = _all()


def spec(name: str) -> dict:
    s = _cp()[name]
    return dict(map=MAP_FIX.get(s["map_name"].strip(), s["map_name"].strip()),
                dataset=ast.literal_eval(s["pedestrian_datasets"])[0],
                start_t=float(ast.literal_eval(s["datasets_start_t"])[0]),
                ped_range=tuple(ast.literal_eval(s["ped_ranges"])[0]),
                start_goal=ast.literal_eval(s["robot_start_goal"]), max_time=float(s["max_time"]))


def window(name: str, margin_s: float = 10.0):
    """(lo, hi) recording frames replayed by the episode, widened by margin_s on both sides.

    As agents/humans/recorded_human.py: simulator time t = (frame - start_frame) / fps + start_t, start_frame =
    first frame of pedestrian ped_range[0] + 1 (or of the recording if that pedestrian does not exist)."""
    e = spec(name)
    fps = dataset_params(e["dataset"])["fps"]
    d = to_map_coords(e["dataset"])
    first = d[d[:, 1] == e["ped_range"][0] + 1, 0]
    start_frame = first.min() if len(first) else d[:, 0].min()
    lo = start_frame - e["start_t"] * fps
    hi = start_frame + (e["max_time"] - e["start_t"]) * fps
    return lo - margin_s * fps, hi + margin_s * fps


def write_ini(tests, out: str = OURS) -> str:
    cp = _cp()
    lines = ["[episodes_params]", f"tests={list(tests)}", "without_robot=False", ""]
    for t in tests:
        lines.append(f"[{t}]")
        for k, v in cp[t].items():
            if k == "map_name":
                v = MAP_FIX.get(v.strip(), v.strip())
            lines.append(f"{k}={v}")
        lines.append("")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as f:
        f.write("\n".join(lines))
    return out


if __name__ == "__main__":
    for n in EPISODES:
        e = spec(n)
        lo, hi = window(n)
        d = to_map_coords(e["dataset"])
        print(f"{n:28s} {e['map']:6s} {e['dataset']:7s} peds {e['ped_range']} max_time {e['max_time']:.0f}  "
              f"window frames {lo:.0f}-{hi:.0f} of {d[:, 0].min():.0f}-{d[:, 0].max():.0f}")
