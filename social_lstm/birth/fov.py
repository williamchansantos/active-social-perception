"""Appearance events relative to a robot's field of view (FOV).

The datasets record when a pedestrian is first *annotated in the scene*. A robot
with a forward-looking camera instead sees a new pedestrian when they first enter
its viewing cone, which happens in two ways:

    "scene_entry"  the pedestrian enters the scene already inside the cone
    "walked_in"    the pedestrian was in the scene (outside the cone) and then
                   walked into it

This module places a synthetic, stationary robot in a scene, replays the
annotation frame by frame and records the first frame each pedestrian is inside
the cone. The result is a `SceneAppearances` object, so every analysis and model
in this package works on it unchanged.

ASSUMPTIONS
- The robot does not move or turn (a fixed pose per scene).
- No occlusion: a pedestrian inside the cone is seen, whoever is in front.
- Only the first entry into the cone counts; re-entries are counted separately
  in `extras["reentries"]` but not treated as births.
- Pedestrians inside the cone in the scene's first frame are left-censored,
  exactly like pedestrians present in the first frame for scene-level births.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from social_lstm.birth.appearance import (
    DEFAULT_DT,
    AppearanceEvent,
    SceneAppearances,
    load_obsmat,
    resolve_obsmat,
    scene_name_from_path,
    tracks_from_frames,
)


@dataclass
class RobotView:
    """A stationary camera: position, heading and a circular-sector field of view."""

    position: np.ndarray  # (2,) x, y in meters
    heading: float  # radians, 0 = +x, counter-clockwise
    fov_deg: float = 60.0  # full horizontal opening angle
    max_range: float = 8.0  # meters

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: which (N, 2) points are inside the cone."""
        pts = np.asarray(points, dtype=float).reshape(-1, 2) - self.position
        dist = np.linalg.norm(pts, axis=1)
        bearing = np.arctan2(pts[:, 1], pts[:, 0]) - self.heading
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
        return (dist <= self.max_range) & (np.abs(bearing) <= math.radians(self.fov_deg) / 2)

    def outline(self, n: int = 40) -> np.ndarray:
        """(n + 2, 2) polygon of the cone for plotting."""
        half = math.radians(self.fov_deg) / 2
        angles = self.heading + np.linspace(-half, half, n)
        arc = self.position + self.max_range * np.column_stack([np.cos(angles), np.sin(angles)])
        return np.vstack([self.position, arc, self.position])

    @property
    def area(self) -> float:
        return 0.5 * self.max_range**2 * math.radians(self.fov_deg)

    def describe(self) -> str:
        return (f"robot at ({self.position[0]:.1f}, {self.position[1]:.1f}) m, heading {math.degrees(self.heading):.0f} deg, "
                f"FOV {self.fov_deg:.0f} deg, range {self.max_range:.0f} m")


def principal_axis(positions: np.ndarray) -> np.ndarray:
    """Unit vector along the main spread of the annotated positions (the dominant
    walking direction in a corridor-like scene), oriented towards +x (or +y if the
    axis is vertical)."""
    centered = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    return axis


def place_robot(all_positions: np.ndarray, placement: str = "center", fov_deg: float = 60.0,
                max_range: float = 8.0) -> RobotView:
    """Default robot poses derived from the scene geometry.

    "center": at the centre of the annotated area, looking along the principal axis.
    "edge":   at the edge of the annotated area on the principal axis, looking inward.
    """
    lo, hi = all_positions.min(axis=0), all_positions.max(axis=0)
    center = 0.5 * (lo + hi)
    axis = principal_axis(all_positions)
    heading = math.atan2(axis[1], axis[0])
    if placement == "center":
        position = center
    elif placement == "edge":
        # walk from the centre backwards along the axis until the bounding box edge
        extent = np.min(np.where(axis != 0, (hi - lo) / (2 * np.abs(axis) + 1e-12), np.inf))
        position = center - axis * extent
    else:
        raise ValueError(f"unknown placement {placement!r}")
    return RobotView(position=np.asarray(position, dtype=float), heading=heading, fov_deg=fov_deg, max_range=max_range)


def extract_fov_appearances(path: str, view: RobotView, dt: float = DEFAULT_DT,
                            name: Optional[str] = None) -> SceneAppearances:
    """Replay a scene and record each pedestrian's first frame inside `view`."""
    obsmat = resolve_obsmat(path)
    frame_to_peds = load_obsmat(obsmat)
    frames_sorted = sorted(frame_to_peds)
    frame_step = int(np.median(np.diff(frames_sorted))) if len(frames_sorted) > 1 else 1
    first_frame, last_frame = frames_sorted[0], frames_sorted[-1]

    events: List[AppearanceEvent] = []
    reentries = 0
    for pid, track in tracks_from_frames(frame_to_peds).items():
        frames = np.array([f for f, _ in track])
        pts = np.array([xy for _, xy in track], dtype=float)
        inside = view.contains(pts)
        if not inside.any():
            continue
        k = int(np.argmax(inside))  # index of the first in-cone frame
        # count separate entries (False -> True transitions after the first)
        reentries += int(max(np.sum(np.diff(inside.astype(int)) == 1) - (0 if inside[0] else 1), 0))
        if k + 1 < len(track):
            elapsed = (frames[k + 1] - frames[k]) / frame_step * dt
            velocity = (pts[k + 1] - pts[k]) / elapsed
        else:
            velocity = np.full(2, np.nan)
        events.append(
            AppearanceEvent(
                ped_id=pid,
                frame=int(frames[k]),
                time_s=(frames[k] - first_frame) / frame_step * dt,
                position=pts[k],
                velocity=velocity,
                present_at_start=(frames[k] == first_frame),
                num_frames=int(inside.sum()),
                origin="scene_entry" if k == 0 else "walked_in",
            )
        )
    events.sort(key=lambda e: (e.frame, e.ped_id))

    all_positions = np.array([xy for f in frames_sorted for xy in frame_to_peds[f].values()], dtype=float)
    in_cone = all_positions[view.contains(all_positions)]
    return SceneAppearances(
        name=name or f"{scene_name_from_path(obsmat)}_fov",
        path=obsmat,
        events=events,
        frame_step=frame_step,
        dt=dt,
        first_frame=first_frame,
        last_frame=last_frame,
        all_positions=in_cone,  # positions visible to the robot: defines the model's bounds
        extras={
            "reentries": float(reentries),
            "robot_x": float(view.position[0]),
            "robot_y": float(view.position[1]),
            "robot_heading_deg": math.degrees(view.heading),
            "fov_deg": view.fov_deg,
            "max_range": view.max_range,
            "scene_positions": all_positions,  # for plotting context
        },
    )


def origin_counts(scene: SceneAppearances, include_initial: bool = False) -> Dict[str, int]:
    counts: Dict[str, int] = {"scene_entry": 0, "walked_in": 0}
    for e in scene.arrivals(include_initial):
        counts[e.origin] = counts.get(e.origin, 0) + 1
    return counts
