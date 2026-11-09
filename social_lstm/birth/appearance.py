"""Empirical analysis of pedestrian appearance events.

An *appearance event* is the first annotated frame of a pedestrian id. The
datasets only tell us when a pedestrian first appears in the *annotation* of
the scene, which is not the same as entering a robot's field of view; that
distinction is left to later work (see the module-level ASSUMPTIONS note).

ASSUMPTIONS
- Every annotated frame step is `dt` seconds apart (0.4 s for ETH/UCY: stated in
  the ETH info.txt files and confirmed by `infer_timestep_from_velocities`).
- A pedestrian present in the very first annotated frame did not "arrive"
  during the recording; such events are flagged `present_at_start=True` and
  excluded from the arrival-rate estimate by default (they are left-censored).
- A pedestrian id is a single person. The ETH/UCY ids are unique per scene.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from social_lstm.dataset import load_obsmat

DEFAULT_DT = 0.4  # seconds between annotated frames in ETH/UCY

# Default scene directories (relative to the repository root).
DEFAULT_SCENES: Tuple[str, ...] = (
    "data/eth/ewap_dataset/seq_eth",
    "data/eth/ewap_dataset/seq_hotel",
    "data/ucy/zara01",
    "data/ucy/zara02",
    "data/ucy/students03",
)


@dataclass
class AppearanceEvent:
    """First observation of one pedestrian."""

    ped_id: int
    frame: int
    time_s: float  # seconds since the first annotated frame of the scene
    position: np.ndarray  # (2,) x, y in meters
    velocity: np.ndarray  # (2,) m/s from the first two frames; NaN if only one frame
    present_at_start: bool  # True if the pedestrian is in the scene's first frame
    num_frames: int  # length of the pedestrian's track in annotated frames
    origin: str = "scene"  # "scene" (first annotated), or for field-of-view births:
    #   "scene_entry" (entered the scene already inside the cone) / "walked_in" (was in
    #   the scene before, then moved into the cone)


@dataclass
class SceneAppearances:
    """All appearance events of one scene plus the context needed to interpret them."""

    name: str
    path: str
    events: List[AppearanceEvent]
    frame_step: int  # frame-id difference between consecutive annotated frames
    dt: float  # seconds per annotated frame step
    first_frame: int
    last_frame: int
    all_positions: np.ndarray  # (N, 2) every annotated position, for plotting context
    destinations: Optional[np.ndarray] = None  # (M, 2) ETH destinations.txt, if present
    extras: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Recording duration covered by the annotation, in seconds."""
        return (self.last_frame - self.first_frame) / self.frame_step * self.dt

    @property
    def num_peds(self) -> int:
        return len(self.events)

    def arrivals(self, include_initial: bool = False) -> List[AppearanceEvent]:
        """Events that count as arrivals *during* the recording."""
        if include_initial:
            return list(self.events)
        return [e for e in self.events if not e.present_at_start]

    def arrival_rate(self, include_initial: bool = False) -> float:
        """Mean arrivals per second (the scalar rate of a homogeneous Poisson process)."""
        return len(self.arrivals(include_initial)) / self.duration_s

    def positions(self, include_initial: bool = False) -> np.ndarray:
        """(K, 2) appearance positions."""
        ev = self.arrivals(include_initial)
        return np.array([e.position for e in ev]).reshape(-1, 2)

    def velocities(self, include_initial: bool = False) -> np.ndarray:
        """(K, 2) initial velocities (may contain NaN rows)."""
        ev = self.arrivals(include_initial)
        return np.array([e.velocity for e in ev]).reshape(-1, 2)

    def times(self, include_initial: bool = False) -> np.ndarray:
        """(K,) appearance times in seconds, sorted."""
        return np.sort([e.time_s for e in self.arrivals(include_initial)])


def scene_name_from_path(path: str) -> str:
    """'data/ucy/zara01/obsmat.txt' -> 'zara01'; a directory maps to its basename."""
    p = path if os.path.isdir(path) else os.path.dirname(path)
    return os.path.basename(os.path.normpath(p))


def resolve_obsmat(path: str) -> str:
    """Accept either a scene directory or a direct obsmat.txt path."""
    if os.path.isdir(path):
        return os.path.join(path, "obsmat.txt")
    return path


def infer_timestep_from_velocities(obsmat_path: str, min_speed: float = 0.3) -> float:
    """Sanity check for `dt`: the obsmat files store a velocity column, so the
    ratio |delta position| / |velocity| between consecutive frames of one pedestrian
    is the time step the annotators used. Returns the median over all moving
    pedestrians (slow ones are skipped because the ratio is numerically noisy)."""
    data = np.loadtxt(obsmat_path)
    ratios: List[np.ndarray] = []
    for pid in np.unique(data[:, 1]):
        rows = data[data[:, 1] == pid]
        rows = rows[np.argsort(rows[:, 0])]
        if len(rows) < 2:
            continue
        dpos = np.hypot(np.diff(rows[:, 2]), np.diff(rows[:, 4]))
        speed = np.hypot(rows[:-1, 5], rows[:-1, 7])
        ok = speed > min_speed
        ratios.append(dpos[ok] / speed[ok])
    return float(np.median(np.concatenate(ratios))) if ratios else float("nan")


def tracks_from_frames(frame_to_peds: dict) -> Dict[int, List[Tuple[int, Tuple[float, float]]]]:
    """Invert {frame: {ped: (x, y)}} into {ped: [(frame, (x, y)), ...]} sorted by frame."""
    tracks: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
    for frame in sorted(frame_to_peds):
        for pid, xy in frame_to_peds[frame].items():
            tracks.setdefault(pid, []).append((frame, xy))
    return tracks


def extract_appearances(path: str, dt: float = DEFAULT_DT, name: Optional[str] = None) -> SceneAppearances:
    """Load one scene and build its list of appearance events.

    `path` is a scene directory or an obsmat.txt file. If the scene directory
    contains destinations.txt (ETH scenes), the destinations are loaded too.
    """
    obsmat = resolve_obsmat(path)
    frame_to_peds = load_obsmat(obsmat)
    frames_sorted = sorted(frame_to_peds)
    frame_step = int(np.median(np.diff(frames_sorted))) if len(frames_sorted) > 1 else 1
    first_frame, last_frame = frames_sorted[0], frames_sorted[-1]

    events: List[AppearanceEvent] = []
    for pid, track in tracks_from_frames(frame_to_peds).items():
        f0, p0 = track[0]
        if len(track) >= 2:
            f1, p1 = track[1]
            elapsed = (f1 - f0) / frame_step * dt
            velocity = (np.asarray(p1) - np.asarray(p0)) / elapsed
        else:
            velocity = np.full(2, np.nan)
        events.append(
            AppearanceEvent(
                ped_id=pid,
                frame=f0,
                time_s=(f0 - first_frame) / frame_step * dt,
                position=np.asarray(p0, dtype=float),
                velocity=np.asarray(velocity, dtype=float),
                present_at_start=(f0 == first_frame),
                num_frames=len(track),
            )
        )
    events.sort(key=lambda e: (e.frame, e.ped_id))

    all_positions = np.array(
        [xy for frame in frames_sorted for xy in frame_to_peds[frame].values()], dtype=float
    )
    dest_path = os.path.join(os.path.dirname(obsmat), "destinations.txt")
    destinations = np.loadtxt(dest_path).reshape(-1, 2) if os.path.isfile(dest_path) else None

    return SceneAppearances(
        name=name or scene_name_from_path(obsmat),
        path=obsmat,
        events=events,
        frame_step=frame_step,
        dt=dt,
        first_frame=first_frame,
        last_frame=last_frame,
        all_positions=all_positions,
        destinations=destinations,
        extras={"dt_from_velocities": infer_timestep_from_velocities(obsmat)},
    )


def load_scenes(paths: Sequence[str], dt: float = DEFAULT_DT) -> List[SceneAppearances]:
    return [extract_appearances(p, dt=dt) for p in paths]


def edge_distance(points: np.ndarray, all_positions: np.ndarray) -> np.ndarray:
    """Distance (m) from each point to the nearest edge of the bounding box of all
    annotated positions. Small values mean "appeared at the border of the
    annotated area", large values mean "appeared in the middle" (e.g. stepped
    out of a door, or was un-occluded)."""
    lo, hi = all_positions.min(axis=0), all_positions.max(axis=0)
    return np.minimum(points - lo, hi - points).min(axis=1)


def summary_row(scene: SceneAppearances, include_initial: bool = False, edge_margin: float = 1.0) -> dict:
    """Per-scene statistics as a flat dict (one row of the summary table)."""
    arrivals = scene.arrivals(include_initial)
    pos = scene.positions(include_initial)
    speeds = np.linalg.norm(scene.velocities(include_initial), axis=1)
    edge = edge_distance(pos, scene.all_positions) if len(pos) else np.array([])
    return {
        "scene": scene.name,
        "peds": scene.num_peds,
        "present_at_start": sum(e.present_at_start for e in scene.events),
        "arrivals": len(arrivals),
        "duration_s": scene.duration_s,
        "rate_per_s": scene.arrival_rate(include_initial),
        "rate_per_min": 60.0 * scene.arrival_rate(include_initial),
        "mean_gap_s": scene.duration_s / max(len(arrivals), 1),
        "median_speed": float(np.nanmedian(speeds)) if len(speeds) else float("nan"),
        "frac_near_edge": float(np.mean(edge < edge_margin)) if len(edge) else float("nan"),
        "frame_step": scene.frame_step,
        "dt_check_s": scene.extras.get("dt_from_velocities", float("nan")),
    }


def format_summary_table(rows: Sequence[dict]) -> str:
    """Render summary rows as a fixed-width text table."""
    # (key, header, width, value format)
    cols = [
        ("scene", "scene", 11, "{:<11s}"),
        ("peds", "peds", 5, "{:5d}"),
        ("present_at_start", "at_start", 8, "{:8d}"),
        ("arrivals", "arrivals", 8, "{:8d}"),
        ("duration_s", "duration_s", 10, "{:10.1f}"),
        ("rate_per_s", "rate_per_s", 10, "{:10.4f}"),
        ("rate_per_min", "rate_per_min", 12, "{:12.2f}"),
        ("mean_gap_s", "mean_gap_s", 10, "{:10.1f}"),
        ("median_speed", "med_speed_mps", 13, "{:13.2f}"),
        ("frac_near_edge", "frac_edge<1m", 12, "{:12.2f}"),
        ("frame_step", "frame_step", 10, "{:10d}"),
        ("dt_check_s", "dt_check_s", 10, "{:10.3f}"),
    ]
    head = " ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (_, h, w, _) in enumerate(cols))
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(" ".join(fmt.format(r[k]) for k, _, _, fmt in cols))
    return "\n".join(lines)
