"""
Match statistics tracker for broadcast football video.

Accumulates per-frame data in pitch-space (centimetres) to compute:
- Player speed (instantaneous and average, in km/h)
- Distance covered per player (in metres)
- Total passes (ball transitions between players)
- Team possession (percentage of frames each team controls the ball)

All four stats require pitch projection (ViewTransformer) to be available.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer

# Maximum plausible player speed (km/h).  Displacements that exceed this
# between consecutive frames are treated as tracking / homography noise.
_MAX_SPEED_KMH = 40.0


class StatsTracker:
    """Accumulates match statistics frame-by-frame in pitch-space."""

    def __init__(
        self,
        fps: float,
        proximity_cm: float = 350.0,
        min_hold_frames: int = 3,
    ):
        self.fps = fps
        self.dt = 1.0 / fps                     # seconds between frames

        # --- per-track movement ---
        self._prev_xy: dict[int, np.ndarray] = {}
        self._total_dist_cm: dict[int, float] = defaultdict(float)
        self._speeds_kmh: dict[int, list[float]] = defaultdict(list)
        self._track_team: dict[int, int] = {}    # most-recent team assignment

        # --- possession ---
        self._possession_frames: dict[int, int] = {0: 0, 1: 0}
        self._total_ball_frames: int = 0

        # --- pass detection (ball-holder transitions) ---
        self._proximity_cm = proximity_cm
        self._min_hold_frames = min_hold_frames
        self._passes_by_team: dict[int, int] = {0: 0, 1: 0}
        self._total_passes: int = 0
        self._holder_tid: int | None = None
        self._holder_team: int | None = None
        self._holder_frames: int = 0

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        players_and_gk: sv.Detections,
        ball: sv.Detections,
        transformer: ViewTransformer,
    ) -> None:
        """Feed one frame of team-assigned detections."""
        if len(players_and_gk) == 0 or players_and_gk.tracker_id is None:
            return

        # Project player feet to pitch-space (cm)
        anchors = players_and_gk.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_xy = transformer.transform_points(points=anchors)

        self._update_movement(players_and_gk, pitch_xy)

        # Possession / passes need ball position
        if len(ball) > 0:
            ball_anchor = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            ball_pitch = transformer.transform_points(points=ball_anchor)[0]
            self._update_possession(players_and_gk, pitch_xy, ball_pitch)

    # ------------------------------------------------------------------
    # Movement (speed + distance)
    # ------------------------------------------------------------------

    def _update_movement(
        self,
        dets: sv.Detections,
        pitch_xy: np.ndarray,
    ) -> None:
        max_disp_cm = (_MAX_SPEED_KMH / 0.036) * self.dt  # cm per frame cap

        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            xy = pitch_xy[i]

            # Store latest team assignment (ignore referees = 2)
            if dets.class_id is not None:
                cid = int(dets.class_id[i])
                if cid in (0, 1):
                    self._track_team[tid] = cid

            if tid in self._prev_xy:
                disp = float(np.linalg.norm(xy - self._prev_xy[tid]))
                if disp <= max_disp_cm:
                    self._total_dist_cm[tid] += disp
                    speed = (disp / self.dt) * 0.036   # cm/s -> km/h
                    self._speeds_kmh[tid].append(speed)

            self._prev_xy[tid] = xy.copy()

    # ------------------------------------------------------------------
    # Possession & passes
    # ------------------------------------------------------------------

    def _update_possession(
        self,
        dets: sv.Detections,
        pitch_xy: np.ndarray,
        ball_xy: np.ndarray,
    ) -> None:
        dists = np.linalg.norm(pitch_xy - ball_xy[None, :], axis=1)
        idx = int(np.argmin(dists))

        if float(dists[idx]) > self._proximity_cm:
            self._flush_holder()
            self._holder_tid = None
            self._holder_team = None
            self._holder_frames = 0
            return

        tid = int(dets.tracker_id[idx])
        team = int(dets.class_id[idx]) if dets.class_id is not None else None

        if team not in (0, 1):
            return  # ignore referee "possession"

        # Count possession frame for this team
        self._possession_frames[team] += 1
        self._total_ball_frames += 1

        # Holder tracking for pass detection
        if self._holder_tid == tid:
            self._holder_frames += 1
        else:
            self._flush_holder()
            self._holder_tid = tid
            self._holder_team = team
            self._holder_frames = 1

    def _flush_holder(self) -> None:
        """Record a pass when a valid ball-holder spell ends."""
        if (
            self._holder_tid is not None
            and self._holder_team is not None
            and self._holder_frames >= self._min_hold_frames
        ):
            self._passes_by_team[self._holder_team] += 1
            self._total_passes += 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a JSON-serializable summary of all accumulated stats."""
        total_bf = max(self._total_ball_frames, 1)
        possession_pct = {
            f"team_{k}": round(v / total_bf * 100, 1)
            for k, v in self._possession_frames.items()
        }

        # Per-player stats
        all_tids = set(self._total_dist_cm.keys()) | set(self._speeds_kmh.keys())
        player_stats = {}
        for tid in sorted(all_tids):
            speeds = self._speeds_kmh.get(tid, [])
            player_stats[str(tid)] = {
                "team_id": self._track_team.get(tid),
                "distance_m": round(self._total_dist_cm.get(tid, 0.0) / 100, 1),
                "avg_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else 0.0,
                "max_speed_kmh": round(float(np.max(speeds)), 1) if speeds else 0.0,
            }

        return {
            "match_stats": {
                "total_passes": self._total_passes,
                "passes_by_team": {
                    f"team_{k}": v for k, v in self._passes_by_team.items()
                },
                "possession_pct": possession_pct,
            },
            "player_stats": player_stats,
        }

    def save(self, path: Path) -> None:
        """Write stats summary to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        path.write_text(json.dumps(data, indent=2))
