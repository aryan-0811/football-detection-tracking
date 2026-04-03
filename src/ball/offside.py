"""
Offside detection for broadcast football video.

Works entirely in pitch-space coordinates (centimetres) via ViewTransformer.
Detects passes by tracking ball-player proximity, then checks whether any
attacking teammate was in an offside position at the moment the ball was played.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PassEvent:
    """A detected pass (ball played by one player)."""
    frame_idx: int
    passer_track_id: int
    passer_team_id: int
    ball_pitch_xy: list[float]          # [x, y] in cm at pass moment

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OffsidePlayer:
    """One player caught in an offside position."""
    track_id: int
    pitch_xy: list[float]               # [x, y] in cm


@dataclass
class OffsideEvent:
    """An offside triggered by a pass."""
    frame_idx: int
    pass_event: PassEvent
    offside_line_x: float               # x-coord of the offside line (cm)
    offside_players: list[OffsidePlayer] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class FrameOffsideState:
    """Per-frame snapshot passed to the annotators / bird-eye renderer."""
    active: bool = False                 # whether to draw offside visuals this frame
    offside_line_x: float | None = None  # pitch-space x of offside line
    offside_track_ids: set[int] = field(default_factory=set)
    attacking_left: bool | None = None   # True ⇒ attacking team goes toward x=0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pitch_xy(dets: sv.Detections, transformer: ViewTransformer) -> np.ndarray:
    """Return (N, 2) pitch-space coordinates for detections (bottom-centre)."""
    if len(dets) == 0:
        return np.empty((0, 2), dtype=np.float64)
    px = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    return transformer.transform_points(points=px)


def _ball_pitch_xy(ball: sv.Detections, transformer: ViewTransformer) -> np.ndarray | None:
    """Return (2,) pitch coords of the ball, or None if no detection."""
    if len(ball) == 0:
        return None
    px = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pts = transformer.transform_points(points=px)
    return pts[0]


# ---------------------------------------------------------------------------
# Attacking-direction estimator
# ---------------------------------------------------------------------------

class _AttackDirectionEstimator:
    """
    Accumulates team centroids and decides which goal each team attacks.

    Convention:
      team_attacks_right[team_id] = True  ⟹  team attacks toward x = pitch_length
      team_attacks_right[team_id] = False ⟹  team attacks toward x = 0
    """

    def __init__(self, min_samples: int = 30, pitch_length: float = 12_000.0):
        self.min_samples = min_samples
        self.pitch_length = pitch_length
        self._sum_x: dict[int, float] = {0: 0.0, 1: 0.0}
        self._count: dict[int, int] = {0: 0, 1: 0}
        self.team_attacks_right: dict[int, bool] | None = None   # locked once decided

    @property
    def ready(self) -> bool:
        return self.team_attacks_right is not None

    def update(self, players: sv.Detections, pitch_xy: np.ndarray) -> None:
        """Feed player positions; auto-locks direction once enough data."""
        if self.ready:
            return
        if len(players) == 0 or players.class_id is None:
            return
        for i, cid in enumerate(players.class_id):
            cid = int(cid)
            if cid not in (0, 1):
                continue
            self._sum_x[cid] += float(pitch_xy[i, 0])
            self._count[cid] += 1

        if self._count[0] >= self.min_samples and self._count[1] >= self.min_samples:
            avg0 = self._sum_x[0] / self._count[0]
            avg1 = self._sum_x[1] / self._count[1]
            # Team with lower avg x is in their own left half → attacks right
            self.team_attacks_right = {
                0: avg0 < avg1,
                1: avg1 < avg0,
            }

    def goal_x(self, team_id: int) -> float | None:
        """Return the x-coordinate of the goal the team attacks toward."""
        if not self.ready:
            return None
        if self.team_attacks_right[team_id]:
            return self.pitch_length
        return 0.0


# ---------------------------------------------------------------------------
# Possession tracker
# ---------------------------------------------------------------------------

@dataclass
class _PossessionState:
    track_id: int
    team_id: int
    start_frame: int
    frames: int = 1


class _PossessionTracker:
    """Tracks which player currently has the ball (in pitch-space)."""

    def __init__(self, proximity_cm: float = 350.0, min_frames: int = 3):
        self.proximity_cm = proximity_cm
        self.min_frames = min_frames
        self.current: _PossessionState | None = None

    def update(
        self,
        frame_idx: int,
        ball_xy: np.ndarray,
        players_and_gk: sv.Detections,
        pitch_xy: np.ndarray,
    ) -> _PossessionState | None:
        """
        Update possession and return the *previous* possession state if a
        hand-off just occurred (i.e. the ball moved to a different player).
        Returns None if no hand-off happened this frame.
        """
        nearest_tid, nearest_cid, dist = self._nearest(ball_xy, players_and_gk, pitch_xy)

        # No player close enough → possession lost
        if nearest_tid is None or dist is None or dist > self.proximity_cm:
            prev = self._flush_if_valid()
            self.current = None
            return prev

        # Same player still has the ball
        if self.current is not None and self.current.track_id == nearest_tid:
            self.current.frames += 1
            return None

        # Different player is now closest → hand-off
        prev = self._flush_if_valid()
        self.current = _PossessionState(
            track_id=nearest_tid,
            team_id=nearest_cid,
            start_frame=frame_idx,
        )
        return prev

    # ------------------------------------------------------------------
    def _flush_if_valid(self) -> _PossessionState | None:
        if self.current is not None and self.current.frames >= self.min_frames:
            return self.current
        return None

    @staticmethod
    def _nearest(
        ball_xy: np.ndarray,
        dets: sv.Detections,
        pitch_xy: np.ndarray,
    ) -> tuple[int | None, int | None, float | None]:
        if len(dets) == 0 or dets.tracker_id is None:
            return None, None, None
        dists = np.linalg.norm(pitch_xy - ball_xy[None, :], axis=1)
        idx = int(np.argmin(dists))
        tid = int(dets.tracker_id[idx])
        cid = int(dets.class_id[idx]) if dets.class_id is not None else None
        return tid, cid, float(dists[idx])


# ---------------------------------------------------------------------------
# Offside checker
# ---------------------------------------------------------------------------

def check_offside(
    passer_team_id: int,
    ball_xy: np.ndarray,
    players_and_gk: sv.Detections,
    pg_pitch_xy: np.ndarray,
    passer_track_id: int,
    goal_x: float,
    pitch_length: float = 12_000.0,
    margin_cm: float = 30.0,
) -> tuple[float | None, list[OffsidePlayer]]:
    """
    Check offside for all teammates of the passer at the moment the ball is played.

    Returns (offside_line_x, list_of_offside_players).
    offside_line_x is None when there are not enough defenders to evaluate.
    """
    if len(players_and_gk) == 0 or players_and_gk.class_id is None:
        return None, []

    attacking_right = goal_x == pitch_length
    halfway_x = pitch_length / 2.0

    # --- defenders = opposing team players + GKs ---
    def_mask = players_and_gk.class_id.astype(int) != passer_team_id
    # Exclude referees (class_id 2) from defenders
    ref_mask = players_and_gk.class_id.astype(int) == 2
    def_mask = def_mask & ~ref_mask

    n_defenders = int(def_mask.sum())
    if n_defenders == 0:
        return None, []

    def_x = pg_pitch_xy[def_mask, 0]

    # Sort defenders by proximity to the goal the attacking team targets
    if attacking_right:
        # attacking toward x = pitch_length → defenders sorted descending (closest to goal first)
        sorted_x = np.sort(def_x)[::-1]
    else:
        # attacking toward x = 0 → defenders sorted ascending
        sorted_x = np.sort(def_x)

    if n_defenders >= 2:
        # Normal case: offside line = second-last defender
        offside_line_x = sorted_x[1]
    else:
        # Only 1 defender visible — assume the GK is off-screen near the goal
        # line, so this single defender is effectively the second-last player.
        offside_line_x = sorted_x[0]

    ball_x = float(ball_xy[0])

    # --- check each attacking teammate ---
    offside_players: list[OffsidePlayer] = []
    tids = players_and_gk.tracker_id
    cids = players_and_gk.class_id.astype(int)

    for i in range(len(players_and_gk)):
        if tids is None:
            continue
        tid = int(tids[i])
        cid = int(cids[i])
        if cid != passer_team_id:
            continue                 # not a teammate
        if tid == passer_track_id:
            continue                 # exclude the passer themselves
        px = float(pg_pitch_xy[i, 0])

        # Must be in opponent's half
        in_opp_half = px > halfway_x if attacking_right else px < halfway_x
        if not in_opp_half:
            continue

        # Must be closer to opponent goal than ball AND second-last defender
        if attacking_right:
            beyond_ball = px > ball_x + margin_cm
            beyond_def = px > offside_line_x + margin_cm
        else:
            beyond_ball = px < ball_x - margin_cm
            beyond_def = px < offside_line_x - margin_cm

        if beyond_ball and beyond_def:
            offside_players.append(OffsidePlayer(
                track_id=tid,
                pitch_xy=[float(pg_pitch_xy[i, 0]), float(pg_pitch_xy[i, 1])],
            ))

    return offside_line_x, offside_players


# ---------------------------------------------------------------------------
# Main detector  (called once per frame from the pipeline)
# ---------------------------------------------------------------------------

class OffsideDetector:
    """
    Orchestrates pass detection → offside checking → display state.

    Usage in the pipeline loop::

        offside_det = OffsideDetector()
        ...
        for frame_idx, ...:
            state = offside_det.update(frame_idx, ball, players, goalkeepers, transformer)
            # state.active / state.offside_line_x / state.offside_track_ids
    """

    def __init__(
        self,
        proximity_cm: float = 350.0,
        min_possession_frames: int = 3,
        offside_margin_cm: float = 30.0,
        display_frames: int = 50,
        pitch_length: float = 12_000.0,
        direction_min_samples: int = 30,
    ):
        self.offside_margin_cm = offside_margin_cm
        self.display_frames = display_frames
        self.pitch_length = pitch_length

        self._possession = _PossessionTracker(
            proximity_cm=proximity_cm,
            min_frames=min_possession_frames,
        )
        self._direction = _AttackDirectionEstimator(
            min_samples=direction_min_samples,
            pitch_length=pitch_length,
        )

        # event logs
        self.pass_events: list[PassEvent] = []
        self.offside_events: list[OffsideEvent] = []

        # display state (persists for display_frames after last offside)
        self._display_until: int = -1
        self._last_state = FrameOffsideState()

    # ------------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        ball: sv.Detections,
        players: sv.Detections,
        goalkeepers: sv.Detections,
        transformer: ViewTransformer,
    ) -> FrameOffsideState:
        """Process one frame; returns the visual state for annotators."""

        # Merge players + GKs for a full picture
        players_and_gk = sv.Detections.merge([players, goalkeepers])
        if len(players_and_gk) > 0 and players_and_gk.class_id is not None:
            players_and_gk.class_id = players_and_gk.class_id.astype(int)

        # Project to pitch-space
        pg_pitch = _pitch_xy(players_and_gk, transformer)
        ball_xy = _ball_pitch_xy(ball, transformer)

        # Feed the direction estimator (uses players only, not GKs)
        if len(players) > 0:
            p_pitch = _pitch_xy(players, transformer)
            self._direction.update(players, p_pitch)

        # If ball not detected, keep previous display state
        if ball_xy is None:
            return self._tick_display(frame_idx)

        # Update possession tracker
        prev_poss = self._possession.update(frame_idx, ball_xy, players_and_gk, pg_pitch)

        # prev_poss is non-None when possession just changed (= ball was played)
        if prev_poss is not None and self._direction.ready:
            self._on_pass(frame_idx, prev_poss, ball_xy, players_and_gk, pg_pitch)

        return self._tick_display(frame_idx)

    # ------------------------------------------------------------------
    def _on_pass(
        self,
        frame_idx: int,
        poss: _PossessionState,
        ball_xy: np.ndarray,
        players_and_gk: sv.Detections,
        pg_pitch: np.ndarray,
    ) -> None:
        pe = PassEvent(
            frame_idx=frame_idx,
            passer_track_id=poss.track_id,
            passer_team_id=poss.team_id,
            ball_pitch_xy=[float(ball_xy[0]), float(ball_xy[1])],
        )
        self.pass_events.append(pe)

        goal_x = self._direction.goal_x(poss.team_id)
        if goal_x is None:
            return

        offside_line_x, off_players = check_offside(
            passer_team_id=poss.team_id,
            ball_xy=ball_xy,
            players_and_gk=players_and_gk,
            pg_pitch_xy=pg_pitch,
            passer_track_id=poss.track_id,
            goal_x=goal_x,
            pitch_length=self.pitch_length,
            margin_cm=self.offside_margin_cm,
        )

        if off_players:
            oe = OffsideEvent(
                frame_idx=frame_idx,
                pass_event=pe,
                offside_line_x=offside_line_x,
                offside_players=off_players,
            )
            self.offside_events.append(oe)

            # Set display state
            attacking_right = self._direction.team_attacks_right[poss.team_id]
            self._last_state = FrameOffsideState(
                active=True,
                offside_line_x=offside_line_x,
                offside_track_ids={p.track_id for p in off_players},
                attacking_left=not attacking_right,
            )
            self._display_until = frame_idx + self.display_frames

    # ------------------------------------------------------------------
    def _tick_display(self, frame_idx: int) -> FrameOffsideState:
        if frame_idx > self._display_until:
            return FrameOffsideState()          # inactive
        return self._last_state

    # ------------------------------------------------------------------
    def save_events(self, path: Path) -> None:
        """Write all detected passes and offsides to a JSON file."""
        data = {
            "passes": [pe.to_dict() for pe in self.pass_events],
            "offsides": [oe.to_dict() for oe in self.offside_events],
        }
        path.parent.mkdir(parents=True, exist_ok=True)

        def _default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        path.write_text(json.dumps(data, indent=2, default=_default))
