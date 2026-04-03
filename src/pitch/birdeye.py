from __future__ import annotations

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer

from src.ball.offside import FrameOffsideState


def render_birdeye_frame(
    pitch_config: SoccerPitchConfiguration,
    transformer: ViewTransformer,
    ball: sv.Detections,
    players_and_gk: sv.Detections,
    referees: sv.Detections,
    offside_state: FrameOffsideState | None = None,
) -> np.ndarray:
    """
    Creates a radar-style pitch frame:
    - ball in white
    - team 0 in blue
    - team 1 in pink
    - refs/other in yellow
    """

    pitch_img = draw_pitch(pitch_config)

    if len(ball) > 0:
        ball_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=ball_xy)
        pitch_img = draw_points_on_pitch(
            config=pitch_config,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch_img,
        )

    # Collect offside track IDs for red-marker override
    _offside_tids: set[int] = set()
    if offside_state is not None and offside_state.active:
        _offside_tids = offside_state.offside_track_ids

    if len(players_and_gk) > 0:
        pg_xy = players_and_gk.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_pg_xy = transformer.transform_points(points=pg_xy)

        # Build per-player offside mask
        _off_mask = np.zeros(len(players_and_gk), dtype=bool)
        if _offside_tids and players_and_gk.tracker_id is not None:
            for i, tid in enumerate(players_and_gk.tracker_id):
                if int(tid) in _offside_tids:
                    _off_mask[i] = True

        mask0 = (players_and_gk.class_id.astype(int) == 0) & ~_off_mask
        if mask0.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[mask0],
                face_color=sv.Color.from_hex("#00BFFF"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img,
            )

        mask1 = (players_and_gk.class_id.astype(int) == 1) & ~_off_mask
        if mask1.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[mask1],
                face_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img,
            )

        mask2 = (players_and_gk.class_id.astype(int) == 2) & ~_off_mask
        if mask2.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[mask2],
                face_color=sv.Color.from_hex("#FFD700"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img,
            )

        # Draw offside players in red
        if _off_mask.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[_off_mask],
                face_color=sv.Color.from_hex("#FF0000"),
                edge_color=sv.Color.WHITE,
                radius=18,
                pitch=pitch_img,
            )

    if len(referees) > 0:
        ref_xy = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ref_xy = transformer.transform_points(points=ref_xy)
        pitch_img = draw_points_on_pitch(
            config=pitch_config,
            xy=pitch_ref_xy,
            face_color=sv.Color.from_hex("#FFD700"),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch_img,
        )

    # Draw offside line
    if offside_state is not None and offside_state.active and offside_state.offside_line_x is not None:
        _draw_offside_line(pitch_img, pitch_config, offside_state.offside_line_x)

    return pitch_img


def _draw_offside_line(
    img: np.ndarray,
    config: SoccerPitchConfiguration,
    line_x_cm: float,
) -> None:
    """Draw a vertical red dashed line at the offside x-coordinate on the pitch image."""
    h, w = img.shape[:2]
    # Pitch coords: x in [0, length], y in [0, width]
    px = int(line_x_cm / config.length * w)
    px = max(0, min(px, w - 1))
    # Draw dashed line
    dash_len, gap_len = 12, 8
    y = 0
    while y < h:
        y_end = min(y + dash_len, h)
        cv2.line(img, (px, y), (px, y_end), (0, 0, 255), 2)
        y = y_end + gap_len