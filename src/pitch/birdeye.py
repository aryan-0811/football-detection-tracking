from __future__ import annotations

import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer


def render_birdeye_frame(
    pitch_config: SoccerPitchConfiguration,
    transformer: ViewTransformer,
    ball: sv.Detections,
    players_and_gk: sv.Detections,
    referees: sv.Detections,
) -> np.ndarray:
    """
    Creates a radar-style pitch frame:
    - ball in white
    - team 0 in blue
    - team 1 in pink
    - refs/other in yellow
    """

    pitch_img = draw_pitch(pitch_config)

    # --- Project anchors (bottom-center) ---
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

    if len(players_and_gk) > 0:
        pg_xy = players_and_gk.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_pg_xy = transformer.transform_points(points=pg_xy)

        # team 0
        mask0 = players_and_gk.class_id.astype(int) == 0
        if mask0.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[mask0],
                face_color=sv.Color.from_hex("#00BFFF"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img,
            )

        # team 1
        mask1 = players_and_gk.class_id.astype(int) == 1
        if mask1.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[mask1],
                face_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_img,
            )

        # other (e.g. unknown GK = 2)
        mask2 = players_and_gk.class_id.astype(int) == 2
        if mask2.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config,
                xy=pitch_pg_xy[mask2],
                face_color=sv.Color.from_hex("#FFD700"),
                edge_color=sv.Color.BLACK,
                radius=16,
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

    return pitch_img