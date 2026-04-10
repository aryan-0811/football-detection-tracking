#!/usr/bin/env python3
"""
Generate a side-by-side figure for the project report:
  Left panel  – broadcast frame with detected pitch keypoints (coloured dots + confidence)
  Right panel – bird's-eye pitch projection with player positions

Usage:
    python -m scripts.fig_homography --source input_videos/input.mp4 --frame 175
    python -m scripts.fig_homography --source input_videos/input.mp4 --frame 175 --output figures/homography_registration.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from src.ball.detector import BallDetector
from src.pipeline.detections import detections_from_ultralytics
from src.pitch.roboflow_pitch import RoboflowPitch, PitchConfig
from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_frame(video_path: Path, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total:
        cap.release()
        raise ValueError(f"Frame {frame_number} out of range (video has {total} frames)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number}")
    return frame


def draw_keypoints_on_frame(
    frame: np.ndarray,
    kps: sv.KeyPoints,
    kp_conf: float,
    pitch_config: SoccerPitchConfiguration,
) -> np.ndarray:
    """Draw keypoints as coloured circles with confidence scores on the frame."""
    out = frame.copy()
    pts = np.array(kps.xy[0], dtype=float)
    conf = np.array(kps.confidence[0], dtype=float)
    colors_hex = pitch_config.colors

    for i, ((x, y), c) in enumerate(zip(pts, conf)):
        if c < kp_conf:
            continue
        xi, yi = int(round(x)), int(round(y))

        # Use the pitch config colour for this keypoint
        hex_col = colors_hex[i] if i < len(colors_hex) else "#00FF00"
        bgr = tuple(int(hex_col.lstrip("#")[j:j+2], 16) for j in (4, 2, 0))

        cv2.circle(out, (xi, yi), 8, bgr, -1)
        cv2.circle(out, (xi, yi), 8, (255, 255, 255), 2)

        label = f"{i}: {c:.2f}"
        cv2.putText(out, label, (xi + 10, yi - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(out, label, (xi + 10, yi - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)

    return out


def build_birdeye(
    pitch_config: SoccerPitchConfiguration,
    transformer,
    players: sv.Detections,
    goalkeepers: sv.Detections,
    referees: sv.Detections,
    ball: sv.Detections | None = None,
) -> np.ndarray:
    """Render bird's-eye pitch with player markers."""
    pitch_img = draw_pitch(pitch_config)

    # Draw ball
    if ball is not None and len(ball) > 0:
        ball_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=ball_xy)
        pitch_img = draw_points_on_pitch(
            config=pitch_config, xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK, radius=10, pitch=pitch_img,
        )

    players_and_gk = sv.Detections.merge([players, goalkeepers])
    if len(players_and_gk) > 0 and players_and_gk.class_id is not None:
        pg_xy = players_and_gk.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_xy = transformer.transform_points(points=pg_xy)

        mask0 = players_and_gk.class_id.astype(int) == 0
        if mask0.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config, xy=pitch_xy[mask0],
                face_color=sv.Color.from_hex("#00BFFF"),
                edge_color=sv.Color.BLACK, radius=16, pitch=pitch_img,
            )
        mask1 = players_and_gk.class_id.astype(int) == 1
        if mask1.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config, xy=pitch_xy[mask1],
                face_color=sv.Color.from_hex("#FF1493"),
                edge_color=sv.Color.BLACK, radius=16, pitch=pitch_img,
            )
        mask2 = players_and_gk.class_id.astype(int) == 2
        if mask2.any():
            pitch_img = draw_points_on_pitch(
                config=pitch_config, xy=pitch_xy[mask2],
                face_color=sv.Color.from_hex("#FFD700"),
                edge_color=sv.Color.BLACK, radius=16, pitch=pitch_img,
            )

    if len(referees) > 0:
        ref_xy = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ref = transformer.transform_points(points=ref_xy)
        pitch_img = draw_points_on_pitch(
            config=pitch_config, xy=pitch_ref,
            face_color=sv.Color.from_hex("#FFD700"),
            edge_color=sv.Color.BLACK, radius=16, pitch=pitch_img,
        )

    return pitch_img


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side homography registration figure")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--frame", type=int, required=True, help="Frame number (0-indexed)")
    parser.add_argument("--model", default="models/object_detection/best.pt",
                        help="YOLO object detection weights")
    parser.add_argument("--output", default="outputs/homography_registration.png",
                        help="Output PNG path")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--kp-conf", type=float, default=0.5,
                        help="Keypoint confidence threshold")
    parser.add_argument("--ball-model", default="models/ball_detection/best.pt",
                        help="Dedicated ball detection YOLO weights")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    parser.add_argument("--team-device", default="cpu", help="Device for team classifier")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    pitch_config = SoccerPitchConfiguration()

    # ── Read frame ────────────────────────────────────────────────────────
    frame = read_frame(source, args.frame)
    print(f"Frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")

    # ── Pitch keypoints ───────────────────────────────────────────────────
    pitch = RoboflowPitch(
        PitchConfig(stride=1, kp_conf=args.kp_conf),
        pitch_config=pitch_config,
    )
    pitch.load_from_env()
    kps = pitch.maybe_infer_keypoints(frame, frame_idx=0)
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        raise RuntimeError("Pitch keypoint detection returned no results")

    conf_arr = np.array(kps.confidence[0], dtype=float)
    n_good = int((conf_arr >= args.kp_conf).sum())
    print(f"Keypoints: {n_good}/{len(conf_arr)} above conf={args.kp_conf}")

    # Build transformer for projection
    transformer = pitch.maybe_get_transformer(frame, frame_idx=0)
    if transformer is None:
        raise RuntimeError("Could not build ViewTransformer (too few keypoints?)")

    # ── Object detection + team classification ────────────────────────────
    model = YOLO(str(Path(args.model).expanduser().resolve()))

    # Warmup team classifier
    assigner = TeamAssigner(device=args.team_device, smooth_window=1)
    warmup = WarmupConfig(seconds=5.0, stride=30, max_crops=400,
                          conf=max(args.conf, 0.25), iou=0.7)
    n_crops = assigner.fit_from_video(str(source), model, player_class_id=2,
                                       imgsz=args.imgsz, warmup=warmup)
    print(f"Team classifier fit: {n_crops} crops")

    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    dets = detections_from_ultralytics(results)

    players = dets[dets.class_id == 2]
    goalkeepers = dets[dets.class_id == 1]
    referees = dets[dets.class_id == 3]

    players = assigner.predict_players(frame, players)
    if len(goalkeepers) > 0:
        gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
        goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)
    if len(referees) > 0:
        referees.class_id = np.full(len(referees), 2, dtype=int)

    # ── Ball detection ──────────────────────────────────────────────────
    ball_model_path = Path(args.ball_model).expanduser().resolve()
    ball_detector = BallDetector(model_path=ball_model_path, conf=0.05,
                                 max_jump_px=80.0, min_conf=0.25, imgsz=args.imgsz)
    ball = ball_detector.predict(frame)

    print(f"Detections: {len(players)} players, {len(goalkeepers)} GK, "
          f"{len(referees)} ref, {len(ball)} ball")

    # ── Left panel: keypoints on frame ────────────────────────────────────
    kp_frame = draw_keypoints_on_frame(frame, kps, args.kp_conf, pitch_config)
    # Draw ball marker on left panel
    if len(ball) > 0:
        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFD700"), base=20, height=17)
        kp_frame = triangle_annotator.annotate(scene=kp_frame, detections=ball)
    kp_frame_rgb = cv2.cvtColor(kp_frame, cv2.COLOR_BGR2RGB)

    # ── Right panel: bird's-eye projection ────────────────────────────────
    birdeye = build_birdeye(pitch_config, transformer, players, goalkeepers, referees, ball)
    birdeye_rgb = cv2.cvtColor(birdeye, cv2.COLOR_BGR2RGB)

    # ── Compose figure ────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18, 6),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    ax_left.imshow(kp_frame_rgb)
    ax_left.axis("off")
    ax_left.text(0.5, -0.04, "(a)", transform=ax_left.transAxes,
                 fontsize=18, ha="center")

    ax_right.imshow(birdeye_rgb)
    ax_right.axis("off")
    ax_right.text(0.5, -0.06, "(b)", transform=ax_right.transAxes,
                  fontsize=18, ha="center")

    # Legend for right panel
    legend_patches = [
        mpatches.Patch(color="#00BFFF", label="Team A"),
        mpatches.Patch(color="#FF1493", label="Team B"),
        mpatches.Patch(color="#FFD700", label="Referee"),
        mpatches.Patch(color="#FFFFFF", label="Ball"),
    ]
    ax_right.legend(handles=legend_patches, loc="lower right", fontsize=9,
                    framealpha=0.85)

    plt.tight_layout()
    fig.savefig(str(output), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
