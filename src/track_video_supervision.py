#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.run import run_video_team_classification


def parse_args():
    p = argparse.ArgumentParser(description="YOLO tracking + Team classification + Supervision annotation")
    p.add_argument("--model", required=True, help="Path to YOLO weights")
    p.add_argument("--source", required=True, help="Path to input video")
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("--name", default=None, help="Output base name (no extension). Default: input video stem")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--show-id", action="store_true")
    p.add_argument("--font-scale", type=float, default=0.7)
    p.add_argument("--box-thickness", type=int, default=2)

    # detector class IDs
    p.add_argument("--ball-id", type=int, default=0)
    p.add_argument("--goalkeeper-id", type=int, default=1)
    p.add_argument("--player-id", type=int, default=2)
    p.add_argument("--referee-id", type=int, default=3)

    # team warmup + smoothing
    p.add_argument("--warmup-seconds", type=float, default=10.0)
    p.add_argument("--warmup-stride", type=int, default=30)
    p.add_argument("--max-warmup-crops", type=int, default=800)
    p.add_argument("--team-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--team-smooth", type=int, default=30)

    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (preview/testing)")

    # Pitch keypoints debug output
    p.add_argument("--pitch-debug", action="store_true",
                        help="Write an extra video with Roboflow pitch keypoints overlaid on frames.")
    p.add_argument("--pitch-stride", type=int, default=15,
                        help="Run pitch keypoints inference every N frames. Default: 15.")
    p.add_argument("--kp-conf", type=float, default=0.5,
                        help="Minimum keypoint confidence to draw. Default: 0.5.")
    
    # birdeye view
    p.add_argument("--birdeye", action="store_true", help="Write bird-eye radar video output.")
    p.add_argument("--side-by-side", action="store_true", help="Write side-by-side video (left main, right bird-eye).")

    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_name = args.name or source_path.stem
    target = outdir / f"{base_name}_team_tracked.mp4"

    run_video_team_classification(
        model_path=model_path,
        source_path=source_path,
        target_video_path=target,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        show_id=args.show_id,
        font_scale=args.font_scale,
        box_thickness=args.box_thickness,
        ball_id=args.ball_id,
        goalkeeper_id=args.goalkeeper_id,
        player_id=args.player_id,
        referee_id=args.referee_id,
        warmup_seconds=args.warmup_seconds,
        warmup_stride=args.warmup_stride,
        max_warmup_crops=args.max_warmup_crops,
        team_device=args.team_device,
        team_smooth=args.team_smooth,
        max_frames=args.max_frames,
        pitch_debug=args.pitch_debug,
        pitch_stride=args.pitch_stride,
        kp_conf=args.kp_conf,
        birdeye=args.birdeye,
        side_by_side=args.side_by_side,
    )


if __name__ == "__main__":
    main()
