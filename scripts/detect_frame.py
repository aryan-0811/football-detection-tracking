#!/usr/bin/env python3
"""
Single-frame YOLO object detection with annotated bounding boxes.

Usage:
    python -m scripts.detect_frame --source input_videos/input.mp4 --frame 175
    python -m scripts.detect_frame --source input_videos/input.mp4 --frame 175 --output outputs/my_frame.png
    python -m scripts.detect_frame --source input_videos/input.mp4 --frame 175 --model models/object_detection/best.pt

Draws bounding boxes for all four classes (ball, goalkeeper, player, referee)
with distinct colours, class labels, and confidence scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.pipeline.detections import detections_from_ultralytics

# ── Class config ──────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Ball", 1: "Goalkeeper", 2: "Player", 3: "Referee"}

CLASS_COLORS = sv.ColorPalette.from_hex([
    "#FF6600",  # 0 Ball       — orange
    "#00CC44",  # 1 Goalkeeper — green
    "#00AAFF",  # 2 Player     — blue
    "#FFD700",  # 3 Referee    — gold
])


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


def annotate_frame(frame: np.ndarray, detections: sv.Detections,
                   box_thickness: int, font_scale: float) -> np.ndarray:
    # Build labels: "ClassName 0.92"
    labels = []
    for i in range(len(detections)):
        cls = int(detections.class_id[i])
        conf = float(detections.confidence[i])
        name = CLASS_NAMES.get(cls, f"cls_{cls}")
        labels.append(f"{name} {conf:.2f}")

    # Annotators using per-class colour palette
    box_annotator = sv.BoxAnnotator(color=CLASS_COLORS, thickness=box_thickness)
    label_annotator = sv.LabelAnnotator(
        color=CLASS_COLORS,
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_scale=font_scale,
        text_padding=6,
    )

    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Single-frame YOLO detection with annotated output")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to extract (0-indexed)")
    parser.add_argument("--model", default="models/object_detection/best.pt", help="Path to YOLO weights")
    parser.add_argument("--output", default=None, help="Output PNG path (default: outputs/detection_frame_<N>.png)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--font-scale", type=float, default=0.5, help="Label font scale")
    parser.add_argument("--box-thickness", type=int, default=2, help="Bounding box thickness")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        outdir = Path("outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        output = outdir / f"detection_frame_{args.frame}.png"

    # Read frame
    frame = read_frame(source, args.frame)
    h, w = frame.shape[:2]
    print(f"Frame {args.frame}: {w}x{h}")

    # Run YOLO inference (no tracking)
    model = YOLO(str(model_path))
    results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    detections = detections_from_ultralytics(results)

    # Print detection summary
    print(f"Detections: {len(detections)}")
    for cls_id, name in sorted(CLASS_NAMES.items()):
        count = int(np.sum(detections.class_id == cls_id)) if len(detections) > 0 else 0
        print(f"  {name}: {count}")

    # Annotate and save
    annotated = annotate_frame(frame, detections, args.box_thickness, args.font_scale)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
