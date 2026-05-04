#!/usr/bin/env python3
"""
Generic-detector video annotation: runs YOLO on every frame and writes a video
with per-class coloured bounding boxes and labels (Ball / Goalkeeper / Player / Referee).

Matches the visual format of project-report/images/03-methodology/generic_detector_output.png.

Usage:
    python -m scripts.detect_video --source input_videos/input.mp4
    python -m scripts.detect_video --source input_videos/input.mp4 --output outputs/generic_detector.mp4
    python -m scripts.detect_video --source input_videos/input.mp4 --max-frames 250 --show-conf
    python -m scripts.detect_video --source input_videos/input.mp4 --start 100 --end 400
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from src.pipeline.detections import detections_from_ultralytics

CLASS_NAMES = {0: "Ball", 1: "Goalkeeper", 2: "Player", 3: "Referee"}

CLASS_COLORS = sv.ColorPalette.from_hex([
    "#FF6600",  # 0 Ball       — orange
    "#00CC44",  # 1 Goalkeeper — green
    "#00AAFF",  # 2 Player     — blue
    "#FFD700",  # 3 Referee    — gold
])

# BGR fills for cv2 label boxes (mirror CLASS_COLORS hex above)
CLASS_BGR = {
    0: (0, 102, 255),    # Ball
    1: (68, 204, 0),     # Goalkeeper
    2: (255, 170, 0),    # Player
    3: (0, 215, 255),    # Referee
}

# Per-class text colour: referee uses black, others white-on-coloured-pill
CLASS_TEXT_BGR = {
    0: (255, 255, 255),
    1: (255, 255, 255),
    2: (255, 255, 255),
    3: (0, 0, 0),
}
# Classes that get a black outline behind the white text
OUTLINE_CLASSES = {0, 1, 2}


def build_annotators(box_thickness: int):
    return sv.BoxAnnotator(color=CLASS_COLORS, thickness=box_thickness)


def draw_labels(scene: np.ndarray, detections: sv.Detections, labels: list[str],
                font_scale: float, text_thickness: int, padding: int = 6) -> np.ndarray:
    """Draw class-name labels with per-class fill, text colour, and optional black outline."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = scene.shape[:2]
    for i in range(len(detections)):
        cls = int(detections.class_id[i])
        x1, y1, x2, _ = detections.xyxy[i].astype(int)
        text = labels[i]
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        bx1 = x1
        by2 = y1
        by1 = by2 - th - 2 * padding - baseline // 2
        # If pill would clip above frame, drop it inside the box instead
        if by1 < 0:
            by1 = y1
            by2 = by1 + th + 2 * padding + baseline // 2
        bx2 = min(w - 1, bx1 + tw + 2 * padding)
        bx1 = max(0, bx1)
        cv2.rectangle(scene, (bx1, by1), (bx2, by2), CLASS_BGR[cls], thickness=-1)
        tx = bx1 + padding
        ty = by2 - padding - baseline // 2
        if cls in OUTLINE_CLASSES:
            cv2.putText(scene, text, (tx, ty), font, font_scale, (0, 0, 0),
                        text_thickness + 2, cv2.LINE_AA)
        if cls == 3:
            cv2.putText(scene, text, (tx, ty), font, font_scale, (255, 255, 255),
                        text_thickness + 2, cv2.LINE_AA)
        cv2.putText(scene, text, (tx, ty), font, font_scale, CLASS_TEXT_BGR[cls],
                    text_thickness, cv2.LINE_AA)
    return scene


def annotate(frame: np.ndarray, detections: sv.Detections, show_conf: bool,
             box_ann: sv.BoxAnnotator, font_scale: float, text_thickness: int) -> np.ndarray:
    if len(detections) == 0:
        return frame
    if show_conf:
        labels = [
            f"{CLASS_NAMES.get(int(c), f'cls_{int(c)}')} {float(p):.2f}"
            for c, p in zip(detections.class_id, detections.confidence)
        ]
    else:
        labels = [CLASS_NAMES.get(int(c), f"cls_{int(c)}") for c in detections.class_id]
    out = box_ann.annotate(scene=frame.copy(), detections=detections)
    out = draw_labels(out, detections, labels, font_scale, text_thickness)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate annotated detection video (no tracking)")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--model", default="models/object_detection/best.pt", help="Path to YOLO weights")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: outputs/<source-stem>_detection.mp4)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    parser.add_argument("--start", type=int, default=0, help="First frame index (0-indexed)")
    parser.add_argument("--end", type=int, default=None, help="Stop *before* this frame index")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap on number of frames written")
    parser.add_argument("--font-scale", type=float, default=0.6, help="Label font scale")
    parser.add_argument("--box-thickness", type=int, default=2, help="Bounding box thickness")
    parser.add_argument("--text-thickness", type=int, default=1,
                        help="Label text stroke thickness (>=2 renders bold)")
    parser.add_argument("--show-conf", action="store_true",
                        help="Append confidence score to each label (off by default to match the report figure)")
    parser.add_argument("--device", default=None, help="YOLO device (e.g. cpu, cuda, mps)")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        outdir = Path("outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        output = outdir / f"{source.stem}_detection.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = max(0, args.start)
    end = total if args.end is None else min(args.end, total)
    if end <= start:
        cap.release()
        raise ValueError(f"Empty range: start={start} end={end}")
    if args.max_frames is not None:
        end = min(end, start + args.max_frames)

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {output}")

    model = YOLO(str(model_path))
    box_ann = build_annotators(args.box_thickness)

    print(f"Source : {source}  ({w}x{h} @ {fps:.2f} fps, {total} frames)")
    print(f"Range  : [{start}, {end})  ({end - start} frames)")
    print(f"Model  : {model_path}")
    print(f"Output : {output}")

    class_totals = {cid: 0 for cid in CLASS_NAMES}
    n_written = 0
    pbar = tqdm(total=end - start, unit="f", desc="detect")
    try:
        while True:
            idx = start + n_written
            if idx >= end:
                break
            ret, frame = cap.read()
            if not ret:
                break
            kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
            if args.device is not None:
                kwargs["device"] = args.device
            results = model(frame, **kwargs)[0]
            detections = detections_from_ultralytics(results)
            if len(detections) > 0:
                for cid in CLASS_NAMES:
                    class_totals[cid] += int(np.sum(detections.class_id == cid))
            annotated = annotate(frame, detections, args.show_conf, box_ann,
                                 args.font_scale, args.text_thickness)
            writer.write(annotated)
            n_written += 1
            pbar.update(1)
    finally:
        pbar.close()
        writer.release()
        cap.release()

    print(f"Wrote {n_written} frames to {output}")
    print("Detections per class (totals across written frames):")
    for cid, name in sorted(CLASS_NAMES.items()):
        print(f"  {name}: {class_totals[cid]}")


if __name__ == "__main__":
    main()
