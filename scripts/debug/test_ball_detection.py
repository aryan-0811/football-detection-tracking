# test_ball_detection.py
# Test a ball-only YOLO model on a video and save an annotated MP4 using Supervision.
#
# Usage (macOS CPU):
#   python test_ball_detection.py \
#     --model models/ball_detection/best.pt \
#     --source input_videos/input.mp4 \
#     --out outputs/ball_only.mp4 \
#     --imgsz 1280 --conf 0.05 --iou 0.5 \
#     --device cpu \
#     --max-frames 0 \
#     --show-fps
#
# Notes:
# - This script tests the BALL model only (no team assigner, no birdeye).
# - Keep --conf low for better recall (0.03–0.10 is common for balls).
# - Output is written with Supervision's VideoSink.

import argparse
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import math

import numpy as np
import supervision as sv
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to ball model .pt (e.g. models/ball_detection/best.pt)")
    p.add_argument("--source", required=True, help="Path to input video")
    p.add_argument("--out", default="outputs/ball_only_supervision.mp4", help="Path to output annotated video")
    p.add_argument("--imgsz", type=int, default=1280, help="Ultralytics inference size")
    p.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--device", default="cpu", help="cpu, mps, or CUDA device id like 0")
    p.add_argument("--max-frames", type=int, default=0, help="0=full video, else stop after N frames")
    p.add_argument("--show-fps", action="store_true", help="Overlay running FPS")
    p.add_argument("--show-conf", action="store_true", help="Show conf value in label")
    p.add_argument("--thickness", type=int, default=2, help="Box thickness")
    p.add_argument("--text-scale", type=float, default=0.7, help="Label text scale")
    return p.parse_args()


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def ultralytics_to_sv_detections(result) -> sv.Detections:
    """
    Convert a single Ultralytics Result to supervision.Detections.
    Works for a standard detect model.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return sv.Detections.empty()

    xyxy = result.boxes.xyxy.cpu().numpy()
    confidence = result.boxes.conf.cpu().numpy()
    class_id = result.boxes.cls.cpu().numpy().astype(int)

    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def select_best_ball_detection(
    det: sv.Detections,
    prev_center: Optional[np.ndarray],
    max_jump_px: float = 350.0,
    min_conf: float = 0.08,
) -> tuple[sv.Detections, Optional[np.ndarray]]:
    if len(det) == 0:
        return det, prev_center

    xyxy = det.xyxy
    centers = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2))

    # If no previous, pick highest confidence (or first)
    if prev_center is None or det.confidence is None:
        best_i = int(np.argmax(det.confidence)) if det.confidence is not None else 0
        # confidence gate (still applies)
        if det.confidence is not None and float(det.confidence[best_i]) < min_conf:
            return sv.Detections.empty(), prev_center
        return det[[best_i]], centers[best_i]

    # Pick closest to previous position
    dists = np.linalg.norm(centers - prev_center[None, :], axis=1)
    best_i = int(np.argmin(dists))

    # If it's too far, treat as missing (your current behaviour)
    if float(dists[best_i]) > max_jump_px:
        return sv.Detections.empty(), prev_center

    # if it's close but not confident enough, treat as missing (prevents sock snaps)
    if det.confidence is not None and float(det.confidence[best_i]) < min_conf:
        return sv.Detections.empty(), prev_center

    return det[[best_i]], centers[best_i]


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)
    out_path = Path(args.out)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    ensure_parent(out_path)

    # Video info + IO
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    frame_gen = sv.get_video_frames_generator(str(source_path))

    # Load model
    yolo = YOLO(str(model_path))

    # Annotators
    # Color: green for ball boxes
    color = sv.Color.from_hex("#00FF00")
    box_annotator = sv.BoxAnnotator(color=color, thickness=args.thickness)
    label_annotator = sv.LabelAnnotator(color=color, text_scale=args.text_scale, text_thickness=2)

    # For FPS overlay
    t0 = time.perf_counter()
    frames_done = 0

    prev_center: Optional[np.ndarray] = None

    with sv.VideoSink(str(out_path), video_info=video_info) as sink:
        total = args.max_frames if args.max_frames > 0 else video_info.total_frames
        for frame_idx, frame in enumerate(tqdm(frame_gen, total=total, desc="Processing frames")):
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            # Ultralytics works fine with numpy BGR; it internally handles conversion.
            # (If you ever see odd colors, convert to RGB before predict.)
            results = yolo.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            r0 = results[0]
            det = ultralytics_to_sv_detections(r0)
            det, prev_center = select_best_ball_detection(det, prev_center, max_jump_px=350.0, min_conf=0.10)

            annotated = frame.copy()

            if len(det) > 0:
                annotated = box_annotator.annotate(scene=annotated, detections=det)

                # Build labels (ball model is usually single-class "ball")
                if args.show_conf and det.confidence is not None:
                    labels = [f"ball {c:.2f}" for c in det.confidence]
                else:
                    labels = ["ball"] * len(det)

                annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)

            if args.show_fps:
                frames_done += 1
                elapsed = max(time.perf_counter() - t0, 1e-6)
                fps_now = frames_done / elapsed
                sv.draw_text(
                    scene=annotated,
                    text=f"FPS: {fps_now:.1f}",
                    text_anchor=sv.Point(x=10, y=10),
                    text_scale=1.0,
                    text_thickness=2,
                )

            sink.write_frame(annotated)

            if (frame_idx + 1) % 50 == 0:
                print(f"Processed {frame_idx + 1} frames...")

    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()