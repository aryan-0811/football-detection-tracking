# #!/usr/bin/env python3
# """
# Track football objects in a video using Ultralytics YOLO (BoT-SORT/ByteTrack) and annotate with Supervision.

# Output: annotated video saved to outputs/<video_stem>_sv_tracked.mp4

# Usage:
#   python src/track_video_supervision.py --model models/best.pt --source input_videos/input.mp4
# """

# from __future__ import annotations

# import argparse
# from pathlib import Path

# import numpy as np
# import supervision as sv
# from tqdm import tqdm
# from ultralytics import YOLO


# def parse_args():
#     p = argparse.ArgumentParser(description="YOLO tracking + Supervision annotation pipeline")
#     p.add_argument("--model", required=True, help="Path to YOLO weights (e.g., models/object_detection/best.pt)")
#     p.add_argument("--source", required=True, help="Path to input video (e.g., input_videos/input.mp4)")
#     p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
#     p.add_argument("--name", default=None, help="Output base name (no extension). Default: input video stem")
#     p.add_argument("--tracker", default="bytetrack.yaml", choices=["botsort.yaml", "bytetrack.yaml"],
#                    help="Tracker (default: botsort.yaml)")
#     p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280)")
#     p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
#     p.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
#     p.add_argument("--show-id", action="store_true",
#                    help="If set, include track IDs in labels (useful to verify tracking).")
#     p.add_argument("--font-scale", type=float, default=0.7,
#                    help="Label font scale (default: 0.6). Smaller: 0.4-0.5")
#     p.add_argument("--box-thickness", type=int, default=2, help="Box thickness (default: 2)")
#     return p.parse_args()


# def main():
#     args = parse_args()

#     model_path = Path(args.model).expanduser().resolve()
#     source_path = Path(args.source).expanduser().resolve()
#     outdir = Path(args.outdir).expanduser().resolve()
#     outdir.mkdir(parents=True, exist_ok=True)

#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found: {model_path}")
#     if not source_path.exists():
#         raise FileNotFoundError(f"Video not found: {source_path}")

#     base_name = args.name or source_path.stem
#     target_video_path = outdir / f"{base_name}_tracked.mp4"

#     # Video IO
#     video_info = sv.VideoInfo.from_video_path(str(source_path))
#     frame_generator = sv.get_video_frames_generator(str(source_path))
#     video_sink = sv.VideoSink(str(target_video_path), video_info=video_info)

#     # Annotators
#     palette = sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700'])
#     box_annotator = sv.BoxAnnotator(color=palette, thickness=args.box_thickness)
#     # ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=args.box_thickness) # 3 colors. as 3 classes
#     # triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)

#     label_annotator = sv.LabelAnnotator(
#         color=palette,
#         text_color=sv.Color.from_hex("#000000"),
#         text_scale=args.font_scale,
#     )

#     # Load YOLO model
#     model = YOLO(str(model_path))

#     # Tracking stream from Ultralytics
#     results_stream = model.track(
#         source=str(source_path),
#         tracker=args.tracker,
#         imgsz=args.imgsz,
#         conf=args.conf,
#         iou=args.iou,
#         stream=True,
#         verbose=False,
#     )

#     with video_sink:
#         for frame, r in tqdm(zip(frame_generator, results_stream), total=video_info.total_frames):
#             if r.boxes is None or len(r.boxes) == 0:
#                 video_sink.write_frame(frame)
#                 continue

#             # Extract data from Ultralytics Results
#             xyxy = r.boxes.xyxy.cpu().numpy()                  # (N,4)
#             confs = r.boxes.conf.cpu().numpy()                 # (N,)
#             class_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
#             track_ids = None
#             if r.boxes.id is not None:
#                 track_ids = r.boxes.id.cpu().numpy().astype(int)

#             # Convert to Supervision Detections
#             detections = sv.Detections(
#                 xyxy=xyxy,
#                 confidence=confs,
#                 class_id=class_ids,
#             )

#             # Build labels: "Class 0.87" (and optionally "ID:12")
#             names = r.names  # dict: class_id -> class_name
#             labels = []
#             for i in range(len(detections)):
#                 cname = names[int(class_ids[i])] if names else str(int(class_ids[i]))
#                 c = float(confs[i])
#                 if args.show_id and track_ids is not None:
#                     labels.append(f"{cname} {c:.2f} ID:{int(track_ids[i])}")
#                 else:
#                     labels.append(f"{cname} {c:.2f}")

#             # # ball detections
#             # ball_detections = detections[detections.class_id == 0]
#             # ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=7) # pad the box for ball as the triangle was very close to the ball.

#             # all_detections = detections[detections.class_id != 0]
#             # all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True) # Does non-max supression
#             # all_detections.class_id = all_detections.class_id - 1

#             # Annotate
#             annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
#             # annotated = ellipse_annotator.annotate(scene=frame.copy(), detections=all_detections)
#             annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
#             # annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)

#             video_sink.write_frame(annotated)

#     print("Saved:", target_video_path)
#     print(f"Tracker: {args.tracker} | imgsz={args.imgsz} conf={args.conf} iou={args.iou}")


# if __name__ == "__main__":
#     main()
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
    )


if __name__ == "__main__":
    main()
