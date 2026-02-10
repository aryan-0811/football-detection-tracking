from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO
import os

@dataclass
class TrackConfig:
    model_path: Path
    source_video: Path
    output_dir: Path
    output_name: Optional[str]
    tracker: str
    imgsz: int
    conf: float
    iou: float
    save_jsonl: bool

def parse_args() -> TrackConfig:
    p = argparse.ArgumentParser(description="YOLO football tracking pipeline (annotated video + optional JSONL).")
    p.add_argument("--model", required=True, help="Path to weights (e.g., models/best.pt)")
    p.add_argument("--source", required=True, help="Path to input video (e.g., input_videos/input.mp4)")
    p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--name", default=None, help="Output base name (without extension). Defaults to input video stem.")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["botsort.yaml", "bytetrack.yaml"],
                   help="Tracker to use (default: bytetrack.yaml)")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280)")
    p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    p.add_argument("--save-jsonl", action="store_true",
                   help="Also save per-frame tracking results to outputs/<name>.jsonl")
    args = p.parse_args()

    return TrackConfig(
        model_path=Path(args.model),
        source_video=Path(args.source),
        output_dir=Path(args.outdir),
        output_name=args.name,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_jsonl=args.save_jsonl,
    )

def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")

def get_video_meta(video_path: Path) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n

def main() -> None:
    import os
    from pathlib import Path
    from ultralytics import YOLO

    cfg = parse_args()
    ensure_exists(cfg.model_path, "Model weights")
    ensure_exists(cfg.source_video, "Source video")

    # Always resolve outputs to an absolute path inside your repo
    outdir = cfg.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_name = cfg.output_name or cfg.source_video.stem
    final_video = outdir / f"{base_name}_tracked.mp4"

    print("CWD:", os.getcwd())
    print("Model:", cfg.model_path.resolve())
    print("Video:", cfg.source_video.resolve())
    print("Outputs dir:", outdir)

    model = YOLO(str(cfg.model_path))

    # Run tracking + save annotated video (Ultralytics will write into outdir/<base_name>/...)
    model.track(
        source=str(cfg.source_video),
        tracker=cfg.tracker,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        save=True,
        project=str(outdir),   # force saving under your project's outputs folder
        name=base_name,        # outputs/<base_name>/
        exist_ok=True          # overwrite same folder instead of exp/exp2
    )

    run_dir = outdir / base_name
    candidates = sorted(run_dir.glob("*.mp4"))
    if candidates:
        candidates[-1].replace(final_video)
        print("Final annotated video saved to:", final_video)
    else:
        print(f"Tracking finished. Check: {run_dir}")



if __name__ == "__main__":
    main()