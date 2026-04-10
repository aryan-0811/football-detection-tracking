"""
Extract player torso crops from a video warmup phase, fit the team classifier,
and save a UMAP scatter plot of SigLIP embeddings coloured by team label.

Usage:
    python -m scripts.plot_team_umap --source input_videos/input.mp4 \
        --model models/object_detection/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO

# Re-use project modules
from src.team.assigner import WarmupConfig
from src.team.cropping import crop_torso
from sports.common.team import TeamClassifier


def collect_crops(source: str, model: YOLO, player_id: int, imgsz: int,
                  warmup: WarmupConfig) -> list[np.ndarray]:
    video_info = sv.VideoInfo.from_video_path(source)
    max_frames = int(video_info.fps * warmup.seconds)
    frame_gen = sv.get_video_frames_generator(source, stride=warmup.stride)

    crops = []
    frames_seen = 0
    for frame in frame_gen:
        frames_seen += warmup.stride
        if frames_seen > max_frames:
            break
        r = model.predict(frame, imgsz=imgsz, conf=warmup.conf,
                          iou=warmup.iou, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        for box, cid in zip(xyxy, class_ids):
            if cid != player_id:
                continue
            if (box[2] - box[0]) * (box[3] - box[1]) < warmup.min_box_area:
                continue
            c = crop_torso(frame, box)
            if c is not None:
                crops.append(c)
            if len(crops) >= warmup.max_crops:
                break
        if len(crops) >= warmup.max_crops:
            break

    return crops


def main():
    parser = argparse.ArgumentParser(description="UMAP scatter plot of team embeddings")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="models/object_detection/best.pt",
                        help="YOLO model path")
    parser.add_argument("--player-id", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--output", default="outputs/team_clusters_umap.png")
    args = parser.parse_args()

    warmup = WarmupConfig(seconds=args.warmup_seconds)
    model = YOLO(args.model)

    print(f"Collecting player crops from {args.source} ...")
    crops = collect_crops(args.source, model, args.player_id, args.imgsz, warmup)
    print(f"Collected {len(crops)} crops")

    if len(crops) < 50:
        print(f"ERROR: only {len(crops)} crops found, need at least 50", file=sys.stderr)
        sys.exit(1)

    # Build classifier and fix UMAP seed for reproducible layout
    tc = TeamClassifier(device=args.device)
    import umap
    tc.reducer = umap.UMAP(n_components=3, random_state=42)
    print("Extracting SigLIP features ...")
    features = tc.extract_features(crops)

    print("Fitting UMAP ...")
    projections = tc.reducer.fit_transform(features)

    print("Fitting KMeans ...")
    tc.cluster_model.fit(projections)
    labels = tc.cluster_model.predict(projections)

    # Plot first two UMAP dimensions
    colours = {0: "#1f77b4", 1: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for team_id, label in [(0, "Team 1"), (1, "Team 2")]:
        mask = labels == team_id
        ax.scatter(projections[mask, 0], projections[mask, 1],
                   c=colours[team_id], label=label, alpha=0.6, s=60,
                   edgecolors="none")

    ax.set_xlabel("UMAP dimension 1", fontsize=14)
    ax.set_ylabel("UMAP dimension 2", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13, frameon=False)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
