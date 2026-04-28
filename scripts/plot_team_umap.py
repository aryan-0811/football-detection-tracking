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
import math
from PIL import Image
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


def save_labelled_crops(crops, labels, out_dir: str, max_per_team: int | None = None):
    out_dir = Path(out_dir)
    counts = {0: 0, 1: 0}

    for crop, label in zip(crops, labels):
        label = int(label)
        if max_per_team is not None and counts[label] >= max_per_team:
            continue

        team_dir = out_dir / f"team_{label + 1}"
        team_dir.mkdir(parents=True, exist_ok=True)

        # Convert BGR -> RGB if needed
        crop_rgb = crop[..., ::-1]
        Image.fromarray(crop_rgb).save(team_dir / f"crop_{counts[label]:03d}.png")
        counts[label] += 1


def save_team_crop_grid(crops, labels, out_path: str, max_per_team: int = 12):
    team0 = [c for c, l in zip(crops, labels) if int(l) == 0][:max_per_team]
    team1 = [c for c, l in zip(crops, labels) if int(l) == 1][:max_per_team]

    ncols = max(len(team0), len(team1))
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.4 * ncols, 2.6), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for j, crop in enumerate(team0):
        axes[0, j].imshow(crop[..., ::-1])
        axes[0, j].axis("off")

    for j, crop in enumerate(team1):
        axes[1, j].imshow(crop[..., ::-1])
        axes[1, j].axis("off")

    axes[0, 0].set_ylabel("Team 1", fontsize=12)
    axes[1, 0].set_ylabel("Team 2", fontsize=12)

    # fig.suptitle("Representative warmup torso crops", fontsize=14)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    parser.add_argument("--crops-dir", default="outputs/team_crops",
                        help="Directory to save labelled torso crops")
    parser.add_argument("--crops-grid", default="outputs/team_crops_grid.png",
                        help="Path to save a grid of representative labelled crops")
    parser.add_argument("--max-crops-per-team", type=int, default=12,
                        help="Maximum number of crops per team to save in the grid")
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

    print("Saving labelled torso crops ...")
    save_labelled_crops(
        crops,
        labels,
        args.crops_dir,
        max_per_team=args.max_crops_per_team
    )

    print("Saving crop grid ...")
    save_team_crop_grid(
        crops,
        labels,
        args.crops_grid,
        max_per_team=args.max_crops_per_team
    )

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
