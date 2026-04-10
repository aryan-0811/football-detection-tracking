"""
Generate a 3-panel figure illustrating the dedicated ball detector's
temporal proximity filtering pipeline.

Usage:
    python scripts/fig_ball_temporal.py <video_path> [--frame FRAME]

Examples:
    python scripts/fig_ball_temporal.py input_videos/input.mp4
    python scripts/fig_ball_temporal.py input_videos/input.mp4 --frame 350

When no --frame is given the script automatically scans the video to find
a frame with multiple ball detections (≥2 candidates at conf≥0.05).

The script runs the ball YOLO model at conf=0.05 on frames (t-1) and t,
then visualises:
  Panel 1 — all low-threshold candidates on frame t
  Panel 2 — prior confirmed position + displacement gate on frame t-1
  Panel 3 — selected ball after temporal filtering on frame t

Saves to: outputs/ball_temporal_filtering.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ultralytics import YOLO

# ── constants (match src/ball/detector.py defaults) ──────────────────
CONF_THRESH = 0.05
MAX_JUMP_PX = 80.0
MIN_CONF = 0.25
BALL_MODEL = Path("models/ball_detection/best.pt")
OUTPUT_PATH = Path("outputs/ball_temporal_filtering.png")


def _center(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def extract_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Could not read frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_model(model: YOLO, frame: np.ndarray):
    results = model.predict(source=frame, conf=CONF_THRESH, imgsz=1280, verbose=False)
    if not results or not hasattr(results[0], "boxes") or len(results[0].boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return xyxy, conf


def select_best(xyxy, conf, prev_center):
    """Reproduce the temporal filtering logic from BallDetector._select_detection."""
    if len(xyxy) == 0:
        return None, None, None
    centers = np.array([_center(b) for b in xyxy])
    if prev_center is None:
        idx = int(np.argmax(conf))
    else:
        dists = np.linalg.norm(centers - prev_center[None, :], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > MAX_JUMP_PX:
            return None, None, None
    if conf[idx] < MIN_CONF:
        return None, None, None
    return idx, centers[idx], conf[idx]


def find_multi_detection_frame(
    video_path: str,
    model: YOLO,
    min_candidates: int = 2,
    sample_every: int = 5,
    max_frames: int = 3000,
) -> int:
    """Scan the video and return the first frame with ≥ min_candidates detections."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, max_frames)
    print(f"Scanning up to {limit} frames (every {sample_every}) for ≥{min_candidates} candidates …")
    best_frame = -1
    best_count = 0
    for idx in range(1, limit, sample_every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        xyxy, conf = run_model(model, frame_rgb)
        n = len(xyxy)
        if n >= min_candidates:
            print(f"  → frame {idx}: {n} candidates ✓")
            cap.release()
            return idx
        if n > best_count:
            best_count = n
            best_frame = idx
    cap.release()
    if best_frame < 1:
        sys.exit("No ball detections found in the scanned range.")
    print(f"  No frame with ≥{min_candidates} candidates found; "
          f"using best frame {best_frame} ({best_count} candidate(s)).")
    return best_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame number (auto-detected if omitted)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = YOLO(str(BALL_MODEL))

    if args.frame is not None:
        t = args.frame
        if t < 1 or t >= total:
            sys.exit(f"Frame {t} out of range (video has {total} frames, need t>=1)")
    else:
        cap.release()
        t = find_multi_detection_frame(args.video, model)
        cap = cv2.VideoCapture(args.video)

    # Read frames t-1 and t
    frame_prev = extract_frame(cap, t - 1)
    frame_curr = extract_frame(cap, t)
    cap.release()

    # Run model on both frames
    xyxy_prev, conf_prev = run_model(model, frame_prev)
    xyxy_curr, conf_curr = run_model(model, frame_curr)
    print(f"Using frame t={t}: {len(xyxy_curr)} candidates on frame t, "
          f"{len(xyxy_prev)} on frame t-1")

    # Simulate detector state: select best on frame t-1 (no prior → highest conf)
    _, prev_center, _ = select_best(xyxy_prev, conf_prev, prev_center=None)

    # Select best on frame t using prior from t-1
    sel_idx, sel_center, sel_conf = select_best(xyxy_curr, conf_curr, prev_center)

    # ── Figure — 2-over-1 layout ──────────────────────────────────────
    # Top row: Panel 1 (candidates) + Panel 2 (prior + gate)
    # Bottom row: Panel 3 (result) centred
    plt.style.use("default")
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.patch.set_alpha(0.0)  # transparent background
    # 3 rows × 2 cols: top image pair each spans 1 col, bottom image
    # occupies the middle 2/3 of the second row for balanced sizing.
    # height_ratios: top row slightly taller so each half-width panel
    # appears similar in size to the centred bottom panel.
    gs = fig.add_gridspec(2, 6, hspace=0.08, wspace=0.06,
                          height_ratios=[1, 1],
                          top=0.93, bottom=0.02, left=0.01, right=0.99)

    ax1 = fig.add_subplot(gs[0, :3])   # top-left
    ax2 = fig.add_subplot(gs[0, 3:])   # top-right
    ax3 = fig.add_subplot(gs[1, 1:5])  # bottom centre (4/6 width)

    # ── Panel 1 (top-left): All candidates on frame t ───────────────
    ax1.imshow(frame_curr)
    for i in range(len(xyxy_curr)):
        cx, cy = _center(xyxy_curr[i])
        c = conf_curr[i]
        radius = max(8, (xyxy_curr[i][2] - xyxy_curr[i][0]) / 2)
        circle = plt.Circle((cx, cy), radius, fill=False, edgecolor="red",
                            linewidth=1.8, linestyle="-")
        ax1.add_patch(circle)
        ax1.text(cx + radius + 4, cy - 4, f"{c:.2f}", color="red",
                 fontsize=11, fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
    ax1.set_title(f"(a)  Frame $t$ — all candidates (conf ≥ {CONF_THRESH})",
                  fontsize=12, pad=8)
    ax1.axis("off")

    # ── Panel 2 (top-right): Previous position + gate on frame t-1 ──
    ax2.imshow(frame_prev)
    if prev_center is not None:
        px, py = prev_center
        img_h, img_w = frame_prev.shape[:2]
        # Small cross marker so it doesn't cover the ball
        ax2.plot(px, py, marker="+", color="lime", markersize=10, markeredgewidth=2)
        # Dashed circle for displacement gate
        gate = plt.Circle((px, py), MAX_JUMP_PX, fill=False, edgecolor="cyan",
                          linewidth=1.5, linestyle="--")
        ax2.add_patch(gate)
        # Place labels away from the ball using annotate arrows
        label_x = px - 120 if px > img_w * 0.4 else px + 60
        label_y = py - 90 if py > img_h * 0.3 else py + 90
        ax2.annotate("prior position", xy=(px, py), xytext=(label_x, label_y),
                     color="lime", fontsize=10, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.7, pad=2, edgecolor="none"),
                     arrowprops=dict(arrowstyle="-", color="lime", lw=1, ls="--"))
        gate_edge_x = px + MAX_JUMP_PX
        gate_label_x = gate_edge_x + 30
        gate_label_y = py + 50 if py < img_h * 0.7 else py - 50
        ax2.annotate(f"max jump = {MAX_JUMP_PX:.0f} px",
                     xy=(gate_edge_x, py), xytext=(gate_label_x, gate_label_y),
                     color="cyan", fontsize=10, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.7, pad=2, edgecolor="none"),
                     arrowprops=dict(arrowstyle="-", color="cyan", lw=1, ls="--"))
    else:
        ax2.text(0.5, 0.5, "No ball detected\nin frame t−1",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=14, color="yellow")
    ax2.set_title("(b)  Frame $t{-}1$ — prior position and displacement gate",
                  fontsize=12, pad=8)
    ax2.axis("off")

    # ── Panel 3 (bottom, centred): Selected ball on frame t ──────────
    ax3.imshow(frame_curr)
    if sel_center is not None:
        sx, sy = sel_center
        radius = max(10, (xyxy_curr[sel_idx][2] - xyxy_curr[sel_idx][0]) / 2)
        circle = plt.Circle((sx, sy), radius, fill=True, facecolor="lime",
                            edgecolor="white", linewidth=2, alpha=0.7)
        ax3.add_patch(circle)
        ax3.text(sx + radius + 6, sy - 6, f"conf = {sel_conf:.2f}", color="lime",
                 fontsize=12, fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
    else:
        ax3.text(0.5, 0.5, "No candidate passed\ntemporal filter",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=14, color="yellow")
    ax3.set_title("(c)  Frame $t$ — selected position after temporal filtering",
                  fontsize=12, pad=8)
    ax3.axis("off")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
