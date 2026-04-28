#!/usr/bin/env python3
"""
Generate a raw-frame figure illustrating image-space goalkeeper resolution.

The figure shows:
  - outfield players coloured by assigned team
  - team centroids marked in the same colours
  - goalkeeper highlighted
  - assignment line from goalkeeper to the chosen team centroid

Usage:
    python -m scripts.fig_gk_resolver_frame \
        --source input_videos/input.mp4 \
        --frame 175 \
        --output images/03-methodology/gk_resolver_frame.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.pipeline.detections import detections_from_ultralytics
from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id


PLAYER_ID = 2
GK_ID = 1

# BGR colours for OpenCV
TEAM_COLOURS = {
    0: (255, 191, 0),    # #00BFFF, cyan-blue
    1: (147, 20, 255),   # #FF1493, pink
}

GK_COLOUR = (0, 255, 255)      # yellow
LINE_COLOUR = (255, 255, 255)  # white


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_idx < 0 or frame_idx >= total:
        cap.release()
        raise ValueError(f"Frame {frame_idx} out of range; video has {total} frames")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")

    return frame


def box_center_xy(box: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = box.astype(float)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bottom_center_xy(box: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = box.astype(float)
    return int((x1 + x2) / 2), int(y2)


def measure_boxed_label(
    text: str,
    anchor: tuple[int, int],
    *,
    side: str = "right",
    valign: str = "above",
    scale: float = 0.58,
    pad_x: int = 7,
    pad_y: int = 5,
    thickness: int = 2,
) -> tuple[int, int, int, int, int, int]:
    """
    Compute the rectangle for a boxed label without drawing it.

    Returns:
        (x1, y1, x2, y2, text_x, text_y)
    """
    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    box_w = tw + 2 * pad_x
    box_h = th + baseline + 2 * pad_y
    gap = 10

    if side == "right":
        x1 = x + gap
    else:
        x1 = x - gap - box_w

    if valign == "above":
        y2 = y - 6
        y1 = y2 - box_h
    else:
        y1 = y + 6
        y2 = y1 + box_h

    tx = x1 + pad_x
    ty = y2 - pad_y - baseline

    return x1, y1, x1 + box_w, y2, tx, ty


def clamp_label_rect(
    rect: tuple[int, int, int, int, int, int],
    image_shape: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """Clamp a measured label rectangle to image bounds."""
    x1, y1, x2, y2, tx, ty = rect
    h, w = image_shape[:2]

    dx = 0
    dy = 0

    if x1 < 4:
        dx = 4 - x1
    elif x2 > w - 4:
        dx = (w - 4) - x2

    if y1 < 4:
        dy = 4 - y1
    elif y2 > h - 4:
        dy = (h - 4) - y2

    return x1 + dx, y1 + dy, x2 + dx, y2 + dy, tx + dx, ty + dy


def rect_overlap(
    a: tuple[int, int, int, int, int, int],
    b: tuple[int, int, int, int, int, int],
    margin: int = 6,
) -> bool:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    return not (
        ax2 + margin < bx1
        or bx2 + margin < ax1
        or ay2 + margin < by1
        or by2 + margin < ay1
    )


def draw_boxed_label_from_rect(
    img: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int, int, int],
    *,
    bg: tuple[int, int, int] = (24, 24, 24),
    fg: tuple[int, int, int] = (255, 255, 255),
    border: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.58,
    thickness: int = 2,
) -> None:
    """Draw a clean boxed label from a precomputed rectangle."""
    x1, y1, x2, y2, tx, ty = rect

    cv2.rectangle(img, (x1, y1), (x2, y2), bg, thickness=-1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, thickness=2)
    cv2.putText(
        img,
        text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        fg,
        thickness,
        cv2.LINE_AA,
    )


def draw_boxed_label(
    img: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    *,
    side: str = "right",
    valign: str = "above",
    scale: float = 0.58,
    bg: tuple[int, int, int] = (24, 24, 24),
    fg: tuple[int, int, int] = (255, 255, 255),
    border: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int, int, int, int]:
    """Measure, clamp, draw, and return a boxed label rectangle."""
    rect = measure_boxed_label(text, anchor, side=side, valign=valign, scale=scale)
    rect = clamp_label_rect(rect, img.shape)
    draw_boxed_label_from_rect(
        img,
        text,
        rect,
        bg=bg,
        fg=fg,
        border=border,
        scale=scale,
    )
    return rect


def draw_centroid_marker(
    img: np.ndarray,
    point: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    """Draw a team-coloured centroid marker with black outline."""
    x, y = point
    s = 14

    # black outline
    cv2.line(img, (x - s, y), (x + s, y), (0, 0, 0), 6, cv2.LINE_AA)
    cv2.line(img, (x, y - s), (x, y + s), (0, 0, 0), 6, cv2.LINE_AA)

    # coloured cross
    cv2.line(img, (x - s, y), (x + s, y), colour, 3, cv2.LINE_AA)
    cv2.line(img, (x, y - s), (x, y + s), colour, 3, cv2.LINE_AA)

    # centre dot
    cv2.circle(img, (x, y), 5, colour, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 8, (0, 0, 0), 2, cv2.LINE_AA)


def draw_player_box(
    img: np.ndarray,
    box: np.ndarray,
    colour: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate goalkeeper resolver frame")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--frame", type=int, required=True, help="Frame index")
    parser.add_argument("--model", default="models/object_detection/best.pt")
    parser.add_argument("--output", default="images/03-methodology/gk_resolver_frame.png")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--team-device", default="cpu")
    parser.add_argument("--warmup-seconds", type=float, default=20.0)
    parser.add_argument("--warmup-stride", type=int, default=30)
    parser.add_argument("--max-warmup-crops", type=int, default=800)

    # Optional visual tuning
    parser.add_argument("--box-thickness", type=int, default=3)
    parser.add_argument("--gk-thickness", type=int, default=4)
    parser.add_argument("--line-thickness", type=int, default=2)

    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = read_frame(source, args.frame)
    model = YOLO(args.model)

    # Fit team classifier from warmup frames.
    assigner = TeamAssigner(device=args.team_device, smooth_window=1)
    warmup = WarmupConfig(
        seconds=args.warmup_seconds,
        stride=args.warmup_stride,
        max_crops=args.max_warmup_crops,
        conf=max(args.conf, 0.25),
        iou=args.iou,
    )

    n_crops = assigner.fit_from_video(
        str(source),
        model,
        player_class_id=PLAYER_ID,
        imgsz=args.imgsz,
        warmup=warmup,
    )
    print(f"Team classifier fitted with {n_crops} crops")

    result = model.predict(
        frame,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )[0]

    dets = detections_from_ultralytics(result)

    players = dets[dets.class_id == PLAYER_ID]
    goalkeepers = dets[dets.class_id == GK_ID]

    if len(players) == 0:
        raise RuntimeError("No players detected in this frame")
    if len(goalkeepers) == 0:
        raise RuntimeError("No goalkeeper detected in this frame; choose another frame")

    # Assign teams to outfield players.
    players = assigner.predict_players(frame, players)
    players.class_id = players.class_id.astype(int)

    # Resolve goalkeeper team using the project implementation.
    gk_team_ids = resolve_goalkeepers_team_id(players, goalkeepers)

    annotated = frame.copy()

    # Draw outfield players and collect bottom-centres for centroids.
    team_points: dict[int, list[tuple[int, int]]] = {0: [], 1: []}

    for i in range(len(players)):
        team_id = int(players.class_id[i])
        colour = TEAM_COLOURS.get(team_id, (200, 200, 200))

        draw_player_box(annotated, players.xyxy[i], colour, thickness=args.box_thickness)

        bc = bottom_center_xy(players.xyxy[i])
        if team_id in team_points:
            team_points[team_id].append(bc)

    # Compute centroids.
    centroids: dict[int, tuple[int, int]] = {}
    for team_id, pts in team_points.items():
        if not pts:
            continue

        arr = np.asarray(pts, dtype=float)
        cx, cy = arr.mean(axis=0)
        centroids[team_id] = (int(cx), int(cy))

    # Draw centroid markers.
    for team_id, centroid in centroids.items():
        colour = TEAM_COLOURS.get(team_id, (255, 255, 255))
        draw_centroid_marker(annotated, centroid, colour)

    # Prepare centroid labels without drawing them twice.
    centroid_label_specs: dict[int, dict[str, object]] = {}

    for team_id, centroid in centroids.items():
        text = f"Team {team_id}"
        colour = TEAM_COLOURS.get(team_id, (255, 255, 255))

        if team_id == 0:
            side, valign = "left", "above"
        else:
            side, valign = "right", "below"

        rect = measure_boxed_label(
            text,
            centroid,
            side=side,
            valign=valign,
            scale=0.58,
            pad_x=7,
            pad_y=5,
            thickness=2,
        )
        rect = clamp_label_rect(rect, annotated.shape)

        centroid_label_specs[team_id] = {
            "text": text,
            "rect": rect,
            "border": colour,
        }

    # If the two centroid labels overlap, nudge Team 1 downward.
    if 0 in centroid_label_specs and 1 in centroid_label_specs:
        if rect_overlap(
            centroid_label_specs[0]["rect"],  # type: ignore[arg-type]
            centroid_label_specs[1]["rect"],  # type: ignore[arg-type]
        ):
            team1_centroid = centroids[1]
            rect = measure_boxed_label(
                "Team 1",
                team1_centroid,
                side="right",
                valign="below",
                scale=0.58,
                pad_x=7,
                pad_y=5,
                thickness=2,
            )
            x1, y1, x2, y2, tx, ty = rect
            rect = (x1, y1 + 28, x2, y2 + 28, tx, ty + 28)
            rect = clamp_label_rect(rect, annotated.shape)
            centroid_label_specs[1]["rect"] = rect

    # Draw each centroid label exactly once.
    for team_id, spec in centroid_label_specs.items():
        draw_boxed_label_from_rect(
            annotated,
            str(spec["text"]),
            spec["rect"],  # type: ignore[arg-type]
            bg=(24, 24, 24),
            fg=(255, 255, 255),
            border=spec["border"],  # type: ignore[arg-type]
            scale=0.58,
            thickness=2,
        )

    # Draw goalkeepers and assignment lines.
    for i in range(len(goalkeepers)):
        x1, y1, x2, y2 = goalkeepers.xyxy[i].astype(int)
        gk_centre = box_center_xy(goalkeepers.xyxy[i])

        assigned_team = int(gk_team_ids[i])
        assigned_colour = TEAM_COLOURS.get(assigned_team, GK_COLOUR)

        # Goalkeeper bounding box.
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            GK_COLOUR,
            args.gk_thickness,
        )

        # Draw assignment line behind label but over boxes.
        if assigned_team in centroids:
            cv2.line(
                annotated,
                gk_centre,
                centroids[assigned_team],
                LINE_COLOUR,
                args.line_thickness,
                cv2.LINE_AA,
            )
            cv2.circle(annotated, gk_centre, 7, assigned_colour, -1, cv2.LINE_AA)
            cv2.circle(annotated, gk_centre, 10, (0, 0, 0), 2, cv2.LINE_AA)

        # Goalkeeper label.
        draw_boxed_label(
            annotated,
            f"GK assigned to Team {assigned_team}",
            (x1, y1),
            side="right",
            valign="above",
            scale=0.62,
            bg=(24, 24, 24),
            fg=(255, 255, 255),
            border=GK_COLOUR,
        )

    cv2.imwrite(str(output), annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()