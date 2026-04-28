#!/usr/bin/env python3
"""
Generate a clean 32-keypoint pitch template for the methodology chapter.

Usage:
    python -m scripts.fig_pitch_keypoint_template \
        --output outputs/pitch_keypoint_template.png

Optional:
    python -m scripts.fig_pitch_keypoint_template \
        --output outputs/pitch_keypoint_template.png \
        --highlight 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np

from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration


def hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to OpenCV BGR."""
    hex_colour = hex_colour.lstrip("#")
    r = int(hex_colour[0:2], 16)
    g = int(hex_colour[2:4], 16)
    b = int(hex_colour[4:6], 16)
    return b, g, r


def get_pitch_vertices(config: SoccerPitchConfiguration) -> np.ndarray:
    """
    Return the 32 pitch-template vertices from the sports configuration.

    The homography code uses the same SoccerPitchConfiguration, so using these
    vertices keeps the report figure aligned with the actual DLT point order.
    """
    if not hasattr(config, "vertices"):
        raise AttributeError(
            "SoccerPitchConfiguration has no 'vertices' attribute. "
            "Inspect your installed sports configuration and replace this "
            "function with the attribute that stores the keypoint template."
        )

    vertices = np.asarray(config.vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError(f"Expected vertices to have shape (N, 2), got {vertices.shape}")

    if len(vertices) != 32:
        print(f"WARNING: expected 32 vertices, found {len(vertices)}")

    return vertices


def pitch_to_image_xy(
    points: np.ndarray,
    image_shape: tuple[int, int, int],
    config: SoccerPitchConfiguration,
    padding: int,
) -> np.ndarray:
    """
    Convert pitch coordinates to image pixel coordinates.

    This mirrors the usual draw_pitch convention: the pitch is rendered inside
    a padded image, with pitch x/y coordinates scaled into the image plane.
    """
    h, w = image_shape[:2]
    usable_w = w - 2 * padding
    usable_h = h - 2 * padding

    pitch_length = float(config.length)
    pitch_width = float(config.width)

    out = points.copy().astype(float)
    out[:, 0] = padding + out[:, 0] * usable_w / pitch_length
    out[:, 1] = padding + out[:, 1] * usable_h / pitch_width
    return out


def draw_keypoint_template(
    config: SoccerPitchConfiguration,
    output: Path,
    highlight: int | None = None,
    padding: int = 50,
) -> None:
    pitch = draw_pitch(config=config, padding=padding)
    vertices = get_pitch_vertices(config)

    xy = pitch_to_image_xy(vertices, pitch.shape, config, padding)

    colours = getattr(config, "colors", ["#00FF00"] * len(vertices))

    for i, (x, y) in enumerate(xy):
        x_i, y_i = int(round(x)), int(round(y))

        if highlight is not None and i == highlight:
            radius = 16
            fill = (0, 0, 255)       # red in BGR
            outline = (255, 255, 255)
            thickness = 3
            label_colour = (0, 0, 255)
        else:
            radius = 11
            fill = hex_to_bgr(colours[i]) if i < len(colours) else (0, 255, 0)
            outline = (255, 255, 255)
            thickness = 2
            label_colour = (0, 0, 0)

        cv2.circle(pitch, (x_i, y_i), radius, fill, -1)
        cv2.circle(pitch, (x_i, y_i), radius, outline, thickness)

        # Offset labels slightly so they do not sit directly on top of markers.
        label = str(i)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness_main = 2
        thickness_outline = 4

        # Default placement: bottom-right of the keypoint
        dx = 12
        dy = 22

        # Measure text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness_main
        )

        # Proposed anchor point
        tx = x_i + dx
        ty = y_i + dy

        img_h, img_w = pitch.shape[:2]
        margin = 8

        # If text would go beyond the right border, move it left of the keypoint
        if tx + text_w > img_w - margin:
            tx = x_i - text_w - 12

        # If text would go beyond the bottom border, move it above a bit
        if ty + baseline > img_h - margin:
            ty = y_i - 10

        # If text would go beyond the top border, push it downward
        if ty - text_h < margin:
            ty = y_i + text_h + 10

        # White outline
        cv2.putText(
            pitch,
            label,
            (tx, ty),
            font,
            font_scale,
            (255, 255, 255),
            thickness_outline,
            cv2.LINE_AA,
        )

        # Main text
        cv2.putText(
            pitch,
            label,
            (tx, ty),
            font,
            font_scale,
            label_colour,
            thickness_main,
            cv2.LINE_AA,
        )

    if highlight is not None:
        cv2.putText(
            pitch,
            f"highlighted: kp_{highlight}",
            (padding, pitch.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            pitch,
            f"highlighted: kp_{highlight}",
            (padding, pitch.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), pitch)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pitch keypoint template figure")
    parser.add_argument(
        "--output",
        default="images/03-methodology/pitch_keypoint_template.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--highlight",
        type=int,
        default=None,
        help="Optional keypoint index to highlight, e.g. 16",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=50,
        help="Pitch padding in pixels",
    )
    args = parser.parse_args()

    config = SoccerPitchConfiguration()
    draw_keypoint_template(
        config=config,
        output=Path(args.output),
        highlight=args.highlight,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()