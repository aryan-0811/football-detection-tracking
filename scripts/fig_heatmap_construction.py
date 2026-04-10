"""Generate side-by-side figure: raw trajectory points vs Gaussian-smoothed heatmap."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
import cv2


def main():
    config = SoccerPitchConfiguration()
    pitch_img_bgr = draw_pitch(config)
    pitch_img = cv2.cvtColor(pitch_img_bgr, cv2.COLOR_BGR2RGB)

    # Load raw heatmap grid (280 x 480) — pick track with highest max for clarity
    heat = np.load("outputs/input_heatmaps/track_1_raw_heat.npy")
    grid_h, grid_w = heat.shape  # 280, 480

    # Extract individual point locations from the count grid
    ys, xs = np.where(heat > 0)
    # Convert grid coords to pitch cm coords
    pitch_length = float(config.length)
    pitch_width = float(config.width)
    px = xs / (grid_w - 1) * pitch_length
    py = ys / (grid_h - 1) * pitch_width

    # Repeat points by their count for scatter density
    counts = heat[ys, xs].astype(int)
    px_rep = np.repeat(px, counts)
    py_rep = np.repeat(py, counts)

    # Build smoothed heatmap
    blur_ksize = 31
    heat_smooth = cv2.GaussianBlur(heat.astype(np.float32), (blur_ksize, blur_ksize), 0)
    if heat_smooth.max() > 0:
        heat_smooth /= heat_smooth.max()
    # Gamma boost for visibility
    heat_smooth = np.power(heat_smooth, 0.22)

    # Resize to pitch image dims
    ph, pw = pitch_img.shape[:2]
    heat_resized = cv2.resize(heat_smooth, (pw, ph), interpolation=cv2.INTER_CUBIC)
    if heat_resized.max() > 0:
        heat_resized /= heat_resized.max()

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw trajectory points on pitch
    ax1.imshow(pitch_img, extent=[0, pitch_length, pitch_width, 0])
    ax1.scatter(px_rep, py_rep, s=8, c="#FF1493", alpha=0.5, edgecolors="none", zorder=5)
    ax1.set_xlim(0, pitch_length)
    ax1.set_ylim(pitch_width, 0)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.text(0.5, -0.06, "(a)", transform=ax1.transAxes, ha="center", va="top",
             fontsize=18)

    # Right: Gaussian-smoothed heatmap overlaid on pitch
    ax2.imshow(pitch_img, extent=[0, pitch_length, pitch_width, 0])
    # Overlay heatmap with transparency
    heat_color = plt.cm.jet(heat_resized)  # RGBA
    # Mask low-intensity regions as transparent
    alpha_channel = np.where(heat_resized > 0.02, heat_resized * 0.85, 0.0)
    heat_color[..., 3] = alpha_channel
    ax2.imshow(heat_color, extent=[0, pitch_length, pitch_width, 0], interpolation="bilinear")
    ax2.set_xlim(0, pitch_length)
    ax2.set_ylim(pitch_width, 0)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.text(0.5, -0.06, "(b)", transform=ax2.transAxes, ha="center", va="top",
             fontsize=18)

    plt.tight_layout()
    plt.savefig("outputs/heatmap_construction.png", dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close()
    print("Saved figures/heatmap_construction.png")


if __name__ == "__main__":
    main()
