from pathlib import Path
import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration
from src.heatmap.player_heatmap import TrackHeatmapStore

heat_path = Path("outputs/input_heatmaps_preview/track_5_raw_heat.npy")
out_path = Path("outputs/debug/test_heatmap_render.png")

pitch_config = SoccerPitchConfiguration()
store = TrackHeatmapStore(pitch_config=pitch_config)

heat = np.load(heat_path)
img = store._render_heatmap_image(heat=heat, title="Render Test")

cv2.imwrite(str(out_path), img)
print("Saved:", out_path)