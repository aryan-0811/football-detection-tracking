from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration


def _get_pitch_dimensions(pitch_config: SoccerPitchConfiguration) -> tuple[float, float]:
    """
    SoccerPitchConfiguration in this project uses centimetres.
    """
    return float(pitch_config.length), float(pitch_config.width)


@dataclass
class TrackHeatmapStore:
    pitch_config: SoccerPitchConfiguration
    grid_size: tuple[int, int] = (480, 280)   # (width, height) in pitch-space bins
    min_samples_to_save: int = 30

    pitch_length: float = field(init=False)
    pitch_width: float = field(init=False)

    counts_by_track: dict[int, np.ndarray] = field(init=False, default_factory=dict)
    samples_by_track: dict[int, int] = field(init=False, default_factory=lambda: defaultdict(int))
    team_votes_by_track: dict[int, Counter] = field(init=False, default_factory=lambda: defaultdict(Counter))

    def __post_init__(self) -> None:
        self.pitch_length, self.pitch_width = _get_pitch_dimensions(self.pitch_config)
        self.grid_w, self.grid_h = self.grid_size

    def _ensure_track(self, track_id: int) -> None:
        if track_id not in self.counts_by_track:
            self.counts_by_track[track_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def _pitch_xy_to_grid_xy(self, x: float, y: float) -> tuple[int, int] | None:
        """
        Convert pitch coordinates (in cm) to heatmap grid coordinates.
        """
        if not np.isfinite(x) or not np.isfinite(y):
            return None

        if x < 0 or y < 0 or x > self.pitch_length or y > self.pitch_width:
            return None

        gx = int(np.clip((x / self.pitch_length) * (self.grid_w - 1), 0, self.grid_w - 1))
        gy = int(np.clip((y / self.pitch_width) * (self.grid_h - 1), 0, self.grid_h - 1))
        return gx, gy

    def update(self, detections: sv.Detections, transformer) -> None:
        """
        Accumulate projected BOTTOM_CENTER positions for tracked detections.
        Expected detections: players + goalkeepers, already team-labelled.
        """
        if transformer is None or len(detections) == 0:
            return

        if detections.tracker_id is None:
            return

        anchors = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_points = transformer.transform_points(points=anchors)

        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            if track_id is None:
                continue

            try:
                track_id = int(track_id)
            except Exception:
                continue

            x = float(pitch_points[i][0])
            y = float(pitch_points[i][1])

            grid_xy = self._pitch_xy_to_grid_xy(x, y)
            if grid_xy is None:
                continue

            gx, gy = grid_xy

            self._ensure_track(track_id)
            self.counts_by_track[track_id][gy, gx] += 1.0
            self.samples_by_track[track_id] += 1

            if detections.class_id is not None:
                cls_id = int(detections.class_id[i])
                self.team_votes_by_track[track_id][cls_id] += 1

    def _render_heatmap_image(
        self,
        heat: np.ndarray,
        title: str,
        max_alpha: float = 1.0,
        blur_ksize: int = 31,
        gamma: float = 0.22,
        min_visible: int = 2,
    ) -> np.ndarray:
        pitch_img = draw_pitch(self.pitch_config).copy()

        if heat.max() <= 0:
            cv2.putText(
                pitch_img,
                title,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return pitch_img

        heat_norm = heat.astype(np.float32)
        heat_norm /= heat_norm.max()

        if blur_ksize % 2 == 0:
            blur_ksize += 1
        heat_norm = cv2.GaussianBlur(heat_norm, (blur_ksize, blur_ksize), 0)

        ph, pw = pitch_img.shape[:2]
        heat_norm = cv2.resize(heat_norm, (pw, ph), interpolation=cv2.INTER_CUBIC)

        if heat_norm.max() > 0:
            heat_norm = heat_norm / heat_norm.max()

        heat_boost = np.power(heat_norm, gamma)
        heat_u8 = np.clip(heat_boost * 255.0, 0, 255).astype(np.uint8)

        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        visible_mask = heat_u8 > min_visible
        alpha_map = (heat_boost * max_alpha).astype(np.float32)

        out = pitch_img.copy().astype(np.float32)
        pitch_f = pitch_img.astype(np.float32)
        heat_f = heat_color.astype(np.float32)

        idx = visible_mask
        a = alpha_map[idx][:, None]
        out[idx] = (1.0 - a) * pitch_f[idx] + a * heat_f[idx]
        out = np.clip(out, 0, 255).astype(np.uint8)

        cv2.putText(
            out,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def save_all(
        self,
        out_dir: Path,
        top_n: int | None = None,
    ) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)

        ranked = sorted(
            self.samples_by_track.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

        if top_n is not None:
            ranked = ranked[:top_n]

        saved_paths: list[Path] = []

        for track_id, n_samples in ranked:
            if n_samples < self.min_samples_to_save:
                continue

            heat = self.counts_by_track.get(track_id)
            if heat is None or heat.max() <= 0:
                continue

            team_votes = self.team_votes_by_track.get(track_id, Counter())
            if len(team_votes) > 0:
                major_team = int(team_votes.most_common(1)[0][0])
                title = f"Track {track_id} | team={major_team} | samples={n_samples}"
                filename = f"track_{track_id}_team_{major_team}_samples_{n_samples}.png"
            else:
                title = f"Track {track_id} | samples={n_samples}"
                filename = f"track_{track_id}_samples_{n_samples}.png"

            np.save(out_dir / f"track_{track_id}_raw_heat.npy", heat)

            out_img = self._render_heatmap_image(heat=heat, title=title)
            out_path = out_dir / filename
            cv2.imwrite(str(out_path), out_img)
            saved_paths.append(out_path)

        return saved_paths