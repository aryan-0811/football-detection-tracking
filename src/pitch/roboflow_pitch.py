from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer


@dataclass
class PitchConfig:
    stride: int = 15           # run field model every N frames
    kp_conf: float = 0.5       # keypoint confidence threshold
    infer_conf: float = 0.3    # Roboflow infer() confidence parameter
    min_good_kps: int = 6      # require at least this many confident points to update cache


class RoboflowPitch:
    """
    Roboflow pitch keypoints helper with:
    - stride-based inference
    - caching of last valid keypoints
    - optional cached ViewTransformer builder (for bird-eye step)
    """

    def __init__(self, cfg: PitchConfig, pitch_config: Optional[SoccerPitchConfiguration] = None):
        self.cfg = cfg
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

        self.model = None
        self.last_kps: Optional[sv.KeyPoints] = None
        self.last_transformer: Optional[ViewTransformer] = None
        self.last_update_frame_idx: Optional[int] = None

    def load_from_env(self) -> None:
        load_dotenv()
        api_key = os.getenv("ROBOFLOW_API_KEY")
        model_id = os.getenv("RF_FIELD_MODEL_ID")

        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY missing in .env")
        if not model_id:
            raise RuntimeError("RF_FIELD_MODEL_ID missing in .env")

        self.model = get_model(model_id=model_id, api_key=api_key)

    def maybe_infer_keypoints(self, frame: np.ndarray, frame_idx: int) -> Optional[sv.KeyPoints]:
        """
        Runs inference only on stride frames; caches last valid keypoints.
        Returns cached keypoints (may be None initially).
        """
        if self.model is None:
            raise RuntimeError("Pitch model not loaded. Call load_from_env() first.")

        stride = max(1, int(self.cfg.stride))
        if frame_idx % stride != 0:
            return self.last_kps

        try:
            result = self.model.infer(frame, confidence=float(self.cfg.infer_conf))[0]
            kps = sv.KeyPoints.from_inference(result)

            if self._is_good(kps):
                self.last_kps = kps
                self.last_update_frame_idx = frame_idx
                # invalidate transformer until rebuilt (or rebuild immediately in maybe_get_transformer)
                self.last_transformer = None

        except Exception as e:
            print(f"[pitch] infer failed at frame {frame_idx}: {e}")

        return self.last_kps

    def maybe_get_transformer(self, frame: np.ndarray, frame_idx: int) -> Optional[ViewTransformer]:
        """
        Next-step helper: returns a cached ViewTransformer built from cached keypoints.
        It updates keypoints on stride frames and builds transformer if possible.
        """
        kps = self.maybe_infer_keypoints(frame, frame_idx)
        if kps is None:
            return None

        if self.last_transformer is not None:
            return self.last_transformer

        # Build transformer from confident correspondences
        if kps.xy is None or kps.confidence is None or len(kps.xy) == 0:
            return None

        conf = np.array(kps.confidence[0], dtype=float)
        mask = conf >= float(self.cfg.kp_conf)

        if int(mask.sum()) < int(self.cfg.min_good_kps):
            return None

        frame_reference_points = np.array(kps.xy[0], dtype=float)[mask]
        pitch_reference_points = np.array(self.pitch_config.vertices, dtype=float)[mask]

        try:
            self.last_transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points,
            )
        except Exception as e:
            print(f"[pitch] transformer build failed at frame {frame_idx}: {e}")
            return None

        return self.last_transformer

    def draw_keypoints_overlay(self, frame: np.ndarray, kps: Optional[sv.KeyPoints]) -> np.ndarray:
        """
        Draws keypoints as circles + indices for debugging.
        """
        out = frame.copy()

        if kps is None or kps.xy is None or kps.confidence is None or len(kps.xy) == 0:
            cv2.putText(out, "Pitch KPs: none", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return out

        pts = np.array(kps.xy[0], dtype=float)
        conf = np.array(kps.confidence[0], dtype=float)
        mask = conf >= float(self.cfg.kp_conf)

        kept = 0
        for i, ((x, y), ok) in enumerate(zip(pts, mask)):
            if not bool(ok):
                continue
            kept += 1
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(out, (xi, yi), 6, (0, 255, 0), -1)
            cv2.putText(out, str(i), (xi + 8, yi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        hud = f"Pitch KPs: {kept}/{len(pts)} (conf>={self.cfg.kp_conf:.2f})"
        if self.last_update_frame_idx is not None:
            hud += f" | last_update={self.last_update_frame_idx}"
        cv2.putText(out, hud, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return out

    def _is_good(self, kps: sv.KeyPoints) -> bool:
        if kps is None or kps.xy is None or kps.confidence is None:
            return False
        if len(kps.xy) == 0:
            return False
        conf = np.array(kps.confidence[0], dtype=float)
        good = int((conf >= float(self.cfg.kp_conf)).sum())
        return good >= int(self.cfg.min_good_kps)