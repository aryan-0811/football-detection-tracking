from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO


def _empty_ball_detection() -> sv.Detections:
    return sv.Detections.empty()


def _xyxy_center(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(float)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def _result_to_sv_detections(result) -> sv.Detections:
    """Convert one Ultralytics result into sv.Detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return _empty_ball_detection()

    xyxy = boxes.xyxy.cpu().numpy()
    confidence = boxes.conf.cpu().numpy()

    if getattr(boxes, "cls", None) is not None:
        class_id = boxes.cls.cpu().numpy().astype(int)
    else:
        class_id = np.zeros(len(xyxy), dtype=int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )


@dataclass
class BallDetector:
    model_path: Path
    conf: float = 0.05
    max_jump_px: float = 80.0
    min_conf: float = 0.25
    imgsz: int | None = 1280

    def __post_init__(self) -> None:
        self.model = YOLO(str(self.model_path))
        self.prev_center: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_center = None

    def _select_detection(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return _empty_ball_detection()

        xyxy = detections.xyxy
        conf = detections.confidence if detections.confidence is not None else np.ones(len(detections), dtype=float)

        centers = np.array([_xyxy_center(box) for box in xyxy], dtype=np.float32)

        if self.prev_center is None:
            best_idx = int(np.argmax(conf))
        else:
            dists = np.linalg.norm(centers - self.prev_center[None, :], axis=1)
            best_idx = int(np.argmin(dists))

            if float(dists[best_idx]) > float(self.max_jump_px):
                return _empty_ball_detection()

        if float(conf[best_idx]) < float(self.min_conf):
            return _empty_ball_detection()

        chosen = detections[np.array([best_idx])]
        self.prev_center = centers[best_idx]
        return chosen

    def predict(self, frame: np.ndarray) -> sv.Detections:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results:
            return _empty_ball_detection()

        detections = _result_to_sv_detections(results[0])
        return self._select_detection(detections)