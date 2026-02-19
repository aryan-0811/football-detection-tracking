from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Deque, Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

from sports.common.team import TeamClassifier
from .cropping import crop_torso

@dataclass
class WarmupConfig:
    seconds: float = 10.0
    stride: int = 30
    max_crops: int = 800
    min_box_area: int = 35 * 35
    conf: float = 0.25
    iou: float = 0.5


class TrackMajorityVote:
    def __init__(self, window: int = 30):
        self.window = window
        self.hist: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id: int, pred: int) -> int:
        h = self.hist[int(track_id)]
        h.append(int(pred))
        return int(np.bincount(np.array(h, dtype=int)).argmax())


class TeamAssigner:
    """
    Unsupervised team assignment using TeamClassifier (SigLIP + UMAP + KMeans).

    Usage:
      assigner = TeamAssigner(device="cuda", smooth_window=30)
      assigner.fit_from_video(video_path, yolo_model, player_class_id=2, warmup=WarmupConfig(...))
      players = assigner.predict_players(frame, players_detections)  # class_id becomes {0,1}
    """
    def __init__(self, device: str = "cuda", smooth_window: int = 30):
        self.classifier = TeamClassifier(device=device)
        self.smoother = TrackMajorityVote(window=smooth_window)
        self.is_fit = False

    def fit_from_video(
        self,
        source_path: str,
        model: YOLO,
        player_class_id: int,
        imgsz: int,
        warmup: WarmupConfig = WarmupConfig(),
    ) -> int:
        video_info = sv.VideoInfo.from_video_path(source_path)
        max_frames = int(video_info.fps * warmup.seconds)

        frame_gen = sv.get_video_frames_generator(source_path, stride=warmup.stride)

        crops: List[np.ndarray] = []
        frames_seen = 0

        for frame in frame_gen:
            frames_seen += warmup.stride
            if frames_seen > max_frames:
                break

            r = model.predict(frame, imgsz=imgsz, conf=warmup.conf, iou=warmup.iou, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)

            for box, cid in zip(xyxy, class_ids):
                if cid != player_class_id:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) < warmup.min_box_area:
                    continue
                c = crop_torso(frame, box)
                if c is not None:
                    crops.append(c)
                if len(crops) >= warmup.max_crops:
                    break

            if len(crops) >= warmup.max_crops:
                break

        if len(crops) < 50:
            raise RuntimeError(f"Not enough crops to fit TeamClassifier: {len(crops)}")

        self.classifier.fit(crops)
        self.is_fit = True
        return len(crops)

    def predict_players(self, frame: np.ndarray, players: sv.Detections) -> sv.Detections:
        if not self.is_fit:
            raise RuntimeError("TeamAssigner not fit yet. Call fit_from_video() first.")

        if len(players) == 0:
            return players

        crops = []
        keep = []
        for box in players.xyxy:
            c = crop_torso(frame, box)
            if c is None:
                keep.append(False)
            else:
                keep.append(True)
                crops.append(c)

        keep = np.array(keep, dtype=bool)
        players = players[keep]
        if len(players) == 0:
            return players

        team_ids = self.classifier.predict(crops)  # 0/1
        players.class_id = np.asarray(team_ids, dtype=int)

        if players.tracker_id is not None:
            for i, tid in enumerate(players.tracker_id):
                players.class_id[i] = self.smoother.update(int(tid), int(players.class_id[i]))

        return players
