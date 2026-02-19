from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from .detections import detections_from_ultralytics
from .annotate import make_team_annotators, make_team_labels
from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id


def run_video_team_classification(
    model_path: Path,
    source_path: Path,
    target_video_path: Path,
    tracker: str,
    imgsz: int,
    conf: float,
    iou: float,
    show_id: bool,
    font_scale: float,
    box_thickness: int,
    # detector class IDs
    ball_id: int = 0,
    goalkeeper_id: int = 1,
    player_id: int = 2,
    referee_id: int = 3,
    # warmup
    warmup_seconds: float = 10.0,
    warmup_stride: int = 30,
    max_warmup_crops: int = 800,
    team_device: str = "cuda",
    team_smooth: int = 30,
):
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    frame_generator = sv.get_video_frames_generator(str(source_path))
    video_sink = sv.VideoSink(str(target_video_path), video_info=video_info)

    model = YOLO(str(model_path))

    # ---- Warmup fit ----
    assigner = TeamAssigner(device=team_device, smooth_window=team_smooth)
    warmup = WarmupConfig(seconds=warmup_seconds, stride=warmup_stride, max_crops=max_warmup_crops,
                          conf=max(conf, 0.25), iou=iou)
    n_crops = assigner.fit_from_video(str(source_path), model, player_class_id=player_id, imgsz=imgsz, warmup=warmup)
    print(f"TeamClassifier fit complete. Crops used: {n_crops}")

    # ---- Tracking stream ----
    results_stream = model.track(
        source=str(source_path),
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        stream=True,
        verbose=False,
    )

    box_annot, label_annot = make_team_annotators(box_thickness=box_thickness, font_scale=font_scale)
    triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)

    with video_sink:
        for frame, r in tqdm(zip(frame_generator, results_stream), total=video_info.total_frames):
            dets = detections_from_ultralytics(r)
            if len(dets) == 0:
                video_sink.write_frame(frame)
                continue

            # split by detector classes
            ball = dets[dets.class_id == ball_id]
            goalkeepers = dets[dets.class_id == goalkeeper_id]
            players = dets[dets.class_id == player_id]
            referees = dets[dets.class_id == referee_id]

            # team assign players -> class_id becomes 0/1
            players = assigner.predict_players(frame, players)

            # assign GK to nearest team centroid -> 0/1; unknown -> 2
            if len(goalkeepers) > 0:
                gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
                goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)

            # referees as Other (2)
            if len(referees) > 0:
                referees.class_id = np.full(len(referees), 2, dtype=int)

            # merge for annotation (ball ignored in this v2 stage)
            team_dets = sv.Detections.merge([players, goalkeepers, referees])
            team_dets.class_id = team_dets.class_id.astype(int)

            labels = make_team_labels(team_dets, show_id=show_id)
            annotated = box_annot.annotate(scene=frame.copy(), detections=team_dets)
            annotated = label_annot.annotate(scene=annotated, detections=team_dets, labels=labels)
            annotated = triangle_annotator.annotate(scene=annotated, detections=ball)

            video_sink.write_frame(annotated)

    print("Saved:", target_video_path)
