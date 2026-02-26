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
import cv2 
from src.pitch.roboflow_pitch import RoboflowPitch, PitchConfig
from contextlib import ExitStack
from sports.configs.soccer import SoccerPitchConfiguration
from src.pitch.birdeye import render_birdeye_frame
from sports.annotators.soccer import draw_pitch

def make_empty_radar(pitch_config: SoccerPitchConfiguration, text: str | None = None) -> np.ndarray:
    radar = draw_pitch(pitch_config)
    if text:
        cv2.putText(
            radar,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
    return radar


def compose_main_with_radar_bottom_center(
    main_frame: np.ndarray,
    radar_frame: np.ndarray,
    *,
    radar_width_ratio: float = 0.42,   # radar width as a fraction of main width
    pad: int = 16,                     # padding between main and radar + outer padding
    bg_color: tuple[int, int, int] = (0, 0, 0),  # BGR background
) -> np.ndarray:
    """
    Output layout:
      [ main_frame (full width) ]
      [   small radar centered  ]
    """
    mh, mw = main_frame.shape[:2]
    rh, rw = radar_frame.shape[:2]

    # Resize radar to target width (keep aspect ratio)
    target_w = max(1, int(mw * radar_width_ratio))
    target_h = max(1, int(rh * (target_w / rw)))
    radar_small = cv2.resize(radar_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # New canvas
    out_h = mh + pad + target_h + pad
    out = np.full((out_h, mw, 3), bg_color, dtype=np.uint8)

    # Place main on top
    out[0:mh, 0:mw] = main_frame

    # Place radar centered below
    x0 = (mw - target_w) // 2
    y0 = mh + pad
    out[y0:y0 + target_h, x0:x0 + target_w] = radar_small

    return out


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
    # limit detection to frames
    max_frames: int | None = None,
    # keypoints detection on video frames
    pitch_debug: bool = False,
    pitch_stride: int = 15,
    kp_conf: float = 0.5,
    # bird eye view
    birdeye: bool = False,
    side_by_side: bool = False,
):
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    frame_generator = sv.get_video_frames_generator(str(source_path))
    # main output
    video_sink = sv.VideoSink(str(target_video_path), video_info=video_info)

    # pitch config used for transformer + drawing
    pitch_config = SoccerPitchConfiguration()

    # --- shared pitch model/helper (load once if needed) ---
    pitch = None
    if pitch_debug or birdeye or side_by_side:
        pitch = RoboflowPitch(
            PitchConfig(stride=pitch_stride, kp_conf=kp_conf),
            pitch_config=pitch_config,
        )
        pitch.load_from_env()

    # --- pitch debug output (overlay keypoints on real frames) ---
    pitch_sink = None
    pitch_debug_path = None
    if pitch_debug:
        outdir = target_video_path.parent
        base = target_video_path.stem.replace("_team_tracked", "")
        suffix = "_pitch_debug_preview.mp4" if max_frames is not None else "_pitch_debug.mp4"
        pitch_debug_path = outdir / f"{base}{suffix}"
        pitch_sink = sv.VideoSink(str(pitch_debug_path), video_info=video_info)
        print("Pitch debug enabled:", pitch_debug_path)

    # --- pitch debug output (overlay keypoints on real frames) ---
    pitch_sink = None
    pitch_debug_path = None
    if pitch_debug:
        outdir = target_video_path.parent
        base = target_video_path.stem.replace("_team_tracked", "")
        suffix = "_pitch_debug_preview.mp4" if max_frames is not None else "_pitch_debug.mp4"
        pitch_debug_path = outdir / f"{base}{suffix}"
        pitch_sink = sv.VideoSink(str(pitch_debug_path), video_info=video_info)
        print("Pitch debug enabled:", pitch_debug_path)

    # --- bird-eye (radar) output ---
    birdeye_sink = None
    birdeye_path = None
    if birdeye:
        outdir = target_video_path.parent
        base = target_video_path.stem.replace("_team_tracked", "")
        suffix = "_birdeye_preview.mp4" if max_frames is not None else "_birdeye.mp4"
        birdeye_path = outdir / f"{base}{suffix}"
        print("Birdeye enabled:", birdeye_path)

    # --- side-by-side output (left: annotated, right: radar) ---
    sbs_sink = None
    sbs_path = None
    if side_by_side:
        outdir = target_video_path.parent
        base = target_video_path.stem.replace("_team_tracked", "")
        suffix = "_side_by_side_preview.mp4" if max_frames is not None else "_side_by_side.mp4"
        sbs_path = outdir / f"{base}{suffix}"
        print("Side-by-side enabled:", sbs_path)

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

    processed = 0
    with ExitStack() as stack:
        stack.enter_context(video_sink)
        if pitch_sink is not None:
            stack.enter_context(pitch_sink)

        try:
            for frame, r in tqdm(zip(frame_generator, results_stream), total=video_info.total_frames):
                processed += 1
                if max_frames is not None and processed > max_frames:
                    break

                frame_idx = processed - 1

                # --- pitch debug overlay  ---
                if pitch_debug and pitch_sink is not None and pitch is not None:
                    kps = pitch.maybe_infer_keypoints(frame, frame_idx=frame_idx)
                    dbg = pitch.draw_keypoints_overlay(frame, kps)
                    pitch_sink.write_frame(dbg)

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

                # --- bird-eye / side-by-side ---
                if (birdeye_path is not None) or (sbs_path is not None):
                    if pitch is None:
                        continue

                    transformer = pitch.maybe_get_transformer(frame, frame_idx=frame_idx)
                    if transformer is None:
                        # fallback radar frame so output videos remain playable / same length
                        radar = make_empty_radar(pitch_config, text="NO TRANSFORMER")
                    
                    else:
                        players_and_gk = sv.Detections.merge([players, goalkeepers])
                        players_and_gk.class_id = players_and_gk.class_id.astype(int)

                        radar = render_birdeye_frame(
                            pitch_config=pitch_config,
                            transformer=transformer,
                            ball=ball,
                            players_and_gk=players_and_gk,
                            referees=referees,
                        )

                    # ---- Lazy-init birdeye sink with correct size ----
                    if birdeye_path is not None:
                        if birdeye_sink is None:
                            rh, rw = radar.shape[:2]
                            birdeye_info = sv.VideoInfo(
                                width=rw,
                                height=rh,
                                fps=video_info.fps,
                                total_frames=video_info.total_frames,
                            )
                            birdeye_sink = sv.VideoSink(str(birdeye_path), video_info=birdeye_info)
                            birdeye_sink.__enter__()
                        birdeye_sink.write_frame(radar)

                    # ---- Lazy-init sbs sink with correct size ----
                    if sbs_path is not None:
                        # Resize radar to match main frame height, then concat horizontally
                        # h = annotated.shape[0]
                        # rh, rw = radar.shape[:2]
                        # new_w = int(rw * (h / rh))
                        # radar_resized = cv2.resize(radar, (new_w, h), interpolation=cv2.INTER_AREA)

                        # sbs = cv2.hconcat([annotated, radar_resized])
                        sbs = compose_main_with_radar_bottom_center(
                            main_frame=annotated,
                            radar_frame=radar,
                            radar_width_ratio=0.40,  # tweak this
                            pad=16,
                        )

                        if sbs_sink is None:
                            sh, sw = sbs.shape[:2]
                            sbs_info = sv.VideoInfo(
                                width=sw,
                                height=sh,
                                fps=video_info.fps,
                                total_frames=video_info.total_frames,
                            )
                            sbs_sink = sv.VideoSink(str(sbs_path), video_info=sbs_info)
                            sbs_sink.__enter__()

                        sbs_sink.write_frame(sbs)
        finally:
            # Close lazily-opened sinks
            if birdeye_sink is not None:
                birdeye_sink.__exit__(None, None, None)
            if sbs_sink is not None:
                sbs_sink.__exit__(None, None, None)

    print("Saved:", target_video_path)
    
    if pitch_debug_path is not None:
        print("Saved pitch debug:", pitch_debug_path)
    if birdeye_path is not None:
        print("Saved birdeye:", birdeye_path)
    if sbs_path is not None:
        print("Saved side-by-side:", sbs_path)
