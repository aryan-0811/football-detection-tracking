from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from tqdm import tqdm
from ultralytics import YOLO

from src.ball.detector import BallDetector
from src.heatmap.player_heatmap import TrackHeatmapStore
from src.pitch.birdeye import render_birdeye_frame
from src.pitch.roboflow_pitch import RoboflowPitch, PitchConfig
from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id
from .annotate import make_team_annotators, make_team_labels
from .detections import detections_from_ultralytics


def _output_path(target_video_path: Path, label: str, preview: bool, ext: str = ".mp4") -> Path:
    """Build an output path like '{base}_{label}[_preview]{ext}' next to the main output."""
    base = target_video_path.stem.replace("_team_tracked", "")
    preview_tag = "_preview" if preview else ""
    return target_video_path.parent / f"{base}_{label}{preview_tag}{ext}"


class _LazyVideoSink:
    """VideoSink that defers initialization until the first frame (when dimensions are known)."""

    def __init__(self, path: Path, video_info: sv.VideoInfo):
        self._path = path
        self._video_info = video_info
        self._sink: sv.VideoSink | None = None

    def write_frame(self, frame: np.ndarray) -> None:
        if self._sink is None:
            h, w = frame.shape[:2]
            info = sv.VideoInfo(
                width=w, height=h,
                fps=self._video_info.fps,
                total_frames=self._video_info.total_frames,
            )
            self._sink = sv.VideoSink(str(self._path), video_info=info)
            self._sink.__enter__()
        self._sink.write_frame(frame)

    def close(self) -> None:
        if self._sink is not None:
            self._sink.__exit__(None, None, None)


def _make_empty_radar(pitch_config: SoccerPitchConfiguration, text: str | None = None) -> np.ndarray:
    radar = draw_pitch(pitch_config)
    if text:
        cv2.putText(
            radar, text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
        )
    return radar


def compose_main_with_radar_bottom_center(
    main_frame: np.ndarray,
    radar_frame: np.ndarray,
    *,
    radar_width_ratio: float = 0.42,   # radar width as a fraction of main width
    pad: int = 16,                     # padding between main and radar
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

    out_h = mh + pad + target_h + pad
    out = np.full((out_h, mw, 3), bg_color, dtype=np.uint8)
    out[0:mh, 0:mw] = main_frame

    # Place radar centered below main frame
    x0 = (mw - target_w) // 2
    y0 = mh + pad
    out[y0:y0 + target_h, x0:x0 + target_w] = radar_small

    return out


def _render_and_write_radar(
    pitch_config: SoccerPitchConfiguration,
    transformer,
    ball: sv.Detections,
    players_and_gk: sv.Detections,
    referees: sv.Detections,
    annotated: np.ndarray,
    birdeye_writer: _LazyVideoSink | None,
    sbs_writer: _LazyVideoSink | None,
) -> None:
    """Render the bird-eye radar frame and write to birdeye / side-by-side sinks."""
    if birdeye_writer is None and sbs_writer is None:
        return

    if transformer is None:
        radar = _make_empty_radar(pitch_config, text="NO TRANSFORMER")
    else:
        radar = render_birdeye_frame(
            pitch_config=pitch_config,
            transformer=transformer,
            ball=ball,
            players_and_gk=players_and_gk,
            referees=referees,
        )

    if birdeye_writer is not None:
        birdeye_writer.write_frame(radar)

    if sbs_writer is not None:
        sbs = compose_main_with_radar_bottom_center(
            main_frame=annotated,
            radar_frame=radar,
            radar_width_ratio=0.40,
            pad=16,
        )
        sbs_writer.write_frame(sbs)


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
    max_frames: int | None = None,
    # pitch keypoints
    pitch_debug: bool = False,
    pitch_stride: int = 15,
    kp_conf: float = 0.5,
    # bird-eye view
    birdeye: bool = False,
    side_by_side: bool = False,
    # heatmaps
    save_heatmaps: bool = False,
    heatmap_top_n: int | None = None,
    heatmap_min_samples: int = 30,
    # ball model
    ball_model_path: Path | None = None,
    ball_conf: float = 0.05,
    ball_max_jump_px: float = 80.0,
    ball_min_conf: float = 0.25,
):
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    frame_generator = sv.get_video_frames_generator(str(source_path))
    video_sink = sv.VideoSink(str(target_video_path), video_info=video_info)
    preview = max_frames is not None

    pitch_config = SoccerPitchConfiguration()

    pitch = None
    if pitch_debug or birdeye or side_by_side:
        pitch = RoboflowPitch(
            PitchConfig(stride=pitch_stride, kp_conf=kp_conf),
            pitch_config=pitch_config,
        )
        pitch.load_from_env()

    heatmap_store = None
    heatmap_dir = None
    if save_heatmaps:
        heatmap_dir = _output_path(target_video_path, "heatmaps", preview, ext="")
        heatmap_store = TrackHeatmapStore(
            pitch_config=pitch_config,
            min_samples_to_save=heatmap_min_samples,
        )
        print("Heatmaps enabled:", heatmap_dir)

    pitch_sink = None
    pitch_debug_path = None
    if pitch_debug:
        pitch_debug_path = _output_path(target_video_path, "pitch_debug", preview)
        pitch_sink = sv.VideoSink(str(pitch_debug_path), video_info=video_info)
        print("Pitch debug enabled:", pitch_debug_path)

    birdeye_path = _output_path(target_video_path, "birdeye", preview) if birdeye else None
    birdeye_writer = _LazyVideoSink(birdeye_path, video_info) if birdeye_path else None
    if birdeye_path:
        print("Birdeye enabled:", birdeye_path)

    sbs_path = _output_path(target_video_path, "side_by_side", preview) if side_by_side else None
    sbs_writer = _LazyVideoSink(sbs_path, video_info) if sbs_path else None
    if sbs_path:
        print("Side-by-side enabled:", sbs_path)

    model = YOLO(str(model_path))

    ball_detector = None
    if ball_model_path is not None:
        ball_detector = BallDetector(
            model_path=ball_model_path,
            conf=ball_conf,
            max_jump_px=ball_max_jump_px,
            min_conf=ball_min_conf,
            imgsz=imgsz,
        )

    assigner = TeamAssigner(device=team_device, smooth_window=team_smooth)
    warmup = WarmupConfig(seconds=warmup_seconds, stride=warmup_stride, max_crops=max_warmup_crops,
                          conf=max(conf, 0.25), iou=iou)
    n_crops = assigner.fit_from_video(str(source_path), model, player_class_id=player_id, imgsz=imgsz, warmup=warmup)
    print(f"TeamClassifier fit complete. Crops used: {n_crops}")

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

                if pitch_debug and pitch_sink is not None and pitch is not None:
                    kps = pitch.maybe_infer_keypoints(frame, frame_idx=frame_idx)
                    dbg = pitch.draw_keypoints_overlay(frame, kps)
                    pitch_sink.write_frame(dbg)

                dets = detections_from_ultralytics(r)

                if ball_detector is not None:
                    ball = ball_detector.predict(frame)
                else:
                    ball = sv.Detections.empty()

                # No tracked detections — still annotate ball and write radar outputs
                if len(dets) == 0:
                    annotated = frame.copy()
                    if len(ball) > 0:
                        annotated = triangle_annotator.annotate(scene=annotated, detections=ball)

                    video_sink.write_frame(annotated)

                    if (birdeye_writer or sbs_writer) and pitch is not None:
                        transformer = pitch.maybe_get_transformer(frame, frame_idx=frame_idx)
                        _render_and_write_radar(
                            pitch_config, transformer, ball,
                            sv.Detections.empty(), sv.Detections.empty(),
                            annotated, birdeye_writer, sbs_writer,
                        )
                    continue

                goalkeepers = dets[dets.class_id == goalkeeper_id]
                players = dets[dets.class_id == player_id]
                referees = dets[dets.class_id == referee_id]

                players = assigner.predict_players(frame, players)

                # Assign each goalkeeper to the nearest team centroid (0/1); unknown -> 2
                if len(goalkeepers) > 0:
                    gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
                    goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)

                if len(referees) > 0:
                    referees.class_id = np.full(len(referees), 2, dtype=int)

                team_dets = sv.Detections.merge([players, goalkeepers, referees])
                team_dets.class_id = team_dets.class_id.astype(int)

                labels = make_team_labels(team_dets, show_id=show_id)
                annotated = box_annot.annotate(scene=frame.copy(), detections=team_dets)
                annotated = label_annot.annotate(scene=annotated, detections=team_dets, labels=labels)
                annotated = triangle_annotator.annotate(scene=annotated, detections=ball)

                video_sink.write_frame(annotated)

                if (birdeye_writer or sbs_writer or heatmap_store) and pitch is not None:
                    transformer = pitch.maybe_get_transformer(frame, frame_idx=frame_idx)

                    players_and_gk = sv.Detections.merge([players, goalkeepers])
                    if len(players_and_gk) > 0 and players_and_gk.class_id is not None:
                        players_and_gk.class_id = players_and_gk.class_id.astype(int)

                    if transformer is not None and heatmap_store is not None and len(players_and_gk) > 0:
                        heatmap_store.update(players_and_gk, transformer)

                    _render_and_write_radar(
                        pitch_config, transformer, ball,
                        players_and_gk, referees,
                        annotated, birdeye_writer, sbs_writer,
                    )
        finally:
            if birdeye_writer is not None:
                birdeye_writer.close()
            if sbs_writer is not None:
                sbs_writer.close()

    print("Saved:", target_video_path)

    if heatmap_store is not None and heatmap_dir is not None:
        saved_heatmaps = heatmap_store.save_all(
            out_dir=heatmap_dir,
            top_n=heatmap_top_n,
        )
        print(f"Saved {len(saved_heatmaps)} heatmaps to: {heatmap_dir}")

    if pitch_debug_path is not None:
        print("Saved pitch debug:", pitch_debug_path)
    if birdeye_path is not None:
        print("Saved birdeye:", birdeye_path)
    if sbs_path is not None:
        print("Saved side-by-side:", sbs_path)
