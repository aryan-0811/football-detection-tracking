#!/usr/bin/env python3
"""
Export the final video assets used in the project screencast.

Runs each asset as a small wrapper around the existing pipeline modules under
`src/` (detection, tracking, team assignment, pitch keypoints, bird-eye,
offside, stats). Visual style for class labels mirrors `scripts/detect_video.py`.

Usage:
    python scripts/export_screencast_assets.py \
        --source-video input_videos/input.mp4 \
        --output-dir outputs/screencast_assets \
        --start 00:00:05 --end 00:00:35 \
        --assets all

    # Only a subset
    python scripts/export_screencast_assets.py --source-video input_videos/input.mp4 \
        --assets raw,detection_all,tracking_ids

    # Dry-run (no video produced)
    python scripts/export_screencast_assets.py --source-video input_videos/input.mp4 \
        --assets all --dry-run
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from tqdm import tqdm
from ultralytics import YOLO

# Allow `python scripts/export_screencast_assets.py …` from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project modules
from src.ball.detector import BallDetector
from src.ball.offside import OffsideDetector
from src.pipeline.detections import detections_from_ultralytics
from src.pitch.birdeye import render_birdeye_frame
from src.pitch.roboflow_pitch import PitchConfig, RoboflowPitch
from src.stats.tracker import StatsTracker
from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id


# ---------------------------------------------------------------------------
# Visual style — mirrors scripts/detect_video.py so labels look identical.
# ---------------------------------------------------------------------------

GENERIC_CLASS_NAMES = {0: "Ball", 1: "Goalkeeper", 2: "Player", 3: "Referee"}

GENERIC_CLASS_PALETTE = sv.ColorPalette.from_hex([
    "#FF6600",  # 0 Ball       — orange
    "#00CC44",  # 1 Goalkeeper — green
    "#00AAFF",  # 2 Player     — blue
    "#FFD700",  # 3 Referee    — gold
])

GENERIC_CLASS_BGR = {
    0: (0, 102, 255),
    1: (68, 204, 0),
    2: (255, 170, 0),
    3: (0, 215, 255),
}
GENERIC_TEXT_BGR = {
    0: (255, 255, 255),
    1: (255, 255, 255),
    2: (255, 255, 255),
    3: (0, 0, 0),
}
GENERIC_OUTLINE_CLASSES = {0, 1, 2}


# Team-coloured palette used by every "final system" asset.
# 0 = Team A (cyan), 1 = Team B (magenta), 2 = referee/other (gold), 3 = ball (white).
TEAM_PALETTE = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700", "#FFFFFF"])
TEAM_NAMES = {0: "Team A", 1: "Team B", 2: "Referee", 3: "Ball"}
TEAM_BGR = {
    0: (255, 191, 0),     # cyan
    1: (147, 20, 255),    # magenta/pink
    2: (0, 215, 255),     # gold
    3: (255, 255, 255),   # white
}
TEAM_TEXT_BGR = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (0, 0, 0),
    3: (0, 0, 0),
}
TEAM_OUTLINE_CLASSES = {1}


def _draw_pill_label(
    scene: np.ndarray,
    xyxy: np.ndarray,
    text: str,
    fill_bgr: tuple[int, int, int],
    text_bgr: tuple[int, int, int],
    *,
    outline: bool = False,
    font_scale: float = 0.6,
    text_thickness: int = 1,
    padding: int = 6,
) -> None:
    """Pill-style label that mirrors the look of scripts/detect_video.py."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = scene.shape[:2]
    x1, y1, x2, _ = xyxy.astype(int)
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    bx1 = x1
    by2 = y1
    by1 = by2 - th - 2 * padding - baseline // 2
    if by1 < 0:
        by1 = y1
        by2 = by1 + th + 2 * padding + baseline // 2
    bx2 = min(w - 1, bx1 + tw + 2 * padding)
    bx1 = max(0, bx1)
    cv2.rectangle(scene, (bx1, by1), (bx2, by2), fill_bgr, thickness=-1)
    tx = bx1 + padding
    ty = by2 - padding - baseline // 2
    if outline:
        cv2.putText(scene, text, (tx, ty), font, font_scale, (0, 0, 0),
                    text_thickness + 2, cv2.LINE_AA)
    cv2.putText(scene, text, (tx, ty), font, font_scale, text_bgr,
                text_thickness, cv2.LINE_AA)


def _draw_generic_labels(
    scene: np.ndarray,
    detections: sv.Detections,
    *,
    font_scale: float,
    text_thickness: int,
) -> np.ndarray:
    """Generic-detector style: per-class colour pill, class name only (no conf)."""
    for i in range(len(detections)):
        cls = int(detections.class_id[i])
        text = GENERIC_CLASS_NAMES.get(cls, f"cls_{cls}")
        _draw_pill_label(
            scene, detections.xyxy[i], text,
            fill_bgr=GENERIC_CLASS_BGR.get(cls, (200, 200, 200)),
            text_bgr=GENERIC_TEXT_BGR.get(cls, (0, 0, 0)),
            outline=cls in GENERIC_OUTLINE_CLASSES,
            font_scale=font_scale,
            text_thickness=text_thickness,
        )
    return scene


def _draw_team_labels(
    scene: np.ndarray,
    detections: sv.Detections,
    *,
    font_scale: float,
    text_thickness: int,
    label_for: Optional[Callable[[int, sv.Detections], Optional[str]]] = None,
) -> np.ndarray:
    """Team-coloured pill label. `label_for(i, dets)` returns the label or None to skip."""
    for i in range(len(detections)):
        cls = int(detections.class_id[i])
        text = label_for(i, detections) if label_for else TEAM_NAMES.get(cls, f"cls_{cls}")
        if text is None:
            continue
        _draw_pill_label(
            scene, detections.xyxy[i], text,
            fill_bgr=TEAM_BGR.get(cls, (200, 200, 200)),
            text_bgr=TEAM_TEXT_BGR.get(cls, (0, 0, 0)),
            outline=cls in TEAM_OUTLINE_CLASSES,
            font_scale=font_scale,
            text_thickness=text_thickness,
        )
    return scene


# ---------------------------------------------------------------------------
# CLI / config plumbing
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    source: Path
    outdir: Path
    object_model: Path
    ball_model: Path
    tracker: str
    imgsz: int
    conf: float
    iou: float
    ball_conf: float
    ball_min_conf: float
    ball_max_jump_px: float
    pitch_stride: int
    kp_conf: float
    warmup_seconds: float
    warmup_stride: int
    max_warmup_crops: int
    team_device: str
    team_smooth: int
    font_scale: float
    box_thickness: int
    text_thickness: int
    start_frame: int
    end_frame: int   # exclusive
    fps: float
    width: int
    height: int
    overwrite: bool
    dry_run: bool
    seed: int

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


ALL_ASSETS = [
    "raw",
    "detection_all",
    "generic_ball_fragmented",
    "dedicated_ball_consistent",
    "team_tracking",
    "tracking_ids",
    "pitch_keypoints",
    "pitch_2d",
    "offside",
    "pass_possession",
]

ASSET_FILES = {
    "raw": "raw_broadcast_10s.mp4",
    "detection_all": "detection_all_classes.mp4",
    "generic_ball_fragmented": "generic_detector_ball_fragmented.mp4",
    "dedicated_ball_consistent": "dedicated_ball_detector_consistent.mp4",
    "team_tracking": "team_coloured_tracking.mp4",
    "tracking_ids": "tracking_ids_only.mp4",
    "pitch_keypoints": "pitch_keypoints_broadcast.mp4",
    "pitch_2d": "pitch_2d_same_clip.mp4",
    "offside": "offside_detection_broadcast_and_2d.mp4",
    "pass_possession": "pass_detection_possession.mp4",
}


def _parse_timestamp(ts: str) -> float:
    """Accept HH:MM:SS, MM:SS, plain seconds (int/float)."""
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = parts
            return m * 60 + s
    return float(ts)


def _setup_logging(verbose: bool) -> None:
    # `force=True` displaces any handlers installed by upstream imports (e.g.
    # the `inference` package configures rich logging on import).
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stderr,
        force=True,
    )


def _resolve_frame_range(source: Path, start: Optional[str], end: Optional[str]) -> tuple[int, int, float, int, int]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    s_frame = 0
    e_frame = total
    if start is not None:
        s_frame = max(0, int(round(_parse_timestamp(start) * fps)))
    if end is not None:
        e_frame = min(total, int(round(_parse_timestamp(end) * fps)))
    if e_frame <= s_frame:
        raise ValueError(f"Empty range: start={s_frame} end={e_frame}")
    return s_frame, e_frame, fps, w, h


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for: {path}")
    return writer


def _seek(cap: cv2.VideoCapture, frame_idx: int) -> None:
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


def _should_skip(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        logging.info("[skip] %s already exists (use --overwrite to regenerate)", path.name)
        return True
    return False


def _letterbox(frame: np.ndarray, target_w: int, target_h: int,
               bg: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Resize frame to fit (target_w, target_h) with letterbox padding."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Shared resources (lazily built so a partial --assets list stays cheap)
# ---------------------------------------------------------------------------

@dataclass
class _LazyResources:
    cfg: ExportConfig
    object_model: Optional[YOLO] = None
    ball_detector_cls: Optional[type[BallDetector]] = None
    team_assigner: Optional[TeamAssigner] = None
    pitch: Optional[RoboflowPitch] = None
    pitch_config: SoccerPitchConfiguration = None  # type: ignore

    def __post_init__(self):
        self.pitch_config = SoccerPitchConfiguration()

    def get_object_model(self) -> YOLO:
        if self.object_model is None:
            logging.info("Loading object detection model: %s", self.cfg.object_model)
            self.object_model = YOLO(str(self.cfg.object_model))
        return self.object_model

    def make_ball_detector(self) -> BallDetector:
        # Each call creates its own (so smoothing state is fresh per asset)
        return BallDetector(
            model_path=self.cfg.ball_model,
            conf=self.cfg.ball_conf,
            min_conf=self.cfg.ball_min_conf,
            max_jump_px=self.cfg.ball_max_jump_px,
            imgsz=self.cfg.imgsz,
        )

    def get_team_assigner(self) -> TeamAssigner:
        if self.team_assigner is None:
            assigner = TeamAssigner(device=self.cfg.team_device, smooth_window=self.cfg.team_smooth)
            warmup = WarmupConfig(
                seconds=self.cfg.warmup_seconds,
                stride=self.cfg.warmup_stride,
                max_crops=self.cfg.max_warmup_crops,
                conf=max(self.cfg.conf, 0.25),
                iou=self.cfg.iou,
            )
            logging.info("Fitting TeamClassifier (warmup %.1fs)…", self.cfg.warmup_seconds)
            n_crops = assigner.fit_from_video(
                str(self.cfg.source), self.get_object_model(),
                player_class_id=2, imgsz=self.cfg.imgsz, warmup=warmup,
            )
            logging.info("TeamClassifier fitted on %d crops", n_crops)
            self.team_assigner = assigner
        return self.team_assigner

    def get_pitch(self) -> RoboflowPitch:
        if self.pitch is None:
            logging.info("Loading Roboflow pitch keypoint model from .env…")
            pitch = RoboflowPitch(
                PitchConfig(stride=self.cfg.pitch_stride, kp_conf=self.cfg.kp_conf),
                pitch_config=self.pitch_config,
            )
            pitch.load_from_env()
            self.pitch = pitch
        return self.pitch


# ---------------------------------------------------------------------------
# Per-asset helpers
# ---------------------------------------------------------------------------

@contextmanager
def _opened_clip(cfg: ExportConfig):
    cap = cv2.VideoCapture(str(cfg.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.source}")
    _seek(cap, cfg.start_frame)
    try:
        yield cap
    finally:
        cap.release()


def _team_pipeline_step(
    frame: np.ndarray,
    raw_dets: sv.Detections,
    res: _LazyResources,
    ball_detector: Optional[BallDetector],
    *,
    ball_id: int = 0,
    goalkeeper_id: int = 1,
    player_id: int = 2,
    referee_id: int = 3,
) -> tuple[sv.Detections, sv.Detections, sv.Detections, sv.Detections, sv.Detections]:
    """
    Run team prediction + GK resolution + (optional) dedicated ball detection.

    Returns (team_dets, players_and_gk, referees, ball, players_only_team)
    where class_id of team_dets/players_and_gk is in {0, 1, 2}.
    """
    if len(raw_dets) == 0:
        ball = ball_detector.predict(frame) if ball_detector else sv.Detections.empty()
        return (sv.Detections.empty(), sv.Detections.empty(), sv.Detections.empty(),
                ball, sv.Detections.empty())

    goalkeepers = raw_dets[raw_dets.class_id == goalkeeper_id]
    players = raw_dets[raw_dets.class_id == player_id]
    referees = raw_dets[raw_dets.class_id == referee_id]

    players = res.get_team_assigner().predict_players(frame, players)

    if len(goalkeepers) > 0:
        gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
        goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)

    if len(referees) > 0:
        referees.class_id = np.full(len(referees), 2, dtype=int)

    if ball_detector is not None:
        ball = ball_detector.predict(frame)
    else:
        # Fall back to ball detections from the generic model
        ball = raw_dets[raw_dets.class_id == ball_id]

    team_dets = sv.Detections.merge([players, goalkeepers, referees])
    if len(team_dets) > 0 and team_dets.class_id is not None:
        team_dets.class_id = team_dets.class_id.astype(int)

    players_and_gk = sv.Detections.merge([players, goalkeepers])
    if len(players_and_gk) > 0 and players_and_gk.class_id is not None:
        players_and_gk.class_id = players_and_gk.class_id.astype(int)

    return team_dets, players_and_gk, referees, ball, players


def _yolo_track_one(model: YOLO, frame: np.ndarray, cfg: ExportConfig) -> sv.Detections:
    """Run model.track on a single frame with persisted state."""
    r = model.track(
        frame, persist=True, tracker=cfg.tracker,
        imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False,
    )[0]
    return detections_from_ultralytics(r)


def _yolo_predict_one(model: YOLO, frame: np.ndarray, cfg: ExportConfig) -> sv.Detections:
    r = model.predict(
        frame, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False,
    )[0]
    return detections_from_ultralytics(r)


def _draw_ball_triangle(scene: np.ndarray, ball: sv.Detections,
                        color_bgr: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Small downward-pointing triangle above the ball box."""
    if len(ball) == 0:
        return
    for box in ball.xyxy:
        x1, y1, x2, _ = box.astype(int)
        cx = (x1 + x2) // 2
        ty = max(0, y1 - 18)
        pts = np.array([[cx - 10, ty - 14], [cx + 10, ty - 14], [cx, ty]], dtype=np.int32)
        cv2.drawContours(scene, [pts], 0, color_bgr, -1)
        cv2.drawContours(scene, [pts], 0, (0, 0, 0), 2)


# ---------------------------------------------------------------------------
# Asset 1 — raw_broadcast_10s.mp4
# ---------------------------------------------------------------------------

def export_raw_clip(cfg: ExportConfig, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:raw] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))
    with _opened_clip(cfg) as cap:
        for _ in tqdm(range(cfg.n_frames), desc="raw", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
    writer.release()
    logging.info("[asset:raw] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 2 — detection_all_classes.mp4
# ---------------------------------------------------------------------------

def export_detection_all_classes(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:detection_all] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    model = res.get_object_model()
    box_ann = sv.BoxAnnotator(color=GENERIC_CLASS_PALETTE, thickness=cfg.box_thickness)
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))
    with _opened_clip(cfg) as cap:
        for _ in tqdm(range(cfg.n_frames), desc="detection_all", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            dets = _yolo_predict_one(model, frame, cfg)
            annotated = box_ann.annotate(scene=frame.copy(), detections=dets)
            _draw_generic_labels(annotated, dets,
                                 font_scale=cfg.font_scale,
                                 text_thickness=cfg.text_thickness)
            writer.write(annotated)
    writer.release()
    logging.info("[asset:detection_all] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 3 — generic_detector_ball_fragmented.mp4
# ---------------------------------------------------------------------------

def export_generic_detector_clip(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:generic_ball_fragmented] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    model = res.get_object_model()
    box_ann = sv.BoxAnnotator(color=GENERIC_CLASS_PALETTE, thickness=cfg.box_thickness)
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))

    # 70-frame "ball blackout" window centred in the clip; reveal the ball on
    # at most ~10 randomly-chosen frames inside it.
    rng = random.Random(cfg.seed)
    blackout_len = min(70, max(0, cfg.n_frames))
    blackout_start = max(0, (cfg.n_frames - blackout_len) // 2)
    blackout_end = blackout_start + blackout_len
    if blackout_len > 0:
        n_visible = min(10, blackout_len)
        visible_in_window = set(
            rng.sample(range(blackout_start, blackout_end), n_visible)
        )
    else:
        visible_in_window = set()
    logging.info("[asset:generic_ball_fragmented] blackout=[%d,%d) visible=%d",
                 blackout_start, blackout_end, len(visible_in_window))

    with _opened_clip(cfg) as cap:
        for idx in tqdm(range(cfg.n_frames), desc="generic_ball", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            dets = _yolo_predict_one(model, frame, cfg)
            in_blackout = blackout_start <= idx < blackout_end
            if in_blackout and idx not in visible_in_window:
                # drop ball detections (class 0 in generic detector)
                keep = dets.class_id != 0
                dets = dets[keep]
            annotated = box_ann.annotate(scene=frame.copy(), detections=dets)
            _draw_generic_labels(annotated, dets,
                                 font_scale=cfg.font_scale,
                                 text_thickness=cfg.text_thickness)
            writer.write(annotated)
    writer.release()
    logging.info("[asset:generic_ball_fragmented] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 4 — dedicated_ball_detector_consistent.mp4
# ---------------------------------------------------------------------------

def export_dedicated_ball_detector_clip(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:dedicated_ball_consistent] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    model = res.get_object_model()
    ball_detector = res.make_ball_detector()
    box_ann = sv.BoxAnnotator(color=GENERIC_CLASS_PALETTE, thickness=cfg.box_thickness)
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))

    with _opened_clip(cfg) as cap:
        for _ in tqdm(range(cfg.n_frames), desc="dedicated_ball", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            dets = _yolo_predict_one(model, frame, cfg)
            # drop ball from the generic detector — the dedicated model
            # supplies its own (more consistent) ball box.
            non_ball = dets[dets.class_id != 0]
            ball = ball_detector.predict(frame)
            if len(ball) > 0:
                # Force ball class id to 0 so it picks up the orange palette.
                ball.class_id = np.zeros(len(ball), dtype=int)
            merged = sv.Detections.merge([non_ball, ball]) if len(ball) > 0 else non_ball
            annotated = box_ann.annotate(scene=frame.copy(), detections=merged)
            _draw_generic_labels(annotated, merged,
                                 font_scale=cfg.font_scale,
                                 text_thickness=cfg.text_thickness)
            writer.write(annotated)
    writer.release()
    logging.info("[asset:dedicated_ball_consistent] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 5 — team_coloured_tracking.mp4
# ---------------------------------------------------------------------------

def export_team_coloured_tracking(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:team_tracking] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    model = res.get_object_model()
    res.get_team_assigner()  # warm up
    ball_detector = res.make_ball_detector()
    box_ann = sv.BoxAnnotator(color=TEAM_PALETTE, thickness=cfg.box_thickness)
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))

    with _opened_clip(cfg) as cap:
        for _ in tqdm(range(cfg.n_frames), desc="team_tracking", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            raw = _yolo_track_one(model, frame, cfg)
            team_dets, _, _, ball, _ = _team_pipeline_step(frame, raw, res, ball_detector)

            annotated = frame.copy()
            if len(team_dets) > 0:
                annotated = box_ann.annotate(scene=annotated, detections=team_dets)
                _draw_team_labels(annotated, team_dets,
                                  font_scale=cfg.font_scale,
                                  text_thickness=cfg.text_thickness)
            _draw_ball_triangle(annotated, ball, color_bgr=(255, 255, 255))
            writer.write(annotated)
    writer.release()
    logging.info("[asset:team_tracking] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 6 — tracking_ids_only.mp4
# ---------------------------------------------------------------------------

def export_tracking_ids_only(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:tracking_ids] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    model = res.get_object_model()
    box_color = (60, 220, 60)   # neutral green; not team-coloured
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))

    with _opened_clip(cfg) as cap:
        for _ in tqdm(range(cfg.n_frames), desc="tracking_ids", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            dets = _yolo_track_one(model, frame, cfg)
            annotated = frame.copy()
            if len(dets) > 0:
                for i in range(len(dets)):
                    x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, cfg.box_thickness)
                    if dets.tracker_id is None:
                        continue
                    tid = int(dets.tracker_id[i])
                    _draw_pill_label(
                        annotated, dets.xyxy[i], f"ID:{tid}",
                        fill_bgr=box_color, text_bgr=(0, 0, 0),
                        outline=False,
                        font_scale=cfg.font_scale,
                        text_thickness=cfg.text_thickness,
                    )
            writer.write(annotated)
    writer.release()
    logging.info("[asset:tracking_ids] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 7 — pitch_keypoints_broadcast.mp4
# ---------------------------------------------------------------------------

def export_pitch_keypoints(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:pitch_keypoints] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    try:
        pitch = res.get_pitch()
    except RuntimeError as e:
        logging.warning("[asset:pitch_keypoints] pitch model unavailable: %s — TODO: set ROBOFLOW_API_KEY/RF_FIELD_MODEL_ID", e)
        return

    model = res.get_object_model()
    res.get_team_assigner()
    ball_detector = res.make_ball_detector()
    box_ann = sv.BoxAnnotator(color=TEAM_PALETTE, thickness=max(1, cfg.box_thickness - 1))
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))

    with _opened_clip(cfg) as cap:
        for idx in tqdm(range(cfg.n_frames), desc="pitch_keypoints", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            kps = pitch.maybe_infer_keypoints(frame, frame_idx=idx)
            annotated = frame.copy()

            # Draw keypoints (no indices — keep it uncluttered for the slide).
            if kps is not None and kps.xy is not None and kps.confidence is not None and len(kps.xy) > 0:
                pts = np.array(kps.xy[0], dtype=float)
                conf = np.array(kps.confidence[0], dtype=float)
                for (x, y), c in zip(pts, conf):
                    if c < cfg.kp_conf:
                        continue
                    cv2.circle(annotated, (int(round(x)), int(round(y))), 7, (0, 255, 0), -1)
                    cv2.circle(annotated, (int(round(x)), int(round(y))), 7, (0, 0, 0), 1)

            # Light overlay of team-coloured boxes + ball (no labels).
            raw = _yolo_track_one(model, frame, cfg)
            team_dets, _, _, ball, _ = _team_pipeline_step(frame, raw, res, ball_detector)
            if len(team_dets) > 0:
                annotated = box_ann.annotate(scene=annotated, detections=team_dets)
            _draw_ball_triangle(annotated, ball)
            writer.write(annotated)
    writer.release()
    logging.info("[asset:pitch_keypoints] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 8 — pitch_2d_same_clip.mp4
# ---------------------------------------------------------------------------

def export_pitch_2d(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:pitch_2d] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    try:
        pitch = res.get_pitch()
    except RuntimeError as e:
        logging.warning("[asset:pitch_2d] pitch model unavailable: %s — TODO: set ROBOFLOW_API_KEY/RF_FIELD_MODEL_ID", e)
        return

    model = res.get_object_model()
    res.get_team_assigner()
    ball_detector = res.make_ball_detector()

    # Determine bird-eye canvas size from a single reference render.
    sample = draw_pitch(res.pitch_config)
    sample_h, sample_w = sample.shape[:2]
    writer = _open_writer(out_path, cfg.fps, (sample_w, sample_h))

    with _opened_clip(cfg) as cap:
        for idx in tqdm(range(cfg.n_frames), desc="pitch_2d", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            transformer = pitch.maybe_get_transformer(frame, frame_idx=idx)
            raw = _yolo_track_one(model, frame, cfg)
            team_dets, players_and_gk, referees, ball, _ = _team_pipeline_step(
                frame, raw, res, ball_detector,
            )

            if transformer is None:
                radar = draw_pitch(res.pitch_config)
            else:
                radar = render_birdeye_frame(
                    pitch_config=res.pitch_config,
                    transformer=transformer,
                    ball=ball,
                    players_and_gk=players_and_gk,
                    referees=referees,
                )
            # Ensure the writer sees a fixed size.
            if radar.shape[:2] != (sample_h, sample_w):
                radar = cv2.resize(radar, (sample_w, sample_h), interpolation=cv2.INTER_AREA)
            writer.write(radar)
    writer.release()
    logging.info("[asset:pitch_2d] saved → %s", out_path)


# ---------------------------------------------------------------------------
# Asset 9 — offside_detection_broadcast_and_2d.mp4
# ---------------------------------------------------------------------------

def export_offside_clip(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:offside] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    try:
        pitch = res.get_pitch()
    except RuntimeError as e:
        logging.warning("[asset:offside] pitch model unavailable: %s — TODO: set ROBOFLOW_API_KEY/RF_FIELD_MODEL_ID", e)
        return

    model = res.get_object_model()
    res.get_team_assigner()
    ball_detector = res.make_ball_detector()
    offside = OffsideDetector(pitch_length=float(res.pitch_config.length))

    box_ann = sv.BoxAnnotator(color=TEAM_PALETTE, thickness=cfg.box_thickness)

    # Output is the broadcast clip followed by the 2-D pitch clip,
    # both normalised to the broadcast resolution so cv2 only sees one size.
    target_w, target_h = cfg.width, cfg.height
    writer = _open_writer(out_path, cfg.fps, (target_w, target_h))

    radar_frames: list[np.ndarray] = []   # buffer 2-D so we can append after broadcast

    n_offside_events_seen = 0
    with _opened_clip(cfg) as cap:
        for idx in tqdm(range(cfg.n_frames), desc="offside_broadcast", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            transformer = pitch.maybe_get_transformer(frame, frame_idx=idx)
            raw = _yolo_track_one(model, frame, cfg)
            team_dets, players_and_gk, referees, ball, players = _team_pipeline_step(
                frame, raw, res, ball_detector,
            )

            state = offside.update(
                frame_idx=idx,
                ball=ball,
                players=players,
                goalkeepers=team_dets[team_dets.class_id != 2] if len(team_dets) else sv.Detections.empty(),
                transformer=transformer,
            ) if transformer is not None else None

            annotated = frame.copy()
            if len(team_dets) > 0:
                annotated = box_ann.annotate(scene=annotated, detections=team_dets)
                offside_tids = state.offside_track_ids if state is not None else set()

                def _label(i: int, dets: sv.Detections) -> Optional[str]:
                    if state is None or not state.active or not offside_tids:
                        return None
                    if dets.tracker_id is None:
                        return None
                    return "Offside flag" if int(dets.tracker_id[i]) in offside_tids else None

                _draw_team_labels(annotated, team_dets,
                                  font_scale=cfg.font_scale,
                                  text_thickness=cfg.text_thickness,
                                  label_for=_label)

                # Re-outline offside players in red so they pop on the slide.
                if state is not None and state.active and offside_tids and team_dets.tracker_id is not None:
                    for i, tid in enumerate(team_dets.tracker_id):
                        if int(tid) not in offside_tids:
                            continue
                        x1, y1, x2, y2 = team_dets.xyxy[i].astype(int)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), cfg.box_thickness + 1)

            _draw_ball_triangle(annotated, ball)
            writer.write(annotated)

            # Build 2-D pitch frame for this idx and stash it.
            if transformer is None:
                radar = draw_pitch(res.pitch_config)
            else:
                radar = render_birdeye_frame(
                    pitch_config=res.pitch_config,
                    transformer=transformer,
                    ball=ball,
                    players_and_gk=players_and_gk,
                    referees=referees,
                    offside_state=state,
                )
            radar_frames.append(_letterbox(radar, target_w, target_h))

            n_offside_events_seen = max(n_offside_events_seen, len(offside.offside_events))

    # Append 2-D pitch sequence
    for radar in tqdm(radar_frames, desc="offside_2d", unit="f"):
        writer.write(radar)
    writer.release()

    if n_offside_events_seen == 0:
        logging.warning("[asset:offside] No offside events fired in this range — "
                        "pick a clip that contains a real offside, or extend the warmup. "
                        "TODO: surface a hand-tagged offside frame for the deck.")
    logging.info("[asset:offside] saved → %s (%d offside events)",
                 out_path, n_offside_events_seen)


# ---------------------------------------------------------------------------
# Asset 10 — pass_detection_possession.mp4
# ---------------------------------------------------------------------------

def _draw_possession_hud(scene: np.ndarray, possession: dict[int, int],
                         total_passes: int, pass_flash: Optional[str]) -> None:
    h, w = scene.shape[:2]
    pad = 16
    box_w = 360
    box_h = 130
    overlay = scene.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, scene, 0.45, 0, scene)

    total = max(possession.get(0, 0) + possession.get(1, 0), 1)
    p0 = possession.get(0, 0) / total * 100
    p1 = possession.get(1, 0) / total * 100

    cv2.putText(scene, f"Team A  {p0:5.1f}%", (pad + 12, pad + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_BGR[0], 2, cv2.LINE_AA)
    cv2.putText(scene, f"Team B  {p1:5.1f}%", (pad + 12, pad + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEAM_BGR[1], 2, cv2.LINE_AA)
    cv2.putText(scene, f"Passes : {total_passes}", (pad + 12, pad + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if pass_flash:
        # right-side flash banner
        text = pass_flash
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        x0 = w - tw - 40
        y0 = 40
        cv2.rectangle(scene, (x0 - 16, y0 - th - 12), (x0 + tw + 16, y0 + 16),
                      (0, 215, 255), -1)
        cv2.putText(scene, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 3, cv2.LINE_AA)


def export_pass_possession_clip(cfg: ExportConfig, res: _LazyResources, out_path: Path) -> None:
    if cfg.dry_run or _should_skip(out_path, cfg.overwrite):
        logging.info("[asset:pass_possession] %s", "(dry-run)" if cfg.dry_run else "skipped")
        return
    try:
        pitch = res.get_pitch()
    except RuntimeError as e:
        logging.warning("[asset:pass_possession] pitch model unavailable: %s — TODO: set ROBOFLOW_API_KEY/RF_FIELD_MODEL_ID", e)
        return

    model = res.get_object_model()
    res.get_team_assigner()
    ball_detector = res.make_ball_detector()
    stats = StatsTracker(fps=cfg.fps)

    box_ann = sv.BoxAnnotator(color=TEAM_PALETTE, thickness=cfg.box_thickness)
    writer = _open_writer(out_path, cfg.fps, (cfg.width, cfg.height))

    last_pass_count = 0
    flash_until = -1
    flash_text: Optional[str] = None

    with _opened_clip(cfg) as cap:
        for idx in tqdm(range(cfg.n_frames), desc="pass_possession", unit="f"):
            ok, frame = cap.read()
            if not ok:
                break
            transformer = pitch.maybe_get_transformer(frame, frame_idx=idx)
            raw = _yolo_track_one(model, frame, cfg)
            team_dets, players_and_gk, _, ball, _ = _team_pipeline_step(
                frame, raw, res, ball_detector,
            )

            if transformer is not None and len(players_and_gk) > 0:
                stats.update(players_and_gk, ball, transformer)

            # pylint: disable=protected-access  (read-only HUD peek into stats internals)
            if stats._total_passes > last_pass_count:
                team = stats._holder_team if stats._holder_team is not None else 0
                flash_text = f"PASS! Team {'A' if team == 0 else 'B'}"
                flash_until = idx + int(round(cfg.fps * 1.2))
                last_pass_count = stats._total_passes

            annotated = frame.copy()
            if len(team_dets) > 0:
                annotated = box_ann.annotate(scene=annotated, detections=team_dets)
            _draw_ball_triangle(annotated, ball)
            _draw_possession_hud(
                annotated,
                possession=dict(stats._possession_frames),
                total_passes=stats._total_passes,
                pass_flash=flash_text if idx <= flash_until else None,
            )
            writer.write(annotated)
    writer.release()
    logging.info("[asset:pass_possession] saved → %s (%d passes)",
                 out_path, stats._total_passes)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _dispatch(asset: str, cfg: ExportConfig, res: _LazyResources) -> None:
    out_path = cfg.outdir / ASSET_FILES[asset]
    if asset == "raw":
        export_raw_clip(cfg, out_path)
    elif asset == "detection_all":
        export_detection_all_classes(cfg, res, out_path)
    elif asset == "generic_ball_fragmented":
        export_generic_detector_clip(cfg, res, out_path)
    elif asset == "dedicated_ball_consistent":
        export_dedicated_ball_detector_clip(cfg, res, out_path)
    elif asset == "team_tracking":
        export_team_coloured_tracking(cfg, res, out_path)
    elif asset == "tracking_ids":
        export_tracking_ids_only(cfg, res, out_path)
    elif asset == "pitch_keypoints":
        export_pitch_keypoints(cfg, res, out_path)
    elif asset == "pitch_2d":
        export_pitch_2d(cfg, res, out_path)
    elif asset == "offside":
        export_offside_clip(cfg, res, out_path)
    elif asset == "pass_possession":
        export_pass_possession_clip(cfg, res, out_path)
    else:
        raise ValueError(f"Unknown asset: {asset}")


def _select_assets(spec: str) -> list[str]:
    spec = spec.strip().lower()
    if spec in ("all", ""):
        return list(ALL_ASSETS)
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ASSET_FILES]
    if unknown:
        raise ValueError(f"Unknown asset(s): {unknown}. Choose from {ALL_ASSETS}")
    # Preserve canonical ordering
    return [a for a in ALL_ASSETS if a in requested]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-video", required=True, help="Input MP4 path")
    p.add_argument("--output-dir", default="outputs/screencast_assets",
                   help="Directory where MP4 assets are written")
    p.add_argument("--assets", default="all",
                   help=f"Comma-separated asset names or 'all'. Choices: {ALL_ASSETS}")
    p.add_argument("--start", default=None, help="Start timestamp (HH:MM:SS, MM:SS, or seconds)")
    p.add_argument("--end", default=None, help="End timestamp (exclusive)")
    p.add_argument("--object-model", default="models/object_detection/best.pt")
    p.add_argument("--ball-model", default="models/ball_detection/best.pt")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--ball-conf", type=float, default=0.05)
    p.add_argument("--ball-min-conf", type=float, default=0.25)
    p.add_argument("--ball-max-jump-px", type=float, default=80.0)
    p.add_argument("--pitch-stride", type=int, default=15)
    p.add_argument("--kp-conf", type=float, default=0.5)
    p.add_argument("--warmup-seconds", type=float, default=10.0)
    p.add_argument("--warmup-stride", type=int, default=30)
    p.add_argument("--max-warmup-crops", type=int, default=800)
    p.add_argument("--team-device", default="cpu", choices=["cuda", "cpu", "mps"])
    p.add_argument("--team-smooth", type=int, default=30)
    p.add_argument("--font-scale", type=float, default=0.6)
    p.add_argument("--box-thickness", type=int, default=2)
    p.add_argument("--text-thickness", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260503,
                   help="RNG seed for the generic-ball blackout window")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render assets even if output files already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="List which assets would be generated without writing them")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    source = Path(args.source_video).expanduser().resolve()
    if not source.exists():
        logging.error("Source video not found: %s", source)
        return 2

    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    assets = _select_assets(args.assets)
    s_frame, e_frame, fps, w, h = _resolve_frame_range(source, args.start, args.end)

    cfg = ExportConfig(
        source=source,
        outdir=outdir,
        object_model=Path(args.object_model).expanduser().resolve(),
        ball_model=Path(args.ball_model).expanduser().resolve(),
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        ball_conf=args.ball_conf,
        ball_min_conf=args.ball_min_conf,
        ball_max_jump_px=args.ball_max_jump_px,
        pitch_stride=args.pitch_stride,
        kp_conf=args.kp_conf,
        warmup_seconds=args.warmup_seconds,
        warmup_stride=args.warmup_stride,
        max_warmup_crops=args.max_warmup_crops,
        team_device=args.team_device,
        team_smooth=args.team_smooth,
        font_scale=args.font_scale,
        box_thickness=args.box_thickness,
        text_thickness=args.text_thickness,
        start_frame=s_frame,
        end_frame=e_frame,
        fps=fps,
        width=w,
        height=h,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    logging.info("Source : %s  (%dx%d @ %.2f fps)", source, w, h, fps)
    logging.info("Range  : frames [%d, %d)  (%d frames, ~%.1fs)",
                 s_frame, e_frame, cfg.n_frames, cfg.n_frames / fps)
    logging.info("Outdir : %s", outdir)
    logging.info("Assets : %s", ", ".join(assets))
    if args.dry_run:
        logging.info("(dry-run) — no videos will be written")
        for a in assets:
            logging.info("  would write: %s", outdir / ASSET_FILES[a])
        return 0

    res = _LazyResources(cfg=cfg)
    failures: list[tuple[str, Exception]] = []
    for asset in assets:
        try:
            _dispatch(asset, cfg, res)
        except KeyboardInterrupt:
            logging.warning("Interrupted by user during '%s'", asset)
            raise
        except Exception as e:  # noqa: BLE001 — log & continue with remaining assets
            logging.exception("[asset:%s] failed: %s", asset, e)
            failures.append((asset, e))

    if failures:
        logging.error("Completed with %d failure(s):", len(failures))
        for name, err in failures:
            logging.error("  %s: %s", name, err)
        return 1
    logging.info("All requested assets generated under %s", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
