
import os
import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model
from ultralytics import YOLO

from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id

load_dotenv()

# ---- CONFIG ----
VIDEO_PATH = "input_videos/input.mp4"          # change if needed
DETECTOR_WEIGHTS = "models/object_detection/best.pt"            # change if needed
FRAME_IDX = 300                                # change if needed

# detector class IDs
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Roboflow pitch model (from .env)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
RF_FIELD_MODEL_ID = os.getenv("RF_FIELD_MODEL_ID")  # e.g. football-field-detection-f07vi-hwn6o/2

# Homography / keypoints
PITCH_INFER_CONF = 0.3
KP_CONF = 0.5
MIN_KP = 6

# YOLO predict params for this single frame
YOLO_IMGSZ = 1280
YOLO_CONF = 0.2
YOLO_IOU = 0.5

# Team warmup fit params
TEAM_DEVICE = "cpu"        # macOS: cpu
WARMUP_SECONDS = 10.0
WARMUP_STRIDE = 30
MAX_WARMUP_CROPS = 800
TEAM_SMOOTH = 30           # not critical for one frame, but fine

assert ROBOFLOW_API_KEY, "Missing ROBOFLOW_API_KEY in .env"
assert RF_FIELD_MODEL_ID, "Missing RF_FIELD_MODEL_ID in .env"

os.makedirs("outputs/debug", exist_ok=True)

# ---- LOAD MODELS ----
det_model = YOLO(DETECTOR_WEIGHTS)
field_model = get_model(model_id=RF_FIELD_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# ---- FIT TEAM ASSIGNER (warmup) ----
assigner = TeamAssigner(device=TEAM_DEVICE, smooth_window=TEAM_SMOOTH)
warmup_cfg = WarmupConfig(
    seconds=WARMUP_SECONDS,
    stride=WARMUP_STRIDE,
    max_crops=MAX_WARMUP_CROPS,
    conf=max(YOLO_CONF, 0.25),
    iou=YOLO_IOU,
)
n_crops = assigner.fit_from_video(
    source_path=VIDEO_PATH,
    model=det_model,
    player_class_id=PLAYER_ID,
    imgsz=YOLO_IMGSZ,
    warmup=warmup_cfg,
)
print("TeamClassifier fit crops:", n_crops)

# ---- READ ONE FRAME ----
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ok, frame = cap.read()
cap.release()
assert ok and frame is not None, f"Could not read frame {FRAME_IDX}"

# ---- PITCH KEYPOINTS -> TRANSFORM ----
rf_res = field_model.infer(frame, confidence=PITCH_INFER_CONF)[0]
kps = sv.KeyPoints.from_inference(rf_res)

pitch_cfg = SoccerPitchConfiguration()
pitch_vertices = np.array(pitch_cfg.vertices, dtype=np.float32)

mask = kps.confidence[0] > KP_CONF
src = kps.xy[0][mask].astype(np.float32)           # image points
dst = pitch_vertices[mask].astype(np.float32)      # pitch points
print("Pitch keypoints used:", int(mask.sum()))

assert src.shape[0] >= MIN_KP, f"Not enough pitch keypoints for transform. Got {src.shape[0]}, need {MIN_KP}"
transformer = ViewTransformer(source=src, target=dst)

# ---- DETECT OBJECTS ON THIS FRAME ----
r = det_model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]

# Convert to supervision detections
if hasattr(sv.Detections, "from_ultralytics"):
    dets = sv.Detections.from_ultralytics(r)
else:
    # fallback; most modern supervision has from_ultralytics
    dets = sv.Detections.from_inference(r)

ball = dets[dets.class_id == BALL_ID]
goalkeepers = dets[dets.class_id == GOALKEEPER_ID]
players = dets[dets.class_id == PLAYER_ID]
referees = dets[dets.class_id == REFEREE_ID]

# ---- TEAM CLASSIFY PLAYERS ----
players = assigner.predict_players(frame, players)  # players.class_id becomes 0/1

# ---- ASSIGN GOALKEEPERS TO A TEAM ----
if len(goalkeepers) > 0:
    gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
    goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)  # unknown->2

# Referees -> 2 ("Other")
if len(referees) > 0:
    referees.class_id = np.full(len(referees), 2, dtype=int)
    

# ---- PROJECT ANCHORS ----
def anchors(d: sv.Detections) -> np.ndarray:
    return d.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(d) else np.empty((0, 2), dtype=np.float32)

ball_xy = anchors(ball)
players_xy = anchors(players)
gk_xy = anchors(goalkeepers)
refs_xy = anchors(referees)

team0_img = np.vstack([players_xy[players.class_id == 0], gk_xy[goalkeepers.class_id == 0]]) if (len(players_xy) or len(gk_xy)) else np.empty((0, 2), dtype=np.float32)
team1_img = np.vstack([players_xy[players.class_id == 1], gk_xy[goalkeepers.class_id == 1]]) if (len(players_xy) or len(gk_xy)) else np.empty((0, 2), dtype=np.float32)

ball_pitch = transformer.transform_points(ball_xy) if len(ball_xy) else np.empty((0, 2), dtype=np.float32)
t0_pitch = transformer.transform_points(team0_img) if len(team0_img) else np.empty((0, 2), dtype=np.float32)
t1_pitch = transformer.transform_points(team1_img) if len(team1_img) else np.empty((0, 2), dtype=np.float32)
refs_pitch = transformer.transform_points(refs_xy) if len(refs_xy) else np.empty((0, 2), dtype=np.float32)

# ---- DRAW BIRD-EYE ----
pitch_img = draw_pitch(pitch_cfg)

if len(ball_pitch):
    pitch_img = draw_points_on_pitch(
        config=pitch_cfg, xy=ball_pitch,
        face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=pitch_img
    )

if len(t0_pitch):
    pitch_img = draw_points_on_pitch(
        config=pitch_cfg, xy=t0_pitch,
        face_color=sv.Color.from_hex("00BFFF"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img
    )

if len(t1_pitch):
    pitch_img = draw_points_on_pitch(
        config=pitch_cfg, xy=t1_pitch,
        face_color=sv.Color.from_hex("FF1493"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img
    )

if len(refs_pitch):
    pitch_img = draw_points_on_pitch(
        config=pitch_cfg, xy=refs_pitch,
        face_color=sv.Color.from_hex("FFD700"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img
    )

out_path = f"outputs/debug/birdeye_team_frame_{FRAME_IDX}.jpg"
pitch_img = np.ascontiguousarray(pitch_img, dtype=np.uint8)
cv2.imwrite(out_path, cv2.cvtColor(pitch_img, cv2.COLOR_RGB2BGR))
print("Saved:", out_path)