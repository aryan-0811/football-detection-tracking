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

load_dotenv()

# ---- CONFIG ----
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("RF_FIELD_MODEL_ID")

VIDEO_PATH = "input_videos/input.mp4"      # change if needed
DETECTOR_WEIGHTS = "models/object_detection/best.pt"        # change if needed
FRAME_IDX = 300

# detector class ids (your defaults)
BALL_ID = 0
PLAYER_ID = 2

KP_CONF = 0.5
MIN_POINTS = 6

assert API_KEY and MODEL_ID, "Missing ROBOFLOW_API_KEY or RF_FIELD_MODEL_ID"

# ---- LOAD MODELS ----
field_model = get_model(model_id=MODEL_ID, api_key=API_KEY)
det_model = YOLO(DETECTOR_WEIGHTS)

# ---- READ FRAME ----
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ok, frame = cap.read()
cap.release()
assert ok, f"Could not read frame {FRAME_IDX}"

# ---- PITCH TRANSFORM ----
result = field_model.infer(frame, confidence=0.3)[0]
kps = sv.KeyPoints.from_inference(result)

conf = kps.confidence[0]
mask = conf > KP_CONF

src = kps.xy[0][mask].astype(np.float32)
cfg = SoccerPitchConfiguration()
dst = np.array(cfg.vertices, dtype=np.float32)[mask]

print("Confident pitch points:", int(mask.sum()))
assert src.shape[0] >= MIN_POINTS, "Not enough pitch keypoints for transform."

transformer = ViewTransformer(source=src, target=dst)

# ---- DETECT OBJECTS ----
r = det_model.predict(frame, imgsz=1280, conf=0.2, iou=0.5, verbose=False)[0]
dets = sv.Detections.from_ultralytics(r) if hasattr(sv.Detections, "from_ultralytics") else sv.Detections.from_inference(r)

ball = dets[dets.class_id == BALL_ID]
players = dets[dets.class_id == PLAYER_ID]

# ---- ANCHORS -> PROJECT ----
ball_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(ball) else np.empty((0,2), dtype=np.float32)
players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(players) else np.empty((0,2), dtype=np.float32)

ball_pitch = transformer.transform_points(ball_xy) if len(ball_xy) else np.empty((0,2), dtype=np.float32)
players_pitch = transformer.transform_points(players_xy) if len(players_xy) else np.empty((0,2), dtype=np.float32)

# ---- DRAW BIRD-EYE ----
pitch_img = draw_pitch(cfg)

if len(ball_pitch):
    pitch_img = draw_points_on_pitch(
        config=cfg, xy=ball_pitch,
        face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=pitch_img
    )

if len(players_pitch):
    pitch_img = draw_points_on_pitch(
        config=cfg, xy=players_pitch,
        face_color=sv.Color.from_hex("00BFFF"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img
    )

os.makedirs("outputs/debug", exist_ok=True)
out_path = f"outputs/debug/birdeye_objects_frame_{FRAME_IDX}.jpg"
pitch_img = np.ascontiguousarray(pitch_img, dtype=np.uint8)
cv2.imwrite(out_path, cv2.cvtColor(pitch_img, cv2.COLOR_RGB2BGR))
print("Saved:", out_path)