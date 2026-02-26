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
VIDEO_PATH = "input_videos/input.mp4"
DETECTOR_WEIGHTS = "models/object_detection/best.pt"
FRAME_IDX = 300

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
RF_FIELD_MODEL_ID = os.getenv("RF_FIELD_MODEL_ID")

PITCH_INFER_CONF = 0.3
KP_CONF = 0.5
MIN_KP = 6

YOLO_IMGSZ = 1280
YOLO_CONF = 0.2
YOLO_IOU = 0.5

TEAM_DEVICE = "cpu"
WARMUP_SECONDS = 10.0
WARMUP_STRIDE = 30
MAX_WARMUP_CROPS = 800
TEAM_SMOOTH = 30

os.makedirs("outputs/debug", exist_ok=True)

assert ROBOFLOW_API_KEY, "Missing ROBOFLOW_API_KEY in .env"
assert RF_FIELD_MODEL_ID, "Missing RF_FIELD_MODEL_ID in .env"

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
src = kps.xy[0][mask].astype(np.float32)
dst = pitch_vertices[mask].astype(np.float32)
print("Pitch keypoints used:", int(mask.sum()))
assert src.shape[0] >= MIN_KP, f"Not enough pitch keypoints for transform. Got {src.shape[0]}, need {MIN_KP}"
transformer = ViewTransformer(source=src, target=dst)

# ---- DETECT OBJECTS ON THIS FRAME ----
r = det_model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
dets = sv.Detections.from_ultralytics(r)

ball = dets[dets.class_id == BALL_ID]
goalkeepers = dets[dets.class_id == GOALKEEPER_ID]
players = dets[dets.class_id == PLAYER_ID]
referees = dets[dets.class_id == REFEREE_ID]

# ---- TEAM CLASSIFY PLAYERS ----
players = assigner.predict_players(frame, players)  # players.class_id becomes 0/1

# ---- ASSIGN GOALKEEPERS TO A TEAM ----
if len(goalkeepers) > 0:
    gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
    goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)

# Referees -> 2 ("Other")
if len(referees) > 0:
    referees.class_id = np.full(len(referees), 2, dtype=int)

# ---- VIDEO FRAME ANNOTATION (v2 style) ----
palette = sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"])  # Team0, Team1, Other
box_annot = sv.BoxAnnotator(color=palette, thickness=2)
label_annot = sv.LabelAnnotator(color=palette, text_color=sv.Color.from_hex("#000000"), text_scale=0.7)
triangle_annot = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)

# optional: overlay pitch keypoints on the video frame too
vertex_annot = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=6)

team_dets = sv.Detections.merge([players, goalkeepers, referees])
team_dets.class_id = team_dets.class_id.astype(int)

labels = []
name = {0: "Team0", 1: "Team1", 2: "Other"}
for i in range(len(team_dets)):
    labels.append(name.get(int(team_dets.class_id[i]), "Other"))

video_annot = frame.copy()
video_annot = box_annot.annotate(scene=video_annot, detections=team_dets)
video_annot = label_annot.annotate(scene=video_annot, detections=team_dets, labels=labels)
video_annot = triangle_annot.annotate(scene=video_annot, detections=ball)
video_annot = vertex_annot.annotate(scene=video_annot, key_points=kps)  # comment out if you don’t want kps on video

# ---- PROJECT ANCHORS + DRAW PITCH ----
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

pitch_img = draw_pitch(pitch_cfg)

if len(ball_pitch):
    pitch_img = draw_points_on_pitch(config=pitch_cfg, xy=ball_pitch,
                                     face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK, radius=10, pitch=pitch_img)
if len(t0_pitch):
    pitch_img = draw_points_on_pitch(config=pitch_cfg, xy=t0_pitch,
                                     face_color=sv.Color.from_hex("00BFFF"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img)
if len(t1_pitch):
    pitch_img = draw_points_on_pitch(config=pitch_cfg, xy=t1_pitch,
                                     face_color=sv.Color.from_hex("FF1493"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img)
if len(refs_pitch):
    pitch_img = draw_points_on_pitch(config=pitch_cfg, xy=refs_pitch,
                                     face_color=sv.Color.from_hex("FFD700"), edge_color=sv.Color.BLACK, radius=14, pitch=pitch_img)

# pitch_img is RGB; convert to BGR for OpenCV
pitch_bgr = cv2.cvtColor(np.ascontiguousarray(pitch_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

# ---- SIDE BY SIDE ----
# resize pitch to match video height
vh, vw = video_annot.shape[:2]
ph, pw = pitch_bgr.shape[:2]
new_w = int(pw * (vh / ph))
pitch_resized = cv2.resize(pitch_bgr, (new_w, vh), interpolation=cv2.INTER_AREA)

# optional divider
divider = np.full((vh, 6, 3), 40, dtype=np.uint8)

side_by_side = np.hstack([video_annot, divider, pitch_resized])

out_path = f"outputs/debug/side_by_side_frame_{FRAME_IDX}.jpg"
cv2.imwrite(out_path, side_by_side)
print("Saved:", out_path)