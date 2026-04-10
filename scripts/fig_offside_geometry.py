#!/usr/bin/env python3
"""
Generate a side-by-side offside geometry figure for the project report.

Left panel (a):  broadcast frame with bounding boxes, offside player highlighted in red.
Right panel (b): bird-eye pitch view with team-coloured positions, offside line, ball,
                 second-last defender, and attacking direction arrow.

Reads an offside event from the pipeline JSON log. Uses single-frame YOLO
prediction (no tracking) and matches offside players by pitch-space proximity
to the positions recorded in the JSON, avoiding the slow multi-frame tracker.

Usage:
    python scripts/fig_offside_geometry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIDEO = ROOT / "input_videos" / "input.mp4"
OBJECT_MODEL = ROOT / "models" / "object_detection" / "best.pt"
BALL_MODEL = ROOT / "models" / "ball_detection" / "best.pt"
OFFSIDE_JSON = ROOT / "outputs" / "input_offside_events.json"
OUTPUT = ROOT / "outputs" / "offside_geometry.png"

# -- load offside event ---------------------------------------------------
with open(OFFSIDE_JSON) as f:
    data = json.load(f)

# Pick frame 243 (~10s, goalkeeper visible in frame)
event = next(e for e in data["offsides"] if e["frame_idx"] == 243)

TARGET_FRAME = event["frame_idx"]
OFFSIDE_LINE_X = event["offside_line_x"]
OFFSIDE_PLAYERS_JSON = event["offside_players"]  # [{track_id, pitch_xy}]
PASSER_TEAM = event["pass_event"]["passer_team_id"]  # 0
BALL_PITCH_XY = np.array(event["pass_event"]["ball_pitch_xy"])

print(f"Target frame: {TARGET_FRAME}, offside line: {OFFSIDE_LINE_X:.0f} cm")

# -- pitch config ----------------------------------------------------------
pitch_config = SoccerPitchConfiguration()
PITCH_LEN = pitch_config.length  # 12000
PITCH_WID = pitch_config.width   # 7000

# -- extract frame ---------------------------------------------------------
cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
ret, frame_bgr = cap.read()
cap.release()
assert ret, f"Could not read frame {TARGET_FRAME}"
print(f"Frame: {frame_bgr.shape}")

# -- single-frame YOLO prediction (fast, no tracker) ----------------------
model = YOLO(str(OBJECT_MODEL))
result = model.predict(source=frame_bgr, imgsz=1280, conf=0.2, iou=0.5, verbose=False)[0]

from src.pipeline.detections import detections_from_ultralytics
dets = detections_from_ultralytics(result)

PLAYER_ID, GK_ID, REF_ID = 2, 1, 3
goalkeepers = dets[dets.class_id == GK_ID]
players = dets[dets.class_id == PLAYER_ID]
referees = dets[dets.class_id == REF_ID]

# -- lightweight team assignment (small warmup) ----------------------------
from src.team.assigner import TeamAssigner, WarmupConfig
from src.team.gk_resolver import resolve_goalkeepers_team_id

assigner = TeamAssigner(device="cpu", smooth_window=1)
warmup = WarmupConfig(seconds=10, stride=30, max_crops=400, conf=0.25, iou=0.5)
n_crops = assigner.fit_from_video(str(VIDEO), model, player_class_id=PLAYER_ID, imgsz=1280, warmup=warmup)
print(f"Team fit: {n_crops} crops")

players = assigner.predict_players(frame_bgr, players)

if len(goalkeepers) > 0:
    gk_team = resolve_goalkeepers_team_id(players, goalkeepers)
    goalkeepers.class_id = np.where(gk_team < 0, 2, gk_team).astype(int)
if len(referees) > 0:
    referees.class_id = np.full(len(referees), 2, dtype=int)

team_dets = sv.Detections.merge([players, goalkeepers, referees])
team_dets.class_id = team_dets.class_id.astype(int)

players_and_gk = sv.Detections.merge([players, goalkeepers])
if len(players_and_gk) > 0:
    players_and_gk.class_id = players_and_gk.class_id.astype(int)

# -- ball detection --------------------------------------------------------
from src.ball.detector import BallDetector

ball_det = BallDetector(model_path=BALL_MODEL, conf=0.05, max_jump_px=80, min_conf=0.25, imgsz=1280)
ball = ball_det.predict(frame_bgr)
print(f"Ball detections: {len(ball)}")

# -- pitch keypoints / ViewTransformer ------------------------------------
from src.pitch.roboflow_pitch import RoboflowPitch, PitchConfig

pitch = RoboflowPitch(PitchConfig(stride=1, kp_conf=0.5), pitch_config=pitch_config)
pitch.load_from_env()
transformer = pitch.maybe_get_transformer(frame_bgr, frame_idx=0)
assert transformer is not None, "Failed to build ViewTransformer"

# -- project to pitch space ------------------------------------------------
pg_xy = players_and_gk.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
pg_pitch = transformer.transform_points(points=pg_xy)

ball_pitch = None
if len(ball) > 0:
    ball_frame_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    ball_pitch = transformer.transform_points(points=ball_frame_xy)[0]

# -- match offside players by pitch-space proximity to JSON positions ------
# Since we didn't run the tracker, we match detections to the JSON offside
# positions using nearest-neighbour in pitch space.
offside_det_indices: list[int] = []
for op in OFFSIDE_PLAYERS_JSON:
    json_xy = np.array(op["pitch_xy"])
    dists = np.linalg.norm(pg_pitch - json_xy[None, :], axis=1)
    nearest = int(np.argmin(dists))
    if dists[nearest] < 500:  # within 5 m tolerance
        offside_det_indices.append(nearest)
        print(f"  Matched offside player (JSON tid {op['track_id']}) → det index {nearest}, dist {dists[nearest]:.0f} cm")

# Map back to team_dets indices for the broadcast frame annotation.
# players_and_gk boxes are a subset of team_dets; find them by xyxy match.
offside_team_det_indices: list[int] = []
for oi in offside_det_indices:
    box = players_and_gk.xyxy[oi]
    for j in range(len(team_dets)):
        if np.allclose(team_dets.xyxy[j], box, atol=1.0):
            offside_team_det_indices.append(j)
            break

# =========================================================================
# PANEL (a): broadcast frame with bounding boxes
# =========================================================================
# Custom palette: Team0=cyan-blue, Team1=pink, Referee=yellow
palette = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
box_annot = sv.EllipseAnnotator(color=palette, thickness=2)
triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFD700"), base=20, height=17)

annotated = box_annot.annotate(scene=frame_bgr.copy(), detections=team_dets)
annotated = triangle_annotator.annotate(scene=annotated, detections=ball)

# Highlight offside players
for j in offside_team_det_indices:
    x1, y1, x2, y2 = team_dets.xyxy[j].astype(int)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
    txt = "OFFSIDE"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    lbl_x = x1 - tw - 6  # place label to the left of the box
    lbl_x = max(0, lbl_x)  # clamp to frame edge
    cv2.rectangle(annotated, (lbl_x, y1 - th - 10), (lbl_x + tw + 4, y1), (0, 0, 255), -1)
    cv2.putText(annotated, txt, (lbl_x + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# =========================================================================
# PANEL (b): bird-eye pitch view
# =========================================================================
pitch_img = draw_pitch(pitch_config)
ph, pw = pitch_img.shape[:2]

def pitch_to_px(x_cm: float, y_cm: float) -> tuple[int, int]:
    return int(x_cm / PITCH_LEN * pw), int(y_cm / PITCH_WID * ph)

# Team colours (BGR): Team0=#00BFFF, Team1=#FF1493, Referee=yellow
TEAM_COLORS = {0: (255, 191, 0), 1: (147, 20, 255), 2: (0, 215, 255)}

# Draw players/GKs
offside_idx_set = set(offside_det_indices)
for i in range(len(players_and_gk)):
    cid = int(players_and_gk.class_id[i])
    x_cm, y_cm = pg_pitch[i]
    px, py = pitch_to_px(x_cm, y_cm)
    if i in offside_idx_set:
        cv2.circle(pitch_img, (px, py), 18, (255, 255, 255), -1)
        cv2.circle(pitch_img, (px, py), 18, (0, 0, 255), 3)
        cv2.circle(pitch_img, (px, py), 15, (0, 0, 255), -1)
    else:
        color = TEAM_COLORS.get(cid, (200, 200, 200))
        cv2.circle(pitch_img, (px, py), 16, (0, 0, 0), -1)
        cv2.circle(pitch_img, (px, py), 14, color, -1)

# Referees
if len(referees) > 0:
    ref_xy = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    ref_pitch = transformer.transform_points(points=ref_xy)
    for i in range(len(referees)):
        px, py = pitch_to_px(ref_pitch[i, 0], ref_pitch[i, 1])
        cv2.circle(pitch_img, (px, py), 16, (0, 0, 0), -1)
        cv2.circle(pitch_img, (px, py), 14, TEAM_COLORS[2], -1)

# Ball — white dot
if ball_pitch is not None:
    bpx, bpy = pitch_to_px(ball_pitch[0], ball_pitch[1])
    cv2.circle(pitch_img, (bpx, bpy), 12, (0, 0, 0), -1)
    cv2.circle(pitch_img, (bpx, bpy), 10, (255, 255, 255), -1)

# Second-last defender (diamond marker)
# Team 0 attacks left (toward x=0): offside_line < halfway confirms this.
def_mask = (players_and_gk.class_id.astype(int) != PASSER_TEAM) & \
           (players_and_gk.class_id.astype(int) != 2)
if def_mask.any():
    def_x = pg_pitch[def_mask, 0]
    def_indices = np.where(def_mask)[0]
    sorted_order = np.argsort(def_x)  # ascending (closest to x=0 goal)
    sl_idx = def_indices[sorted_order[1]] if len(sorted_order) >= 2 else def_indices[sorted_order[0]]
    sl_x, sl_y = pg_pitch[sl_idx]
    sl_px, sl_py = pitch_to_px(sl_x, sl_y)
    d = 22
    diamond = np.array([[sl_px, sl_py - d], [sl_px + d, sl_py],
                         [sl_px, sl_py + d], [sl_px - d, sl_py]], dtype=np.int32)
    cv2.fillPoly(pitch_img, [diamond], (147, 20, 255))  # #FF1493 (BGR) for team 1
    cv2.polylines(pitch_img, [diamond], True, (0, 0, 0), 2)

# Offside line — red dashed vertical
off_px = max(0, min(int(OFFSIDE_LINE_X / PITCH_LEN * pw), pw - 1))
y = 0
while y < ph:
    y_end = min(y + 16, ph)
    cv2.line(pitch_img, (off_px, y), (off_px, y_end), (0, 0, 255), 3)
    y = y_end + 10

# Place label to the left of the line
(tw, _), _ = cv2.getTextSize("offside line", cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
cv2.putText(pitch_img, "offside line", (off_px - tw - 8, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)

# Attacking direction arrow (team 0 attacks left)
arr_y = ph - 40
cv2.arrowedLine(pitch_img, (int(pw * 0.55), arr_y), (int(pw * 0.35), arr_y),
                (0, 0, 0), 3, tipLength=0.15)
cv2.putText(pitch_img, "attacking direction", (int(pw * 0.35) + 30, arr_y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)

pitch_img_rgb = cv2.cvtColor(pitch_img, cv2.COLOR_BGR2RGB)

# =========================================================================
# Compose figure
# =========================================================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 4.5),
                                  gridspec_kw={"width_ratios": [1.1, 1]})

ax_a.imshow(annotated_rgb)
ax_a.set_axis_off()
ax_a.text(0.5, -0.04, "(a)", transform=ax_a.transAxes,
          ha="center", va="top", fontsize=18)

ax_b.imshow(pitch_img_rgb)
ax_b.set_axis_off()
ax_b.text(0.5, -0.04, "(b)", transform=ax_b.transAxes,
          ha="center", va="top", fontsize=18)

plt.tight_layout(pad=0.5)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(OUTPUT), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {OUTPUT}")
