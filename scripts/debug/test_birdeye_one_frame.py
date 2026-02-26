import os
import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("RF_FIELD_MODEL_ID")
VIDEO_PATH = "input_videos/input.mp4"   # change if needed
FRAME_IDX = 300                         # try different frames if needed

KP_CONF = 0.5
MIN_POINTS = 6

assert API_KEY, "ROBOFLOW_API_KEY missing"
assert MODEL_ID, "RF_FIELD_MODEL_ID missing"

# Load Roboflow model
model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# Read one frame
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ok, frame = cap.read()
cap.release()
assert ok, f"Could not read frame {FRAME_IDX}"

# Inference -> KeyPoints (Roboflow format)
result = model.infer(frame, confidence=0.3)[0]
kps = sv.KeyPoints.from_inference(result)

conf = kps.confidence[0]
mask = conf > KP_CONF

src = kps.xy[0][mask].astype(np.float32)  # image points
cfg = SoccerPitchConfiguration()
dst = np.array(cfg.vertices, dtype=np.float32)[mask]  # pitch template points

print("Confident points:", int(mask.sum()))

os.makedirs("outputs/debug", exist_ok=True)

# Draw pitch
pitch_img = draw_pitch(cfg)

if src.shape[0] >= MIN_POINTS:
    transformer = ViewTransformer(source=src, target=dst)
    proj = transformer.transform_points(points=src)

    pitch_img = draw_points_on_pitch(
        config=cfg,
        xy=proj,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=6,
        pitch=pitch_img
    )
else:
    print(f"Not enough points for transform (need {MIN_POINTS}).")

out_path = f"outputs/debug/birdeye_frame_{FRAME_IDX}.jpg"
pitch_img = np.ascontiguousarray(pitch_img, dtype=np.uint8)
cv2.imwrite(out_path, cv2.cvtColor(pitch_img, cv2.COLOR_RGB2BGR))
print("Saved:", out_path)