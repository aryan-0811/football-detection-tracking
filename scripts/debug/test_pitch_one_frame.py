import os
import cv2
import numpy as np
import supervision as sv
from sports.common.view import ViewTransformer
from dotenv import load_dotenv
from inference import get_model
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

load_dotenv()  # loads .env from repo root

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("RF_FIELD_MODEL_ID")  # e.g. football-field-detection-f07vi-hwn6o/2
VIDEO_PATH = "input_videos/input.mp4"      # change if needed
FRAME_IDX = 300                            # try 0, 300, 600, ...

assert API_KEY, "ROBOFLOW_API_KEY not found in .env"
assert MODEL_ID, "RF_FIELD_MODEL_ID not found in .env"

model = get_model(model_id=MODEL_ID, api_key=API_KEY)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
ok, frame = cap.read()
cap.release()
assert ok, f"Could not read frame {FRAME_IDX} from {VIDEO_PATH}"

result = model.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(
    xy=frame_reference_points[np.newaxis, ...])

# draw keypoints on the real frame and save
vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=6)
annotated = vertex_annotator.annotate(scene=frame.copy(), key_points=frame_reference_key_points)

os.makedirs("outputs/debug", exist_ok=True)
out_path = f"outputs/debug/roboflow_pitch_kps_frame_{FRAME_IDX}.jpg"
cv2.imwrite(out_path, annotated)
print("Saved:", out_path)

CONFIG = SoccerPitchConfiguration()

annotated_frame = draw_pitch(CONFIG)
out_path = f"outputs/debug/birdeye_pitch_kps_frame_{FRAME_IDX}.jpg"
cv2.imwrite(out_path, annotated_frame)


edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#00BFFF'),
    thickness=2, edges=CONFIG.edges)
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    radius=8)
vertex_annotator_2 = sv.VertexAnnotator(
    color=sv.Color.from_hex('#00BFFF'),
    radius=8)

frame_generator = sv.get_video_frames_generator(VIDEO_PATH, start=200)
frame = next(frame_generator)

result = model.infer(frame, confidence=0.3)[0]
key_points = sv.KeyPoints.from_inference(result)

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(
    xy=frame_reference_points[np.newaxis, ...])

pitch_reference_points = np.array(CONFIG.vertices)[filter]

transformer = ViewTransformer(
    source=pitch_reference_points,
    target=frame_reference_points
)

pitch_all_points = np.array(CONFIG.vertices)
frame_all_points = transformer.transform_points(points=pitch_all_points)

frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = edge_annotator.annotate(
    scene=annotated_frame,
    key_points=frame_all_key_points)
annotated_frame = vertex_annotator_2.annotate(
    scene=annotated_frame,
    key_points=frame_all_key_points)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=frame_reference_key_points)

# sv.plot_image(annotated_frame)
