# Football Detection, Tracking & Analysis (Third Year Project)

A computer vision pipeline for analysing broadcast football videos. It detects players, goalkeepers, referees, and the ball, tracks them across frames, classifies teams unsupervised, projects positions onto a 2D pitch, and generates per-player heatmaps.

## Capabilities

- **v1 — Detection + Tracking:** YOLO object detection with ByteTrack / BoT-SORT multi-object tracking, annotated via Supervision.
- **v2 — Team Classification (Unsupervised):** Assigns players to Team 0 / Team 1 using **SigLIP embeddings -> UMAP -> KMeans** (via Roboflow's `sports` TeamClassifier), with track-level majority-vote smoothing and a goalkeeper-to-team heuristic.
- **v3 — Bird-eye View:** Projects player positions onto a 2D pitch diagram using Roboflow-hosted pitch keypoint detection + ViewTransformer homography, with stride-based caching. Outputs a radar video and an optional side-by-side composite.
- **v4 — Ball Detection:** Dedicated YOLO ball-only model with temporal smoothing (closest-center selection + max-jump filtering) for robust ball tracking independent of the main detector.

---

## Repository structure

```
src/
  track_video_supervision.py  — CLI entrypoint, parses args and calls run_pipeline()
  pipeline/
    run.py        — main orchestration: warmup phase -> frame-by-frame tracking loop -> output writing
    detections.py — converts Ultralytics results -> supervision.Detections
    annotate.py   — colour palette and label formatting
  team/
    assigner.py   — TeamAssigner: warmup fitting (SigLIP -> UMAP -> KMeans) + per-frame prediction + track smoothing
    cropping.py   — torso crop extraction (jersey-focused upper body)
    gk_resolver.py — goalkeeper-to-team assignment via nearest centroid distance
  ball/
    detector.py   — dedicated YOLO ball model with temporal smoothing
  pitch/
    roboflow_pitch.py — Roboflow keypoint inference (stride-based, cached), builds ViewTransformer
    birdeye.py        — bird-eye radar view rendering with team-coloured markers
  heatmap/
    player_heatmap.py — per-track heatmap accumulation in pitch-space, saves PNG + NPY
model_training/       — Jupyter notebooks and scripts used to train the YOLO models
scripts/
  run.sh              — convenience wrapper with mode presets (preview / full / debug)
  debug/              — standalone test scripts for individual components
models/               — YOLO weights (not committed; see Setup step 5)
input_videos/         — input videos (not committed)
outputs/              — generated outputs (not committed)
```

---

## Setup

Requires **Python 3.11** (for HuggingFace Transformers compatibility).

### 1) Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install the Roboflow sports library (TeamClassifier)

This project uses Roboflow's `sports` library for team classification. It must be installed from source:

```bash
git clone https://github.com/roboflow/sports.git
pip install -e ./sports
```

### 4) Environment variables

Copy the example file and fill in your Roboflow credentials:

```bash
cp .env.example .env
# Edit .env and set ROBOFLOW_API_KEY and RF_FIELD_MODEL_ID
```

These are loaded automatically via `python-dotenv`. Required for bird-eye view (v3) — the pitch keypoint detection model is hosted on Roboflow.

### 5) Model weights

Trained YOLO model weights (`.pt` files) are **not included** in this repository. You must either:
- Train your own models using the notebooks in `model_training/`
- Obtain pre-trained weights separately

Place weights in:
- `models/object_detection/best.pt` — main player/goalkeeper/referee detector
- `models/ball_detection/best.pt` — dedicated ball-only detector (v4)

### 6) Input video

Place a broadcast football video in `input_videos/`. The pipeline expects standard broadcast camera angles — it will not work well on close-up or tactical camera footage.

---

## Usage

All commands must be run from the **project root**.

### Quick start with `scripts/run.sh`

The easiest way to run the pipeline. Three modes are available:

```bash
# Preview: 100 frames, all outputs enabled (good for testing)
./scripts/run.sh preview input_videos/input.mp4

# Full: process the entire video with birdeye + side-by-side + heatmaps
./scripts/run.sh full input_videos/input.mp4

# Debug: 300 frames, pitch keypoint overlay only
./scripts/run.sh debug input_videos/input.mp4
```

Default parameters (model paths, thresholds, devices) are configured at the top of `scripts/run.sh` — edit them there to match your setup.

### Direct invocation

For full control over all parameters:

```bash
python -m src.track_video_supervision \
  --model models/object_detection/best.pt \
  --source input_videos/input.mp4 \
  --tracker bytetrack.yaml \
  --imgsz 1280 \
  --conf 0.2 \
  --iou 0.5 \
  --team-device cpu \
  --warmup-seconds 12 \
  --warmup-stride 30 \
  --max-warmup-crops 800 \
  --team-smooth 30 \
  --show-id \
  --ball-model models/ball_detection/best.pt \
  --ball-conf 0.05 \
  --ball-max-jump-px 80 \
  --ball-min-conf 0.25 \
  --birdeye \
  --side-by-side \
  --pitch-stride 15 \
  --kp-conf 0.5 \
  --save-heatmaps \
  --heatmap-top-n 5 \
  --heatmap-min-samples 40 \
  --max-frames 200
```

**Platform note:** On macOS, always use `--team-device cpu` (CUDA is not available). On machines with an NVIDIA GPU, use `--team-device cuda`.

---

## Outputs

All outputs are written to `outputs/` (gitignored). For input video `input.mp4`, the pipeline produces:

| Output | Flag | Description |
|--------|------|-------------|
| `input_team_tracked.mp4` | *(always)* | Main annotated video with bounding boxes, team labels, track IDs, and ball markers |
| `input_birdeye.mp4` | `--birdeye` | Bird-eye radar view showing player positions projected onto a 2D pitch |
| `input_side_by_side.mp4` | `--side-by-side` | Composite video: main annotated frame on top, radar view below |
| `input_pitch_debug.mp4` | `--pitch-debug` | Debug video with Roboflow pitch keypoints overlaid on raw frames |
| `input_heatmaps/` | `--save-heatmaps` | Per-player heatmap PNGs (and raw `.npy` arrays) showing pitch-space activity |

When `--max-frames` is set, output filenames include a `_preview` suffix.

---

## Pipeline overview

The pipeline runs in two phases:

### 1. Warmup phase (runs once per video)

Samples frames across the first N seconds of video (`--warmup-seconds`), detects players, crops their upper-body (torso/jersey) regions, then fits an unsupervised team classifier:

**SigLIP image embeddings -> UMAP dimensionality reduction -> KMeans clustering (k=2)**

This determines which players belong to which team for the rest of the video.

### 2. Main processing loop (frame-by-frame)

```
Frame
 -> YOLO detection + ByteTrack/BoT-SORT tracking
 -> Split detections by class (players, goalkeepers, referees)
 -> Predict team (0/1) for each player using fitted classifier
 -> Smooth predictions via track-level majority vote (--team-smooth window)
 -> Resolve goalkeeper teams by nearest-centroid distance to player groups
 -> (Optional) Dedicated ball model detection with temporal coherence filtering
 -> (Optional) Pitch keypoint inference (every --pitch-stride frames, cached between)
 -> (Optional) Bird-eye projection via ViewTransformer homography
 -> (Optional) Heatmap accumulation in pitch-space
 -> Write annotated frame(s) to output video(s)
```

---

## Configuration

### Tracker options

- `--tracker bytetrack.yaml` (default)
- `--tracker botsort.yaml`

### YOLO class IDs

The default class IDs match the custom-trained model:

| ID | Class |
|----|-------|
| 0 | ball |
| 1 | goalkeeper |
| 2 | player |
| 3 | referee |

Override with `--ball-id`, `--goalkeeper-id`, `--player-id`, `--referee-id` if your model uses different IDs.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run as a module from the project root (not as a script path):

```bash
python -m src.track_video_supervision ...
```

### `ModuleNotFoundError: No module named 'sports'`

Install the Roboflow sports library from source (see Setup step 3).

### `Failed building wheel for tokenizers`

Use Python 3.11:

```bash
brew install python@3.11   # macOS
```

### `Torch not compiled with CUDA enabled`

Use CPU mode: `--team-device cpu`

---

## Acknowledgements & Attribution

### Core Frameworks
- **Ultralytics YOLO** (AGPL-3.0) — object detection and tracking infrastructure
- **Supervision** (MIT) — detection annotation and video processing utilities
- **Roboflow `sports` library** — team classification pipeline (SigLIP embeddings -> UMAP -> KMeans)
- **Roboflow Inference SDK** — hosted pitch keypoint detection

### ML Dependencies
- **HuggingFace Transformers** (Apache 2.0) — SigLIP vision model for image embeddings
- **UMAP** (BSD-3) — dimensionality reduction for embedding clustering
- **scikit-learn** (BSD-3) — KMeans clustering
- **OpenCV** (Apache 2.0) — video I/O and image processing
- **NumPy** (BSD-3) — numerical operations

### Tracking Algorithms
- **ByteTrack** (Zhang et al., 2022) — via Ultralytics integration
- **BoT-SORT** (Ahmetoglu et al., 2023) — via Ultralytics integration

### Datasets
- **football-updated-2024** (Roboflow, CC BY 4.0) — object detection training data
- **football-field-detection** (Roboflow, CC BY 4.0) — pitch keypoint training data
- **ball_detection** (Roboflow, CC BY 4.0) — ball-only detection training data
