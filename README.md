# Football Detection, Tracking & Team Classification (Third Year Project)

This repository contains my Third Year Project pipeline for analysing broadcast football videos.

Current capabilities:
- **v1 — Detection + Tracking:** Detect football objects (players, goalkeepers, referees, ball) with YOLO and track them across frames using Ultralytics tracking (ByteTrack / BoT-SORT) + Supervision for annotation.
- **v2 — Team Classification (Unsupervised):** Assign players to **Team 0 / Team 1** using **SigLIP embeddings -> UMAP -> KMeans** (via Roboflow’s `sports` TeamClassifier), with **track-level smoothing** and a **goalkeeper-to-team** heuristic.

---

## Repository structure

- `src/`
  - `track_video_supervision.py` — CLI entrypoint (runs the full pipeline)
  - `pipeline/`
    - `run.py` — warmup + main video loop orchestration
    - `detections.py` — converts Ultralytics results → `supervision.Detections`
    - `annotate.py` — annotation helpers (colors/labels)
  - `team/`
    - `cropping.py` — torso crop helper (jersey-focused crops)
    - `assigner.py` — TeamAssigner (fit + predict + smoothing)
    - `gk_resolver.py` — assign goalkeepers to nearest team centroid
- `models/` — YOLO weights (not committed)
- `input_videos/` — input videos (not committed)
- `outputs/` — generated outputs (not committed)
- `artifacts/` — training runs / predictions / dataset metadata (not committed)

---

## Setup

### 1) Create and activate a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

### 2) Install dependencies

```bash
pip install supervision ultralytics tqdm numpy opencv-python umap-learn scikit-learn
pip install "transformers==4.46.3" "accelerate>=0.34.0" "sentencepiece" "protobuf"
```

### 3) Install the sports dependency (TeamClassifier)

This project uses Roboflow’s `sports` library for team classification.

Clone it somewhere on your machine (outside or inside this repo - either is fine), then install it in editable mode:

```bash
pip install -e "<PATH_TO_SPORTS_REPO>"
```

---

## Usage

### Run detection + tracking + team classification (v1 + v2)

Run from the project root:

```bash
python -m src.track_video_supervision \
  --model models/<YOUR_WEIGHTS>.pt \
  --source input_videos/<YOUR_VIDEO>.mp4 \
  --tracker bytetrack.yaml \
  --imgsz 1280 \
  --conf 0.2 \
  --iou 0.5 \
  --team-device cpu \
  --warmup-seconds 12 \
  --warmup-stride 30 \
  --max-warmup-crops 800 \
  --team-smooth 30 \
  --show-id
```

Output:

- `outputs/<video_stem>_team_tracked.mp4`

**Notes**
- **macOS:** run with `--team-device cpu` (CUDA isn’t typically available on macOS).
- **Team classification:** done **unsupervised per video**. The pipeline runs a short **warm-up** to sample player crops and fit **KMeans**, then assigns teams for the rest of the video.

---

## How team classification works (v2)

1. ### Warmup fit (once per video):
  - Sample frames with stride (`--warmup-stride`) for `--warmup-seconds`
  - Run player detection on those frames
  - Crop **upper-body (torso)** regions from player boxes
  - Compute **SigLIP embeddings -> UMAP projection -> KMeans clustering (2 teams)**

2. ### Main video loop:
  - Track objects per frame (ByteTrack / BoT-SORT via Ultralytics)
  - For each detected player, predict team (0/1) from torso crops
  - Apply **track-level smoothing** (`--team-smooth`) to avoid flicker
  - Assign goalkeepers to the nearest team centroid (image-space)
  - Label referees as `Other`

---

## Configuration

### Tracker options

  - `--tracker bytetrack.yaml` (default)
  - `--tracker botsort.yaml`

### Detector class IDs

Defaults are commonly:

  - `0`: ball
  - `1`: goalkeeper
  - `2`: player
  - `3`: referee

If your trained detector uses different IDs, pass:

  - `--ball-id`
  - `--goalkeeper-id`
  - `--player-id`
  - `--referee-id`

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run as a module from project root:

`python -m src.track_video_supervision ...`

Ensure these exist:

`touch src/__init__.py src/pipeline/__init__.py src/team/__init__.py`

### `ModuleNotFoundError: No module named 'sports'`

Install the sports repo:

`pip install -e "<PATH_TO_SPORTS_REPO>"`

### `Failed building wheel for tokenizers`

Use Python 3.11 (recommended for Transformers stack). On macOS:

`brew install python@3.11`

### `Torch not compiled with CUDA enabled`

Use CPU mode:

`--team-device cpu`

---

## Acknowledgements

- Ultralytics YOLO
- Supervision
- Roboflow `sports` (TeamClassifier: SigLIP + UMAP + KMeans)

---