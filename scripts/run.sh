#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run.sh                         # preview on default input
#   ./scripts/run.sh input_videos/x.mp4      # preview on provided input
#   ./scripts/run.sh full input_videos/x.mp4 # full run
#   ./scripts/run.sh debug input_videos/x.mp4# pitch debug only

MODE="preview"
SOURCE="input_videos/input.mp4"   # default input video path

# Parse args: allow either [mode] [source] or [source]
if [[ $# -ge 1 ]]; then
  if [[ "$1" == "preview" || "$1" == "full" || "$1" == "debug" ]]; then
    MODE="$1"
    [[ $# -ge 2 ]] && SOURCE="$2"
  else
    SOURCE="$1"
  fi
fi

# ---------- Defaults (edit once) ----------
MODEL="models/object_detection/best.pt"   # YOLO Object Detection weights path
TRACKER="bytetrack.yaml"                 # tracker config (bytetrack.yaml or botsort.yaml)

IMGSZ=1280     # YOLO inference image size (larger = better detection, slower) for object detection
CONF=0.2       # YOLO confidence threshold (higher = fewer detections)
IOU=0.5        # YOLO NMS IoU threshold (higher = keeps more overlapping boxes)

TEAM_DEVICE="cpu"     # team classifier device: cpu on macOS; cuda on NVIDIA
WARMUP_SECONDS=20     # seconds of video used to collect player crops for clustering
WARMUP_STRIDE=30      # sample every N frames during warmup (higher = fewer crops)
MAX_WARMUP_CROPS=800  # cap number of crops used to fit team classifier
TEAM_SMOOTH=30        # smoothing window over track IDs (majority vote length)

PITCH_STRIDE=15       # run Roboflow pitch keypoints every N frames (caches in between)
KP_CONF=0.5           # min keypoint confidence to accept/use for transformer

SHOW_ID=true          # show tracker ID labels on players (true/false)
FONT_SCALE=0.7        # label text size
BOX_THICKNESS=2       # bounding box thickness
# ----------------------------------------

# Mode presets (you generally don’t edit these; tweak defaults above instead)
MAX_FRAMES=""         # stop after N frames (preview/testing); empty = full video
PITCH_DEBUG=false     # output pitch keypoint overlay debug video
BIRDEYE=false         # output bird-eye (radar) video
SIDE_BY_SIDE=false    # output combined video (main + radar)

case "$MODE" in
  preview)
    MAX_FRAMES=300        # process only first 300 frames
    PITCH_DEBUG=true      # write pitch debug overlay video
    BIRDEYE=true          # write radar video
    SIDE_BY_SIDE=true     # write combined video
    ;;
  full)
    MAX_FRAMES=""         # no frame cap
    PITCH_DEBUG=false     # usually off (saves time/output)
    BIRDEYE=true
    SIDE_BY_SIDE=true
    ;;
  debug)
    MAX_FRAMES=300
    PITCH_DEBUG=true
    BIRDEYE=false
    SIDE_BY_SIDE=false
    ;;
  *)
    echo "Unknown mode: $MODE (use preview|full|debug)"
    exit 1
    ;;
esac

# Build the command
CMD=(python -m src.track_video_supervision
  --model "$MODEL"              # YOLO weights
  --source "$SOURCE"            # input video
  --tracker "$TRACKER"          # bytetrack/botsort config
  --imgsz "$IMGSZ"              # inference size
  --conf "$CONF"                # detection confidence threshold
  --iou "$IOU"                  # NMS IoU threshold

  --team-device "$TEAM_DEVICE"  # team classifier device
  --warmup-seconds "$WARMUP_SECONDS"
  --warmup-stride "$WARMUP_STRIDE"
  --max-warmup-crops "$MAX_WARMUP_CROPS"
  --team-smooth "$TEAM_SMOOTH"

  --pitch-stride "$PITCH_STRIDE"
  --kp-conf "$KP_CONF"

  --font-scale "$FONT_SCALE"
  --box-thickness "$BOX_THICKNESS"
)

# Optional flags
if [[ "$SHOW_ID" == "true" ]]; then
  CMD+=(--show-id)               # include track IDs in labels
fi

if [[ -n "$MAX_FRAMES" ]]; then
  CMD+=(--max-frames "$MAX_FRAMES")  # stop early for preview
fi

if [[ "$PITCH_DEBUG" == "true" ]]; then
  CMD+=(--pitch-debug)           # write pitch debug video
fi
if [[ "$BIRDEYE" == "true" ]]; then
  CMD+=(--birdeye)               # write bird-eye radar video
fi
if [[ "$SIDE_BY_SIDE" == "true" ]]; then
  CMD+=(--side-by-side)          # write combined video
fi

echo "Mode: $MODE"
echo "Source: $SOURCE"
echo "Running:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"