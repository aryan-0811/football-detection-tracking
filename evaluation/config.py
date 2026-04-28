"""
Centralised configuration for the evaluation harness.

All paths, clip definitions, success-criterion thresholds, hyper-parameter
mirrors and table-formatting options live here so that individual scripts
remain thin. Edit this file (rather than the scripts) when adding clips or
adjusting thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve project root relative to this file so the harness is location-agnostic.
EVAL_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = EVAL_DIR.parent

DATA_DIR: Path = EVAL_DIR / "data"
CLIPS_DIR: Path = DATA_DIR / "clips"
GT_DIR: Path = DATA_DIR / "ground_truth"
PIPELINE_OUT_DIR: Path = DATA_DIR / "pipeline_outputs"

RESULTS_DIR: Path = EVAL_DIR / "results"
TABLES_DIR: Path = RESULTS_DIR / "tables"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
RAW_DIR: Path = RESULTS_DIR / "raw"
LOGS_DIR: Path = RESULTS_DIR / "logs"

# Optional pre-computed Ultralytics validation reports.
# If a YAML/JSON report exists here for a given detector, 01_detection_metrics
# will read it directly instead of re-running validation.
DETECTION_REPORTS_DIR: Path = DATA_DIR / "detection_reports"


# ---------------------------------------------------------------------------
# Clip registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClipDef:
    """A single evaluation clip."""

    name: str
    source: Path
    fps: float
    duration_s: float
    notes: str = ""


# Edit this list when adding clips. The `name` field is used as the directory
# name under data/ground_truth/<name>/ and data/pipeline_outputs/<name>/.
CLIPS: list[ClipDef] = [
    ClipDef(
        name="clip_01_premier_league",
        source=CLIPS_DIR / "clip_01_premier_league.mp4",
        fps=25.0,
        duration_s=30.0,
        notes="Placeholder. Replace with real clip definitions.",
    ),
]


def get_clip(name: str) -> ClipDef:
    """Return a ClipDef by name; raise KeyError if not registered."""
    for clip in CLIPS:
        if clip.name == name:
            return clip
    raise KeyError(f"Clip '{name}' not registered in config.CLIPS")


# ---------------------------------------------------------------------------
# Success-criterion thresholds (from the project specification)
# ---------------------------------------------------------------------------

METRIC_THRESHOLDS: dict[str, float] = {
    "detection_mAP50": 0.80,   # criterion (i)
    "tracking_HOTA": 0.45,     # criterion (ii)
    "team_F1": 0.85,           # criterion (iii)
    "pass_F1": 0.60,           # criterion (iv) — pass detection
}


# ---------------------------------------------------------------------------
# Pipeline hyper-parameters mirrored from scripts/run.sh
# ---------------------------------------------------------------------------
# These are reported alongside results so each table is self-describing. Update
# whenever run.sh defaults change.
PIPELINE_HYPERPARAMS: dict[str, Any] = {
    "detector_model": "models/object_detection/best.pt",
    "ball_model": "models/ball_detection/best.pt",
    "tracker_default": "bytetrack",
    "team_warmup_seconds": 8,
    "team_device": "cpu",
    "team_smoothing_window": 30,
    "speed_cap_kmh": 40.0,
    "pitch_keypoint_stride": 15,
    "pitch_dimensions_m": (105.0, 68.0),
    "ball_max_jump_px": 80,
    "yolo_classes": {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3},
}


# ---------------------------------------------------------------------------
# Matching tolerances
# ---------------------------------------------------------------------------

PASS_FRAME_TOLERANCE: int = 10           # +/- frames for pass TP matching
OFFSIDE_FRAME_TOLERANCE: int = 5         # +/- frames for offside event matching
TRACK_IOU_THRESHOLD: float = 0.5         # for ID matching in tracking eval


# ---------------------------------------------------------------------------
# LaTeX table style
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatexTableOptions:
    float_format: str = "%.3f"
    na_rep: str = "--"
    index: bool = False
    bold_rows: bool = False
    column_format: str | None = None
    escape: bool = False


LATEX_TABLE_OPTIONS = LatexTableOptions()


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PLOT_STYLE: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# Reference / comparison numbers from related work
# ---------------------------------------------------------------------------
# These are embedded as reference rows in the comparison tables. Numbers come
# from the cited papers; see project-report/references.bib.
LITERATURE_BASELINES: dict[str, dict[str, Any]] = {
    "tracking_soccernet_2023": {
        "source": "Cioppa et al. 2022 / Magera et al. 2025 (SoccerNet-Tracking)",
        "HOTA": 0.535,
        "MOTA": 0.727,
        "IDF1": 0.616,
    },
    "team_istasse_2019": {
        "source": "Istasse et al. 2019 colour-histogram baseline",
        "F1": 0.78,
    },
    "homography_theiner_2022": {
        "source": "Theiner & Eggert 2022",
        "mean_error_m": 1.12,
    },
    "homography_tvcalib_2023": {
        "source": "Theiner & Eggert 2023 (TVCalib)",
        "mean_error_m": 0.87,
    },
}


def ensure_results_tree() -> None:
    """Create the results subdirectories if missing. Idempotent."""
    for d in (TABLES_DIR, FIGURES_DIR, RAW_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
