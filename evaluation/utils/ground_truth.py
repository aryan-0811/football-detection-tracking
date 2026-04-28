"""
Ground-truth loaders.

The harness expects ground truth under data/ground_truth/<clip_name>/. Each
loader takes a clip name, resolves the expected path under config.GT_DIR, and
returns a typed structure or raises FileNotFoundError with an actionable
message.

Expected GT layout (per clip):

    ground_truth/<clip>/
        tracks_mot.txt              # MOT-format
        team_labels.json            # {track_id: team_id}
        passes.json                 # [{frame, passer_id, receiver_id, team_id}]
        offsides.json               # [{frame, offside_track_ids}]
        homography_keypoints.json   # [{frame, image_xy: [x,y], pitch_xy: [x,y]}]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation import config
from evaluation.utils.io_helpers import read_json, read_mot


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def gt_dir(clip_name: str) -> Path:
    """Return the GT directory for a clip (no existence check)."""
    return config.GT_DIR / clip_name


def pipeline_out_dir(clip_name: str) -> Path:
    return config.PIPELINE_OUT_DIR / clip_name


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def load_gt_tracks(clip_name: str) -> pd.DataFrame:
    """Load hand-labelled MOT tracks. Returns a DataFrame (see io_helpers.read_mot)."""
    p = gt_dir(clip_name) / "tracks_mot.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"GT tracks missing for clip '{clip_name}': expected {p}. "
            f"Provide MOT-format hand labels."
        )
    return read_mot(p)


def load_predicted_tracks(clip_name: str, tracker: str) -> pd.DataFrame:
    """
    Load predicted MOT tracks for a given tracker variant.

    Filename convention: pipeline_outputs/<clip>/tracks_<tracker>.txt
    """
    p = pipeline_out_dir(clip_name) / f"tracks_{tracker}.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"Predicted tracks missing for clip '{clip_name}' tracker '{tracker}': "
            f"expected {p}. See INSTRUMENTATION_TODO.md item (2)."
        )
    return read_mot(p)


# ---------------------------------------------------------------------------
# Team labels
# ---------------------------------------------------------------------------

@dataclass
class TeamLabels:
    """Mapping from (string) track_id to team_id."""
    by_track: dict[int, int]


def load_gt_team_labels(clip_name: str) -> TeamLabels:
    p = gt_dir(clip_name) / "team_labels.json"
    if not p.exists():
        raise FileNotFoundError(
            f"GT team labels missing for clip '{clip_name}': expected {p}."
        )
    raw = read_json(p)
    if not isinstance(raw, dict):
        raise ValueError(f"team_labels.json must be a JSON object: {p}")
    return TeamLabels(by_track={int(k): int(v) for k, v in raw.items()})


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

@dataclass
class PassEvent:
    frame: int
    passer_id: int
    receiver_id: int
    team_id: int | None = None


def load_gt_passes(clip_name: str) -> list[PassEvent]:
    p = gt_dir(clip_name) / "passes.json"
    if not p.exists():
        raise FileNotFoundError(
            f"GT passes missing for clip '{clip_name}': expected {p}."
        )
    raw = read_json(p)
    if not isinstance(raw, list):
        raise ValueError(f"passes.json must be a JSON list: {p}")
    return [
        PassEvent(
            frame=int(item["frame"]),
            passer_id=int(item["passer_id"]),
            receiver_id=int(item["receiver_id"]),
            team_id=int(item["team_id"]) if "team_id" in item else None,
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Offsides
# ---------------------------------------------------------------------------

@dataclass
class OffsideEvent:
    frame: int
    offside_track_ids: list[int]


def load_gt_offsides(clip_name: str) -> list[OffsideEvent]:
    p = gt_dir(clip_name) / "offsides.json"
    if not p.exists():
        raise FileNotFoundError(
            f"GT offsides missing for clip '{clip_name}': expected {p}."
        )
    raw = read_json(p)
    if not isinstance(raw, list):
        raise ValueError(f"offsides.json must be a JSON list: {p}")
    return [
        OffsideEvent(
            frame=int(item["frame"]),
            offside_track_ids=[int(x) for x in item.get("offside_track_ids", [])],
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Homography keypoints
# ---------------------------------------------------------------------------

@dataclass
class HomographyKeypoint:
    frame: int
    image_xy: tuple[float, float]
    pitch_xy: tuple[float, float]
    label: str | None = None


def load_gt_homography_keypoints(clip_name: str) -> list[HomographyKeypoint]:
    p = gt_dir(clip_name) / "homography_keypoints.json"
    if not p.exists():
        raise FileNotFoundError(
            f"GT homography keypoints missing for clip '{clip_name}': expected {p}."
        )
    raw = read_json(p)
    if not isinstance(raw, list):
        raise ValueError(f"homography_keypoints.json must be a JSON list: {p}")
    out: list[HomographyKeypoint] = []
    for item in raw:
        ix, iy = item["image_xy"]
        px, py = item["pitch_xy"]
        out.append(HomographyKeypoint(
            frame=int(item["frame"]),
            image_xy=(float(ix), float(iy)),
            pitch_xy=(float(px), float(py)),
            label=item.get("label"),
        ))
    return out


# ---------------------------------------------------------------------------
# Pipeline outputs
# ---------------------------------------------------------------------------

def load_offside_events(clip_name: str) -> list[dict[str, Any]]:
    """Load the pipeline's <clip>_offside_events.json output."""
    p = pipeline_out_dir(clip_name) / f"{clip_name}_offside_events.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Pipeline offside events missing: expected {p}."
        )
    raw = read_json(p)
    if not isinstance(raw, list):
        raise ValueError(f"{p} should contain a JSON list of events")
    return raw


def load_stats(clip_name: str) -> dict[str, Any]:
    """Load the pipeline's <clip>_stats.json output."""
    p = pipeline_out_dir(clip_name) / f"{clip_name}_stats.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Pipeline stats missing: expected {p}."
        )
    raw = read_json(p)
    if not isinstance(raw, dict):
        raise ValueError(f"{p} should contain a JSON object")
    return raw


def load_timings(clip_name: str) -> pd.DataFrame:
    """Load per-stage timings CSV. See INSTRUMENTATION_TODO.md item (1)."""
    p = pipeline_out_dir(clip_name) / "timings.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Per-stage timings missing for '{clip_name}': expected {p}. "
            f"See INSTRUMENTATION_TODO.md item (1)."
        )
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Generic clip-presence check
# ---------------------------------------------------------------------------

def gt_available(clip_name: str, files: list[str]) -> tuple[bool, list[str]]:
    """
    Return (all_present, missing_list) for given relative GT filenames.

    Used by scripts to skip a clip with a clear log line rather than crash.
    """
    missing: list[str] = []
    base = gt_dir(clip_name)
    for fn in files:
        if not (base / fn).exists():
            missing.append(str(base / fn))
    return (len(missing) == 0, missing)
