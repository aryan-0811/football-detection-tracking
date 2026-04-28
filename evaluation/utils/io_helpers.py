"""
Defensive I/O helpers.

All functions follow two rules:

1. If a file is missing, raise FileNotFoundError with an actionable message
   (caller decides whether to skip the clip or abort).
2. Never silently return empty data.

Supported formats: JSON, CSV (via pandas), NPY, MOT-Challenge tracks.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    """Read a JSON file. Raises FileNotFoundError or json.JSONDecodeError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write JSON, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default, ensure_ascii=False)
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def maybe_read_json(path: str | Path) -> Any | None:
    """Return parsed JSON or None if the file is missing."""
    p = Path(path)
    if not p.exists():
        return None
    return read_json(p)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parents."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# NPY
# ---------------------------------------------------------------------------

def write_npy(arr: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    return p


def read_npy(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NPY not found: {p}")
    return np.load(p)


# ---------------------------------------------------------------------------
# MOT-Challenge tracks
# ---------------------------------------------------------------------------
# Format (one row per detection per frame):
#   frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
# We use only the first 7 columns. x,y,z are -1 for 2D MOT.

MOT_COLUMNS = [
    "frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
    "conf", "x", "y", "z",
]


def read_mot(path: str | Path) -> pd.DataFrame:
    """
    Load a MOT-format tracks file.

    Returns a DataFrame with columns: frame (int), id (int),
    bb_left, bb_top, bb_width, bb_height (float), conf (float).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MOT tracks file not found: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"MOT tracks file is empty: {p}")

    df = pd.read_csv(p, header=None, names=MOT_COLUMNS[:10],
                     usecols=range(min(10, _csv_num_cols(p))))
    if df.empty:
        raise ValueError(f"MOT tracks file parsed to zero rows: {p}")

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    for col in ("bb_left", "bb_top", "bb_width", "bb_height"):
        df[col] = df[col].astype(float)
    if "conf" in df.columns:
        df["conf"] = df["conf"].astype(float)
    return df


def _csv_num_cols(p: Path) -> int:
    with p.open("r") as f:
        first = f.readline().strip()
    if not first:
        return 0
    return len(first.split(","))


def write_mot(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to MOT format. Required columns: frame, id, bb_left,
    bb_top, bb_width, bb_height. conf defaults to 1.0; x/y/z to -1."""
    required = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"write_mot: missing columns {missing}")

    out = df.copy()
    if "conf" not in out.columns:
        out["conf"] = 1.0
    for c in ("x", "y", "z"):
        if c not in out.columns:
            out[c] = -1

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out[MOT_COLUMNS].to_csv(p, header=False, index=False)
    return p


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def require_files(paths: Iterable[str | Path], *, context: str = "") -> list[Path]:
    """Validate that every path exists. Returns paths as Path objects."""
    resolved = [Path(p) for p in paths]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        ctx = f" [{context}]" if context else ""
        raise FileNotFoundError(
            f"Required input files missing{ctx}: " + ", ".join(missing)
        )
    return resolved


def write_markdown(path: str | Path, body: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single row to a CSV. Creates header if file is new."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
