"""
Shared metric primitives.

These are intentionally framework-agnostic and unit-tested in
evaluation/tests/. Heavyweight metrics (HOTA) live in their dedicated scripts
and use external libraries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Intersection-over-union for two boxes given in (left, top, w, h) format.
    Returns 0 if either box is degenerate.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between two sets of xywh boxes. Returns (Na, Nb).
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=float)
    a = boxes_a.astype(float)
    b = boxes_b.astype(float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    ix1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    iy1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=union > 0)
    return out


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf1(tp: int, fp: int, fn: int) -> PRF1:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return PRF1(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)


def macro_f1(per_class: Iterable[PRF1]) -> float:
    items = list(per_class)
    if not items:
        return 0.0
    return float(np.mean([x.f1 for x in items]))


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Return (matrix, ordered_labels). Rows = true, cols = predicted."""
    if labels is None:
        labels = sorted({*y_true, *y_pred})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            m[label_to_idx[t], label_to_idx[p]] += 1
    return m, list(labels)


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------

def reprojection_errors_px(
    image_xy: np.ndarray,
    pitch_xy: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Project image points through homography H into pitch space and return
    Euclidean errors in pitch units (input units of pitch_xy).
    """
    if image_xy.shape[0] == 0:
        return np.array([], dtype=float)
    pts = np.concatenate([image_xy, np.ones((image_xy.shape[0], 1))], axis=1)
    proj = (H @ pts.T).T  # (N,3)
    w = proj[:, 2:3]
    proj_xy = proj[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)
    diffs = proj_xy - pitch_xy
    return np.linalg.norm(diffs, axis=1)


def summary_stats(values: np.ndarray) -> dict[str, float]:
    """Common summary stats for a 1-D array. Empty input → all NaN."""
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "n": int(values.size),
    }


# ---------------------------------------------------------------------------
# Pass-event matching
# ---------------------------------------------------------------------------

def match_events_with_tolerance(
    gt: list[tuple[int, int, int]],
    pred: list[tuple[int, int, int]],
    *,
    frame_tol: int,
) -> tuple[int, int, int, list[tuple[int, int]]]:
    """
    Match (frame, passer_id, receiver_id) tuples between gt and pred.

    A predicted event matches a GT event if same passer & receiver IDs and
    |frame_pred - frame_gt| <= frame_tol. One-to-one matching, greedy on
    smallest frame distance.

    Returns (TP, FP, FN, list_of_matches as (gt_idx, pred_idx)).
    """
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    # Build candidate pairs sorted by |df|.
    candidates = []
    for gi, (gf, gp, gr) in enumerate(gt):
        for pi, (pf, pp, pr) in enumerate(pred):
            if gp != pp or gr != pr:
                continue
            df = abs(gf - pf)
            if df <= frame_tol:
                candidates.append((df, gi, pi))
    candidates.sort()

    for _, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gi, pi))

    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, matches


# ---------------------------------------------------------------------------
# Track-label smoothing
# ---------------------------------------------------------------------------

def smooth_labels_majority(labels: Sequence[int], window: int) -> list[int]:
    """
    Centred majority-vote smoother. window must be >= 1; ties prefer the
    incoming label. Used to evaluate the smoothing-vs-raw ablation in
    03_team_classification_metrics.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    out: list[int] = []
    n = len(labels)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chunk = labels[lo:hi]
        vals, counts = np.unique(chunk, return_counts=True)
        out.append(int(vals[int(np.argmax(counts))]))
    return out


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str],
                       value_cols: list[str]) -> pd.DataFrame:
    """Group and report mean ± std per value column."""
    g = df.groupby(group_cols)[value_cols]
    mean = g.mean().add_suffix("_mean")
    std = g.std().add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()
