"""Unit tests for metrics_helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.utils import metrics_helpers as M


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def test_iou_identical_boxes() -> None:
    a = (0, 0, 10, 10)
    assert M.iou_xywh(a, a) == pytest.approx(1.0)


def test_iou_disjoint() -> None:
    assert M.iou_xywh((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_half_overlap() -> None:
    # Two 10x10 boxes overlapping 5x10 = 50 area; union = 200 - 50 = 150
    iou = M.iou_xywh((0, 0, 10, 10), (5, 0, 10, 10))
    assert iou == pytest.approx(50 / 150)


def test_iou_degenerate_zero_area() -> None:
    assert M.iou_xywh((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_iou_matrix_shape_and_values() -> None:
    a = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10]], dtype=float)
    m = M.iou_matrix(a, b)
    assert m.shape == (2, 1)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[1, 0] == 0.0


def test_iou_matrix_empty() -> None:
    m = M.iou_matrix(np.zeros((0, 4)), np.zeros((0, 4)))
    assert m.shape == (0, 0)


# ---------------------------------------------------------------------------
# PRF1
# ---------------------------------------------------------------------------

def test_prf1_basic() -> None:
    r = M.prf1(tp=8, fp=2, fn=2)
    assert r.precision == pytest.approx(0.8)
    assert r.recall == pytest.approx(0.8)
    assert r.f1 == pytest.approx(0.8)


def test_prf1_zero_safe() -> None:
    r = M.prf1(tp=0, fp=0, fn=0)
    assert r.precision == 0.0 and r.recall == 0.0 and r.f1 == 0.0


def test_macro_f1_average() -> None:
    a = M.prf1(tp=1, fp=1, fn=1)  # f1 = 0.5
    b = M.prf1(tp=4, fp=0, fn=0)  # f1 = 1.0
    assert M.macro_f1([a, b]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def test_confusion_matrix_simple() -> None:
    m, labels = M.confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1])
    assert labels == [0, 1]
    assert m.tolist() == [[1, 1], [0, 2]]


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

def test_reprojection_identity_homography() -> None:
    H = np.eye(3)
    img = np.array([[10.0, 20.0], [30.0, 40.0]])
    pitch = img.copy()
    err = M.reprojection_errors_px(img, pitch, H)
    assert np.allclose(err, [0.0, 0.0])


def test_reprojection_translation() -> None:
    # H translates image points by (+5, +5); GT pitch is the original image,
    # so per-point error should be sqrt(50).
    H = np.array([[1, 0, 5], [0, 1, 5], [0, 0, 1]], dtype=float)
    img = np.array([[0.0, 0.0], [10.0, 10.0]])
    pitch = img.copy()
    err = M.reprojection_errors_px(img, pitch, H)
    expected = math.sqrt(50)
    assert np.allclose(err, [expected, expected])


def test_summary_stats_empty() -> None:
    s = M.summary_stats(np.array([]))
    assert s["n"] == 0
    assert math.isnan(s["mean"])


def test_summary_stats_basic() -> None:
    s = M.summary_stats(np.array([1, 2, 3, 4]))
    assert s["mean"] == pytest.approx(2.5)
    assert s["median"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------

def test_match_events_exact() -> None:
    gt = [(10, 1, 2), (50, 3, 4)]
    pred = [(11, 1, 2), (50, 3, 4)]
    tp, fp, fn, _ = M.match_events_with_tolerance(gt, pred, frame_tol=5)
    assert (tp, fp, fn) == (2, 0, 0)


def test_match_events_outside_tolerance() -> None:
    gt = [(10, 1, 2)]
    pred = [(30, 1, 2)]
    tp, fp, fn, _ = M.match_events_with_tolerance(gt, pred, frame_tol=5)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_events_one_to_one() -> None:
    # Two predictions for the same GT — only one should match.
    gt = [(10, 1, 2)]
    pred = [(10, 1, 2), (12, 1, 2)]
    tp, fp, fn, _ = M.match_events_with_tolerance(gt, pred, frame_tol=5)
    assert (tp, fp, fn) == (1, 1, 0)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def test_smoothing_majority_vote() -> None:
    labels = [0, 0, 1, 0, 0]
    out = M.smooth_labels_majority(labels, window=3)
    assert out == [0, 0, 0, 0, 0]


def test_smoothing_window_one_is_identity() -> None:
    labels = [0, 1, 2, 1, 0]
    assert M.smooth_labels_majority(labels, window=1) == labels


def test_smoothing_invalid_window() -> None:
    with pytest.raises(ValueError):
        M.smooth_labels_majority([0, 1], window=0)
