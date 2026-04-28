"""Unit tests for io_helpers. Run with: pytest evaluation/tests"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.utils import io_helpers


def test_write_then_read_json(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": np.float32(0.5)}
    io_helpers.write_json(payload, p)
    out = io_helpers.read_json(p)
    assert out["a"] == 1
    assert out["b"] == [1, 2, 3]
    assert abs(out["c"] - 0.5) < 1e-6


def test_read_json_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io_helpers.read_json(tmp_path / "missing.json")


def test_maybe_read_json_missing_returns_none(tmp_path: Path) -> None:
    assert io_helpers.maybe_read_json(tmp_path / "missing.json") is None


def test_csv_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    io_helpers.write_csv(df, p)
    out = io_helpers.read_csv(p)
    assert out.equals(df)


def test_npy_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "x.npy"
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    io_helpers.write_npy(arr, p)
    out = io_helpers.read_npy(p)
    assert np.allclose(out, arr)


def test_mot_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "tracks.txt"
    df = pd.DataFrame({
        "frame": [1, 1, 2],
        "id": [1, 2, 1],
        "bb_left": [10.0, 20.0, 11.0],
        "bb_top": [30.0, 40.0, 31.0],
        "bb_width": [50.0, 60.0, 50.0],
        "bb_height": [70.0, 80.0, 70.0],
    })
    io_helpers.write_mot(df, p)
    loaded = io_helpers.read_mot(p)
    assert list(loaded["frame"]) == [1, 1, 2]
    assert list(loaded["id"]) == [1, 2, 1]
    assert loaded.loc[0, "bb_left"] == 10.0


def test_read_mot_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        io_helpers.read_mot(tmp_path / "no.txt")


def test_read_mot_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError):
        io_helpers.read_mot(p)


def test_require_files(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"; a.write_text("x")
    b = tmp_path / "b.txt"
    with pytest.raises(FileNotFoundError) as exc:
        io_helpers.require_files([a, b], context="unit-test")
    assert "unit-test" in str(exc.value)
    assert "b.txt" in str(exc.value)


def test_append_csv_row(tmp_path: Path) -> None:
    p = tmp_path / "out.csv"
    io_helpers.append_csv_row(p, {"a": 1, "b": 2})
    io_helpers.append_csv_row(p, {"a": 3, "b": 4})
    df = pd.read_csv(p)
    assert list(df["a"]) == [1, 3]
    assert list(df["b"]) == [2, 4]
