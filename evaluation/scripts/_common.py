"""
Boilerplate shared by every analysis script: argparse setup, logging, and
clip iteration. Imported by each `0X_*.py` so the per-script files stay focused
on the actual analysis logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

from evaluation import config


def make_parser(script_name: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=script_name, description=description)
    p.add_argument(
        "--clips", nargs="*", default=None,
        help="Subset of clip names to evaluate (default: all clips in config).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=config.RESULTS_DIR,
        help="Override results directory (default: evaluation/results).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def configure_logging(script_name: str, level: str = "INFO",
                      output_dir: Path = config.RESULTS_DIR) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{script_name}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level))

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter(fmt))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)
    root.addHandler(sh)

    return logging.getLogger(script_name)


def select_clips(names: list[str] | None) -> list[config.ClipDef]:
    """Return ClipDefs for the requested names, or all configured clips."""
    if names is None:
        return list(config.CLIPS)
    by_name = {c.name: c for c in config.CLIPS}
    out: list[config.ClipDef] = []
    for n in names:
        if n not in by_name:
            raise SystemExit(f"Unknown clip '{n}'. Add it to config.CLIPS.")
        out.append(by_name[n])
    return out


def resolve_subdirs(output_dir: Path) -> dict[str, Path]:
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    raw = output_dir / "raw"
    logs = output_dir / "logs"
    for d in (tables, figures, raw, logs):
        d.mkdir(parents=True, exist_ok=True)
    return {"tables": tables, "figures": figures, "raw": raw, "logs": logs}


def apply_plot_style() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    plt.rcParams.update(config.PLOT_STYLE)
