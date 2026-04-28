"""
01_detection_metrics
====================

Produces per-class detection metrics for the four-class generic detector and
the two single-class detectors (ball, pitch keypoints).

Serves: §3.4 forward reference and Success Criterion (i) — mAP@50 >= 0.80.

Inputs
------
Reads pre-computed Ultralytics validation reports from
`evaluation/data/detection_reports/`. Two formats are supported:

A) `<detector>.json` — direct dump of `ultralytics.YOLO.val(...).results_dict`.
   Expected keys (Ultralytics names, suffixed with B for bbox):
       "metrics/precision(B)", "metrics/recall(B)",
       "metrics/mAP50(B)", "metrics/mAP50-95(B)"
   plus optional per-class arrays under "per_class" with structure:
       {"<class_name>": {"P": float, "R": float, "F1": float,
                         "mAP50": float, "mAP5095": float, "support": int}}

B) `<detector>.yaml` — same structure, YAML-serialised.

If neither file exists for a given detector, the row is skipped with a clear
log message. The script never re-runs validation itself (that would require
loading model weights and the test set).

Outputs
-------
- results/tables/detection_metrics_full.csv
- results/raw/detection_metrics.json
- results/figures/confusion_matrix_generic.pdf  (if confusion matrix dumped)
- results/figures/per_class_pr_curves.pdf       (if PR data dumped)
- results/tables/detection_metrics_summary.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation import config
from evaluation.scripts._common import (
    apply_plot_style, configure_logging, make_parser, resolve_subdirs,
)
from evaluation.utils import io_helpers

logger = logging.getLogger("01_detection_metrics")


DETECTORS = [
    {"name": "generic_4class", "classes": ["ball", "goalkeeper", "player", "referee"]},
    {"name": "ball_specialist", "classes": ["ball"]},
    {"name": "pitch_keypoints", "classes": ["keypoint"]},
]


def _load_report(detector_name: str) -> dict[str, Any] | None:
    """Try .json then .yaml. Return None if neither exists."""
    base = config.DETECTION_REPORTS_DIR / detector_name
    j, y = base.with_suffix(".json"), base.with_suffix(".yaml")
    if j.exists():
        return io_helpers.read_json(j)
    if y.exists():
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML missing — cannot parse %s", y)
            return None
        return yaml.safe_load(y.read_text())
    return None


def _rows_from_report(detector: dict[str, Any], report: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    name = detector["name"]
    classes = detector["classes"]

    # Per-class block if available.
    per_class = report.get("per_class") or {}
    for cls in classes:
        block = per_class.get(cls, {})
        rows.append({
            "detector": name,
            "class": cls,
            "P": float(block.get("P", float("nan"))),
            "R": float(block.get("R", float("nan"))),
            "F1": float(block.get("F1", float("nan"))),
            "mAP50": float(block.get("mAP50", float("nan"))),
            "mAP5095": float(block.get("mAP5095", float("nan"))),
            "support": int(block.get("support", -1)),
        })

    # Aggregate "all" row from top-level keys.
    rows.append({
        "detector": name,
        "class": "all",
        "P": float(report.get("metrics/precision(B)", float("nan"))),
        "R": float(report.get("metrics/recall(B)", float("nan"))),
        "F1": float("nan"),  # Ultralytics doesn't expose top-level F1
        "mAP50": float(report.get("metrics/mAP50(B)", float("nan"))),
        "mAP5095": float(report.get("metrics/mAP50-95(B)", float("nan"))),
        "support": -1,
    })
    return rows


def _plot_confusion_matrix(report: dict[str, Any], out_path: Path) -> bool:
    """Plot the normalised confusion matrix if the report contains one."""
    cm = report.get("confusion_matrix")
    classes = report.get("class_names")
    if cm is None or classes is None:
        return False
    arr = np.asarray(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    norm = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix (row-normalised)")
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return True


def _plot_pr_curves(report: dict[str, Any], out_path: Path) -> bool:
    """Plot PR curves per class if the report dumped PR points."""
    curves = report.get("pr_curves")
    if not curves:
        return False
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls_name, pts in curves.items():
        rec = np.asarray(pts.get("recall", []))
        pre = np.asarray(pts.get("precision", []))
        if rec.size and pre.size:
            ax.plot(rec, pre, label=cls_name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Per-class precision–recall curves")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return True


def _verdict(rows: list[dict[str, Any]]) -> str:
    threshold = config.METRIC_THRESHOLDS["detection_mAP50"]
    generic_all = [r for r in rows
                   if r["detector"] == "generic_4class" and r["class"] == "all"]
    if not generic_all:
        return f"No `generic_4class` aggregate row found; cannot evaluate mAP50 >= {threshold}."
    m = generic_all[0]["mAP50"]
    if np.isnan(m):
        return "mAP50 not present in the report — provide it to assess Criterion (i)."
    status = "MET" if m >= threshold else "NOT MET"
    return (f"Generic-detector mAP@50 = {m:.3f} (threshold {threshold:.2f}). "
            f"Criterion (i) **{status}**.")


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("01_detection_metrics", __doc__ or "")
    args = parser.parse_args(argv)

    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("01_detection_metrics", args.log_level, args.output_dir)

    rows: list[dict[str, Any]] = []
    raw_dump: dict[str, Any] = {}
    figures_made: list[str] = []
    missing: list[str] = []

    for det in DETECTORS:
        report = _load_report(det["name"])
        if report is None:
            logger.warning("No report found for detector '%s' — skipping. "
                           "Drop a JSON/YAML report at %s.json or .yaml.",
                           det["name"], config.DETECTION_REPORTS_DIR / det["name"])
            missing.append(det["name"])
            continue
        rows.extend(_rows_from_report(det, report))
        raw_dump[det["name"]] = report

        if det["name"] == "generic_4class":
            cm_path = sub["figures"] / "confusion_matrix_generic.pdf"
            if _plot_confusion_matrix(report, cm_path):
                figures_made.append(str(cm_path))
            pr_path = sub["figures"] / "per_class_pr_curves.pdf"
            if _plot_pr_curves(report, pr_path):
                figures_made.append(str(pr_path))

    if not rows:
        logger.error("No detection reports loaded. Drop reports under %s and re-run.",
                     config.DETECTION_REPORTS_DIR)
        return 2

    df = pd.DataFrame(rows, columns=[
        "detector", "class", "P", "R", "F1", "mAP50", "mAP5095", "support",
    ])
    csv_path = sub["tables"] / "detection_metrics_full.csv"
    io_helpers.write_csv(df, csv_path)
    json_path = sub["raw"] / "detection_metrics.json"
    io_helpers.write_json({"rows": rows, "raw_reports": raw_dump,
                           "missing_detectors": missing}, json_path)

    verdict = _verdict(rows)
    md = (
        "# Detection metrics — summary\n\n"
        f"{verdict}\n\n"
        f"- Full table: `{csv_path.relative_to(config.EVAL_DIR)}`\n"
        f"- Raw JSON: `{json_path.relative_to(config.EVAL_DIR)}`\n"
        f"- Figures: {', '.join(figures_made) if figures_made else 'none'}\n"
    )
    if missing:
        md += "\n## Missing reports\n" + "\n".join(f"- `{m}`" for m in missing) + "\n"
    md_path = sub["tables"] / "detection_metrics_summary.md"
    io_helpers.write_markdown(md_path, md)

    print(
        f"01_detection_metrics: wrote {len(df)} rows to {csv_path.name}, "
        f"{len(figures_made)} figures, summary at {md_path.name}. {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
