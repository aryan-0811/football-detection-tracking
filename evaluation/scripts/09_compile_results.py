"""
09_compile_results
==================

Aggregates outputs from scripts 01–08 into a single set of report-ready files:

- results/tables/all_results.csv          — master table with one row per metric
- results/CHAPTER_4_DATA.md               — index keyed by report section
- results/SUCCESS_CRITERIA.md             — verdict on each criterion
- results/MISSING_DATA.md                 — list of inputs / experiments still
                                            needed to fill in the harness

Can be run two ways:

  python -m evaluation.scripts.09_compile_results
      → assumes 01..08 have already produced artefacts in results/.

  python -m evaluation.scripts.09_compile_results --run-all
      → invokes 01..08 programmatically before compiling.

The script never crashes on missing intermediate outputs; it lists them in
`MISSING_DATA.md` so the user can see exactly what to do next.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation import config
from evaluation.scripts._common import (
    apply_plot_style, configure_logging, make_parser, resolve_subdirs,
    select_clips,
)
from evaluation.utils import io_helpers

logger = logging.getLogger("09_compile_results")


SCRIPTS_IN_ORDER = [
    "evaluation.scripts.01_detection_metrics",
    "evaluation.scripts.02_tracking_metrics",
    "evaluation.scripts.03_team_classification_metrics",
    "evaluation.scripts.04_homography_metrics",
    "evaluation.scripts.05_pass_offside_metrics",
    "evaluation.scripts.06_speed_distance_analysis",
    "evaluation.scripts.07_runtime_profiling",
    "evaluation.scripts.08_error_budget",
]


def _run_all(clips: list[str] | None, output_dir: Path) -> dict[str, int]:
    results: dict[str, int] = {}
    for mod_name in SCRIPTS_IN_ORDER:
        mod = importlib.import_module(mod_name)
        argv = []
        if clips:
            argv += ["--clips", *clips]
        argv += ["--output-dir", str(output_dir)]
        try:
            rc = mod.main(argv)
        except SystemExit as se:
            rc = int(se.code or 0)
        except Exception as e:
            logger.error("%s crashed: %s", mod_name, e)
            rc = 99
        results[mod_name] = rc
    return results


def _safe_read_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        logger.warning("could not read %s: %s", p, e)
        return None


def _flatten_to_master(tables_dir: Path) -> pd.DataFrame:
    """Pull a small, hand-picked set of headline numbers from each table CSV."""
    rows: list[dict[str, Any]] = []

    # Detection
    df = _safe_read_csv(tables_dir / "detection_metrics_full.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.2_detection", "metric": f"{r['detector']}/{r['class']}",
                         "P": r.get("P"), "R": r.get("R"),
                         "F1": r.get("F1"), "mAP50": r.get("mAP50"),
                         "mAP5095": r.get("mAP5095")})

    # Tracking
    df = _safe_read_csv(tables_dir / "tracking_metrics.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.3_tracking",
                         "metric": f"{r['clip']}/{r['tracker']}",
                         "HOTA": r.get("HOTA"), "MOTA": r.get("MOTA"),
                         "IDF1": r.get("IDF1")})

    # Team
    df = _safe_read_csv(tables_dir / "team_classification_metrics.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.4_team",
                         "metric": f"{r['clip']}/{r['condition']}/{r['class']}",
                         "P": r.get("P"), "R": r.get("R"), "F1": r.get("F1")})

    # Homography
    df = _safe_read_csv(tables_dir / "homography_reprojection_error.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.5_homography",
                         "metric": f"{r['clip']}/{r['condition']}",
                         "mean_error_m": r.get("mean_error_m"),
                         "p95_error_m": r.get("p95_error_m")})

    # Pass / offside
    df = _safe_read_csv(tables_dir / "pass_detection_metrics.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.6_pass",
                         "metric": f"{r['clip']}/passes",
                         "P": r.get("P"), "R": r.get("R"), "F1": r.get("F1"),
                         "TP": r.get("TP"), "FP": r.get("FP"), "FN": r.get("FN")})

    # Speed
    df = _safe_read_csv(tables_dir / "speed_distance_analysis.csv")
    if df is not None and not df.empty:
        rows.append({
            "section": "4.7_speed", "metric": "tracks_total",
            "value": int(len(df)),
        })
        rows.append({
            "section": "4.7_speed", "metric": "tracks_at_cap_pct",
            "value": float(100.0 * df["at_cap"].sum() / len(df)),
        })

    # Runtime
    df = _safe_read_csv(tables_dir / "runtime_profile.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.8_runtime",
                         "metric": f"stage/{r['stage']}",
                         "mean_ms": r.get("mean_ms"),
                         "p95_ms": r.get("p95_ms"),
                         "fraction_of_total": r.get("fraction_of_total")})

    # Error budget
    df = _safe_read_csv(tables_dir / "error_budget.csv")
    if df is not None:
        for _, r in df.iterrows():
            rows.append({"section": "4.9_error_budget",
                         "metric": f"{r['clip']}/{r['condition']}",
                         "metric_l1_possession": r.get("metric_l1_possession"),
                         "computable": r.get("computable")})

    return pd.DataFrame(rows)


def _check_success_criteria(tables_dir: Path) -> list[dict[str, Any]]:
    """Pull headline numbers and compare against config.METRIC_THRESHOLDS."""
    out: list[dict[str, Any]] = []

    # (i) Detection mAP50 (generic, "all")
    det = _safe_read_csv(tables_dir / "detection_metrics_full.csv")
    val: float | None = None
    if det is not None:
        sub = det[(det["detector"] == "generic_4class") & (det["class"] == "all")]
        if not sub.empty:
            val = float(sub["mAP50"].iloc[0])
    out.append({
        "criterion": "(i) Detection mAP@50 >= 0.80",
        "threshold": config.METRIC_THRESHOLDS["detection_mAP50"],
        "value": val,
        "met": _met(val, config.METRIC_THRESHOLDS["detection_mAP50"]),
        "source": "tables/detection_metrics_full.csv",
    })

    # (ii) Tracking HOTA mean >= 0.45 (excluding reference rows)
    tr = _safe_read_csv(tables_dir / "tracking_metrics.csv")
    val = None
    if tr is not None:
        own = tr[tr.get("is_reference", 0) == 0] if "is_reference" in tr.columns else tr
        if not own.empty:
            val = float(own["HOTA"].mean())
    out.append({
        "criterion": "(ii) Tracking HOTA >= 0.45",
        "threshold": config.METRIC_THRESHOLDS["tracking_HOTA"],
        "value": val,
        "met": _met(val, config.METRIC_THRESHOLDS["tracking_HOTA"]),
        "source": "tables/tracking_metrics.csv",
    })

    # (iii) Team smoothed track-level macro-F1 >= 0.85
    team = _safe_read_csv(tables_dir / "team_classification_metrics.csv")
    val = None
    if team is not None:
        sub = team[(team["condition"] == "smoothed_track_level") & (team["class"] == "macro")]
        if not sub.empty:
            val = float(sub["F1"].mean())
    out.append({
        "criterion": "(iii) Team-classification F1 >= 0.85",
        "threshold": config.METRIC_THRESHOLDS["team_F1"],
        "value": val,
        "met": _met(val, config.METRIC_THRESHOLDS["team_F1"]),
        "source": "tables/team_classification_metrics.csv",
    })

    # (iv) Pass F1 >= 0.60
    p = _safe_read_csv(tables_dir / "pass_detection_metrics.csv")
    val = float(p["F1"].mean()) if p is not None and not p.empty else None
    out.append({
        "criterion": "(iv) Pass-detection F1 >= 0.60",
        "threshold": config.METRIC_THRESHOLDS["pass_F1"],
        "value": val,
        "met": _met(val, config.METRIC_THRESHOLDS["pass_F1"]),
        "source": "tables/pass_detection_metrics.csv",
    })
    return out


def _met(v: float | None, t: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "UNKNOWN"
    return "MET" if v >= t else "NOT MET"


def _list_missing(tables_dir: Path, figures_dir: Path) -> list[str]:
    expected_tables = [
        "detection_metrics_full.csv",
        "tracking_metrics.csv",
        "team_classification_metrics.csv",
        "homography_reprojection_error.csv",
        "pass_detection_metrics.csv",
        "speed_distance_analysis.csv",
        "runtime_profile.csv",
        "error_budget.csv",
    ]
    expected_figs = [
        "confusion_matrix_generic.pdf",
        "tracking_hota_comparison.pdf",
        "warmup_umap.pdf",
        "reprojection_error_distribution.pdf",
        "pass_event_timeline.pdf",
        "speed_distribution.pdf",
        "runtime_breakdown.pdf",
        "error_budget_waterfall.pdf",
    ]
    missing = []
    for f in expected_tables:
        if not (tables_dir / f).exists():
            missing.append(f"tables/{f}")
    for f in expected_figs:
        if not (figures_dir / f).exists():
            missing.append(f"figures/{f}")
    return missing


def _make_parser_local():
    parser = make_parser("09_compile_results", __doc__ or "")
    parser.add_argument("--run-all", action="store_true",
                        help="Invoke 01..08 before compiling.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser_local()
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("09_compile_results", args.log_level, args.output_dir)

    if args.run_all:
        clips_arg = args.clips if args.clips else None
        rcs = _run_all(clips_arg, args.output_dir)
        for k, v in rcs.items():
            logger.info("ran %s → rc=%d", k, v)

    master = _flatten_to_master(sub["tables"])
    if not master.empty:
        io_helpers.write_csv(master, sub["tables"] / "all_results.csv")

    crit = _check_success_criteria(sub["tables"])
    crit_md = ["# Success-criterion verdict\n"]
    crit_md.append("| criterion | threshold | observed | status | source |")
    crit_md.append("|---|---|---|---|---|")
    for c in crit:
        v = c["value"]
        v_str = f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "—"
        crit_md.append(f"| {c['criterion']} | {c['threshold']:.2f} | {v_str} | "
                       f"**{c['met']}** | `{c['source']}` |")
    io_helpers.write_markdown(args.output_dir / "SUCCESS_CRITERIA.md",
                              "\n".join(crit_md) + "\n")

    chapter_md = [
        "# Chapter 4 data index",
        "",
        "Use this document to find the table or figure backing each citation "
        "in the evaluation chapter. All paths are relative to "
        "`evaluation/results/`.",
        "",
        "## §4.2 Detection",
        "- Table: `tables/detection_metrics_full.csv`",
        "- Verdict: `tables/detection_metrics_summary.md`",
        "- Figures: `figures/confusion_matrix_generic.pdf`, "
        "`figures/per_class_pr_curves.pdf`",
        "",
        "## §4.3 Tracking",
        "- Table: `tables/tracking_metrics.csv` (with SoccerNet reference row)",
        "- Aggregate: `tables/tracking_summary.csv`",
        "- Figure: `figures/tracking_hota_comparison.pdf`",
        "- Verdict: `tables/tracking_summary.md`",
        "",
        "## §4.4 Team classification",
        "- Table: `tables/team_classification_metrics.csv`",
        "- Ablation: `tables/team_classification_ablation.csv`",
        "- Figures: `figures/warmup_umap.pdf`, "
        "`figures/team_smoothing_effect_<clip>.pdf`",
        "- Verdict: `tables/team_summary.md`",
        "",
        "## §4.5 Homography",
        "- Table: `tables/homography_reprojection_error.csv`",
        "- Figures: `figures/reprojection_error_distribution.pdf`, "
        "`figures/per_keypoint_contribution.pdf`",
        "- Verdict: `tables/homography_summary.md`",
        "",
        "## §4.6 Passes & offsides",
        "- Tables: `tables/pass_detection_metrics.csv`, "
        "`tables/offside_event_analysis.csv`",
        "- Figure: `figures/pass_event_timeline.pdf`",
        "- Verdict: `tables/tactical_summary.md`",
        "",
        "## §4.7 Speed / distance",
        "- Tables: `tables/speed_distance_analysis.csv`, "
        "`tables/speed_cap_activation_summary.csv`",
        "- Figures: `figures/speed_distribution.pdf`, "
        "`figures/cap_activation_per_track.pdf`",
        "- Verdict: `tables/speed_summary.md`",
        "",
        "## §4.8 Runtime",
        "- Tables: `tables/runtime_profile.csv`, "
        "`tables/runtime_profile_per_clip.csv`",
        "- Figures: `figures/runtime_breakdown.pdf`, "
        "`figures/runtime_timeline.pdf`",
        "- Verdict: `tables/runtime_summary.md`",
        "",
        "## §4.9 Error budget (bonus)",
        "- Table: `tables/error_budget.csv`",
        "- Figure: `figures/error_budget_waterfall.pdf`",
        "- Verdict: `tables/error_budget_summary.md`",
        "",
        "## Master rollup",
        "- `tables/all_results.csv`",
        "- `SUCCESS_CRITERIA.md`",
        "- `MISSING_DATA.md`",
    ]
    io_helpers.write_markdown(args.output_dir / "CHAPTER_4_DATA.md",
                              "\n".join(chapter_md) + "\n")

    missing = _list_missing(sub["tables"], sub["figures"])
    miss_md = ["# Missing data / experiments\n"]
    if not missing:
        miss_md.append("All expected tables and figures are present.\n")
    else:
        miss_md.append("The following expected outputs are missing. "
                       "For each, consult the relevant script's docstring "
                       "and `INSTRUMENTATION_TODO.md`.\n")
        for m in missing:
            miss_md.append(f"- `{m}`")
        miss_md.append("")
    io_helpers.write_markdown(args.output_dir / "MISSING_DATA.md",
                              "\n".join(miss_md) + "\n")

    print(f"09_compile_results: {len(master)} master rows, "
          f"{len(missing)} expected outputs missing. "
          f"See SUCCESS_CRITERIA.md and CHAPTER_4_DATA.md in {args.output_dir.name}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
