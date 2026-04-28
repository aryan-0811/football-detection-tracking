"""
07_runtime_profiling
====================

End-to-end runtime characterisation. Computes per-stage mean / p50 / p95 / share
of total time, compares per-frame totals against the 33 ms real-time budget
(30 fps), and tracks variance across clips.

Serves: §3.10 / §4.7.

Inputs
------
Per clip:
- data/pipeline_outputs/<clip>/timings.csv     (NEW — INSTRUMENTATION_TODO.md item 1)

Schema:
    frame, stage, ms

Outputs
-------
- results/tables/runtime_profile.csv              (per-stage stats, all clips merged)
- results/tables/runtime_profile_per_clip.csv     (same, broken down by clip)
- results/figures/runtime_breakdown.pdf
- results/figures/runtime_timeline.pdf
- results/raw/runtime_profile.json
- results/tables/runtime_summary.md
"""

from __future__ import annotations

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
from evaluation.utils.ground_truth import load_timings

logger = logging.getLogger("07_runtime_profiling")

REAL_TIME_BUDGET_MS = 1000.0 / 30.0  # 33.33 ms at 30 fps


def _per_stage_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("stage")["ms"]
    out = pd.DataFrame({
        "mean_ms": g.mean(),
        "p50_ms": g.median(),
        "p95_ms": g.quantile(0.95),
        "max_ms": g.max(),
        "n": g.count(),
    }).reset_index()
    total_mean = out["mean_ms"].sum()
    out["fraction_of_total"] = out["mean_ms"] / total_mean if total_mean > 0 else 0.0
    return out.sort_values("mean_ms", ascending=False).reset_index(drop=True)


def _per_frame_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Total ms per (clip, frame). Assumes all stages report on the same frame."""
    return df.groupby(["clip", "frame"])["ms"].sum().reset_index(name="total_ms")


def _plot_breakdown(per_stage: pd.DataFrame, out_path: Path) -> None:
    if per_stage.empty:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(per_stage["stage"], per_stage["mean_ms"], edgecolor="black")
    ax.axvline(REAL_TIME_BUDGET_MS, color="red", ls="--", lw=1,
               label=f"30-fps budget ({REAL_TIME_BUDGET_MS:.1f} ms)")
    ax.set_xlabel("mean ms / frame")
    ax.set_title("Per-stage runtime breakdown (mean across all clips/frames)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def _plot_timeline(totals: pd.DataFrame, out_path: Path) -> None:
    if totals.empty:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for clip, sub in totals.groupby("clip"):
        ax.plot(sub["frame"], sub["total_ms"], lw=0.6, alpha=0.7, label=str(clip))
    ax.axhline(REAL_TIME_BUDGET_MS, color="red", ls="--", lw=1,
               label=f"30 fps ({REAL_TIME_BUDGET_MS:.1f} ms)")
    ax.set_xlabel("frame"); ax.set_ylabel("total ms")
    ax.set_title("Per-frame total runtime")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("07_runtime_profiling", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("07_runtime_profiling", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for clip in clips:
        try:
            df = load_timings(clip.name)
        except FileNotFoundError as e:
            skipped.append((clip.name, str(e)))
            logger.warning("skipping %s: %s", clip.name, e)
            continue
        if df.empty:
            skipped.append((clip.name, "timings.csv was empty"))
            continue
        if "ms" not in df.columns or "stage" not in df.columns:
            skipped.append((clip.name, "timings.csv missing 'ms'/'stage' columns"))
            continue
        df = df.copy()
        df["clip"] = clip.name
        frames.append(df)

    if not frames:
        logger.error("No timings.csv loaded. See INSTRUMENTATION_TODO.md item 1.")
        return 2

    big = pd.concat(frames, ignore_index=True)

    overall = _per_stage_stats(big)
    io_helpers.write_csv(overall, sub["tables"] / "runtime_profile.csv")

    per_clip = (big.groupby(["clip", "stage"])["ms"]
                .agg(mean_ms="mean", p50_ms="median",
                     p95_ms=lambda x: float(np.percentile(x, 95)))
                .reset_index())
    io_helpers.write_csv(per_clip, sub["tables"] / "runtime_profile_per_clip.csv")

    totals = _per_frame_totals(big)
    n_over_budget = int((totals["total_ms"] > REAL_TIME_BUDGET_MS).sum())
    pct_over = 100.0 * n_over_budget / len(totals) if len(totals) else 0.0
    mean_total = float(totals["total_ms"].mean())
    p95_total = float(np.percentile(totals["total_ms"], 95))

    _plot_breakdown(overall, sub["figures"] / "runtime_breakdown.pdf")
    _plot_timeline(totals, sub["figures"] / "runtime_timeline.pdf")

    io_helpers.write_json({
        "overall_per_stage": overall.to_dict(orient="records"),
        "per_clip_per_stage": per_clip.to_dict(orient="records"),
        "totals_summary": {
            "mean_total_ms": mean_total,
            "p95_total_ms": p95_total,
            "frames_over_budget": n_over_budget,
            "pct_over_budget": pct_over,
            "budget_ms": REAL_TIME_BUDGET_MS,
        },
        "skipped": skipped,
    }, sub["raw"] / "runtime_profile.json")

    feasibility = ("real-time at 30 fps" if mean_total <= REAL_TIME_BUDGET_MS
                   else "below real-time at 30 fps")
    md_lines = [
        "# Runtime profile — summary",
        "",
        f"Mean total per-frame: **{mean_total:.2f} ms** "
        f"(budget {REAL_TIME_BUDGET_MS:.2f} ms). p95 = {p95_total:.2f} ms.",
        f"Frames over budget: {n_over_budget} / {len(totals)} ({pct_over:.1f} %).",
        f"Verdict: pipeline is **{feasibility}** on the test hardware.",
        "",
        "## Top stages by mean cost",
        "",
        "| stage | mean_ms | p95_ms | share |",
        "|---|---|---|---|",
    ]
    for _, row in overall.head(8).iterrows():
        md_lines.append(
            f"| {row['stage']} | {row['mean_ms']:.2f} | {row['p95_ms']:.2f} | "
            f"{row['fraction_of_total'] * 100:.1f}% |"
        )
    md_lines += [
        "",
        "## Hardware context",
        "",
        "These numbers reflect the hardware on which the pipeline was run "
        "(M2 Pro, CPU-only inference). Document any GPU acceleration or "
        "model quantisation in the report alongside these tables.",
    ]
    if skipped:
        md_lines += ["", "## Skipped clips", *[f"- {c}: {r}" for c, r in skipped]]
    md_path = sub["tables"] / "runtime_summary.md"
    io_helpers.write_markdown(md_path, "\n".join(md_lines) + "\n")

    print(f"07_runtime_profiling: mean total {mean_total:.2f} ms, "
          f"p95 {p95_total:.2f} ms; {pct_over:.1f}% over 33 ms. "
          f"{feasibility}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
