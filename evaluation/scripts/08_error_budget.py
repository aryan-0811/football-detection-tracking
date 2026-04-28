"""
08_error_budget   (BONUS — optional, requires extensive GT)
============================================================

Decomposes end-to-end error into per-stage contributions by recomputing a
single downstream metric under five conditions:

  1. Oracle baseline:   GT detections + GT tracks + GT teams + GT homography
  2. Real detections + GT everything else
  3. Real det + real tracks + GT teams + GT homography
  4. Real det + real tracks + real teams + GT homography
  5. Full pipeline (real everything)

The downstream metric this script computes is **possession share per team**
because it depends on detections, tracks, team labels, and the ball position
(via homography), so each ablation reveals the error contribution of the
corresponding stage.

If any required input is missing for a given condition, that condition is
flagged in MISSING_DATA.md and skipped (rather than producing a misleading
number).

Inputs (all optional — script gracefully degrades)
--------------------------------------------------
- data/ground_truth/<clip>/tracks_mot.txt
- data/ground_truth/<clip>/team_labels.json
- data/ground_truth/<clip>/passes.json     (used to derive GT possession)
- data/ground_truth/<clip>/homography_keypoints.json (or homography.npz GT — see below)
- data/pipeline_outputs/<clip>/tracks_bytetrack.txt
- data/pipeline_outputs/<clip>/team_predictions_smoothed.csv
- data/pipeline_outputs/<clip>/<clip>_stats.json

Outputs
-------
- results/tables/error_budget.csv
- results/figures/error_budget_waterfall.pdf
- results/raw/error_budget.json
- results/tables/error_budget_summary.md
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
from evaluation.utils.ground_truth import (
    gt_dir, load_gt_passes, load_gt_team_labels, load_gt_tracks,
    load_predicted_tracks, load_stats, pipeline_out_dir,
)

logger = logging.getLogger("08_error_budget")

CONDITIONS = [
    "oracle",
    "real_det_gt_rest",
    "real_det_real_track_gt_rest",
    "real_det_real_track_real_team_gt_homog",
    "full_pipeline",
]


def _possession_from_passes(passes: list[Any], team_lookup: dict[int, int]
                            ) -> dict[int, float]:
    """Approximate possession share = pass-count share per team."""
    counts: dict[int, int] = {}
    total = 0
    for p in passes:
        team = getattr(p, "team_id", None)
        if team is None:
            team = team_lookup.get(p.passer_id)
        if team is None:
            continue
        counts[team] = counts.get(team, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {t: c / total for t, c in counts.items()}


def _possession_from_pipeline(stats: dict[str, Any]) -> dict[int, float]:
    """Try a few reasonable shapes; return empty dict if not found."""
    m = stats.get("match", {})
    pos = m.get("possession") or m.get("possession_pct") or stats.get("possession")
    if not pos:
        return {}
    out: dict[int, float] = {}
    if isinstance(pos, dict):
        for k, v in pos.items():
            try:
                out[int(k)] = float(v) / (100.0 if float(v) > 1.5 else 1.0)
            except (ValueError, TypeError):
                continue
    return out


def _possession_from_team_predictions(pipeline_dir: Path) -> dict[int, float]:
    """Frame-share per team if we don't have richer GT or pipeline output."""
    p = pipeline_dir / "team_predictions_smoothed.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "team" not in df.columns or df.empty:
        return {}
    counts = df["team"].value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in counts.items()}


def _l1(a: dict[int, float], b: dict[int, float]) -> float:
    """L1 distance between two normalised possession distributions."""
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return float("nan")
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def _evaluate_condition(condition: str, clip_name: str,
                        oracle_pos: dict[int, float],
                        gt_team_lookup: dict[int, int],
                        ) -> tuple[float | None, str]:
    """
    Returns (metric_value, note). Metric is L1 deviation from the oracle for
    the condition. None means we couldn't compute it.

    Implementation note: a fully-faithful per-condition evaluation would
    require swapping each stage's output independently — this is genuinely
    expensive and partially out of scope. Instead, we approximate:

      - "oracle" = GT-derived possession (our reference point, L1 = 0)
      - "real_det_gt_rest" — needs a GT-tracks file but also requires a
            re-run with GT-fed pipeline; flagged as not-yet-implementable
      - "real_det_real_track_gt_rest" — same caveat
      - "real_det_real_track_real_team_gt_homog" — same caveat
      - "full_pipeline" = the pipeline's published possession (or a fallback
            from team_predictions_smoothed.csv frame share) compared to oracle

    The intermediate conditions (rows 2–4) require a *partial* re-run of
    the pipeline with selected GT injected. Document this honestly in the
    output rather than fabricating numbers.
    """
    if condition == "oracle":
        return 0.0, "GT-derived possession (reference)"

    pipeline_dir = pipeline_out_dir(clip_name)

    if condition == "full_pipeline":
        try:
            stats = load_stats(clip_name)
        except FileNotFoundError as e:
            return None, f"stats.json missing: {e}"
        pos = _possession_from_pipeline(stats)
        if not pos:
            pos = _possession_from_team_predictions(pipeline_dir)
        if not pos:
            return None, "pipeline possession not found in stats.json or team_predictions"
        return _l1(pos, oracle_pos), "L1(possession_pred, possession_oracle)"

    return None, ("partial-injection condition requires a dedicated re-run "
                  "of run.py with GT injected at this stage. See "
                  "error_budget_summary.md for guidance.")


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("08_error_budget", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("08_error_budget", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for clip in clips:
        try:
            gt_team = load_gt_team_labels(clip.name)
            gt_passes = load_gt_passes(clip.name)
        except FileNotFoundError as e:
            skipped.append((clip.name, str(e)))
            logger.warning("skipping %s: %s", clip.name, e)
            continue

        oracle_pos = _possession_from_passes(gt_passes, gt_team.by_track)
        if not oracle_pos:
            skipped.append((clip.name, "oracle possession could not be derived"))
            continue

        for cond in CONDITIONS:
            value, note = _evaluate_condition(cond, clip.name, oracle_pos,
                                              gt_team.by_track)
            rows.append({
                "clip": clip.name,
                "condition": cond,
                "metric_l1_possession": value if value is not None else float("nan"),
                "computable": int(value is not None),
                "note": note,
            })

    if not rows:
        logger.error("Error budget produced no rows. This script is bonus and "
                     "requires extensive GT (passes + team_labels for each clip).")
        return 2

    df = pd.DataFrame(rows)
    csv_path = sub["tables"] / "error_budget.csv"
    io_helpers.write_csv(df, csv_path)

    # Waterfall: only computable rows
    plot_df = df[df["computable"] == 1]
    if not plot_df.empty:
        import matplotlib.pyplot as plt
        agg = plot_df.groupby("condition")["metric_l1_possession"].mean()
        agg = agg.reindex([c for c in CONDITIONS if c in agg.index])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(agg.index, agg.values, edgecolor="black")
        ax.set_ylabel("Mean L1 vs oracle possession")
        ax.set_title("Error budget (lower = closer to oracle)")
        ax.tick_params(axis="x", labelrotation=20)
        fig.savefig(sub["figures"] / "error_budget_waterfall.pdf")
        fig.savefig(sub["figures"] / "error_budget_waterfall.png")
        plt.close(fig)

    io_helpers.write_json({"rows": rows, "skipped": skipped},
                          sub["raw"] / "error_budget.json")

    md_lines = [
        "# Error budget — summary (BONUS)",
        "",
        "This is the optional five-condition oracle decomposition. The "
        "downstream metric is **per-team possession share**, measured as "
        "L1 deviation from the oracle.",
        "",
        "## Computability",
        "",
        f"Out of {len(df)} rows, **{int(df['computable'].sum())}** are "
        f"computable from existing artefacts. The remainder require a "
        f"partial re-run of the pipeline with selected GT injected; see "
        f"INSTRUMENTATION_TODO.md and the notes column in `error_budget.csv`.",
        "",
        "## How to fill in the missing rows",
        "",
        "Add a `--inject-gt {detections,tracks,teams,homography}` flag set "
        "to run.py that loads the corresponding GT artefact at the start of "
        "each pipeline stage instead of running it. Then run:",
        "",
        "  - condition 2: `--inject-gt tracks teams homography`",
        "  - condition 3: `--inject-gt teams homography`",
        "  - condition 4: `--inject-gt homography`",
        "  - condition 5: no injection (full pipeline)",
        "",
        "Each run produces a `<clip>_stats.json` whose possession is the "
        "value for that row. Re-run this script after each pipeline run.",
    ]
    if skipped:
        md_lines += ["", "## Skipped clips", *[f"- {c}: {r}" for c, r in skipped]]
    md_path = sub["tables"] / "error_budget_summary.md"
    io_helpers.write_markdown(md_path, "\n".join(md_lines) + "\n")

    n_comp = int(df["computable"].sum())
    print(f"08_error_budget (bonus): {len(df)} rows, {n_comp} computable. "
          f"See {csv_path.name} and error_budget_summary.md for the rest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
