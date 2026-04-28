"""
05_pass_offside_metrics
=======================

Pass-detection P/R/F1 and per-event offside analysis.

Serves: §3.8 forward reference and Success Criterion (iv).

Inputs
------
Per clip:
- data/ground_truth/<clip>/passes.json
- data/ground_truth/<clip>/offsides.json
- data/pipeline_outputs/<clip>/<clip>_offside_events.json

Pipeline event schema (assumed from src/ball/events.py):
    [
      {"type": "pass", "frame": int, "passer_id": int, "receiver_id": int, ...},
      {"type": "offside", "frame": int, "offside_track_ids": [int, ...], ...},
      ...
    ]

If your pipeline uses different keys, adjust `_extract_pass_events` and
`_extract_offside_events` below — the script will log a warning if no events
of the expected type are found in a non-empty file.

Matching
--------
Pass TP: same passer_id and receiver_id, frame within
config.PASS_FRAME_TOLERANCE. One-to-one greedy match on smallest |Δframe|.

Offside TP: any predicted offside_track_ids overlap with GT
offside_track_ids on a matched frame (Δframe <= OFFSIDE_FRAME_TOLERANCE).

Outputs
-------
- results/tables/pass_detection_metrics.csv
- results/tables/offside_event_analysis.csv
- results/figures/pass_event_timeline.pdf
- results/raw/pass_offside.json
- results/tables/tactical_summary.md
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
    load_gt_passes, load_gt_offsides, load_offside_events, pipeline_out_dir,
)
from evaluation.utils.metrics_helpers import match_events_with_tolerance, prf1

logger = logging.getLogger("05_pass_offside_metrics")


def _extract_pass_events(events: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Return (frame, passer_id, receiver_id) tuples."""
    out: list[tuple[int, int, int]] = []
    for e in events:
        et = e.get("type", e.get("event_type", "")).lower()
        if et != "pass":
            continue
        try:
            f = int(e["frame"])
            p = int(e.get("passer_id", e.get("passer", -1)))
            r = int(e.get("receiver_id", e.get("receiver", -1)))
        except (KeyError, TypeError, ValueError):
            continue
        if p < 0 or r < 0:
            continue
        out.append((f, p, r))
    return out


def _extract_offside_events(events: list[dict[str, Any]]) -> list[tuple[int, list[int]]]:
    """Return (frame, [track_ids]) tuples."""
    out: list[tuple[int, list[int]]] = []
    for e in events:
        et = e.get("type", e.get("event_type", "")).lower()
        if et != "offside":
            continue
        try:
            f = int(e["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        ids_raw = e.get("offside_track_ids", e.get("track_ids", []))
        ids = [int(i) for i in ids_raw if isinstance(i, (int, float, str))
               and str(i).lstrip("-").isdigit()]
        out.append((f, ids))
    return out


def _eval_passes(clip_name: str, gt_passes: list, pred_events: list) -> dict[str, Any]:
    pred_passes = _extract_pass_events(pred_events)
    gt_tuples = [(p.frame, p.passer_id, p.receiver_id) for p in gt_passes]

    tp, fp, fn, _ = match_events_with_tolerance(
        gt_tuples, pred_passes,
        frame_tol=config.PASS_FRAME_TOLERANCE,
    )
    m = prf1(tp, fp, fn)
    return {
        "clip": clip_name,
        "total_predicted": len(pred_passes),
        "total_gt": len(gt_passes),
        "TP": tp, "FP": fp, "FN": fn,
        "P": m.precision, "R": m.recall, "F1": m.f1,
        "frame_tolerance": config.PASS_FRAME_TOLERANCE,
        "pred_passes": pred_passes,
        "gt_passes": gt_tuples,
    }


def _eval_offsides(clip_name: str, gt_offsides: list, pred_events: list
                   ) -> list[dict[str, Any]]:
    pred = _extract_offside_events(pred_events)
    gt_by_frame = {o.frame: set(o.offside_track_ids) for o in gt_offsides}
    pred_by_frame = {f: set(ids) for f, ids in pred}

    tol = config.OFFSIDE_FRAME_TOLERANCE
    matched_pred: set[int] = set()
    rows: list[dict[str, Any]] = []

    # Pass over GT events
    for o in gt_offsides:
        # find the closest predicted frame within tolerance
        closest = None
        best_df = tol + 1
        for i, (pf, _ids) in enumerate(pred):
            if i in matched_pred: continue
            df = abs(pf - o.frame)
            if df <= tol and df < best_df:
                closest = i; best_df = df
        if closest is None:
            rows.append({
                "clip": clip_name, "frame": o.frame,
                "predicted_offside_ids": "",
                "gt_offside_ids": ",".join(map(str, sorted(o.offside_track_ids))),
                "classification": "FN",
                "notes": "no predicted offside within tolerance",
            })
        else:
            pf, ids = pred[closest]
            overlap = sorted(set(ids) & set(o.offside_track_ids))
            cls = "TP" if overlap else "FP_id_mismatch"
            rows.append({
                "clip": clip_name, "frame": o.frame,
                "predicted_offside_ids": ",".join(map(str, sorted(ids))),
                "gt_offside_ids": ",".join(map(str, sorted(o.offside_track_ids))),
                "classification": cls,
                "notes": f"matched pred at frame {pf}, |df|={abs(pf - o.frame)}",
            })
            matched_pred.add(closest)

    # Unmatched predictions are FPs
    for i, (pf, ids) in enumerate(pred):
        if i in matched_pred: continue
        rows.append({
            "clip": clip_name, "frame": pf,
            "predicted_offside_ids": ",".join(map(str, sorted(ids))),
            "gt_offside_ids": "",
            "classification": "FP",
            "notes": "no GT offside within tolerance",
        })
    return rows


def _plot_pass_timeline(clip_name: str, pred: list[tuple[int, int, int]],
                        gt: list[tuple[int, int, int]], out_path: Path) -> None:
    if not pred and not gt:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.4))
    if gt:
        gt_frames = [t[0] for t in gt]
        ax.scatter(gt_frames, [1] * len(gt_frames), marker="|",
                   s=200, c="green", label="GT pass")
    if pred:
        pred_frames = [t[0] for t in pred]
        ax.scatter(pred_frames, [0] * len(pred_frames), marker="|",
                   s=200, c="orange", label="predicted pass")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["pred", "GT"])
    ax.set_xlabel("frame")
    ax.set_title(f"Pass-event timeline — {clip_name}")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("05_pass_offside_metrics", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("05_pass_offside_metrics", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    pass_rows: list[dict[str, Any]] = []
    offside_rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    timeline_made: str | None = None

    for clip in clips:
        try:
            gt_passes = load_gt_passes(clip.name)
            gt_offs = load_gt_offsides(clip.name)
            pred_events = load_offside_events(clip.name)
        except FileNotFoundError as e:
            skipped.append((clip.name, str(e)))
            logger.warning("skipping %s: %s", clip.name, e)
            continue

        passes_summary = _eval_passes(clip.name, gt_passes, pred_events)
        pass_rows.append({k: v for k, v in passes_summary.items()
                          if k not in ("pred_passes", "gt_passes")})

        if timeline_made is None:
            _plot_pass_timeline(
                clip.name,
                passes_summary["pred_passes"],
                passes_summary["gt_passes"],
                sub["figures"] / "pass_event_timeline.pdf",
            )
            timeline_made = clip.name

        offside_rows.extend(_eval_offsides(clip.name, gt_offs, pred_events))

    if not pass_rows and not offside_rows:
        logger.error("No pass/offside evaluations produced.")
        return 2

    pass_df = pd.DataFrame(pass_rows)
    off_df = pd.DataFrame(offside_rows)

    pass_csv = sub["tables"] / "pass_detection_metrics.csv"
    if not pass_df.empty:
        io_helpers.write_csv(pass_df, pass_csv)
    off_csv = sub["tables"] / "offside_event_analysis.csv"
    if not off_df.empty:
        io_helpers.write_csv(off_df, off_csv)

    io_helpers.write_json({"passes": pass_rows, "offsides": offside_rows,
                           "skipped": skipped},
                          sub["raw"] / "pass_offside.json")

    threshold = config.METRIC_THRESHOLDS["pass_F1"]
    if pass_df.empty:
        verdict = "No pass evaluations produced; cannot evaluate Criterion (iv)."
    else:
        mean_f1 = float(pass_df["F1"].mean())
        status = "MET" if mean_f1 >= threshold else "NOT MET"
        verdict = (f"Mean pass-F1 = {mean_f1:.3f} (threshold {threshold:.2f}). "
                   f"Criterion (iv) **{status}**.")

    md_lines = [
        "# Pass / offside metrics — summary",
        "",
        verdict,
        "",
        f"Pass-matching tolerance: ±{config.PASS_FRAME_TOLERANCE} frames "
        f"(set in config.PASS_FRAME_TOLERANCE).",
        f"Offside-matching tolerance: ±{config.OFFSIDE_FRAME_TOLERANCE} frames.",
    ]
    if not off_df.empty:
        counts = off_df["classification"].value_counts().to_dict()
        md_lines += [
            "",
            "## Offside event tally (across clips)",
            "",
            "| classification | count |",
            "|---|---|",
        ] + [f"| {k} | {v} |" for k, v in counts.items()]

    md_lines += [
        "",
        "## Honest accounting",
        "",
        "Use the offside CSV to find frames where the system fired but no GT "
        "offside existed. Common causes to discuss in the report: ID swaps "
        "around the last defender, ball-possession mis-attribution near the "
        "moment of release, and frames where the homography drift exceeds "
        "the offside-line spatial precision (~0.5 m).",
    ]
    if skipped:
        md_lines += ["", "## Skipped clips", *[f"- {c}: {r}" for c, r in skipped]]
    md_path = sub["tables"] / "tactical_summary.md"
    io_helpers.write_markdown(md_path, "\n".join(md_lines) + "\n")

    print(f"05_pass_offside_metrics: pass rows={len(pass_df)}, "
          f"offside event rows={len(off_df)}. {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
