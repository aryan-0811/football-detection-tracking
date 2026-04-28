"""
02_tracking_metrics
===================

Computes HOTA / MOTA / IDF1 for each tracker variant (ByteTrack, BoT-SORT)
against hand-labelled MOT ground truth, then aggregates across clips.

Serves: §3.5 forward reference and Success Criterion (ii) — HOTA >= 0.45.

Inputs
------
- data/ground_truth/<clip>/tracks_mot.txt
- data/pipeline_outputs/<clip>/tracks_<tracker>.txt
  (Add export per INSTRUMENTATION_TODO.md item 2.)

Implementation
--------------
Prefers `trackeval` (academic-standard HOTA from SoccerNet-Tracking) if
installed; otherwise falls back to `motmetrics`, which provides MOTA/IDF1 and a
HOTA-like score derived from DetA*AssA. The fallback is documented inline so
you can disclose it in the report.

Comparison context
------------------
A reference row is appended for the published SoccerNet-Tracking 2023 baseline
(see config.LITERATURE_BASELINES["tracking_soccernet_2023"]).

Outputs
-------
- results/tables/tracking_metrics.csv
- results/tables/tracking_summary.csv
- results/raw/tracking_metrics.json
- results/figures/tracking_hota_comparison.pdf
- results/tables/tracking_summary.md
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
    gt_dir, load_gt_tracks, load_predicted_tracks, pipeline_out_dir,
)

logger = logging.getLogger("02_tracking_metrics")

TRACKERS = ["bytetrack", "botsort"]


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _have_trackeval() -> bool:
    try:
        import trackeval  # noqa: F401
        return True
    except Exception:
        return False


def _have_motmetrics() -> bool:
    try:
        import motmetrics  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# motmetrics fallback (HOTA approx via sqrt(DetA*AssA))
# ---------------------------------------------------------------------------

def _eval_motmetrics(gt: pd.DataFrame, pred: pd.DataFrame,
                     iou_threshold: float = 0.5) -> dict[str, float]:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt["frame"]).union(pred["frame"]))

    for f in frames:
        g = gt[gt["frame"] == f]
        p = pred[pred["frame"] == f]
        gt_ids = g["id"].tolist()
        p_ids = p["id"].tolist()

        if len(gt_ids) == 0 and len(p_ids) == 0:
            acc.update([], [], [])
            continue

        gt_boxes = g[["bb_left", "bb_top", "bb_width", "bb_height"]].values
        p_boxes = p[["bb_left", "bb_top", "bb_width", "bb_height"]].values
        dist = mm.distances.iou_matrix(gt_boxes, p_boxes, max_iou=1 - iou_threshold)
        acc.update(gt_ids, p_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "mota", "motp", "idf1", "num_switches", "num_fragmentations",
        "num_unique_objects", "num_objects", "num_predictions",
        "idtp", "idfp", "idfn",
    ], name="run")

    motp = float(summary["motp"].iloc[0])
    det_a = 1.0 - min(1.0, motp) if not np.isnan(motp) else float("nan")
    idtp = float(summary["idtp"].iloc[0]); idfp = float(summary["idfp"].iloc[0])
    idfn = float(summary["idfn"].iloc[0])
    ass_a = idtp / (idtp + idfp + idfn) if (idtp + idfp + idfn) > 0 else float("nan")
    hota_approx = float(np.sqrt(det_a * ass_a)) if not np.isnan(det_a * ass_a) else float("nan")

    return {
        "HOTA": hota_approx,
        "DetA": float(det_a),
        "AssA": float(ass_a),
        "MOTA": float(summary["mota"].iloc[0]),
        "IDF1": float(summary["idf1"].iloc[0]),
        "IDsw": int(summary["num_switches"].iloc[0]),
        "Frag": int(summary["num_fragmentations"].iloc[0]),
        "_backend": "motmetrics_approx",
    }


def _eval_trackeval(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    """Compute HOTA via TrackEval. Implementation note: TrackEval expects
    files on disk in a strict directory layout; we write temporary files."""
    import tempfile, shutil
    import trackeval  # type: ignore

    tmp = Path(tempfile.mkdtemp(prefix="trackeval_"))
    try:
        gt_seq = tmp / "gt" / "MOT-eval" / "seq01"
        gt_seq.mkdir(parents=True)
        (gt_seq / "gt").mkdir()
        io_helpers.write_mot(gt, gt_seq / "gt" / "gt.txt")
        seqinfo = (
            "[Sequence]\nname=seq01\nimDir=img1\nframeRate=25\n"
            f"seqLength={int(gt['frame'].max())}\nimWidth=1920\nimHeight=1080\n"
        )
        (gt_seq / "seqinfo.ini").write_text(seqinfo)
        seq_list = tmp / "gt" / "MOT-eval" / "seqmaps" / "MOT-eval-train.txt"
        seq_list.parent.mkdir(parents=True, exist_ok=True)
        seq_list.write_text("name\nseq01\n")

        pred_dir = tmp / "trackers" / "MOT-eval" / "tracker" / "data"
        pred_dir.mkdir(parents=True)
        io_helpers.write_mot(pred, pred_dir / "seq01.txt")

        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg["DISPLAY_LESS_PROGRESS"] = True
        eval_cfg["PRINT_RESULTS"] = False
        eval_cfg["PRINT_CONFIG"] = False
        eval_cfg["OUTPUT_SUMMARY"] = False
        eval_cfg["LOG_ON_ERROR"] = None
        dataset_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_cfg.update({
            "GT_FOLDER": str(tmp / "gt" / "MOT-eval"),
            "TRACKERS_FOLDER": str(tmp / "trackers" / "MOT-eval"),
            "BENCHMARK": "MOT-eval",
            "SPLIT_TO_EVAL": "train",
            "TRACKERS_TO_EVAL": ["tracker"],
            "CLASSES_TO_EVAL": ["pedestrian"],
        })
        evaluator = trackeval.Evaluator(eval_cfg)
        ds = [trackeval.datasets.MotChallenge2DBox(dataset_cfg)]
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                   trackeval.metrics.Identity()]
        out, _ = evaluator.evaluate(ds, metrics)
        res = out["MotChallenge2DBox"]["tracker"]["seq01"]["pedestrian"]
        hota = float(np.mean(res["HOTA"]["HOTA"]))
        deta = float(np.mean(res["HOTA"]["DetA"]))
        assa = float(np.mean(res["HOTA"]["AssA"]))
        return {
            "HOTA": hota,
            "DetA": deta,
            "AssA": assa,
            "MOTA": float(res["CLEAR"]["MOTA"]),
            "IDF1": float(res["Identity"]["IDF1"]),
            "IDsw": int(res["CLEAR"]["IDSW"]),
            "Frag": int(res["CLEAR"]["Frag"]),
            "_backend": "trackeval",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _eval_pair(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    if _have_trackeval():
        try:
            return _eval_trackeval(gt, pred)
        except Exception as e:
            logger.warning("TrackEval failed (%s); falling back to motmetrics.", e)
    if _have_motmetrics():
        return _eval_motmetrics(gt, pred)
    raise RuntimeError(
        "Neither trackeval nor motmetrics is installed. "
        "Run: pip install py-motmetrics"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_hota(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    pivot = (df[df["is_reference"] == 0]
             .pivot(index="clip", columns="tracker", values="HOTA"))
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_ylabel("HOTA"); ax.set_xlabel("Clip")
    ax.set_ylim(0, 1)
    ax.axhline(config.METRIC_THRESHOLDS["tracking_HOTA"],
               color="red", ls="--", lw=1, label="threshold (0.45)")
    ax.set_title("HOTA per clip per tracker")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = make_parser("02_tracking_metrics", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("02_tracking_metrics", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    skipped: list[str] = []

    for clip in clips:
        try:
            gt = load_gt_tracks(clip.name)
        except FileNotFoundError as e:
            logger.warning("skipping clip '%s' because %s", clip.name, e)
            skipped.append(clip.name)
            continue

        for tracker in TRACKERS:
            try:
                pred = load_predicted_tracks(clip.name, tracker)
            except FileNotFoundError as e:
                logger.warning("skipping clip '%s' tracker '%s' (%s)",
                               clip.name, tracker, e)
                continue

            try:
                m = _eval_pair(gt, pred)
            except Exception as e:
                logger.error("evaluation failed for clip=%s tracker=%s: %s",
                             clip.name, tracker, e)
                continue

            row = {
                "clip": clip.name,
                "tracker": tracker,
                "is_reference": 0,
                **m,
            }
            rows.append(row)
            raw[f"{clip.name}/{tracker}"] = m

    # Reference row from the literature
    lit = config.LITERATURE_BASELINES["tracking_soccernet_2023"]
    rows.append({
        "clip": "literature",
        "tracker": "SoccerNet-2023",
        "is_reference": 1,
        "HOTA": lit["HOTA"], "DetA": float("nan"), "AssA": float("nan"),
        "MOTA": lit["MOTA"], "IDF1": lit["IDF1"], "IDsw": -1, "Frag": -1,
        "_backend": "literature",
    })

    if not rows:
        logger.error("No tracking evaluations produced. Check INSTRUMENTATION_TODO.md item 2.")
        return 2

    df = pd.DataFrame(rows)
    csv_path = sub["tables"] / "tracking_metrics.csv"
    io_helpers.write_csv(df, csv_path)

    # Aggregate (excluding reference rows)
    own = df[df["is_reference"] == 0]
    if not own.empty:
        agg = (own.groupby("tracker")[["HOTA", "MOTA", "IDF1"]]
               .agg(["mean", "std"]).round(4))
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index()
        io_helpers.write_csv(agg, sub["tables"] / "tracking_summary.csv")

    io_helpers.write_json({"rows": rows, "skipped": skipped}, sub["raw"] / "tracking_metrics.json")
    _plot_hota(df, sub["figures"] / "tracking_hota_comparison.pdf")

    # Verdict
    threshold = config.METRIC_THRESHOLDS["tracking_HOTA"]
    own_mean_hota = float(own["HOTA"].mean()) if not own.empty else float("nan")
    if np.isnan(own_mean_hota):
        verdict = "Insufficient data to evaluate Criterion (ii)."
    else:
        status = "MET" if own_mean_hota >= threshold else "NOT MET"
        verdict = (f"Mean HOTA across clips = {own_mean_hota:.3f} "
                   f"(threshold {threshold:.2f}). Criterion (ii) **{status}**.")

    md = (
        "# Tracking metrics — summary\n\n"
        f"{verdict}\n\n"
        f"Backend used: {own['_backend'].iloc[0] if not own.empty else 'n/a'}.\n\n"
        f"Reference row included: SoccerNet-2023 (HOTA={lit['HOTA']}, "
        f"MOTA={lit['MOTA']}, IDF1={lit['IDF1']}).\n\n"
        "## Honest discussion\n\n"
        "Update this section with: which tracker won on which clip, "
        "scenarios where each underperformed (occlusion clusters, fast "
        "camera pans, near-uniform team-strip frames), and whether the "
        "absolute HOTA gap to SoccerNet-2023 (which trains on much more "
        "data) is acceptable for the report's narrative.\n"
    )
    if skipped:
        md += "\n## Skipped clips\n" + "\n".join(f"- {c}" for c in skipped) + "\n"
    md_path = sub["tables"] / "tracking_summary.md"
    io_helpers.write_markdown(md_path, md)

    print(f"02_tracking_metrics: {len(own)} eval rows across "
          f"{own['clip'].nunique() if not own.empty else 0} clips. {verdict} "
          f"Tables: {csv_path.name}; figure: tracking_hota_comparison.pdf.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
