"""
03_team_classification_metrics
==============================

F1 / precision / recall for team classification, with five linked experiments:
raw per-frame F1, smoothed track-level F1, goalkeeper-resolver accuracy,
crop-strategy ablation (when an alternate prediction file is present), and a
UMAP scatter of the warm-up embeddings.

Serves: §3.6 forward reference and Success Criterion (iii) — F1 >= 0.85.

Inputs
------
Per clip:
- data/ground_truth/<clip>/team_labels.json          {track_id: team_id}
- data/pipeline_outputs/<clip>/team_predictions_raw.csv      (item 3)
- data/pipeline_outputs/<clip>/team_predictions_smoothed.csv (item 3)
- data/pipeline_outputs/<clip>/team_predictions_fullbbox.csv (optional, ablation)
- data/pipeline_outputs/<clip>/warmup_embeddings.npz         (item 6, optional)
- data/pipeline_outputs/<clip>/tracks_bytetrack.txt          (for GK class info,
                                                             optional)

CSV format:
    frame, track_id, team

Outputs
-------
- results/tables/team_classification_metrics.csv
- results/tables/team_classification_ablation.csv
- results/figures/warmup_umap.pdf
- results/figures/team_smoothing_effect.pdf
- results/tables/team_summary.md
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
    gt_dir, load_gt_team_labels, pipeline_out_dir,
)
from evaluation.utils.metrics_helpers import prf1, macro_f1, PRF1

logger = logging.getLogger("03_team_classification_metrics")


def _load_predictions(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    expected = {"frame", "track_id", "team"}
    if not expected.issubset(df.columns):
        logger.warning("%s missing columns %s", path, expected - set(df.columns))
        return None
    df["track_id"] = df["track_id"].astype(int)
    df["team"] = df["team"].astype(int)
    return df


def _per_class_prf1(y_true: list[int], y_pred: list[int]) -> dict[int, PRF1]:
    classes = sorted(set(y_true))
    out = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        out[c] = prf1(tp, fp, fn)
    return out


def _evaluate(condition: str, clip_name: str, pred_df: pd.DataFrame,
              gt_by_track: dict[int, int]) -> list[dict[str, Any]]:
    """Score predictions and emit per-class + macro rows for a condition."""
    if pred_df is None or pred_df.empty:
        return []

    # Restrict to tracks we have GT for
    mask = pred_df["track_id"].isin(gt_by_track.keys())
    df = pred_df[mask].copy()
    if df.empty:
        logger.warning("clip=%s condition=%s: no overlap between predicted and GT track IDs",
                       clip_name, condition)
        return []

    df["gt_team"] = df["track_id"].map(gt_by_track)

    rows: list[dict[str, Any]] = []
    per_class = _per_class_prf1(df["gt_team"].tolist(), df["team"].tolist())
    for cls, m in per_class.items():
        rows.append({
            "clip": clip_name, "condition": condition, "class": f"team_{cls}",
            "P": m.precision, "R": m.recall, "F1": m.f1, "support": m.tp + m.fn,
        })
    rows.append({
        "clip": clip_name, "condition": condition, "class": "macro",
        "P": float(np.mean([m.precision for m in per_class.values()])),
        "R": float(np.mean([m.recall for m in per_class.values()])),
        "F1": macro_f1(per_class.values()),
        "support": sum(m.tp + m.fn for m in per_class.values()),
    })
    return rows


def _track_level_majority(pred_df: pd.DataFrame) -> dict[int, int]:
    """For each track, return the majority-vote predicted team (final label)."""
    out = {}
    for tid, g in pred_df.groupby("track_id"):
        vals, counts = np.unique(g["team"].values, return_counts=True)
        out[int(tid)] = int(vals[int(np.argmax(counts))])
    return out


def _plot_smoothing_effect(clip_name: str, raw: pd.DataFrame,
                           smoothed: pd.DataFrame, out_path: Path) -> None:
    """Per-frame raw vs smoothed labels for a few representative tracks."""
    if raw is None or smoothed is None or raw.empty:
        return
    track_ids = raw["track_id"].value_counts().head(4).index.tolist()
    if not track_ids:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(track_ids), 1, figsize=(7, 1.8 * len(track_ids)),
                             sharex=True)
    if len(track_ids) == 1:
        axes = [axes]
    for ax, tid in zip(axes, track_ids):
        r = raw[raw["track_id"] == tid].sort_values("frame")
        s = smoothed[smoothed["track_id"] == tid].sort_values("frame")
        ax.step(r["frame"], r["team"], where="post", label="raw", alpha=0.7)
        ax.step(s["frame"], s["team"], where="post", label="smoothed",
                lw=2, alpha=0.9)
        ax.set_ylabel(f"track {tid}")
        ax.set_yticks(sorted(set(r["team"]) | set(s["team"])))
    axes[-1].set_xlabel("frame"); axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Smoothing effect (clip={clip_name})")
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def _plot_umap(clip_name: str, npz_path: Path, out_path: Path) -> bool:
    if not npz_path.exists():
        return False
    data = np.load(npz_path)
    emb = data.get("embeddings"); cls = data.get("cluster")
    centroids = data.get("centroids")
    if emb is None or cls is None or emb.ndim != 2 or emb.shape[1] < 2:
        return False
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=cls, cmap="tab10", s=10, alpha=0.8)
    if centroids is not None and centroids.ndim == 2 and centroids.shape[1] >= 2:
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=120,
                   c="black", edgecolors="white", linewidths=1, label="centroid")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Warm-up UMAP embeddings — {clip_name}")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("03_team_classification_metrics", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("03_team_classification_metrics", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    umap_made = False

    for clip in clips:
        try:
            gt = load_gt_team_labels(clip.name)
        except FileNotFoundError as e:
            skipped.append((clip.name, str(e)))
            logger.warning("skipping %s (%s)", clip.name, e)
            continue

        out_dir = pipeline_out_dir(clip.name)
        raw = _load_predictions(out_dir / "team_predictions_raw.csv")
        smoothed = _load_predictions(out_dir / "team_predictions_smoothed.csv")
        fullbbox = _load_predictions(out_dir / "team_predictions_fullbbox.csv")

        if raw is None and smoothed is None:
            skipped.append((clip.name, "no team_predictions_*.csv"))
            logger.warning("skipping %s: no team_predictions_raw/smoothed.csv", clip.name)
            continue

        if raw is not None:
            rows.extend(_evaluate("raw_per_frame", clip.name, raw, gt.by_track))
        if smoothed is not None:
            rows.extend(_evaluate("smoothed_per_frame", clip.name, smoothed, gt.by_track))
            # Track-level: take majority across the smoothed stream
            track_pred = _track_level_majority(smoothed)
            track_df = pd.DataFrame([
                {"frame": -1, "track_id": tid, "team": pred}
                for tid, pred in track_pred.items()
            ])
            rows.extend(_evaluate("smoothed_track_level", clip.name, track_df, gt.by_track))
        if fullbbox is not None:
            rows.extend(_evaluate("ablation_fullbbox_per_frame", clip.name, fullbbox, gt.by_track))

        # GK-only — if predictions include GK tracks (track IDs in GT marked as GK).
        # Without an explicit GK marker, this row is skipped; the user can label
        # GKs with team_id 99 in team_labels.json to opt in.
        gk_track_ids = [tid for tid, t in gt.by_track.items() if t in (90, 91, 92, 93, 94, 95)]
        if gk_track_ids and smoothed is not None:
            gk_df = smoothed[smoothed["track_id"].isin(gk_track_ids)]
            if not gk_df.empty:
                rows.extend(_evaluate("gk_only", clip.name, gk_df,
                                      {tid: gt.by_track[tid] for tid in gk_track_ids}))

        # Plots
        if raw is not None and smoothed is not None:
            _plot_smoothing_effect(
                clip.name, raw, smoothed,
                sub["figures"] / f"team_smoothing_effect_{clip.name}.pdf",
            )

        if not umap_made:
            if _plot_umap(clip.name, out_dir / "warmup_embeddings.npz",
                          sub["figures"] / "warmup_umap.pdf"):
                umap_made = True

    if not rows:
        logger.error("No team-classification rows produced. See INSTRUMENTATION_TODO.md item 3.")
        return 2

    df = pd.DataFrame(rows)
    main_csv = sub["tables"] / "team_classification_metrics.csv"
    io_helpers.write_csv(df, main_csv)

    # Ablation pivot: condition vs F1 macro per clip
    macro = df[df["class"] == "macro"]
    abl = macro.pivot_table(index="clip", columns="condition", values="F1").reset_index()
    io_helpers.write_csv(abl, sub["tables"] / "team_classification_ablation.csv")

    io_helpers.write_json({"rows": rows, "skipped": skipped},
                          sub["raw"] / "team_classification.json")

    # Reference row → markdown summary
    threshold = config.METRIC_THRESHOLDS["team_F1"]
    smoothed_macro = macro[macro["condition"] == "smoothed_track_level"]
    if smoothed_macro.empty:
        verdict = "Smoothed track-level F1 not produced — cannot evaluate Criterion (iii)."
    else:
        mean_f1 = float(smoothed_macro["F1"].mean())
        status = "MET" if mean_f1 >= threshold else "NOT MET"
        verdict = (f"Mean smoothed track-level macro-F1 = {mean_f1:.3f} "
                   f"(threshold {threshold:.2f}). Criterion (iii) **{status}**.")

    lit = config.LITERATURE_BASELINES["team_istasse_2019"]
    md = (
        "# Team-classification metrics — summary\n\n"
        f"{verdict}\n\n"
        f"Reference: {lit['source']} colour-histogram baseline reports F1 ≈ {lit['F1']}.\n\n"
        "## Smoothing benefit\n\n"
        "The ablation table compares raw per-frame, smoothed per-frame, and "
        "smoothed track-level predictions. Discuss in the report whether "
        "smoothing's gain comes mostly from majority-vote across noisy frames "
        "or from suppressing transient mis-classifications during occlusion.\n\n"
        f"{'UMAP scatter saved.' if umap_made else 'No warmup_embeddings.npz found in any clip — UMAP scatter skipped.'}\n"
    )
    if skipped:
        md += "\n## Skipped clips\n" + "\n".join(f"- {c}: {r}" for c, r in skipped) + "\n"
    md_path = sub["tables"] / "team_summary.md"
    io_helpers.write_markdown(md_path, md)

    print(f"03_team_classification_metrics: {len(df)} rows, "
          f"ablation table written. {verdict} "
          f"UMAP plot: {'yes' if umap_made else 'skipped'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
