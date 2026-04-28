"""
04_homography_metrics
=====================

Reprojection-error analysis for the pitch homography. Includes the kp_16
ablation, the stride caching ablation, and a per-keypoint contribution study.

Serves: §3.7 forward references (kp_16 outlier impact, stride caching ablation).

Inputs
------
Per clip:
- data/ground_truth/<clip>/homography_keypoints.json
    list of {frame, image_xy: [x, y], pitch_xy: [x, y], label?}
- data/pipeline_outputs/<clip>/homographies.npz
    arrays: frames (F,), H (F, 3, 3)   — see INSTRUMENTATION_TODO.md item 4

Method
------
1. For each GT keypoint at frame f, look up the pipeline's homography for the
   nearest cached frame (carry-forward semantics) and project the image point
   into pitch space. The pitch coordinate system used by the pipeline is the
   105 m × 68 m FIFA pitch (config.PIPELINE_HYPERPARAMS["pitch_dimensions_m"]).
   Errors are reported in metres and in equivalent pixels (back-projected
   through pitch-to-image scaling assuming a 1080-px display, see code).

2. kp_16 ablation: re-fit a homography per frame with kp_16 included vs
   excluded from the correspondence set, by treating one of the GT keypoints
   labelled "kp_16" as an extra control point. Requires at least 4 GT points
   per frame; clips that don't satisfy this constraint produce a clear note
   in the markdown summary.

3. Stride caching ablation: compare reprojection error at stride=1 (per-frame
   inference) vs the stride used in the run (default 15). The pipeline output
   reflects whatever stride was used during inference; this script can only
   *simulate* a stride-1 condition if a separate `homographies_stride_1.npz`
   is present. Otherwise it documents the gap.

Outputs
-------
- results/tables/homography_reprojection_error.csv
- results/figures/reprojection_error_distribution.pdf
- results/figures/per_keypoint_contribution.pdf
- results/raw/homography_metrics.json
- results/tables/homography_summary.md
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
    load_gt_homography_keypoints, pipeline_out_dir,
)
from evaluation.utils.metrics_helpers import reprojection_errors_px, summary_stats

logger = logging.getLogger("04_homography_metrics")


def _load_homographies(clip_name: str, suffix: str = "") -> tuple[np.ndarray, np.ndarray] | None:
    fname = f"homographies{suffix}.npz" if suffix else "homographies.npz"
    p = pipeline_out_dir(clip_name) / fname
    if not p.exists():
        return None
    data = np.load(p)
    frames = np.asarray(data["frames"], dtype=np.int64)
    H = np.asarray(data["H"], dtype=np.float64)
    if H.ndim != 3 or H.shape[1:] != (3, 3) or H.shape[0] != frames.shape[0]:
        logger.warning("Malformed homographies file %s", p)
        return None
    return frames, H


def _carry_forward(frames: np.ndarray, H: np.ndarray, target: int) -> np.ndarray | None:
    """Return the most recent H whose frame <= target."""
    if frames.size == 0:
        return None
    mask = frames <= target
    if not mask.any():
        return H[0]  # carry the first one back if nothing earlier
    idx = int(np.where(mask)[0][-1])
    return H[idx]


def _project_image_to_pitch(image_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.concatenate([image_xy, np.ones((image_xy.shape[0], 1))], axis=1)
    proj = (H @ pts.T).T
    w = proj[:, 2:3]
    proj_xy = proj[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)
    return proj_xy


def _refit_homography(image_pts: np.ndarray, pitch_pts: np.ndarray) -> np.ndarray | None:
    """DLT homography. Requires >= 4 correspondences."""
    if image_pts.shape[0] < 4:
        return None
    try:
        import cv2
        H, _ = cv2.findHomography(image_pts, pitch_pts, method=0)
        return H
    except ImportError:
        # Pure-NumPy DLT fallback.
        A = []
        for (x, y), (X, Y) in zip(image_pts, pitch_pts):
            A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
            A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
        A = np.asarray(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        return H / H[2, 2]


def _evaluate_per_frame(kpts: list, frames_arr: np.ndarray,
                        H_arr: np.ndarray, condition_name: str,
                        clip_name: str, exclude_kp_16: bool = False
                        ) -> tuple[pd.DataFrame, list[float]]:
    """Returns (per-keypoint dataframe, per-frame mean error list)."""
    pitch_w_m, pitch_h_m = config.PIPELINE_HYPERPARAMS["pitch_dimensions_m"]
    pitch_diag = float(np.hypot(pitch_w_m, pitch_h_m))

    by_frame: dict[int, list[Any]] = {}
    for kp in kpts:
        if exclude_kp_16 and kp.label == "kp_16":
            continue
        by_frame.setdefault(kp.frame, []).append(kp)

    rows: list[dict[str, Any]] = []
    per_frame_means: list[float] = []
    for frame, kp_list in sorted(by_frame.items()):
        H = _carry_forward(frames_arr, H_arr, frame)
        if H is None:
            continue
        img = np.array([k.image_xy for k in kp_list], dtype=float)
        gt_pitch = np.array([k.pitch_xy for k in kp_list], dtype=float)
        proj = _project_image_to_pitch(img, H)
        errs_m = np.linalg.norm(proj - gt_pitch, axis=1)
        per_frame_means.append(float(np.mean(errs_m)))
        for kp, e in zip(kp_list, errs_m):
            rows.append({
                "clip": clip_name,
                "condition": condition_name,
                "frame": frame,
                "label": kp.label or "unlabelled",
                "error_m": float(e),
                "error_pct_diag": float(e / pitch_diag * 100.0),
            })
    return pd.DataFrame(rows), per_frame_means


def _aggregate(df: pd.DataFrame, condition: str, clip: str) -> dict[str, Any]:
    sub = df[(df["condition"] == condition) & (df["clip"] == clip)]
    if sub.empty:
        return {"clip": clip, "condition": condition,
                "mean_error_m": float("nan"), "p95_error_m": float("nan"),
                "mean_error_cm": float("nan"), "p95_error_cm": float("nan"),
                "n": 0}
    e_m = sub["error_m"].values
    return {
        "clip": clip,
        "condition": condition,
        "mean_error_m": float(np.mean(e_m)),
        "p95_error_m": float(np.percentile(e_m, 95)),
        "mean_error_cm": float(np.mean(e_m) * 100.0),
        "p95_error_cm": float(np.percentile(e_m, 95) * 100.0),
        "n": int(e_m.size),
    }


def _plot_distribution(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, g in df.groupby("condition"):
        ax.hist(g["error_m"], bins=40, alpha=0.55, label=cond, edgecolor="black")
    ax.set_xlabel("Reprojection error (m)"); ax.set_ylabel("count")
    ax.set_title("Reprojection-error distribution")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def _plot_per_keypoint(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    import matplotlib.pyplot as plt
    g = df.groupby("label")["error_m"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(g))))
    ax.barh(g.index.astype(str), g.values, edgecolor="black")
    ax.set_xlabel("Mean reprojection error (m)")
    ax.set_title("Per-keypoint contribution (lower = more reliable)")
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("04_homography_metrics", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("04_homography_metrics", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    all_kp_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []

    for clip in clips:
        try:
            kpts = load_gt_homography_keypoints(clip.name)
        except FileNotFoundError as e:
            skipped.append((clip.name, str(e)))
            logger.warning("skipping %s: %s", clip.name, e)
            continue

        H_pack = _load_homographies(clip.name, suffix="")
        if H_pack is None:
            skipped.append((clip.name, "homographies.npz missing"))
            logger.warning("skipping %s: homographies.npz not found "
                           "(see INSTRUMENTATION_TODO.md item 4).", clip.name)
            continue
        frames_arr, H_arr = H_pack

        # (A) Pipeline-default condition (i.e. whatever stride / kp_16 logic the run used)
        df_default, _ = _evaluate_per_frame(kpts, frames_arr, H_arr,
                                            condition_name="pipeline_default",
                                            clip_name=clip.name)
        if not df_default.empty:
            all_kp_rows.append(df_default)
            summary_rows.append(_aggregate(df_default, "pipeline_default", clip.name))

        # (B) kp_16 ablation: re-fit per frame using GT correspondences with/without kp_16
        # This is the cleanest expression of the kp_16 outlier question.
        for cond_name, exclude in (("with_kp16", False), ("without_kp16", True)):
            rows: list[dict[str, Any]] = []
            by_frame: dict[int, list] = {}
            for kp in kpts:
                if exclude and kp.label == "kp_16":
                    continue
                by_frame.setdefault(kp.frame, []).append(kp)
            for frame, kp_list in by_frame.items():
                if len(kp_list) < 5:  # need extras for refit + held-out test
                    continue
                # Leave-one-out evaluation: refit on N-1 points, test on held-out.
                img_all = np.array([k.image_xy for k in kp_list], dtype=float)
                pitch_all = np.array([k.pitch_xy for k in kp_list], dtype=float)
                for i, kp in enumerate(kp_list):
                    img_fit = np.delete(img_all, i, axis=0)
                    pitch_fit = np.delete(pitch_all, i, axis=0)
                    H = _refit_homography(img_fit, pitch_fit)
                    if H is None:
                        continue
                    proj = _project_image_to_pitch(img_all[i:i + 1], H)
                    err = float(np.linalg.norm(proj[0] - pitch_all[i]))
                    rows.append({
                        "clip": clip.name, "condition": cond_name, "frame": frame,
                        "label": kp.label or "unlabelled",
                        "error_m": err,
                        "error_pct_diag": err / float(np.hypot(*config.PIPELINE_HYPERPARAMS["pitch_dimensions_m"])) * 100.0,
                    })
            if rows:
                df_abl = pd.DataFrame(rows)
                all_kp_rows.append(df_abl)
                summary_rows.append(_aggregate(df_abl, cond_name, clip.name))
            else:
                notes.append(f"{clip.name}: insufficient keypoints per frame "
                             f"for the kp_16 ablation (need >= 5).")

        # (C) Stride ablation
        H_pack_s1 = _load_homographies(clip.name, suffix="_stride_1")
        if H_pack_s1 is not None:
            frames_s1, H_s1 = H_pack_s1
            df_s1, _ = _evaluate_per_frame(kpts, frames_s1, H_s1,
                                           condition_name="stride_1",
                                           clip_name=clip.name)
            if not df_s1.empty:
                all_kp_rows.append(df_s1)
                summary_rows.append(_aggregate(df_s1, "stride_1", clip.name))
                # stride_15 row from the default file (pipeline_default is stride_15 in practice)
                summary_rows.append(_aggregate(df_default, "stride_15", clip.name)
                                    if not df_default.empty else None)
        else:
            notes.append(f"{clip.name}: stride-1 homographies missing — "
                         "stride caching ablation not run for this clip.")

    summary_rows = [r for r in summary_rows if r]
    if not summary_rows:
        logger.error("No homography evaluations produced. See INSTRUMENTATION_TODO.md item 4.")
        return 2

    summary_df = pd.DataFrame(summary_rows)
    csv_path = sub["tables"] / "homography_reprojection_error.csv"
    io_helpers.write_csv(summary_df, csv_path)

    big_df = pd.concat(all_kp_rows, ignore_index=True) if all_kp_rows else pd.DataFrame()
    io_helpers.write_json({"summary": summary_rows, "notes": notes,
                           "n_per_keypoint_rows": len(big_df),
                           "skipped": skipped},
                          sub["raw"] / "homography_metrics.json")

    if not big_df.empty:
        _plot_distribution(big_df, sub["figures"] / "reprojection_error_distribution.pdf")
        _plot_per_keypoint(big_df, sub["figures"] / "per_keypoint_contribution.pdf")

    # Reference rows from literature for the markdown
    theiner = config.LITERATURE_BASELINES["homography_theiner_2022"]
    tvcalib = config.LITERATURE_BASELINES["homography_tvcalib_2023"]

    md_lines = [
        "# Homography reprojection-error — summary",
        "",
        "Mean error per condition (across clips):",
        "",
        "| condition | mean_m | p95_m | n |",
        "|---|---|---|---|",
    ]
    agg = (summary_df.groupby("condition")[["mean_error_m", "p95_error_m", "n"]]
           .agg({"mean_error_m": "mean", "p95_error_m": "mean", "n": "sum"}))
    for cond, row in agg.iterrows():
        md_lines.append(
            f"| {cond} | {row['mean_error_m']:.3f} | {row['p95_error_m']:.3f} | {int(row['n'])} |"
        )
    md_lines += [
        "",
        f"Reference: {theiner['source']} ≈ {theiner['mean_error_m']} m mean.",
        f"Reference: {tvcalib['source']} ≈ {tvcalib['mean_error_m']} m mean.",
        "",
        "## kp_16 impact",
        "",
        "Compare the `with_kp16` and `without_kp16` rows above. If excluding "
        "kp_16 reduces error materially, this confirms the chapter's "
        "narrative that kp_16 is an outlier. Otherwise, soften the claim.",
        "",
        "## Stride caching trade-off",
        "",
        "The pipeline runs at stride=15 for inference cost. Compare `stride_1` "
        "vs `stride_15` (or `pipeline_default`) above. The runtime saving "
        "is reported in `runtime_profile.csv` (script 07).",
    ]
    if notes:
        md_lines += ["", "## Notes", *[f"- {n}" for n in notes]]
    md_path = sub["tables"] / "homography_summary.md"
    io_helpers.write_markdown(md_path, "\n".join(md_lines) + "\n")

    print(f"04_homography_metrics: {len(summary_df)} summary rows, "
          f"{len(big_df)} per-keypoint rows. "
          f"See {csv_path.name} and {md_path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
