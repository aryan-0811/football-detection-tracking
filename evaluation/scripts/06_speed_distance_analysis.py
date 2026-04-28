"""
06_speed_distance_analysis
==========================

Validates the 40 km/h speed cap and characterises the per-track speed/distance
distribution.

Serves: §3.8 forward reference about speed cap activation.

Inputs
------
Per clip:
- data/pipeline_outputs/<clip>/<clip>_stats.json       (already produced)
- data/pipeline_outputs/<clip>/speeds.csv              (NEW — INSTRUMENTATION_TODO.md item 5)

Stats schema (assumed):
    {
      "per_player": {"<track_id>": {"max_speed_kmh": float, "avg_speed_kmh": float,
                                     "distance_m": float, "n_frames": int, ...}, ...},
      "match": {...}
    }

speeds.csv schema:
    frame, track_id, speed_kmh, capped

Analyses
--------
1. Distribution of max_speed_kmh across tracks; count at-cap tracks.
2. For tracks at cap, fraction of frames where the cap fired (from speeds.csv).
3. Per-track speed histograms for representative tracks.
4. Plausibility: how many tracks have avg speed in 7–13 km/h?

Outputs
-------
- results/tables/speed_distance_analysis.csv
- results/tables/speed_cap_activation_summary.csv
- results/figures/speed_distribution.pdf
- results/figures/cap_activation_per_track.pdf
- results/tables/speed_summary.md
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
from evaluation.utils.ground_truth import load_stats, pipeline_out_dir

logger = logging.getLogger("06_speed_distance_analysis")

CAP = config.PIPELINE_HYPERPARAMS["speed_cap_kmh"]
PLAUSIBLE_AVG_KMH_LO = 7.0
PLAUSIBLE_AVG_KMH_HI = 13.0
AT_CAP_TOL = 0.5  # km/h slack — anything within this is "at cap"


def _per_player_table(stats: dict[str, Any], clip_name: str) -> pd.DataFrame:
    pp = stats.get("per_player", stats.get("players", {}))
    if not isinstance(pp, dict) or not pp:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tid, blob in pp.items():
        try:
            tid_i = int(tid)
        except ValueError:
            continue
        if not isinstance(blob, dict):
            continue
        max_s = float(blob.get("max_speed_kmh", float("nan")))
        avg_s = float(blob.get("avg_speed_kmh", float("nan")))
        dist = float(blob.get("distance_m", blob.get("total_distance_m", float("nan"))))
        n_frames = int(blob.get("n_frames", -1))
        rows.append({
            "clip": clip_name, "track_id": tid_i,
            "max_speed_kmh": max_s, "avg_speed_kmh": avg_s,
            "distance_m": dist, "n_frames": n_frames,
            "at_cap": int(max_s >= CAP - AT_CAP_TOL) if not np.isnan(max_s) else 0,
            "plausible_avg": int(PLAUSIBLE_AVG_KMH_LO <= avg_s <= PLAUSIBLE_AVG_KMH_HI)
                            if not np.isnan(avg_s) else 0,
        })
    return pd.DataFrame(rows)


def _per_frame_speeds(clip_name: str) -> pd.DataFrame | None:
    p = pipeline_out_dir(clip_name) / "speeds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if not {"frame", "track_id", "speed_kmh", "capped"}.issubset(df.columns):
        logger.warning("speeds.csv missing columns: %s", p)
        return None
    return df


def _cap_activation_table(speeds: pd.DataFrame, clip_name: str) -> pd.DataFrame:
    if speeds is None or speeds.empty:
        return pd.DataFrame()
    g = speeds.groupby("track_id")
    rows = []
    for tid, sub in g:
        n = len(sub)
        n_cap = int(sub["capped"].sum())
        rows.append({
            "clip": clip_name,
            "track_id": int(tid),
            "n_frames": n,
            "n_cap_frames": n_cap,
            "frac_cap": n_cap / n if n > 0 else 0.0,
            "max_pre_cap_kmh": float(sub["speed_kmh"].max()) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot_speed_distribution(per_player: pd.DataFrame, out_path: Path) -> None:
    if per_player.empty:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_player["max_speed_kmh"].dropna(), bins=20,
            alpha=0.6, label="max", edgecolor="black")
    ax.hist(per_player["avg_speed_kmh"].dropna(), bins=20,
            alpha=0.6, label="avg", edgecolor="black")
    ax.axvline(CAP, color="red", ls="--", lw=1, label=f"cap ({CAP} km/h)")
    ax.set_xlabel("speed (km/h)"); ax.set_ylabel("number of tracks")
    ax.set_title("Per-track max and average speed distribution")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def _plot_cap_activation(activation: pd.DataFrame, out_path: Path) -> None:
    if activation.empty:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(activation["frac_cap"], bins=20, edgecolor="black")
    ax.set_xlabel("fraction of frames at cap")
    ax.set_ylabel("number of tracks")
    ax.set_title("How often is the speed cap binding per track?")
    fig.savefig(out_path); fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser("06_speed_distance_analysis", __doc__ or "")
    args = parser.parse_args(argv)
    apply_plot_style()
    sub = resolve_subdirs(args.output_dir)
    configure_logging("06_speed_distance_analysis", args.log_level, args.output_dir)

    clips = select_clips(args.clips)
    per_player_all: list[pd.DataFrame] = []
    activation_all: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for clip in clips:
        try:
            stats = load_stats(clip.name)
        except FileNotFoundError as e:
            skipped.append((clip.name, str(e)))
            logger.warning("skipping %s: %s", clip.name, e)
            continue
        pp = _per_player_table(stats, clip.name)
        if pp.empty:
            skipped.append((clip.name, "stats.json had empty per_player"))
            continue
        per_player_all.append(pp)

        speeds = _per_frame_speeds(clip.name)
        if speeds is not None:
            activation_all.append(_cap_activation_table(speeds, clip.name))
        else:
            logger.warning("clip=%s: speeds.csv missing — cap-activation per frame not computed.",
                           clip.name)

    if not per_player_all:
        logger.error("No per-player stats produced. Check pipeline_outputs/*/<clip>_stats.json.")
        return 2

    pp_df = pd.concat(per_player_all, ignore_index=True)
    io_helpers.write_csv(pp_df, sub["tables"] / "speed_distance_analysis.csv")

    if activation_all:
        act_df = pd.concat(activation_all, ignore_index=True)
        io_helpers.write_csv(act_df, sub["tables"] / "speed_cap_activation_summary.csv")
    else:
        act_df = pd.DataFrame()

    _plot_speed_distribution(pp_df, sub["figures"] / "speed_distribution.pdf")
    if not act_df.empty:
        _plot_cap_activation(act_df, sub["figures"] / "cap_activation_per_track.pdf")

    n_total = len(pp_df)
    n_at_cap = int(pp_df["at_cap"].sum())
    n_plausible = int(pp_df["plausible_avg"].sum())
    pct_at_cap = 100.0 * n_at_cap / n_total if n_total else 0.0
    pct_plausible = 100.0 * n_plausible / n_total if n_total else 0.0

    md_lines = [
        "# Speed / distance analysis — summary",
        "",
        f"Speed cap: **{CAP} km/h**.",
        f"Tracks total: {n_total}.",
        f"Tracks reaching the cap (max ≥ {CAP - AT_CAP_TOL} km/h): "
        f"{n_at_cap} ({pct_at_cap:.1f} %).",
        f"Tracks with average speed in [{PLAUSIBLE_AVG_KMH_LO}, {PLAUSIBLE_AVG_KMH_HI}] km/h: "
        f"{n_plausible} ({pct_plausible:.1f} %).",
    ]
    if not act_df.empty:
        med_frac = float(act_df["frac_cap"].median())
        max_frac = float(act_df["frac_cap"].max())
        md_lines += [
            "",
            f"Per-frame cap activation (from speeds.csv): "
            f"median fraction-at-cap per track = {med_frac:.4f}; "
            f"max = {max_frac:.4f}.",
        ]
    else:
        md_lines += [
            "",
            "speeds.csv was missing for every clip — per-frame cap "
            "activation not computed (see INSTRUMENTATION_TODO.md item 5).",
        ]
    md_lines += [
        "",
        "## Verdict on cap appropriateness",
        "",
        "If a substantial number of tracks repeatedly hit the cap, the cap "
        "is biting and should be discussed: is it suppressing real top-end "
        "sprints, or catching homography-induced jitter? Inspect the "
        "highest-frac_cap tracks visually before defending the choice.",
    ]
    if skipped:
        md_lines += ["", "## Skipped", *[f"- {c}: {r}" for c, r in skipped]]
    md_path = sub["tables"] / "speed_summary.md"
    io_helpers.write_markdown(md_path, "\n".join(md_lines) + "\n")

    print(f"06_speed_distance_analysis: {n_total} tracks across "
          f"{pp_df['clip'].nunique()} clips. "
          f"{n_at_cap} at-cap, {n_plausible} plausible. "
          f"Tables in {sub['tables'].name}/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
