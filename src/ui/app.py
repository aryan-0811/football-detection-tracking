"""Streamlit UI for the football video analysis pipeline.

Runs the existing CLI (`python -m src.track_video_supervision`) in a subprocess;
this module contains no pipeline logic of its own.

Launch from the project root:
    streamlit run src/ui/app.py
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "input_videos"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL = "models/object_detection/best.pt"
DEFAULT_BALL_MODEL = "models/ball_detection/best.pt"


def list_inputs() -> list[Path]:
    if not INPUT_DIR.exists():
        return []
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"})


def snapshot_outputs() -> set[Path]:
    if not OUTPUT_DIR.exists():
        return set()
    return {p for p in OUTPUT_DIR.rglob("*") if p.is_file()}


def build_command(source: Path, opts: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.track_video_supervision",
        "--model", opts["model"],
        "--source", str(source),
        "--tracker", opts["tracker"],
        "--imgsz", str(opts["imgsz"]),
        "--conf", str(opts["conf"]),
        "--iou", str(opts["iou"]),
        "--team-device", opts["team_device"],
        "--team-smooth", str(opts["team_smooth"]),
        "--pitch-stride", str(opts["pitch_stride"]),
        "--kp-conf", str(opts["kp_conf"]),
        "--ball-model", opts["ball_model"],
        "--ball-conf", str(opts["ball_conf"]),
    ]
    if opts["max_frames"]:
        cmd += ["--max-frames", str(opts["max_frames"])]
    if opts["show_id"]:
        cmd.append("--show-id")
    if opts["birdeye"]:
        cmd.append("--birdeye")
    if opts["side_by_side"]:
        cmd.append("--side-by-side")
    if opts["pitch_debug"]:
        cmd.append("--pitch-debug")
    if opts["heatmaps"]:
        cmd.append("--save-heatmaps")
    if opts["offside"]:
        cmd.append("--offside")
    if opts["stats"]:
        cmd.append("--save-stats")
    return cmd


def run_and_stream(cmd: list[str], log_area) -> int:
    lines: list[str] = ["$ " + " ".join(shlex.quote(c) for c in cmd)]
    log_area.code("\n".join(lines), language="bash")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_area.code("\n".join(lines[-400:]), language="bash")
    return proc.wait()


# ------------------------------------------------------------------ UI
st.set_page_config(page_title="Football Analysis Pipeline", layout="wide")
st.title("Football Video Analysis")
st.caption("Thin wrapper around `python -m src.track_video_supervision`.")

with st.sidebar:
    st.header("Input")
    source_mode = st.radio("Source", ["Select existing", "Upload"], horizontal=True)

    source_path: Path | None = None
    if source_mode == "Select existing":
        inputs = list_inputs()
        if inputs:
            choice = st.selectbox("Video in input_videos/", [p.name for p in inputs])
            source_path = INPUT_DIR / choice
        else:
            st.warning(f"No videos in {INPUT_DIR}")
    else:
        up = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])
        if up is not None:
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            dest = INPUT_DIR / up.name
            dest.write_bytes(up.getbuffer())
            source_path = dest
            st.success(f"Saved to {dest.relative_to(PROJECT_ROOT)}")

    st.header("Mode")
    mode = st.radio("Run mode", ["preview (100 frames)", "full video"], index=0)
    max_frames = 100 if mode.startswith("preview") else 0

    st.header("Tracker & outputs")
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
    col1, col2 = st.columns(2)
    with col1:
        birdeye = st.checkbox("Bird's-eye", value=True)
        side_by_side = st.checkbox("Side-by-side", value=True)
        pitch_debug = st.checkbox("Pitch debug", value=False)
    with col2:
        heatmaps = st.checkbox("Heatmaps", value=True)
        stats = st.checkbox("Match stats", value=True)
        offside = st.checkbox("Offside", value=True)
    show_id = st.checkbox("Show track IDs", value=True)

    st.header("Thresholds")
    conf = st.slider("conf", 0.0, 1.0, 0.20, 0.01)
    iou = st.slider("iou", 0.0, 1.0, 0.50, 0.01)
    ball_conf = st.slider("ball_conf", 0.0, 1.0, 0.05, 0.01)
    pitch_stride = st.slider("pitch_stride", 1, 60, 15)
    kp_conf = st.slider("kp_conf", 0.0, 1.0, 0.50, 0.01)
    team_smooth = st.slider("team_smooth", 1, 120, 30)

    with st.expander("Advanced"):
        imgsz = st.number_input("imgsz", 320, 1920, 1280, 32)
        team_device = st.selectbox("team_device", ["cpu", "cuda"], index=0)
        model = st.text_input("YOLO model", DEFAULT_MODEL)
        ball_model = st.text_input("Ball model", DEFAULT_BALL_MODEL)

run_btn = st.button("Run pipeline", type="primary", disabled=source_path is None)

st.subheader("Logs")
log_area = st.empty()

if run_btn and source_path is not None:
    opts = dict(
        model=model, tracker=tracker, imgsz=imgsz, conf=conf, iou=iou,
        team_device=team_device, team_smooth=team_smooth,
        pitch_stride=pitch_stride, kp_conf=kp_conf,
        ball_model=ball_model, ball_conf=ball_conf,
        max_frames=max_frames, show_id=show_id,
        birdeye=birdeye, side_by_side=side_by_side, pitch_debug=pitch_debug,
        heatmaps=heatmaps, offside=offside, stats=stats,
    )
    cmd = build_command(source_path, opts)
    before = snapshot_outputs()
    with st.spinner("Running pipeline..."):
        rc = run_and_stream(cmd, log_area)
    if rc == 0:
        st.success("Pipeline completed.")
    else:
        st.error(f"Pipeline exited with code {rc}.")
    after = snapshot_outputs()
    st.session_state["new_outputs"] = sorted(p for p in (after - before))

def _render_stats(path: Path) -> None:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        st.warning(f"Could not parse {path.name}: {e}")
        return

    match = data.get("match_stats", {})
    players = data.get("player_stats", {})

    passes_by_team = match.get("passes_by_team", {}) or {}
    possession = match.get("possession_pct", {}) or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total passes", match.get("total_passes", 0))
    c2.metric("Team 0 passes", passes_by_team.get("team_0", 0))
    c3.metric("Team 1 passes", passes_by_team.get("team_1", 0))
    c4.metric("Team 0 possession", f"{possession.get('team_0', 0):.1f}%")
    c5.metric("Team 1 possession", f"{possession.get('team_1', 0):.1f}%")

    if players:
        rows = [
            {
                "track_id": int(tid),
                "team": ps.get("team_id"),
                "distance_m": ps.get("distance_m"),
                "avg_speed_kmh": ps.get("avg_speed_kmh"),
                "max_speed_kmh": ps.get("max_speed_kmh"),
            }
            for tid, ps in players.items()
        ]
        rows.sort(key=lambda r: (r["team"] if r["team"] is not None else 99, -(r["distance_m"] or 0)))
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_offside(path: Path) -> None:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        st.warning(f"Could not parse {path.name}: {e}")
        return
    passes = data.get("passes", []) or []
    offsides = data.get("offsides", []) or []
    c1, c2 = st.columns(2)
    c1.metric("Passes detected", len(passes))
    c2.metric("Offside events", len(offsides))
    if offsides:
        with st.expander(f"Offside events ({len(offsides)})"):
            st.json(offsides)
    if passes:
        with st.expander(f"Pass events ({len(passes)})"):
            st.json(passes)


_HEATMAP_RE = re.compile(r"track_(\d+)_team_(\d+)_samples_(\d+)\.png$")


def _render_heatmaps(pngs: list[Path]) -> None:
    parsed = []
    for p in pngs:
        m = _HEATMAP_RE.search(p.name)
        if m:
            parsed.append((int(m.group(3)), int(m.group(1)), int(m.group(2)), p))
        else:
            parsed.append((0, -1, -1, p))
    parsed.sort(key=lambda t: -t[0])

    cols_per_row = 3
    for i in range(0, len(parsed), cols_per_row):
        row = parsed[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (samples, tid, team, p) in zip(cols, row):
            with col:
                caption = f"Track {tid} · Team {team} · {samples} samples" if tid >= 0 else p.name
                st.image(str(p), caption=caption, use_container_width=True)


new_outputs = st.session_state.get("new_outputs", [])
if new_outputs:
    st.subheader("Results")

    videos = [p for p in new_outputs if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
    heatmap_pngs = [p for p in new_outputs if p.suffix.lower() == ".png" and "heatmap" in p.parent.name.lower()]
    stats_jsons = [p for p in new_outputs if p.suffix.lower() == ".json" and "stats" in p.name.lower()]
    offside_jsons = [p for p in new_outputs if p.suffix.lower() == ".json" and "offside" in p.name.lower()]
    accounted = set(videos) | set(heatmap_pngs) | set(stats_jsons) | set(offside_jsons)
    others = [p for p in new_outputs if p not in accounted]

    tabs = st.tabs(["Videos", "Heatmaps", "Match stats", "Offside", "Other files"])

    with tabs[0]:
        if not videos:
            st.info("No videos generated.")
        for v in videos:
            st.markdown(f"**{v.name}**  \n`{v.relative_to(PROJECT_ROOT)}`")
            try:
                st.video(str(v))
            except Exception as e:
                st.warning(f"Could not embed {v.name}: {e}")

    with tabs[1]:
        if heatmap_pngs:
            _render_heatmaps(heatmap_pngs)
        else:
            st.info("No heatmaps generated (enable the Heatmaps toggle).")

    with tabs[2]:
        if stats_jsons:
            for p in stats_jsons:
                st.caption(str(p.relative_to(PROJECT_ROOT)))
                _render_stats(p)
                st.download_button(
                    f"Download {p.name}", data=p.read_bytes(),
                    file_name=p.name, key=f"dl-{p}",
                )
        else:
            st.info("No match stats file (enable Match stats).")

    with tabs[3]:
        if offside_jsons:
            for p in offside_jsons:
                st.caption(str(p.relative_to(PROJECT_ROOT)))
                _render_offside(p)
                st.download_button(
                    f"Download {p.name}", data=p.read_bytes(),
                    file_name=p.name, key=f"dl-{p}",
                )
        else:
            st.info("No offside events file (enable Offside).")

    with tabs[4]:
        if others:
            for p in others:
                rel = p.relative_to(PROJECT_ROOT)
                try:
                    st.download_button(
                        label=f"Download {rel}", data=p.read_bytes(),
                        file_name=p.name, key=f"dl-other-{rel}",
                    )
                except Exception as e:
                    st.warning(f"Could not read {rel}: {e}")
        else:
            st.info("No other files.")
