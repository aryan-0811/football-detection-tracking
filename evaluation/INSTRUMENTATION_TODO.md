# Instrumentation TODO — additions required in `src/`

The evaluation harness assumes six debug-mode dumps that the current pipeline does not emit. Add the following to `src/pipeline/run.py` (search anchors given as comments). Each new file must land under `evaluation/data/pipeline_outputs/<clip_name>/`.

For all items: gate the dump behind a CLI flag that defaults to **off**, so production runs are not slowed down. Add the flag in `src/track_video_supervision.py` and thread it through to `run_pipeline()`.

```python
# Suggested CLI flags (add in track_video_supervision.py):
parser.add_argument("--debug-eval-dir", type=Path, default=None,
                    help="If set, dump per-frame eval artefacts to this dir.")
parser.add_argument("--debug-tracker-name", type=str, default="bytetrack",
                    help="Tag used for tracks_<name>.txt export.")
```

The directory you pass must equal `evaluation/data/pipeline_outputs/<clip_name>/`.

---

## (1) Per-stage timings — `timings.csv`

**Format:** one row per frame per stage.

```
frame, stage, ms
0, detection, 12.4
0, tracking, 1.8
0, team_predict, 4.0
...
```

**Where to add:** in `run_pipeline()` (search for `for frame_idx, frame in enumerate(...)` or your tracking loop).

```python
# Add at top of run.py:
import csv, time
_DEBUG_TIMINGS_FILE = None  # set up at start of run_pipeline if --debug-eval-dir

# In run_pipeline (after argument unpack):
if debug_eval_dir is not None:
    debug_eval_dir.mkdir(parents=True, exist_ok=True)
    _DEBUG_TIMINGS_FILE = (debug_eval_dir / "timings.csv").open("w", newline="")
    _timings_writer = csv.writer(_DEBUG_TIMINGS_FILE)
    _timings_writer.writerow(["frame", "stage", "ms"])

# Wrap each stage in the per-frame loop:
def _timed(stage: str, frame_idx: int, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if _DEBUG_TIMINGS_FILE is not None:
        _timings_writer.writerow([frame_idx, stage, (time.perf_counter() - t0) * 1000.0])
    return out

# Stages to instrument (every call site in the loop):
detections = _timed("detection", frame_idx, detector, frame)
tracked = _timed("tracking", frame_idx, tracker.update_with_detections, detections)
teams = _timed("team_predict", frame_idx, team_assigner.predict, frame, players)
ball = _timed("ball_detect", frame_idx, ball_detector, frame)
keypoints = _timed("pitch_keypoints", frame_idx, pitch_module.update, frame_idx, frame)
annotated = _timed("annotate", frame_idx, annotate, frame, ...)

# At end of run_pipeline:
if _DEBUG_TIMINGS_FILE is not None:
    _DEBUG_TIMINGS_FILE.close()
```

---

## (2) Track export in MOT format — `tracks_<tracker>.txt`

**Format:** MOT-Challenge (frame, id, bb_left, bb_top, bb_w, bb_h, conf, -1, -1, -1).

**Where to add:** alongside the existing tracking call (search for `tracker.update_with_detections` or wherever tracked detections are produced).

```python
# At start of run_pipeline (after debug_eval_dir setup):
_TRACKS_FILE = None
if debug_eval_dir is not None:
    name = debug_tracker_name or "tracker"
    _TRACKS_FILE = (debug_eval_dir / f"tracks_{name}.txt").open("w")

# Each frame, after tracker.update_with_detections gives back `tracked`:
if _TRACKS_FILE is not None and len(tracked) > 0:
    for det_idx in range(len(tracked)):
        x1, y1, x2, y2 = tracked.xyxy[det_idx]
        tid = int(tracked.tracker_id[det_idx])
        cls = int(tracked.class_id[det_idx]) if tracked.class_id is not None else -1
        # MOT eval uses player + GK only; skip ball/referee for the tracking eval
        if cls in (PLAYER_ID, GOALKEEPER_ID):
            conf = float(tracked.confidence[det_idx]) if tracked.confidence is not None else 1.0
            _TRACKS_FILE.write(
                f"{frame_idx + 1},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                f"{conf:.4f},-1,-1,-1\n"
            )

# At end:
if _TRACKS_FILE is not None: _TRACKS_FILE.close()
```

To produce both `tracks_bytetrack.txt` and `tracks_botsort.txt`, run the pipeline twice with `--tracker bytetrack` and `--tracker botsort` respectively, and `--debug-tracker-name` matching.

---

## (3) Per-frame team predictions — `team_predictions_raw.csv` and `team_predictions_smoothed.csv`

**Format:**

```
frame, track_id, raw_team, smoothed_team
```

**Where to add:** inside `src/team/assigner.py` `TeamAssigner`, expose both raw and smoothed predictions per frame; in the loop in `run_pipeline()` write them out.

```python
# In TeamAssigner.predict(...) return both raw and smoothed:
def predict(self, frame, players):
    raw = self._classify_each(frame, players)
    smoothed = self._apply_smoothing(raw, players.tracker_id)
    return raw, smoothed

# In run_pipeline, after the team predict call:
if debug_eval_dir is not None:
    if _TEAM_RAW_F is None:
        _TEAM_RAW_F = (debug_eval_dir / "team_predictions_raw.csv").open("w", newline="")
        _TEAM_SMOOTH_F = (debug_eval_dir / "team_predictions_smoothed.csv").open("w", newline="")
        _team_raw_w = csv.writer(_TEAM_RAW_F); _team_raw_w.writerow(["frame", "track_id", "team"])
        _team_smooth_w = csv.writer(_TEAM_SMOOTH_F); _team_smooth_w.writerow(["frame", "track_id", "team"])

    for tid, t_raw, t_smooth in zip(players.tracker_id, raw, smoothed):
        _team_raw_w.writerow([frame_idx, int(tid), int(t_raw)])
        _team_smooth_w.writerow([frame_idx, int(tid), int(t_smooth)])
```

---

## (4) Per-frame homography — `homographies.npz`

**Format:** an NPZ archive with two arrays:
- `frames`: int array (F,)
- `H`: float array (F, 3, 3)

**Where to add:** wherever the `ViewTransformer` is updated (search for `ViewTransformer(` or the pitch module's stride-based recompute).

```python
# At start:
_H_FRAMES, _H_LIST = [], []

# After every ViewTransformer fit:
if debug_eval_dir is not None and view_transformer is not None:
    _H_FRAMES.append(frame_idx)
    _H_LIST.append(view_transformer.H.copy())  # whatever the underlying matrix attr is

# At end:
if debug_eval_dir is not None and _H_FRAMES:
    np.savez(debug_eval_dir / "homographies.npz",
             frames=np.array(_H_FRAMES, dtype=np.int64),
             H=np.stack(_H_LIST, axis=0).astype(np.float64))
```

If the homography is cached on a stride, write only the frames where it was actually recomputed; `04_homography_metrics.py` carries the value forward as needed.

---

## (5) Per-frame per-track speeds — `speeds.csv`

**Format:**

```
frame, track_id, speed_kmh, capped
```

`capped` is 1 if the speed cap was triggered for that frame.

**Where to add:** in `src/stats/` (wherever speeds are computed). Expose the pre-cap value too.

```python
# In the speed-computation function, capture both pre- and post-cap:
raw_speed = compute_raw_speed(...)
capped_speed = min(raw_speed, SPEED_CAP_KMH)
was_capped = raw_speed > SPEED_CAP_KMH

# In run_pipeline:
if debug_eval_dir is not None:
    if _SPEED_F is None:
        _SPEED_F = (debug_eval_dir / "speeds.csv").open("w", newline="")
        _speed_w = csv.writer(_SPEED_F)
        _speed_w.writerow(["frame", "track_id", "speed_kmh", "capped"])
    for tid, raw_s, capped in per_frame_speeds:
        _speed_w.writerow([frame_idx, int(tid), float(raw_s), int(capped)])
```

---

## (6) Warm-up embeddings — `warmup_embeddings.npz`

**Format:** NPZ with:
- `embeddings`: (N, D) float — the 2-D UMAP-projected embeddings
- `cluster`: (N,) int — KMeans assignment
- `centroids`: (K, D) float — UMAP-space centroids

**Where to add:** in `TeamAssigner.fit(...)` after UMAP and KMeans fitting.

```python
# In TeamAssigner.fit(...):
if debug_eval_dir is not None:
    np.savez(debug_eval_dir / "warmup_embeddings.npz",
             embeddings=self.umap_embeddings.astype(np.float32),
             cluster=self.kmeans.labels_.astype(np.int64),
             centroids=self.kmeans.cluster_centers_.astype(np.float32))
```

If the model uses `siglip_features → umap → kmeans`, save the **UMAP-projected** features (2-D) so `03_team_classification_metrics.py` can plot them directly.

---

## Sanity check

After adding all six dumps, a single run:

```bash
./scripts/run.sh full input_videos/<clip>.mp4 \
    --debug-eval-dir evaluation/data/pipeline_outputs/<clip_name> \
    --debug-tracker-name bytetrack
```

should produce, under `evaluation/data/pipeline_outputs/<clip_name>/`, every file listed in the README's "Data layout" section *except* `tracks_botsort.txt` (run again with `--tracker botsort` for that). If any file is missing, the relevant evaluation script will print a clear error pointing back at this document.
