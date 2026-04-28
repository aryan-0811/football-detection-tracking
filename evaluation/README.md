# Evaluation Harness

This folder contains the evaluation pipeline for the football analysis project. It consumes ground-truth data and pipeline outputs and produces every metric, table, and figure required for Chapter 4 of the report.

## Layout

```
evaluation/
  config.py                 Paths, clip registry, thresholds, hyper-parameter mirror.
  utils/                    Shared I/O, GT loaders, metric primitives.
  scripts/                  01..09 — runnable in order, also importable.
  results/
    tables/                 CSVs (LaTeX-ready) and *_summary.md verdicts.
    figures/                PDFs/PNGs.
    raw/                    JSON dumps for reproducibility.
    logs/                   Per-script log files.
  data/
    clips/                  Raw clip videos (you provide).
    ground_truth/<clip>/    Hand-labelled GT (you provide).
    pipeline_outputs/<clip>/Pipeline JSON/MP4/MOT exports.
  tests/                    Unit tests for utils.
  INSTRUMENTATION_TODO.md   Snippets to paste into run.py.
  requirements.txt          Pinned deps.
```

## Setup

```bash
cd "/Users/aryan/Desktop/Uni/Third Year Project/football-detection-tracking"
source .venv/bin/activate
pip install -r evaluation/requirements.txt

# Optional: TrackEval (academic-standard HOTA)
pip install "git+https://github.com/JonathonLuiten/TrackEval@master"
```

The harness uses `python -m evaluation.scripts.<name>` so you must run it from the project root.

## Data layout you must provide

For each clip you want to evaluate, create:

```
data/ground_truth/<clip_name>/
  tracks_mot.txt              MOT-Challenge format hand labels
  team_labels.json            {"<track_id>": <team_id>, ...}
  passes.json                 [{"frame": int, "passer_id": int, "receiver_id": int, "team_id": int}, ...]
  offsides.json               [{"frame": int, "offside_track_ids": [int, ...]}, ...]
  homography_keypoints.json   [{"frame": int, "image_xy": [x,y], "pitch_xy": [x,y], "label": str}, ...]

data/pipeline_outputs/<clip_name>/
  <clip_name>_offside_events.json   (already produced by run.py)
  <clip_name>_stats.json            (already produced by run.py)
  tracks_bytetrack.txt              (NEW — see INSTRUMENTATION_TODO.md item 2)
  tracks_botsort.txt                (NEW — same)
  team_predictions_raw.csv          (NEW — item 3)
  team_predictions_smoothed.csv     (NEW — item 3)
  homographies.npz                  (NEW — item 4)
  speeds.csv                        (NEW — item 5)
  warmup_embeddings.npz             (NEW — item 6)
  timings.csv                       (NEW — item 1)
```

Then register the clip in `config.CLIPS`.

## Instrumentation required in `src/`

Before scripts 02 and 04–07 can run, six debug-mode dumps need to be added to `run.py`. Exact code snippets and search anchors are in [INSTRUMENTATION_TODO.md](INSTRUMENTATION_TODO.md).

## Run order

```bash
python -m evaluation.scripts.01_detection_metrics
python -m evaluation.scripts.02_tracking_metrics
python -m evaluation.scripts.03_team_classification_metrics
python -m evaluation.scripts.04_homography_metrics
python -m evaluation.scripts.05_pass_offside_metrics
python -m evaluation.scripts.06_speed_distance_analysis
python -m evaluation.scripts.07_runtime_profiling
python -m evaluation.scripts.08_error_budget          # bonus, requires extra GT
python -m evaluation.scripts.09_compile_results
```

Common flags:

- `--clips clip_a clip_b` to restrict to a subset (default: every clip in `config.CLIPS`).
- `--output-dir <path>` to redirect outputs (default `evaluation/results/`).

Each script writes a `results/logs/<script>.log` file and prints a one-paragraph summary at the end.

## What each script produces

| Script | Section it serves | Key outputs |
|---|---|---|
| 01_detection_metrics | §3.4 / Criterion (i) | `detection_metrics_full.csv`, confusion-matrix and PR figures |
| 02_tracking_metrics | §3.5 / Criterion (ii) | `tracking_metrics.csv`, HOTA bar chart vs SoccerNet baseline |
| 03_team_classification_metrics | §3.6 / Criterion (iii) | F1 raw/smoothed/GK, UMAP scatter, smoothing-effect figure |
| 04_homography_metrics | §3.7 | Reprojection-error CSV (with kp_16 ablation, stride ablation) |
| 05_pass_offside_metrics | §3.8 / Criterion (iv) | Pass P/R/F1, per-event offside table |
| 06_speed_distance_analysis | §3.8 (cap) | Speed cap activation analysis, distribution plots |
| 07_runtime_profiling | §3.10 / §4.7 | Per-stage time breakdown vs 33 ms budget |
| 08_error_budget | Bonus | Five-condition oracle waterfall |
| 09_compile_results | §4.* meta | `all_results.csv`, `CHAPTER_4_DATA.md`, `SUCCESS_CRITERIA.md`, `MISSING_DATA.md` |

## First Run

1. `pip install -r evaluation/requirements.txt`.
2. `pytest evaluation/tests` — confirms helpers are healthy.
3. Add at least one clip to `config.CLIPS` and place its files under `data/`.
4. Apply the instrumentation in [INSTRUMENTATION_TODO.md](INSTRUMENTATION_TODO.md) and re-run your pipeline so the new dumps appear under `data/pipeline_outputs/<clip>/`.
5. Run scripts 01–09 in order.
6. Read `results/SUCCESS_CRITERIA.md` and `results/CHAPTER_4_DATA.md`.
7. Review `results/MISSING_DATA.md` to see what (if anything) still needs to be produced.

## Troubleshooting

- **`ModuleNotFoundError: evaluation`** — run from the project root, not from inside `evaluation/`.
- **`FileNotFoundError: GT tracks missing`** — the script names exactly which file is missing. Either provide it or remove that clip from the run via `--clips`.
- **HOTA scores are zero** — check that the clip name matches between GT and prediction filenames; check that frame numbers align (GT and predictions both 1-indexed).
- **`detection_metrics.json` empty** — drop your Ultralytics validation report into `data/detection_reports/`. Format details in `01_detection_metrics.py` docstring.
- **Script 08 reports "skipped"** — error-budget GT is partial; consult `MISSING_DATA.md`.

## Conventions

- All scripts use `pathlib.Path`, `argparse`, and `logging`.
- Script-level imports re-export a `main()` function so `09_compile_results.py` can call them programmatically.
- CSVs are written with the column order documented in each script's docstring.
- Figures are saved as PDF (preferred for LaTeX) and PNG.

## Notes on honest reporting

The harness intentionally surfaces negative results: smoothing that hurts, ablations that show no effect, false positives in the pass detector, frames where the speed cap is binding. Each `*_summary.md` includes a discussion section that you should expand with your own analysis before pasting into the report.
