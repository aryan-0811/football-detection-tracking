"""
Evaluate team classification on unseen crops from the same match using a percentage split.

This mirrors the intended pipeline:
  1) Export player torso crops in chronological order.
  2) Manually label the CSV with team labels.
  3) Fit TeamClassifier only on the first N% of crop rows.
  4) Freeze the fitted reducer + KMeans model.
  5) Evaluate only on labelled crops from the remaining rows.

Your CSV format is supported directly:
    crop_id,crop_path,frame,x1,y1,x2,y2,team,notes

Default split:
    first 35% rows  -> warm-up / fit split
    remaining 65%   -> unseen evaluation split

Example export:
    python evaluate_team_classification_percent_split.py export \
        --source input_videos/input.mp4 \
        --model models/object_detection/best.pt \
        --out-dir outputs/team_eval_percent \
        --stride 25 \
        --max-crops 300

Then fill the team column in:
    outputs/team_eval_percent/labels_template.csv

Use 0 for one team and 1 for the other team.
Leave unclear crops blank or set to ignore.

Example evaluate:
    python evaluate_team_classification_percent_split.py evaluate \
        --labels outputs/team_eval_percent/labels_template.csv \
        --out-dir outputs/team_eval_percent \
        --warmup-percent 35 \
        --device cpu \
        --save-predictions

Important:
  - The first warmup-percent rows are used for fitting, even if the team column is blank.
  - The remaining rows are NEVER used for fitting.
  - Metrics are computed only on labelled rows in the evaluation split.
  - KMeans labels are arbitrary. By default, mapping_source=auto uses labelled warm-up
    crops to map cluster IDs to team IDs if enough labels exist. Otherwise it falls back
    to best label mapping on the eval split. That fallback is only for interpreting
    cluster IDs during scoring; it does not refit the classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import supervision as sv
from ultralytics import YOLO

# Project modules used by your existing implementation.
from src.team.cropping import crop_torso
from sports.common.team import TeamClassifier


IGNORE_VALUES = {"", "ignore", "ignored", "x", "?", "na", "n/a", "none", "null", "-1"}


@dataclass
class CropRow:
    row_index: int
    crop_id: str
    crop_path: Path
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    team: int | None
    notes: str = ""


def parse_team(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in IGNORE_VALUES:
        return None
    try:
        label = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Invalid team label {value!r}. Use 0, 1, blank, or ignore.") from exc
    if label not in (0, 1):
        raise ValueError(f"Invalid team label {value!r}. Use only 0 or 1.")
    return label


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_crop_bgr(crop: np.ndarray, out_path: Path) -> None:
    """Save a project crop. Project frames/crops are usually BGR, so convert to RGB for PNG."""
    ensure_dir(out_path.parent)
    if crop.ndim == 3 and crop.shape[2] == 3:
        img = Image.fromarray(crop[..., ::-1])
    else:
        img = Image.fromarray(crop)
    img.save(out_path)


def load_crop_as_project_array(path: Path, assume_saved_as_rgb: bool = True) -> np.ndarray:
    """
    Load a crop image for TeamClassifier.

    Crops exported by this script are normal RGB PNG files on disk, but your existing
    pipeline passes BGR numpy crops into TeamClassifier. By default this converts RGB
    back to BGR to match that behaviour.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    if assume_saved_as_rgb:
        return arr[..., ::-1].copy()
    return arr.copy()


def make_contact_sheet(rows: list[CropRow], out_path: Path, title: str, cols: int = 6, thumb_w: int = 140) -> None:
    if not rows:
        return
    ensure_dir(out_path.parent)

    thumbs: list[Image.Image] = []
    for row in rows:
        img = Image.open(row.crop_path).convert("RGB")
        scale = thumb_w / max(1, img.width)
        thumb_h = max(1, int(img.height * scale))
        thumbs.append(img.resize((thumb_w, thumb_h)))

    label_h = 34
    title_h = 40
    cell_h = max(t.height for t in thumbs) + label_h
    rows_n = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, title_h + rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), title, fill="black")

    for idx, (row, thumb) in enumerate(zip(rows, thumbs)):
        r = idx // cols
        c = idx % cols
        x = c * thumb_w
        y = title_h + r * cell_h
        sheet.paste(thumb, (x, y))
        label = f"{row.crop_id}\nf={row.frame}"
        draw.text((x + 4, y + thumb.height + 2), label, fill="black")

    sheet.save(out_path)


def export_crops(args: argparse.Namespace) -> None:
    source = str(args.source)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    contact_dir = out_dir / "contact_sheets"
    ensure_dir(crops_dir)
    ensure_dir(contact_dir)

    model = YOLO(args.model)
    frame_gen = sv.get_video_frames_generator(source, stride=args.stride)

    rows: list[CropRow] = []
    crop_idx = 0
    frame_idx = 0

    print(f"Exporting crops from {source}")
    print(f"stride={args.stride}, max_crops={args.max_crops}")

    for frame in frame_gen:
        frame_idx += args.stride
        if args.max_crops is not None and crop_idx >= args.max_crops:
            break

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, cid in zip(xyxy, class_ids):
            if int(cid) != args.player_id:
                continue
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if area < args.min_box_area:
                continue
            if args.max_crops is not None and crop_idx >= args.max_crops:
                break

            crop = crop_torso(frame, box)
            if crop is None:
                continue

            crop_id = f"crop_{crop_idx:05d}"
            crop_path = crops_dir / f"{crop_id}.png"
            save_crop_bgr(crop, crop_path)

            rows.append(
                CropRow(
                    row_index=crop_idx,
                    crop_id=crop_id,
                    crop_path=crop_path,
                    frame=frame_idx,
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    team=None,
                    notes="",
                )
            )
            crop_idx += 1

    if not rows:
        raise RuntimeError("No crops exported. Check --player-id, --conf, --model, and video path.")

    labels_csv = out_dir / "labels_template.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["crop_id", "crop_path", "frame", "x1", "y1", "x2", "y2", "team", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "crop_id": row.crop_id,
                    "crop_path": str(row.crop_path),
                    "frame": row.frame,
                    "x1": f"{row.x1:.2f}",
                    "y1": f"{row.y1:.2f}",
                    "x2": f"{row.x2:.2f}",
                    "y2": f"{row.y2:.2f}",
                    "team": "",
                    "notes": "",
                }
            )

    make_contact_sheet(
        rows[: args.contact_sheet_limit],
        contact_dir / "crops_contact_sheet.png",
        f"Exported crops in CSV order. n={len(rows)}",
    )

    default_warmup_count = split_index_for_percent(len(rows), args.warmup_percent)
    print("Done.")
    print(f"Exported total crops: {len(rows)}")
    print(f"Default {args.warmup_percent:.1f}% warm-up split would use first {default_warmup_count} crops for fitting")
    print(f"Default eval split would use remaining {len(rows) - default_warmup_count} crops for scoring")
    print(f"Fill team labels in: {labels_csv}")
    print(f"Contact sheet saved in: {contact_dir}")


def read_rows(labels_csv: Path) -> list[CropRow]:
    base_dir = labels_csv.parent
    rows: list[CropRow] = []

    with labels_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"crop_id", "crop_path", "frame", "x1", "y1", "x2", "y2", "team", "notes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "CSV missing required columns: "
                f"{sorted(missing)}. Expected header: crop_id,crop_path,frame,x1,y1,x2,y2,team,notes"
            )

        for idx, raw in enumerate(reader):
            crop_path = Path(raw["crop_path"])
            if not crop_path.is_absolute() and not crop_path.exists():
                crop_path = base_dir / crop_path

            rows.append(
                CropRow(
                    row_index=idx,
                    crop_id=raw["crop_id"],
                    crop_path=crop_path,
                    frame=int(float(raw["frame"])),
                    x1=float(raw.get("x1", 0) or 0),
                    y1=float(raw.get("y1", 0) or 0),
                    x2=float(raw.get("x2", 0) or 0),
                    y2=float(raw.get("y2", 0) or 0),
                    team=parse_team(raw.get("team")),
                    notes=raw.get("notes", ""),
                )
            )

    if len(rows) < 2:
        raise RuntimeError("Need at least 2 crop rows so there can be a warm-up and evaluation split.")
    return rows


def split_index_for_percent(n_rows: int, warmup_percent: float) -> int:
    if not (0.0 < warmup_percent < 100.0):
        raise ValueError("--warmup-percent must be between 0 and 100, e.g. 35")
    if n_rows < 2:
        raise ValueError("Need at least 2 rows to split")

    n_warmup = int(round(n_rows * (warmup_percent / 100.0)))
    # Always leave at least one crop in each split.
    n_warmup = max(1, min(n_rows - 1, n_warmup))
    return n_warmup


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("No labelled evaluation crops found.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for cls in [0, 1]:
        cls_tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        cls_fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        cls_fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) else 0.0
        r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        macro_precision += p / 2
        macro_recall += r / 2
        macro_f1 += f / 2

    return {
        "n": float(len(y_true)),
        "accuracy": accuracy,
        "precision_team1": precision,
        "recall_team1": recall,
        "f1_team1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def apply_mapping(cluster_labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(c)] for c in cluster_labels], dtype=int)


def choose_mapping_from_labels(cluster_labels: np.ndarray, y_true: np.ndarray) -> tuple[dict[int, int], dict[str, float]]:
    """Try both cluster-team mappings and choose the one with higher macro-F1, then accuracy."""
    candidates = [
        {0: 0, 1: 1},
        {0: 1, 1: 0},
    ]
    scored = []
    for mapping in candidates:
        y_pred = apply_mapping(cluster_labels, mapping)
        m = binary_metrics(y_true, y_pred)
        scored.append((mapping, m))
    scored.sort(key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]), reverse=True)
    return scored[0]


def evaluate(args: argparse.Namespace) -> None:
    labels_csv = Path(args.labels)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows = read_rows(labels_csv)
    n_warmup = split_index_for_percent(len(rows), args.warmup_percent)
    warmup_rows = rows[:n_warmup]
    eval_all_rows = rows[n_warmup:]
    eval_rows = [r for r in eval_all_rows if r.team is not None]
    labelled_warmup_rows = [r for r in warmup_rows if r.team is not None]

    if len(warmup_rows) < args.min_warmup_crops:
        raise RuntimeError(
            f"Only {len(warmup_rows)} warm-up crops found from the first {args.warmup_percent:.1f}% rows. "
            f"Need at least {args.min_warmup_crops}. Export more crops or lower --min-warmup-crops."
        )
    if len(eval_rows) == 0:
        raise RuntimeError(
            "No labelled crops found in the evaluation split. "
            f"Label some rows after row index {n_warmup - 1}, which is the first row after the warm-up split."
        )

    print("Split based on CSV row order:")
    print(f"  total rows:             {len(rows)}")
    print(f"  warm-up percent:        {args.warmup_percent:.2f}%")
    print(f"  warm-up rows for fit:   0 to {n_warmup - 1}  (n={len(warmup_rows)})")
    print(f"  eval rows for scoring:  {n_warmup} to {len(rows) - 1}  (n={len(eval_all_rows)}, labelled={len(eval_rows)})")

    for row in warmup_rows + eval_rows:
        if not row.crop_path.exists():
            raise FileNotFoundError(f"Crop image not found: {row.crop_path}")

    print("Loading warm-up crops...")
    warmup_crops = [load_crop_as_project_array(r.crop_path, assume_saved_as_rgb=not args.no_bgr_conversion) for r in warmup_rows]

    print("Loading labelled unseen eval crops...")
    eval_crops = [load_crop_as_project_array(r.crop_path, assume_saved_as_rgb=not args.no_bgr_conversion) for r in eval_rows]

    print("Building TeamClassifier...")
    tc = TeamClassifier(device=args.device)

    if args.fix_umap_seed:
        try:
            import umap

            tc.reducer = umap.UMAP(n_components=args.umap_components, random_state=args.umap_seed)
        except Exception as exc:
            print(f"WARNING: Could not replace reducer with seeded UMAP: {exc}", file=sys.stderr)

    print(f"Extracting features for warm-up crops only: n={len(warmup_crops)}")
    warmup_features = tc.extract_features(warmup_crops)

    print("Fitting reducer on warm-up crops only...")
    warmup_proj = tc.reducer.fit_transform(warmup_features)

    print("Fitting KMeans on warm-up crops only...")
    tc.cluster_model.fit(warmup_proj)
    warmup_clusters = tc.cluster_model.predict(warmup_proj)

    print(f"Extracting features for unseen eval crops: n={len(eval_crops)}")
    eval_features = tc.extract_features(eval_crops)

    print("Transforming unseen crops with frozen reducer...")
    if not hasattr(tc.reducer, "transform"):
        raise RuntimeError(
            "The reducer has no transform() method, so unseen crops cannot be projected without refitting. "
            "Use UMAP or another reducer that supports transform()."
        )
    eval_proj = tc.reducer.transform(eval_features)

    print("Predicting unseen crops with frozen KMeans...")
    eval_clusters = tc.cluster_model.predict(eval_proj)
    y_true_eval = np.asarray([r.team for r in eval_rows], dtype=int)

    mapping_source = args.mapping_source
    mapping_note = ""
    if mapping_source == "auto":
        mapping_source = "warmup" if len(labelled_warmup_rows) >= args.min_mapping_labels else "eval_best"

    if mapping_source == "warmup":
        if len(labelled_warmup_rows) < args.min_mapping_labels:
            raise RuntimeError(
                f"mapping-source=warmup requires at least {args.min_mapping_labels} labelled warm-up crops. "
                f"Found {len(labelled_warmup_rows)}. Label more of the first {n_warmup} rows or use --mapping-source eval_best."
            )
        idxs = [r.row_index for r in labelled_warmup_rows]
        y_true_mapping = np.asarray([r.team for r in labelled_warmup_rows], dtype=int)
        cluster_mapping_labels = warmup_clusters[idxs]
        mapping, mapping_metrics = choose_mapping_from_labels(cluster_mapping_labels, y_true_mapping)
        mapping_note = "Cluster-to-team mapping chosen using labelled warm-up crops only."
    elif mapping_source == "eval_best":
        mapping, mapping_metrics = choose_mapping_from_labels(eval_clusters, y_true_eval)
        mapping_note = (
            "Cluster-to-team mapping chosen using eval labels. This handles arbitrary KMeans label IDs for scoring, "
            "but the eval crops were not used to fit UMAP or KMeans."
        )
    else:
        raise ValueError("mapping_source must be auto, warmup, or eval_best")

    y_pred_eval = apply_mapping(eval_clusters, mapping)
    metrics = binary_metrics(y_true_eval, y_pred_eval)

    summary = {
        "split_method": "csv_row_order_percent",
        "warmup_percent": args.warmup_percent,
        "total_crop_rows": len(rows),
        "warmup_row_start_inclusive": 0,
        "warmup_row_end_inclusive": n_warmup - 1,
        "eval_row_start_inclusive": n_warmup,
        "eval_row_end_inclusive": len(rows) - 1,
        "n_warmup_crops_used_for_fit": len(warmup_rows),
        "n_eval_crops_available": len(eval_all_rows),
        "n_eval_crops_scored": len(eval_rows),
        "n_labelled_warmup_crops_used_for_mapping": len(labelled_warmup_rows) if mapping_source == "warmup" else 0,
        "mapping_source": mapping_source,
        "cluster_to_team_mapping": mapping,
        "mapping_note": mapping_note,
        "metrics": metrics,
        "mapping_split_metrics": mapping_metrics,
    }

    summary_json = out_dir / "percent_split_eval_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv = out_dir / "percent_split_eval_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_method",
                "warmup_percent",
                "total_crop_rows",
                "n_warmup_crops_used_for_fit",
                "n_eval_crops_available",
                "n_eval_crops_scored",
                "n_labelled_warmup_crops_used_for_mapping",
                "mapping_source",
                "cluster0_team",
                "cluster1_team",
                "n",
                "accuracy",
                "precision_team1",
                "recall_team1",
                "f1_team1",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "tp",
                "tn",
                "fp",
                "fn",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split_method": "csv_row_order_percent",
                "warmup_percent": args.warmup_percent,
                "total_crop_rows": len(rows),
                "n_warmup_crops_used_for_fit": len(warmup_rows),
                "n_eval_crops_available": len(eval_all_rows),
                "n_eval_crops_scored": len(eval_rows),
                "n_labelled_warmup_crops_used_for_mapping": len(labelled_warmup_rows) if mapping_source == "warmup" else 0,
                "mapping_source": mapping_source,
                "cluster0_team": mapping[0],
                "cluster1_team": mapping[1],
                **metrics,
            }
        )

    if args.save_predictions:
        pred_csv = out_dir / "percent_split_eval_predictions.csv"
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "crop_id",
                    "crop_path",
                    "row_index",
                    "frame",
                    "true_team",
                    "cluster_label",
                    "pred_team",
                    "correct",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "notes",
                ],
            )
            writer.writeheader()
            for row, cluster, pred in zip(eval_rows, eval_clusters, y_pred_eval):
                writer.writerow(
                    {
                        "crop_id": row.crop_id,
                        "crop_path": str(row.crop_path),
                        "row_index": row.row_index,
                        "frame": row.frame,
                        "true_team": row.team,
                        "cluster_label": int(cluster),
                        "pred_team": int(pred),
                        "correct": int(pred == row.team),
                        "x1": f"{row.x1:.2f}",
                        "y1": f"{row.y1:.2f}",
                        "x2": f"{row.x2:.2f}",
                        "y2": f"{row.y2:.2f}",
                        "notes": row.notes,
                    }
                )
        print(f"Saved predictions: {pred_csv}")

    print("\nEvaluation complete: fitted on first percentage of crops, scored on unseen remaining crops.")
    print(mapping_note)
    print(f"Cluster mapping: cluster 0 -> team {mapping[0]}, cluster 1 -> team {mapping[1]}")
    print(f"Warm-up crops used for fit: {len(warmup_rows)} / {len(rows)} ({args.warmup_percent:.2f}%)")
    print(f"Unseen labelled eval crops scored: {len(eval_rows)} / {len(eval_all_rows)}")
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:       {metrics['precision_team1'] * 100:.2f}%  (team 1 as positive class)")
    print(f"Recall:          {metrics['recall_team1'] * 100:.2f}%  (team 1 as positive class)")
    print(f"F1:              {metrics['f1_team1'] * 100:.2f}%  (team 1 as positive class)")
    print(f"Macro Precision: {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:    {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1:        {metrics['macro_f1'] * 100:.2f}%")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved JSON:    {summary_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate team classifier using first N percent of crop rows as warm-up")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_p = subparsers.add_parser("export", help="Export torso crops and a CSV labelling template")
    export_p.add_argument("--source", required=True, type=Path, help="Input video path")
    export_p.add_argument("--model", default="models/object_detection/best.pt", help="YOLO model path")
    export_p.add_argument("--out-dir", default="outputs/team_eval_percent", type=Path)
    export_p.add_argument("--player-id", type=int, default=2, help="YOLO class ID for players")
    export_p.add_argument("--imgsz", type=int, default=1280)
    export_p.add_argument("--conf", type=float, default=0.25)
    export_p.add_argument("--iou", type=float, default=0.7)
    export_p.add_argument("--min-box-area", type=float, default=1000.0)
    export_p.add_argument("--stride", type=int, default=25, help="Process every Nth frame")
    export_p.add_argument("--max-crops", type=int, default=300, help="Maximum total crops to export; set -1 for no limit")
    export_p.add_argument("--warmup-percent", type=float, default=35.0, help="Only used for reporting the expected split after export")
    export_p.add_argument("--contact-sheet-limit", type=int, default=120)

    eval_p = subparsers.add_parser("evaluate", help="Fit on first N percent of crop rows and evaluate on the rest")
    eval_p.add_argument("--labels", required=True, type=Path, help="CSV with header: crop_id,crop_path,frame,x1,y1,x2,y2,team,notes")
    eval_p.add_argument("--out-dir", default="outputs/team_eval_percent", type=Path)
    eval_p.add_argument("--warmup-percent", type=float, default=35.0, help="Percentage of CSV rows used for warm-up fitting")
    eval_p.add_argument("--device", default="cpu")
    eval_p.add_argument("--mapping-source", choices=["auto", "warmup", "eval_best"], default="auto")
    eval_p.add_argument("--min-warmup-crops", type=int, default=50)
    eval_p.add_argument("--min-mapping-labels", type=int, default=10)
    eval_p.add_argument("--save-predictions", action="store_true")
    eval_p.add_argument("--fix-umap-seed", action=argparse.BooleanOptionalAction, default=True)
    eval_p.add_argument("--umap-components", type=int, default=3)
    eval_p.add_argument("--umap-seed", type=int, default=42)
    eval_p.add_argument(
        "--no-bgr-conversion",
        action="store_true",
        help="Use this only if your crop images are already in the exact colour channel order expected by TeamClassifier.",
    )

    return parser


def normalize_limits(args: argparse.Namespace) -> None:
    if hasattr(args, "max_crops") and args.max_crops is not None and args.max_crops < 0:
        args.max_crops = None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    normalize_limits(args)

    if args.command == "export":
        export_crops(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
