#!/usr/bin/env python3
"""
Evaluate your two-team clustering classifier without needing SportsMOT team labels.

Workflow
--------
1) Export player torso crops from your own video:
   python evaluate_team_classification_own_video.py export \
       --source input_videos/input.mp4 \
       --model models/object_detection/best.pt \
       --out-dir outputs/team_eval_manual \
       --stride 25 \
       --max-crops 300

2) Open outputs/team_eval_manual/labels_template.csv and fill the `team` column:
       team = 0 for one team
       team = 1 for the other team
       leave unsure rows blank, x, or ignore

3) Evaluate:
   python evaluate_team_classification_own_video.py evaluate \
       --labels outputs/team_eval_manual/labels_template.csv \
       --out-dir outputs/team_eval_manual \
       --device cpu

This script is intentionally label-flip safe:
K-Means cluster 0/1 can swap between runs, so metrics are computed for both
cluster-to-team mappings and the better mapping is reported.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# These imports are from your project. They match your existing plot_team_umap.py script.
try:
    from src.team.assigner import WarmupConfig
    from src.team.cropping import crop_torso
    from sports.common.team import TeamClassifier
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: Could not import your team-classification modules.\n"
        "Run this script from your project root, where src/ and sports/ are importable.\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    raise


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_crop_as_project_array(path: Path) -> np.ndarray:
    """
    Crops were exported as RGB images for viewing.
    Your existing script flips BGR -> RGB before saving, so here we flip RGB -> BGR
    to mimic the original crop array passed into TeamClassifier.
    """
    arr_rgb = np.array(Image.open(path).convert("RGB"))
    return arr_rgb[..., ::-1].copy()


def export_crops(
    source: str,
    model_path: str,
    out_dir: str,
    player_id: int,
    imgsz: int,
    stride: int,
    conf: float,
    iou: float,
    min_box_area: float,
    max_crops: int,
    contact_sheet_cols: int,
) -> None:
    try:
        import supervision as sv
        from ultralytics import YOLO
    except Exception as exc:
        print(
            "ERROR: export mode requires `supervision` and `ultralytics`.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise

    out = Path(out_dir)
    crops_dir = out / "crops"
    _safe_mkdir(crops_dir)

    model = YOLO(model_path)
    frame_gen = sv.get_video_frames_generator(source, stride=stride)

    rows: list[dict[str, str]] = []
    crop_paths_for_sheet: list[Path] = []

    crop_idx = 0
    for frame_i, frame in enumerate(frame_gen):
        frame_number = frame_i * stride + 1

        result = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, cid in zip(xyxy, class_ids):
            if int(cid) != int(player_id):
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < min_box_area:
                continue

            crop = crop_torso(frame, box)
            if crop is None:
                continue

            crop_id = f"crop_{crop_idx:05d}"
            crop_path = crops_dir / f"{crop_id}.png"

            # Convert BGR -> RGB for saving/viewing, consistent with your existing script.
            crop_rgb = crop[..., ::-1]
            Image.fromarray(crop_rgb).save(crop_path)

            rows.append(
                {
                    "crop_id": crop_id,
                    "crop_path": str(crop_path),
                    "frame": str(frame_number),
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "team": "",
                    "notes": "",
                }
            )
            crop_paths_for_sheet.append(crop_path)
            crop_idx += 1

            if crop_idx >= max_crops:
                break

        if crop_idx >= max_crops:
            break

    labels_csv = out / "labels_template.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_id",
                "crop_path",
                "frame",
                "x1",
                "y1",
                "x2",
                "y2",
                "team",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if crop_paths_for_sheet:
        save_contact_sheets(crop_paths_for_sheet, out / "contact_sheets", cols=contact_sheet_cols)

    print(f"Exported {len(rows)} crops")
    print(f"Fill team labels here: {labels_csv}")
    print(f"Use 0 for one team, 1 for the other. Leave uncertain rows blank or write ignore.")
    print(f"Contact sheets saved in: {out / 'contact_sheets'}")


def save_contact_sheets(crop_paths: list[Path], out_dir: Path, cols: int = 5, tile_w: int = 160, tile_h: int = 210) -> None:
    _safe_mkdir(out_dir)

    # Use default PIL font to avoid system-font assumptions.
    font = ImageFont.load_default()

    per_sheet = cols * 5
    for sheet_idx in range(math.ceil(len(crop_paths) / per_sheet)):
        batch = crop_paths[sheet_idx * per_sheet : (sheet_idx + 1) * per_sheet]
        rows = math.ceil(len(batch) / cols)
        sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
        draw = ImageDraw.Draw(sheet)

        for i, crop_path in enumerate(batch):
            r = i // cols
            c = i % cols
            x = c * tile_w
            y = r * tile_h

            img = Image.open(crop_path).convert("RGB")
            img.thumbnail((tile_w - 12, tile_h - 42))

            img_x = x + (tile_w - img.width) // 2
            img_y = y + 22
            sheet.paste(img, (img_x, img_y))

            draw.text((x + 6, y + 5), crop_path.stem, fill="black", font=font)

        sheet.save(out_dir / f"contact_sheet_{sheet_idx:03d}.jpg", quality=95)


def read_labelled_rows(labels_csv: str) -> tuple[list[Path], np.ndarray, list[dict[str, str]]]:
    path = Path(labels_csv)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    crop_paths: list[Path] = []
    y_true: list[int] = []
    rows_kept: list[dict[str, str]] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"crop_path", "team"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Labels CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            team_raw = str(row.get("team", "")).strip().lower()
            if team_raw in {"", "x", "ignore", "ignored", "na", "n/a", "none", "?"}:
                continue
            if team_raw not in {"0", "1"}:
                print(f"Skipping row with invalid team label {team_raw!r}: {row}", file=sys.stderr)
                continue

            crop_path = Path(row["crop_path"])
            if not crop_path.exists():
                print(f"Skipping missing crop path: {crop_path}", file=sys.stderr)
                continue

            crop_paths.append(crop_path)
            y_true.append(int(team_raw))
            rows_kept.append(row)

    if len(crop_paths) < 20:
        raise ValueError(
            f"Only {len(crop_paths)} labelled crops found. "
            "For a meaningful evaluation, label at least ~50 crops, ideally 100+."
        )

    return crop_paths, np.asarray(y_true, dtype=int), rows_kept


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_team0: float
    recall_team0: float
    f1_team0: float
    precision_team1: float
    recall_team1: float
    f1_team1: float
    tp0: int
    fp0: int
    fn0: int
    tp1: int
    fp1: int
    fn1: int


def _prf_for_class(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[float, float, float, int, int, int]:
    tp = int(np.sum((y_true == cls) & (y_pred == cls)))
    fp = int(np.sum((y_true != cls) & (y_pred == cls)))
    fn = int(np.sum((y_true == cls) & (y_pred != cls)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1, tp, fp, fn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    accuracy = float(np.mean(y_true == y_pred))

    p0, r0, f10, tp0, fp0, fn0 = _prf_for_class(y_true, y_pred, 0)
    p1, r1, f11, tp1, fp1, fn1 = _prf_for_class(y_true, y_pred, 1)

    return Metrics(
        accuracy=accuracy,
        precision_macro=(p0 + p1) / 2,
        recall_macro=(r0 + r1) / 2,
        f1_macro=(f10 + f11) / 2,
        precision_team0=p0,
        recall_team0=r0,
        f1_team0=f10,
        precision_team1=p1,
        recall_team1=r1,
        f1_team1=f11,
        tp0=tp0,
        fp0=fp0,
        fn0=fn0,
        tp1=tp1,
        fp1=fp1,
        fn1=fn1,
    )


def evaluate_labels(
    labels_csv: str,
    out_dir: str,
    device: str,
    random_state: int,
    save_predictions: bool,
) -> None:
    crop_paths, y_true, rows = read_labelled_rows(labels_csv)
    crops = [_load_crop_as_project_array(p) for p in crop_paths]

    tc = TeamClassifier(device=device)

    # Your existing plotting script sets a seeded 3D UMAP before KMeans.
    try:
        import umap
        tc.reducer = umap.UMAP(n_components=3, random_state=random_state)
    except Exception as exc:
        print(f"WARNING: Could not set seeded UMAP reducer: {exc}", file=sys.stderr)

    print(f"Extracting features for {len(crops)} labelled crops ...")
    features = tc.extract_features(crops)

    print("Fitting UMAP/reducer ...")
    projections = tc.reducer.fit_transform(features)

    print("Fitting KMeans ...")
    tc.cluster_model.fit(projections)
    cluster_labels = tc.cluster_model.predict(projections).astype(int)

    # KMeans labels are arbitrary. Evaluate both possible mappings.
    pred_same = cluster_labels.copy()
    pred_flipped = 1 - cluster_labels

    metrics_same = compute_metrics(y_true, pred_same)
    metrics_flipped = compute_metrics(y_true, pred_flipped)

    if (metrics_flipped.f1_macro, metrics_flipped.accuracy) > (metrics_same.f1_macro, metrics_same.accuracy):
        best_mapping = "flipped: cluster 0 -> team 1, cluster 1 -> team 0"
        y_pred = pred_flipped
        metrics = metrics_flipped
    else:
        best_mapping = "same: cluster 0 -> team 0, cluster 1 -> team 1"
        y_pred = pred_same
        metrics = metrics_same

    out = Path(out_dir)
    _safe_mkdir(out)

    summary_path = out / "team_eval_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_labelled_crops",
                "best_mapping",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_team0",
                "recall_team0",
                "f1_team0",
                "precision_team1",
                "recall_team1",
                "f1_team1",
                "tp0",
                "fp0",
                "fn0",
                "tp1",
                "fp1",
                "fn1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "n_labelled_crops": len(y_true),
                "best_mapping": best_mapping,
                "accuracy": f"{metrics.accuracy:.6f}",
                "precision_macro": f"{metrics.precision_macro:.6f}",
                "recall_macro": f"{metrics.recall_macro:.6f}",
                "f1_macro": f"{metrics.f1_macro:.6f}",
                "precision_team0": f"{metrics.precision_team0:.6f}",
                "recall_team0": f"{metrics.recall_team0:.6f}",
                "f1_team0": f"{metrics.f1_team0:.6f}",
                "precision_team1": f"{metrics.precision_team1:.6f}",
                "recall_team1": f"{metrics.recall_team1:.6f}",
                "f1_team1": f"{metrics.f1_team1:.6f}",
                "tp0": metrics.tp0,
                "fp0": metrics.fp0,
                "fn0": metrics.fn0,
                "tp1": metrics.tp1,
                "fp1": metrics.fp1,
                "fn1": metrics.fn1,
            }
        )

    if save_predictions:
        pred_path = out / "team_eval_predictions.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys()) + ["cluster_label", "predicted_team", "correct"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, cluster, pred, true in zip(rows, cluster_labels, y_pred, y_true):
                out_row = dict(row)
                out_row["cluster_label"] = int(cluster)
                out_row["predicted_team"] = int(pred)
                out_row["correct"] = int(pred == true)
                writer.writerow(out_row)
        print(f"Predictions saved to: {pred_path}")

    print("\nTeam classification evaluation")
    print("------------------------------")
    print(f"Labelled crops:   {len(y_true)}")
    print(f"Best mapping:     {best_mapping}")
    print(f"Accuracy:         {metrics.accuracy * 100:.2f}%")
    print(f"Precision macro:  {metrics.precision_macro * 100:.2f}%")
    print(f"Recall macro:     {metrics.recall_macro * 100:.2f}%")
    print(f"F1 macro:         {metrics.f1_macro * 100:.2f}%")
    print(f"\nSummary saved to: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual crop-labelling workflow for team-classification evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export player torso crops and a CSV template for manual labels.")
    export.add_argument("--source", required=True, help="Input video path.")
    export.add_argument("--model", default="models/object_detection/best.pt", help="YOLO model path.")
    export.add_argument("--out-dir", default="outputs/team_eval_manual", help="Output directory.")
    export.add_argument("--player-id", type=int, default=2, help="YOLO class ID for player.")
    export.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    export.add_argument("--stride", type=int, default=25, help="Process every Nth frame.")
    export.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    export.add_argument("--iou", type=float, default=0.7, help="YOLO IoU threshold.")
    export.add_argument("--min-box-area", type=float, default=500.0, help="Ignore boxes smaller than this area.")
    export.add_argument("--max-crops", type=int, default=300, help="Maximum crops to export.")
    export.add_argument("--contact-sheet-cols", type=int, default=5, help="Number of crop columns per contact sheet.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate clustering against manually labelled crops.")
    evaluate.add_argument("--labels", required=True, help="Filled labels_template.csv path.")
    evaluate.add_argument("--out-dir", default="outputs/team_eval_manual", help="Output directory.")
    evaluate.add_argument("--device", default="cpu", help="Device passed to TeamClassifier, e.g. cpu or cuda.")
    evaluate.add_argument("--random-state", type=int, default=42, help="UMAP random seed.")
    evaluate.add_argument("--save-predictions", action="store_true", help="Save per-crop predictions CSV.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        export_crops(
            source=args.source,
            model_path=args.model,
            out_dir=args.out_dir,
            player_id=args.player_id,
            imgsz=args.imgsz,
            stride=args.stride,
            conf=args.conf,
            iou=args.iou,
            min_box_area=args.min_box_area,
            max_crops=args.max_crops,
            contact_sheet_cols=args.contact_sheet_cols,
        )
    elif args.command == "evaluate":
        evaluate_labels(
            labels_csv=args.labels,
            out_dir=args.out_dir,
            device=args.device,
            random_state=args.random_state,
            save_predictions=args.save_predictions,
        )
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
