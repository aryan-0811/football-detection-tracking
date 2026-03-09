#!/usr/bin/env python3
"""
Train a ball-only YOLO model using Ultralytics, then evaluate on val/test and
(optionally) run a small prediction sample — all in one run.

Example:
  python train_ball.py \
    --data /scratch/football/ball_dataset/data.yaml \
    --model yolov8x.pt \
    --batch 8 --epochs 150 --imgsz 1280 \
    --device 0 --workers 8 --cache --patience 30 \
    --name ball_yolov8x_1280_baseline \
    --project /scratch/football/runs \
    --plots \
    --eval \
    --predict-sample 50
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run a command and stream stdout/stderr."""
    print("\nRunning command:\n", " ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML with PyYAML if available, otherwise fail with a helpful message."""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            "PyYAML is required to read data.yaml for automatic val/test evaluation.\n"
            "Install it with: pip install pyyaml\n"
            f"Import error: {e}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_dataset_root(data_yaml_path: Path) -> Path:
    """
    Returns the folder Ultralytics should treat as dataset root.
    For Roboflow exports, relative paths in data.yaml are resolved from data.yaml's parent.
    """
    return data_yaml_path.parent.resolve()


def infer_split_images_dir(data_yaml: Dict[str, Any], data_yaml_path: Path, split_key: str) -> Optional[Path]:
    """
    Return resolved images directory for train/val/test, if present in data.yaml.
    Handles relative paths like ../valid/images (Roboflow YOLO export).
    """
    v = data_yaml.get(split_key)
    if not v:
        return None
    p = Path(str(v))
    if p.is_absolute():
        return p
    # resolve relative to data.yaml parent
    return (data_yaml_path.parent / p).resolve()


def main() -> None:
    p = argparse.ArgumentParser()

    # Dataset path OR Roboflow download options
    p.add_argument("--data", required=True, help="Path to data.yaml OR 'auto' if using Roboflow download.")
    p.add_argument("--rf_api_key", default=None, help="Roboflow API key (or set ROBOFLOW_API_KEY env var).")
    p.add_argument("--rf_workspace", default=None, help="Roboflow workspace name.")
    p.add_argument("--rf_project", default=None, help="Roboflow project name.")
    p.add_argument("--rf_version", type=int, default=None, help="Roboflow dataset version number.")
    p.add_argument("--rf_format", default="yolov8", help="Export format (keep as yolov8).")

    # Training params
    p.add_argument("--model", default="yolov8x.pt")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")  # "0" for GPU, "cpu" for CPU
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--name", default="ball_yolov8x_1280_baseline")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--plots", action="store_true")

    # One-shot extras
    p.add_argument("--eval", action="store_true", help="After training, run yolo val on val + test splits.")
    p.add_argument(
        "--predict-sample",
        type=int,
        default=0,
        help="After training, run predictions on N test images (0 disables).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Prediction confidence threshold for sample predictions (use low for ball recall).",
    )

    args = p.parse_args()

    data_yaml = args.data

    # Optional: Roboflow dataset download
    if args.data.lower() == "auto":
        try:
            from roboflow import Roboflow
        except Exception as e:
            raise SystemExit(
                "roboflow package not installed. Install it or provide --data /path/to/data.yaml.\n"
                f"Error: {e}"
            )

        api_key = args.rf_api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise SystemExit("Missing Roboflow API key. Use --rf_api_key or set ROBOFLOW_API_KEY env var.")
        if not (args.rf_workspace and args.rf_project and args.rf_version):
            raise SystemExit("For --data auto, you must set --rf_workspace --rf_project --rf_version.")

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(args.rf_workspace).project(args.rf_project)
        dataset = project.version(args.rf_version).download(args.rf_format)

        data_yaml = str(Path(dataset.location) / "data.yaml")

    data_yaml_path = Path(data_yaml).expanduser().resolve()
    if not data_yaml_path.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml_path}")

    # Read YAML so we can find test split etc.
    data_yaml_obj = read_yaml(data_yaml_path)

    # Train
    cmd_train = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model={args.model}",
        f"data={str(data_yaml_path)}",
        f"batch={args.batch}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"patience={args.patience}",
        f"name={args.name}",
        f"project={args.project}",
    ]
    if args.cache:
        cmd_train.append("cache=True")
    if args.plots:
        cmd_train.append("plots=True")

    run(cmd_train)

    run_dir = Path(args.project).expanduser().resolve() / args.name
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    print("\n=== Training outputs ===")
    print("Run dir:", run_dir)
    print("Weights dir:", weights_dir)
    print("Best weights:", best_pt)
    print("Last weights:", last_pt)

    # Evaluate (val + test) using best.pt if available, else last.pt
    if args.eval:
        weights = best_pt if best_pt.exists() else last_pt
        if not weights.exists():
            raise SystemExit(f"No weights found at {best_pt} or {last_pt}. Training likely failed.")

        # Always evaluate on val (standard)
        eval_val_name = f"{args.name}_eval_val"
        cmd_val = [
            "yolo",
            "task=detect",
            "mode=val",
            f"model={str(weights)}",
            f"data={str(data_yaml_path)}",
            f"imgsz={args.imgsz}",
            f"device={args.device}",
            "split=val",
            f"project={args.project}",
            f"name={eval_val_name}",
            "plots=True",
        ]
        run(cmd_val)

        # Evaluate on test if data.yaml provides a test split
        if data_yaml_obj.get("test"):
            eval_test_name = f"{args.name}_eval_test"
            cmd_test = [
                "yolo",
                "task=detect",
                "mode=val",
                f"model={str(weights)}",
                f"data={str(data_yaml_path)}",
                f"imgsz={args.imgsz}",
                f"device={args.device}",
                "split=test",
                f"project={args.project}",
                f"name={eval_test_name}",
                "plots=True",
            ]
            run(cmd_test)
        else:
            print("\nNo 'test' entry found in data.yaml; skipping test evaluation.")

    # Prediction sample (use test/images if available)
    if args.predict_sample > 0:
        weights = best_pt if best_pt.exists() else last_pt
        if not weights.exists():
            raise SystemExit(f"No weights found at {best_pt} or {last_pt}. Training likely failed.")

        test_images_dir = infer_split_images_dir(data_yaml_obj, data_yaml_path, "test")
        if not test_images_dir or not test_images_dir.exists():
            print("\nTest images directory not found; skipping predict sample.")
        else:
            # Create a temp folder with a subset list to avoid predicting on thousands of images
            sample_list = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
            sample_list = sample_list[: args.predict_sample]

            sample_dir = run_dir / "artifacts" / "test_sample_images"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for pth in sample_list:
                # symlink if possible, else copy
                dst = sample_dir / pth.name
                if dst.exists():
                    continue
                try:
                    dst.symlink_to(pth)
                except Exception:
                    import shutil
                    shutil.copy2(pth, dst)

            pred_name = f"{args.name}_preds_test_sample"
            cmd_pred = [
                "yolo",
                "task=detect",
                "mode=predict",
                f"model={str(weights)}",
                f"source={str(sample_dir)}",
                f"imgsz={args.imgsz}",
                f"conf={args.conf}",
                f"device={args.device}",
                "save=True",
                "save_txt=False",
                f"project={args.project}",
                f"name={pred_name}",
            ]
            run(cmd_pred)

    print("\n✅ One-shot run complete.")
    print("Main run folder:", run_dir)
    if args.eval:
        print("Val eval folder:", Path(args.project) / f"{args.name}_eval_val")
        if data_yaml_obj.get("test"):
            print("Test eval folder:", Path(args.project) / f"{args.name}_eval_test")
    if args.predict_sample > 0:
        print("Pred sample folder:", Path(args.project) / f"{args.name}_preds_test_sample")


if __name__ == "__main__":
    main()