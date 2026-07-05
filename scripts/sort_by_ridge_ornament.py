# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.temple_engine import DESKTOP_APP_DIR, collect_images  # noqa: E402


RIDGE_MODEL = DESKTOP_APP_DIR / "models" / "roof_ridge_ornament_best.pt"
DEFAULT_IGNORE_LABELS = {"long", "龙"}


def safe_folder_name(label: str) -> str:
    cleaned = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in label).strip()
    return cleaned or "unknown"


def unique_destination(folder: Path, original_name: str) -> Path:
    dst = folder / original_name
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    idx = 2
    while True:
        candidate = folder / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def normalize_ignored(labels: Sequence[str]) -> set[str]:
    ignored = set(DEFAULT_IGNORE_LABELS)
    ignored.update(label.strip() for label in labels if label.strip())
    return {label.lower() for label in ignored} | ignored


def run_ridge_detection(
    model: Any,
    image_path: Path,
    ignored_labels: set[str],
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int,
) -> Dict[str, Any]:
    result = model(str(image_path), conf=conf, iou=iou, max_det=max_det, imgsz=imgsz, verbose=False)[0]
    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "status": "skipped",
            "reason": "no ridge ornament detected",
            "detections": [],
        }

    classes = (
        result.boxes.cls.cpu().numpy().astype(int)
        if result.boxes.cls is not None
        else np.zeros(len(result.boxes), dtype=int)
    )
    confs = (
        result.boxes.conf.cpu().numpy()
        if result.boxes.conf is not None
        else np.ones(len(classes), dtype=float)
    )

    detections: List[Dict[str, Any]] = []
    valid: List[Dict[str, Any]] = []
    for cls_id, score in zip(classes, confs):
        label = str(names.get(int(cls_id), f"class_{int(cls_id)}"))
        item = {
            "class_id": int(cls_id),
            "label": label,
            "confidence": float(score),
            "ignored": label in ignored_labels or label.lower() in ignored_labels,
        }
        detections.append(item)
        if not item["ignored"]:
            valid.append(item)

    if not valid:
        return {
            "status": "skipped",
            "reason": "only ignored labels detected",
            "detections": detections,
        }

    best = max(valid, key=lambda item: item["confidence"])
    return {
        "status": "copied",
        "reason": "",
        "label": best["label"],
        "confidence": best["confidence"],
        "detections": detections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sort images into one ridge-ornament label folder each.")
    parser.add_argument("--input", required=True, help="Source image folder.")
    parser.add_argument("--output", required=True, help="Parent output folder for per-label subfolders.")
    parser.add_argument("--ignore-label", action="append", default=[], help="Labels to ignore. Defaults include long and 龙.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not RIDGE_MODEL.exists():
        raise FileNotFoundError(f"Missing ridge ornament model: {RIDGE_MODEL}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir)
    ignored_labels = normalize_ignored(args.ignore_label)

    from ultralytics import YOLO

    model = YOLO(str(RIDGE_MODEL))

    records: List[Dict[str, Any]] = []
    used_sources: set[str] = set()
    copied = 0
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    progress_path = output_dir / "ridge_sort_progress.txt"

    for idx, image_path in enumerate(images, 1):
        source_key = str(image_path.resolve())
        record: Dict[str, Any] = {
            "input_image": str(image_path),
            "output_image": "",
            "status": "skipped",
            "label": "",
            "confidence": "",
            "reason": "",
            "detections_json": "[]",
        }
        try:
            if source_key in used_sources:
                record["reason"] = "source image already copied"
            else:
                result = run_ridge_detection(
                    model=model,
                    image_path=image_path,
                    ignored_labels=ignored_labels,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    imgsz=args.imgsz,
                )
                record["status"] = result.get("status", "skipped")
                record["reason"] = result.get("reason", "")
                record["detections_json"] = json.dumps(result.get("detections", []), ensure_ascii=False)
                if record["status"] == "copied":
                    label = str(result["label"])
                    label_dir = output_dir / safe_folder_name(label)
                    label_dir.mkdir(parents=True, exist_ok=True)
                    dst = unique_destination(label_dir, image_path.name)
                    shutil.copy2(image_path, dst)
                    used_sources.add(source_key)
                    copied += 1
                    record["label"] = label
                    record["confidence"] = float(result["confidence"])
                    record["output_image"] = str(dst)
        except Exception as exc:
            record["status"] = "error"
            record["reason"] = str(exc)

        records.append(record)

        if idx == 1 or idx % max(args.progress_every, 1) == 0 or idx == len(images):
            skipped = sum(1 for item in records if item.get("status") == "skipped")
            errors = sum(1 for item in records if item.get("status") == "error")
            progress_path.write_text(
                "\n".join([
                    f"started: {started}",
                    f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"input: {input_dir}",
                    f"output: {output_dir}",
                    f"processed_images: {idx}/{len(images)}",
                    f"copied_images: {copied}",
                    f"skipped: {skipped}",
                    f"errors: {errors}",
                ]),
                encoding="utf-8",
            )
            print(f"[{idx}/{len(images)}] copied={copied} skipped={skipped} errors={errors}", flush=True)

    csv_path = output_dir / "ridge_sort_records.csv"
    json_path = output_dir / "ridge_sort_records.json"
    summary_path = output_dir / "ridge_sort_summary.json"

    fields = [
        "input_image",
        "output_image",
        "status",
        "label",
        "confidence",
        "reason",
        "detections_json",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    label_counts: Dict[str, int] = {}
    for item in records:
        if item.get("status") != "copied":
            continue
        label = str(item.get("label", ""))
        label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_dir),
        "output": str(output_dir),
        "input_images": len(images),
        "copied_images": copied,
        "skipped": sum(1 for item in records if item.get("status") == "skipped"),
        "errors": sum(1 for item in records if item.get("status") == "error"),
        "ignored_labels": sorted(str(label) for label in ignored_labels),
        "label_counts": label_counts,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
