# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "data" / "ultralytics_config"))

from backend.temple_engine import DESKTOP_APP_DIR, collect_images  # noqa: E402


RIDGE_MODEL = DESKTOP_APP_DIR / "models" / "roof_ridge_ornament_best.pt"
LONG_LABELS = {"long", "龙"}


def normalize_labels(labels: Sequence[str]) -> set[str]:
    normalized = {label.strip() for label in labels if label.strip()}
    return normalized | {label.lower() for label in normalized}


def is_long_label(label: str) -> bool:
    return label in LONG_LABELS or label.lower() in LONG_LABELS


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def clean_polygon(points: Any, width: int, height: int) -> List[List[float]]:
    cleaned: List[List[float]] = []
    last: Optional[tuple[float, float]] = None
    for raw_x, raw_y in np.asarray(points).tolist():
        x = min(max(float(raw_x), 0.0), max(float(width - 1), 0.0))
        y = min(max(float(raw_y), 0.0), max(float(height - 1), 0.0))
        point = (round(x, 2), round(y, 2))
        if point == last:
            continue
        cleaned.append([point[0], point[1]])
        last = point
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def box_polygon(box: Sequence[float], width: int, height: int) -> List[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return clean_polygon(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        width,
        height,
    )


def labelme_detections(model: Any, image_path: Path, ignored_labels: set[str], args: argparse.Namespace) -> Dict[str, Any]:
    width, height = image_size(image_path)
    result = model(
        str(image_path),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        imgsz=int(args.imgsz),
        verbose=False,
    )[0]
    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    if result.boxes is None or len(result.boxes) == 0:
        return {"status": "skipped", "reason": "no detection", "width": width, "height": height}

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
    boxes = result.boxes.xyxy.cpu().numpy()
    mask_polygons = list(result.masks.xy) if getattr(result, "masks", None) is not None and result.masks is not None else []

    non_long_candidates: List[Dict[str, Any]] = []
    long_candidates: List[Dict[str, Any]] = []
    ignored_count = 0
    for idx, (cls_id, score, box) in enumerate(zip(classes, confs, boxes)):
        label = str(names.get(int(cls_id), f"class_{int(cls_id)}"))
        if label in ignored_labels or label.lower() in ignored_labels:
            ignored_count += 1
            continue

        if idx < len(mask_polygons):
            points = clean_polygon(mask_polygons[idx], width, height)
            polygon_source = "mask"
        elif args.allow_box_fallback:
            points = box_polygon(box, width, height)
            polygon_source = "box_fallback"
        else:
            continue

        if len(points) < 3:
            continue

        item = {
            "class_id": int(cls_id),
            "label": label,
            "confidence": float(score),
            "points": points,
            "polygon_source": polygon_source,
            "width": width,
            "height": height,
        }
        if is_long_label(label):
            long_candidates.append(item)
        else:
            non_long_candidates.append(item)

    selected = list(long_candidates)
    if non_long_candidates:
        selected.append(max(non_long_candidates, key=lambda item: item["confidence"]))

    if not selected:
        reason = "only ignored label detected" if ignored_count else "no valid polygon"
        return {"status": "skipped", "reason": reason, "width": width, "height": height}

    return {
        "status": "annotated",
        "detections": selected,
        "width": width,
        "height": height,
    }


def write_labelme_json(image_path: Path, output_json: Path, dets: List[Dict[str, Any]], width: int, height: int, labelme_version: str) -> None:
    data = {
        "version": labelme_version,
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    for det in dets:
        data["shapes"].append({
            "label": det["label"],
            "points": det["points"],
            "group_id": None,
            "description": f"confidence={det['confidence']:.4f}; source={det['polygon_source']}",
            "shape_type": "polygon",
            "flags": {},
        })
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one LabelMe polygon per image from ridge ornament segmentation.")
    parser.add_argument("--input", required=True, help="Folder containing images.")
    parser.add_argument("--output-json-dir", default="", help="Folder for LabelMe JSON files. Defaults to input folder.")
    parser.add_argument("--ignore-label", action="append", default=[], help="Labels to ignore.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--allow-box-fallback", action="store_true", help="Use box polygon if a mask polygon is missing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output_json_dir) if args.output_json_dir else input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not RIDGE_MODEL.exists():
        raise FileNotFoundError(f"Missing ridge ornament model: {RIDGE_MODEL}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    ignored_labels = normalize_labels(args.ignore_label)

    try:
        import labelme

        labelme_version = getattr(labelme, "__version__", "6.1.0")
    except Exception:
        labelme_version = "6.1.0"

    from ultralytics import YOLO

    model = YOLO(str(RIDGE_MODEL))

    records: List[Dict[str, Any]] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    annotated = 0
    progress_path = output_dir / "labelme_polygon_progress.txt"

    for idx, image_path in enumerate(images, 1):
        output_json = output_dir / f"{image_path.stem}.json"
        record: Dict[str, Any] = {
            "input_image": str(image_path),
            "json": str(output_json),
            "status": "skipped",
            "label": "",
            "confidence": "",
            "polygon_points": "",
            "reason": "",
        }
        try:
            if output_json.exists() and not args.overwrite:
                record["reason"] = "json exists; not overwritten"
            else:
                det = labelme_detections(model, image_path, ignored_labels, args)
                if det.get("status") == "annotated":
                    detections = det["detections"]
                    write_labelme_json(image_path, output_json, detections, det["width"], det["height"], labelme_version)
                    annotated += 1
                    labels = [item["label"] for item in detections]
                    confidences = [float(item["confidence"]) for item in detections]
                    record.update({
                        "status": "annotated",
                        "label": ";".join(labels),
                        "confidence": max(confidences) if confidences else "",
                        "polygon_points": sum(len(item["points"]) for item in detections),
                        "reason": f"shapes={len(detections)}",
                    })
                else:
                    record["reason"] = det.get("reason", "")
        except (OSError, UnidentifiedImageError) as exc:
            record["status"] = "error"
            record["reason"] = f"image open failed: {exc}"
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
                    f"json_output: {output_dir}",
                    f"processed_images: {idx}/{len(images)}",
                    f"annotated: {annotated}",
                    f"skipped: {skipped}",
                    f"errors: {errors}",
                ]),
                encoding="utf-8",
            )
            print(f"[{idx}/{len(images)}] annotated={annotated} skipped={skipped} errors={errors}", flush=True)

    csv_path = output_dir / "labelme_polygon_records.csv"
    json_path = output_dir / "labelme_polygon_records.json"
    summary_path = output_dir / "labelme_polygon_summary.json"
    fields = ["input_image", "json", "status", "label", "confidence", "polygon_points", "reason"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    label_counts: Dict[str, int] = {}
    for record in records:
        if record.get("status") == "annotated":
            for label in str(record.get("label", "")).split(";"):
                if not label:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1

    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_dir),
        "json_output": str(output_dir),
        "input_images": len(images),
        "annotated": annotated,
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
