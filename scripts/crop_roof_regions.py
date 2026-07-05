# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.temple_engine import DESKTOP_APP_DIR, collect_images  # noqa: E402


BODY_MODEL = DESKTOP_APP_DIR / "models" / "body_yolo_best.pt"
ROOF_MODEL = DESKTOP_APP_DIR / "models" / "roof_yolo_best.pt"


def clamp_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def run_yolo(model: Any, image_path: Path, conf: float, iou: float, max_det: int) -> List[Dict[str, Any]]:
    result = model(str(image_path), conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes), dtype=float)
    detections: List[Dict[str, Any]] = []
    for box, score in zip(boxes, confs):
        detections.append({"box": tuple(float(v) for v in box), "confidence": float(score)})
    return detections


def unique_crop_path(output_dir: Path, image_path: Path, crop_index: int, suffix: str) -> Path:
    base = f"{image_path.stem}_roof_{crop_index:02d}{suffix}"
    candidate = output_dir / base
    if not candidate.exists():
        return candidate

    serial = 2
    while True:
        candidate = output_dir / f"{image_path.stem}_roof_{crop_index:02d}_{serial}{suffix}"
        if not candidate.exists():
            return candidate
        serial += 1


def save_crop(img: Image.Image, box: Tuple[int, int, int, int], output_path: Path) -> None:
    crop = img.crop(box)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        crop.save(output_path, quality=95, subsampling=0)
    else:
        crop.save(output_path)


def process_image(
    image_path: Path,
    output_dir: Path,
    body_model: Any,
    roof_model: Any,
    body_conf: float,
    roof_conf: float,
    iou: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with Image.open(image_path) as opened:
            img = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        return [{
            "input_image": str(image_path),
            "output_image": "",
            "status": "skipped",
            "reason": f"image open failed: {exc}",
        }]

    width, height = img.size
    body_dets = run_yolo(body_model, image_path, conf=body_conf, iou=iou, max_det=max_det)
    if not body_dets:
        return [{
            "input_image": str(image_path),
            "output_image": "",
            "status": "skipped",
            "reason": "no temple building facade/body detected",
            "image_width": width,
            "image_height": height,
        }]

    roof_dets = run_yolo(roof_model, image_path, conf=roof_conf, iou=iou, max_det=max_det)
    if not roof_dets:
        return [{
            "input_image": str(image_path),
            "output_image": "",
            "status": "skipped",
            "reason": "no roof region detected",
            "image_width": width,
            "image_height": height,
            "body_count": len(body_dets),
            "body_max_confidence": max(det["confidence"] for det in body_dets),
        }]

    suffix = image_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        suffix = ".jpg"

    crop_index = 0
    for det in sorted(roof_dets, key=lambda item: item["confidence"], reverse=True):
        x1, y1, x2, y2 = clamp_box(det["box"], width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_index += 1
        output_path = unique_crop_path(output_dir, image_path, crop_index, suffix)
        save_crop(img, (x1, y1, x2, y2), output_path)
        records.append({
            "input_image": str(image_path),
            "output_image": str(output_path),
            "status": "cropped",
            "reason": "",
            "image_width": width,
            "image_height": height,
            "body_count": len(body_dets),
            "body_max_confidence": max(det["confidence"] for det in body_dets),
            "roof_count": len(roof_dets),
            "roof_confidence": det["confidence"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "crop_width": x2 - x1,
            "crop_height": y2 - y1,
        })

    if records:
        return records

    return [{
        "input_image": str(image_path),
        "output_image": "",
        "status": "skipped",
        "reason": "roof boxes were invalid after clamping",
        "image_width": width,
        "image_height": height,
        "body_count": len(body_dets),
        "body_max_confidence": max(det["confidence"] for det in body_dets),
        "roof_count": len(roof_dets),
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop roof regions from temple facade images using Agent YOLO tools.")
    parser.add_argument("--input", required=True, help="Source image folder.")
    parser.add_argument("--output", required=True, help="Output folder for roof crops.")
    parser.add_argument("--body-conf", type=float, default=0.25)
    parser.add_argument("--roof-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not BODY_MODEL.exists():
        raise FileNotFoundError(f"Missing body YOLO model: {BODY_MODEL}")
    if not ROOF_MODEL.exists():
        raise FileNotFoundError(f"Missing roof YOLO model: {ROOF_MODEL}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir)

    from ultralytics import YOLO

    body_model = YOLO(str(BODY_MODEL))
    roof_model = YOLO(str(ROOF_MODEL))

    records: List[Dict[str, Any]] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    progress_path = output_dir / "roof_crop_progress.txt"

    for idx, image_path in enumerate(images, 1):
        try:
            image_records = process_image(
                image_path=image_path,
                output_dir=output_dir,
                body_model=body_model,
                roof_model=roof_model,
                body_conf=args.body_conf,
                roof_conf=args.roof_conf,
                iou=args.iou,
                max_det=args.max_det,
            )
        except Exception as exc:
            image_records = [{
                "input_image": str(image_path),
                "output_image": "",
                "status": "error",
                "reason": str(exc),
            }]
        records.extend(image_records)

        if idx == 1 or idx % max(args.progress_every, 1) == 0 or idx == len(images):
            cropped = sum(1 for item in records if item.get("status") == "cropped")
            skipped = sum(1 for item in records if item.get("status") == "skipped")
            errors = sum(1 for item in records if item.get("status") == "error")
            progress_path.write_text(
                "\n".join([
                    f"started: {started}",
                    f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"input: {input_dir}",
                    f"output: {output_dir}",
                    f"processed_images: {idx}/{len(images)}",
                    f"cropped_regions: {cropped}",
                    f"skipped_records: {skipped}",
                    f"error_records: {errors}",
                ]),
                encoding="utf-8",
            )
            print(f"[{idx}/{len(images)}] cropped={cropped} skipped={skipped} errors={errors}", flush=True)

    csv_path = output_dir / "roof_crop_records.csv"
    json_path = output_dir / "roof_crop_records.json"
    summary_path = output_dir / "roof_crop_summary.json"

    fields = [
        "input_image",
        "output_image",
        "status",
        "reason",
        "image_width",
        "image_height",
        "body_count",
        "body_max_confidence",
        "roof_count",
        "roof_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_width",
        "crop_height",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    input_images_with_crop = len({item["input_image"] for item in records if item.get("status") == "cropped"})
    summary = {
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(input_dir),
        "output": str(output_dir),
        "input_images": len(images),
        "images_with_crop": input_images_with_crop,
        "cropped_regions": sum(1 for item in records if item.get("status") == "cropped"),
        "skipped_records": sum(1 for item in records if item.get("status") == "skipped"),
        "error_records": sum(1 for item in records if item.get("status") == "error"),
        "csv": str(csv_path),
        "json": str(json_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
