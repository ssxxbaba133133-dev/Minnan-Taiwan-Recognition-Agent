# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.temple_engine import DESKTOP_APP_DIR, collect_images, image_sharpness  # noqa: E402


def decode_bgr(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def unique_destination(output_dir: Path, original_name: str) -> Path:
    dst = output_dir / original_name
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    idx = 2
    while True:
        candidate = output_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def evaluate_image(img_path: Path, yolo_model: Any, min_sharpness: float) -> Dict[str, Any]:
    sharp = image_sharpness(img_path)
    keep = sharp >= min_sharpness
    reason: List[str] = []
    if not keep:
        reason.append(f"sharpness {sharp:.1f} < {min_sharpness:.1f}")

    box_score = 0.0
    best_metrics: Dict[str, float] = {}
    if keep:
        img = decode_bgr(img_path)
        if img is None:
            keep = False
            reason.append("image decode failed")
        else:
            h, w = img.shape[:2]
            if min(w, h) < 320:
                keep = False
                reason.append(f"image too small {w}x{h}")

            if keep:
                result = yolo_model(str(img_path), conf=0.35, iou=0.7, max_det=8, verbose=False)[0]
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                confs = (
                    result.boxes.conf.cpu().numpy()
                    if result.boxes is not None and result.boxes.conf is not None
                    else np.ones(len(boxes), dtype=float)
                )
                if len(boxes) == 0:
                    keep = False
                    reason.append("no building body detected")
                else:
                    best = -1.0
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = [float(v) for v in box]
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        area_ratio = max(0.0, bw * bh / max(w * h, 1))
                        width_ratio = bw / max(w, 1)
                        height_ratio = bh / max(h, 1)
                        cx = (x1 + x2) / 2.0 / max(w, 1)
                        cy = (y1 + y2) / 2.0 / max(h, 1)
                        bottom_ratio = y2 / max(h, 1)
                        centrality = max(0.0, 1.0 - abs(cx - 0.5) * 2.0)
                        vertical_ok = 1.0 if 0.22 <= cy <= 0.72 and bottom_ratio >= 0.42 else 0.0
                        score = (
                            area_ratio * 0.46
                            + width_ratio * 0.20
                            + height_ratio * 0.14
                            + centrality * 0.14
                            + float(conf) * 0.06
                        )
                        if vertical_ok <= 0.0:
                            score *= 0.65
                        if score > best:
                            best = score
                            best_metrics = {
                                "confidence": float(conf),
                                "area_ratio": float(area_ratio),
                                "width_ratio": float(width_ratio),
                                "height_ratio": float(height_ratio),
                                "centrality": float(centrality),
                                "bottom_ratio": float(bottom_ratio),
                                "center_y": float(cy),
                            }
                    box_score = float(max(best, 0.0))

                    if keep and best_metrics:
                        checks: List[Tuple[str, float, float]] = [
                            ("body confidence", best_metrics["confidence"], 0.45),
                            ("body area", best_metrics["area_ratio"], 0.10),
                            ("body width", best_metrics["width_ratio"], 0.24),
                            ("body height", best_metrics["height_ratio"], 0.16),
                            ("body centered", best_metrics["centrality"], 0.35),
                            ("body front-framed", best_metrics["bottom_ratio"], 0.42),
                        ]
                        for label, value, threshold in checks:
                            if value < threshold:
                                keep = False
                                reason.append(f"{label} {value:.2f} < {threshold:.2f}")
                    if keep and box_score < 0.24:
                        keep = False
                        reason.append(f"weak facade score {box_score:.2f}")

    record: Dict[str, Any] = {
        "task": "建筑主体区域识别",
        "input_image": str(img_path),
        "sharpness": sharp,
        "facade_score": box_score,
        "keep": keep,
        "reason": "; ".join(reason),
    }
    record.update(best_metrics)
    return record


def write_json(records: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter clear temple front-facade photos into a folder.")
    parser.add_argument("--input", required=True, help="Source folder containing images.")
    parser.add_argument("--output", required=True, help="Destination folder for kept original images.")
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    model_path = DESKTOP_APP_DIR / "models" / "body_yolo_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing YOLO model: {model_path}")

    from ultralytics import YOLO

    yolo_model = YOLO(str(model_path))
    records: List[Dict[str, Any]] = []
    csv_path = output_dir / "facade_filter_results.csv"
    json_path = output_dir / "facade_filter_results.json"
    progress_path = output_dir / "facade_filter_progress.txt"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    kept = 0

    fields = [
        "task",
        "input_image",
        "output_image",
        "sharpness",
        "facade_score",
        "keep",
        "reason",
        "confidence",
        "area_ratio",
        "width_ratio",
        "height_ratio",
        "centrality",
        "bottom_ratio",
        "center_y",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, img_path in enumerate(images, 1):
            try:
                record = evaluate_image(img_path, yolo_model, args.min_sharpness)
                if record["keep"]:
                    dst = unique_destination(output_dir, img_path.name)
                    shutil.copy2(img_path, dst)
                    record["output_image"] = str(dst)
                    kept += 1
                else:
                    record["output_image"] = ""
            except Exception as exc:
                record = {
                    "task": "建筑主体区域识别",
                    "input_image": str(img_path),
                    "output_image": "",
                    "sharpness": 0.0,
                    "facade_score": 0.0,
                    "keep": False,
                    "reason": f"error: {exc}",
                }
            records.append(record)
            writer.writerow({key: record.get(key, "") for key in fields})

            if idx == 1 or idx % max(args.progress_every, 1) == 0 or idx == len(images):
                progress_path.write_text(
                    "\n".join(
                        [
                            f"started: {started}",
                            f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            f"source: {input_path}",
                            f"target: {output_dir}",
                            f"processed: {idx}/{len(images)}",
                            f"kept: {kept}",
                            f"rejected: {idx - kept}",
                        ]
                    ),
                    encoding="utf-8",
                )
                f.flush()

    write_json(records, json_path)
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total": len(images),
        "kept": kept,
        "rejected": len(images) - kept,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    (output_dir / "facade_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
