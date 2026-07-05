# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.temple_engine import DESKTOP_APP_DIR, image_sharpness  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def decode_bgr(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def iter_top_level_images(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def run_yolo(model: Any, img_path: Path, conf: float, iou: float, max_det: int) -> List[Dict[str, Any]]:
    result = model(str(img_path), conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes), dtype=float)
    return [{"box": tuple(float(v) for v in box), "confidence": float(score)} for box, score in zip(boxes, confs)]


def box_metrics(box: Tuple[float, float, float, float], w: int, h: int) -> Dict[str, float]:
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(w), x2), min(float(h), y2)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / max(w, 1)
    cy = (y1 + y2) / 2.0 / max(h, 1)
    centrality = max(0.0, 1.0 - abs(cx - 0.5) * 2.0)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": bw,
        "height": bh,
        "area_ratio": bw * bh / max(w * h, 1),
        "width_ratio": bw / max(w, 1),
        "height_ratio": bh / max(h, 1),
        "center_x": cx,
        "center_y": cy,
        "centrality": centrality,
        "bottom_ratio": y2 / max(h, 1),
        "top_ratio": y1 / max(h, 1),
    }


def horizontal_overlap(a: Dict[str, float], b: Dict[str, float]) -> float:
    overlap = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    return overlap / max(min(a["width"], b["width"]), 1.0)


def choose_body(dets: List[Dict[str, Any]], w: int, h: int) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for det in dets:
        m = box_metrics(det["box"], w, h)
        score = (
            m["area_ratio"] * 0.42
            + m["width_ratio"] * 0.20
            + m["height_ratio"] * 0.16
            + m["centrality"] * 0.16
            + det["confidence"] * 0.06
        )
        if not (0.24 <= m["center_y"] <= 0.76 and m["bottom_ratio"] >= 0.46):
            score *= 0.65
        if score > best_score:
            best_score = score
            best = {"score": score, "confidence": det["confidence"], **m}
    return best


def choose_roof(dets: List[Dict[str, Any]], w: int, h: int, body: Dict[str, float]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for det in dets:
        m = box_metrics(det["box"], w, h)
        overlap = horizontal_overlap(m, body)
        above = 1.0 if m["center_y"] < body["center_y"] else 0.0
        score = (
            m["width_ratio"] * 0.25
            + m["area_ratio"] * 0.20
            + m["centrality"] * 0.14
            + overlap * 0.22
            + above * 0.10
            + det["confidence"] * 0.09
        )
        if m["center_y"] > 0.62:
            score *= 0.55
        if score > best_score:
            best_score = score
            best = {"score": score, "confidence": det["confidence"], "overlap_with_body": overlap, **m}
    return best


def evaluate(img_path: Path, body_model: Any, roof_model: Any, min_sharpness: float) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "input_image": str(img_path),
        "keep": False,
        "reason": "",
    }
    reasons: List[str] = []
    sharp = image_sharpness(img_path)
    record["sharpness"] = sharp
    if sharp < min_sharpness:
        reasons.append(f"sharpness {sharp:.1f} < {min_sharpness:.1f}")

    img = decode_bgr(img_path)
    if img is None:
        reasons.append("image decode failed")
        record["reason"] = "; ".join(reasons)
        return record
    h, w = img.shape[:2]
    record["image_width"] = w
    record["image_height"] = h
    if min(w, h) < 320:
        reasons.append(f"image too small {w}x{h}")

    body_dets = run_yolo(body_model, img_path, conf=0.35, iou=0.70, max_det=8)
    if not body_dets:
        reasons.append("no building body detected")
        record["reason"] = "; ".join(reasons)
        return record
    body = choose_body(body_dets, w, h)
    if body is None:
        reasons.append("no valid building body")
        record["reason"] = "; ".join(reasons)
        return record
    for key, value in body.items():
        record[f"body_{key}"] = value

    roof_dets = run_yolo(roof_model, img_path, conf=0.25, iou=0.70, max_det=8)
    if not roof_dets:
        reasons.append("no roof region detected")
        record["reason"] = "; ".join(reasons)
        return record
    roof = choose_roof(roof_dets, w, h, body)
    if roof is None:
        reasons.append("no valid roof region")
        record["reason"] = "; ".join(reasons)
        return record
    for key, value in roof.items():
        record[f"roof_{key}"] = value

    checks: List[Tuple[str, float, float]] = [
        ("body confidence", body["confidence"], 0.58),
        ("body area", body["area_ratio"], 0.20),
        ("body width", body["width_ratio"], 0.42),
        ("body height", body["height_ratio"], 0.24),
        ("body centered", body["centrality"], 0.52),
        ("body bottom", body["bottom_ratio"], 0.52),
        ("roof confidence", roof["confidence"], 0.36),
        ("roof width", roof["width_ratio"], 0.32),
        ("roof height", roof["height_ratio"], 0.07),
        ("roof centered", roof["centrality"], 0.42),
        ("roof/body overlap", roof["overlap_with_body"], 0.45),
    ]
    for label, value, threshold in checks:
        if value < threshold:
            reasons.append(f"{label} {value:.2f} < {threshold:.2f}")

    if roof["center_y"] >= body["center_y"]:
        reasons.append(f"roof not above body {roof['center_y']:.2f} >= {body['center_y']:.2f}")
    if roof["top_ratio"] > 0.48:
        reasons.append(f"roof too low {roof['top_ratio']:.2f} > 0.48")
    if body["area_ratio"] > 0.94 and roof["area_ratio"] < 0.06:
        reasons.append("body fills image but roof is too weak")

    record["strict_score"] = (
        body["score"] * 0.48
        + roof["score"] * 0.40
        + roof["overlap_with_body"] * 0.08
        + min(sharp / 800.0, 1.0) * 0.04
    )
    record["keep"] = len(reasons) == 0
    record["reason"] = "; ".join(reasons)
    return record


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-pass stricter temple facade filter.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-sharpness", type=float, default=80.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = iter_top_level_images(input_dir)

    from ultralytics import YOLO

    body_model = YOLO(str(DESKTOP_APP_DIR / "models" / "body_yolo_best.pt"))
    roof_model = YOLO(str(DESKTOP_APP_DIR / "models" / "roof_yolo_best.pt"))

    records: List[Dict[str, Any]] = []
    kept = 0
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    csv_path = output_dir / "strict_facade_results.csv"
    json_path = output_dir / "strict_facade_results.json"
    progress_path = output_dir / "strict_facade_progress.txt"

    fields = [
        "input_image",
        "output_image",
        "keep",
        "reason",
        "strict_score",
        "sharpness",
        "image_width",
        "image_height",
        "body_confidence",
        "body_score",
        "body_area_ratio",
        "body_width_ratio",
        "body_height_ratio",
        "body_center_y",
        "body_centrality",
        "body_bottom_ratio",
        "roof_confidence",
        "roof_score",
        "roof_area_ratio",
        "roof_width_ratio",
        "roof_height_ratio",
        "roof_center_y",
        "roof_centrality",
        "roof_overlap_with_body",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, img_path in enumerate(images, 1):
            try:
                record = evaluate(img_path, body_model, roof_model, args.min_sharpness)
                if record.get("keep"):
                    dst = unique_destination(output_dir, img_path.name)
                    shutil.copy2(img_path, dst)
                    record["output_image"] = str(dst)
                    kept += 1
                else:
                    record["output_image"] = ""
            except Exception as exc:
                record = {
                    "input_image": str(img_path),
                    "output_image": "",
                    "keep": False,
                    "reason": f"error: {exc}",
                }
            records.append(record)
            writer.writerow({field: record.get(field, "") for field in fields})

            if idx == 1 or idx % max(args.progress_every, 1) == 0 or idx == len(images):
                progress_path.write_text(
                    "\n".join(
                        [
                            f"started: {started}",
                            f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            f"source: {input_dir}",
                            f"target: {output_dir}",
                            f"processed: {idx}/{len(images)}",
                            f"kept: {kept}",
                            f"rejected: {idx - kept}",
                        ]
                    ),
                    encoding="utf-8",
                )
                f.flush()

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "input_path": str(input_dir),
        "output_dir": str(output_dir),
        "total": len(images),
        "kept": kept,
        "rejected": len(images) - kept,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    (output_dir / "strict_facade_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
