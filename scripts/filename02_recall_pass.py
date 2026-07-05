# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.temple_engine import DESKTOP_APP_DIR, collect_images, image_sharpness  # noqa: E402
from scripts.high_precision_facade_pass import color_features  # noqa: E402
from scripts.inspect_image_structure_features import features as structure_features  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def suffix_number(path: Path) -> str | None:
    match = re.search(r"_(\d+)(?:_\d+)?$", path.stem)
    return match.group(1) if match else None


def decode_size(path: Path) -> Tuple[int, int]:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def yolo_stats(result: Any, image_h: int) -> Dict[str, float]:
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "count": 0.0,
            "max_conf": 0.0,
            "upper_count": 0.0,
            "upper_max_conf": 0.0,
        }
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes), dtype=float)
    upper_confs: List[float] = []
    for box, conf in zip(boxes, confs):
        y1, y2 = float(box[1]), float(box[3])
        center_y = (y1 + y2) / 2.0 / max(image_h, 1)
        if center_y <= 0.72:
            upper_confs.append(float(conf))
    return {
        "count": float(len(boxes)),
        "max_conf": float(max(confs) if len(confs) else 0.0),
        "upper_count": float(len(upper_confs)),
        "upper_max_conf": float(max(upper_confs) if upper_confs else 0.0),
    }


def likely_interior_or_detail(color: Dict[str, float], structure: Dict[str, float]) -> bool:
    top_blue = float(color.get("top_blue_sky_ratio", 0.0))
    top_green = float(color.get("top_green_ratio", 0.0))
    top_ext = float(color.get("top_exterior_cue_ratio", 0.0))
    ext = float(color.get("exterior_cue_ratio", 0.0))
    bottom_gray = float(color.get("bottom_gray_ratio", 0.0))
    warm = float(color.get("warm_ratio", 0.0))
    gold = float(color.get("gold_ratio", 0.0))
    red = float(color.get("red_ratio", 0.0))
    dark = float(color.get("dark_ratio", 0.0))
    saturation = float(color.get("saturation", 0.0))
    edge = float(structure.get("edge_ratio", 0.0))
    long_lines = float(structure.get("long_line_count", 0.0))

    outdoor_signal = top_blue + top_green + min(bottom_gray, 0.6) * 0.7
    hot_interior = (
        warm > 0.62
        and top_blue < 0.03
        and top_green < 0.05
        and bottom_gray < 0.36
        and top_ext < 0.42
    )
    saturated_detail = saturation > 130 and top_blue < 0.04 and top_green < 0.04 and bottom_gray < 0.34
    dark_altar = dark > 0.42 and outdoor_signal < 0.20
    red_gold_detail = (red > 0.35 or gold > 0.24) and ext < 0.24 and bottom_gray < 0.35
    plaque_like = edge > 0.24 and long_lines > 130 and outdoor_signal < 0.30
    return bool(hot_interior or saturated_detail or dark_altar or red_gold_detail or plaque_like)


def keep_filename02_candidate(
    sharpness: float,
    color: Dict[str, float],
    structure: Dict[str, float],
    roof: Dict[str, float],
    ornament: Dict[str, float],
    min_sharpness: float,
) -> Tuple[bool, str, float]:
    reasons: List[str] = []
    if sharpness < min_sharpness:
        reasons.append(f"sharpness {sharpness:.1f} < {min_sharpness:.1f}")

    top_blue = float(color.get("top_blue_sky_ratio", 0.0))
    top_green = float(color.get("top_green_ratio", 0.0))
    top_ext = float(color.get("top_exterior_cue_ratio", 0.0))
    bottom_gray = float(color.get("bottom_gray_ratio", 0.0))
    outdoor_signal = top_blue + top_green + min(bottom_gray, 0.65) * 0.75

    roof_conf = float(roof.get("upper_max_conf", roof.get("max_conf", 0.0)))
    ornament_conf = float(ornament.get("upper_max_conf", ornament.get("max_conf", 0.0)))
    ornament_count = float(ornament.get("upper_count", ornament.get("count", 0.0)))
    has_roof = roof_conf >= 0.55
    has_ornament = ornament_conf >= 0.72 or (ornament_conf >= 0.58 and ornament_count >= 2)
    has_weak_roof_with_outdoor = roof_conf >= 0.42 and outdoor_signal >= 0.32 and top_ext >= 0.36

    if not (has_roof or has_ornament or has_weak_roof_with_outdoor):
        reasons.append(
            f"weak roof/ornament roof={roof_conf:.2f} ornament={ornament_conf:.2f} count={ornament_count:.0f}"
        )
    if likely_interior_or_detail(color, structure) and roof_conf < 0.74:
        reasons.append("looks like interior/plaque/detail")

    score = (
        min(roof_conf / 0.90, 1.0) * 0.42
        + min(ornament_conf / 0.90, 1.0) * 0.24
        + min(ornament_count / 4.0, 1.0) * 0.10
        + min(outdoor_signal / 0.75, 1.0) * 0.16
        + min(top_ext / 0.80, 1.0) * 0.08
    )
    return len(reasons) == 0, "; ".join(reasons), float(score)


def unique_destination(output_dir: Path, original_name: str, used_names: set[str]) -> Path:
    dst = output_dir / original_name
    if dst.name not in used_names and not dst.exists():
        used_names.add(dst.name)
        return dst
    stem, suffix = dst.stem, dst.suffix
    idx = 2
    while True:
        candidate = output_dir / f"{stem}_{idx}{suffix}"
        if candidate.name not in used_names and not candidate.exists():
            used_names.add(candidate.name)
            return candidate
        idx += 1


def copy_existing_images(folder: Path, output_dir: Path, used_names: set[str]) -> int:
    copied = 0
    if not folder or not folder.exists():
        return copied
    for src in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        dst = unique_destination(output_dir, src.name, used_names)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover likely temple front-facades using _02 filename convention.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-folder", action="append", default=[])
    parser.add_argument("--min-sharpness", type=float, default=35.0)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_names = {p.name for p in output_dir.iterdir() if p.is_file()}
    included = 0
    for folder in args.include_folder:
        included += copy_existing_images(Path(folder), output_dir, used_names)

    images = [p for p in collect_images(source) if suffix_number(p) == "02"]
    from ultralytics import YOLO

    roof_model = YOLO(str(DESKTOP_APP_DIR / "models" / "roof_yolo_best.pt"))
    ornament_model = YOLO(str(DESKTOP_APP_DIR / "models" / "roof_ridge_ornament_best.pt"))

    records: List[Dict[str, Any]] = []
    kept = 0
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    progress_path = output_dir / "filename02_recall_progress.txt"

    for idx, img_path in enumerate(images, 1):
        record: Dict[str, Any] = {"input_image": str(img_path), "keep": False, "output_image": ""}
        try:
            w, h = decode_size(img_path)
            record["image_width"] = w
            record["image_height"] = h
            if min(w, h) < 240:
                record["reason"] = f"image too small {w}x{h}"
                records.append(record)
                continue

            sharp = image_sharpness(img_path)
            record["sharpness"] = sharp
            color = color_features(img_path)
            structure = structure_features(img_path)
            roof_result = roof_model(str(img_path), conf=0.18, iou=0.70, max_det=8, verbose=False)[0]
            ornament_result = ornament_model(str(img_path), conf=0.25, iou=0.70, max_det=50, imgsz=960, verbose=False)[0]
            roof = yolo_stats(roof_result, h)
            ornament = yolo_stats(ornament_result, h)
            keep, reason, score = keep_filename02_candidate(sharp, color, structure, roof, ornament, args.min_sharpness)

            record.update({f"color_{k}": v for k, v in color.items()})
            record.update({f"structure_{k}": v for k, v in structure.items()})
            record.update({f"roof_{k}": v for k, v in roof.items()})
            record.update({f"ornament_{k}": v for k, v in ornament.items()})
            record["filename02_recall_score"] = score
            record["keep"] = keep
            record["reason"] = reason
            if keep and img_path.name not in used_names:
                dst = unique_destination(output_dir, img_path.name, used_names)
                shutil.copy2(img_path, dst)
                record["output_image"] = str(dst)
                kept += 1
            elif keep:
                record["reason"] = "already included by include-folder"
        except Exception as exc:
            record["reason"] = f"error: {exc}"
        records.append(record)

        if idx == 1 or idx % max(args.progress_every, 1) == 0 or idx == len(images):
            progress_path.write_text(
                "\n".join(
                    [
                        f"started: {started}",
                        f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"source: {source}",
                        f"output: {output_dir}",
                        f"processed: {idx}/{len(images)}",
                        f"included_existing: {included}",
                        f"new_kept: {kept}",
                    ]
                ),
                encoding="utf-8",
            )

    results_path = output_dir / "filename02_recall_results.json"
    results_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    total_images = sum(1 for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    summary = {
        "source": str(source),
        "output_dir": str(output_dir),
        "filename02_total": len(images),
        "included_existing": included,
        "new_kept": kept,
        "total_output_images": total_images,
        "results_json": str(results_path),
    }
    (output_dir / "filename02_recall_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
