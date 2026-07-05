# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ratio(mask) -> float:
    return float(mask.mean())


def color_features(path: Path) -> Dict[str, float]:
    rgb = read_rgb(path)
    if rgb is None:
        return {"decode_failed": 1.0}
    h, w = rgb.shape[:2]
    resize_w = min(w, 512)
    resize_h = max(1, int(h * resize_w / max(w, 1)))
    small = cv2.resize(rgb, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    r = small[:, :, 0].astype(np.int16)
    g = small[:, :, 1].astype(np.int16)
    b = small[:, :, 2].astype(np.int16)
    hh = small.shape[0]
    top = slice(0, max(1, hh // 3))
    bottom = slice(max(0, hh * 2 // 3), hh)

    warm = (((hue <= 28) | (hue >= 170)) & (sat > 70) & (val > 50))
    gold = ((hue >= 18) & (hue <= 45) & (sat > 60) & (val > 70))
    red = (((hue <= 10) | (hue >= 170)) & (sat > 80) & (val > 50))
    dark = val < 55
    bright_neutral = (val > 145) & (sat < 70)
    white_sky_wall = (val > 175) & (sat < 95)
    blue_sky = (b > r + 12) & (b > g + 4) & (val > 110) & (sat > 18)
    green = (g > r + 10) & (g > b + 4) & (val > 55)
    gray_ground = (val > 80) & (sat < 65)
    exterior = blue_sky | green | bright_neutral

    return {
        "brightness": float(val.mean()),
        "saturation": float(sat.mean()),
        "warm_ratio": ratio(warm),
        "gold_ratio": ratio(gold),
        "red_ratio": ratio(red),
        "dark_ratio": ratio(dark),
        "bright_neutral_ratio": ratio(bright_neutral),
        "top_white_sky_ratio": ratio(white_sky_wall[top, :]),
        "top_blue_sky_ratio": ratio(blue_sky[top, :]),
        "top_green_ratio": ratio(green[top, :]),
        "bottom_gray_ratio": ratio(gray_ground[bottom, :]),
        "exterior_cue_ratio": ratio(exterior),
        "top_exterior_cue_ratio": ratio(exterior[top, :]),
    }


def f(record: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def high_precision_keep(record: Dict[str, Any], features: Dict[str, float]) -> tuple[bool, str, float]:
    reasons: List[str] = []

    body_area = f(record, "body_area_ratio")
    body_width = f(record, "body_width_ratio")
    body_height = f(record, "body_height_ratio")
    body_center_y = f(record, "body_center_y")
    body_bottom = f(record, "body_bottom_ratio")
    body_centrality = f(record, "body_centrality")
    roof_area = f(record, "roof_area_ratio")
    roof_width = f(record, "roof_width_ratio")
    roof_height = f(record, "roof_height_ratio")
    roof_center_y = f(record, "roof_center_y")
    roof_overlap = f(record, "roof_overlap_with_body")
    strict_score = f(record, "strict_score")

    top_ext = features.get("top_exterior_cue_ratio", 0.0)
    ext = features.get("exterior_cue_ratio", 0.0)
    bottom_gray = features.get("bottom_gray_ratio", 0.0)
    warm = features.get("warm_ratio", 0.0)
    red = features.get("red_ratio", 0.0)
    gold = features.get("gold_ratio", 0.0)
    dark = features.get("dark_ratio", 0.0)
    brightness = features.get("brightness", 0.0)
    saturation = features.get("saturation", 0.0)

    if features.get("decode_failed"):
        reasons.append("decode failed")
    if body_width < 0.52:
        reasons.append(f"body too narrow {body_width:.2f} < 0.52")
    if body_area < 0.32:
        reasons.append(f"body too small {body_area:.2f} < 0.32")
    if body_height < 0.42:
        reasons.append(f"body too short {body_height:.2f} < 0.42")
    if body_centrality < 0.66:
        reasons.append(f"body off center {body_centrality:.2f} < 0.66")
    if body_center_y > 0.70:
        reasons.append(f"body too low {body_center_y:.2f} > 0.70")
    if body_bottom < 0.62:
        reasons.append(f"body not grounded {body_bottom:.2f} < 0.62")
    if roof_width < 0.48:
        reasons.append(f"roof too narrow {roof_width:.2f} < 0.48")
    if roof_area < 0.10:
        reasons.append(f"roof too small {roof_area:.2f} < 0.10")
    if roof_overlap < 0.72:
        reasons.append(f"roof/body overlap weak {roof_overlap:.2f} < 0.72")
    if roof_center_y >= body_center_y - 0.10:
        reasons.append(f"roof not clearly above body {roof_center_y:.2f} vs {body_center_y:.2f}")
    if strict_score < 0.70:
        reasons.append(f"strict score low {strict_score:.2f} < 0.70")

    exterior_ok = top_ext >= 0.30 or ext >= 0.36 or bottom_gray >= 0.50
    if not exterior_ok:
        reasons.append(f"weak exterior cues top={top_ext:.2f} ext={ext:.2f} ground={bottom_gray:.2f}")

    likely_closeup_or_interior = (
        (warm > 0.55 and top_ext < 0.28 and bottom_gray < 0.40)
        or (dark > 0.42 and top_ext < 0.22)
        or (red > 0.35 and ext < 0.25)
        or (gold > 0.22 and top_ext < 0.24 and bottom_gray < 0.42)
        or (brightness < 70 and top_ext < 0.28)
        or (saturation > 135 and top_ext < 0.22 and bottom_gray < 0.42)
    )
    if likely_closeup_or_interior:
        reasons.append("color pattern looks like interior/deity/plaque/detail closeup")

    if roof_height > 0.68 and top_ext < 0.30:
        reasons.append(f"roof box fills image without exterior context {roof_height:.2f}")
    if body_area > 0.94 and top_ext < 0.24 and bottom_gray < 0.35:
        reasons.append("body fills image but exterior context is weak")

    score = (
        strict_score * 0.42
        + min(top_ext / 0.65, 1.0) * 0.22
        + min(ext / 0.55, 1.0) * 0.14
        + min(bottom_gray / 0.70, 1.0) * 0.10
        + min(body_width / 0.90, 1.0) * 0.06
        + min(roof_width / 0.90, 1.0) * 0.06
    )
    return len(reasons) == 0, "; ".join(reasons), float(score)


def load_records(path: Path) -> List[Dict[str, Any]]:
    records_path = path / "strict_facade_results.json"
    if records_path.exists():
        data = json.loads(records_path.read_text(encoding="utf-8"))
        return [item for item in data if item.get("keep") and item.get("output_image")]
    return [{"input_image": str(p), "output_image": str(p), "keep": True} for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]


def unique_destination(output_dir: Path, original_name: str) -> Path:
    dst = output_dir / original_name
    if not dst.exists():
        return dst
    stem, suffix = dst.stem, dst.suffix
    idx = 2
    while True:
        candidate = output_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="High precision post-filter for temple exterior front facades.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_dir)
    out_records: List[Dict[str, Any]] = []
    kept = 0
    csv_path = output_dir / "high_precision_facade_results.csv"
    json_path = output_dir / "high_precision_facade_results.json"
    fields = [
        "input_image",
        "output_image",
        "keep",
        "reason",
        "high_precision_score",
        "strict_score",
        "body_area_ratio",
        "body_width_ratio",
        "body_height_ratio",
        "body_center_y",
        "body_bottom_ratio",
        "roof_area_ratio",
        "roof_width_ratio",
        "roof_height_ratio",
        "roof_center_y",
        "roof_overlap_with_body",
        "top_exterior_cue_ratio",
        "exterior_cue_ratio",
        "bottom_gray_ratio",
        "warm_ratio",
        "gold_ratio",
        "red_ratio",
        "dark_ratio",
        "brightness",
        "saturation",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        for record in records:
            src = Path(str(record.get("output_image") or record.get("input_image")))
            features = color_features(src)
            keep, reason, hp_score = high_precision_keep(record, features)
            out = dict(record)
            out.update(features)
            out["keep"] = keep
            out["reason"] = reason
            out["high_precision_score"] = hp_score
            if keep:
                dst = unique_destination(output_dir, src.name)
                shutil.copy2(src, dst)
                out["output_image"] = str(dst)
                kept += 1
            else:
                out["output_image"] = ""
            out_records.append(out)
            writer.writerow({field: out.get(field, "") for field in fields})

    json_path.write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "input_path": str(input_dir),
        "output_dir": str(output_dir),
        "total": len(records),
        "kept": kept,
        "rejected": len(records) - kept,
        "csv": str(csv_path),
        "json": str(json_path),
    }
    (output_dir / "high_precision_facade_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
