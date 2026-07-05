# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESKTOP_APP_DIR = PROJECT_ROOT / "desktop_app"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ULTRALYTICS_CONFIG_DIR = DATA_DIR / "ultralytics_config"
try:
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    ULTRALYTICS_CONFIG_DIR = Path(tempfile.gettempdir()) / "temple_agent_ultralytics_config"
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ARCHIVE_EXTS = {".zip", ".rar"}
REGION_DETECTION_TASKS = {
    "\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b": {
        "model_file": "body_yolo_best.pt",
        "label": "\u5efa\u7b51\u4e3b\u4f53\u533a\u57df",
        "color": (14, 165, 233),
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 20,
    },
    "\u5efa\u7b51\u5c4b\u9876\u533a\u57df\u8bc6\u522b": {
        "model_file": "roof_yolo_best.pt",
        "label": "\u5efa\u7b51\u5c4b\u9876\u533a\u57df",
        "color": (34, 197, 94),
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 20,
    },
}


def _find_desktop_app_file() -> Path:
    exact = DESKTOP_APP_DIR / "pyqt5正式版.py"
    if exact.exists():
        return exact
    matches = sorted(DESKTOP_APP_DIR.glob("pyqt5*.py"))
    if not matches:
        raise FileNotFoundError(f"Cannot find desktop app python file in {DESKTOP_APP_DIR}")
    return matches[0]


class TempleRecognitionEngine:
    def __init__(self) -> None:
        self._module = None
        self._engine = None
        self._lock = threading.Lock()
        self._region_yolo_cache: Dict[str, Any] = {}

    @property
    def module(self):
        if self._module is None:
            app_file = _find_desktop_app_file()
            spec = importlib.util.spec_from_file_location("temple_desktop_app", app_file)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {app_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
        return self._module

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self.module.InferenceEngine()
        return self._engine

    def tasks(self) -> List[Dict[str, Any]]:
        model_paths = getattr(self.module, "MODEL_PATHS", {})
        task_labels = getattr(self.module, "TASK_LABELS", {})
        tasks = []
        for name, model_path in model_paths.items():
            tasks.append({
                "name": name,
                "labels": task_labels.get(name, []),
                "model_path": model_path,
                "model_exists": bool(model_path) and Path(model_path).exists(),
            })
        for name, cfg in REGION_DETECTION_TASKS.items():
            model_path = DESKTOP_APP_DIR / "models" / cfg["model_file"]
            if not any(item["name"] == name for item in tasks):
                tasks.append({
                    "name": name,
                    "labels": [cfg["label"]],
                    "model_path": str(model_path),
                    "model_exists": model_path.exists(),
                })
        return tasks

    def predict_image(self, task_name: str, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        if task_name in REGION_DETECTION_TASKS:
            return self._predict_region_detection(task_name, image_path, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            pred_name, conf, topk, detail, vis_img, chinese_pred = self.engine.predict(task_name, str(image_path))

        result_id = f"{image_path.stem}_{int(time.time() * 1000)}"
        result_image = None
        if vis_img is not None:
            result_image = output_dir / f"{result_id}_result.png"
            vis_img.save(result_image)

        return {
            "task": task_name,
            "input_image": str(image_path),
            "prediction": pred_name,
            "prediction_cn": chinese_pred,
            "confidence": float(conf),
            "topk": [{"label": name, "confidence": float(prob)} for name, prob in topk],
            "detail": detail,
            "result_image": str(result_image) if result_image else None,
        }

    def _load_region_yolo(self, task_name: str):
        if task_name in self._region_yolo_cache:
            return self._region_yolo_cache[task_name]
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for YOLO region detection tasks.") from exc
        cfg = REGION_DETECTION_TASKS[task_name]
        model_path = DESKTOP_APP_DIR / "models" / cfg["model_file"]
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        model = YOLO(str(model_path))
        self._region_yolo_cache[task_name] = model
        return model

    def _predict_region_detection(self, task_name: str, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = REGION_DETECTION_TASKS[task_name]
        model = self._load_region_yolo(task_name)
        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        results = model(
            str(image_path),
            conf=float(cfg.get("conf", 0.25)),
            iou=float(cfg.get("iou", 0.70)),
            max_det=int(cfg.get("max_det", 20)),
            verbose=False,
        )
        detections: List[Dict[str, Any]] = []
        if results and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.ones(len(boxes), dtype=float)
            for box, score in zip(boxes, confs):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({"box": (x1, y1, x2, y2), "confidence": float(score)})

        vis_img = orig_img.copy()
        draw = ImageDraw.Draw(vis_img)
        color = tuple(cfg.get("color", (34, 197, 94)))
        label = str(cfg.get("label", task_name))
        try:
            font = ImageFont.truetype("msyh.ttc", 22)
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            text = label
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_y = max(0, y1 - text_h - 8)
            draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 8], fill=color)
            draw.text((x1 + 4, label_y + 2), text, fill=(255, 255, 255), font=font)

        result_id = f"{image_path.stem}_{int(time.time() * 1000)}"
        result_image = output_dir / f"{result_id}_result.png"
        vis_img.save(result_image)

        count = len(detections)
        avg_conf = float(sum(det["confidence"] for det in detections) / count) if count else 0.0
        pred_cn = f"{label}{count}\u4e2a" if count else f"\u672a\u68c0\u6d4b\u5230{label}"
        detail_lines = [
            f"{task_name}",
            f"\u6a21\u578b\u6587\u4ef6\uff1a{cfg['model_file']}",
            f"\u68c0\u6d4b\u5230\u533a\u57df\u6570\uff1a{count}",
        ]
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["box"]
            detail_lines.append(f"\u533a\u57df{i}: ({x1}, {y1}) - ({x2}, {y2})")

        return {
            "task": task_name,
            "input_image": str(image_path),
            "prediction": "detected" if count else "not_detected",
            "prediction_cn": pred_cn,
            "confidence": avg_conf,
            "topk": [{"label": label, "confidence": avg_conf}] if count else [],
            "detail": "\n".join(detail_lines),
            "result_image": str(result_image),
        }


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_archive(path: Path) -> bool:
    return path.suffix.lower() in ARCHIVE_EXTS


def collect_images(path: Path) -> List[Path]:
    if path.is_file() and is_image(path):
        return [path]
    if path.is_dir():
        return [p for p in path.rglob("*") if p.is_file() and is_image(p)]
    return []


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (target_dir / member.filename).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Unsafe archive member path: {member.filename}")
            zf.extract(member, target_dir)


def _find_bandizip_cli() -> Optional[str]:
    candidates = [
        shutil.which("bz.exe"),
        r"D:\Bandizip\bz.exe",
        r"C:\Program Files\Bandizip\bz.exe",
        r"C:\Program Files (x86)\Bandizip\bz.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    if suffix == ".zip":
        _safe_extract_zip(archive_path, target_dir)
        return target_dir
    if suffix == ".rar":
        bz = _find_bandizip_cli()
        if not bz:
            raise RuntimeError("RAR archive support requires Bandizip command line tool bz.exe.")
        cmd = [bz, "x", "-y", "-aoa", f"-o:{target_dir}", str(archive_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Failed to extract RAR archive with Bandizip: {detail}")
        return target_dir
    raise RuntimeError(f"Unsupported archive type: {archive_path.suffix}")


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    extract_archive(zip_path, target_dir)
    return target_dir


def write_result_files(results: List[Dict[str, Any]], output_dir: Path, base_name: str = "results") -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{base_name}.json"
    csv_path = output_dir / f"{base_name}.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "task",
        "input_image",
        "image",
        "prediction",
        "prediction_cn",
        "confidence",
        "sharpness",
        "facade_score",
        "keep",
        "reason",
        "output_image",
        "result_image",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({k: item.get(k, "") for k in fields})
    return {"json": str(json_path), "csv": str(csv_path)}


def _clean_polygon_points(points: Any, width: int, height: int) -> List[tuple[float, float]]:
    cleaned: List[tuple[float, float]] = []
    last: Optional[tuple[float, float]] = None
    for raw_x, raw_y in np.asarray(points).tolist():
        x = min(max(float(raw_x), 0.0), max(float(width - 1), 0.0))
        y = min(max(float(raw_y), 0.0), max(float(height - 1), 0.0))
        point = (round(x, 2), round(y, 2))
        if point == last:
            continue
        cleaned.append(point)
        last = point
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _yolo_seg_line(class_id: int, points: List[tuple[float, float]], width: int, height: int) -> str:
    values = [str(int(class_id))]
    for x, y in points:
        values.append(f"{min(max(x / max(width, 1), 0.0), 1.0):.6f}")
        values.append(f"{min(max(y / max(height, 1), 0.0), 1.0):.6f}")
    return " ".join(values)


def _select_largest_yolo_box(result: Any) -> Optional[tuple[tuple[int, int, int, int], float]]:
    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes), dtype=float)
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    return tuple(int(v) for v in boxes[best_idx]), float(confs[best_idx])


def _unique_dataset_name(image_path: Path, used_names: set[str]) -> str:
    stem = image_path.stem
    suffix = image_path.suffix.lower() or ".jpg"
    candidate = f"{stem}{suffix}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    idx = 2
    while True:
        candidate = f"{stem}_{idx}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        idx += 1


def _zip_dir(source_dir: Path, zip_path: Path) -> str:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(source_dir))
    return str(zip_path)


def generate_ridge_yolo_labels(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Generate one YOLO segmentation label file per image using ridge ornament masks."""
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for ridge ornament YOLO label generation.") from exc

    source_path = input_path
    if input_path.is_file() and is_archive(input_path):
        source_path = extract_archive(input_path, output_dir / "_extracted_input")
    images = collect_images(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    roof_model_path = DESKTOP_APP_DIR / "models" / "roof_yolo_best.pt"
    ridge_model_path = DESKTOP_APP_DIR / "models" / "roof_ridge_ornament_best.pt"
    if not roof_model_path.exists():
        raise FileNotFoundError(f"Missing roof model: {roof_model_path}")
    if not ridge_model_path.exists():
        raise FileNotFoundError(f"Missing ridge ornament model: {ridge_model_path}")

    roof_model = YOLO(str(roof_model_path))
    ridge_model = YOLO(str(ridge_model_path))
    class_names = getattr(ridge_model, "names", {}) or {}
    ordered_names = [str(class_names.get(i, f"class_{i}")) for i in sorted(class_names)]
    if not ordered_names:
        ordered_names = ["龙", "珠", "塔", "瓶", "人物"]

    (output_dir / "classes.txt").write_text("\n".join(ordered_names) + "\n", encoding="utf-8")
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {output_dir.as_posix()}",
            "train: images",
            "val: images",
            "names:",
            *[f"  {idx}: {name}" for idx, name in enumerate(ordered_names)],
            "",
        ]),
        encoding="utf-8",
    )

    records: List[Dict[str, Any]] = []
    used_names: set[str] = set()
    total_polygons = 0

    for image_path in images:
        dataset_name = _unique_dataset_name(image_path, used_names)
        dataset_image = images_dir / dataset_name
        label_path = labels_dir / f"{Path(dataset_name).stem}.txt"
        record: Dict[str, Any] = {
            "input_image": str(image_path),
            "dataset_image": str(dataset_image),
            "label_file": str(label_path),
            "status": "empty",
            "polygons": 0,
            "reason": "",
        }
        try:
            shutil.copy2(image_path, dataset_image)
            orig_img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = orig_img.size
            roof_result = roof_model(str(image_path), conf=0.25, iou=0.70, max_det=20, verbose=False)[0]
            roof_det = _select_largest_yolo_box(roof_result)
            if roof_det is None:
                label_path.write_text("", encoding="utf-8")
                record["reason"] = "no roof detected"
                records.append(record)
                continue

            (rx1, ry1, rx2, ry2), roof_conf = roof_det
            box_w = max(1, int(rx2) - int(rx1))
            box_h = max(1, int(ry2) - int(ry1))
            pad_x = int(round(box_w * 0.06)) + 12
            pad_y = int(round(box_h * 0.06)) + 12
            crop_x1 = max(0, int(rx1) - pad_x)
            crop_y1 = max(0, int(ry1) - pad_y)
            crop_x2 = min(orig_w, int(rx2) + pad_x)
            crop_y2 = min(orig_h, int(ry2) + pad_y)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                label_path.write_text("", encoding="utf-8")
                record["reason"] = "invalid roof box"
                records.append(record)
                continue

            roi = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            ridge_result = ridge_model(
                np.array(roi),
                imgsz=960,
                conf=0.25,
                iou=0.70,
                max_det=100,
                retina_masks=True,
                verbose=False,
            )[0]
            if getattr(ridge_result, "boxes", None) is None or len(ridge_result.boxes) == 0:
                label_path.write_text("", encoding="utf-8")
                record.update({"roof_confidence": roof_conf, "reason": "no ridge ornament detected"})
                records.append(record)
                continue

            classes = (
                ridge_result.boxes.cls.cpu().numpy().astype(int)
                if ridge_result.boxes.cls is not None
                else np.zeros(len(ridge_result.boxes), dtype=int)
            )
            confs = (
                ridge_result.boxes.conf.cpu().numpy()
                if ridge_result.boxes.conf is not None
                else np.ones(len(classes), dtype=float)
            )
            mask_polygons = (
                list(ridge_result.masks.xy)
                if getattr(ridge_result, "masks", None) is not None and ridge_result.masks is not None
                else []
            )
            lines: List[str] = []
            labels: List[str] = []
            confidences: List[float] = []
            for idx, (class_id, score) in enumerate(zip(classes, confs)):
                if idx >= len(mask_polygons):
                    continue
                mapped = [(float(x) + crop_x1, float(y) + crop_y1) for x, y in mask_polygons[idx]]
                points = _clean_polygon_points(mapped, orig_w, orig_h)
                if len(points) < 3:
                    continue
                lines.append(_yolo_seg_line(int(class_id), points, orig_w, orig_h))
                labels.append(str(class_names.get(int(class_id), f"class_{int(class_id)}")))
                confidences.append(float(score))

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if lines:
                total_polygons += len(lines)
                record.update({
                    "status": "annotated",
                    "polygons": len(lines),
                    "labels": ";".join(labels),
                    "max_confidence": max(confidences) if confidences else "",
                    "roof_confidence": roof_conf,
                    "reason": "",
                })
            else:
                record.update({"roof_confidence": roof_conf, "reason": "no valid mask polygon"})
        except Exception as exc:
            try:
                label_path.write_text("", encoding="utf-8")
            except Exception:
                pass
            record.update({"status": "error", "reason": str(exc)})
        records.append(record)

    records_json = output_dir / "ridge_yolo_label_records.json"
    records_csv = output_dir / "ridge_yolo_label_records.csv"
    summary_json = output_dir / "ridge_yolo_label_summary.json"
    records_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["input_image", "dataset_image", "label_file", "status", "polygons", "labels", "max_confidence", "roof_confidence", "reason"]
    with records_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "label_count": len(list(labels_dir.glob("*.txt"))),
        "annotated_images": sum(1 for item in records if item.get("status") == "annotated"),
        "empty_images": sum(1 for item in records if item.get("status") == "empty"),
        "error_images": sum(1 for item in records if item.get("status") == "error"),
        "polygons": total_polygons,
        "classes": ordered_names,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "data_yaml": str(data_yaml),
        "records_json": str(records_json),
        "records_csv": str(records_csv),
        "summary_json": str(summary_json),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = output_dir / "ridge_yolo_labels.zip"
    summary["zip"] = _zip_dir(output_dir, zip_path)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def image_sharpness(image_path: Path) -> float:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_facade_images(input_path: Path, output_dir: Path, min_sharpness: float = 80.0) -> Dict[str, Any]:
    """Filter clear front-facade temple photos and copy the original files only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_path
    if input_path.is_file() and is_archive(input_path):
        source_path = extract_archive(input_path, output_dir / "_extracted_input")
    images = collect_images(source_path)
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    yolo_model = None
    try:
        from ultralytics import YOLO

        model_path = DESKTOP_APP_DIR / "models" / "body_yolo_best.pt"
        if model_path.exists():
            yolo_model = YOLO(str(model_path))
    except Exception:
        yolo_model = None

    for img_path in images:
        sharp = image_sharpness(img_path)
        reason = []
        keep = sharp >= min_sharpness
        if not keep:
            reason.append(f"sharpness {sharp:.1f} < {min_sharpness:.1f}")

        box_score = 0.0
        best_metrics: Dict[str, float] = {}
        if keep and yolo_model is not None:
            try:
                result = yolo_model(str(img_path), conf=0.35, iou=0.7, max_det=8, verbose=False)[0]
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                confs = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else np.ones(len(boxes), dtype=float)
                if len(boxes) == 0:
                    keep = False
                    reason.append("no building body detected")
                else:
                    data = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    h, w = img.shape[:2]
                    if min(w, h) < 320:
                        keep = False
                        reason.append(f"image too small {w}x{h}")
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
                        score = area_ratio * 0.46 + width_ratio * 0.20 + height_ratio * 0.14 + centrality * 0.14 + float(conf) * 0.06
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
                        if best_metrics["confidence"] < 0.45:
                            keep = False
                            reason.append(f"body confidence {best_metrics['confidence']:.2f} < 0.45")
                        if best_metrics["area_ratio"] < 0.10:
                            keep = False
                            reason.append(f"body area {best_metrics['area_ratio']:.2f} < 0.10")
                        if best_metrics["width_ratio"] < 0.24:
                            keep = False
                            reason.append(f"body width {best_metrics['width_ratio']:.2f} < 0.24")
                        if best_metrics["height_ratio"] < 0.16:
                            keep = False
                            reason.append(f"body height {best_metrics['height_ratio']:.2f} < 0.16")
                        if best_metrics["centrality"] < 0.35:
                            keep = False
                            reason.append(f"body not centered {best_metrics['centrality']:.2f}")
                        if best_metrics["bottom_ratio"] < 0.42:
                            keep = False
                            reason.append(f"body not front-framed {best_metrics['bottom_ratio']:.2f}")
                    if keep and box_score < 0.24:
                        keep = False
                        reason.append(f"weak facade score {box_score:.2f}")
            except Exception as exc:
                keep = False
                reason.append(f"yolo skipped: {exc}")
        elif keep and yolo_model is None:
            keep = False
            reason.append("building body model unavailable")

        record = {
            "task": "\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b",
            "image": str(img_path),
            "input_image": str(img_path),
            "sharpness": sharp,
            "facade_score": box_score,
            "keep": keep,
            "reason": "; ".join(reason),
        }
        record.update(best_metrics)
        if keep:
            temple_name = img_path.parent.name or "temple"
            ext = img_path.suffix.lower() or ".jpg"
            new_name = f"{temple_name}_{len(kept) + 1:05d}{ext}"
            dst = output_dir / new_name
            shutil.copy2(img_path, dst)
            record["output_image"] = str(dst)
            kept.append(record)
        else:
            rejected.append(record)

    files = write_result_files(kept + rejected, output_dir, base_name="facade_filter")
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total": len(images),
        "kept": len(kept),
        "rejected": len(rejected),
        "kept_preview": kept[:12],
        "records": kept + rejected,
        "files": files,
    }


recognition_engine = TempleRecognitionEngine()
