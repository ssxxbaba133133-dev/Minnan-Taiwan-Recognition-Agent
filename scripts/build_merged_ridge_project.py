# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASSES = ["龙", "珠", "塔", "瓶", "人物"]


@dataclass
class Instance:
    label: str
    coords: list[float]


@dataclass
class Record:
    source: str
    image_path: Path
    original_label_path: Path
    stem: str
    instances: list[Instance] = field(default_factory=list)
    assigned_name: str = ""

    @property
    def labels(self) -> Counter[str]:
        return Counter(instance.label for instance in self.instances)


def read_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}


def class_name(label: str) -> str:
    text = str(label).strip()
    if text.lower() == "long":
        return "龙"
    return text


def normalize_points(points: Iterable[Iterable[float]], width: int, height: int) -> list[float]:
    coords: list[float] = []
    for point in points:
        pair = list(point)
        if len(pair) < 2:
            continue
        x = min(max(float(pair[0]) / max(width, 1), 0.0), 1.0)
        y = min(max(float(pair[1]) / max(height, 1), 0.0), 1.0)
        coords.extend([x, y])
    return coords


def find_image_for_label(image_dir: Path, label_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{label_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    for candidate in image_dir.glob(f"{label_path.stem}.*"):
        if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file():
            return candidate
    return None


def collect_old_yolo_records(dataset_dir: Path, classes: list[str]) -> list[Record]:
    records: list[Record] = []
    for subset in ("train", "val"):
        image_dir = dataset_dir / "images" / subset
        label_dir = dataset_dir / "labels" / subset
        if not image_dir.exists() or not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image_for_label(image_dir, label_path)
            if image_path is None:
                continue
            instances: list[Instance] = []
            for raw_line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw_line.strip().split()
                if len(parts) < 7:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    continue
                if class_id < 0 or class_id >= len(classes):
                    continue
                if len(coords) < 6 or len(coords) % 2 != 0:
                    continue
                instances.append(Instance(label=classes[class_id], coords=coords))
            if instances:
                records.append(
                    Record(
                        source=f"old_{subset}",
                        image_path=image_path,
                        original_label_path=label_path,
                        stem=label_path.stem,
                        instances=instances,
                    )
                )
    return records


def collect_new_labelme_records(labelme_dir: Path, classes: list[str]) -> list[Record]:
    records: list[Record] = []
    for image_path in sorted(p for p in labelme_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        label_path = image_path.with_suffix(".json")
        if not label_path.exists():
            continue
        data = read_json(label_path)
        with Image.open(image_path) as img:
            width, height = img.size
        instances: list[Instance] = []
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            label = class_name(shape.get("label", ""))
            points = shape.get("points") or []
            shape_type = shape.get("shape_type") or "polygon"
            if not label or shape_type != "polygon" or len(points) < 3:
                continue
            if label not in classes:
                classes.append(label)
            coords = normalize_points(points, width, height)
            if len(coords) >= 6 and len(coords) % 2 == 0:
                instances.append(Instance(label=label, coords=coords))
        if instances:
            records.append(
                Record(
                    source="new_labelme",
                    image_path=image_path,
                    original_label_path=label_path,
                    stem=label_path.stem,
                    instances=instances,
                )
            )
    return records


def safe_name(path: Path, used: set[str]) -> str:
    name = path.name
    if name not in used:
        used.add(name)
        return name
    stem, suffix = path.stem, path.suffix
    idx = 2
    while True:
        candidate = f"{stem}_{idx}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def assign_unique_names(records: list[Record]) -> None:
    used: set[str] = set()
    for record in sorted(records, key=lambda item: (item.source, item.image_path.name, str(item.image_path))):
        record.assigned_name = safe_name(record.image_path, used)


def count_instances(records: Iterable[Record]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.labels)
    return counts


def count_images_with_class(records: Iterable[Record]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for label in record.labels:
            counts[label] += 1
    return counts


def stratified_split(records: list[Record], classes: list[str], val_ratio: float, seed: int) -> tuple[list[Record], list[Record]]:
    rng = random.Random(seed)
    remaining = records[:]
    rng.shuffle(remaining)
    total_instances = count_instances(records)
    total_images = count_images_with_class(records)
    target_val_images = max(1, round(len(records) * val_ratio))
    target_instance_counts = {label: total_instances[label] * val_ratio for label in classes}
    target_image_counts = {label: total_images[label] * val_ratio for label in classes}

    val: list[Record] = []
    val_instances: Counter[str] = Counter()
    val_images: Counter[str] = Counter()

    def split_error(next_instances: Counter[str], next_images: Counter[str], size: int) -> float:
        instance_error = sum((next_instances[label] - target_instance_counts[label]) ** 2 for label in classes)
        image_error = sum((next_images[label] - target_image_counts[label]) ** 2 for label in classes)
        size_error = (size - target_val_images) ** 2
        return instance_error + image_error * 0.35 + size_error * 0.10

    def add_record(record: Record) -> None:
        val.append(record)
        val_instances.update(record.labels)
        for label in record.labels:
            val_images[label] += 1
        remaining.remove(record)

    # Represent each class in validation when possible.
    for label in sorted(classes, key=lambda name: total_instances[name]):
        if total_instances[label] <= 1:
            continue
        if val_instances[label] > 0:
            continue
        candidates = [record for record in remaining if record.labels[label] > 0]
        if not candidates:
            continue
        candidates.sort(
            key=lambda record: (
                split_error(val_instances + record.labels, val_images + Counter(record.labels.keys()), len(val) + 1),
                sum(record.labels.values()),
                record.assigned_name,
            )
        )
        add_record(candidates[0])

    while len(val) < target_val_images and remaining:
        scored: list[tuple[float, float, Record]] = []
        current_error = split_error(val_instances, val_images, len(val))
        for record in remaining:
            trial_images = val_images.copy()
            for label in record.labels:
                trial_images[label] += 1
            trial_instances = val_instances + record.labels
            gain = current_error - split_error(trial_instances, trial_images, len(val) + 1)
            rare_bonus = sum(record.labels[label] / max(total_instances[label], 1) for label in record.labels)
            scored.append((gain + rare_bonus * 0.20, rng.random(), record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        add_record(scored[0][2])

    train = remaining
    return train, val


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_project_files(old_project: Path, new_project: Path) -> None:
    new_project.mkdir(parents=True, exist_ok=True)
    scripts_src = old_project / "scripts"
    scripts_dst = new_project / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    if scripts_src.exists():
        for item in scripts_src.iterdir():
            if item.name in {"__pycache__", "runs"}:
                continue
            if item.is_file() and item.suffix.lower() in {".py", ".yaml", ".yml", ".pt"}:
                shutil.copy2(item, scripts_dst / item.name)

    for name in ("README_training.md", "requirements.txt", "yolo26s-seg.pt", "yolo26n.pt"):
        src = old_project / name
        if src.exists() and src.is_file():
            shutil.copy2(src, new_project / name)


def copy_labelme_reviewed(new_labelme: Path, new_project: Path) -> None:
    dst = new_project / "labelme_reviewed_new"
    reset_dir(dst)
    for item in new_labelme.iterdir():
        if item.is_file() and (item.suffix.lower() in IMAGE_SUFFIXES or item.suffix.lower() == ".json"):
            shutil.copy2(item, dst / item.name)


def write_dataset(output_dir: Path, classes: list[str], train: list[Record], val: list[Record]) -> None:
    reset_dir(output_dir)
    class_to_id = {label: idx for idx, label in enumerate(classes)}
    for subset, records in (("train", train), ("val", val)):
        image_dir = output_dir / "images" / subset
        label_dir = output_dir / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            image_dst = image_dir / record.assigned_name
            label_dst = label_dir / f"{Path(record.assigned_name).stem}.txt"
            shutil.copy2(record.image_path, image_dst)
            lines = []
            for instance in record.instances:
                if instance.label not in class_to_id:
                    continue
                coords = " ".join(f"{value:.6f}" for value in instance.coords)
                lines.append(f"{class_to_id[instance.label]} {coords}")
            label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_data = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: label for idx, label in enumerate(classes)},
    }
    (output_dir / "roof_ridge_ornament.yaml").write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def write_reports(output_dir: Path, classes: list[str], all_records: list[Record], train: list[Record], val: list[Record]) -> None:
    all_instances = count_instances(all_records)
    train_instances = count_instances(train)
    val_instances = count_instances(val)
    all_images = count_images_with_class(all_records)
    train_images = count_images_with_class(train)
    val_images = count_images_with_class(val)

    with (output_dir / "class_distribution.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "class",
            "total_instances",
            "train_instances",
            "val_instances",
            "val_instance_ratio",
            "images_with_class",
            "train_images_with_class",
            "val_images_with_class",
            "val_image_ratio",
        ])
        for label in classes:
            total_i = all_instances[label]
            total_img = all_images[label]
            writer.writerow([
                label,
                total_i,
                train_instances[label],
                val_instances[label],
                f"{val_instances[label] / total_i:.4f}" if total_i else "0.0000",
                total_img,
                train_images[label],
                val_images[label],
                f"{val_images[label] / total_img:.4f}" if total_img else "0.0000",
            ])

    with (output_dir / "labels_per_file.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["subset", "source", "image", "original_image", "instances", *classes])
        for subset, records in (("train", train), ("val", val)):
            for record in sorted(records, key=lambda item: item.assigned_name):
                labels = record.labels
                writer.writerow([
                    subset,
                    record.source,
                    record.assigned_name,
                    str(record.image_path),
                    sum(labels.values()),
                    *[labels[label] for label in classes],
                ])

    source_counts: dict[str, Any] = {}
    for source in sorted({record.source for record in all_records}):
        source_records = [record for record in all_records if record.source == source]
        source_counts[source] = {
            "images": len(source_records),
            "instances": dict(count_instances(source_records)),
        }

    summary = {
        "dataset_dir": str(output_dir.resolve()),
        "classes": classes,
        "total_images": len(all_records),
        "train_images": len(train),
        "val_images": len(val),
        "total_instances": dict(all_instances),
        "train_instances": dict(train_instances),
        "val_instances": dict(val_instances),
        "images_with_class": dict(all_images),
        "train_images_with_class": dict(train_images),
        "val_images_with_class": dict(val_images),
        "sources": source_counts,
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a merged roof-ridge ornament YOLO segmentation project.")
    parser.add_argument("--old-project", required=True)
    parser.add_argument("--new-project", required=True)
    parser.add_argument("--new-labelme", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    old_project = Path(args.old_project)
    new_project = Path(args.new_project)
    new_labelme = Path(args.new_labelme)
    old_dataset = old_project / "datasets" / "roof_ridge_ornament_yolo"
    old_yaml = old_dataset / "roof_ridge_ornament.yaml"
    output_dataset = new_project / "datasets" / "roof_ridge_ornament_yolo"

    if not old_project.exists():
        raise FileNotFoundError(f"Old project not found: {old_project}")
    if not old_dataset.exists():
        raise FileNotFoundError(f"Old YOLO dataset not found: {old_dataset}")
    if not new_labelme.exists():
        raise FileNotFoundError(f"New LabelMe folder not found: {new_labelme}")
    if any(new_project.iterdir()) if new_project.exists() else False:
        allowed = {"datasets", "scripts", "labelme_reviewed_new", "README_training.md", "requirements.txt", "yolo26s-seg.pt", "yolo26n.pt"}
        unexpected = [item.name for item in new_project.iterdir() if item.name not in allowed]
        if unexpected:
            raise RuntimeError(f"New project is not empty and has unexpected files: {unexpected}")

    yaml_data = read_yaml(old_yaml)
    names = yaml_data.get("names") or {}
    if isinstance(names, dict):
        classes = [str(names[idx]) for idx in sorted(names, key=lambda value: int(value))]
    elif isinstance(names, list):
        classes = [str(item) for item in names]
    else:
        classes = DEFAULT_CLASSES[:]
    if not classes:
        classes = DEFAULT_CLASSES[:]

    copy_project_files(old_project, new_project)
    copy_labelme_reviewed(new_labelme, new_project)

    records = collect_old_yolo_records(old_dataset, classes)
    records.extend(collect_new_labelme_records(new_labelme, classes))
    if not records:
        raise RuntimeError("No labeled records found.")
    assign_unique_names(records)
    train, val = stratified_split(records, classes, args.val_ratio, args.seed)
    write_dataset(output_dataset, classes, train, val)
    write_reports(output_dataset, classes, records, train, val)

    summary = read_json(output_dataset / "merge_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
