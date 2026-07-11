# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics_config"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TASKS = [
    "塌寿三分类",
    "屋顶四分类",
    "开间分类",
    "瓦片分类",
    "屋脊装饰识别",
    "建筑主体区域识别",
    "建筑屋顶区域识别",
]


def make_test_image(path: Path) -> None:
    image = Image.new("RGB", (512, 512), (232, 226, 211))
    draw = ImageDraw.Draw(image)
    draw.rectangle((90, 230, 422, 455), fill=(158, 54, 45), outline=(70, 35, 25), width=6)
    draw.polygon([(55, 235), (256, 72), (457, 235)], fill=(62, 88, 84), outline=(35, 45, 42))
    draw.rectangle((206, 320, 306, 455), fill=(95, 51, 35))
    image.save(path)


def main() -> int:
    from backend.temple_engine import recognition_engine

    module = recognition_engine.module
    engine = module.InferenceEngine(device="cpu")
    with tempfile.TemporaryDirectory(prefix="temple_agent_smoke_") as tmp:
        image_path = Path(tmp) / "synthetic_temple.png"
        make_test_image(image_path)
        for task in TASKS:
            prediction, confidence, _topk, _detail, _visual, _chinese = engine.predict(task, str(image_path))
            print(f"[OK] {task}: {prediction!s} (confidence={float(confidence):.4f})")
            engine.clear_task_cache(task)
            gc.collect()
    print("[OK] 7 个任务均完成一次 CPU 前向推理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
