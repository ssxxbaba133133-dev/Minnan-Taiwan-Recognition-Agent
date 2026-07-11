# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data" / "ultralytics_config"))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODEL_API_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("MODEL_NAME", "local-smoke-test")

EXPECTED_TASKS = {
    "塌寿三分类",
    "屋顶四分类",
    "开间分类",
    "瓦片分类",
    "屋脊装饰识别",
    "建筑主体区域识别",
    "建筑屋顶区域识别",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-models", action="store_true")
    args = parser.parse_args()

    from backend.app import app  # noqa: F401
    from backend.temple_engine import recognition_engine

    tasks = recognition_engine.tasks()
    names = {item["name"] for item in tasks}
    if names != EXPECTED_TASKS:
        raise RuntimeError(f"任务清单不符：{sorted(names)}")
    missing = [item["model_path"] for item in tasks if not item["model_exists"]]
    if missing:
        raise RuntimeError(f"模型文件缺失：{missing}")
    print(f"[OK] 后端导入成功，7 个标准任务均已找到本地模型。")

    if args.load_models:
        module = recognition_engine.module
        engine = module.InferenceEngine(device="cpu")
        for task in sorted(EXPECTED_TASKS):
            loaded = engine.load_model(task)
            print(f"[OK] 模型可加载：{task}")
            engine.clear_task_cache(task)
            del loaded
            gc.collect()
        print("[OK] 7 个任务模型全部完成离线加载测试。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
