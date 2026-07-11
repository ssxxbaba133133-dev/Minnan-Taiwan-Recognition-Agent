# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "fastapi", "uvicorn", "python-multipart", "opencv-python", "numpy",
    "pillow", "pandas", "openpyxl", "torch", "torchvision", "timm",
    "ultralytics", "PyQt5",
]


def main() -> int:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "executable": "runtime/python.exe",
        "packages": versions,
        "sys_prefix_is_runtime": Path(sys.prefix).resolve() == (ROOT / "runtime").resolve(),
    }
    target = ROOT / "runtime-manifest.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] 已生成 {target.name}")
    return 0 if all(versions.values()) and payload["sys_prefix_is_runtime"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
