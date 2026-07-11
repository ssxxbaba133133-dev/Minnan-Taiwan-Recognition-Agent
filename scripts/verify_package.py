# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_models(full: bool) -> list[str]:
    errors: list[str] = []
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = manifest.get("models", [])
    if len(records) != manifest.get("model_count"):
        errors.append("模型清单数量不一致")
    actual_total = 0
    for record in records:
        path = ROOT / record["file"]
        if not path.is_file():
            errors.append(f"缺少模型：{record['file']}")
            continue
        size = path.stat().st_size
        actual_total += size
        if size != record["bytes"]:
            errors.append(f"模型大小不符：{record['file']}（{size} != {record['bytes']}）")
            continue
        if full and sha256(path) != record["sha256"].upper():
            errors.append(f"模型 SHA256 不符：{record['file']}")
        print(f"[OK] {record['file']} ({size / 1024 / 1024:.2f} MB)")
    if actual_total != manifest.get("total_bytes"):
        errors.append(f"模型总大小不符：{actual_total} != {manifest.get('total_bytes')}")
    return errors


def verify_imports() -> list[str]:
    errors: list[str] = []
    modules = [
        "fastapi", "uvicorn", "multipart", "cv2", "numpy", "PIL",
        "pandas", "openpyxl", "torch", "torchvision", "timm",
        "ultralytics", "PyQt5",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
            print(f"[OK] Python 模块：{name}")
        except Exception as exc:
            errors.append(f"Python 模块不可用：{name}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the portable agent package.")
    parser.add_argument("--quick", action="store_true", help="Check model existence and byte sizes.")
    parser.add_argument("--full", action="store_true", help="Also verify SHA256 hashes.")
    parser.add_argument("--imports", action="store_true", help="Import every required runtime module.")
    args = parser.parse_args()

    errors = verify_models(full=args.full)
    if args.imports:
        errors.extend(verify_imports())
    if errors:
        print("\n验证失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\n完整性验证通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
