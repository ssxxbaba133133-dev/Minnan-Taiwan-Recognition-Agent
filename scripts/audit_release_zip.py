# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path


REQUIRED = {
    "runtime/python.exe",
    "models-manifest.json",
    "config/runtime.conf",
    "desktop_app/models/body_yolo_best.pt",
    "desktop_app/models/kaijian_best_swinv2.pth",
    "desktop_app/models/roof_resnet34_best.pth",
    "desktop_app/models/roof_ridge_ornament_best.pt",
    "desktop_app/models/roof_yolo_best.pt",
    "desktop_app/models/tashou_best.pth",
    "desktop_app/models/tile_best.pt",
    "scripts/launcher.py",
}
FORBIDDEN = re.compile(
    r"(^|/)\.env$|(^|/)\.build/|"
    r"outputs/.+\.(?:log|png|jpe?g|json|csv)$|"
    r"data/(?:uploads|chat_uploads)/.+\.(?:png|jpe?g|zip)$|"
    r"ultralytics_config/Ultralytics/",
    re.IGNORECASE,
)


def clean(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name.replace("\\", "/")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    path = args.zip_path.resolve()
    with zipfile.ZipFile(path) as archive:
        names = {clean(info.filename) for info in archive.infolist()}
    missing = sorted(REQUIRED - names)
    forbidden = sorted(name for name in names if FORBIDDEN.search(name))
    print(f"ZIP: {path}")
    print(f"Size: {path.stat().st_size} bytes ({path.stat().st_size / 1024**3:.3f} GiB)")
    print(f"SHA256: {file_hash(path)}")
    print(f"Entries: {len(names)}")
    print(f"Missing required: {len(missing)}")
    print(f"Forbidden entries: {len(forbidden)}")
    for item in missing:
        print(f"MISSING: {item}")
    for item in forbidden[:30]:
        print(f"FORBIDDEN: {item}")
    return 1 if missing or forbidden else 0


if __name__ == "__main__":
    raise SystemExit(main())
