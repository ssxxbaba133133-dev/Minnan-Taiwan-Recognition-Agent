# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paged contact sheets for all images in a folder.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--thumb-w", type=int, default=220)
    parser.add_argument("--thumb-h", type=int, default=150)
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    manifest = []
    cols = max(1, args.cols)
    per_page = max(1, args.per_page)
    label_h = 40

    for page_index, page_start in enumerate(range(0, len(images), per_page), 1):
        page = images[page_start : page_start + per_page]
        rows = math.ceil(len(page) / cols) if page else 1
        sheet = Image.new("RGB", (cols * args.thumb_w, rows * (args.thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(page):
            global_index = page_start + offset + 1
            x0 = (offset % cols) * args.thumb_w
            y0 = (offset // cols) * (args.thumb_h + label_h)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((args.thumb_w, args.thumb_h), Image.LANCZOS)
                x = x0 + (args.thumb_w - img.width) // 2
                y = y0 + (args.thumb_h - img.height) // 2
                sheet.paste(img, (x, y))
            except Exception as exc:
                draw.text((x0 + 4, y0 + 10), f"ERR: {exc}", fill=(180, 0, 0))
            draw.text((x0 + 4, y0 + args.thumb_h + 2), f"{global_index:03d}", fill=(180, 0, 0))
            draw.text((x0 + 42, y0 + args.thumb_h + 2), path.name[:26], fill=(0, 0, 0))
            manifest.append({"index": global_index, "path": str(path), "page": page_index})
        out_path = out_dir / f"page_{page_index:02d}.jpg"
        sheet.save(out_path, quality=92)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(images), "pages": math.ceil(len(images) / per_page), "manifest": str(out_dir / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
