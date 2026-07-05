# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a random image contact sheet for quick review.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    images = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    random.seed(args.seed)
    sample = random.sample(images, min(args.count, len(images)))

    thumb_w, thumb_h = 220, 150
    label_h = 34
    cols = max(1, args.cols)
    rows = math.ceil(len(sample) / cols) if sample else 1
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(sample):
        x0 = (i % cols) * thumb_w
        y0 = (i // cols) * (thumb_h + label_h)
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            x = x0 + (thumb_w - img.width) // 2
            y = y0 + (thumb_h - img.height) // 2
            sheet.paste(img, (x, y))
            label = f"{i + 1:02d} {path.name[:28]}"
            draw.text((x0 + 4, y0 + thumb_h + 2), label, fill=(0, 0, 0))
        except Exception as exc:
            draw.text((x0 + 4, y0 + 10), f"ERR {path.name}: {exc}", fill=(180, 0, 0))

    sheet.save(out, quality=92)
    out.with_suffix(".json").write_text(
        json.dumps([str(p) for p in sample], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
