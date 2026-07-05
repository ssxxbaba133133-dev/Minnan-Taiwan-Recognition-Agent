# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    resize_w = min(w, 512)
    resize_h = max(1, int(h * resize_w / max(w, 1)))
    return cv2.resize(gray, (resize_w, resize_h), interpolation=cv2.INTER_AREA)


def features(path: Path) -> dict:
    gray = read_gray(path)
    if gray is None:
        return {"error": "decode failed"}
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 70, 160)
    edge_ratio = float((edges > 0).mean())
    top = edges[: max(1, h // 3), :]
    mid = edges[h // 3 : max(h // 3 + 1, 2 * h // 3), :]
    bottom = edges[max(0, 2 * h // 3) :, :]

    # Small connected components on a binary high-contrast image are a decent
    # proxy for dense plaque/tablet text.
    norm = cv2.equalizeHist(gray)
    binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = 0
    tiny = 0
    total = max(gray.size, 1)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if 3 <= ww <= 45 and 3 <= hh <= 45 and 8 <= area <= 700:
            small += 1
        if 2 <= ww <= 24 and 2 <= hh <= 24 and 5 <= area <= 220:
            tiny += 1
    small_density = small / (total / 10000.0)
    tiny_density = tiny / (total / 10000.0)

    # Long straight-line density helps identify board/tablet frames.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=max(45, w // 6), maxLineGap=8)
    line_count = 0 if lines is None else len(lines)
    return {
        "edge_ratio": edge_ratio,
        "top_edge_ratio": float((top > 0).mean()),
        "mid_edge_ratio": float((mid > 0).mean()),
        "bottom_edge_ratio": float((bottom > 0).mean()),
        "small_component_density": float(small_density),
        "tiny_component_density": float(tiny_density),
        "long_line_count": float(line_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-json", required=True)
    args = parser.parse_args()
    sample = json.loads(Path(args.sample_json).read_text(encoding="utf-8"))
    for idx, item in enumerate(sample, 1):
        f = features(Path(item))
        print(f"{idx:02d} {Path(item).name}")
        print("    " + " ".join(f"{k}={v:.3f}" for k, v in f.items() if isinstance(v, (float, int))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
