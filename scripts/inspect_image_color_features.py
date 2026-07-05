# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def features(path: Path) -> dict:
    rgb = read_rgb(path)
    if rgb is None:
        return {"error": "decode failed"}
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (min(w, 512), max(1, int(h * min(w, 512) / max(w, 1)))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    r = small[:, :, 0].astype(np.int16)
    g = small[:, :, 1].astype(np.int16)
    b = small[:, :, 2].astype(np.int16)
    hh = small.shape[0]
    top = slice(0, max(1, hh // 3))
    bot = slice(max(0, hh * 2 // 3), hh)

    warm = (((hue <= 28) | (hue >= 170)) & (sat > 70) & (val > 50))
    gold = ((hue >= 18) & (hue <= 45) & (sat > 60) & (val > 70))
    red = (((hue <= 10) | (hue >= 170)) & (sat > 80) & (val > 50))
    dark = val < 55
    bright_neutral = (val > 145) & (sat < 70)
    white_wall_sky = (val > 175) & (sat < 95)
    blue_sky = (b > r + 12) & (b > g + 4) & (val > 110) & (sat > 18)
    green = (g > r + 10) & (g > b + 4) & (val > 55)
    gray_ground = (val > 80) & (sat < 65)

    def ratio(mask):
        return float(mask.mean())

    return {
        "brightness": float(val.mean()),
        "saturation": float(sat.mean()),
        "warm_ratio": ratio(warm),
        "gold_ratio": ratio(gold),
        "red_ratio": ratio(red),
        "dark_ratio": ratio(dark),
        "bright_neutral_ratio": ratio(bright_neutral),
        "top_bright_neutral_ratio": ratio(bright_neutral[top, :]),
        "top_white_sky_ratio": ratio(white_wall_sky[top, :]),
        "top_blue_sky_ratio": ratio(blue_sky[top, :]),
        "top_green_ratio": ratio(green[top, :]),
        "bottom_gray_ratio": ratio(gray_ground[bot, :]),
        "exterior_cue_ratio": ratio(blue_sky | green | bright_neutral),
        "top_exterior_cue_ratio": ratio((blue_sky | green | bright_neutral)[top, :]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-json", required=True)
    args = parser.parse_args()
    sample = json.loads(Path(args.sample_json).read_text(encoding="utf-8"))
    for idx, item in enumerate(sample, 1):
        f = features(Path(item))
        print(f"{idx:02d} {Path(item).name}")
        print(
            "    "
            + " ".join(
                f"{k}={v:.3f}" for k, v in f.items() if isinstance(v, (float, int))
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
