# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def image_data_url(path: Path, max_side: int = 768, quality: int = 85) -> str:
    with Image.open(path).convert("RGB") as img:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            img.save(tmp_path, format="JPEG", quality=quality)
            data = tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def chat_completion(base_url: str, model: str, image_path: Path, timeout: int) -> dict:
    prompt = (
        "Classify this image for a temple facade filtering task. "
        "Return only JSON, no markdown. Schema: "
        "{\"is_front_facade\": true|false, \"confidence\": 0.0-1.0, \"reason\": \"short\"}. "
        "Positive means an exterior, front-facing view of a temple building/main hall, "
        "with the main facade or entrance and roof visible. "
        "Reject deity/statue photos, shrine or altar interiors, plaques/stone tablets, "
        "decorative closeups, roof-only images, side/oblique views, partial crops, and generic gates/arches "
        "that are not the main temple building facade."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test LM Studio vision classification for temple facade images.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=os.getenv("LMSTUDIO_VISION_MODEL", "google/gemma-3-12b"))
    parser.add_argument("--base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    try:
        data = chat_completion(args.base_url, args.model, Path(args.image), args.timeout)
        message = data.get("choices", [{}])[0].get("message", {})
        print(message.get("content", ""))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
