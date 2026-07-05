# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_ID = "nvidia/LocateAnything-3B"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Local path config. Edit these two values for everyday runs.
# DEFAULT_INPUT_PATH can be one image file or a folder containing images.
DEFAULT_INPUT_PATH = r"E:\TempleRecognitionAgent\data\uploads"
DEFAULT_OUTPUT_DIR = r"E:\TempleRecognitionAgent\outputs\locateanything_local"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float16"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_LOCAL_FILES_ONLY = True

BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)
POINT_TAG_RE = re.compile(r"<point>\s*(.*?)\s*</point>", re.I | re.S)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def collect_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []


def build_prompt(query: str, mode: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")
    lowered = query.lower()
    if lowered.startswith(("locate ", "detect ", "point ", "please locate", "find ")):
        return query
    if mode == "single":
        return f"Locate a single instance that matches the following description: {query}."
    if mode == "point":
        return f"Point to: {query}."
    return f"Locate all the instances that match the following description: {query}."


def dtype_from_name(torch: Any, name: str, device: str):
    value = (name or "").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "full"}:
        return torch.float32
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def scale_box(values: Tuple[float, float, float, float], width: int, height: int) -> Dict[str, int]:
    x1, y1, x2, y2 = values
    box = {
        "x1": int(round(x1 / 1000.0 * width)),
        "y1": int(round(y1 / 1000.0 * height)),
        "x2": int(round(x2 / 1000.0 * width)),
        "y2": int(round(y2 / 1000.0 * height)),
    }
    box["x1"] = max(0, min(width, box["x1"]))
    box["x2"] = max(0, min(width, box["x2"]))
    box["y1"] = max(0, min(height, box["y1"]))
    box["y2"] = max(0, min(height, box["y2"]))
    if box["x2"] < box["x1"]:
        box["x1"], box["x2"] = box["x2"], box["x1"]
    if box["y2"] < box["y1"]:
        box["y1"], box["y2"] = box["y2"], box["y1"]
    return box


def scale_point(values: Tuple[float, float], width: int, height: int) -> Dict[str, int]:
    x, y = values
    return {
        "x": max(0, min(width, int(round(x / 1000.0 * width)))),
        "y": max(0, min(height, int(round(y / 1000.0 * height)))),
    }


def parse_output(text: str, width: int, height: int) -> Dict[str, List[Dict[str, int]]]:
    boxes: List[Dict[str, int]] = []
    points: List[Dict[str, int]] = []
    for match in BOX_TAG_RE.finditer(text or ""):
        values = [float(value) for value in NUM_RE.findall(match.group(1))]
        if len(values) >= 4:
            box = scale_box((values[0], values[1], values[2], values[3]), width, height)
            if box["x2"] > box["x1"] and box["y2"] > box["y1"]:
                boxes.append(box)
        elif len(values) == 2:
            points.append(scale_point((values[0], values[1]), width, height))
    for match in POINT_TAG_RE.finditer(text or ""):
        values = [float(value) for value in NUM_RE.findall(match.group(1))]
        if len(values) >= 2:
            points.append(scale_point((values[0], values[1]), width, height))
    return {"boxes": boxes, "points": points}


def draw_result(image: Image.Image, boxes: List[Dict[str, int]], points: List[Dict[str, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("msyh.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    for index, box in enumerate(boxes, 1):
        draw.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]], outline=(239, 68, 68), width=4)
        label = str(index)
        bbox = draw.textbbox((box["x1"], box["y1"]), label, font=font)
        label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, box["y1"] - label_h - 8)
        draw.rectangle([box["x1"], label_y, box["x1"] + label_w + 10, label_y + label_h + 8], fill=(239, 68, 68))
        draw.text((box["x1"] + 5, label_y + 2), label, fill=(255, 255, 255), font=font)

    for index, point in enumerate(points, 1):
        x, y = point["x"], point["y"]
        radius = 8
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(14, 165, 233), outline=(255, 255, 255), width=2)
        draw.text((x + 10, y - 10), str(index), fill=(14, 165, 233), font=font)

    canvas.save(output_path)


class LocateAnythingRunner:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype_name: str,
        max_new_tokens: int,
        generation_mode: str,
        local_files_only: bool = DEFAULT_LOCAL_FILES_ONLY,
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self.dtype_name = dtype_name
        self.max_new_tokens = max_new_tokens
        self.generation_mode = generation_mode
        self.local_files_only = local_files_only
        self.torch = None
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.device = ""
        self.dtype = None

    def load(self) -> None:
        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Missing or broken dependencies. Install CUDA PyTorch first, then run: "
                f"pip install -r locateanything_local/requirements.txt. Original error: {type(exc).__name__}: {exc}"
            ) from exc

        self.torch = torch
        self.device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype_from_name(torch, self.dtype_name, self.device)
        print(f"[load] model={self.model_id} device={self.device} dtype={self.dtype}", file=sys.stderr)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self.model.eval()

    def locate_image(self, image_path: Path, prompt: str, output_dir: Path) -> Dict[str, Any]:
        if self.model is None:
            self.load()

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(self.processor, "py_apply_chat_template"):
            text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if hasattr(self.processor, "process_vision_info"):
            images, videos = self.processor.process_vision_info(messages)
        else:
            images, videos = [image], None

        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generate_kwargs: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "generation_mode": self.generation_mode,
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        if "pixel_values" in inputs:
            generate_kwargs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        if inputs.get("image_grid_hws") is not None:
            generate_kwargs["image_grid_hws"] = inputs["image_grid_hws"]

        try:
            with self.torch.no_grad():
                output = self.model.generate(**generate_kwargs)
        except TypeError:
            generate_kwargs.pop("generation_mode", None)
            with self.torch.no_grad():
                output = self.model.generate(**generate_kwargs)

        answer = self.decode_output(output, inputs["input_ids"])
        parsed = parse_output(answer, width, height)
        result_image = output_dir / f"{image_path.stem}_locate_{int(time.time() * 1000)}.png"
        draw_result(image, parsed["boxes"], parsed["points"], result_image)
        return {
            "image": str(image_path),
            "width": width,
            "height": height,
            "prompt": prompt,
            "answer": answer,
            "boxes": parsed["boxes"],
            "points": parsed["points"],
            "result_image": str(result_image),
        }

    def decode_output(self, output: Any, input_ids: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, (list, tuple)) and output:
            first = output[0]
            if isinstance(first, str):
                return first.strip()
            output = first
        if hasattr(output, "shape"):
            sequence = output[0] if len(output.shape) > 1 else output
            prompt_len = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else 0
            try:
                return self.tokenizer.decode(sequence[prompt_len:], skip_special_tokens=False).strip()
            except Exception:
                return self.tokenizer.decode(sequence, skip_special_tokens=False).strip()
        return str(output).strip()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not (args.input or "").strip():
        print("No input path configured. Edit DEFAULT_INPUT_PATH near the top of locate.py or pass --input.", file=sys.stderr)
        return 2
    input_path = Path(args.input).expanduser()
    images = collect_images(input_path)
    if not images:
        print(f"No images found: {input_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt or build_prompt(args.query, args.mode)

    runner = LocateAnythingRunner(
        model_id=args.model_id,
        device=args.device,
        dtype_name=args.dtype,
        max_new_tokens=args.max_new_tokens,
        generation_mode=args.generation_mode,
        local_files_only=not args.allow_download,
    )

    results = []
    for index, image_path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {image_path}", file=sys.stderr)
        try:
            item = runner.locate_image(image_path, prompt, output_dir)
            item["ok"] = True
            item["error"] = None
        except Exception as exc:
            item = {
                "ok": False,
                "error": str(exc),
                "image": str(image_path),
                "prompt": prompt,
                "answer": "",
                "boxes": [],
                "points": [],
                "result_image": "",
            }
            print(f"[error] {image_path}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                results.append(item)
                break
        results.append(item)

    summary = {
        "model_id": args.model_id,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "prompt": prompt,
        "count": len(results),
        "success_count": sum(1 for item in results if item.get("ok")),
        "box_count": sum(len(item.get("boxes", [])) for item in results),
        "point_count": sum(len(item.get("points", [])) for item in results),
        "results": results,
    }
    json_path = output_dir / "locate_results.json"
    save_json(json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] wrote {json_path}", file=sys.stderr)
    return 0 if summary["success_count"] == len(results) else 1


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nvidia/LocateAnything-3B locally on one image or a folder.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT_PATH, help="Image file or folder.")
    parser.add_argument("-q", "--query", required=True, help="Target description, for example: roof ridge dragon.")
    parser.add_argument("--mode", choices=["all", "single", "point"], default="all", help="Prompt template mode.")
    parser.add_argument("--prompt", default="", help="Full prompt. If set, it overrides --query and --mode template.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--model-id", default=os.getenv("LOCATE_ANYTHING_MODEL_ID", DEFAULT_MODEL_ID), help="Hugging Face model id or local model path.")
    parser.add_argument("--device", default=os.getenv("LOCATE_ANYTHING_DEVICE", DEFAULT_DEVICE), help="cuda, cuda:0, or cpu. Defaults to cuda if available.")
    parser.add_argument("--dtype", default=os.getenv("LOCATE_ANYTHING_DTYPE", DEFAULT_DTYPE), help="bfloat16, float16, or float32.")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LOCATE_ANYTHING_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))))
    parser.add_argument("--generation-mode", default=os.getenv("LOCATE_ANYTHING_GENERATION_MODE", "hybrid"))
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face network checks/downloads instead of using local cache only.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue when an image fails.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
