# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_ID = "nvidia/LocateAnything-3B"
BOX_TAG_RE = re.compile(r"<box>\s*(.*?)\s*</box>", re.I | re.S)
POINT_TAG_RE = re.compile(r"<point>\s*(.*?)\s*</point>", re.I | re.S)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class LocateAnythingError(RuntimeError):
    pass


def build_locate_prompt(query: str, mode: str = "all") -> str:
    query = (query or "").strip()
    if not query:
        raise LocateAnythingError("LocateAnything query is empty.")

    lowered = query.lower()
    if lowered.startswith(("locate ", "detect ", "point ", "please locate", "find ")):
        return query
    if mode == "single":
        return f"Locate a single instance that matches the following description: {query}."
    if mode == "point":
        return f"Point to: {query}."
    return f"Locate all the instances that match the following description: {query}."


def _dtype_from_name(torch: Any, name: str, device: str):
    value = (name or "").strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "full"}:
        return torch.float32
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _scale_box(values: Tuple[float, float, float, float], width: int, height: int) -> Dict[str, int]:
    x1, y1, x2, y2 = values
    coords = {
        "x1": int(round(x1 / 1000.0 * width)),
        "y1": int(round(y1 / 1000.0 * height)),
        "x2": int(round(x2 / 1000.0 * width)),
        "y2": int(round(y2 / 1000.0 * height)),
    }
    coords["x1"] = max(0, min(width, coords["x1"]))
    coords["x2"] = max(0, min(width, coords["x2"]))
    coords["y1"] = max(0, min(height, coords["y1"]))
    coords["y2"] = max(0, min(height, coords["y2"]))
    if coords["x2"] < coords["x1"]:
        coords["x1"], coords["x2"] = coords["x2"], coords["x1"]
    if coords["y2"] < coords["y1"]:
        coords["y1"], coords["y2"] = coords["y2"], coords["y1"]
    return coords


def _scale_point(values: Tuple[float, float], width: int, height: int) -> Dict[str, int]:
    x, y = values
    px = int(round(x / 1000.0 * width))
    py = int(round(y / 1000.0 * height))
    return {"x": max(0, min(width, px)), "y": max(0, min(height, py))}


def parse_locate_anything_output(text: str, width: int, height: int) -> Dict[str, List[Dict[str, int]]]:
    boxes: List[Dict[str, int]] = []
    points: List[Dict[str, int]] = []

    for match in BOX_TAG_RE.finditer(text or ""):
        values = [float(value) for value in NUM_RE.findall(match.group(1))]
        if len(values) >= 4:
            box = _scale_box((values[0], values[1], values[2], values[3]), width, height)
            if box["x2"] > box["x1"] and box["y2"] > box["y1"]:
                boxes.append(box)
        elif len(values) == 2:
            points.append(_scale_point((values[0], values[1]), width, height))

    for match in POINT_TAG_RE.finditer(text or ""):
        values = [float(value) for value in NUM_RE.findall(match.group(1))]
        if len(values) >= 2:
            points.append(_scale_point((values[0], values[1]), width, height))

    return {"boxes": boxes, "points": points}


class LocateAnythingClient:
    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        self.model_id = model_id or os.getenv("LOCATE_ANYTHING_MODEL_ID") or DEFAULT_MODEL_ID
        self.requested_device = device or os.getenv("LOCATE_ANYTHING_DEVICE") or ""
        self.requested_dtype = dtype or os.getenv("LOCATE_ANYTHING_DTYPE") or ""
        self.max_new_tokens = int(max_new_tokens or os.getenv("LOCATE_ANYTHING_MAX_NEW_TOKENS", "8192"))
        self._lock = threading.Lock()
        self._loaded = False
        self._torch = None
        self._tokenizer = None
        self._processor = None
        self._model = None
        self._device = ""
        self._dtype = None

    @property
    def device(self) -> str:
        return self._device or self.requested_device or "auto"

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                from transformers import AutoModel, AutoProcessor, AutoTokenizer
            except Exception as exc:
                raise LocateAnythingError(
                    "LocateAnything dependencies are missing. Install torch, transformers, peft, decord, and lmdb first."
                ) from exc

            device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = _dtype_from_name(torch, self.requested_dtype, device)

            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)
            model.eval()

            self._torch = torch
            self._tokenizer = tokenizer
            self._processor = processor
            self._model = model
            self._device = device
            self._dtype = dtype
            self._loaded = True

    def locate(self, image_path: Path, query: str, output_dir: Optional[Path] = None, mode: str = "all") -> Dict[str, Any]:
        self._load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._processor is not None
        assert self._model is not None

        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        prompt = build_locate_prompt(query, mode=mode)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        processor = self._processor
        if hasattr(processor, "py_apply_chat_template"):
            text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if hasattr(processor, "process_vision_info"):
            images, videos = processor.process_vision_info(messages)
        else:
            images, videos = [image], None

        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generate_kwargs: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "tokenizer": self._tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "generation_mode": os.getenv("LOCATE_ANYTHING_GENERATION_MODE", "hybrid"),
            "use_cache": True,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }
        if "pixel_values" in inputs:
            generate_kwargs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        if inputs.get("image_grid_hws") is not None:
            generate_kwargs["image_grid_hws"] = inputs["image_grid_hws"]

        try:
            with self._torch.no_grad():
                output = self._model.generate(**generate_kwargs)
        except TypeError:
            generate_kwargs.pop("generation_mode", None)
            with self._torch.no_grad():
                output = self._model.generate(**generate_kwargs)

        answer = self._decode_output(output, inputs["input_ids"])
        parsed = parse_locate_anything_output(answer, width, height)
        result_image = None
        if output_dir is not None:
            result_image = self._draw_result(image, parsed["boxes"], parsed["points"], output_dir, image_path.stem)

        located = bool(parsed["boxes"] or parsed["points"])
        return {
            "task": "LocateAnything-3B",
            "input_image": str(image_path),
            "query": query,
            "prompt": prompt,
            "answer": answer,
            "boxes": parsed["boxes"],
            "points": parsed["points"],
            "prediction": "located" if located else "not_found",
            "prediction_cn": f"{len(parsed['boxes'])} boxes, {len(parsed['points'])} points",
            "confidence": 0.0,
            "topk": [],
            "detail": answer,
            "result_image": str(result_image) if result_image else None,
        }

    def _decode_output(self, output: Any, input_ids: Any) -> str:
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
                generated = sequence[prompt_len:]
                text = self._tokenizer.decode(generated, skip_special_tokens=False)
                return text.strip()
            except Exception:
                text = self._tokenizer.decode(sequence, skip_special_tokens=False)
                return text.strip()
        return str(output).strip()

    def _draw_result(self, image: Image.Image, boxes: List[Dict[str, int]], points: List[Dict[str, int]], output_dir: Path, stem: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        try:
            font = ImageFont.truetype("msyh.ttc", 22)
        except Exception:
            font = ImageFont.load_default()

        for index, box in enumerate(boxes, 1):
            xy = [box["x1"], box["y1"], box["x2"], box["y2"]]
            draw.rectangle(xy, outline=(239, 68, 68), width=4)
            label = str(index)
            bbox = draw.textbbox((box["x1"], box["y1"]), label, font=font)
            label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            label_y = max(0, box["y1"] - label_h - 8)
            draw.rectangle([box["x1"], label_y, box["x1"] + label_w + 10, label_y + label_h + 8], fill=(239, 68, 68))
            draw.text((box["x1"] + 5, label_y + 2), label, fill=(255, 255, 255), font=font)

        for index, point in enumerate(points, 1):
            x, y = point["x"], point["y"]
            r = 8
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(14, 165, 233), outline=(255, 255, 255), width=2)
            draw.text((x + 10, y - 10), str(index), fill=(14, 165, 233), font=font)

        out_path = output_dir / f"{stem}_locate_anything_{int(time.time() * 1000)}.png"
        vis.save(out_path)
        return out_path


_CLIENT: Optional[LocateAnythingClient] = None
_CLIENT_LOCK = threading.Lock()


def get_locate_anything_client() -> LocateAnythingClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = LocateAnythingClient()
    return _CLIENT
