# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .temple_engine import DATA_DIR, OUTPUT_DIR, collect_images, image_sharpness
from scripts.high_precision_facade_pass import color_features
from scripts.inspect_image_structure_features import features as structure_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WORKFLOW_DIR = OUTPUT_DIR / "workflow"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

QUOTED_PATH_RE = re.compile(r'["“”\']([A-Za-z]:\\[^"“”\']+)["“”\']?')
UNQUOTED_PATH_RE = re.compile(r"(?<!\w)([A-Za-z]:\\[^\s\"“”'<>|]+)")
EXCLUDE_LIST_RE = re.compile(r"(?:排除|剔除|不要|去掉|exclude)\s*[:：]?\s*([0-9,\s，、\-]+)", re.I)


def workflow_tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "id": "local_facade_filter",
            "name": "本地正立面宽筛",
            "description": "从本地图片文件夹筛出宫庙建筑正立面，并复制到目标文件夹，同时生成结果表和复核图。",
            "examples": ["把 F:\\台灣宮廟照片 里的宫庙建筑正立面筛选出来，复制到 F:\\新建文件夹"],
        },
        {
            "id": "contact_sheet",
            "name": "分页复核图",
            "description": "把图片文件夹或结果清单做成带编号的分页缩略图，方便人工检查误判。",
            "examples": ["为 F:\\新建文件夹_召回增强正立面 生成复核图"],
        },
        {
            "id": "manual_review_selection",
            "name": "按编号复核复制",
            "description": "根据复核图编号排除误图，并把保留图片复制到新文件夹。",
            "examples": ["按编号排除 1,3,7，从 manifest.json 复制复核结果到 F:\\复核后"],
        },
        {
            "id": "big_recall",
            "name": "大召回",
            "description": "按文件名 _02 规律补召回可能漏掉的正立面，并自动生成复核清单。",
            "examples": ["对 F:\\台灣宮廟照片 做大召回，输出到 F:\\新建文件夹_大召回正立面"],
        },
        {
            "id": "multistage",
            "name": "多轮复核",
            "description": "按宽筛、严格筛、高精度筛的顺序运行多阶段流程，并输出每轮结果与复核图。",
            "examples": ["对 F:\\台灣宮廟照片 跑多轮复核流程"],
        },
        {
            "id": "summary",
            "name": "结果统计",
            "description": "读取已有结果文件，统计保留、排除数量和主要误判原因。",
            "examples": ["统计 F:\\新建文件夹_大召回正立面 的误判原因"],
        },
    ]


def _strip_path_noise(value: str) -> str:
    return value.strip().rstrip(".,;，。；：:!?)]}】」'")


def extract_local_paths(message: str) -> List[str]:
    msg = message or ""
    found: List[str] = []
    for match in QUOTED_PATH_RE.finditer(msg):
        found.append(_strip_path_noise(match.group(1)))
    for match in UNQUOTED_PATH_RE.finditer(msg):
        found.append(_strip_path_noise(match.group(1)))

    deduped: List[str] = []
    seen = set()
    for path in found:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _message_without_paths(message: str) -> str:
    msg = message or ""
    msg = QUOTED_PATH_RE.sub(" ", msg)
    msg = UNQUOTED_PATH_RE.sub(" ", msg)
    return re.sub(r"\s+", " ", msg).strip()


def extract_exclude_indices(message: str) -> List[int]:
    msg = message or ""
    match = EXCLUDE_LIST_RE.search(msg)
    if not match:
        return []
    raw = match.group(1)
    values: List[int] = []
    for token in re.split(r"[,，、\s]+", raw.strip()):
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            if left.isdigit() and right.isdigit():
                start, end = int(left), int(right)
                if start <= end:
                    values.extend(range(start, end + 1))
                else:
                    values.extend(range(end, start + 1))
            continue
        if token.isdigit():
            values.append(int(token))
    return sorted(set(v for v in values if v > 0))


def _unique_destination(output_dir: Path, original_name: str, used_names: set[str]) -> Path:
    dst = output_dir / original_name
    if dst.name not in used_names and not dst.exists():
        used_names.add(dst.name)
        return dst
    stem, suffix = dst.stem, dst.suffix
    idx = 2
    while True:
        candidate = output_dir / f"{stem}_{idx}{suffix}"
        if candidate.name not in used_names and not candidate.exists():
            used_names.add(candidate.name)
            return candidate
        idx += 1


def _run_python_script(script_name: str, args: Sequence[str], timeout: int = 7200) -> Dict[str, Any]:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0:
        detail = (proc.stderr or stdout or "").strip()
        raise RuntimeError(detail or f"{script_name} failed with exit code {proc.returncode}")
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except Exception:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stdout[start : end + 1])
            except Exception:
                pass
        return {"raw_stdout": stdout}


def _load_manifest_entries(source: Path) -> List[Dict[str, Any]]:
    if source.is_file() and source.suffix.lower() == ".json":
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            data = []
        entries: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for idx, item in enumerate(data, 1):
                if isinstance(item, str):
                    path = item
                elif isinstance(item, dict):
                    path = item.get("output_image") or item.get("input_image") or item.get("path") or ""
                else:
                    continue
                if path:
                    entries.append({"index": idx, "path": path})
        return entries

    if source.is_dir():
        images = _list_images(source)
        return [{"index": idx, "path": str(path)} for idx, path in enumerate(images, 1)]
    return []


def _list_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if not path.is_dir():
        return []

    top_level = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if top_level:
        return top_level
    return collect_images(path)


def _iter_top_level_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _image_size(path: Path) -> Tuple[int, int]:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return w, h


def _render_paged_contact_sheet(images: List[Path], out_dir: Path, per_page: int, cols: int, thumb_w: int, thumb_h: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    label_h = 40
    cols = max(1, cols)
    per_page = max(1, per_page)

    page_paths: List[str] = []
    for page_index, page_start in enumerate(range(0, len(images), per_page), 1):
        page = images[page_start : page_start + per_page]
        rows = max(1, int(np.ceil(len(page) / cols)))
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(page):
            global_index = page_start + offset + 1
            x0 = (offset % cols) * thumb_w
            y0 = (offset // cols) * (thumb_h + label_h)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
                x = x0 + (thumb_w - img.width) // 2
                y = y0 + (thumb_h - img.height) // 2
                sheet.paste(img, (x, y))
            except Exception as exc:
                draw.text((x0 + 4, y0 + 10), f"ERR: {exc}", fill=(180, 0, 0))
            draw.text((x0 + 4, y0 + thumb_h + 2), f"{global_index:03d}", fill=(180, 0, 0))
            draw.text((x0 + 42, y0 + thumb_h + 2), path.name[:26], fill=(0, 0, 0))
            manifest.append({"index": global_index, "path": str(path), "page": page_index})
        out_path = out_dir / f"page_{page_index:02d}.jpg"
        sheet.save(out_path, quality=92)
        page_paths.append(str(out_path))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "count": len(images),
        "pages": len(page_paths),
        "manifest": str(manifest_path),
        "page_images": page_paths,
    }


def create_paged_contact_sheets(source: Path, output_dir: Path, per_page: int = 50, cols: int = 5, thumb_w: int = 220, thumb_h: int = 150) -> Dict[str, Any]:
    images = _list_images(source)
    if not images and source.is_file() and source.suffix.lower() == ".json":
        entries = _load_manifest_entries(source)
        images = [Path(item["path"]) for item in entries if Path(item["path"]).exists()]
    if not images:
        raise FileNotFoundError(f"No images found in {source}")
    return _render_paged_contact_sheet(images, output_dir, per_page, cols, thumb_w, thumb_h)


def copy_manual_review_selection(manifest_or_folder: Path, output_dir: Path, exclude: Sequence[int]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exclude_set = {int(x) for x in exclude if int(x) > 0}
    entries = _load_manifest_entries(manifest_or_folder)
    if not entries:
        raise FileNotFoundError(f"No manifest or images found in {manifest_or_folder}")

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    used_names = {p.name for p in output_dir.iterdir() if p.is_file()}

    for item in entries:
        idx = int(item["index"])
        src = Path(item["path"])
        if idx in exclude_set:
            rejected.append(item)
            continue
        if not src.exists():
            rejected.append({**item, "error": "source missing"})
            continue
        dst = _unique_destination(output_dir, src.name, used_names)
        shutil.copy2(src, dst)
        kept_item = dict(item)
        kept_item["output_image"] = str(dst)
        kept.append(kept_item)

    summary = {
        "manifest": str(manifest_or_folder),
        "output_dir": str(output_dir),
        "total": len(entries),
        "kept": len(kept),
        "rejected": len(rejected),
        "exclude": sorted(exclude_set),
    }
    (output_dir / "manual_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manual_review_kept.json").write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manual_review_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **summary,
        "kept_preview": kept[:12],
        "files": {
            "summary": str(output_dir / "manual_review_summary.json"),
            "kept": str(output_dir / "manual_review_kept.json"),
            "rejected": str(output_dir / "manual_review_rejected.json"),
        },
    }


def summarize_review_results(source: Path) -> Dict[str, Any]:
    candidates: List[Path] = []
    if source.is_file():
        candidates.append(source)
    elif source.is_dir():
        for name in [
            "filename02_recall_results.json",
            "manual_review_kept.json",
            "manual_review_rejected.json",
            "manual_review_summary.json",
            "facade_filter_results.json",
            "strict_facade_results.json",
            "high_precision_facade_results.json",
        ]:
            p = source / name
            if p.exists():
                candidates.append(p)
        candidates.extend(sorted(source.glob("*results.json")))

    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and {"kept", "rejected"} <= set(data.keys()):
            return {"source": str(candidate), "summary": data}
        if isinstance(data, list):
            keep_count = sum(1 for item in data if isinstance(item, dict) and item.get("keep"))
            reason_counter = Counter()
            for item in data:
                if not isinstance(item, dict) or item.get("keep"):
                    continue
                reason = str(item.get("reason", "")).strip()
                if not reason:
                    continue
                key = reason.split("; ", 1)[0]
                reason_counter[key] += 1
            return {
                "source": str(candidate),
                "summary": {
                    "total": len(data),
                    "kept": keep_count,
                    "rejected": len(data) - keep_count,
                    "top_reject_reasons": reason_counter.most_common(10),
                },
            }
    raise FileNotFoundError(f"No readable result file found in {source}")


def _looks_like_contact_request(message: str) -> bool:
    msg = _message_without_paths(message).lower()
    words = ["复核图", "contact sheet", "contactsheet", "缩略图", "分页", "预览图", "页面预览", "手工复核"]
    return any(word in msg for word in words)


def _looks_like_selection_request(message: str) -> bool:
    msg = _message_without_paths(message)
    return any(word in msg for word in ["排除编号", "排除", "剔除", "不要这些", "保留这些", "copy", "复制"]) and bool(extract_exclude_indices(msg))


def _looks_like_big_recall_request(message: str) -> bool:
    msg = _message_without_paths(message).lower()
    words = ["大召回", "召回增强", "补召回", "提高召回", "_02", "filename02", "二号图", "二图"]
    return any(word in msg for word in words)


def _looks_like_multistage_request(message: str) -> bool:
    msg = _message_without_paths(message)
    words = ["多轮复核", "纠错流程", "复核流程", "分阶段筛选", "多阶段筛选", "先宽后严", "宽筛再严筛"]
    return any(word in msg for word in words)


def _looks_like_local_facade_filter_request(message: str) -> bool:
    msg = _message_without_paths(message)
    filter_words = ["筛选", "筛出", "找出", "挑出", "复制", "拷贝", "保存", "输出", "放到", "放进"]
    facade_words = ["正立面", "正面", "正向", "正脸", "门面", "正门", "庙门", "宫庙建筑", "建筑主体"]
    return any(word in msg for word in filter_words) and any(word in msg for word in facade_words)


def _resolve_source_output(message: str, input_roots: List[Path], run_id: str) -> Tuple[Optional[Path], Path]:
    paths = [Path(p) for p in extract_local_paths(message)]
    intent_msg = _message_without_paths(message)
    default_output = WORKFLOW_DIR / run_id
    default_output.mkdir(parents=True, exist_ok=True)

    if len(paths) >= 2:
        source = paths[0] if paths[0].exists() else (input_roots[0] if input_roots else None)
        return source, paths[1]
    if len(paths) == 1:
        source = paths[0]
        if input_roots and any(word in intent_msg for word in ["到", "输出", "导出", "复制到", "保存到", "放到"]):
            return input_roots[0], source
        if source.exists():
            return source, default_output
        return None, source
    if input_roots:
        return input_roots[0], default_output
    return None, default_output


def _workflow_response(reply: str, route: Dict[str, Any], output_dir: Path, files: Dict[str, str], images: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "reply": reply,
        "route": route,
        "results": results,
        "files": files,
        "images": images,
        "sources": [],
        "show_files": True,
        "output_dir": str(output_dir),
    }


def _run_big_recall(source: Path, output_dir: Path, include_folder: Optional[Path] = None) -> Dict[str, Any]:
    args = ["--source", str(source), "--output", str(output_dir)]
    if include_folder and include_folder.exists():
        args.extend(["--include-folder", str(include_folder)])
    return _run_python_script("filename02_recall_pass.py", args, timeout=14400)


def _run_broad_facade_filter(source: Path, output_dir: Path) -> Dict[str, Any]:
    return _run_python_script(
        "filter_facade_to_folder.py",
        ["--input", str(source), "--output", str(output_dir), "--min-sharpness", "80.0", "--progress-every", "200"],
        timeout=14400,
    )


def _run_multistage_facade_workflow(source: Path, output_dir: Path) -> Dict[str, Any]:
    stages = {
        "broad": output_dir / "01_broad_body",
        "strict": output_dir / "02_strict_body_roof",
        "high_precision": output_dir / "03_high_precision",
    }
    broad = _run_python_script(
        "filter_facade_to_folder.py",
        ["--input", str(source), "--output", str(stages["broad"]), "--min-sharpness", "80.0", "--progress-every", "200"],
        timeout=14400,
    )
    strict = _run_python_script(
        "strict_facade_second_pass.py",
        ["--input", str(stages["broad"]), "--output", str(stages["strict"]), "--min-sharpness", "80.0", "--progress-every", "100"],
        timeout=14400,
    )
    high = _run_python_script(
        "high_precision_facade_pass.py",
        ["--input", str(stages["strict"]), "--output", str(stages["high_precision"])],
        timeout=14400,
    )
    strict_sheets = _render_paged_contact_sheet(_list_images(stages["strict"]), output_dir / "04_strict_review", 50, 5, 220, 150)
    high_sheets = _render_paged_contact_sheet(_list_images(stages["high_precision"]), output_dir / "05_high_precision_review", 50, 5, 220, 150)
    return {
        "stages": stages,
        "broad": broad,
        "strict": strict,
        "high_precision": high,
        "strict_sheets": strict_sheets,
        "high_sheets": high_sheets,
    }


def maybe_handle_workflow_request(message: str, input_roots: List[Path], run_id: str) -> Optional[Dict[str, Any]]:
    msg = message or ""
    intent_msg = _message_without_paths(msg)
    source, output_dir = _resolve_source_output(msg, input_roots, run_id)
    exclude = extract_exclude_indices(msg)
    path_values = extract_local_paths(msg)
    explicit_output = len(path_values) >= 2 or (
        len(path_values) == 1
        and bool(input_roots)
        and any(word in intent_msg for word in ["到", "输出", "导出", "复制到", "保存到", "放到"])
    )

    if _looks_like_selection_request(msg):
        if source is None:
            return {
                "reply": "我需要一个复核清单或包含图片的文件夹路径，才能按编号排除并复制结果。",
                "route": {"intent": "workflow", "tool": "selection"},
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
                "show_files": False,
                "output_dir": "",
            }
        selection_output = output_dir if explicit_output else output_dir / "manual_selection"
        result = copy_manual_review_selection(source, selection_output, exclude)
        images = [
            {
                "url": item["output_image"],
                "title": Path(item["output_image"]).name,
                "detail": f"index={item['index']}",
            }
            for item in result.get("kept_preview", [])
            if item.get("output_image")
        ]
        reply = f"已按编号复核并复制到：{result['output_dir']}。共保留 {result['kept']} 张，排除 {result['rejected']} 张。"
        return _workflow_response(reply, {"intent": "workflow", "tool": "manual_review_selection"}, Path(result["output_dir"]), result["files"], images, result.get("kept_preview", []))

    if _looks_like_big_recall_request(msg):
        if source is None:
            return {
                "reply": "我需要一个源文件夹路径，才能跑大召回流程。",
                "route": {"intent": "workflow", "tool": "big_recall"},
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
                "show_files": False,
                "output_dir": "",
            }
        big_output = output_dir if explicit_output else output_dir / "big_recall"
        include_folder = input_roots[0] if input_roots and input_roots[0].exists() and input_roots[0] != source else None
        result = _run_big_recall(source, big_output, include_folder=include_folder)
        out_dir = Path(result["output_dir"])
        review_output = WORKFLOW_DIR / run_id / "big_recall_review"
        sheet = create_paged_contact_sheets(out_dir, review_output, per_page=50, cols=5, thumb_w=220, thumb_h=150) if _iter_top_level_images(out_dir) else {}
        images = [
            {"url": item, "title": Path(item).name, "detail": "big recall review page"}
            for item in sheet.get("page_images", [])[:MAX_PREVIEW_PAGES]
        ]
        reply = (
            f"已完成大召回：源文件 {result.get('filename02_total', 0)} 张，新保留 {result.get('new_kept', 0)} 张，"
            f"合并后输出 {result.get('total_output_images', 0)} 张。结果在：{result.get('output_dir', '')}。"
        )
        return _workflow_response(
            reply,
            {"intent": "workflow", "tool": "big_recall"},
            big_output,
            {
                "summary": str(Path(result["output_dir"]) / "filename02_recall_summary.json"),
                "results": result.get("results_json", ""),
                "review_manifest": sheet.get("manifest", ""),
            },
            images,
            [],
        )

    if _looks_like_multistage_request(msg):
        if source is None:
            return {
                "reply": "我需要一个源文件夹路径，才能跑多轮复核流程。",
                "route": {"intent": "workflow", "tool": "multistage"},
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
                "show_files": False,
                "output_dir": "",
            }
        multi_output = output_dir if explicit_output else output_dir / "multistage"
        result = _run_multistage_facade_workflow(source, multi_output)
        images = []
        for item in result["high_sheets"]["page_images"][:MAX_PREVIEW_PAGES]:
            p = Path(item)
            images.append({"url": item, "title": p.name, "detail": "high precision review page"})
        broad_total = result["broad"].get("kept", 0)
        strict_total = result["strict"].get("kept", 0)
        high_total = result["high_precision"].get("kept", 0)
        reply = f"已完成多轮复核：宽筛 {broad_total} 张，严格 {strict_total} 张，高精度 {high_total} 张。复核页在：{multi_output}。"
        return _workflow_response(
            reply,
            {"intent": "workflow", "tool": "multistage"},
            multi_output,
            {
                "broad": result["broad"].get("json", ""),
                "strict": result["strict"].get("json", ""),
                "high_precision": result["high_precision"].get("json", ""),
                "strict_manifest": result["strict_sheets"].get("manifest", ""),
                "high_manifest": result["high_sheets"].get("manifest", ""),
            },
            images,
            [],
        )

    if _looks_like_local_facade_filter_request(msg) and source is not None:
        facade_output = output_dir if explicit_output else output_dir / "facade_filter"
        result = _run_broad_facade_filter(source, facade_output)
        out_dir = Path(result.get("output_dir") or facade_output)
        images = _iter_top_level_images(out_dir)
        contact_output = WORKFLOW_DIR / run_id / "facade_filter_review"
        sheet = create_paged_contact_sheets(out_dir, contact_output, per_page=50, cols=5, thumb_w=220, thumb_h=150) if images else {}
        preview_images = [
            {"url": item, "title": Path(item).name, "detail": "facade review page"}
            for item in sheet.get("page_images", [])[:MAX_PREVIEW_PAGES]
        ]
        files = {
            "csv": result.get("csv", ""),
            "json": result.get("json", ""),
            "summary": str(out_dir / "facade_filter_summary.json"),
        }
        if sheet.get("manifest"):
            files["review_manifest"] = sheet["manifest"]
        reply = (
            f"已完成本地正立面宽筛：共检查 {result.get('total', 0)} 张，保留 {result.get('kept', 0)} 张，"
            f"复制到：{out_dir}。我也生成了分页复核图，方便你后续按编号排除误图。"
        )
        return _workflow_response(reply, {"intent": "workflow", "tool": "local_facade_filter"}, out_dir, files, preview_images, [])

    if _looks_like_contact_request(msg):
        if source is None:
            return {
                "reply": "我需要一个图片文件夹或清单路径，才能生成分页复核图。",
                "route": {"intent": "workflow", "tool": "contact_sheet"},
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
                "show_files": False,
                "output_dir": "",
            }
        contact_output = output_dir if explicit_output else output_dir / "contact_sheets"
        result = create_paged_contact_sheets(source, contact_output, per_page=50, cols=5, thumb_w=220, thumb_h=150)
        images = [
            {
                "url": item,
                "title": Path(item).name,
                "detail": "contact sheet page",
            }
            for item in result["page_images"][:MAX_PREVIEW_PAGES]
        ]
        reply = f"已生成分页复核图，共 {result['count']} 张，分成 {result['pages']} 页，清单在：{result['manifest']}。"
        return _workflow_response(reply, {"intent": "workflow", "tool": "contact_sheet"}, contact_output, {"manifest": result["manifest"]}, images, [])

    if any(word in intent_msg for word in ["漏检", "误判", "原因统计", "统计原因", "summary", "top reason"]):
        if source is None:
            return None
        summary = summarize_review_results(source)
        stats = summary.get("summary", {})
        reply = f"已统计 {summary.get('source', '')}：总计 {stats.get('total', 0)} 张，保留 {stats.get('kept', 0)} 张，排除 {stats.get('rejected', 0)} 张。"
        if stats.get("top_reject_reasons"):
            top = "；".join(f"{name} ×{count}" for name, count in stats["top_reject_reasons"][:5])
            reply += f" 主要误判原因：{top}。"
        return _workflow_response(reply, {"intent": "workflow", "tool": "summary"}, source if source.is_dir() else source.parent, {}, [], [])

    return None


MAX_PREVIEW_PAGES = 12
