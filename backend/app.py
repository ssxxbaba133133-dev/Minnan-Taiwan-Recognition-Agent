# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import time
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .lmstudio_client import LMStudioClient
from .facade_workflow import maybe_handle_workflow_request, workflow_tool_specs
from .locate_anything_client import LocateAnythingError, get_locate_anything_client
from .temple_engine import DATA_DIR, OUTPUT_DIR, collect_images, extract_archive, filter_facade_images, generate_ridge_yolo_labels, is_archive, recognition_engine, write_result_files
from .web_search import search_web


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Temple Recognition Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []


class FacadeFilterRequest(BaseModel):
    input_path: str
    output_dir: Optional[str] = None
    min_sharpness: float = 80.0


MAX_CHAT_PREVIEW_IMAGES = 12
ENABLE_LOCATE_ANYTHING = os.getenv("ENABLE_LOCATE_ANYTHING", "0").strip().lower() in {"1", "true", "yes", "on"}
BODY_REGION_TASK = "\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b"
ROOF_REGION_TASK = "\u5efa\u7b51\u5c4b\u9876\u533a\u57df\u8bc6\u522b"


def _public_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if str(path).startswith(("http://", "https://", "file://", "/files/")):
        return path
    p = Path(path)
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT.resolve())
        return f"/files/{rel.as_posix()}"
    except Exception:
        try:
            if p.is_absolute() and p.exists():
                return p.as_uri()
        except Exception:
            pass
        return path


def _save_upload(upload: UploadFile, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(upload.filename or f"upload_{int(time.time())}").name
    dst = target_dir / filename
    with dst.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dst


def _form_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _task_names() -> List[str]:
    return [item["name"] for item in recognition_engine.tasks()]


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _looks_facade_filter_request(message: str, input_count: int = 0, has_archive: bool = False) -> bool:
    """Deterministic guardrail for the batch facade-screening tool.

    This tool must win before the LLM router whenever the user is asking to
    select/copy/filter many temple front-facade photos. Otherwise the generic
    visual router may choose classification or ridge-ornament tasks.
    """
    msg = message or ""
    filter_words = [
        "\u7b5b\u9009", "\u7b5b\u4e00\u4e0b", "\u7b5b\u51fa", "\u6311\u9009", "\u6311\u51fa", "\u9009\u51fa",
        "\u627e\u51fa", "\u627e\u5230", "\u8fc7\u6ee4", "\u4fdd\u7559", "\u7559\u4e0b", "\u5254\u9664",
        "\u6392\u9664", "\u53bb\u6389", "\u5220\u9664", "\u5220\u6389", "\u653e\u5230", "\u653e\u8fdb",
        "\u5b58\u5230", "\u4fdd\u5b58", "\u65b0\u6587\u4ef6\u5939", "\u590d\u5236", "\u62f7\u8d1d",
        "\u6574\u7406", "\u5f52\u7c7b",
    ]
    facade_words = [
        "\u6b63\u7acb\u9762", "\u7acb\u9762", "\u6b63\u9762", "\u6b63\u5411", "\u6b63\u8138", "\u95e8\u9762", "\u5e99\u95e8", "\u6b63\u95e8",
        "\u5927\u95e8", "\u6709\u62cd\u5230", "\u62cd\u5230", "\u5efa\u7b51\u4e3b\u4f53", "\u4e3b\u4f53\u533a\u57df",
        "\u5b8c\u6574\u4e3b\u4f53", "\u5bab\u5e99\u5efa\u7b51", "\u5bab\u5e99", "\u5bfa\u5e99", "\u5e99\u5b87",
    ]
    batch_words = [
        "\u6279\u91cf", "\u5927\u91cf", "\u4e00\u6279", "\u8fd9\u6279", "\u8fd9\u4e9b", "\u91cc\u9762",
        "\u6570\u636e\u96c6", "\u6587\u4ef6\u5939", "\u76ee\u5f55", "\u538b\u7f29\u5305", "\u5305\u91cc",
        "\u56fe\u7247", "\u7167\u7247", "\u56fe\u50cf", "\u6e05\u6670", "\u7b26\u5408\u8981\u6c42",
    ]
    other_task_words = [
        "\u5c4b\u810a", "\u810a\u9970", "\u74e6\u7247", "\u5c4b\u9876\u6837\u5f0f", "\u5c4b\u9876\u533a\u57df",
        "\u5efa\u7b51\u5c4b\u9876", "\u5f00\u95f4", "\u584c\u5bff",
    ]
    selection_question_words = [
        "\u54ea\u4e9b", "\u54ea\u51e0\u5f20", "\u54ea\u5f20", "\u6709\u6ca1\u6709", "\u662f\u5426", "\u5305\u542b",
        "\u7b26\u5408", "\u9700\u8981\u7684", "\u80fd\u7528\u7684",
    ]
    has_filter = _contains_any(msg, filter_words)
    has_facade = _contains_any(msg, facade_words)
    has_other_task = _contains_any(msg, other_task_words)
    is_batch = input_count > 1 or has_archive
    if has_facade:
        if has_filter:
            return True
        if is_batch and _contains_any(msg, selection_question_words) and not has_other_task:
            return True
    if not has_filter:
        return False
    # For many uploaded images or archives, a generic "filter/select photos" request
    # should use the facade filter unless the user explicitly named another task.
    if is_batch and _contains_any(msg, batch_words) and not has_other_task:
        return True
    return False


def _facade_filter_route(message: str) -> dict:
    return {
        "intent": "facade_filter",
        "tasks": [BODY_REGION_TASK],
        "needs_export": _wants_export(message),
    }


def _looks_ridge_yolo_annotation_request(message: str) -> bool:
    msg = (message or "").lower()
    ridge_words = ["屋脊", "脊饰", "脊飾", "屋脊装饰", "屋脊裝飾", "ridge", "ornament"]
    yolo_words = ["yolo", "标注", "標註", "标签", "標籤", "label", "annotation", "txt"]
    polygon_words = ["多边形", "多邊形", "polygon", "分割", "seg", "segmentation", "mask"]
    output_words = ["生成", "导出", "導出", "输出", "輸出", "文件", "标注文件", "標註文件", "兼容"]
    return (
        _contains_any(msg, ridge_words)
        and _contains_any(msg, yolo_words)
        and (_contains_any(msg, polygon_words) or _contains_any(msg, output_words))
    )


def _looks_locate_anything_request(message: str) -> bool:
    if not ENABLE_LOCATE_ANYTHING:
        return False
    msg = (message or "").lower()
    markers = [
        "locateanything",
        "locate anything",
        "\u901a\u7528\u5b9a\u4f4d",
        "\u901a\u7528\u89c6\u89c9\u5b9a\u4f4d",
        "\u76ee\u6807\u5b9a\u4f4d",
        "\u89c6\u89c9\u5b9a\u4f4d",
        "\u7528\u8fd9\u4e2a\u6a21\u578b",
        "\u7528\u5b9a\u4f4d\u6a21\u578b",
    ]
    return any(marker in msg for marker in markers)


def _locate_anything_query(message: str) -> str:
    query = (message or "").strip()
    replacements = [
        "LocateAnything",
        "locateanything",
        "Locate Anything",
        "locate anything",
        "\u901a\u7528\u5b9a\u4f4d",
        "\u901a\u7528\u89c6\u89c9\u5b9a\u4f4d",
        "\u76ee\u6807\u5b9a\u4f4d",
        "\u89c6\u89c9\u5b9a\u4f4d",
        "\u7528\u8fd9\u4e2a\u6a21\u578b",
        "\u7528\u5b9a\u4f4d\u6a21\u578b",
    ]
    for token in replacements:
        query = query.replace(token, " ")
    query = " ".join(query.replace("\uff1a", " ").replace(":", " ").split()).strip(" ,.;\uff0c\u3002")
    return query or (message or "").strip()


def _locate_anything_response(query: str, inputs: List[Path], output_dir: Path, mode: str = "all") -> dict:
    if not inputs:
        raise HTTPException(status_code=400, detail="\u6ca1\u6709\u627e\u5230\u53ef\u7528\u4e8e\u5b9a\u4f4d\u7684\u56fe\u7247\u3002")
    results = []
    try:
        client = get_locate_anything_client()
        for image_path in inputs:
            item = client.locate(image_path, query=query, output_dir=output_dir, mode=mode)
            item["ok"] = True
            item["error"] = None
            item["result_image_url"] = _public_path(item.get("result_image"))
            results.append(item)
    except LocateAnythingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    files_out = write_result_files(results, output_dir, base_name="locate_anything_results")
    box_count = sum(len(item.get("boxes", [])) for item in results)
    point_count = sum(len(item.get("points", [])) for item in results)
    images = [
        {
            "url": item.get("result_image_url"),
            "title": f"LocateAnything - {item.get('query', '')}",
            "detail": item.get("answer", ""),
        }
        for item in results[:MAX_CHAT_PREVIEW_IMAGES]
        if item.get("result_image_url")
    ]
    return {
        "reply": f"LocateAnything \u5df2\u5b8c\u6210\u76ee\u6807\u5b9a\u4f4d\uff1a\u5171\u5904\u7406 {len(inputs)} \u5f20\u56fe\uff0c\u5f97\u5230 {box_count} \u4e2a\u6846\u548c {point_count} \u4e2a\u70b9\u3002",
        "route": {"intent": "locate_anything", "query": query, "mode": mode},
        "tasks": ["LocateAnything-3B"],
        "count": len(results),
        "success_count": len(results),
        "error_count": 0,
        "results": results[:200],
        "images": images,
        "files": {k: _public_path(v) for k, v in files_out.items()},
        "sources": [],
        "show_files": True,
        "output_dir": str(output_dir),
    }


def _choose_tasks(message: str) -> List[str]:
    msg = message or ""
    tasks = _task_names()
    roof_region_words = ["\u5c4b\u9876\u533a\u57df", "\u5c4b\u9876\u4f4d\u7f6e", "\u5c4b\u9876\u68c0\u6d4b", "\u5c4b\u9876\u6846", "\u6846\u51fa\u5c4b\u9876"]

    def has(name: str) -> bool:
        return name in tasks

    selected: List[str] = []
    if any(word in msg for word in ["\u7efc\u5408", "\u5404\u79cd", "\u6784\u6210", "\u8981\u7d20", "\u5168\u90e8", "\u591a\u4efb\u52a1", "\u7c7b\u578b"]):
        for name in ["\u5c4b\u810a\u88c5\u9970\u8bc6\u522b", "\u5c4b\u9876\u56db\u5206\u7c7b", "\u5f00\u95f4\u5206\u7c7b", "\u74e6\u7247\u5206\u7c7b"]:
            if has(name):
                selected.append(name)
    else:
        if any(word in msg for word in ["\u4e3b\u4f53\u533a\u57df", "\u5efa\u7b51\u4e3b\u4f53", "\u5bab\u5e99\u4e3b\u4f53", "\u4e3b\u4f53\u68c0\u6d4b", "\u4e3b\u4f53\u4f4d\u7f6e", "\u4e3b\u4f53\u6846"]):
            if has(BODY_REGION_TASK):
                selected.append(BODY_REGION_TASK)
        if any(word in msg for word in roof_region_words):
            if has(ROOF_REGION_TASK):
                selected.append(ROOF_REGION_TASK)
        rules = [
            ("\u5c4b\u810a\u88c5\u9970\u8bc6\u522b", ["\u5c4b\u810a", "\u810a\u9970", "\u88c5\u9970", "\u9f99", "\u73e0", "\u5854", "\u74f6", "\u4eba\u7269"]),
            ("\u5c4b\u9876\u56db\u5206\u7c7b", ["\u5c4b\u9876", "\u5c4b\u9762", "\u4e09\u5ddd", "\u65ad\u6a90"]),
            ("\u5f00\u95f4\u5206\u7c7b", ["\u5f00\u95f4", "\u9762\u9614"]),
            ("\u74e6\u7247\u5206\u7c7b", ["\u74e6", "\u74e6\u7247", "\u677f\u74e6", "\u7b52\u74e6"]),
        ]
        for task_name, keywords in rules:
            if has(task_name) and any(keyword in msg for keyword in keywords):
                selected.append(task_name)

        if any(word in msg for word in roof_region_words) and not any(word in msg for word in ["\u6837\u5f0f", "\u7c7b\u578b", "\u5206\u7c7b"]):
            selected = [task for task in selected if task != "\u5c4b\u9876\u56db\u5206\u7c7b"]

    if not selected and BODY_REGION_TASK in tasks:
        selected = [BODY_REGION_TASK]
    return selected


def _looks_general_chat(message: str) -> bool:
    msg = (message or "").lower()
    markers = [
        "\u4f60\u662f\u4ec0\u4e48\u6a21\u578b",
        "\u4f60\u662f\u8c01",
        "\u4f60\u53eb\u4ec0\u4e48",
        "\u4f60\u80fd\u505a\u4ec0\u4e48",
        "\u4ecb\u7ecd\u4e00\u4e0b\u4f60",
        "\u6a21\u578b",
        "model",
        "lm studio",
        "lmstudio",
    ]
    return any(marker in msg for marker in markers)


def _deterministic_general_answer(message: str) -> Optional[str]:
    msg = (message or "").lower()
    model_markers = [
        "\u4f60\u662f\u4ec0\u4e48\u6a21\u578b",
        "\u4f60\u7684\u57fa\u7840\u6a21\u578b",
        "\u57fa\u7840\u6a21\u578b",
        "\u5e95\u5c42\u6a21\u578b",
        "\u5927\u6a21\u578b",
        "\u6a21\u578b",
        "model",
    ]
    if any(marker in msg for marker in model_markers):
        model_name = LMStudioClient().model
        return (
            f"\u6211\u7684\u57fa\u7840\u8bed\u8a00\u6a21\u578b\u6765\u81ea\u8fdc\u7a0b OpenAI \u517c\u5bb9 API\uff0c"
            f"\u5f53\u524d\u914d\u7f6e\u7684\u6a21\u578b\u662f {model_name}\uff0c"
            "\u56fe\u50cf\u8bc6\u522b\u90e8\u5206\u4f1a\u6839\u636e\u672c\u673a\u662f\u5426\u63d0\u4f9b YOLO \u548c\u5206\u7c7b\u6743\u91cd\u6765\u8fd0\u884c\u3002"
        )
    identity_markers = [
        "\u4f60\u662f\u8c01",
        "\u4f60\u53eb\u4ec0\u4e48",
        "\u4ecb\u7ecd\u4e00\u4e0b\u4f60",
    ]
    if any(marker in msg for marker in identity_markers):
        return "\u6211\u662f\u95fd\u53f0\u5bab\u5e99\u5efa\u7b51\u56fe\u50cf\u8bc6\u522b Agent\uff0c\u53ef\u4ee5\u5e2e\u4f60\u505a\u56fe\u50cf\u8bc6\u522b\u3001\u6279\u91cf\u7b5b\u9009\u548c\u7ed3\u679c\u6574\u7406\u3002"
    ability_markers = [
        "\u4f60\u80fd\u505a\u4ec0\u4e48",
        "\u6709\u4ec0\u4e48\u529f\u80fd",
        "\u529f\u80fd",
    ]
    if any(marker in msg for marker in ability_markers):
        return (
            "\u6211\u53ef\u4ee5\u8bc6\u522b\u5bab\u5e99\u5efa\u7b51\u7684\u5c4b\u9876\u6837\u5f0f\u3001\u5f00\u95f4\u7c7b\u578b\u3001"
            "\u74e6\u7247\u7c7b\u578b\u3001\u5c4b\u810a\u88c5\u9970\u3001\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u548c\u5efa\u7b51\u5c4b\u9876\u533a\u57df\uff0c"
            "\u4e5f\u80fd\u5bf9\u672c\u5730\u6587\u4ef6\u5939\u505a\u6279\u91cf\u7b5b\u9009\u3001\u751f\u6210\u590d\u6838\u56fe\u3001"
            "\u6309\u7f16\u53f7\u6392\u9664\u3001\u505a\u591a\u8f6e\u590d\u6838\u3001\u751f\u6210\u5927\u53ec\u56de\u7ed3\u679c\u5e76\u5bfc\u51fa\u7ed3\u679c\u3002"
        )
    return None


def _final_reply_text(reply: dict, fallback: str = "") -> str:
    content = (reply.get("content") or "").strip()
    if content:
        return content
    return fallback or "\u6211\u8fd9\u6b21\u6ca1\u6709\u751f\u6210\u6709\u6548\u56de\u7b54\uff0c\u8bf7\u6362\u4e00\u79cd\u95ee\u6cd5\u6216\u91cd\u8bd5\u3002"


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}


def _parse_chat_history(raw_history: str) -> List[dict]:
    try:
        data = json.loads(raw_history or "[]")
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    items: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            items.append({"role": role, "content": content})
    return items


def _prior_history(history_items: List[dict], current_message: str, max_items: int = 12) -> List[dict]:
    items = list(history_items)
    current = (current_message or "").strip()
    if items and items[-1].get("role") == "user" and items[-1].get("content") == current:
        items = items[:-1]
    return items[-max_items:]


def _messages_with_history(system: str, history_items: List[dict], current_message: str) -> List[dict]:
    messages = [{"role": "system", "content": system}]
    messages.extend(_prior_history(history_items, current_message))
    messages.append({"role": "user", "content": current_message})
    return messages


def _history_context_text(history_items: List[dict], current_message: str, max_items: int = 8) -> str:
    lines = []
    for item in _prior_history(history_items, current_message, max_items=max_items):
        role = "User" if item.get("role") == "user" else "Assistant"
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _route_with_llm(
    message: str,
    has_images: bool,
    input_count: int = 0,
    has_archive: bool = False,
    allow_search: bool = True,
    history_items: Optional[List[dict]] = None,
) -> dict:
    task_names = _task_names()
    if has_images and _looks_facade_filter_request(message, input_count=input_count, has_archive=has_archive):
        return _facade_filter_route(message)
    history_context = _history_context_text(history_items or [], message)
    history_instruction = (
        f"Recent conversation:\n{history_context}\n\n"
        if history_context
        else ""
    )
    search_instruction = (
        "If the user asks for latest/current/recent information, online sources, web search, papers, news, websites, external facts, or asks you to look something up, use intent=search. "
        if allow_search
        else "Web search is disabled for this request, so do not use intent=search. Treat online/recent/current information requests as chat unless the image requires visual analysis. "
    )
    intent_schema = "chat|visual|search|facade_filter" if allow_search else "chat|visual|facade_filter"
    prompt = (
        "You are an intent router for a temple image recognition agent. "
        "Return only one JSON object, no markdown. "
        f"Current request has_images={has_images}, input_count={input_count}, has_archive={has_archive}. "
        "If the user asks to filter/select/copy/keep images containing clear front facades, temple front views, temple gates, or main temple building fronts into a folder, use intent=facade_filter and only task \u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b. "
        "For facade_filter never add roof, ridge ornament, tile, bay, or classification tasks. "
        "If the user asks about the assistant itself, the model, abilities, identity, or anything not asking about the image, use intent=chat. "
        f"{search_instruction}"
        "If the user asks about the uploaded/current image, temple building, roof, ridge ornaments, tiles, bays, facade, components, style, count, or recognition result, use intent=visual. "
        "If has_images=false, do not use intent=visual. "
        "Choose tasks from this exact list only: "
        f"{task_names}. "
        "Task mapping: roof style/classification -> \u5c4b\u9876\u56db\u5206\u7c7b; roof region/location/bounding box/detect roof -> \u5efa\u7b51\u5c4b\u9876\u533a\u57df\u8bc6\u522b; building body/main building/facade body region -> \u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b; bays/kaijian -> \u5f00\u95f4\u5206\u7c7b; tiles -> \u74e6\u7247\u5206\u7c7b; ridge ornaments/dragon/pearl/tower/bottle/person -> \u5c4b\u810a\u88c5\u9970\u8bc6\u522b; comprehensive/components/elements -> use roof style, bays, tiles, and ridge ornament tasks. "
        "Also set needs_export=true only if the user asks to export/download/report/csv/json/excel/statistics. "
        f"JSON schema: {{\"intent\":\"{intent_schema}\",\"tasks\":[\"task name\"],\"needs_export\":false}}. "
        f"{history_instruction}"
        f"User query: {message}"
    )
    try:
        reply = LMStudioClient().chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
        route = _parse_json_object(reply.get("content", ""))
    except Exception:
        route = {}

    intent = str(route.get("intent", "")).lower()
    tasks = [task for task in route.get("tasks", []) if task in task_names]
    needs_export = bool(route.get("needs_export", _wants_export(message)))

    if intent == "chat" or _looks_general_chat(message):
        return {"intent": "chat", "tasks": [], "needs_export": needs_export}
    if allow_search and (intent == "search" or _looks_search_request(message)):
        return {"intent": "search", "tasks": [], "needs_export": needs_export}
    if not allow_search and intent == "search":
        intent = "chat"
    if intent == "facade_filter" and has_images:
        return _facade_filter_route(message)
    if not has_images:
        return {"intent": "chat", "tasks": [], "needs_export": needs_export}
    if intent != "visual":
        intent = "visual"
    if not tasks:
        tasks = _choose_tasks(message)
    roof_region_words = ["\u5c4b\u9876\u533a\u57df", "\u5c4b\u9876\u4f4d\u7f6e", "\u5c4b\u9876\u68c0\u6d4b", "\u5c4b\u9876\u6846", "\u6846\u51fa\u5c4b\u9876"]
    if any(word in (message or "") for word in roof_region_words) and not any(word in (message or "") for word in ["\u6837\u5f0f", "\u7c7b\u578b", "\u5206\u7c7b"]):
        tasks = [task for task in tasks if task != "\u5c4b\u9876\u56db\u5206\u7c7b"]
        if ROOF_REGION_TASK in task_names and ROOF_REGION_TASK not in tasks:
            tasks.insert(0, ROOF_REGION_TASK)
    return {"intent": intent, "tasks": tasks, "needs_export": needs_export}


def _looks_search_request(message: str) -> bool:
    msg = (message or "").lower()
    markers = [
        "\u6700\u65b0",
        "\u6700\u8fd1",
        "\u4eca\u5929",
        "\u5f53\u524d",
        "\u65b0\u95fb",
        "\u8054\u7f51",
        "\u7f51\u4e0a",
        "\u641c\u7d22",
        "\u67e5\u4e00\u4e0b",
        "\u8bba\u6587",
        "\u8d44\u6599",
        "\u7f51\u5740",
        "\u6765\u6e90",
        "latest",
        "recent",
        "search",
        "web",
        "paper",
        "news",
    ]
    return any(marker in msg for marker in markers)


def _search_fallback_summary(message: str, usable: List[dict]) -> str:
    titles = [str(src.get("title", "")).strip() for src in usable if src.get("title")]
    snippets = [str(src.get("snippet", "")).strip() for src in usable if src.get("snippet")]
    haystack = "\n".join(titles + snippets)
    themes = []
    if any(word in haystack for word in ["\u88c5\u9970", "\u827a\u672f", "\u5c4b\u810a", "\u526a\u9ecf", "\u4ea4\u8dbe"]):
        themes.append("\u5bab\u5e99\u88c5\u9970\u827a\u672f\u548c\u5c4b\u810a\u88c5\u9970\u7814\u7a76")
    if any(word in haystack for word in ["\u7a7a\u95f4", "\u5efa\u7b51", "\u5883\u57df", "\u5f62\u5236"]):
        themes.append("\u95fd\u53f0\u5bab\u5e99\u5efa\u7b51\u7a7a\u95f4\u4e0e\u5f62\u5236\u7814\u7a76")
    if any(word in haystack for word in ["\u6587\u5316", "\u5386\u53f2", "\u4fdd\u62a4", "\u4f20\u627f"]):
        themes.append("\u5bab\u5e99\u6587\u5316\u5386\u53f2\u3001\u4f20\u627f\u4e0e\u4fdd\u62a4\u7814\u7a76")
    if any(word in haystack.lower() for word in ["\u56fe\u50cf", "\u8bc6\u522b", "\u6df1\u5ea6\u5b66\u4e60", "yolo", "computer vision"]):
        themes.append("\u56fe\u50cf\u8bc6\u522b\u6216\u6570\u5b57\u5316\u65b9\u5411\u7684\u4ea4\u53c9\u7814\u7a76")
    if not themes:
        themes = ["\u5bab\u5e99\u5efa\u7b51\u53ca\u76f8\u5173\u6587\u5316\u7814\u7a76"]

    theme_text = "\u3001".join(themes)
    representative = "\u3001".join(titles[:3])
    if representative:
        return (
            f"\u6839\u636e\u8fd9\u6b21\u8054\u7f51\u641c\u7d22\uff0c\u76f8\u5173\u7814\u7a76\u4e3b\u8981\u96c6\u4e2d\u5728"
            f"{theme_text}\u3002"
            f"\u641c\u7d22\u7ed3\u679c\u4e2d\u8f83\u5177\u4ee3\u8868\u6027\u7684\u6761\u76ee\u5305\u62ec\uff1a{representative}\u3002"
            "\u4ece\u7ed3\u679c\u770b\uff0c\u7eaf\u6df1\u5ea6\u5b66\u4e60\u8bc6\u522b\u65b9\u5411\u7684\u8d44\u6599\u76f8\u5bf9\u5c11\uff0c"
            "\u66f4\u591a\u6587\u732e\u8fd8\u662f\u504f\u5411\u5efa\u7b51\u6587\u5316\u3001\u88c5\u9970\u827a\u672f\u548c\u7a7a\u95f4\u7814\u7a76\u3002"
        )
    return (
        f"\u6839\u636e\u8fd9\u6b21\u8054\u7f51\u641c\u7d22\uff0c\u76f8\u5173\u8d44\u6599\u4e3b\u8981\u96c6\u4e2d\u5728"
        f"{theme_text}\u3002\u4e0b\u65b9\u6765\u6e90\u53ef\u4ee5\u7ee7\u7eed\u70b9\u5f00\u67e5\u770b\u3002"
    )


def _answer_with_search(message: str, sources: List[dict]) -> str:
    usable = [src for src in sources if src.get("url")]
    if not usable:
        return "\u6211\u5c1d\u8bd5\u8054\u7f51\u641c\u7d22\uff0c\u4f46\u6ca1\u6709\u83b7\u53d6\u5230\u53ef\u7528\u7ed3\u679c\uff1b\u53ef\u4ee5\u7a0d\u540e\u518d\u8bd5\uff0c\u6216\u6362\u4e00\u4e2a\u66f4\u5177\u4f53\u7684\u5173\u952e\u8bcd\u3002"
    fallback = _search_fallback_summary(message, usable)
    source_text = "\n".join(
        f"[{i}] {src.get('title', '')}\nURL: {src.get('url', '')}\n???: {src.get('snippet', '')}"
        for i, src in enumerate(usable[:5], 1)
    )
    prompt = (
        "\u4f60\u662f\u95fd\u53f0\u5bab\u5e99\u5efa\u7b51\u56fe\u50cf\u8bc6\u522b Agent\u3002"
        "\u8bf7\u53ea\u6839\u636e\u4e0b\u9762\u7684\u8054\u7f51\u641c\u7d22\u7ed3\u679c\u56de\u7b54\u7528\u6237\uff0c"
        "\u56de\u7b54\u8981\u7b80\u6d01\uff0c\u5982\u679c\u7ed3\u679c\u4e0d\u8db3\u8981\u660e\u786e\u8bf4\u4e0d\u8db3\u3002\n\n"
        f"\u7528\u6237\u95ee\u9898\uff1a{message}\n\n"
        f"\u641c\u7d22\u7ed3\u679c\uff1a\n{source_text}"
    )
    try:
        reply = LMStudioClient().chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=2200)
        content = (reply.get("content") or "").strip()
        if content:
            return content
        return fallback
    except Exception:
        return fallback


def _wants_export(message: str) -> bool:
    msg = message or ""
    keywords = [
        "\u5bfc\u51fa",
        "\u4e0b\u8f7d",
        "\u62a5\u544a",
        "\u8868\u683c",
        "\u6e05\u5355",
        "\u7edf\u8ba1",
        "csv",
        "json",
        "excel",
        "xlsx",
    ]
    return any(keyword in msg.lower() for keyword in keywords)


def _likelihood_phrase(score: float) -> str:
    if score >= 0.85:
        return "\u53ef\u80fd\u6027\u5f88\u9ad8"
    if score >= 0.65:
        return "\u53ef\u80fd\u6027\u8f83\u9ad8"
    if score >= 0.45:
        return "\u6709\u4e00\u5b9a\u53ef\u80fd"
    return "\u53ef\u80fd\u6027\u8f83\u4f4e\uff0c\u5efa\u8bae\u4ec5\u4f5c\u53c2\u8003"


def _summarize_results(message: str, results: List[dict], image_count: int) -> str:
    ok_items = [item for item in results if item.get("ok")]
    if not ok_items:
        return "\u8fd9\u5f20\u56fe\u7247\u6682\u65f6\u6ca1\u6709\u5f97\u5230\u53ef\u9760\u7684\u8bc6\u522b\u7ed3\u679c\u3002"

    task_labels = {
        "\u5c4b\u810a\u88c5\u9970\u8bc6\u522b": "\u5c4b\u810a\u88c5\u9970",
        "\u5c4b\u9876\u56db\u5206\u7c7b": "\u5c4b\u9876\u6837\u5f0f",
        "\u5f00\u95f4\u5206\u7c7b": "\u5f00\u95f4\u7c7b\u578b",
        "\u74e6\u7247\u5206\u7c7b": "\u74e6\u7247\u7c7b\u578b",
        "\u584c\u5bff\u4e09\u5206\u7c7b": "\u584c\u5bff\u7c7b\u578b",
        "YOLO\u68c0\u6d4b+\u5206\u7c7b": "\u68c0\u6d4b\u5206\u7c7b",
        "\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b": "\u5efa\u7b51\u4e3b\u4f53\u533a\u57df",
        "\u5efa\u7b51\u5c4b\u9876\u533a\u57df\u8bc6\u522b": "\u5efa\u7b51\u5c4b\u9876\u533a\u57df",
    }

    parts = []
    for item in ok_items:
        task = item.get("task", "")
        pred = str(item.get("prediction_cn") or item.get("prediction") or "").strip()
        if not pred:
            continue
        conf = float(item.get("confidence", 0.0))
        label = task_labels.get(task, task)
        if task in ["\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b", "\u5efa\u7b51\u5c4b\u9876\u533a\u57df\u8bc6\u522b"]:
            if pred.startswith("\u672a\u68c0\u6d4b\u5230"):
                parts.append(f"\u6ca1\u6709\u660e\u786e\u68c0\u6d4b\u5230{label}")
            else:
                parts.append(f"\u68c0\u6d4b\u5230{pred}\uff0c{_likelihood_phrase(conf)}")
        elif task == "\u5c4b\u810a\u88c5\u9970\u8bc6\u522b":
            parts.append(f"{label}\u8bc6\u522b\u5230{pred}")
        else:
            parts.append(f"{label}\u5927\u6982\u662f\u201c{pred}\u201d\uff0c{_likelihood_phrase(conf)}")

    if not parts:
        return "\u8fd9\u5f20\u56fe\u7247\u6682\u65f6\u6ca1\u6709\u5f97\u5230\u53ef\u9760\u7684\u8bc6\u522b\u7ed3\u679c\u3002"

    joined = "; ".join(parts)
    if image_count > 1:
        return f"\u8fd9\u6279\u56fe\u7247\u7684\u8bc6\u522b\u7ed3\u679c\u662f\uff1a{joined}\u3002"
    return f"\u8fd9\u4e2a\u5bab\u5e99\u5efa\u7b51\u7684{joined}\u3002"


@app.get("/api/health")
def health():
    lm = LMStudioClient()
    models = []
    model_api_ok = False
    model_api_error = None
    try:
        models = lm.models()
        model_api_ok = True
    except Exception as exc:
        model_api_error = str(exc)
    model_api = {"ok": model_api_ok, "base_url": lm.base_url, "model": lm.model, "models": models, "error": model_api_error}
    return {
        "ok": True,
        "model_api": model_api,
        "lmstudio": model_api,
        "tasks": recognition_engine.tasks(),
    }


@app.get("/api/ready")
def ready():
    """Local-only readiness check that never contacts the remote model API."""
    tasks = recognition_engine.tasks()
    return {
        "ok": True,
        "task_count": len(tasks),
        "models_ready": all(item.get("model_exists") for item in tasks),
    }


@app.get("/api/tasks")
def tasks():
    return {"tasks": recognition_engine.tasks()}


@app.get("/api/workflow/tools")
def workflow_tools():
    return {"tools": workflow_tool_specs()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    deterministic = _deterministic_general_answer(req.message)
    if deterministic:
        return {"reply": deterministic, "reasoning_content": "", "model": LMStudioClient().model}
    system = (
        "\u4f60\u662f\u95fd\u53f0\u5bab\u5e99\u5efa\u7b51\u56fe\u50cf\u8bc6\u522b Agent\u3002"
        "\u4f60\u53ef\u4ee5\u5e2e\u52a9\u7528\u6237\u9009\u62e9\u8bc6\u522b\u4efb\u52a1\u3001"
        "\u89e3\u91ca\u7ed3\u679c\u3001\u89c4\u5212\u6279\u91cf\u7b5b\u9009\u548c\u751f\u6210\u62a5\u544a\u3002"
        "\u5f53\u7528\u6237\u9700\u8981\u771f\u6b63\u8bc6\u522b\u56fe\u7247\u65f6\uff0c"
        "\u63d0\u793a\u5176\u5728\u7f51\u9875\u4e2d\u4e0a\u4f20\u56fe\u7247\u5e76\u9009\u62e9\u4efb\u52a1\u3002"
    )
    messages = [{"role": "system", "content": system}]
    for item in req.history[-8:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": req.message})
    try:
        reply = LMStudioClient().chat(messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"reply": _final_reply_text(reply), "reasoning_content": reply.get("reasoning_content", ""), "model": reply["model"]}


@app.post("/api/agent_message")
def agent_message(
    message: str = Form(""),
    history: str = Form("[]"),
    use_image_context: str = Form("0"),
    web_search_enabled: str = Form("0"),
    files: Optional[List[UploadFile]] = File(default=None),
):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    upload_dir = DATA_DIR / "chat_uploads" / run_id
    output_dir = OUTPUT_DIR / "chat" / run_id
    inputs: List[Path] = []
    input_roots: List[Path] = []
    has_archive_upload = False
    history_items = _parse_chat_history(history)

    for upload in files or []:
        saved = _save_upload(upload, upload_dir)
        if is_archive(saved):
            has_archive_upload = True
            extracted = extract_archive(saved, upload_dir / f"{saved.stem}_extracted")
            input_roots.append(extracted)
            inputs.extend(collect_images(extracted))
        else:
            input_roots.append(saved)
            inputs.extend(collect_images(saved))

    if _looks_ridge_yolo_annotation_request(message):
        if not inputs:
            return {
                "reply": "请先上传图片、图片文件夹压缩包，或提供可读取的图片输入；我会直接生成 YOLO segmentation 多边形标注 txt。",
                "route": {"intent": "workflow", "tool": "ridge_yolo_labels"},
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
                "show_files": False,
                "output_dir": "",
            }
        label_root = input_roots[0] if len(input_roots) == 1 else upload_dir
        label_output = OUTPUT_DIR / "ridge_yolo_labels" / run_id
        result = generate_ridge_yolo_labels(label_root, label_output)
        files_out = {
            "zip": result.get("zip", ""),
            "summary": result.get("summary_json", ""),
            "records_json": result.get("records_json", ""),
            "records_csv": result.get("records_csv", ""),
            "data_yaml": result.get("data_yaml", ""),
            "classes": str(label_output / "classes.txt"),
        }
        reply = (
            f"已生成 YOLO segmentation 多边形标注："
            f"{result.get('label_count', 0)} 份 txt，"
            f"其中 {result.get('annotated_images', 0)} 张有脊饰多边形，"
            f"共 {result.get('polygons', 0)} 个目标。"
            f"输出目录：{result.get('output_dir', '')}"
        )
        return {
            "reply": reply,
            "route": {"intent": "workflow", "tool": "ridge_yolo_labels"},
            "results": [],
            "files": {key: _public_path(value) for key, value in files_out.items() if value},
            "images": [],
            "sources": [],
            "show_files": True,
            "output_dir": result.get("output_dir", ""),
            "summary": result,
        }

    workflow_response = maybe_handle_workflow_request(message, input_roots, run_id)
    if workflow_response is not None:
        workflow_response["files"] = {
            key: _public_path(value) or value
            for key, value in workflow_response.get("files", {}).items()
            if value
        }
        workflow_response["images"] = [
            {
                **item,
                "url": _public_path(item.get("url")) or item.get("url"),
            }
            for item in workflow_response.get("images", [])
            if item.get("url")
        ]
        return workflow_response

    if inputs and _looks_locate_anything_request(message):
        locate_output = OUTPUT_DIR / "locate_anything" / run_id
        return _locate_anything_response(_locate_anything_query(message), inputs, locate_output)

    allow_search = _form_bool(web_search_enabled)
    route = _route_with_llm(
        message,
        bool(inputs),
        input_count=len(inputs),
        has_archive=has_archive_upload,
        allow_search=allow_search,
        history_items=history_items,
    )
    # Execution-level guard: batch front-facade screening must not fall through
    # to the generic multi-tool visual branch even if a future router prompt changes.
    if inputs and _looks_facade_filter_request(message, input_count=len(inputs), has_archive=has_archive_upload):
        route = _facade_filter_route(message)

    if allow_search and route.get("intent") == "search":
        sources = search_web(message, limit=5)
        reply = _answer_with_search(message, sources)
        return {
            "reply": reply,
            "route": route,
            "results": [],
            "files": {},
            "images": [],
            "sources": [src for src in sources if src.get("url")],
        }

    if route.get("intent") == "facade_filter":
        if not inputs:
            return {
                "reply": "\u6ca1\u6709\u627e\u5230\u53ef\u7528\u4e8e\u7b5b\u9009\u7684\u56fe\u7247\u3002",
                "route": route,
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
            }
        filter_root = input_roots[0] if len(input_roots) == 1 else upload_dir
        filter_output = OUTPUT_DIR / "facade_filter" / run_id
        result = filter_facade_images(filter_root, filter_output, min_sharpness=80.0)
        kept_preview = result.get("kept_preview", [])[:MAX_CHAT_PREVIEW_IMAGES]
        images = [
            {
                "url": _public_path(item.get("output_image")),
                "title": Path(str(item.get("output_image", ""))).name,
                "detail": item.get("reason", ""),
            }
            for item in kept_preview
            if item.get("output_image")
        ]
        reply = (
            f"\u5df2\u6309\u201c\u5efa\u7b51\u4e3b\u4f53\u533a\u57df\u8bc6\u522b\u201d\u7b5b\u9009\u6e05\u6670\u7684\u5bab\u5e99\u5efa\u7b51\u6b63\u7acb\u9762\u7167\u7247\uff1a"
            f"\u5171\u68c0\u67e5 {result.get('total', 0)} \u5f20\uff0c\u4fdd\u7559 {result.get('kept', 0)} \u5f20\uff0c"
            f"\u5df2\u628a\u539f\u59cb\u56fe\u7247\u590d\u5236\u5230\uff1a{result.get('output_dir', '')}\u3002"
            f"\u5bf9\u8bdd\u4e2d\u53ea\u663e\u793a\u524d {len(images)} \u5f20\u9884\u89c8\uff0c\u5b8c\u6574\u7ed3\u679c\u5728\u65b0\u6587\u4ef6\u5939\u91cc\u3002"
        )
        return {
            "reply": reply,
            "route": route,
            "results": result.get("records", [])[:200],
            "files": {k: _public_path(v) for k, v in result.get("files", {}).items()} if route.get("needs_export") else {},
            "images": images,
            "sources": [],
            "show_files": bool(route.get("needs_export")),
            "output_dir": result.get("output_dir", ""),
        }

    if not inputs or route.get("intent") == "chat":
        try:
            deterministic = _deterministic_general_answer(message)
            if deterministic:
                return {
                    "reply": deterministic,
                    "route": route,
                    "results": [],
                    "files": {},
                    "images": [],
                    "sources": [],
                }
            prompt = (
                "\u4f60\u662f\u95fd\u53f0\u5bab\u5e99\u5efa\u7b51\u56fe\u50cf\u8bc6\u522b Agent\u3002"
                "\u8bf7\u7b80\u6d01\u56de\u7b54\u7528\u6237\u7684\u95ee\u9898\u3002"
                "\u4f60可以参考最近的对话历史，尤其是你上一轮主动询问用户的信息；"
                "\u5982\u679c\u7528\u6237\u7684\u56de\u7b54\u5f88\u77ed\uff0c\u8bf7\u7ed3\u5408\u4e0a\u4e0b\u6587\u7406\u89e3\u3002"
            )
            reply = LMStudioClient().chat(_messages_with_history(prompt, history_items, message))
            return {
                "reply": _final_reply_text(reply),
                "route": route,
                "results": [],
                "files": {},
                "images": [],
                "sources": [],
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    selected_tasks = route.get("tasks") or _choose_tasks(message)
    results = []
    for image_path in inputs:
        for task_name in selected_tasks:
            try:
                item = recognition_engine.predict_image(task_name, image_path, output_dir)
                item["ok"] = True
                item["error"] = None
            except Exception as exc:
                item = {
                    "ok": False,
                    "error": str(exc),
                    "task": task_name,
                    "input_image": str(image_path),
                    "prediction": "",
                    "prediction_cn": "",
                    "confidence": 0.0,
                    "topk": [],
                    "detail": str(exc),
                    "result_image": None,
                }
            item["result_image_url"] = _public_path(item.get("result_image"))
            results.append(item)

    files_out = write_result_files(results, output_dir, base_name="chat_results")
    reply = _summarize_results(message, results, len(inputs))
    preview_items = [item for item in results if item.get("result_image_url")][:MAX_CHAT_PREVIEW_IMAGES]
    images = [
        {
            "url": item.get("result_image_url"),
            "title": f"{item.get('task')} - {item.get('prediction_cn') or item.get('prediction')}",
            "detail": item.get("detail", ""),
        }
        for item in preview_items
    ]
    response = {
        "reply": reply,
        "tasks": selected_tasks,
        "count": len(results),
        "success_count": sum(1 for item in results if item.get("ok")),
        "error_count": sum(1 for item in results if not item.get("ok")),
        "results": results[:200],
        "images": images,
        "files": {k: _public_path(v) for k, v in files_out.items()},
        "sources": [],
        "show_files": bool(route.get("needs_export")),
        "route": route,
    }
    if not response["show_files"]:
        response["files"] = {}
    return response


@app.post("/api/analyze")
def analyze(task: str = Form(...), files: List[UploadFile] = File(...)):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    upload_dir = DATA_DIR / "uploads" / run_id
    output_dir = OUTPUT_DIR / "analysis" / run_id
    inputs: List[Path] = []

    for upload in files:
        saved = _save_upload(upload, upload_dir)
        if is_archive(saved):
            extracted = extract_archive(saved, upload_dir / f"{saved.stem}_extracted")
            inputs.extend(collect_images(extracted))
        else:
            inputs.extend(collect_images(saved))

    if not inputs:
        raise HTTPException(status_code=400, detail="\u6ca1\u6709\u627e\u5230\u53ef\u8bc6\u522b\u7684\u56fe\u7247\u6587\u4ef6\u3002")

    results = []
    for image_path in inputs:
        try:
            item = recognition_engine.predict_image(task, image_path, output_dir)
            item["ok"] = True
            item["error"] = None
        except Exception as exc:
            item = {
                "ok": False,
                "error": str(exc),
                "task": task,
                "input_image": str(image_path),
                "prediction": "",
                "prediction_cn": "",
                "confidence": 0.0,
                "topk": [],
                "detail": str(exc),
                "result_image": None,
            }
        item["result_image_url"] = _public_path(item.get("result_image"))
        results.append(item)
    files_out = write_result_files(results, output_dir, base_name="analysis_results")
    return {
        "task": task,
        "count": len(results),
        "success_count": sum(1 for item in results if item.get("ok")),
        "error_count": sum(1 for item in results if not item.get("ok")),
        "results": results,
        "files": {k: _public_path(v) for k, v in files_out.items()},
    }


@app.post("/api/locate_anything")
def locate_anything(query: str = Form(...), mode: str = Form("all"), files: List[UploadFile] = File(...)):
    if not ENABLE_LOCATE_ANYTHING:
        raise HTTPException(
            status_code=404,
            detail="LocateAnything-3B is not part of the standard portable package.",
        )
    run_id = time.strftime("%Y%m%d_%H%M%S")
    upload_dir = DATA_DIR / "locate_anything_uploads" / run_id
    output_dir = OUTPUT_DIR / "locate_anything" / run_id
    inputs: List[Path] = []

    for upload in files:
        saved = _save_upload(upload, upload_dir)
        if is_archive(saved):
            extracted = extract_archive(saved, upload_dir / f"{saved.stem}_extracted")
            inputs.extend(collect_images(extracted))
        else:
            inputs.extend(collect_images(saved))

    return _locate_anything_response(query, inputs, output_dir, mode=mode)


@app.post("/api/filter_facade")
def filter_facade(req: FacadeFilterRequest):
    input_path = Path(req.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"\u8def\u5f84\u4e0d\u5b58\u5728\uff1a{input_path}")
    output_dir = Path(req.output_dir) if req.output_dir else OUTPUT_DIR / "facade_filter" / time.strftime("%Y%m%d_%H%M%S")
    try:
        result = filter_facade_images(input_path, output_dir, min_sharpness=req.min_sharpness)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    result["files"] = {k: _public_path(v) for k, v in result["files"].items()}
    return result


@app.get("/files/{path:path}")
def files(path: str):
    target = PROJECT_ROOT / path
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="\u6587\u4ef6\u4e0d\u5b58\u5728")
    return FileResponse(target)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

