# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT_BAT = PROJECT_ROOT / "run_agent.bat"
ENV_FILE = PROJECT_ROOT / ".env"


def _parse_key_value_lines(text: str, from_batch: bool = False) -> Dict[str, str]:
    settings: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("::") or line.lower().startswith("rem "):
            continue
        if from_batch and not line.lower().startswith("set "):
            continue
        assignment = line[4:].strip() if line.lower().startswith("set ") else line
        if assignment.lower().startswith("export "):
            assignment = assignment[7:].strip()
        if len(assignment) >= 2 and assignment[0] == assignment[-1] == '"':
            assignment = assignment[1:-1]
        if "=" not in assignment:
            continue
        key, value = assignment.split("=", 1)
        key = key.strip().strip('"').upper()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        if key:
            settings[key] = value
    return settings


def _load_settings_file(path: Path, from_batch: bool = False) -> Dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return _parse_key_value_lines(text, from_batch=from_batch)


def _setting(settings: Dict[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    for name in names:
        value = settings.get(name)
        if value:
            return value
    return default


class LMStudioClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = {}
        settings.update(_load_settings_file(RUN_AGENT_BAT, from_batch=True))
        settings.update(_load_settings_file(ENV_FILE))
        self.base_url = (
            base_url
            or _setting(settings, "MODEL_API_BASE_URL", "OPENAI_BASE_URL", "LMSTUDIO_BASE_URL", default=DEFAULT_BASE_URL)
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else _setting(settings, "MODEL_API_KEY", "OPENAI_API_KEY", "LMSTUDIO_API_KEY")
        self.model = (
            model
            or _setting(settings, "MODEL_NAME", "MODEL_API_MODEL", "OPENAI_MODEL", "LMSTUDIO_MODEL")
        ).strip()
        if not self.model:
            raise RuntimeError(f"MODEL_NAME is not configured. Set it in {ENV_FILE} or {RUN_AGENT_BAT}.")

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 120) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to model API at {self.base_url}: {exc.reason}") from exc

    def models(self) -> List[str]:
        data = self._request("GET", "/models", timeout=15)
        return [item.get("id", "") for item in data.get("data", []) if item.get("id")]

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.4, max_tokens: int = 2048) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._request("POST", "/chat/completions", payload=payload, timeout=180)
        message = data.get("choices", [{}])[0].get("message", {})
        content = _repair_mojibake(message.get("content") or "")
        reasoning_content = _repair_mojibake(message.get("reasoning_content") or "")
        return {
            "model": data.get("model", self.model),
            "content": content,
            "reasoning_content": reasoning_content,
            "raw": data,
        }


def _repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-CP1252 mojibake seen in some local model replies."""
    if not text:
        return text
    markers = ("\u8103", "\u8117", "\u8292", "\u832b", "\u8302", "\u5fd9", "\u732b", "\u83bd", "\u6c13", "\u8305")
    if not any(marker in text for marker in markers):
        return text
    try:
        raw = bytearray()
        for ch in text:
            code = ord(ch)
            if code <= 255:
                raw.append(code)
            else:
                raw.extend(ch.encode("cp1252"))
        repaired = bytes(raw).decode("utf-8")
    except UnicodeError:
        return text
    cjk_count = sum(1 for ch in repaired if "\u4e00" <= ch <= "\u9fff")
    old_cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return repaired if cjk_count > old_cjk_count else text

