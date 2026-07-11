# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYTHON = ROOT / "runtime" / "python.exe"
LOCAL_CONFIG = ROOT / ".env"
DEFAULT_CONFIG = ROOT / "config" / "runtime.conf"
URL = "http://127.0.0.1:7860"


def read_settings(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def ready(timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(f"{URL}/api/ready", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def open_browser() -> None:
    if os.getenv("TEMPLE_AGENT_NO_BROWSER", "0") != "1":
        webbrowser.open(URL)


def open_browser_when_ready() -> None:
    for _ in range(180):
        status = ready()
        if status and status.get("ok"):
            open_browser()
            return
        time.sleep(0.5)


def main() -> int:
    if not RUNTIME_PYTHON.is_file():
        raise FileNotFoundError("缺少内置运行环境 runtime/python.exe。")

    config_path = LOCAL_CONFIG if LOCAL_CONFIG.is_file() else DEFAULT_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError("缺少运行配置 config/runtime.conf。")
    settings = read_settings(config_path)
    if not settings.get("MODEL_API_BASE_URL") or not settings.get("MODEL_NAME"):
        raise RuntimeError("运行配置不完整。")

    os.environ.update(settings)
    os.environ.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "YOLO_CONFIG_DIR": str(ROOT / "data" / "ultralytics_config"),
            "YOLO_OFFLINE": "true",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    check = subprocess.run(
        [str(RUNTIME_PYTHON), str(ROOT / "scripts" / "verify_package.py"), "--quick"],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check.returncode != 0:
        detail = (check.stderr or check.stdout or "").strip()
        raise RuntimeError(f"程序文件不完整。\n{detail}")

    status = ready()
    if status and status.get("ok"):
        print(f"Agent 已经在运行：{URL}")
        open_browser()
        return 0

    print("正在启动闽台宫庙识别 Agent……")
    print("浏览器打开后即可使用；关闭本窗口右上角 × 即可结束 Agent。")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=7860,
        log_level="warning",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as exc:
        print(f"\n[错误] {exc}", file=sys.stderr)
        raise SystemExit(1)
