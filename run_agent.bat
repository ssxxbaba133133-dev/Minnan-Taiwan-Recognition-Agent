@echo off
title Temple Recognition Agent - Remote API
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONDONTWRITEBYTECODE=1
rem Edit .env first. These defaults only keep the process startable before configuration.
if not defined MODEL_API_BASE_URL set MODEL_API_BASE_URL=http://127.0.0.1:1234/v1
if not defined MODEL_NAME set MODEL_NAME=your-remote-model-name
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\agent_control.ps1" run
pause
