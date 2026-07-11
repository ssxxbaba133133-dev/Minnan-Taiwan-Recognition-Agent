@echo off
chcp 65001 >nul
title 闽台宫庙识别 Agent
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "%~dp0runtime\python.exe" (
  echo [ERROR] Missing runtime\python.exe
  pause
  exit /b 1
)
"%~dp0runtime\python.exe" "%~dp0scripts\launcher.py"
if errorlevel 1 pause
