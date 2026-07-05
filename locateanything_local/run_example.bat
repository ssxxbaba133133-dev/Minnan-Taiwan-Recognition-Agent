@echo off
setlocal
cd /d "%~dp0\.."

REM Edit these two values before running.
set "IMAGE_PATH=E:\TempleRecognitionAgent\data\uploads\example.jpg"
set "QUERY=roof ridge dragon"

python locateanything_local\locate.py ^
  --input "%IMAGE_PATH%" ^
  --query "%QUERY%" ^
  --output-dir "outputs\locateanything_local"

pause
