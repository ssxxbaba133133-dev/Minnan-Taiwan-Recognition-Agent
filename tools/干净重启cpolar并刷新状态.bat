@echo off
chcp 65001 >nul
echo [1/5] Stop cpolar service...
D:\cpolar\cpolar.exe service stop

echo [2/5] Kill remaining cpolar.exe processes...
taskkill /IM cpolar.exe /F

echo [3/5] Wait a moment...
timeout /t 3 /nobreak >nul

echo [4/5] Start cpolar service with current config...
D:\cpolar\cpolar.exe service start

echo [5/5] Done. Please refresh http://localhost:9200/#/status/online after 10 seconds.
pause
