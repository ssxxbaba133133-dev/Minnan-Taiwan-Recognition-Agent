@echo off
chcp 65001 >nul
echo ???? Cloudflare Tunnel...
taskkill /IM cloudflared.exe /F >nul 2>nul
taskkill /IM proxychains_win32_x64.exe /F >nul 2>nul
echo ????
pause
