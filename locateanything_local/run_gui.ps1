$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python locateanything_local\gui_pyqt5.py
