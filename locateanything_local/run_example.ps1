$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# Edit these two values before running.
$imagePath = "E:\TempleRecognitionAgent\data\uploads\example.jpg"
$query = "roof ridge dragon"

python locateanything_local\locate.py `
  --input $imagePath `
  --query $query `
  --output-dir "outputs\locateanything_local"
