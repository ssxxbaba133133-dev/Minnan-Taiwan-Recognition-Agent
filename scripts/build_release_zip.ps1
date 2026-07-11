param(
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimePython = Join-Path $ProjectRoot "runtime\python.exe"
if (-not (Test-Path -LiteralPath $RuntimePython)) {
    throw "Portable runtime is missing. Run build_portable_runtime.ps1 first."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $RuntimePython (Join-Path $PSScriptRoot "verify_package.py") --full --imports
if ($LASTEXITCODE -ne 0) { throw "Package verification failed. Packaging stopped." }

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path (Split-Path -Parent $ProjectRoot) "dist"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$version = (Get-Content -LiteralPath (Join-Path $ProjectRoot "VERSION") -Encoding UTF8 | Select-Object -First 1).Trim()
$zipPath = Join-Path $OutputDirectory "TempleRecognitionAgent-Windows-CPU-$version.zip"
if (Test-Path -LiteralPath $zipPath) {
    throw "The output archive already exists: $zipPath"
}

Push-Location $ProjectRoot
try {
    $excludeArgs = @(
        "--exclude=.env",
        "--exclude=.build",
        "--exclude=.git",
        "--exclude=outputs/*",
        "--exclude=data/uploads/*",
        "--exclude=data/chat_uploads/*",
        "--exclude=data/ultralytics_config/*",
        "--exclude=backend/__pycache__",
        "--exclude=desktop_app/__pycache__",
        "--exclude=scripts/__pycache__"
    )
    & tar.exe -a -c -f $zipPath @excludeArgs "."
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the release archive." }
} finally {
    Pop-Location
}
Write-Host "Created: $zipPath" -ForegroundColor Green
