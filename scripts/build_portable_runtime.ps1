param(
    [string]$BuilderPython = "python",
    [string]$PythonEmbedVersion = "3.11.9"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "runtime"
$BuildDir = Join-Path $ProjectRoot ".build"
$DownloadDir = Join-Path $BuildDir "downloads"
$EmbedZip = Join-Path $DownloadDir "python-$PythonEmbedVersion-embed-amd64.zip"
$EmbedUrl = "https://www.python.org/ftp/python/$PythonEmbedVersion/python-$PythonEmbedVersion-embed-amd64.zip"
$SitePackages = Join-Path $RuntimeDir "Lib\site-packages"
$Requirements = Join-Path $ProjectRoot "requirements-portable.txt"

if (Test-Path -LiteralPath $RuntimeDir) {
    throw "The runtime directory already exists. Move it away before rebuilding."
}

New-Item -ItemType Directory -Force -Path $DownloadDir,$RuntimeDir,$SitePackages | Out-Null
if (-not (Test-Path -LiteralPath $EmbedZip)) {
    Write-Host "Downloading the official Python embedded runtime..." -ForegroundColor Cyan
    & curl.exe -L --fail --retry 3 --output $EmbedZip $EmbedUrl
    if ($LASTEXITCODE -ne 0) { throw "Failed to download the Python embedded runtime." }
}

Expand-Archive -LiteralPath $EmbedZip -DestinationPath $RuntimeDir -Force
$pthFile = Join-Path $RuntimeDir "python311._pth"
$pthLines = @("python311.zip", ".", "Lib", "Lib/site-packages", "..", "import site")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllLines($pthFile, $pthLines, $utf8NoBom)

Write-Host "Installing CPU PyTorch and pinned project dependencies..." -ForegroundColor Cyan
$pipArgs = @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
    "--only-binary=:all:", "--target", $SitePackages,
    "--index-url", "https://pypi.org/simple",
    "--extra-index-url", "https://download.pytorch.org/whl/cpu",
    "torch==2.10.0+cpu", "torchvision==0.25.0+cpu",
    "-r", $Requirements
)
& $BuilderPython @pipArgs
if ($LASTEXITCODE -ne 0) { throw "Failed to install portable dependencies." }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONNOUSERSITE = "1"
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot "data\ultralytics_config"
$env:YOLO_OFFLINE = "true"
$runtimePython = Join-Path $RuntimeDir "python.exe"

& $runtimePython (Join-Path $PSScriptRoot "runtime_manifest.py")
if ($LASTEXITCODE -ne 0) { throw "Failed to generate the runtime manifest." }
& $runtimePython (Join-Path $PSScriptRoot "verify_package.py") --full --imports
if ($LASTEXITCODE -ne 0) { throw "Portable runtime verification failed." }
& $runtimePython (Join-Path $PSScriptRoot "smoke_test.py") --load-models
if ($LASTEXITCODE -ne 0) { throw "Portable model loading test failed." }

Write-Host "Portable CPU runtime build and verification completed." -ForegroundColor Green
