param(
    [ValidateSet("menu", "start", "stop", "status", "open", "run")]
    [string]$Action = "menu"
)

$ErrorActionPreference = "Continue"
$Port = 7860
$Url = "http://127.0.0.1:$Port"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UvicornArgs = @("-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "$Port")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path -LiteralPath $EnvFile) {
    foreach ($line in (Get-Content -LiteralPath $EnvFile -Encoding UTF8)) {
        $trimmed = ([string]$line).Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"')
        if ($key) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}
if (-not $env:MODEL_API_BASE_URL -and $env:LMSTUDIO_BASE_URL) {
    $env:MODEL_API_BASE_URL = $env:LMSTUDIO_BASE_URL
}
if (-not $env:MODEL_API_BASE_URL) {
    $env:MODEL_API_BASE_URL = "http://127.0.0.1:1234/v1"
}
if (-not $env:MODEL_NAME -and $env:LMSTUDIO_MODEL) {
    $env:MODEL_NAME = $env:LMSTUDIO_MODEL
}
if (-not $env:MODEL_NAME) {
    $env:MODEL_NAME = "your-remote-model-name"
}

function Add-PythonCandidate {
    param(
        [System.Collections.Generic.List[string]]$Candidates,
        [hashtable]$Seen,
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($Path.Trim('"'))
    if ($expanded -like "*\Microsoft\WindowsApps\python*.exe") {
        return
    }

    $key = $expanded.ToLowerInvariant()
    if (-not $Seen.ContainsKey($key)) {
        $Seen[$key] = $true
        $Candidates.Add($expanded) | Out-Null
    }
}

function Test-PythonCandidate {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }

    try {
        $requiredModules = "uvicorn,fastapi,multipart,cv2,numpy,PIL,pandas,torch,torchvision,timm,ultralytics"
        $probeScript = @"
import importlib.util
import sys

module_names = '$requiredModules'.split(',')
missing = [name for name in module_names if importlib.util.find_spec(name) is None]
print('EXE=' + sys.executable)
print('VERSION=%d.%d.%d' % sys.version_info[:3])
print('MISSING=' + ','.join(missing))
"@
        $probe = & $Path -c $probeScript 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $probe) {
            return $null
        }

        $exeLine = $probe | Where-Object { $_ -like "EXE=*" } | Select-Object -First 1
        $versionLine = $probe | Where-Object { $_ -like "VERSION=*" } | Select-Object -First 1
        $missingLine = $probe | Where-Object { $_ -like "MISSING=*" } | Select-Object -First 1
        if (-not $exeLine -or -not $versionLine -or -not $missingLine) {
            return $null
        }

        $missingModules = @()
        $missingText = ([string]$missingLine).Substring("MISSING=".Length)
        if (-not [string]::IsNullOrWhiteSpace($missingText)) {
            $missingModules = $missingText.Split(",") | Where-Object { $_ }
        }

        return [pscustomobject]@{
            Path = ([string]$exeLine).Substring("EXE=".Length)
            Version = ([string]$versionLine).Substring("VERSION=".Length)
            HasRequiredModules = ($missingModules.Count -eq 0)
            MissingModules = $missingModules
        }
    }
    catch {
        return $null
    }
}

function Get-PythonVersionRank {
    param([string]$Version)

    $preferred = @("3.11", "3.10", "3.12", "3.13")
    for ($i = 0; $i -lt $preferred.Count; $i++) {
        if ($Version -like "$($preferred[$i])*") {
            return $i
        }
    }

    return 100
}

function Get-PythonCandidates {
    $candidates = New-Object System.Collections.Generic.List[string]
    $seen = @{}

    Add-PythonCandidate $candidates $seen $env:TEMPLE_PYTHON
    Add-PythonCandidate $candidates $seen (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    Add-PythonCandidate $candidates $seen (Join-Path $ProjectRoot "venv\Scripts\python.exe")

    foreach ($command in (Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
        Add-PythonCandidate $candidates $seen $command.Source
    }
    foreach ($command in (Get-Command python3.exe -All -ErrorAction SilentlyContinue)) {
        Add-PythonCandidate $candidates $seen $command.Source
    }

    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $launcherOutput = & $pyLauncher.Source -0p 2>$null
            foreach ($line in $launcherOutput) {
                if ($line -match "(?<path>[A-Za-z]:\\.*?python\.exe)") {
                    Add-PythonCandidate $candidates $seen $Matches.path
                }
            }
        }
        catch {
            # The launcher exists even when no Python installation is registered.
        }
    }

    $commonRoots = @(
        "$env:USERPROFILE\.conda\envs",
        "$env:USERPROFILE\miniconda3\envs",
        "$env:USERPROFILE\anaconda3\envs",
        "$env:LOCALAPPDATA\Programs\Python",
        "C:\ProgramData\miniconda3\envs",
        "C:\ProgramData\anaconda3\envs",
        "D:\anaconda3\envs",
        "D:\miniconda3\envs",
        "D:\mambaforge\envs",
        "D:\miniforge3\envs",
        "C:\Program Files\Python313",
        "C:\Program Files\Python312",
        "C:\Program Files\Python311",
        "C:\Program Files\Python310",
        "C:\Python313",
        "C:\Python312",
        "C:\Python311",
        "C:\Python310",
        "D:\anaconda3",
        "D:\miniconda3",
        "D:\mambaforge",
        "D:\miniforge3"
    )

    foreach ($root in $commonRoots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }

        Add-PythonCandidate $candidates $seen (Join-Path $root "python.exe")
        try {
            foreach ($found in (Get-ChildItem -LiteralPath $root -Filter python.exe -Recurse -File -ErrorAction SilentlyContinue)) {
                Add-PythonCandidate $candidates $seen $found.FullName
            }
        }
        catch {
        }
    }

    return $candidates
}

function Resolve-Python {
    param([switch]$Quiet)

    $usable = @()
    foreach ($candidate in (Get-PythonCandidates)) {
        $probe = Test-PythonCandidate $candidate
        if ($probe) {
            $usable += $probe
        }
    }

    if (-not $usable -or $usable.Count -eq 0) {
        return $null
    }

    $selected = $usable |
        Sort-Object @{ Expression = { if ($_.HasRequiredModules) { 0 } else { 1 } } }, @{ Expression = { Get-PythonVersionRank $_.Version } }, @{ Expression = { $_.Path.Length } } |
        Select-Object -First 1

    if (-not $Quiet) {
        Write-Host "Using Python $($selected.Version): $($selected.Path)"
        if (-not $selected.HasRequiredModules) {
            Write-Host "Warning: selected Python is missing required modules: $($selected.MissingModules -join ', ')"
        }
    }
    return $selected.Path
}

function Install-Python {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Host "No usable Python was found, and winget is not available."
        Write-Host "Install Python 3.11 from https://www.python.org/downloads/windows/ and start again."
        return $false
    }

    Write-Host "No usable Python found. Installing Python 3.11 with winget..."
    & $winget.Source install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python installation failed. winget exit code: $LASTEXITCODE"
        return $false
    }

    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $env:Path = "$userPath;$machinePath;$env:Path"
    return $true
}

function Get-OrInstallPython {
    $python = Resolve-Python
    if ($python) {
        return $python
    }

    if (Install-Python) {
        $python = Resolve-Python
        if ($python) {
            return $python
        }
    }

    Write-Host "Python is still unavailable. Check the installation, then run this tool again."
    return $null
}

function Get-AgentProcess {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $conn) {
        $netstatLine = netstat -ano -p tcp |
            Select-String -Pattern "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(?<pid>\d+)" |
            Select-Object -First 1
        if ($netstatLine -and $netstatLine.Matches.Count -gt 0) {
            $conn = [pscustomobject]@{
                OwningProcess = [int]$netstatLine.Matches[0].Groups["pid"].Value
            }
        }
    }
    if (-not $conn) {
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($async.AsyncWaitHandle.WaitOne(500, $false)) {
                $client.EndConnect($async)
                $client.Close()
                return [pscustomobject]@{
                    Id = $null
                    Name = "unknown"
                    Path = ""
                    StartTime = $null
                }
            }
            $client.Close()
        }
        catch {
        }

        return $null
    }

    $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
    if (-not $proc) {
        return [pscustomobject]@{
            Id = $conn.OwningProcess
            Name = "unknown"
            Path = ""
            StartTime = $null
        }
    }

    return [pscustomobject]@{
        Id = $proc.Id
        Name = $proc.ProcessName
        Path = $proc.Path
        StartTime = $proc.StartTime
    }
}

function Get-AgentHealth {
    try {
        return Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 30
    }
    catch {
        return $null
    }
}

function Wait-AgentProcess {
    param([int]$TimeoutSeconds = 90)

    for ($i = 0; $i -lt $TimeoutSeconds; $i++) {
        $proc = Get-AgentProcess
        if ($proc) {
            return $proc
        }
        Start-Sleep -Seconds 1
    }

    return $null
}

function Show-Status {
    $proc = Get-AgentProcess
    if (-not $proc) {
        Write-Host "Status : STOPPED"
        Write-Host "URL    : $Url"
        return $false
    }

    Write-Host "Status : RUNNING"
    Write-Host "PID    : $($proc.Id)"
    Write-Host "URL    : $Url"
    if ($proc.StartTime) {
        Write-Host "Started: $($proc.StartTime)"
    }

    $health = Get-AgentHealth
    if ($health -and $health.model_api) {
        Write-Host "Model  : $($health.model_api.model)"
        Write-Host "API    : $($health.model_api.base_url)"
    }
    elseif ($health) {
        Write-Host "Health : backend responded"
    }
    else {
        Write-Host "Health : port is open, but /api/health did not respond"
    }
    return $true
}

function Start-Agent {
    if (Get-AgentProcess) {
        Write-Host "Agent is already running."
        Show-Status | Out-Null
        return
    }

    $PythonExe = Get-OrInstallPython
    if (-not $PythonExe) {
        return
    }

    Write-Host "Starting Agent..."
    $logDir = Join-Path $ProjectRoot "outputs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stdoutLog = Join-Path $logDir "agent_server.log"
    $stderrLog = Join-Path $logDir "agent_server.err.log"
    $cmd = "cd /d `"$ProjectRoot`" && `"$PythonExe`" $($UvicornArgs -join ' ') 1>`"$stdoutLog`" 2>`"$stderrLog`""
    Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmd) -WorkingDirectory $ProjectRoot -WindowStyle Hidden | Out-Null

    if (Wait-AgentProcess -TimeoutSeconds 90) {
        Write-Host "Agent started."
        Show-Status | Out-Null
    }
    else {
        Write-Host "Agent did not stay running. Try run_agent.bat to see the error output."
        Write-Host "Stdout log: $stdoutLog"
        Write-Host "Stderr log: $stderrLog"
        if (Test-Path -LiteralPath $stderrLog) {
            Get-Content -LiteralPath $stderrLog -Tail 20
        }
    }
}

function Run-AgentForeground {
    $PythonExe = Get-OrInstallPython
    if (-not $PythonExe) {
        return
    }

    Set-Location -LiteralPath $ProjectRoot
    & $PythonExe @UvicornArgs
}

function Stop-Agent {
    $proc = Get-AgentProcess
    if (-not $proc) {
        Write-Host "Agent is already stopped."
        return
    }
    if (-not $proc.Id) {
        Write-Host "Agent port is open, but the process ID could not be detected."
        Write-Host "Close the process using Task Manager, or run this tool as Administrator."
        return
    }

    Write-Host "Stopping Agent PID $($proc.Id)..."
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        Start-Sleep -Seconds 1
    }
    catch {
        Write-Host "Stop failed: $($_.Exception.Message)"
        Write-Host "If this says access is denied, run this tool as Administrator."
        return
    }

    if (Get-AgentProcess) {
        Write-Host "Agent is still running. Run this tool as Administrator if needed."
    }
    else {
        Write-Host "Agent stopped."
    }
}

function Open-Agent {
    if (-not (Get-AgentProcess)) {
        Write-Host "Agent is stopped. Start it first."
        return
    }
    Start-Process $Url
}

function Show-Menu {
    while ($true) {
        Clear-Host
        Write-Host "Temple Recognition Agent Control"
        Write-Host "================================"
        Show-Status | Out-Null
        Write-Host ""
        Write-Host "[1] Start Agent"
        Write-Host "[2] Stop Agent"
        Write-Host "[3] Open Web Page"
        Write-Host "[4] Refresh Status"
        Write-Host "[0] Exit"
        Write-Host ""

        $choice = Read-Host "Choose"
        switch ($choice) {
            "1" { Start-Agent; Pause }
            "2" { Stop-Agent; Pause }
            "3" { Open-Agent; Pause }
            "4" { continue }
            "0" { return }
            default { Write-Host "Unknown choice."; Pause }
        }
    }
}

switch ($Action) {
    "start" { Start-Agent }
    "stop" { Stop-Agent }
    "status" { Show-Status | Out-Null }
    "open" { Open-Agent }
    "run" { Run-AgentForeground }
    default { Show-Menu }
}
