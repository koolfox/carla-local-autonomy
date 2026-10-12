# Works from the repository/backup, or as a standalone downloaded PowerShell script.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RestoreArguments)
$ErrorActionPreference = "Stop"
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    throw "The Windows bootstrap supports x86-64."
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$BootstrapRoot = Join-Path $env:LOCALAPPDATA "CARLA-Vision\restore-bootstrap\uv-0.11.2"
New-Item -ItemType Directory -Force -Path $BootstrapRoot | Out-Null
if ((Get-Item -LiteralPath $BootstrapRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw "The bootstrap cache must not be a symbolic link."
}
$Stage = Join-Path $BootstrapRoot ([Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Stage | Out-Null
try {
    Write-Host "Preparing the verified setup runtime (no system Python changes)..."
    $Archive = Join-Path $Stage "uv.zip"
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/0.11.2/uv-x86_64-pc-windows-msvc.zip" -OutFile $Archive
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant() -ne "171b7ccda1bbd562da6babeffcf533a1c6cc7862cf998da826e1db534fc43e48") {
        throw "Setup runtime checksum mismatch."
    }
    $Payload = Join-Path $Stage "payload"
    Expand-Archive -LiteralPath $Archive -DestinationPath $Payload
    $Candidates = @(Get-ChildItem -LiteralPath $Payload -Recurse -Filter "uv.exe")
    if ($Candidates.Count -ne 1) { throw "Unexpected setup runtime archive layout." }
    $Uv = $Candidates[0].FullName
    $Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
    $Engine = Join-Path $Root "restore_project.py"
    if (-not (Test-Path -LiteralPath $Engine -PathType Leaf)) {
        $Engine = Join-Path $Stage "restore_project.py"
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/koolfox/carla-local-autonomy/refs/heads/main/restore_project.py" -OutFile $Engine
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $BootstrapRoot "python"
    $env:UV_CACHE_DIR = Join-Path $BootstrapRoot "cache"
    $env:UV_NO_CONFIG = "1"
    $env:PYTHONUTF8 = "1"
    & $Uv run --no-project --python 3.12 --managed-python python -B $Engine --uv $Uv @RestoreArguments
    $RestoreExitCode = $LASTEXITCODE
} catch {
    Write-Error $_ -ErrorAction Continue
    $RestoreExitCode = 1
} finally {
    Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
}
exit $RestoreExitCode
