param()
$ErrorActionPreference = "Stop"
$ApplicationRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Python = Join-Path $ApplicationRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    & (Join-Path $PSScriptRoot "install.ps1") -InPlace -NoLaunch
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$env:PYTHONUTF8 = "1"
Push-Location $ApplicationRoot
try {
    & $Python -m carla_vision.launcher --workspace $ApplicationRoot tui
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
