param(
    [string]$RunId = "native-preflight-native-host-kit-pilot-20260727-v003",
    [switch]$ConfirmWorldReload,
    [switch]$ConfirmExclusiveTickOwner
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot
if (Test-Path (Join-Path "runs" $RunId)) {
    throw "Refusing to overwrite existing preflight: runs/$RunId"
}
$env:PYTHONPATH = (Join-Path $KitRoot "source")
$Arguments = @(
    "-m", "carla_vision.native.preflight",
    "--scenario-plan", "scenario-plan/scenario-plan-native-integration-pilot-v1",
    "--dataset-id", "ds-carla0916-native-pilot-v001",
    "--runs-root", "runs",
    "--run-id", $RunId,
    "--host", "172.20.10.7",
    "--port", "2000",
    "--partition", "train",
    "--max-episodes", "1"
)
if ($ConfirmWorldReload) { $Arguments += "--confirm-world-reload" }
if ($ConfirmExclusiveTickOwner) { $Arguments += "--confirm-exclusive-tick-owner" }
& .\.venv\Scripts\python.exe @Arguments
exit $LASTEXITCODE
