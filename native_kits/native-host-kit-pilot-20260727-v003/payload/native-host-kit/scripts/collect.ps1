param(
    [Parameter(Mandatory=$true)][string]$ReadyPreflightRunId,
    [Parameter(Mandatory=$true)]
    [ValidateSet("I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK")]
    [string]$Confirmation
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot
if (Test-Path "datasets\ds-carla0916-native-pilot-v001") {
    throw "Refusing to overwrite existing dataset: datasets\ds-carla0916-native-pilot-v001"
}
$env:PYTHONPATH = (Join-Path $KitRoot "source")
& .\.venv\Scripts\python.exe -m carla_vision.native.host_gate `
    --kit-plan kit-plan.json `
    --preflight "runs\$ReadyPreflightRunId"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m carla_vision.native.worker `
    --scenario-plan "scenario-plan\scenario-plan-native-integration-pilot-v1" `
    --dataset-id "ds-carla0916-native-pilot-v001" `
    --datasets-root datasets `
    --host "172.20.10.7" `
    --port "2000" `
    --partition train `
    --max-episodes 1 `
    --timeout 30 `
    --sensor-timeout 10 `
    --acknowledge-exclusive-tick-owner
exit $LASTEXITCODE
