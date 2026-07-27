param()
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot
$QaRunId = "ds-carla0916-native-pilot-v001-qa"
if (Test-Path (Join-Path "runs" $QaRunId)) {
    throw "Refusing to overwrite existing QA run: runs\$QaRunId"
}
$env:PYTHONPATH = (Join-Path $KitRoot "source")
& .\.venv\Scripts\python.exe -m carla_vision.verification `
    "datasets\ds-carla0916-native-pilot-v001" --reject-unregistered
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m carla_vision.dataset.qa `
    --dataset "datasets\ds-carla0916-native-pilot-v001" `
    --runs-root runs `
    --run-id $QaRunId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m carla_vision.verification `
    "runs\$QaRunId" --reject-unregistered
exit $LASTEXITCODE
