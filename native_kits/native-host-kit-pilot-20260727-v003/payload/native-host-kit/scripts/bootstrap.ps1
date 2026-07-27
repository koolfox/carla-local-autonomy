param()
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$KitRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $KitRoot

py -3.12 scripts\verify_payload.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -3.12 -c "import platform,sys; assert sys.version_info[:2] == (3,12), 'native host kit requires CPython 3.12'; assert platform.system() == 'Windows' and platform.machine().lower() in {'amd64','x86_64'}, 'native host kit requires x86-64 Windows'; print(platform.platform(), sys.version)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path ".venv") {
    throw "Refusing to overwrite existing .venv"
}
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements\windows-py312.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -c "import carla,cv2,matplotlib,msgpack,numpy; print('carla',getattr(carla,'__version__','unknown'),carla.__file__); print('opencv',cv2.__version__); print('matplotlib',matplotlib.__version__); print('msgpack',msgpack.version); print('numpy',numpy.__version__)"
exit $LASTEXITCODE
