param(
    [string]$Destination = "",
    [switch]$InPlace,
    [switch]$NoLaunch
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    throw "Same-computer setup supports Windows x86-64."
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$SourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$IndexPath = Join-Path $SourceRoot "installer-files.json"
function Assert-RegularPath([string]$Path, [string]$Root) {
    $Current = $Path
    while ($Current -and $Current.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) {
        if ((Test-Path -LiteralPath $Current) -and
            ((Get-Item -LiteralPath $Current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Installer paths must not contain symbolic links or junctions: $Current"
        }
        if ($Current -eq $Root) { break }
        $Current = [IO.Path]::GetDirectoryName($Current)
    }
}
if ($InPlace -or -not (Test-Path -LiteralPath $IndexPath)) {
    $ApplicationRoot = $SourceRoot
} else {
    if (-not $Destination) { $Destination = Join-Path $env:LOCALAPPDATA "CARLA-Vision" }
    $ApplicationRoot = [IO.Path]::GetFullPath((Join-Path $Destination "app"))
}
Assert-RegularPath $ApplicationRoot $ApplicationRoot
New-Item -ItemType Directory -Force -Path $ApplicationRoot | Out-Null
$Private = Join-Path $ApplicationRoot ".launcher"
Assert-RegularPath $Private $ApplicationRoot
New-Item -ItemType Directory -Force -Path $Private | Out-Null
$InstallLocks = New-Object 'System.Collections.Generic.List[System.IO.FileStream]'
try {
    # Same byte-range lock as Python msvcrt: cover every TUI entry point and
    # hold through both file replacement and environment synchronization.
    foreach ($LockName in @("launcher.lock", "operation.lock")) {
        $LockPath = Join-Path $Private $LockName
        Assert-RegularPath $LockPath $ApplicationRoot
        $Stream = [IO.File]::Open($LockPath, [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite)
        try {
            if ($Stream.Length -lt 1) { $Stream.SetLength(1) }
            $Stream.Lock(0, 1)
        } catch {
            $Stream.Dispose()
            throw "Close the installed launcher and finish other operations before reinstalling."
        }
        $InstallLocks.Add($Stream)
    }
    $ReceiptPath = Join-Path $Private "services.json"
    Assert-RegularPath $ReceiptPath $ApplicationRoot
    if (Test-Path -LiteralPath $ReceiptPath) {
        $Receipt = Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
        foreach ($Property in $Receipt.services.PSObject.Properties) {
            $Entries = @($Property.Value) + @($Property.Value.descendants)
            foreach ($Entry in $Entries) {
                $Owned = Get-Process -Id $Entry.pid -ErrorAction SilentlyContinue
                if ($Owned) {
                    $Created = ([DateTimeOffset]$Owned.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds() / 1000.0
                    if ([Math]::Abs($Created - [double]$Entry.create_time) -le 0.02) {
                        throw "Stop all launcher-owned services before reinstalling."
                    }
                }
            }
        }
    }
    if (-not $InPlace -and (Test-Path -LiteralPath $IndexPath)) {
        Assert-RegularPath $IndexPath $SourceRoot
        $Index = Get-Content -LiteralPath $IndexPath -Raw | ConvertFrom-Json
        if ($Index.schema_version -ne 1 -or -not $Index.files.Count -or
            $Index.source_commit -notmatch '^[a-f0-9]{40}$') { throw "Invalid installer inventory." }
        $PreviousPath = Join-Path $ApplicationRoot "installer-files.json"
        Assert-RegularPath $PreviousPath $ApplicationRoot
        $Previous = @{}
        if (Test-Path -LiteralPath $PreviousPath) {
            $PreviousIndex = Get-Content -LiteralPath $PreviousPath -Raw | ConvertFrom-Json
            foreach ($Entry in $PreviousIndex.files) { $Previous[[string]$Entry.path] = $Entry }
        }
        # Validate every source and destination before replacing any installed file.
        foreach ($Entry in $Index.files) {
            $Relative = [string]$Entry.path
            $Source = [IO.Path]::GetFullPath((Join-Path $SourceRoot $Relative))
            $Target = [IO.Path]::GetFullPath((Join-Path $ApplicationRoot $Relative))
            if (-not $Source.StartsWith($SourceRoot + "\", [StringComparison]::OrdinalIgnoreCase) -or
                -not $Target.StartsWith($ApplicationRoot + "\", [StringComparison]::OrdinalIgnoreCase)) {
                throw "Installer file path escapes its destination."
            }
            Assert-RegularPath $Source $SourceRoot
            Assert-RegularPath $Target $ApplicationRoot
            if ((Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Entry.sha256) {
                throw "Installer file integrity failed: $Relative"
            }
        }
        foreach ($Entry in $Index.files) {
            $Relative = [string]$Entry.path
            $Source = Join-Path $SourceRoot $Relative
            $Target = Join-Path $ApplicationRoot $Relative
            if ($Entry.model_weight -and (Test-Path -LiteralPath $Target) -and
                (Get-Item -LiteralPath $Target).Length -gt 1024) { continue }
            if (($Relative.StartsWith("configs/") -or $Relative.StartsWith("operator_configs/")) -and
                (Test-Path -LiteralPath $Target)) {
                $Original = $Previous[$Relative]
                if (-not $Original -or
                    (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Original.sha256) {
                    $Target = Join-Path (Join-Path (Join-Path $Private "incoming-defaults") $Index.source_commit) $Relative
                    Assert-RegularPath $Target $ApplicationRoot
                    Write-Host "Preserved customized configuration: $Relative"
                    Write-Host "Incoming default: $Target"
                }
            }
            New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($Target)) | Out-Null
            Copy-Item -LiteralPath $Source -Destination $Target -Force
        }
        Copy-Item -LiteralPath $IndexPath -Destination $PreviousPath -Force
    }
    $UvDirectory = Join-Path $Private "uv"
    Assert-RegularPath $UvDirectory $ApplicationRoot
    New-Item -ItemType Directory -Force -Path $UvDirectory | Out-Null
    $Uv = Join-Path $UvDirectory "uv.exe"
    Assert-RegularPath $Uv $ApplicationRoot
    if (-not (Test-Path -LiteralPath $Uv)) {
        $Bootstrap = Get-Content (Join-Path $ApplicationRoot "deploy\desktop\bootstrap.json") -Raw | ConvertFrom-Json
        $Stage = Join-Path $Private ("uv-install-" + [Guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $Stage | Out-Null
        try {
            $Archive = Join-Path $Stage "uv.zip"
            Write-Host "Downloading the pinned setup runtime..."
            Invoke-WebRequest -Uri $Bootstrap.uv.Windows.url -OutFile $Archive -UseBasicParsing
            if ((Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Bootstrap.uv.Windows.sha256) {
                throw "The setup runtime checksum did not match."
            }
            Expand-Archive -LiteralPath $Archive -DestinationPath (Join-Path $Stage "payload")
            $Executable = Get-ChildItem (Join-Path $Stage "payload") -Recurse -Filter uv.exe | Select-Object -First 1
            if (-not $Executable) { throw "The setup runtime archive did not contain uv.exe." }
            Copy-Item $Executable.FullName -Destination $Uv
        } finally {
            Remove-Item -LiteralPath $Stage -Recurse -Force
        }
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Private "python"
    $env:UV_CACHE_DIR = Join-Path $Private "cache"
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $ApplicationRoot ".venv"
    $env:PYTHONUTF8 = "1"
    Push-Location $ApplicationRoot
    try {
        Write-Host "Installing private Python 3.12 and the full-screen launcher..."
        & $Uv sync --python 3.12 --managed-python --extra launcher --no-dev --inexact --frozen
        if ($LASTEXITCODE -ne 0) { throw "Python/launcher installation failed; correct the error and rerun setup." }
        $Shell = New-Object -ComObject WScript.Shell
        $ShortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "CARLA Vision.lnk"
        $Shortcut = $Shell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath = Join-Path $ApplicationRoot "deploy\desktop\CARLA-Vision.cmd"
        $Shortcut.WorkingDirectory = $ApplicationRoot
        $Shortcut.Description = "Install and run the local CARLA research console"
        $Shortcut.Save()
        Write-Host "Launcher installed. A CARLA Vision desktop shortcut is ready."
    } finally {
        Pop-Location
    }
} finally {
    foreach ($Stream in $InstallLocks) {
        try { $Stream.Unlock(0, 1) } finally { $Stream.Dispose() }
    }
}
if (-not $NoLaunch) {
    & (Join-Path $ApplicationRoot ".venv\Scripts\python.exe") -m carla_vision.launcher --workspace $ApplicationRoot tui
    exit $LASTEXITCODE
}
