# Portable native CARLA collection kit

Kit ID: `native-host-kit-pilot-20260727-v003`  
CARLA endpoint: `172.20.10.7:2000`  
Scenario plan: `scenario-plan-native-integration-pilot-v1`  
Dataset target: `ds-carla0916-native-pilot-v001`  
Planned captures: `50`

This payload is intentionally limited to one native 50-frame integration
episode. It contains the exact project Python sources needed for collection,
the verified scenario plan, hash-pinned CPython 3.12 dependencies for Windows
x86-64 and Linux x86-64, and guarded scripts. It contains no CARLA server,
model weight, secret, or previously collected dataset.

## 1. Verify and bootstrap

Extract the ZIP into a new directory. Do not run it from inside the archive.

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

Ubuntu/Linux:

```bash
chmod +x scripts/*.sh scripts/verify_payload.py
./scripts/bootstrap.sh
```

Both bootstrap scripts first verify `checksums.sha256`, require CPython 3.12
on x86-64, create a new `.venv`, and install every dependency with
`--require-hashes`.

## 2. Read-only preflight

First run without manual confirmations:

```powershell
.\scripts\preflight.ps1 -RunId native-preflight-pilot-host-v1
```

```bash
./scripts/preflight.sh native-preflight-pilot-host-v1
```

Inspect `runs/native-preflight-pilot-host-v1/report.md`. Stop all other CARLA
clients, confirm that the current world is disposable, and confirm this worker
will be the only `world.tick()` owner. Then create a new preflight ID and
explicitly record both confirmations:

```powershell
.\scripts\preflight.ps1 -RunId native-preflight-pilot-host-ready-v1 `
  -ConfirmWorldReload -ConfirmExclusiveTickOwner
```

```bash
./scripts/preflight.sh native-preflight-pilot-host-ready-v1   --confirm-world-reload --confirm-exclusive-tick-owner
```

Do not continue unless its summary says
`ready_for_native_execution: true`, `read_only: true`, and
`simulator_mutated: false`.

## 3. One destructive collection

The collection script accepts only a verified ready preflight that exactly
matches this kit. It also requires the literal confirmation token below.
Map loading/reloading destroys current world actors.

```powershell
.\scripts\collect.ps1 `
  -ReadyPreflightRunId native-preflight-pilot-host-ready-v1 `
  -Confirmation I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK
```

```bash
./scripts/collect.sh native-preflight-pilot-host-ready-v1   I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK
```

## 4. Verify and audit before transfer

```powershell
.\scripts\postrun.ps1
```

```bash
./scripts/postrun.sh
```

Copy both `datasets/ds-carla0916-native-pilot-v001` and `runs/ds-carla0916-native-pilot-v001-qa` back to the main
research workspace. Verify them again after transfer. Do not train or scale
collection until the 50-frame QA montage, frame IDs, labels, actor cleanup,
and asynchronous-world restoration are accepted.
