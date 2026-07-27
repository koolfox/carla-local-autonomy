# Portable Native-Host Kit

Status: implementation complete; real CARLA execution intentionally deferred  
Kit: `native-host-kit-pilot-20260727-v003`  
Target: CPython 3.12, Windows/Linux x86-64, CARLA 0.9.16  
Scope: one Town10HD train episode, 50 front-RGB/instance pairs

## Deliverable

Transfer this file to the machine that can import the official CARLA
PythonAPI:

```text
native_kits/native-host-kit-pilot-20260727-v003/payload/native-host-kit.zip
```

SHA-256:

```text
185779504f5d789f6dd83eba76728814797512ba854ba63b2957547d3d7c377b
```

The outer research object is at
`native_kits/native-host-kit-pilot-20260727-v003`. Its manifest SHA-256 is:

```text
ec0b81358d93aa2edf71e58c6b272d7f3fcb696121356bb1ba8526c01e939165
```

The 1.3 MiB ZIP contains 114 files, including 95 project Python files, the
verified scenario plan, an internal checksum index, two platform-named
requirements locks, and equivalent PowerShell/shell scripts. Both lock files
have SHA-256
`ceaef5c27bbaa0091cc8de17181eaf86600bba21da8ed60be0d2dbbf2d83be17`;
all 14 packages are pinned with hashes, including six official CARLA wheel
hashes.

The kit contains no CARLA server, model weights, secret, previous Dataset, or
full thesis scenario plan.

## Manual acceptance sequence

1. Copy the ZIP and verify its SHA-256 before extracting.
2. Extract it into a new empty directory.
3. Verify/bootstrap:

   Windows PowerShell:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\scripts\bootstrap.ps1
   ```

   Linux:

   ```bash
   chmod +x scripts/*.sh scripts/verify_payload.py
   ./scripts/bootstrap.sh
   ```

4. Run the first read-only preflight without confirmations:

   ```powershell
   .\scripts\preflight.ps1 -RunId native-preflight-pilot-host-v1
   ```

   ```bash
   ./scripts/preflight.sh native-preflight-pilot-host-v1
   ```

5. Inspect `runs/native-preflight-pilot-host-v1/report.md`. Stop all other
   CARLA clients and tick owners, and confirm the current world is disposable.
6. Run a second read-only preflight with a new ID and both confirmations:

   ```powershell
   .\scripts\preflight.ps1 `
     -RunId native-preflight-pilot-host-ready-v1 `
     -ConfirmWorldReload `
     -ConfirmExclusiveTickOwner
   ```

   ```bash
   ./scripts/preflight.sh native-preflight-pilot-host-ready-v1 \
     --confirm-world-reload \
     --confirm-exclusive-tick-owner
   ```

7. Stop unless the second summary contains all three:

   ```text
   ready_for_native_execution true
   read_only                  true
   simulator_mutated          false
   ```

8. Run the single destructive pilot:

   ```powershell
   .\scripts\collect.ps1 `
     -ReadyPreflightRunId native-preflight-pilot-host-ready-v1 `
     -Confirmation I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK
   ```

   ```bash
   ./scripts/collect.sh native-preflight-pilot-host-ready-v1 \
     I_CONFIRM_WORLD_RELOAD_AND_EXCLUSIVE_TICK
   ```

9. Verify and audit:

   ```powershell
   .\scripts\postrun.ps1
   ```

   ```bash
   ./scripts/postrun.sh
   ```

10. Copy these two complete directories back to the research workspace:

    ```text
    datasets/ds-carla0916-native-pilot-v001
    runs/ds-carla0916-native-pilot-v001-qa
    ```

Do not train or scale collection until the QA montage, exact frame IDs,
RGB/instance alignment, construction props, actor inventory, cleanup, and
restoration to asynchronous mode have been inspected by hand.

## Safety and failure behavior

- Bootstrap verifies every payload byte before installation and uses
  `pip --require-hashes`.
- Preflight is read-only and never authorizes itself.
- The collection gate recomputes both kit and preflight semantics and rejects
  any different Dataset ID, endpoint, scenario-plan hash, episode selection,
  or capture count.
- Collection also requires the literal confirmation token shown above.
- Existing Dataset and run directories are never overwritten.
- Map load/reload destroys the current CARLA world actors.
- A crash may leave CARLA in synchronous mode; restart the simulator before
  retrying and use new preflight/run IDs.

Live & Record in the local operator UI is separate: it uses the lightweight
MessagePack bridge, so a `PythonAPI · missing` header does not prevent the live
RT-DETR/YOLO viewer. The official PythonAPI is required only for this native
world-building and synchronous capture path.
