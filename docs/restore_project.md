# Restore the project on a new computer

Use `restore.sh` on macOS/Linux or `Restore.cmd` on Windows. These entry points
prepare a private, checksum-verified uv runtime and managed Python 3.12; you do
not need preinstalled Python, Node or Git for the default restoration.

The default source is `main`, which includes the published models, historical
samples, native launcher and recovery tools. A normal clone of the repository
gets these additions too. Use `--ref <commit>` for a specific version; recovery
resolves the requested ref to one exact commit and checks its model/sample
inventory and compiled console.

## Windows: CARLA and the web app together

Extract/download this branch or use the recovery tools saved beside your local
backup. From its root in PowerShell:

```powershell
.\Restore.cmd --destination "C:\CARLA-Vision" --role local --carla-root "C:\CARLA_0.9.16"
```

To also download the official CARLA 0.9.16 server when it is absent:

```powershell
.\Restore.cmd --destination "C:\CARLA-Vision" --role local --download-carla
```

The local profile installs the locked vision dependencies, official CARLA SDK,
full-screen launcher and bundled recording encoder. Server download is explicit;
an installed server can be reused. Compatible graphics hardware/drivers and
adequate disk space are still required. Setup prepares files without starting
CARLA or changing a running world.

After restoration:

```powershell
cd C:\CARLA-Vision
.\deploy\desktop\CARLA-Vision.cmd
```

Use the TUI's **Setup** tab to inspect the CARLA folder and **Run → Start all**
to start the server, Worker and browser console. This recovery entry point does
not create desktop shortcuts; the [preview installer](desktop_launcher.md) does.

## macOS: web app connected to the CARLA computer

On Apple Silicon with macOS 14 or newer:

```bash
sh restore.sh --destination "$HOME/CARLA-Vision" --role web
cd "$HOME/CARLA-Vision"
# Set the Windows/LAN Worker URL and matching secret in .env.local first.
.venv/bin/carla-operator-ui --open-browser
```

This restores all published models and installs the locked vision/recording
environment, without a local CARLA server or native CARLA PythonAPI. Intel macOS
14+ uses the lean web/recording profile automatically because the locked Torch
and ONNX Runtime wheels target Apple Silicon. Its model files are still restored;
local ML inference needs a supported vision host. `--lean --role web` selects
that dependency profile explicitly on other supported computers.

The native TUI coordinates a same-computer stack only on Windows/Linux x86-64.
For the Mac workflow, run CARLA and the Worker on Windows/Linux. The same-host
TUI binds the Worker to loopback by default; a Mac cannot connect to that address.
Use the existing [LAN Worker command](../README.md#run-the-windows-world-worker)
with `--bind <host-LAN-address> --allow-lan`, a shared Worker token and the
appropriate firewall rule. Keep one Worker owner per port and configure the
Mac's `.env.local` with that URL/token. Restoration does not alter firewall or
Worker exposure automatically.

## Linux: CARLA and the web app together

For a Linux x86-64 host:

```bash
sh restore.sh --destination "$HOME/CARLA-Vision" --role local --carla-root /path/to/CARLA_0.9.16
cd "$HOME/CARLA-Vision"
sh deploy/desktop/launch.sh
```

Use `--download-carla` instead of `--carla-root` to request the official server.

## What is restored

| Source | Included |
| --- | --- |
| GitHub source at one resolved commit | Application, compiled browser console, configuration examples, scripts and documentation |
| Published `models/assets.json` | All ten full weight files (732,845,619 bytes), labels, pinned processor/configuration files and notices; every declared size/SHA-256 checked |
| Published `samples/archive.json` | All 123 historical payloads, including five videos; every SHA-256 checked |
| Sample original paths | Copies under the recorded `output/`, `reports/`, `datasets/` and model-table paths, unless `--no-sample-copies` |
| Locked Python environment | Managed Python 3.12, `.venv`, private uv and recording encoder; vision profile by default except Intel macOS/`--lean` |
| Optional CARLA package | Windows/Linux x86-64 only, when `--download-carla` is requested |
| Optional private local backup | Original local configuration, sessions, native task packages and retained build artifacts |

Downloaded model payloads are hydrated from GitHub LFS even when Git/LFS are
not installed. The default source-archive method creates runnable project files
without Git history. For a shallow Git checkout, install Git and Git LFS first
and add `--git`. That checkout gets an `origin` remote and a `restored` branch.
Do future development on a separate feature branch.

The source archive is obtained over HTTPS at the recorded GitHub commit; model,
sample and uv archive checksums provide their declared payload verification.
The bootstrap itself is trusted source downloaded from this repository. This
does not claim a separately signed installer or upstream CARLA archive checksum.

## Keep private state outside the project

GitHub does not store `.env.local`, machine credentials, private sessions or
installed environments. The local cleanup saved `local-state.zip`, its checksum,
an environment inventory and a cleanup report in a dated directory under the
sibling `CARLA PY.local-backups/` folder. Recovery tools and instructions are
copied there too. Keep that folder, and preferably another copy on a separate
disk, before removing the original project. Never upload its private archive.

To import it while restoring on macOS/Linux:

```bash
sh restore.sh --destination "$HOME/CARLA-Vision" --role web \
  --local-backup "/path/to/CARLA PY.local-backups/<dated-folder>/local-state.zip"
```

On Windows, pass the same `--local-backup "C:\path\local-state.zip"` option to
`Restore.cmd`. All declared backup payloads are verified before import; a
different existing file is preserved and restoration stops. Process receipts,
locks, control files, environments and Python bytecode caches are not imported.
The POSIX scratch file `:memory:.ses` remains in the archive when restoring on
Windows, where its name is invalid. Original machine-local settings and historical
paths are retained; review host addresses and workspace paths for the new machine.

The environment inventory is a reference. Restoration uses the committed
`uv.lock`; it does not blindly reinstall an old environment's freeze output.
Full Git history, unpushed branches and stashes are not recreated by the source
archive. They were preserved locally during cleanup, and should be backed up
separately if they are needed after deleting `.git`.

## Retry and optional profiles

An existing nonempty project is never chosen as a fresh destination. Restoration
normally defaults to `carla-local-autonomy-restored/` in the current
directory; when run from a project root, it chooses a sibling directory instead.
An explicit `--destination` is recommended for a new installation. A failed
download/install leaves a `.restore-state.json` marker at its resolved commit.
Close the target project's TUI and services, then retry with the same destination
and `--resume`; do not use `--resume` to repair arbitrary old checkouts.

```bash
sh restore.sh --destination "$HOME/CARLA-Vision" --role web --resume
```

`--no-setup` restores source/models/samples only. `--research` adds locked
research/experimental dependencies, while `--dev` includes developer tooling.
Neither enables experimental driving automatically. To inspect options with an
existing Python 3.11+ installation:

```bash
python3 restore_project.py --help
python3 restore_project.py --destination ../CARLA-Vision --role web --plan
```

The Python engine's `--plan` does not download or install anything. The shell
and PowerShell wrappers prepare their bootstrap runtime before forwarding any
engine option, including `--plan` or `--help`.

## Verification status

Source/static checks, CLI help/plan and existing payload integrity were checked.
Recovery tests exercise archive rejection, private-file preservation, corrupt
downloads, process locks and published assets; CI runs these on Windows, Linux
and macOS without installing the ML/native runtime. Full fresh-machine restoration,
native installer/TUI execution, inference and live CARLA acceptance remain deferred.
