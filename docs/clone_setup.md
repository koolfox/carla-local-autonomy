# Start in a new environment

This guide brings together the source, retained models, dependency lock and
local connection settings. It does not run tests or claim new simulator or
model acceptance. The default `main` branch includes the published models,
historical samples, native launcher and recovery tools. The [recovery script](restore_project.md)
can prepare these files and dependencies without preinstalled Git or Python.

For Windows/Linux users who want CARLA and the web app on one computer, use the
[full-screen installer/launcher](desktop_launcher.md). Its source bundles do not
need preinstalled Git, Python or Node; they install the private environment and
hydrate model weights from their fixed GitHub source.

## 1. Get the full source and models

Install Git, [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage),
Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). Then:

```bash
git lfs install
git clone https://github.com/koolfox/carla-local-autonomy.git
cd carla-local-autonomy
git lfs pull
```

The clone includes ten weight
files, exact class tables, pinned processor/configuration files and the
[historical sample archive](../samples/README.md). The complete model inventory
and original-file hashes are under [`models/`](../models/README.md).

If a weight file is only a few hundred bytes and begins with
`version https://git-lfs.github.com/spec/v1`, it is an LFS pointer. Complete
`git lfs pull` before model selection. GitHub's ordinary source ZIP is not the
documented way to obtain a complete LFS-enabled working environment.

## 2. Install the locked Operator environment

From the repository root, install every existing supported profile:

```bash
uv sync --all-extras --all-groups --frozen
```

For a smaller install, use the [existing profile commands](../README.md#install-profiles).
Choose the profiles in the [model table](../models/README.md#models-and-runtime-paths)
for the models you intend to use. Install FFmpeg for browser-compatible recording
and saved-video conversion. The reviewed production Svelte bundle is already
tracked and packaged; Node/npm is needed for frontend development, not normal
Operator use.

The same-computer launcher selects managed Python 3.12 and installs a matching
official PythonAPI automatically. The exact CUDA/GPU runtime depends on the new machine; model availability does
not establish GPU compatibility. The CARLA-host Worker has its own lighter
environment and matching official CARLA 0.9.16 PythonAPI, described in
[`native_host_runbook.md`](native_host_runbook.md).

## 3. Set local endpoints and fresh credentials

On macOS/Linux:

```bash
cp .env.local.example .env.local
```

In PowerShell:

```powershell
Copy-Item .env.local.example .env.local
```

Edit `.env.local` for this machine's CARLA/Worker addresses and a fresh matching
Worker token. The existing maintainer's `.env.local` and active session state
are not portable setup inputs. See [`local_configuration.md`](local_configuration.md).

Run the Operator from the repository root so its configured workspace finds
the retained `models/` paths. To prevent Hugging Face metadata requests for the
pinned snapshots, optionally set `HF_HUB_OFFLINE=1` as described in the model guide.

```bash
uv run --no-sync carla-operator-ui
```

This starts the console using the existing entry point. Real preview or driving
still needs the separate CARLA server and authenticated World Worker. The
repository supplies their source/configuration and Operator-side model assets;
it does not embed a CARLA server installation or a machine-specific environment.
The same-computer launcher can prepare these services and optionally download
the official CARLA package through its Setup tab.

## 4. Find the retained examples

[`samples/README.md`](../samples/README.md) links the five original movies,
camera images, three-frame labelled pilot and eight report snapshots. These
remain historical evidence with their recorded limitations; a new checkout does
not turn them into new live results or a complete training benchmark. Runtime
outputs created on the new machine remain in the ignored workspace roots.
