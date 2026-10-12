# Same-computer setup preview

The [Windows ZIP](CARLA-Vision-setup-preview.zip) and
[Linux archive](CARLA-Vision-setup-preview.tar.gz) contain a self-bootstrapping
full-screen launcher and the existing compiled web app, configurations, model
metadata and historical samples. Extract the entire archive; run **Install.cmd**
on Windows or **sh install.sh** on Linux. See [the installation guide](../docs/desktop_launcher.md).

Python, Git and Node do not need to be preinstalled for these bundles. Setup
installs private managed Python and the TUI; **Setup → Install / Repair** installs
the locked runtime and hydrates all ten model weights with SHA-256 verification.
An optional checkbox downloads the official CARLA server. A desktop/menu
shortcut opens the launcher for future sessions.

Reinstallation preserves edited configuration files and puts incoming defaults
in `.launcher/incoming-defaults/` for comparison. Close the launcher and its
owned services before updating; bootstrap checks the lifetime/operation locks
and verifies process creation time alongside the PID.

These are **preview artifacts**, pending Windows/Linux installation, visual and
live-CARLA validation. They are published on a feature branch for review, with
maintainer-owned PR and merge; they are not a validated production release.

[manifest.json](manifest.json) records the exact source commit, file count,
fixed model source, artifact sizes and hashes. [checksums.sha256](checksums.sha256)
covers the archives. Every indexed application file also has a hash inside the
archive. Integrity checks are independent of the deferred runtime acceptance.

The builder reads only committed tracked application inputs and creates valid
LFS pointers instead of duplicating 699 MiB of weights. Weights remain fully
available in GitHub LFS for ordinary clones and installer downloads. Local
environments, credentials, session data, simulator packages and caches are
excluded. The builder excludes this output directory to avoid self-inclusion.
