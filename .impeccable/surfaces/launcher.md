# Local launcher surface

Mode: Operate. Target: `carla_vision/launcher/tui.py`.

The maintainer confirmed Windows first with Linux support, a full-screen TUI,
automatic dependency/model setup and coordinated same-host server/web execution.
This is a narrowly scoped launch surface around the established browser console;
it inherits the console's neutral light surfaces, blue action emphasis and
plain operational language from `web/src/lib/styles/app.css`.

The first screen is a service ledger for CARLA, Worker and web readiness, followed
by Start, Open web app and Stop controls. Setup and Logs are separate tabs.
Avoid a second Garage, invented telemetry, decorative status metrics or any
model/actuation setting that belongs to the web console. Terminal typography and
square divisions are native affordances rather than a replacement product brand.

Keyboard navigation and visible focus must cover all controls. Installation and
startup run outside the UI thread, show their actual stage and keep duplicate
actions disabled. Missing models, no selected CARLA folder, occupied ports,
failed readiness and incomplete cleanup need explicit recovery text.

No runtime, visual or live-CARLA acceptance is asserted today: the maintainer
deferred testing. Review today is limited to source and packaging integrity;
Windows/Linux rendering and native process lifecycle remain tomorrow's gates.

## Observed implementation — 2026-10-12

This record describes the finished source; it does not establish a new global
design system. The browser console's `web/src/lib/styles/app.css` remains the
visual authority. No global `DESIGN.md` or `.impeccable/design.json` was created
for this extension.

The TUI directly reuses the console's background (`#f0efeb`), text (`#202428`),
white surface (`#ffffff`), action blue (`#315bea`) and divider (`#d9d8d2`). Ready,
stopped, unavailable and running states pair words with the incumbent success
green (`#237a55`), muted text (`#6c7278`), danger red (`#b23b3b`) and warning
amber (`#a26017`). Header and Footer use the text color as a dark background
with white text. The host terminal supplies typography; spacing uses terminal
cells, and divisions follow Textual widgets. Widget defaults, terminal color
support and actual rendered contrast have not been observed.

Run is the initial tab. Its order is readiness summary, three-row service
ledger (Service / State / Details), Start all / Open web app / Stop owned
services, current stage, progress and exit note. Setup is a scrollable form
for the CARLA folder, Vision/Research profile, deliberate official CARLA
download, four local ports, Low/Epic quality and off-screen rendering. Logs is
a wrapping, bounded text log. Garage, driving, model choices and recordings
remain in the browser console.

Footer shortcuts are Ctrl+R (Start), Ctrl+S (Stop), Ctrl+B (Open web) and Ctrl+Q
(Exit). Buttons, Inputs and Selects have an explicit blue focus border.
Checkbox, tab and table focus behavior uses Textual defaults and remains a
native verification gate. Start, Stop, Install / Repair and Save settings are
disabled during an operation; keyboard Start/Stop also check the busy flag
and reach backend validation and the operation lock. Open web app is gated
by the latest web readiness snapshot.

The default and explicit `tui` CLI entry points hold `.launcher/launcher.lock`
for the full duration of `LauncherApp.run()`. A conflicting launcher/installer
lock reports that the other surface must be closed first. Backend operation
locks continue to serialize installation, configuration, start and stop.
Native lock contention and release behavior have not been exercised.

Lifecycle and installation work run in a background worker. Reported messages
update both the stage and Logs. Busy progress is indeterminate; the source
does not claim an installation completion percentage. A separate background
status worker is scheduled every three seconds, avoids overlapping probes
and rejects stale callbacks when an operation changes the status generation.
Service log reads append only new bounded chunks with their role and path,
including retained logs from the last run.

The summary names the first unavailable service, directs missing model files
to Setup → Install / Repair and directs a configured start to the CARLA folder.
Operation failures retain their message in the stage/Logs and produce an
error notification. Port, missing CARLA and incomplete cleanup recovery text
comes from the launcher backend. Stop and Exit use existing scene owners and
preserve owned services when cleanup cannot be confirmed; external CARLA is
identified in the ledger as preserved on Stop.

The finish reviewer returned **ship at static correction scope**, scoring all
five listed findings resolved in source. The later installer correction
review separately returned ship at static correction scope with four findings
resolved. This is a fix-list
verdict, not visual or runtime acceptance. See
`docs/audit/launcher_static_review_2026-10-12.md` for evidence and the remaining
Windows/Linux installation, process lifecycle, terminal and live-CARLA gates.
