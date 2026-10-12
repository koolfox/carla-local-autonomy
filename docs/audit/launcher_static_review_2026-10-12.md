# Local launcher static review — 2026-10-12

The finish reviewer returned **ship at static correction scope** and scored
all five listed findings resolved in source. This verdict covers those
corrections only. It does not assert whole-surface, Windows/Linux, visual or
live-CARLA acceptance.

The later installer correction review separately returned **ship at static
correction scope**, with four findings resolved in source. The five-findings UI
verdict below does not score those installer corrections.

The maintainer confirmed Windows first, Linux support and a full-screen TUI
for end-user installation, startup, shutdown and diagnostics. Runtime,
native, visual and CARLA testing were explicitly deferred to tomorrow.
No tests, screenshots, TUI execution, installer execution, service execution
or dependency installation were performed for this review. No commit or push
was performed by the documentation handoff.

## Authority and scope

`PRODUCT.md` supplies product truth and
`.impeccable/surfaces/launcher.md` supplies this surface's Operate brief.
The established browser console remains the interaction surface for Garage,
driving, perception and recordings. `web/src/lib/styles/app.css` is the
incumbent visual authority for this narrow terminal extension.

This handoff read the final `carla_vision/launcher/tui.py`, the relevant
launcher configuration, CLI, installation and service source, the incumbent CSS
and Impeccable's document/new-work guidance. It wrote only the launcher
surface brief and this audit record. It preserved product and incumbent
system files and did not create a global `DESIGN.md` or
`.impeccable/design.json`. No new qualitative brand commitments were inferred.

The initial context pass reported no `CONTEXT_STALE` finding. No further
pre-existing drift evidence was available. Source palette agreement is
documented below; rendered or project-wide visual drift was not assessed.
No drift repair was performed.

After later lock and installer edits, the documenter re-read `config.py` and
`cli.py`, refreshed configuration line references and recorded the TUI lifetime
lock below. The implementer reported no TUI changes. This documentation
recheck is not the independent installer verdict; that separate reviewer
assessed the four corrections recorded below.

## Installer correction verdict

The independent static reviewer scored all four corrections resolved:
customized/unindexed configuration preservation with separate incoming
defaults; lifetime and operation locks held across copying/environment sync;
creation-time-aware upgrade process guards; and matching home-directory fallback
when `XDG_DATA_HOME` is empty. Evidence is in `deploy/desktop/install.ps1`,
`deploy/desktop/copy_install.py`, `deploy/desktop/install.sh` and the launcher
configuration/CLI. No native installer or lifecycle acceptance was performed.

## Observed visual and interaction evidence

| Area | Source evidence | Observation and limit |
| --- | --- | --- |
| Palette | `web/src/lib/styles/app.css:1-26`; `carla_vision/launcher/tui.py:44-66,187-193` | Background `#f0efeb`, text `#202428`, white surfaces, action `#315bea`, divider `#d9d8d2`, success `#237a55`, muted `#6c7278`, danger `#b23b3b` and warning `#a26017` match incumbent values exactly. Service state words accompany color. Actual terminal rendering and contrast remain unverified. |
| Primary action | `web/src/lib/styles/app.css:759-790`; `carla_vision/launcher/tui.py:53-55,89-92,132-134` | Start all and Install / Repair use the incumbent blue with white text. Browser buttons use pixel padding and rounded corners; terminal controls use cell spacing and Textual divisions. Library-provided styles are not a documented global token change. |
| Type and hierarchy | `web/src/lib/styles/app.css:2-3,78-79`; `carla_vision/launcher/tui.py:80-141,143-150` | The browser uses an Inter/system stack and separate code typography. The TUI leaves font choice to the host terminal. Run is initial; summary precedes the three-row service ledger, action row, stage, progress and exit note. Setup and Logs have separate tabs. |
| Form | `carla_vision/launcher/tui.py:96-138`; `carla_vision/launcher/config.py:137-150` | Scrollable Setup exposes the CARLA folder, Vision/Research profiles, deliberate official download, four ports, simulator quality and off-screen rendering. Configuration validates port ranges and conflicts. Narrow-terminal clipping and navigation are deferred. |
| Launcher exclusivity | `carla_vision/launcher/config.py:68-121`; `carla_vision/launcher/cli.py:33-42,62-64` | Both the default and explicit `tui` CLI branches hold `.launcher/launcher.lock` around `LauncherApp.run()`. The source uses a nonblocking OS file lock and reports that another launcher or installer must be closed first on contention. Per-operation locking remains separate. Native exclusivity, contention and release behavior have not been exercised. |
| Keyboard and focus | `web/src/lib/styles/app.css:62-76`; `carla_vision/launcher/tui.py:38-43,55,175-182,280-329`; `carla_vision/launcher/services.py:122-126,273-285,556-578` | Footer binds Ctrl+R/S/B/Q. Buttons, Inputs and Selects have an explicit blue focus border; other focus styles come from Textual. Busy state disables mutation buttons; Start/Stop shortcuts check busy and rely on backend validation and operation locking. Open web checks the latest ready snapshot. Complete keyboard reachability and shortcut/button consistency have not been exercised. |
| Loading and status | `carla_vision/launcher/tui.py:87-95,143-160,175-182,184-247,249-278` | Initial summary/rows say Checking. Reported operation messages update stage and Logs; work runs in background threads. Busy progress is indeterminate and resets at idle, with no measured completion fraction. Status is scheduled every three seconds with one probe and stale-generation rejection. Responsiveness has not been measured. |
| Errors and recovery | `web/src/lib/styles/app.css:982-987`; `carla_vision/launcher/tui.py:198-207,266-278,288-292,313-329`; `carla_vision/launcher/services.py:287-314,549-574` | Summary prioritizes unavailable services, then model repair, then CARLA setup. Backend messages identify missing PythonAPI/models/console, occupied ports, missing CARLA and unconfirmed cleanup. Exceptions go to stage/Logs and a notification. Busy exit reports that completion must be awaited; unsuccessful stop prevents deferred exit. Error presentation and recovery remain unrendered. |
| Diagnostics | `carla_vision/launcher/tui.py:139-140,158-160,243-247`; `carla_vision/launcher/services.py:191-217,442-445` | Logs wrap and cap at 1,500 lines. Service reads use new chunks bounded to 16 KiB per file, role/path labels and retained last-run receipts; the Worker credential is redacted. File access, log refresh and long-path layout remain runtime gates. |

## Finish review corrections

The following table records the reviewer's reported findings and final
scores, checked against the current source. It is not a second full review.

| Listed finding | Final score | Source evidence |
| --- | --- | --- |
| Stop must parse owner replies and await actual scene/recorder cleanup, including a Worker busy/preparing guard. | Resolved in source | `carla_vision/launcher/services.py:456-552`: owner responses are parsed, cleanup errors fail, Drive must be terminal with recording false, preview/capture must be inactive and capture must not hold the world. Native capture cleanup is checked when required. Worker idle requires no scene/research job and rejects `busy`, `preparing`, `research_running` and `preparation.status == preparing`. |
| An absent Worker must permit shutdown of remaining owners when those owners are confirmed idle. | Resolved in source | `carla_vision/launcher/services.py:447-454,519-528`: connection absence is distinguished from HTTP authentication/server errors; finished web owners with a terminal capture phase permit stopping remaining owned processes. |
| A stale status callback must not overwrite operation state. | Resolved in source | `carla_vision/launcher/tui.py:175-182,213-247`: busy transitions increment generation, only one status probe runs, stale or busy callbacks are discarded and operation completion requests fresh status. |
| Logs must expose bounded new subprocess diagnostics and retained failed-run paths. | Resolved in source | `carla_vision/launcher/services.py:191-217,442-445`; `carla_vision/launcher/tui.py:243-247`: bounded offset reads use current or last-run receipts and append role/path-labelled text to Logs. |
| Web readiness must require the HTML entrypoint and compiled console contract. | Resolved in source | `carla_vision/launcher/services.py:91-95,293-296`: readiness reads bootstrap and nonempty HTML at `/`; startup requires `console_available()`. These source checks do not prove successful rendering. |

## Deferred acceptance gates

All gates below remain open. They record the maintainer's deferred validation
scope and are not executed checks.

- **Windows x86-64 first, then Linux x86-64:** install into the private Python
  environment, hydrate model assets, handle existing and deliberately
  downloaded CARLA, and exercise repair and readiness.
- **Native lifecycle and ownership:** start and stop owned processes, preserve
  external CARLA, verify Drive/recording/preview/capture cleanup, exercise
  Worker busy/preparing/absence and failed readiness, and retain diagnostics
  on incomplete cleanup or shutdown. Exercise launcher lifetime and operation
  lock contention and release.
- **Terminal visuals and keyboard:** inspect the full-screen layout, smaller
  terminal sizes, Setup scrolling, port fields, long status/log text, focus
  through every widget, shortcuts, busy states, loading, error visibility,
  terminal color support and contrast on both supported platforms.
- **Browser and live CARLA:** verify the compiled console opens after genuine
  API readiness, then confirm existing Garage, driving, perception and
  recording flows against CARLA 0.9.16 without replacing their ownership or
  model contracts.
