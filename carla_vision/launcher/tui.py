"""Full-screen installation, readiness and lifecycle UI for the local stack."""

from __future__ import annotations

import os
import platform
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .config import LauncherConfig, LauncherError, default_mode, find_carla, remote_token
from .install import setup
from .network import (
    export_connection,
    import_connection,
    lan_addresses,
    windows_firewall,
    windows_prerequisites,
)
from .services import LocalStack


class LauncherApp(App):
    TITLE = "CARLA Vision"
    SUB_TITLE = "Application launcher"
    BINDINGS = [
        ("ctrl+r", "start_stack", "Start"),
        ("ctrl+s", "stop_stack", "Stop"),
        ("ctrl+b", "open_web", "Open web"),
        ("ctrl+q", "quit_launcher", "Exit"),
    ]
    CSS = """
    Screen { background: #f0efeb; color: #202428; }
    Header, Footer { background: #202428; color: #ffffff; }
    FooterKey, FooterLabel { background: #202428; color: #ffffff; }
    FooterKey .footer-key--key, FooterKey .footer-key--description { background: #202428; color: #ffffff; }
    #main { padding: 0 2; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0; }
    #summary { height: auto; min-height: 2; margin-bottom: 1; }
    #services { height: 5; margin-bottom: 1; background: #ffffff; }
    #actions { height: auto; min-height: 3; layout: grid; grid-size: 3; grid-columns: 1fr 1fr 1fr; grid-rows: 3; }
    Button { margin-right: 1; min-width: 12; }
    Button.-primary { background: #315bea; color: #ffffff; }
    Button:focus, Input:focus, Select:focus { border: tall #315bea; }
    #stage { height: auto; min-height: 1; margin-top: 1; }
    #progress { height: 1; margin-top: 1; }
    #setup-form { padding-right: 1; }
    Label { margin-top: 1; margin-bottom: 0; height: auto; }
    Input, Select { width: 1fr; }
    #ports { height: auto; }
    .port-field { width: 1fr; margin-right: 1; height: auto; }
    #setup-actions, #pair-actions { height: auto; margin-top: 1; }
    .settings-group { height: auto; }
    #logs { height: 1fr; background: #ffffff; border: solid #d9d8d2; }
    #exit-note { height: auto; color: #a26017; margin-top: 1; }
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.theme = "textual-light"
        self.root = root
        self.stack = LocalStack(root, self._report)
        self.busy = False
        self.cancel_install = threading.Event()
        self.last_snapshot: dict = {}
        self.exit_when_stopped = False
        self._status_running = False
        self._status_generation = 0
        self._log_offsets: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        config = LauncherConfig.load(self.root)
        detected = find_carla(self.root, config.carla_root)
        yield Header()
        with Vertical(id="main"):
            with TabbedContent(
                initial="run" if (self.root / ".launcher/config.json").exists() else "setup"
            ):
                with TabPane("Run", id="run"):
                    yield Static("Checking local readiness…", id="summary", markup=False)
                    yield DataTable(id="services", cursor_type="row")
                    with Horizontal(id="actions"):
                        yield Button(
                            "Start web app" if config.mode == "web" else "Start all",
                            variant="primary",
                            id="start",
                            disabled=True,
                        )
                        yield Button("Open web app", id="browser", disabled=True)
                        yield Button("Stop", id="stop", disabled=True)
                    yield Static(
                        "Choose Setup for first-time installation.", id="stage", markup=False
                    )
                    yield ProgressBar(total=100, show_eta=False, id="progress")
                    yield Static("", id="exit-note", markup=False)
                with TabPane("Setup", id="setup"):
                    with VerticalScroll(id="setup-form"):
                        yield Static(
                            "Choose what runs here. Install / Repair prepares Python, packages, "
                            "models and recording automatically. Daily use is Start and Stop.",
                            markup=False,
                        )
                        yield Label("Run on this computer")
                        choices = [("Web app — connect to another CARLA computer", "web")]
                        if default_mode() == "local":
                            choices.insert(0, ("CARLA + Worker + web app", "local"))
                        yield Select(choices, value=config.mode, allow_blank=False, id="mode")
                        with Vertical(classes="settings-group", id="local-settings"):
                            yield Label("CARLA folder (detected automatically when possible)")
                            yield Input(
                                value=config.carla_root
                                or (str(detected.parent) if detected else ""),
                                placeholder="Folder containing CarlaUE4.exe or CarlaUE4.sh",
                                id="carla-root",
                            )
                            yield Checkbox(
                                "Download official CARLA 0.9.16 if missing (multi-GB; 35 GiB free)",
                                id="download-carla",
                            )
                            yield Button(
                                "Install CARLA prerequisites…",
                                id="prerequisites",
                                disabled=os.name != "nt",
                            )
                            yield Checkbox(
                                "Share Worker with another computer on this LAN",
                                value=config.share_worker,
                                id="share-worker",
                            )
                            yield Label(
                                "Worker LAN address (choose this computer's connected network)"
                            )
                            addresses = lan_addresses()
                            if config.worker_bind not in addresses:
                                addresses.insert(0, config.worker_bind)
                            yield Select(
                                [(value, value) for value in addresses],
                                value=config.worker_bind,
                                allow_blank=False,
                                id="worker-bind",
                            )
                            yield Button(
                                "Allow LAN in Windows Firewall…",
                                id="firewall",
                                disabled=os.name != "nt",
                            )
                            yield Static(
                                "Windows will ask for administrator permission for scoped Private-network access."
                                if os.name == "nt"
                                else "LAN sharing may require allowing the Worker/CARLA ports in your OS firewall.",
                                markup=False,
                            )
                        yield Label("Connection file path")
                        yield Input(
                            str(self.root / ".launcher/connection.json"), id="connection-file"
                        )
                        with Horizontal(id="pair-actions"):
                            yield Button("Export connection", id="export-connection")
                            yield Button("Import connection", id="import-connection")
                            yield Button("Check connection", id="check-connection")
                        with Vertical(classes="settings-group", id="remote-settings"):
                            with Collapsible(title="Enter connection manually", collapsed=True):
                                yield Label("CARLA host")
                                yield Input(
                                    config.carla_host, placeholder="192.168.1.100", id="carla-host"
                                )
                                yield Label("Worker URL")
                                yield Input(
                                    config.remote_worker_url,
                                    placeholder="http://192.168.1.100:8766",
                                    id="worker-url",
                                )
                                yield Label("Worker token (kept private)")
                                yield Input(
                                    remote_token(self.root), password=True, id="worker-token"
                                )
                                yield Static(
                                    "Import the connection file from the host TUI to fill these automatically. "
                                    "Leave URL and token empty to browse saved recordings offline.",
                                    markup=False,
                                )
                        yield Label("Application packages")
                        yield Select(
                            [("Web / recording — without ML inference", "lean")]
                            if platform.system() == "Darwin"
                            and platform.machine().lower() == "x86_64"
                            else [
                                ("Web / recording — without ML inference", "lean"),
                                ("Vision — all bundled inference models", "vision"),
                                ("Research — vision plus training/evaluation", "research"),
                            ],
                            value=config.profile,
                            allow_blank=False,
                            id="profile",
                        )
                        yield Label("Ports")
                        with Horizontal(id="ports"):
                            for name, value in [
                                ("carla-port", config.carla_port),
                                ("worker-port", config.worker_port),
                                ("web-port", config.web_port),
                                ("traffic-manager-port", config.traffic_manager_port),
                            ]:
                                with Vertical(classes="port-field"):
                                    yield Label(
                                        name.removesuffix("-port").replace("-", " ").title()
                                    )
                                    yield Input(str(value), type="integer", id=name)
                        with Vertical(classes="settings-group", id="render-settings"):
                            yield Label("Simulator quality")
                            yield Select(
                                [("Low — shared-machine default", "Low"), ("Epic", "Epic")],
                                value=config.quality,
                                allow_blank=False,
                                id="quality",
                            )
                            yield Checkbox(
                                "Render off-screen", value=config.offscreen, id="offscreen"
                            )
                        with Horizontal(id="setup-actions"):
                            yield Button("Save settings", id="save-settings")
                            yield Button("Install / Repair", variant="primary", id="install")
                        yield Static(
                            "Local Worker credentials are generated automatically. Connection files "
                            "contain a secret: transfer them privately. Garage settings remain in the web app.",
                            markup=False,
                        )
                with TabPane("Logs", id="log-tab"):
                    yield RichLog(id="logs", wrap=True, markup=False, max_lines=1500)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#services", DataTable)
        table.add_column("Service", key="service", width=10)
        table.add_column("State", key="state", width=12)
        table.add_column("Details", key="details")
        for name in ("carla", "worker", "web"):
            table.add_row(name.capitalize(), "Checking", "", key=name)
        self.query_one("#progress", ProgressBar).update(progress=0)
        self.set_interval(3, self.refresh_status)
        self._show_mode(config=self.stack.config)
        self.refresh_status()

    def _report(self, message: str) -> None:
        if self.is_running:
            self.call_from_thread(self._append_log, message)

    def _append_log(self, message: str) -> None:
        self.query_one("#logs", RichLog).write(Text(message))
        self.query_one("#stage", Static).update(Text(message))

    def _settings(self) -> LauncherConfig:
        return replace(
            self.stack.config,
            mode=str(self.query_one("#mode", Select).value),
            carla_host=self.query_one("#carla-host", Input).value.strip(),
            remote_worker_url=self.query_one("#worker-url", Input).value.strip(),
            share_worker=self.query_one("#share-worker", Checkbox).value,
            worker_bind=str(self.query_one("#worker-bind", Select).value),
            carla_root=self.query_one("#carla-root", Input).value.strip(),
            profile=str(self.query_one("#profile", Select).value),
            quality=str(self.query_one("#quality", Select).value),
            offscreen=self.query_one("#offscreen", Checkbox).value,
            carla_port=int(self.query_one("#carla-port", Input).value),
            worker_port=int(self.query_one("#worker-port", Input).value),
            web_port=int(self.query_one("#web-port", Input).value),
            traffic_manager_port=int(self.query_one("#traffic-manager-port", Input).value),
        )

    @on(Select.Changed, "#mode")
    def mode_changed(self, event: Select.Changed) -> None:
        if event.value in {"local", "web"}:
            self._show_mode(mode=str(event.value))

    @on(Checkbox.Changed, "#share-worker")
    def sharing_changed(self, event: Checkbox.Changed) -> None:
        if event.value and self.query_one("#worker-bind", Select).value == "127.0.0.1":
            addresses = lan_addresses()
            if addresses:
                self.query_one("#worker-bind", Select).value = addresses[0]
            else:
                self.notify(
                    "No private LAN address found. Connect to the network and reopen Setup.",
                    severity="warning",
                )

    def _show_mode(self, config: LauncherConfig | None = None, *, mode: str | None = None) -> None:
        mode = mode or config.mode
        local = mode == "local"
        self.query_one("#local-settings").display = local
        self.query_one("#render-settings").display = local
        self.query_one("#remote-settings").display = not local
        self.query_one("#export-connection", Button).display = local
        self.query_one("#import-connection", Button).display = not local
        self.query_one("#firewall", Button).disabled = self.busy or os.name != "nt"
        self.query_one("#prerequisites", Button).disabled = self.busy or os.name != "nt"
        self.query_one("#prerequisites", Button).display = os.name == "nt"
        self.query_one("#firewall", Button).display = os.name == "nt"
        self.query_one("#start", Button).label = "Start all" if local else "Start web app"

    def _load_connection(self) -> None:
        config = self.stack.config
        self.query_one("#mode", Select).value = config.mode
        self.query_one("#carla-host", Input).value = config.carla_host
        self.query_one("#worker-url", Input).value = config.remote_worker_url
        self.query_one("#carla-port", Input).value = str(config.carla_port)
        self.query_one("#worker-token", Input).value = remote_token(self.root)
        self._show_mode(config=config)

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self._status_generation += 1
        for selector in (
            "#start",
            "#stop",
            "#install",
            "#save-settings",
            "#import-connection",
            "#export-connection",
            "#check-connection",
            "#firewall",
            "#prerequisites",
        ):
            self.query_one(selector, Button).disabled = value
        self.query_one("#progress", ProgressBar).update(total=None if value else 100, progress=0)
        if not value:
            self._show_mode(config=self.stack.config)
            self.refresh_status()

    def _render_status(self, snapshot: dict) -> None:
        self.last_snapshot = snapshot
        table = self.query_one("#services", DataTable)
        colors = {
            "ready": "#237a55",
            "stopped": "#6c7278",
            "unavailable": "#b23b3b",
            "running": "#a26017",
            "external": "#6c7278",
        }
        for role, row in snapshot["services"].items():
            table.update_cell(
                role, "state", Text(row["status"].capitalize(), style=colors[row["status"]])
            )
            detail = row.get("detail", "")
            if row.get("status") == "ready" and role == "carla" and not row.get("owned"):
                detail += " · external; preserved on Stop"
            table.update_cell(role, "details", Text(detail), update_width=True)
        models = snapshot["models"]
        unavailable = [
            (role, row)
            for role, row in snapshot["services"].items()
            if row["status"] == "unavailable"
        ]
        if snapshot["ready"]:
            summary = (
                "Web app is ready."
                if snapshot["config"]["mode"] == "web"
                else "All local services are ready. Open the web app to use Garage."
            )
            if unavailable:
                summary += " Remote Worker needs attention; recordings remain available."
        elif unavailable:
            role, row = unavailable[0]
            summary = f"{role.capitalize()} needs attention: {row.get('detail', 'See Logs.')}"
        elif not models.get("ready"):
            summary = f"Model files: {models.get('present', 0)} / {models.get('total', 0)}. Use Setup → Install / Repair."
        else:
            summary = (
                "Use Start web app; configure the remote connection in Setup for Garage."
                if snapshot["config"]["mode"] == "web"
                else "Models are present. Select the CARLA folder in Setup, then Start all."
            )
        self.query_one("#summary", Static).update(Text(summary))
        self.query_one("#browser", Button).disabled = (
            snapshot["services"]["web"]["status"] != "ready"
        )
        if not self.busy:
            self.query_one("#start", Button).disabled = snapshot["ready"]
            self.query_one("#stop", Button).disabled = not snapshot["owned_services_active"]

    def refresh_status(self) -> None:
        # Thread workers are cooperatively cancelled; never overlap slow API probes.
        if self.busy or self._status_running:
            return
        self._status_running = True
        self._probe_status(self._status_generation, dict(self._log_offsets))

    @work(thread=True, exclusive=True, group="status", exit_on_error=False)
    def _probe_status(self, generation: int, offsets: dict[str, int]) -> None:
        snapshot, error, chunks, updated = None, None, [], offsets
        try:
            snapshot = self.stack.snapshot()
            chunks, updated = self.stack.read_logs(offsets)
        except (OSError, ValueError, LauncherError) as caught:
            error = str(caught)
        finally:
            if self.is_running:
                self.call_from_thread(
                    self._finish_probe, generation, snapshot, error, chunks, updated
                )

    def _finish_probe(
        self,
        generation: int,
        snapshot: dict | None,
        error: str | None,
        chunks: list[dict],
        updated: dict[str, int],
    ) -> None:
        self._status_running = False
        if generation != self._status_generation or self.busy:
            return
        if error:
            self.query_one("#summary", Static).update(Text(error))
        elif snapshot is not None:
            self._render_status(snapshot)
        logs = self.query_one("#logs", RichLog)
        for chunk in chunks:
            logs.write(Text(f"{chunk['role'].capitalize()} · {chunk['path']}"))
            logs.write(Text(chunk["text"]))
        self._log_offsets = updated

    @work(thread=True, exclusive=True, group="operation", exit_on_error=False)
    def operate(
        self,
        operation: str,
        config: LauncherConfig | None = None,
        download: bool = False,
        token: str | None = None,
        connection: Path | None = None,
    ) -> None:
        succeeded = False
        try:
            if operation == "install":
                assert config is not None
                self.stack.configure(config, token=token)
                setup(
                    self.root,
                    download_carla=download and config.mode == "local",
                    report=self._report,
                    cancel=self.cancel_install,
                )
            elif operation == "save":
                assert config is not None
                self.stack.configure(config, token=token)
                self._report("Settings saved. Use Start when ready.")
            elif operation == "import":
                assert connection is not None
                import_connection(self.stack, connection)
                self.call_from_thread(self._load_connection)
            elif operation == "export":
                assert config is not None
                self.stack.configure(config, token=token)
                assert connection is not None
                export_connection(self.stack, connection)
            elif operation == "check":
                assert config is not None
                self.stack.configure(config, token=token)
                self.stack.check_connection()
            elif operation == "prerequisites":
                assert config is not None
                self.stack.configure(config, token=token)
                windows_prerequisites(self.stack)
            elif operation == "firewall":
                assert config is not None
                self.stack.configure(config, token=token)
                windows_firewall(self.stack)
            elif operation == "start":
                if config is not None:
                    self.stack.configure(config, token=token)
                self.stack.start()
            else:
                self.stack.stop()
            succeeded = True
        except Exception as error:
            self._report(str(error))
        finally:
            if self.is_running:
                self.call_from_thread(self._finish_operation, succeeded)

    def _finish_operation(self, succeeded: bool) -> None:
        self._set_busy(False)
        if succeeded and self.exit_when_stopped:
            self.exit()
        elif not succeeded:
            self.exit_when_stopped = False
            self.notify(
                "Operation failed. See the stage message or Logs for recovery.", severity="error"
            )

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        name = event.button.id
        if name == "browser":
            self.action_open_web()
            return
        if self.busy:
            return
        try:
            config = (
                self._settings()
                if name
                in {
                    "install",
                    "save-settings",
                    "prerequisites",
                    "firewall",
                    "export-connection",
                    "check-connection",
                    "start",
                }
                else None
            )
        except (ValueError, LauncherError) as error:
            self.notify(str(error), severity="error")
            return
        operation = {
            "start": "start",
            "stop": "stop",
            "install": "install",
            "save-settings": "save",
            "import-connection": "import",
            "export-connection": "export",
            "check-connection": "check",
            "firewall": "firewall",
            "prerequisites": "prerequisites",
        }.get(name)
        if operation:
            self._set_busy(True)
            self.cancel_install.clear()
            self.operate(
                operation,
                config,
                self.query_one("#download-carla", Checkbox).value,
                self.query_one("#worker-token", Input).value.strip()
                if config and config.mode == "web"
                else None,
                Path(self.query_one("#connection-file", Input).value).expanduser(),
            )

    def action_start_stack(self) -> None:
        if not self.busy:
            try:
                config = self._settings()
            except (ValueError, LauncherError) as error:
                self.notify(str(error), severity="error")
                return
            self._set_busy(True)
            self.operate(
                "start",
                config,
                token=self.query_one("#worker-token", Input).value.strip()
                if config.mode == "web"
                else None,
            )

    def action_stop_stack(self) -> None:
        if not self.busy:
            self._set_busy(True)
            self.operate("stop")

    def action_open_web(self) -> None:
        if self.last_snapshot.get("services", {}).get("web", {}).get("status") == "ready":
            webbrowser.open(self.stack.config.web_url)

    def action_quit_launcher(self) -> None:
        if self.busy:
            self.query_one("#exit-note", Static).update(
                "An operation is active. Wait for completion before exiting."
            )
            return
        try:
            active = self.stack.active()
        except (OSError, ValueError, LauncherError) as error:
            self.notify(str(error), severity="error")
            return
        if active:
            self.exit_when_stopped = True
            self.query_one("#exit-note", Static).update("Stopping owned services before exit…")
            self.action_stop_stack()
        else:
            self.exit()
