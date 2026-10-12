"""Full-screen installation, readiness and lifecycle UI for the local stack."""

from __future__ import annotations

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

from .config import LauncherConfig, LauncherError, find_carla
from .install import setup
from .services import LocalStack


class LauncherApp(App):
    TITLE = "CARLA Vision"
    SUB_TITLE = "Local launcher"
    BINDINGS = [
        ("ctrl+r", "start_stack", "Start"),
        ("ctrl+s", "stop_stack", "Stop"),
        ("ctrl+b", "open_web", "Open web"),
        ("ctrl+q", "quit_launcher", "Exit"),
    ]
    CSS = """
    Screen { background: #f0efeb; color: #202428; }
    Header, Footer { background: #202428; color: #ffffff; }
    #main { padding: 1 2; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 0; }
    #summary { height: auto; min-height: 2; margin-bottom: 1; }
    #services { height: 8; margin-bottom: 1; background: #ffffff; }
    #actions { height: auto; min-height: 3; }
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
    #setup-actions { height: auto; margin-top: 1; }
    #logs { height: 1fr; background: #ffffff; border: solid #d9d8d2; }
    #exit-note { height: auto; color: #a26017; margin-top: 1; }
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
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
            with TabbedContent(initial="run"):
                with TabPane("Run", id="run"):
                    yield Static("Checking local readiness…", id="summary", markup=False)
                    yield DataTable(id="services", cursor_type="row")
                    with Horizontal(id="actions"):
                        yield Button("Start all", variant="primary", id="start", disabled=True)
                        yield Button("Open web app", id="browser", disabled=True)
                        yield Button("Stop owned services", id="stop", disabled=True)
                    yield Static("Choose Setup for first-time installation.", id="stage", markup=False)
                    yield ProgressBar(total=100, show_eta=False, id="progress")
                    yield Static("", id="exit-note", markup=False)
                with TabPane("Setup", id="setup"):
                    with VerticalScroll(id="setup-form"):
                        yield Static(
                            "Install once, then start from Run. CARLA, Worker and the web app use "
                            "this computer's loopback connection.", markup=False,
                        )
                        yield Label("CARLA folder")
                        yield Input(
                            value=config.carla_root or (str(detected.parent) if detected else ""),
                            placeholder="Folder containing CarlaUE4.exe or CarlaUE4.sh", id="carla-root",
                        )
                        yield Label("Installation profile")
                        yield Select(
                            [("Vision — all bundled inference models", "vision"),
                             ("Research — vision plus training/evaluation tools", "research")],
                            value=config.profile, allow_blank=False, id="profile",
                        )
                        yield Checkbox(
                            "Download official CARLA 0.9.16 if no local folder is found "
                            "(multi-GB; at least 35 GiB free)", id="download-carla",
                        )
                        yield Label("Local ports")
                        with Horizontal(id="ports"):
                            for name, value in [
                                ("carla-port", config.carla_port), ("worker-port", config.worker_port),
                                ("web-port", config.web_port), ("traffic-manager-port", config.traffic_manager_port),
                            ]:
                                with Vertical(classes="port-field"):
                                    yield Label(name.removesuffix("-port").replace("-", " ").title())
                                    yield Input(str(value), type="integer", id=name)
                        yield Label("Simulator quality")
                        yield Select(
                            [("Low — shared-machine default", "Low"), ("Epic", "Epic")],
                            value=config.quality, allow_blank=False, id="quality",
                        )
                        yield Checkbox("Render off-screen", value=config.offscreen, id="offscreen")
                        with Horizontal(id="setup-actions"):
                            yield Button("Save settings", id="save-settings")
                            yield Button("Install / Repair", variant="primary", id="install")
                        yield Static(
                            "Credentials are generated locally. The installer does not change "
                            "the web app's scene or driving settings.", markup=False,
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
            carla_root=self.query_one("#carla-root", Input).value.strip(),
            profile=str(self.query_one("#profile", Select).value),
            quality=str(self.query_one("#quality", Select).value),
            offscreen=self.query_one("#offscreen", Checkbox).value,
            carla_port=int(self.query_one("#carla-port", Input).value),
            worker_port=int(self.query_one("#worker-port", Input).value),
            web_port=int(self.query_one("#web-port", Input).value),
            traffic_manager_port=int(self.query_one("#traffic-manager-port", Input).value),
        )

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        self._status_generation += 1
        for selector in ("#start", "#stop", "#install", "#save-settings"):
            self.query_one(selector, Button).disabled = value
        self.query_one("#progress", ProgressBar).update(total=None if value else 100, progress=0)
        if not value:
            self.refresh_status()

    def _render_status(self, snapshot: dict) -> None:
        self.last_snapshot = snapshot
        table = self.query_one("#services", DataTable)
        colors = {"ready": "#237a55", "stopped": "#6c7278", "unavailable": "#b23b3b", "running": "#a26017"}
        for role, row in snapshot["services"].items():
            table.update_cell(role, "state", Text(row["status"].capitalize(), style=colors[row["status"]]))
            detail = row.get("detail", "")
            if row.get("status") == "ready" and role == "carla" and not row.get("owned"):
                detail += " · external; preserved on Stop"
            table.update_cell(role, "details", Text(detail), update_width=True)
        models = snapshot["models"]
        unavailable = [
            (role, row) for role, row in snapshot["services"].items() if row["status"] == "unavailable"
        ]
        if snapshot["ready"]:
            summary = "All local services are ready. Open the web app to use Garage."
        elif unavailable:
            role, row = unavailable[0]
            summary = f"{role.capitalize()} needs attention: {row.get('detail', 'See Logs.')}"
        elif not models.get("ready"):
            summary = f"Model files: {models.get('present', 0)} / {models.get('total', 0)}. Use Setup → Install / Repair."
        else:
            summary = "Models are present. Select the CARLA folder in Setup, then Start all."
        self.query_one("#summary", Static).update(Text(summary))
        self.query_one("#browser", Button).disabled = snapshot["services"]["web"]["status"] != "ready"
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
                self.call_from_thread(self._finish_probe, generation, snapshot, error, chunks, updated)

    def _finish_probe(
        self, generation: int, snapshot: dict | None, error: str | None,
        chunks: list[dict], updated: dict[str, int],
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
    def operate(self, operation: str, config: LauncherConfig | None = None, download: bool = False) -> None:
        succeeded = False
        try:
            if operation == "install":
                assert config is not None
                self.stack.configure(config)
                setup(self.root, download_carla=download, report=self._report, cancel=self.cancel_install)
            elif operation == "save":
                assert config is not None
                self.stack.configure(config)
                self._report("Settings saved. Use Start all when ready.")
            elif operation == "start":
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
            self.notify("Operation failed. See the stage message or Logs for recovery.", severity="error")

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        name = event.button.id
        if name == "browser":
            self.action_open_web()
            return
        if self.busy:
            return
        try:
            config = self._settings() if name in {"install", "save-settings"} else None
        except (ValueError, LauncherError) as error:
            self.notify(str(error), severity="error")
            return
        operation = {"start": "start", "stop": "stop", "install": "install", "save-settings": "save"}.get(name)
        if operation:
            self._set_busy(True)
            self.cancel_install.clear()
            self.operate(operation, config, self.query_one("#download-carla", Checkbox).value)

    def action_start_stack(self) -> None:
        if not self.busy:
            self._set_busy(True)
            self.operate("start")

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
