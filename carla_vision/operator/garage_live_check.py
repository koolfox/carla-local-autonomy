"""One-click live-CARLA check over the Operator's existing World Worker.

This owner only coordinates the world handoff. The checks themselves are the
existing ``GarageAcceptanceRunner``; it gets its own drive manager because the
runner shuts its manager down when it finishes. Like teacher capture, the check
holds the world exclusively, so the Garage preview pauses and Drive is blocked
until it releases.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .garage_drive import _module_available
from .jobs import _atomic_json

SCHEMA_VERSION = "1.0"
DEFAULT_MAP = "Town10HD_Opt"
DEFAULT_VEHICLE = "vehicle.tesla.model3"
DEFAULT_SEED = 20260809


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    checks = [
        *report.get("checks", []),
        *(check for session in report.get("sessions", []) for check in session.get("checks", [])),
    ]
    required = [check for check in checks if check.get("required", True)]
    failed = [str(check.get("check_id")) for check in required if check.get("passed") is not True]
    error = report.get("runner_error") or report.get("shutdown_error")
    return {
        "status": report.get("status", "fail"),
        "passed_checks": len(required) - len(failed),
        "total_checks": len(required),
        "failed_checks": failed[:20],
        "error": error.get("message") if isinstance(error, dict) else None,
        "modes": (report.get("target") or {}).get("modes", []),
        "duration_seconds": report.get("duration_seconds"),
    }


class GarageLiveCheck:
    def __init__(
        self,
        application: Any,
        world_lock: threading.RLock,
        *,
        manager_factory: Callable[[bool], Any] | None = None,
    ) -> None:
        self.application, self.world_lock = application, world_lock
        self._manager_factory = manager_factory or self._default_manager
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.path = application.jobs.sessions_root / "live_check_state.json"
        self._state: dict[str, Any] = {
            "phase": "idle",
            "active": False,
            "holds_world": False,
            "started_at": None,
            "report": None,
            "summary": None,
            "error": None,
        }
        if self.path.is_file():
            try:
                self._state.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                pass
        if self._state["active"]:
            # The Operator restarted mid-check; the runner thread is gone.
            self._state.update(
                phase="failed",
                active=False,
                error="The Operator restarted during the live check; run it again.",
            )
        self._state["holds_world"] = False

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                **self._state,
                "available": self.application.world_worker is not None,
            }

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
            _atomic_json(self.path, self._state)

    def require_world_available(self) -> None:
        if self.state()["holds_world"]:
            raise RuntimeError("The live CARLA check owns the world; wait for it to finish")

    def _default_manager(self, behavior: bool) -> Any:
        from .garage_server import GarageOperatorDriveManager

        return GarageOperatorDriveManager(
            workspace=self.application.workspace,
            carla_host=self.application.carla_host,
            carla_port=self.application.carla_port,
            world_worker=self.application.world_worker,
            experimental_enabled=behavior,
        )

    def start(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict) or set(raw) != {"acknowledge"}:
            raise ValueError("the live check requires only an acknowledgement")
        if raw["acknowledge"] is not True:
            raise ValueError(
                "the live check reloads the map and drives a test vehicle; acknowledgement required"
            )
        if self.application.world_worker is None:
            raise ValueError("connect the World Worker before running the live check")
        if not self.world_lock.acquire(blocking=False):
            raise RuntimeError("Garage is updating; wait for it to finish before the live check")
        try:
            if self.state()["active"]:
                raise RuntimeError("a live check is already running")
            capture = getattr(self.application, "capture", None)
            if capture is not None:
                capture.require_world_available()
            if self.application.drive.state()["status"] in {"starting", "running", "stopping"}:
                raise RuntimeError("Stop & Save the active drive before the live check")
            self._update(
                phase="preparing",
                active=True,
                holds_world=True,
                started_at=datetime.now(UTC).isoformat(),
                report=None,
                summary=None,
                error=None,
            )
        finally:
            self.world_lock.release()
        self._thread = threading.Thread(target=self._run, daemon=True, name="garage-live-check")
        self._thread.start()
        return self.state()

    def _run(self) -> None:
        # Imported here: the acceptance module imports the Garage server, which owns this check.
        from .garage_acceptance import GarageAcceptanceRunner

        workspace = Path(self.application.workspace)
        output = workspace / "runs" / f"garage-acceptance-{_utc_stamp()}.json"
        try:
            with self.world_lock:
                self.application.preview.cancel_preparation()
                self.application.preview.stop_for_drive()
            behavior = _module_available("carla") and _module_available(
                "agents.navigation.behavior_agent"
            )
            self._update(phase="running")
            runner = GarageAcceptanceRunner(
                self._manager_factory(behavior),
                self.application.world_worker,
                host=self.application.carla_host,
                port=self.application.carla_port,
                map_name=DEFAULT_MAP,
                vehicle=DEFAULT_VEHICLE,
                seed=DEFAULT_SEED,
                traffic_count=5,
                walker_count=5,
                camera_fps=30.0,
                repeat=1,
                modes=("manual", "behavior") if behavior else ("manual",),
            )
            report = runner.run()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            summary = _summary(report)
            self._update(
                phase="passed" if summary["status"] == "pass" else "failed",
                active=False,
                holds_world=False,
                report=str(output.relative_to(workspace)),
                summary=summary,
                error=summary["error"],
            )
        except Exception as error:
            self._update(phase="failed", active=False, holds_world=False, error=str(error))

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2)
