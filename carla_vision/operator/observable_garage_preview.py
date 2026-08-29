"""Asynchronous Garage preview preparation backed by observable World Worker state.

The existing Garage preview session remains authoritative for CARLA ownership,
camera setup and cleanup.  This module only moves its long ``start()`` call off
the HTTP request thread and projects World Worker preparation progress into the
normal preview state response.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from .garage_preview import (
    _ACTIVE_DRIVE_STATES,
    GaragePreviewConfig,
    GaragePreviewManager,
    GaragePreviewSession,
)
from .world_worker_client import WorldWorkerClient


class ObservableGaragePreviewSession(GaragePreviewSession):
    """Expose Worker scene-preparation progress while the preview is starting."""

    def snapshot(self) -> dict[str, Any]:
        payload = super().snapshot()
        if payload.get("status") != "starting":
            payload["preparation"] = None
            return payload

        preparation: dict[str, Any] | None = None
        try:
            worker_state = self.world_worker.current_scene()
        except Exception as error:
            preparation = {
                "status": "waiting",
                "stage": "worker_status",
                "requested": {
                    "map": self.config.map_name,
                    "traffic": self.config.traffic_count,
                    "walkers": self.config.walker_count,
                    "prop_preset": self.config.prop_preset,
                    "route_mode": "free",
                },
                "actual": {"traffic": 0, "walkers": 0, "props": 0},
                "history": [],
                "elapsed_seconds": 0.0,
                "error": {
                    "code": type(error).__name__,
                    "message": str(error),
                    "stage": "worker_status",
                },
            }
        else:
            raw = worker_state.get("preparation")
            if isinstance(raw, Mapping):
                preparation = dict(raw)

        if preparation is None:
            with self._lock:
                worker_scene_ready = self._scene is not None
            preparation = {
                "status": "starting",
                "stage": "camera" if worker_scene_ready else "worker_prepare",
                "requested": {
                    "map": self.config.map_name,
                    "traffic": self.config.traffic_count,
                    "walkers": self.config.walker_count,
                    "prop_preset": self.config.prop_preset,
                    "route_mode": "free",
                },
                "actual": {
                    "traffic": self.config.traffic_count if worker_scene_ready else 0,
                    "walkers": self.config.walker_count if worker_scene_ready else 0,
                    "props": 0,
                },
                "history": [],
                "elapsed_seconds": 0.0,
                "error": None,
            }
        payload["preparation"] = preparation
        return payload


class AsyncGaragePreviewManager(GaragePreviewManager):
    """Return from configure immediately and reject duplicate world mutation."""

    def __init__(
        self,
        *,
        carla_host: str,
        carla_port: int,
        world_worker: WorldWorkerClient | None,
        drive_state: Callable[[], Mapping[str, Any]],
        world_mode_lock: threading.RLock,
        session_factory: Callable[..., GaragePreviewSession] = ObservableGaragePreviewSession,
    ) -> None:
        super().__init__(
            carla_host=carla_host,
            carla_port=carla_port,
            world_worker=world_worker,
            drive_state=drive_state,
            world_mode_lock=world_mode_lock,
            session_factory=session_factory,
        )
        self._launch_thread: threading.Thread | None = None

    def _launch_in_progress(self) -> bool:
        with self._lock:
            thread = self._launch_thread
        return thread is not None and thread.is_alive()

    def _worker_supports_observable_prepare(self) -> bool:
        worker = self.world_worker
        if worker is None:
            return False
        health = worker.health()
        capabilities = health.get("capabilities", {})
        return isinstance(capabilities, Mapping) and bool(
            capabilities.get("observable_scene_preparation")
        )

    def configure(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        config = GaragePreviewConfig.from_mapping(raw)
        if self.world_worker is None:
            raise RuntimeError("Garage preview requires a configured World Worker")

        # Older Workers keep current-scene behind the same long prepare lock.
        # Preserve the established synchronous behavior instead of creating a
        # background task whose progress endpoint would itself block.
        if not self._worker_supports_observable_prepare():
            return super().configure(raw)

        with self._world_mode_lock:
            drive_status = str(self._drive_state().get("status", "idle"))
            if drive_status in _ACTIVE_DRIVE_STATES:
                raise RuntimeError("end the active Drive before starting Garage preview")
            if self._launch_in_progress():
                raise RuntimeError("Garage preview preparation is already in progress")

            self._stop_locked(reason="reconfigure")
            session = self._session_factory(
                config,
                carla_host=self.carla_host,
                carla_port=self.carla_port,
                world_worker=self.world_worker,
            )
            thread = threading.Thread(
                target=self._start_session,
                args=(session,),
                name="garage-preview-prepare",
                daemon=True,
            )
            with self._lock:
                self._session = session
                self._last_error = None
                self._launch_thread = thread
            thread.start()
            return session.snapshot()

    def _start_session(self, session: GaragePreviewSession) -> None:
        try:
            session.start()
        except BaseException as error:
            with self._lock:
                if self._session is session:
                    self._last_error = f"{type(error).__name__}: {error}"
        finally:
            with self._lock:
                if self._launch_thread is threading.current_thread():
                    self._launch_thread = None

    def stop(self, raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._launch_in_progress():
            raise RuntimeError(
                "Garage preview preparation is in progress; wait for it to finish before closing"
            )
        return super().stop(raw)

    def stop_for_drive(self) -> None:
        if self._launch_in_progress():
            raise RuntimeError(
                "Garage preview preparation is in progress; wait for it to finish before Drive"
            )
        super().stop_for_drive()


__all__ = ["AsyncGaragePreviewManager", "ObservableGaragePreviewSession"]
