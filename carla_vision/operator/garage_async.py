"""Non-blocking lifecycle facade for Garage preview configuration.

The existing :class:`GaragePreviewManager` remains authoritative for scene
coalescing, Drive/Preview exclusion, cleanup, and CARLA ownership.  This module
only moves one long-running ``configure`` call off the HTTP request thread and
publishes revisioned state through a ``Condition``.  It deliberately has no
FIFO work queue: while one operation is active a second start is rejected.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_TERMINAL_STATUSES = frozenset({"running", "failed"})


@dataclass
class GaragePreviewOperation:
    """One accepted Garage preview configuration operation."""

    operation_id: str
    status: str
    stage: str
    revision: int
    started_monotonic: float
    completed_monotonic: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    preparation: dict[str, Any] | None = None

    def snapshot(self, *, clock: Any = time.monotonic) -> dict[str, Any]:
        end = clock() if self.completed_monotonic is None else self.completed_monotonic
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "operation_id": self.operation_id,
            "status": self.status,
            "stage": self.stage,
            "revision": self.revision,
            "elapsed_seconds": round(max(0.0, float(end) - self.started_monotonic), 3),
            "error": None if self.error is None else dict(self.error),
        }
        if self.preparation is not None:
            payload["preparation"] = copy.deepcopy(self.preparation)
        if self.result is not None:
            payload["result"] = copy.deepcopy(self.result)
        return payload


class GaragePreviewAsyncFacade:
    """Run exactly one Garage ``configure`` call outside the HTTP handler.

    The wrapped manager keeps all existing locking and latest-request semantics.
    This facade does not queue scene requests.  A caller that tries to start a
    second operation while one is active receives a conflict and may retry after
    the terminal lifecycle event.
    """

    def __init__(self, manager: Any, *, clock: Any = time.monotonic) -> None:
        self.manager = manager
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._operations: dict[str, GaragePreviewOperation] = {}
        self._active_operation_id: str | None = None
        self._thread: threading.Thread | None = None
        self._progress_thread: threading.Thread | None = None
        provider = getattr(manager, "preparation_status", None)
        self._progress_provider = provider if callable(provider) else None
        self._progress_interval = 0.25
        self._closed = False

    def start(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Accept one operation and return before CARLA preparation completes."""

        payload = dict(raw)
        with self._condition:
            if self._closed:
                raise RuntimeError("Garage preview async facade is shutting down")
            active = self._active_operation_locked()
            if active is not None and active.status not in _TERMINAL_STATUSES:
                raise RuntimeError("Garage preview configuration is already in progress")

            operation = GaragePreviewOperation(
                operation_id=f"garage_{uuid.uuid4().hex}",
                status="starting",
                stage="accepted",
                revision=1,
                started_monotonic=self._clock(),
            )
            self._operations[operation.operation_id] = operation
            self._active_operation_id = operation.operation_id
            # Keep only a small bounded terminal history for reconnect/readback;
            # this is not a work queue and no old operation is ever executed.
            terminal_ids = [
                operation_id
                for operation_id, item in self._operations.items()
                if operation_id != operation.operation_id and item.status in _TERMINAL_STATUSES
            ]
            for operation_id in terminal_ids[:-7]:
                self._operations.pop(operation_id, None)

            thread = threading.Thread(
                target=self._run,
                args=(operation.operation_id, payload),
                name=f"garage-preview-configure-{operation.operation_id}",
                daemon=True,
            )
            monitor = threading.Thread(
                target=self._monitor_progress,
                args=(operation.operation_id,),
                name=f"garage-preview-progress-{operation.operation_id}",
                daemon=True,
            )
            accepted = operation.snapshot(clock=self._clock)
            self._thread = thread
            self._progress_thread = monitor
            thread.start()
            monitor.start()
            return accepted

    def snapshot(self, operation_id: str) -> dict[str, Any]:
        with self._condition:
            return self._operation_locked(operation_id).snapshot(clock=self._clock)

    def active_snapshot(self) -> dict[str, Any] | None:
        with self._condition:
            active = self._active_operation_locked()
            return None if active is None else active.snapshot(clock=self._clock)

    def wait_for_update(
        self,
        operation_id: str,
        after_revision: int,
        *,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Block until the operation revision advances; never poll the manager."""

        if timeout <= 0.0:
            raise ValueError("timeout must be positive")
        deadline = self._clock() + float(timeout)
        with self._condition:
            while True:
                operation = self._operation_locked(operation_id)
                if operation.revision > after_revision:
                    return operation.snapshot(clock=self._clock)
                remaining = deadline - self._clock()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def close(self) -> None:
        """Prevent future starts; the manager remains responsible for shutdown."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _active_operation_locked(self) -> GaragePreviewOperation | None:
        if self._active_operation_id is None:
            return None
        return self._operations.get(self._active_operation_id)

    def _operation_locked(self, operation_id: str) -> GaragePreviewOperation:
        try:
            return self._operations[str(operation_id)]
        except KeyError as error:
            raise KeyError("Garage preview operation not found") from error

    def _update(
        self,
        operation_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        result: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
        terminal: bool = False,
    ) -> None:
        with self._condition:
            operation = self._operation_locked(operation_id)
            if status is not None:
                operation.status = status
            if stage is not None:
                operation.stage = stage
            if result is not None:
                operation.result = dict(result)
            if error is not None:
                operation.error = {
                    "type": type(error).__qualname__,
                    "message": str(error),
                }
            if terminal:
                operation.completed_monotonic = self._clock()
            operation.revision += 1
            self._condition.notify_all()

    def _monitor_progress(self, operation_id: str) -> None:
        provider = self._progress_provider
        if provider is None:
            return
        last_semantic: dict[str, Any] | None = None
        last_emit = 0.0
        while True:
            with self._condition:
                operation = self._operation_locked(operation_id)
                if operation.status in _TERMINAL_STATUSES or self._closed:
                    return
            try:
                raw = provider()
            except BaseException:
                raw = None
            if isinstance(raw, Mapping) and raw.get("status") == "preparing":
                preparation = copy.deepcopy(dict(raw))
                semantic = copy.deepcopy(preparation)
                semantic.pop("elapsed_seconds", None)
                now = self._clock()
                if semantic != last_semantic or now - last_emit >= 1.0:
                    with self._condition:
                        operation = self._operation_locked(operation_id)
                        if operation.status in _TERMINAL_STATUSES or self._closed:
                            return
                        operation.preparation = preparation
                        operation.stage = str(preparation.get("stage") or "configuring")
                        operation.revision += 1
                        self._condition.notify_all()
                    last_semantic = semantic
                    last_emit = now
            with self._condition:
                operation = self._operation_locked(operation_id)
                if operation.status in _TERMINAL_STATUSES or self._closed:
                    return
                self._condition.wait(self._progress_interval)

    def _run(self, operation_id: str, raw: Mapping[str, Any]) -> None:
        self._update(operation_id, stage="configuring")
        try:
            result = self.manager.configure(raw)
        except BaseException as error:
            self._update(
                operation_id,
                status="failed",
                stage="failed",
                error=error,
                terminal=True,
            )
            return
        self._update(
            operation_id,
            status="running",
            stage="running",
            result=result,
            terminal=True,
        )


__all__ = ["GaragePreviewAsyncFacade", "GaragePreviewOperation"]
