"""Observable World Worker variant for dense CARLA scene preparation.

The legacy :mod:`world_worker` remains the authoritative actor/session owner.
This module subclasses it narrowly so long-running scene population can keep
health and current-scene requests responsive while exposing bounded progress.
It can be executed either as a package module or directly beside
``world_worker.py`` on the CARLA host.
"""

from __future__ import annotations

import copy
import importlib.util
import logging
import sys
import threading
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


def _load_base() -> ModuleType:
    if __package__:
        from . import world_worker

        return world_worker
    path = Path(__file__).resolve().with_name("world_worker.py")
    spec = importlib.util.spec_from_file_location("carla_observable_world_worker_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load World Worker base module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()
SceneConfig = base.SceneConfig
WorkerError = base.WorkerError
WorldWorker = base.WorldWorker


class ObservableWorldWorker(WorldWorker):
    """Keep the Worker control plane responsive during synchronous prepare.

    ``WorldWorker.prepare`` deliberately remains the single authoritative
    implementation for map changes, actor ownership, exact population checks,
    rollback, and lease creation.  It still serializes those simulator
    mutations under its existing lock.  This subclass adds a separate progress
    lock/gate so read-only liveness endpoints never wait behind a dense spawn.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prepare_gate = threading.Lock()
        self._preparation_lock = threading.RLock()
        self._preparation: dict[str, Any] = {
            "status": "idle",
            "stage": "idle",
            "requested": {},
            "actual": {},
            "history": [],
            "error": None,
            "scene_id": None,
            "started_monotonic": None,
            "completed_monotonic": None,
        }
        self._health_cache: dict[str, Any] | None = None
        self._scene_cache: dict[str, Any] | None = None

    def _capabilities(self, client: Any | None = None) -> dict[str, bool]:
        capabilities = super()._capabilities(client)
        capabilities.update(
            {
                "observable_scene_preparation": True,
                "responsive_prepare_health": True,
                "prepare_progress_stages": True,
            }
        )
        return capabilities

    def _preparation_snapshot(self) -> dict[str, Any]:
        with self._preparation_lock:
            payload = copy.deepcopy(self._preparation)
        started = payload.pop("started_monotonic", None)
        completed = payload.pop("completed_monotonic", None)
        if started is None:
            payload["elapsed_seconds"] = 0.0
        else:
            end = self._clock() if completed is None else completed
            payload["elapsed_seconds"] = round(max(0.0, float(end) - float(started)), 3)
        return payload

    def _is_preparing(self) -> bool:
        with self._preparation_lock:
            return self._preparation.get("status") == "preparing"

    def _begin_preparation(self, config: Any) -> None:
        now = self._clock()
        requested = {
            "map": config.map_name,
            "traffic": int(config.traffic_count),
            "walkers": int(config.walker_count),
            "prop_preset": config.prop_preset,
            "route_mode": config.route_mode,
        }
        with self._preparation_lock:
            self._preparation = {
                "status": "preparing",
                "stage": "map",
                "requested": requested,
                "actual": {
                    "traffic": 0,
                    "walkers": 0,
                    "props": 0,
                },
                "history": [{"stage": "map", "elapsed_seconds": 0.0}],
                "error": None,
                "scene_id": None,
                "started_monotonic": now,
                "completed_monotonic": None,
            }
        logger.info(
            "scene_prepare status=preparing stage=map map=%s traffic=%d walkers=%d props=%s route=%s",
            requested["map"],
            requested["traffic"],
            requested["walkers"],
            requested["prop_preset"],
            requested["route_mode"],
        )

    def _set_preparation_stage(self, stage: str, **updates: Any) -> None:
        stage_changed = False
        elapsed = 0.0
        actual: dict[str, Any] = {}
        with self._preparation_lock:
            if self._preparation.get("status") != "preparing":
                return
            if self._preparation.get("stage") != stage:
                started = self._preparation.get("started_monotonic")
                elapsed = 0.0 if started is None else max(0.0, self._clock() - float(started))
                self._preparation["stage"] = stage
                self._preparation.setdefault("history", []).append(
                    {"stage": stage, "elapsed_seconds": round(elapsed, 3)}
                )
                stage_changed = True
            for name, value in updates.items():
                if name in {"traffic", "walkers", "props"}:
                    self._preparation.setdefault("actual", {})[name] = int(value)
                else:
                    self._preparation[name] = value
            actual = dict(self._preparation.get("actual", {}))
        if stage_changed:
            logger.info(
                "scene_prepare status=preparing stage=%s elapsed_seconds=%.3f traffic=%d walkers=%d props=%d",
                stage,
                elapsed,
                int(actual.get("traffic", 0)),
                int(actual.get("walkers", 0)),
                int(actual.get("props", 0)),
            )

    def _fail_preparation(self, error: BaseException) -> None:
        now = self._clock()
        code = getattr(error, "code", type(error).__name__)
        message = getattr(error, "message", str(error))
        with self._preparation_lock:
            failed_stage = str(self._preparation.get("stage", "unknown"))
            started = self._preparation.get("started_monotonic")
            elapsed = 0.0 if started is None else max(0.0, now - float(started))
            self._preparation["status"] = "failed"
            self._preparation["completed_monotonic"] = now
            self._preparation["error"] = {
                "code": str(code),
                "message": str(message),
                "stage": failed_stage,
            }
            self._preparation.setdefault("history", []).append(
                {"stage": "failed", "elapsed_seconds": round(elapsed, 3)}
            )
        logger.error(
            "scene_prepare status=failed stage=%s elapsed_seconds=%.3f code=%s message=%s",
            failed_stage,
            elapsed,
            code,
            message,
        )

    def _complete_preparation(self) -> None:
        self._set_preparation_stage("route")
        self._set_preparation_stage("ready")
        now = self._clock()
        with self._lock:
            scene = self._scene
            scene_id = None if scene is None else str(scene.scene_id)
            actual = {
                "traffic": 0 if scene is None else len(scene.vehicle_actors),
                "walkers": 0 if scene is None else len(scene.walker_actors),
                "props": 0 if scene is None else len(scene.prop_actors),
            }
        with self._preparation_lock:
            started = self._preparation.get("started_monotonic")
            elapsed = 0.0 if started is None else max(0.0, now - float(started))
            self._preparation["status"] = "ready"
            self._preparation["stage"] = "ready"
            self._preparation["scene_id"] = scene_id
            self._preparation["actual"] = actual
            self._preparation["completed_monotonic"] = now
        logger.info(
            "scene_prepare status=ready stage=ready scene_id=%s elapsed_seconds=%.3f traffic=%d walkers=%d props=%d",
            scene_id,
            elapsed,
            actual["traffic"],
            actual["walkers"],
            actual["props"],
        )

    def _prime_control_plane_cache(self) -> None:
        health = super().health()
        scene = super().current_scene()
        with self._preparation_lock:
            self._health_cache = copy.deepcopy(health)
            self._scene_cache = copy.deepcopy(scene)

    def health(self) -> dict[str, Any]:
        preparation = self._preparation_snapshot()
        if preparation["status"] == "preparing":
            with self._preparation_lock:
                cached = copy.deepcopy(self._health_cache)
            if cached is None:
                cached = {
                    "schema_version": base.SCHEMA_VERSION,
                    "worker_api_revision": base.WORKER_API_REVISION,
                    "status": "preparing",
                    "ready": True,
                    "error_code": None,
                    "carla": {
                        "connected": True,
                        "host": self.carla_host,
                        "port": self.carla_port,
                        "client_version": None,
                        "server_version": None,
                        "current_map": None,
                    },
                    "active_scene": None,
                    "capabilities": self._capabilities(None),
                }
            cached["status"] = "preparing"
            cached["preparation"] = preparation
            cached["busy"] = True
            capabilities = dict(cached.get("capabilities", {}))
            capabilities.update(self._capabilities(None))
            cached["capabilities"] = capabilities
            return cached

        payload = super().health()
        payload["preparation"] = preparation
        payload["busy"] = False
        with self._preparation_lock:
            self._health_cache = copy.deepcopy(payload)
        return payload

    def current_scene(self) -> dict[str, Any]:
        preparation = self._preparation_snapshot()
        if preparation["status"] == "preparing":
            with self._preparation_lock:
                cached = copy.deepcopy(self._scene_cache)
            if cached is None:
                cached = {
                    "schema_version": base.SCHEMA_VERSION,
                    "worker_api_revision": base.WORKER_API_REVISION,
                    "status": "idle",
                    "scene": None,
                }
            cached["status"] = "preparing"
            cached["preparation"] = preparation
            return cached

        payload = super().current_scene()
        payload["preparation"] = preparation
        with self._preparation_lock:
            self._scene_cache = copy.deepcopy(payload)
        return payload

    def prepare(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        config = SceneConfig.from_mapping(raw)
        if not self._prepare_gate.acquire(blocking=False):
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "scene_preparing",
                "another scene preparation is already in progress",
            )
        try:
            self._prime_control_plane_cache()
            self._begin_preparation(config)
            try:
                result = super().prepare(raw)
            except BaseException as error:
                self._fail_preparation(error)
                raise
            self._complete_preparation()
            return result
        finally:
            self._prepare_gate.release()

    def configure(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        config = SceneConfig.from_mapping(
            {key: value for key, value in raw.items() if key != "lease_token"}
        )
        if not self._prepare_gate.acquire(blocking=False):
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "scene_preparing",
                "another scene preparation or configuration is already in progress",
            )
        try:
            self._prime_control_plane_cache()
            self._begin_preparation(config)
            scene = self._scene
            self._set_preparation_stage(
                "configure",
                scene_id=scene_id,
                operation="configure",
                traffic=0 if scene is None else len(scene.vehicle_actors),
                walkers=0 if scene is None else len(scene.walker_actors),
                props=0 if scene is None else len(scene.prop_actors),
            )
            try:
                result = super().configure(scene_id, raw)
            except BaseException as error:
                self._fail_preparation(error)
                raise
            self._complete_preparation()
            return result
        finally:
            self._prepare_gate.release()

    def _load_world_isolated(self, target: str) -> tuple[Any, Any]:
        self._set_preparation_stage("map")
        return super()._load_world_isolated(target)

    def _ensure_async_world(self, world: Any) -> None:
        self._set_preparation_stage("world_settings")
        return super()._ensure_async_world(world)

    def _spawn_ego(
        self,
        world: Any,
        config: Any,
        scene_id: str,
        spawn_points: list[Any],
        rng: Any,
        owned: list[Any],
    ) -> tuple[Any, int]:
        self._set_preparation_stage("ego")
        return super()._spawn_ego(world, config, scene_id, spawn_points, rng, owned)

    def _spawn_props(
        self,
        scene_id: str,
        world: Any,
        preset: str,
        ego_transform: Any,
        owned: list[Any],
    ) -> list[Any]:
        self._set_preparation_stage("props")
        actors = super()._spawn_props(scene_id, world, preset, ego_transform, owned)
        self._set_preparation_stage("props", props=len(actors))
        return actors

    def _spawn_traffic(
        self,
        scene_id: str,
        world: Any,
        traffic_manager: Any,
        spawn_points: list[Any],
        ego_spawn_index: int,
        count: int,
        rng: Any,
        owned: list[Any],
    ) -> list[Any]:
        self._set_preparation_stage("traffic")
        actors = super()._spawn_traffic(
            scene_id,
            world,
            traffic_manager,
            spawn_points,
            ego_spawn_index,
            count,
            rng,
            owned,
        )
        existing = 0
        if self._preparation_snapshot().get("operation") == "configure" and self._scene is not None:
            existing = len(self._scene.vehicle_actors)
        self._set_preparation_stage("traffic", traffic=existing + len(actors))
        return actors

    def _spawn_walkers(
        self,
        scene_id: str,
        world: Any,
        count: int,
        rng: Any,
        owned: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        self._set_preparation_stage("walkers")
        walkers, controllers = super()._spawn_walkers(scene_id, world, count, rng, owned)
        existing = 0
        if self._preparation_snapshot().get("operation") == "configure" and self._scene is not None:
            existing = len(self._scene.walker_actors)
        self._set_preparation_stage("walkers", walkers=existing + len(walkers))
        return walkers, controllers

    def _plan_random_route(
        self,
        world: Any,
        ego: Any,
        spawn_points: list[Any],
        spawn_index: int,
        rng: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
        self._set_preparation_stage("route")
        return super()._plan_random_route(world, ego, spawn_points, spawn_index, rng)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the existing authenticated server with ``ObservableWorldWorker``."""

    original = base.WorldWorker
    base.WorldWorker = ObservableWorldWorker
    try:
        return int(base.main(argv))
    finally:
        base.WorldWorker = original


__all__ = ["ObservableWorldWorker", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
