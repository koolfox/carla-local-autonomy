from pathlib import Path

root = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = root / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "carla_vision/native/world_worker.py",
    '''            if parsed.path == "/v1/scenes/prepare":
                result = self.server.worker.prepare(body)
                self._send_json(HTTPStatus.CREATED, result)
                return
            match = _SCENE_PATH.fullmatch(parsed.path)
''',
    '''            if parsed.path == "/v1/scenes/prepare":
                result = self.server.worker.prepare(body)
                self._send_json(HTTPStatus.CREATED, result)
                return
            if parsed.path == "/v1/scenes/preparation/cancel":
                operation = getattr(self.server.worker, "cancel_preparation", None)
                if not callable(operation):
                    raise WorkerError(
                        HTTPStatus.NOT_IMPLEMENTED,
                        "prepare_cancel_unavailable",
                        "this World Worker does not support scene preparation cancellation",
                    )
                result = operation(body)
                self._send_json(HTTPStatus.ACCEPTED, result)
                return
            match = _SCENE_PATH.fullmatch(parsed.path)
''',
)

replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''        self._prepare_gate = threading.Lock()
        self._preparation_lock = threading.RLock()
''',
    '''        self._prepare_gate = threading.Lock()
        self._preparation_lock = threading.RLock()
        self._prepare_cancel = threading.Event()
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''                "prepare_progress_stages": True,
''',
    '''                "prepare_progress_stages": True,
                "prepare_cancellation": True,
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    def _begin_preparation(self, config: Any) -> None:
        now = self._clock()
''',
    '''    def _begin_preparation(self, config: Any) -> None:
        self._prepare_cancel.clear()
        now = self._clock()
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''                "error": None,
                "scene_id": None,
''',
    '''                "error": None,
                "cancel_requested": False,
                "scene_id": None,
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    def _set_preparation_stage(self, stage: str, **updates: Any) -> None:
''',
    '''    def cancel_preparation(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        base._strict_keys(raw, allowed=set(), required=set(), name="prepare cancellation request")
        with self._preparation_lock:
            if self._preparation.get("status") != "preparing":
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_not_preparing",
                    "no scene preparation is currently active",
                )
            self._prepare_cancel.set()
            self._preparation["cancel_requested"] = True
        return {
            "schema_version": base.SCHEMA_VERSION,
            "worker_api_revision": base.WORKER_API_REVISION,
            "status": "cancelling",
            "preparation": self._preparation_snapshot(),
        }

    def _check_prepare_cancelled(self) -> None:
        if self._prepare_cancel.is_set():
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "scene_prepare_cancelled",
                "scene preparation was cancelled before the next CARLA mutation batch",
            )

    def _set_preparation_stage(self, stage: str, **updates: Any) -> None:
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    def _load_world_isolated(self, target: str) -> tuple[Any, Any]:
        self._set_preparation_stage("map")
        return super()._load_world_isolated(target)

    def _ensure_async_world(self, world: Any) -> None:
        self._set_preparation_stage("world_settings")
        return super()._ensure_async_world(world)
''',
    '''    def _load_world_isolated(self, target: str) -> tuple[Any, Any]:
        self._check_prepare_cancelled()
        self._set_preparation_stage("map")
        result = super()._load_world_isolated(target)
        self._check_prepare_cancelled()
        return result

    def _ensure_async_world(self, world: Any) -> None:
        self._check_prepare_cancelled()
        self._set_preparation_stage("world_settings")
        result = super()._ensure_async_world(world)
        self._check_prepare_cancelled()
        return result
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''        # Reaching ego spawn means the base Worker successfully applied
        # TrafficManager/world dynamics immediately before this call.
        self._set_preparation_stage(
''',
    '''        self._check_prepare_cancelled()
        # Reaching ego spawn means the base Worker successfully applied
        # TrafficManager/world dynamics immediately before this call.
        self._set_preparation_stage(
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''        return super()._spawn_ego(world, config, scene_id, spawn_points, rng, owned)

    def _spawn_owned_batch(
''',
    '''        result = super()._spawn_ego(world, config, scene_id, spawn_points, rng, owned)
        self._check_prepare_cancelled()
        return result

    def _spawn_owned_batch(
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    ) -> list[Any | None]:
        actors = super()._spawn_owned_batch(world, requests, owned)
        if self._is_preparing():
''',
    '''    ) -> list[Any | None]:
        self._check_prepare_cancelled()
        actors = super()._spawn_owned_batch(world, requests, owned)
        self._check_prepare_cancelled()
        if self._is_preparing():
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    ) -> list[Any]:
        self._set_preparation_stage("props")
''',
    '''    ) -> list[Any]:
        self._check_prepare_cancelled()
        self._set_preparation_stage("props")
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    ) -> list[Any]:
        self._set_preparation_stage("traffic")
        actors = super()._spawn_traffic(
''',
    '''    ) -> list[Any]:
        self._check_prepare_cancelled()
        self._set_preparation_stage("traffic")
        actors = super()._spawn_traffic(
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    ) -> tuple[list[Any], list[Any]]:
        self._set_preparation_stage("walkers")
''',
    '''    ) -> tuple[list[Any], list[Any]]:
        self._check_prepare_cancelled()
        self._set_preparation_stage("walkers")
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''    ) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
        self._set_preparation_stage("route")
''',
    '''    ) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
        self._check_prepare_cancelled()
        self._set_preparation_stage("route")
''',
)

replace_once(
    "carla_vision/operator/world_worker_client.py",
    '''    def current_scene(self) -> dict[str, Any]:
        return self._request("GET", "/v1/scenes/current")

''',
    '''    def current_scene(self) -> dict[str, Any]:
        return self._request("GET", "/v1/scenes/current")

    def cancel_preparation(self) -> dict[str, Any]:
        """Request cooperative cancellation between bounded CARLA mutation batches."""

        return self._request("POST", "/v1/scenes/preparation/cancel", {})

''',
)

replace_once(
    "carla_vision/operator/garage_preview.py",
    '''from .world_worker_client import (
    WorldWorkerCameraStream,
    WorldWorkerClient,
    WorldWorkerScene,
)
''',
    '''from .world_worker_client import (
    WorldWorkerCameraStream,
    WorldWorkerClient,
    WorldWorkerError,
    WorldWorkerScene,
)
''',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '''    def state(self) -> dict[str, Any]:
''',
    '''    def cancel_preparation(self) -> dict[str, Any] | None:
        """Cancel an in-flight Worker prepare without waiting on the scene lock."""

        worker = self.world_worker
        if worker is None:
            return None
        try:
            return worker.cancel_preparation()
        except WorldWorkerError as error:
            if error.code in {"scene_not_preparing", "prepare_cancel_unavailable"}:
                return None
            raise

    def state(self) -> dict[str, Any]:
''',
)

replace_once(
    "carla_vision/operator/garage_server.py",
    '''                else:
                    result = self.server.application.preview.stop(body)
                    status = HTTPStatus.OK
''',
    '''                else:
                    # If configure currently owns the scene mutation lock, ask
                    # the responsive Worker control plane to stop between CARLA
                    # mutation batches before waiting for normal cleanup.
                    self.server.application.preview.cancel_preparation()
                    result = self.server.application.preview.stop(body)
                    status = HTTPStatus.OK
''',
)

replace_once(
    "pyproject.toml",
    '''carla-garage-acceptance = "carla_vision.operator.garage_acceptance:main"
''',
    '''carla-garage-acceptance = "carla_vision.operator.garage_acceptance:main"
carla-garage-density-acceptance = "carla_vision.operator.garage_density_acceptance:main"
''',
)
