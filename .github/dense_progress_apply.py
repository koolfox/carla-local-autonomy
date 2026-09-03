from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"{path}: marker not found")
    target.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


# Worker progress: include dynamics and update counts after each official CARLA spawn batch.
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''        requested = {
            "map": config.map_name,
            "traffic": int(config.traffic_count),
            "walkers": int(config.walker_count),
            "prop_preset": config.prop_preset,
            "route_mode": config.route_mode,
        }
''',
    '''        requested = {
            "map": config.map_name,
            "traffic": int(config.traffic_count),
            "walkers": int(config.walker_count),
            "prop_preset": config.prop_preset,
            "route_mode": config.route_mode,
            "pedestrian_crossing_factor": float(config.pedestrian_crossing_factor),
            "speed_difference_percent": float(config.speed_difference_percent),
            "following_distance_metres": float(config.following_distance_metres),
        }
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''                "actual": {
                    "traffic": 0,
                    "walkers": 0,
                    "props": 0,
                },
''',
    '''                "actual": {
                    "traffic": 0,
                    "walkers": 0,
                    "props": 0,
                    "pedestrian_crossing_factor": None,
                    "speed_difference_percent": None,
                    "following_distance_metres": None,
                },
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''            for name, value in updates.items():
                if name in {"traffic", "walkers", "props"}:
                    self._preparation.setdefault("actual", {})[name] = int(value)
                else:
                    self._preparation[name] = value
''',
    '''            for name, value in updates.items():
                if name in {"traffic", "walkers", "props"}:
                    self._preparation.setdefault("actual", {})[name] = int(value)
                elif name in {
                    "pedestrian_crossing_factor",
                    "speed_difference_percent",
                    "following_distance_metres",
                }:
                    self._preparation.setdefault("actual", {})[name] = float(value)
                else:
                    self._preparation[name] = value
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''            actual = {
                "traffic": 0 if scene is None else len(scene.vehicle_actors),
                "walkers": 0 if scene is None else len(scene.walker_actors),
                "props": 0 if scene is None else len(scene.prop_actors),
            }
''',
    '''            actual = {
                "traffic": 0 if scene is None else len(scene.vehicle_actors),
                "walkers": 0 if scene is None else len(scene.walker_actors),
                "props": 0 if scene is None else len(scene.prop_actors),
                "pedestrian_crossing_factor": (
                    None if scene is None else float(scene.config.pedestrian_crossing_factor)
                ),
                "speed_difference_percent": (
                    None if scene is None else float(scene.config.speed_difference_percent)
                ),
                "following_distance_metres": (
                    None if scene is None else float(scene.config.following_distance_metres)
                ),
            }
''',
)
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '''        self._set_preparation_stage("ego")
        return super()._spawn_ego(world, config, scene_id, spawn_points, rng, owned)

    def _spawn_props(
''',
    '''        # Reaching ego spawn means the base Worker successfully applied
        # TrafficManager/world dynamics immediately before this call.
        self._set_preparation_stage(
            "ego",
            pedestrian_crossing_factor=config.pedestrian_crossing_factor,
            speed_difference_percent=config.speed_difference_percent,
            following_distance_metres=config.following_distance_metres,
        )
        return super()._spawn_ego(world, config, scene_id, spawn_points, rng, owned)

    def _spawn_owned_batch(
        self,
        world: Any,
        requests: Sequence[Any],
        owned: list[Any],
    ) -> list[Any | None]:
        actors = super()._spawn_owned_batch(world, requests, owned)
        if self._is_preparing():
            stage = str(self._preparation_snapshot().get("stage", ""))
            if stage in {"traffic", "walkers", "props"}:
                self._set_preparation_stage(
                    stage,
                    traffic=sum(item.kind == "traffic" for item in owned),
                    walkers=sum(item.kind == "walker" for item in owned),
                    props=sum(item.kind == "prop" for item in owned),
                )
        return actors

    def _spawn_props(
''',
)

# Preview manager exposes the existing responsive Worker preparation snapshot.
replace_once(
    "carla_vision/operator/garage_preview.py",
    '''        self._session: GaragePreviewSession | None = None
        self._last_error: str | None = None

    def state(self) -> dict[str, Any]:
''',
    '''        self._session: GaragePreviewSession | None = None
        self._last_error: str | None = None

    def preparation_status(self) -> dict[str, Any] | None:
        """Read Worker preparation progress without taking the Garage scene lock."""

        worker = self.world_worker
        if worker is None:
            return None
        try:
            payload = worker.current_scene()
        except Exception:
            # Progress telemetry must never become a second failure path. The
            # authoritative configure call will surface transport/CARLA errors.
            return None
        preparation = payload.get("preparation")
        if not isinstance(preparation, Mapping):
            return None
        return dict(preparation)

    def state(self) -> dict[str, Any]:
''',
)

# SSE lifecycle: monitor Worker progress while the authoritative configure call blocks.
replace_once(
    "carla_vision/operator/garage_async.py",
    '''    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def snapshot(self, *, clock: Any = time.monotonic) -> dict[str, Any]:
''',
    '''    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    preparation: dict[str, Any] | None = None

    def snapshot(self, *, clock: Any = time.monotonic) -> dict[str, Any]:
''',
)
replace_once(
    "carla_vision/operator/garage_async.py",
    '''        if self.result is not None:
            payload["result"] = copy.deepcopy(self.result)
        return payload
''',
    '''        if self.preparation is not None:
            payload["preparation"] = copy.deepcopy(self.preparation)
        if self.result is not None:
            payload["result"] = copy.deepcopy(self.result)
        return payload
''',
)
replace_once(
    "carla_vision/operator/garage_async.py",
    '''        self._active_operation_id: str | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
''',
    '''        self._active_operation_id: str | None = None
        self._thread: threading.Thread | None = None
        self._progress_thread: threading.Thread | None = None
        provider = getattr(manager, "preparation_status", None)
        self._progress_provider = provider if callable(provider) else None
        self._progress_interval = 0.25
        self._closed = False
''',
)
replace_once(
    "carla_vision/operator/garage_async.py",
    '''            accepted = operation.snapshot(clock=self._clock)
            self._thread = thread
            thread.start()
            return accepted
''',
    '''            monitor = threading.Thread(
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
''',
)
replace_once(
    "carla_vision/operator/garage_async.py",
    '''    def _run(self, operation_id: str, raw: Mapping[str, Any]) -> None:
        self._update(operation_id, stage="configuring")
''',
    '''    def _monitor_progress(self, operation_id: str) -> None:
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
''',
)

# Typed frontend progress projection and pure formatting helpers.
replace_once(
    "web/src/lib/domain/garagePreview.ts",
    '''import type { SessionConfig } from './config';

/** Keep this projection aligned with build_garage_preview_request on the Operator. */
''',
    '''import type { SessionConfig } from './config';

export interface GaragePreparationProgress {
  status: string;
  stage: string;
  requested: Record<string, unknown>;
  actual: Record<string, unknown>;
  history?: Array<Record<string, unknown>>;
  elapsed_seconds: number;
  error?: Record<string, unknown> | null;
}

function progressNumber(record: Record<string, unknown>, key: string): number | null {
  const value = Number(record[key]);
  return Number.isFinite(value) ? value : null;
}

export function garagePreparationStageLabel(stage: string): string {
  const labels: Record<string, string> = {
    accepted: 'Starting Garage…',
    configuring: 'Preparing CARLA…',
    map: 'Loading CARLA map…',
    world_settings: 'Applying world settings…',
    ego: 'Spawning ego vehicle…',
    props: 'Spawning scene props…',
    traffic: 'Spawning traffic…',
    walkers: 'Spawning walkers…',
    route: 'Planning route…',
    ready: 'Scene ready',
    running: 'Live CARLA',
    failed: 'Preparation failed'
  };
  return labels[stage] ?? 'Applying Garage settings…';
}

export function garagePreparationSummary(progress: GaragePreparationProgress): string {
  const requestedTraffic = progressNumber(progress.requested, 'traffic');
  const requestedWalkers = progressNumber(progress.requested, 'walkers');
  const actualTraffic = progressNumber(progress.actual, 'traffic');
  const actualWalkers = progressNumber(progress.actual, 'walkers');
  const crossing = progressNumber(progress.actual, 'pedestrian_crossing_factor')
    ?? progressNumber(progress.requested, 'pedestrian_crossing_factor');
  const pieces = [garagePreparationStageLabel(progress.stage)];
  if (requestedTraffic !== null && actualTraffic !== null) {
    pieces.push(`${actualTraffic}/${requestedTraffic} cars`);
  }
  if (requestedWalkers !== null && actualWalkers !== null) {
    pieces.push(`${actualWalkers}/${requestedWalkers} walkers`);
  }
  if (crossing !== null) pieces.push(`crossing ${crossing.toFixed(2)}`);
  const elapsed = Number(progress.elapsed_seconds);
  if (Number.isFinite(elapsed)) pieces.push(`${elapsed.toFixed(1)} s`);
  return pieces.join(' · ');
}

/** Keep this projection aligned with build_garage_preview_request on the Operator. */
''',
)
replace_once(
    "web/src/lib/api/operator.ts",
    '''import type {
  DriveControlRequest,
  DriveState,
  GarageOrbitRequest
} from '$lib/domain/runtime';
''',
    '''import type { GaragePreparationProgress } from '$lib/domain/garagePreview';
import type {
  DriveControlRequest,
  DriveState,
  GarageOrbitRequest
} from '$lib/domain/runtime';
''',
)
replace_once(
    "web/src/lib/api/operator.ts",
    '''  error?: { type?: string; message?: string } | null;
  result?: Record<string, unknown>;
  resolved_config?: Record<string, unknown>;
''',
    '''  error?: { type?: string; message?: string } | null;
  preparation?: GaragePreparationProgress | null;
  result?: Record<string, unknown>;
  resolved_config?: Record<string, unknown>;
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''  import {
    createGarageApplyQueue,
    garagePreviewInputError,
    garagePreviewSignature
  } from '$lib/domain/garagePreview';
''',
    '''  import {
    createGarageApplyQueue,
    garagePreparationStageLabel,
    garagePreparationSummary,
    garagePreviewInputError,
    garagePreviewSignature,
    type GaragePreparationProgress
  } from '$lib/domain/garagePreview';
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''  function lifecycleLabel(stage: string): string {
    if (stage === 'accepted') return 'Starting Garage…';
    if (stage === 'configuring') return 'Preparing CARLA…';
    if (stage === 'running') return 'Live CARLA';
    return 'Applying…';
  }

''',
    '',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''  let configureRetrying = false;
  let lifecycleStage = '';
  let previewEvidence: ConfigurationEvidence | null = null;
''',
    '''  let configureRetrying = false;
  let lifecycleStage = '';
  let preparation: GaragePreparationProgress | null = null;
  let previewEvidence: ConfigurationEvidence | null = null;
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''  function updateLifecycle(operation: GaragePreviewOperation): void {
    if (!destroyed) lifecycleStage = operation.stage;
  }
''',
    '''  function updateLifecycle(operation: GaragePreviewOperation): void {
    if (destroyed) return;
    lifecycleStage = operation.preparation?.stage ?? operation.stage;
    if (operation.preparation) preparation = operation.preparation;
  }
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''    busy: (value) => {
      busy = value;
      if (!value) lifecycleStage = '';
    },
''',
    '''    busy: (value) => {
      busy = value;
      if (!value) {
        lifecycleStage = '';
        preparation = null;
      }
    },
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''      lifecycleStage = '';
      configureError = caught instanceof Error ? caught.message : String(caught);
''',
    '''      lifecycleStage = '';
      preparation = null;
      configureError = caught instanceof Error ? caught.message : String(caught);
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''      lifecycleStage = '';
      configureError = '';
      previewEvidence = response.configuration;
''',
    '''      lifecycleStage = '';
      preparation = null;
      configureError = '';
      previewEvidence = response.configuration;
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''  $: streamSource = active && !driveActive
    ? `/api/garage/preview/stream.mjpg?t=${streamNonce}`
    : '';
''',
    '''  $: preparationText = preparation ? garagePreparationSummary(preparation) : '';
  $: streamSource = active && !driveActive
    ? `/api/garage/preview/stream.mjpg?t=${streamNonce}`
    : '';
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''    lifecycleStage = '';
    configureError = '';
    streamError = '';
''',
    '''    lifecycleStage = '';
    preparation = null;
    configureError = '';
    streamError = '';
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''          <strong>{lifecycleLabel(lifecycleStage)}</strong>
''',
    '''          <strong>{garagePreparationStageLabel(lifecycleStage)}</strong>
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''        <strong>{busy ? lifecycleLabel(lifecycleStage) : 'Waiting for CARLA camera…'}</strong>
''',
    '''        <strong>{busy ? garagePreparationStageLabel(lifecycleStage) : 'Waiting for CARLA camera…'}</strong>
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''        {#if driveActive}Drive active{:else if busy}{lifecycleLabel(lifecycleStage)}{:else if configureRetrying || streamRetryTimer}Reconnecting…{:else if error}Needs attention{:else if dirty}Syncing settings…{:else if active}Live CARLA{:else}Waiting for bridge{/if}
''',
    '''        {#if driveActive}Drive active{:else if busy}{garagePreparationStageLabel(lifecycleStage)}{:else if configureRetrying || streamRetryTimer}Reconnecting…{:else if error}Needs attention{:else if dirty}Syncing settings…{:else if active}Live CARLA{:else}Waiting for bridge{/if}
''',
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '''  {#if !$systemSettings?.workerConnected && !active}
''',
    '''  {#if busy && preparationText}
    <p class="garage-preview-note" aria-live="polite">{preparationText}</p>
  {/if}
  {#if !$systemSettings?.workerConnected && !active}
''',
)

# Backend regression: progress reaches the existing lifecycle stream.
append_once(
    "tests/test_garage_async.py",
    "\ndef test_async_facade_rejects_second_active_start_without_queueing() -> None:\n",
    '''\ndef test_async_facade_pushes_worker_population_progress() -> None:\n    class ProgressManager(_BlockingPreviewManager):\n        def __init__(self) -> None:\n            super().__init__()\n            self.progress: dict[str, Any] | None = None\n\n        def preparation_status(self) -> dict[str, Any] | None:\n            return None if self.progress is None else dict(self.progress)\n\n    manager = ProgressManager()\n    facade = GaragePreviewAsyncFacade(manager)\n    accepted = facade.start({"traffic_count": 64, "walker_count": 40})\n    assert manager.entered.wait(1.0)\n    manager.progress = {\n        "status": "preparing",\n        "stage": "traffic",\n        "requested": {\n            "traffic": 64,\n            "walkers": 40,\n            "pedestrian_crossing_factor": 0.45,\n        },\n        "actual": {\n            "traffic": 32,\n            "walkers": 0,\n            "pedestrian_crossing_factor": 0.45,\n        },\n        "history": [],\n        "elapsed_seconds": 1.2,\n        "error": None,\n    }\n\n    revision = accepted["revision"]\n    progress: dict[str, Any] | None = None\n    deadline = time.monotonic() + 2.0\n    while time.monotonic() < deadline:\n        update = facade.wait_for_update(accepted["operation_id"], revision, timeout=0.5)\n        if update is None:\n            continue\n        revision = int(update["revision"])\n        if update.get("preparation"):\n            progress = update\n            break\n    assert progress is not None\n    assert progress["stage"] == "traffic"\n    assert progress["preparation"]["actual"]["traffic"] == 32\n    assert progress["preparation"]["requested"]["walkers"] == 40\n    assert progress["preparation"]["actual"]["pedestrian_crossing_factor"] == 0.45\n\n    manager.release.set()\n\n''',
)

# Worker regression: requested/applied dynamics and batch counts are factual.
replace_once(
    "tests/test_observable_world_worker.py",
    '''    assert health["preparation"]["requested"]["traffic"] == 40
    assert health["preparation"]["requested"]["walkers"] == 30
''',
    '''    assert health["preparation"]["requested"]["traffic"] == 40
    assert health["preparation"]["requested"]["walkers"] == 30
    assert health["preparation"]["requested"]["pedestrian_crossing_factor"] == 0.2
    assert health["preparation"]["requested"]["speed_difference_percent"] == 12.0
    assert health["preparation"]["requested"]["following_distance_metres"] == 2.0
''',
)
append_once(
    "tests/test_observable_world_worker.py",
    "\ndef test_direct_file_entrypoint_keeps_single_host_script_workflow() -> None:\n",
    '''\ndef test_spawn_batch_hook_reports_incremental_population() -> None:\n    worker = observable.ObservableWorldWorker(start_monitor=False)\n    worker._begin_preparation(observable.SceneConfig(traffic_count=64, walker_count=40))\n    worker._set_preparation_stage("traffic")\n\n    class Item:\n        def __init__(self, kind: str) -> None:\n            self.kind = kind\n\n    owned = [Item("traffic") for _ in range(32)]\n    original = observable.base.WorldWorker._spawn_owned_batch\n    try:\n        observable.base.WorldWorker._spawn_owned_batch = lambda self, world, requests, owned: []\n        worker._spawn_owned_batch(None, [], owned)\n    finally:\n        observable.base.WorldWorker._spawn_owned_batch = original\n\n    assert worker._preparation_snapshot()["actual"]["traffic"] == 32\n\n\ndef test_ego_stage_confirms_applied_traffic_dynamics(monkeypatch: pytest.MonkeyPatch) -> None:\n    worker = observable.ObservableWorldWorker(start_monitor=False)\n    config = observable.SceneConfig(\n        pedestrian_crossing_factor=0.65,\n        speed_difference_percent=-15.0,\n        following_distance_metres=3.5,\n    )\n    worker._begin_preparation(config)\n    monkeypatch.setattr(\n        observable.base.WorldWorker,\n        "_spawn_ego",\n        lambda self, *args, **kwargs: (object(), 4),\n    )\n    worker._spawn_ego(None, config, "scene", [], None, [])\n    actual = worker._preparation_snapshot()["actual"]\n    assert actual["pedestrian_crossing_factor"] == 0.65\n    assert actual["speed_difference_percent"] == -15.0\n    assert actual["following_distance_metres"] == 3.5\n\n''',
)

# Frontend pure formatter tests.
replace_once(
    "web/tests/garage-preview.test.mjs",
    '''  createGarageApplyQueue,
  garagePreviewInputError,
  garagePreviewSignature,
  isTransientGarageError
''',
    '''  createGarageApplyQueue,
  garagePreparationStageLabel,
  garagePreparationSummary,
  garagePreviewInputError,
  garagePreviewSignature,
  isTransientGarageError
''',
)
append_once(
    "web/tests/garage-preview.test.mjs",
    "\ntest('signature includes only the effective parked preview contract', () => {\n",
    '''\ntest('dense preparation progress is factual and human-readable', () => {\n  const progress = {\n    status: 'preparing',\n    stage: 'walkers',\n    requested: { traffic: 64, walkers: 40, pedestrian_crossing_factor: 0.45 },\n    actual: { traffic: 64, walkers: 24, pedestrian_crossing_factor: 0.45 },\n    elapsed_seconds: 7.25\n  };\n  assert.equal(garagePreparationStageLabel('walkers'), 'Spawning walkers…');\n  assert.equal(\n    garagePreparationSummary(progress),\n    'Spawning walkers… · 64/64 cars · 24/40 walkers · crossing 0.45 · 7.3 s'\n  );\n});\n\n''',
)
