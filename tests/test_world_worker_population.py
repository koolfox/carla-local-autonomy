from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from test_world_worker import (
    FakeActor,
    FakeAttribute,
    FakeBlueprint,
    FakeBlueprintLibrary,
    FakeCarla,
    FakeClient,
    FakeClock,
    FakeLocation,
    FakePlanner,
    FakeTrafficManager,
    FakeTransform,
    FakeWorld,
)

from carla_vision.native.world_worker import WorkerError, WorldWorker


def population_kind(blueprint_id: str) -> str:
    if blueprint_id.startswith("vehicle."):
        return "traffic"
    if blueprint_id.startswith("walker.pedestrian."):
        return "walker"
    assert blueprint_id == "controller.ai.walker"
    return "controller"


class BatchSpawnActor:
    """Match CARLA command value semantics, not mutable blueprint references."""

    def __init__(
        self,
        blueprint: FakeBlueprint,
        transform: FakeTransform,
        parent: int | FakeActor | None = None,
    ) -> None:
        self.blueprint = copy.deepcopy(blueprint)
        self.transform = copy.deepcopy(transform)
        self.parent_id = None if parent is None else int(getattr(parent, "id", parent))

    @property
    def kind(self) -> str:
        return population_kind(self.blueprint.id)


class PopulationWorld(FakeWorld):
    """Keep authoritative actors separate from the optional delayed snapshot."""

    def __init__(self, library: FakeBlueprintLibrary) -> None:
        super().__init__("Town10HD_Opt", library)
        self.map.spawn_points = [
            FakeTransform(FakeLocation(float(index * 8), float(index % 7) * 3.0, 0.5))
            for index in range(130)
        ]
        self.spawning_batch = False
        self.delay_batch_visibility = False
        self.pending_snapshot: set[int] = set()
        self.spawned: list[FakeActor] = []
        self.serial_spawn_calls = 0
        self.peak_alive: Counter[str] = Counter()
        self.activation_failures: Counter[str] = Counter()
        self.autopilot_failures = 0
        self.early_start_attempts = 0
        self.events: list[tuple[str, int]] = []
        self.navigation_calls = 0

    def try_spawn_actor(
        self,
        blueprint: FakeBlueprint,
        transform: FakeTransform,
        attach_to: FakeActor | None = None,
    ) -> FakeActor:
        if not self.spawning_batch:
            self.serial_spawn_calls += 1
        actor = super().try_spawn_actor(blueprint, transform, attach_to)
        actor.parent_id = None if attach_to is None else attach_to.id
        self.spawned.append(actor)
        self.events.append(("spawn", actor.id))
        if self.spawning_batch and self.delay_batch_visibility:
            self.pending_snapshot.add(actor.id)
        alive = Counter(population_kind(item.type_id) for item in self.actors.values())
        for kind, count in alive.items():
            self.peak_alive[kind] = max(self.peak_alive[kind], count)

        def destroy() -> None:
            self.events.append(("destroy", actor.id))
            self.pending_snapshot.discard(actor.id)
            FakeActor.destroy(actor)

        def activate(phase: str) -> None:
            self.events.append((phase, actor.id))
            if phase == "start" and (
                actor.id in self.pending_snapshot or actor.parent_id in self.pending_snapshot
            ):
                self.early_start_attempts += 1
                raise RuntimeError("Actor could not be found in the registry before natural tick")
            if self.activation_failures[phase] > 0:
                self.activation_failures[phase] -= 1
                raise RuntimeError(f"injected controller {phase} failure")

        def start() -> None:
            activate("start")
            FakeActor.start(actor)

        def go_to_location(location: FakeLocation) -> None:
            activate("destination")
            FakeActor.go_to_location(actor, location)

        def set_max_speed(speed: float) -> None:
            activate("speed")
            FakeActor.set_max_speed(actor, speed)

        def set_autopilot(enabled: bool, port: int) -> None:
            if self.autopilot_failures > 0 and actor.attributes.get("role_name", "").startswith(
                "world_worker_npc_"
            ):
                self.autopilot_failures -= 1
                raise RuntimeError("injected autopilot activation failure")
            FakeActor.set_autopilot(actor, enabled, port)

        actor.destroy = destroy  # type: ignore[method-assign]
        actor.start = start  # type: ignore[method-assign]
        actor.go_to_location = go_to_location  # type: ignore[method-assign]
        actor.set_max_speed = set_max_speed  # type: ignore[method-assign]
        actor.set_autopilot = set_autopilot  # type: ignore[method-assign]
        return actor

    def get_actor(self, actor_id: int) -> FakeActor | None:
        if actor_id in self.pending_snapshot:
            return None
        return super().get_actor(actor_id)

    def get_actors(self, actor_ids: list[int]) -> list[FakeActor]:
        return [
            self.actors[actor_id]
            for actor_id in actor_ids
            if actor_id in self.actors and actor_id not in self.pending_snapshot
        ]

    def get_random_location_from_navigation(self) -> FakeLocation:
        self.navigation_calls += 1
        return FakeLocation(float(self.navigation_calls * 2), 7.0, 0.5)

    def wait_for_tick(self, seconds: float) -> None:
        self.events.append(("wait", self.wait_count + 1))
        self.pending_snapshot.clear()
        super().wait_for_tick(seconds)

    def live_population(self) -> Counter[str]:
        result: Counter[str] = Counter({"traffic": 0, "walker": 0, "controller": 0})
        result.update(population_kind(actor.type_id) for actor in self.actors.values())
        # The independently spawned ego is not part of requested NPC traffic.
        result["traffic"] -= sum(
            not actor.attributes.get("role_name", "").startswith("world_worker_npc_")
            for actor in self.actors.values()
            if actor.type_id.startswith("vehicle.")
        )
        return result


class BatchClient(FakeClient):
    def __init__(
        self,
        world: PopulationWorld,
        library: FakeBlueprintLibrary,
        traffic_manager: FakeTrafficManager,
    ) -> None:
        super().__init__(world, library, traffic_manager)
        self.batches: list[list[BatchSpawnActor]] = []
        self.tick_flags: list[bool] = []
        self.attempts: Counter[str] = Counter()
        self.failures: dict[str, set[int]] = {}
        self.always_fail: set[str] = set()

    def apply_batch_sync(self, commands: list[BatchSpawnActor], do_tick: bool = False) -> list[Any]:
        assert do_tick is False, "the worker must never tick the shared asynchronous world"
        assert commands, "empty batches are unnecessary CARLA round trips"
        assert all(isinstance(command, BatchSpawnActor) for command in commands)
        self.batches.append(list(commands))
        self.tick_flags.append(do_tick)
        world = self.world
        assert isinstance(world, PopulationWorld)
        responses: list[Any] = []
        for command in commands:
            self.attempts[command.kind] += 1
            failed = command.kind in self.always_fail or self.attempts[command.kind] in (
                self.failures.get(command.kind, set())
            )
            if failed:
                responses.append(SimpleNamespace(actor_id=0, error="spawn collision"))
                continue
            parent = None
            if command.parent_id is not None:
                # Attachment is a server operation and need not be in the client snapshot yet.
                parent = world.actors.get(command.parent_id)
                assert parent is not None, "controller must attach to its spawned walker ID"
                assert parent.type_id.startswith("walker.pedestrian.")
            world.spawning_batch = True
            try:
                actor = world.try_spawn_actor(command.blueprint, command.transform, parent)
            finally:
                world.spawning_batch = False
            responses.append(SimpleNamespace(actor_id=actor.id, error=""))
        return responses

    def batches_for(self, kind: str) -> list[list[BatchSpawnActor]]:
        return [batch for batch in self.batches if batch[0].kind == kind]


@dataclass
class PopulationHarness:
    world: PopulationWorld
    client: FakeClient
    carla: FakeCarla
    worker: WorldWorker
    traffic_manager: FakeTrafficManager


def make_population_harness(
    *, command_capability: bool = True, client_capability: bool = True
) -> PopulationHarness:
    library = FakeBlueprintLibrary()
    for index in (2, 3):
        blueprint = copy.deepcopy(library.find("walker.pedestrian.0001"))
        blueprint.id = f"walker.pedestrian.{index:04d}"
        library.blueprints[blueprint.id] = blueprint
    world = PopulationWorld(library)
    traffic_manager = FakeTrafficManager(8000)
    client_type = BatchClient if client_capability else FakeClient
    client = client_type(world, library, traffic_manager)
    carla = FakeCarla(client)
    if command_capability:
        carla.command = SimpleNamespace(SpawnActor=BatchSpawnActor)
    worker = WorldWorker(
        carla_loader=lambda: carla,
        route_planner_loader=lambda: lambda map_object: FakePlanner(map_object),
        clock=FakeClock(),
        start_monitor=False,
    )
    return PopulationHarness(world, client, carla, worker, traffic_manager)


@pytest.fixture
def population() -> Iterator[PopulationHarness]:
    harness = make_population_harness()
    try:
        yield harness
    finally:
        harness.worker.close()


def assert_exact_population(harness: PopulationHarness, traffic: int, walkers: int) -> None:
    assert harness.world.live_population() == {
        "traffic": traffic,
        "walker": walkers,
        "controller": walkers,
    }
    assert harness.world.peak_alive["traffic"] <= traffic + 1
    assert harness.world.peak_alive["walker"] <= walkers
    assert harness.world.peak_alive["controller"] <= walkers
    for actor in harness.world.actors.values():
        if actor.type_id == "controller.ai.walker":
            assert actor.started
            assert actor.destination is not None
            assert actor.maximum_speed in (1.4, 2.8)
            assert actor.parent_id in harness.world.actors
        elif actor.attributes.get("role_name", "").startswith("world_worker_npc_"):
            assert actor.autopilot
            assert actor.autopilot_port == 8000
            assert actor.id in harness.traffic_manager.lights


def test_spawn_command_snapshots_mutable_blueprint_and_transform() -> None:
    blueprint = FakeBlueprint("vehicle.audi.tt", {"color": FakeAttribute("red")})
    transform = FakeTransform(FakeLocation(5.0, 6.0, 7.0))
    command = BatchSpawnActor(blueprint, transform, 123)
    blueprint.set_attribute("color", "blue")
    transform.location.x = 99.0
    assert str(command.blueprint.get_attribute("color")) == "red"
    assert command.transform.location.x == 5.0
    assert command.parent_id == 123


def test_dense_prepare_uses_19_spawn_round_trips_for_454_population_actors(
    population: PopulationHarness,
) -> None:
    prepared = population.worker.prepare({"traffic_count": 88, "walker_count": 183, "seed": 41})
    scene = prepared["scene"]
    assert scene["traffic_count_requested"] == scene["traffic_count"] == 88
    assert scene["walker_count_requested"] == scene["walker_count"] == 183
    client = population.client
    assert isinstance(client, BatchClient)
    assert [len(batch) for batch in client.batches_for("traffic")] == [32, 32, 24]
    assert [len(batch) for batch in client.batches_for("walker")] == [24] * 7 + [15]
    assert [len(batch) for batch in client.batches_for("controller")] == [24] * 7 + [15]
    assert len(client.batches) == 3 + 16
    assert sum(map(len, client.batches)) == 88 + 2 * 183 == 454
    assert client.tick_flags == [False] * 19
    assert population.world.serial_spawn_calls == 1  # The ego retains its serial retry path.
    assert_exact_population(population, 88, 183)


def test_maximum_configured_population_is_not_clamped(population: PopulationHarness) -> None:
    population.world.map.spawn_points = [
        FakeTransform(FakeLocation(float(index * 8), 0.0, 0.5)) for index in range(260)
    ]
    prepared = population.worker.prepare({"traffic_count": 250, "walker_count": 250})
    assert prepared["scene"]["traffic_count"] == 250
    assert prepared["scene"]["walker_count"] == 250
    assert_exact_population(population, 250, 250)
    client = population.client
    assert isinstance(client, BatchClient)
    assert len(client.batches) == 8 + 11 + 11


@pytest.mark.parametrize("missing", ["command", "client", "both"])
def test_missing_batch_capability_retains_exact_serial_population(missing: str) -> None:
    harness = make_population_harness(
        command_capability=missing not in ("command", "both"),
        client_capability=missing not in ("client", "both"),
    )
    try:
        prepared = harness.worker.prepare({"traffic_count": 5, "walker_count": 7, "seed": 8})
        assert prepared["scene"]["traffic_count"] == 5
        assert prepared["scene"]["walker_count"] == 7
        assert harness.world.serial_spawn_calls == 1 + 5 + 2 * 7
        if isinstance(harness.client, BatchClient):
            assert harness.client.batches == []
        assert_exact_population(harness, 5, 7)
    finally:
        harness.worker.close()


@pytest.mark.parametrize("kind", ["traffic", "walker", "controller"])
def test_partial_spawn_errors_retry_only_the_deficit(
    population: PopulationHarness, kind: str
) -> None:
    client = population.client
    assert isinstance(client, BatchClient)
    client.failures[kind] = {1, 3}
    prepared = population.worker.prepare({"traffic_count": 35, "walker_count": 29, "seed": 17})
    assert prepared["scene"]["traffic_count"] == 35
    assert prepared["scene"]["walker_count"] == 29
    assert_exact_population(population, 35, 29)
    assert client.attempts["traffic"] == 35 + (2 if kind == "traffic" else 0)
    assert client.attempts["walker"] == 29 + (2 if kind in ("walker", "controller") else 0)
    assert client.attempts["controller"] == 29 + (2 if kind == "controller" else 0)
    assert all(len(batch) <= (32 if batch[0].kind == "traffic" else 24) for batch in client.batches)
    if kind == "controller":
        discarded = [actor for actor in population.world.spawned if actor.destroyed]
        assert len(discarded) == 2
        assert all(actor.type_id.startswith("walker.pedestrian.") for actor in discarded)


def test_vehicle_activation_failure_is_rolled_back_before_replacement(
    population: PopulationHarness,
) -> None:
    population.world.autopilot_failures = 2
    prepared = population.worker.prepare({"traffic_count": 35, "seed": 19})
    assert prepared["scene"]["traffic_count"] == 35
    client = population.client
    assert isinstance(client, BatchClient)
    assert client.attempts["traffic"] == 37
    assert len([actor for actor in population.world.spawned if actor.destroyed]) == 2
    assert_exact_population(population, 35, 0)


@pytest.mark.parametrize("phase", ["start", "destination", "speed"])
def test_controller_activation_failure_rolls_back_pair_and_retries(
    population: PopulationHarness, phase: str
) -> None:
    population.world.activation_failures[phase] = 2
    prepared = population.worker.prepare({"walker_count": 29, "seed": 23})
    assert prepared["scene"]["walker_count"] == 29
    client = population.client
    assert isinstance(client, BatchClient)
    assert client.attempts["walker"] == client.attempts["controller"] == 31
    discarded = [actor for actor in population.world.spawned if actor.destroyed]
    assert Counter(population_kind(actor.type_id) for actor in discarded) == {
        "walker": 2,
        "controller": 2,
    }
    for controller in discarded:
        if controller.type_id == "controller.ai.walker":
            assert controller.stopped
            assert population.world.events.index(("destroy", controller.id)) < (
                population.world.events.index(("destroy", controller.parent_id))
            )
    assert_exact_population(population, 0, 29)


@pytest.mark.parametrize(
    "failure", ["traffic_activation", "controller_creation", "controller_activation"]
)
@pytest.mark.parametrize("destroy_result", ["false", "exception"])
def test_unconfirmed_population_rollback_aborts_without_spawning_replacements(
    population: PopulationHarness,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    destroy_result: str,
) -> None:
    client = population.client
    assert isinstance(client, BatchClient)
    traffic = 2 if failure == "traffic_activation" else 0
    walkers = 0 if failure == "traffic_activation" else 2
    rejected_kind = {
        "traffic_activation": "traffic",
        "controller_creation": "walker",
        "controller_activation": "controller",
    }[failure]
    if failure == "traffic_activation":
        population.world.autopilot_failures = 1
    elif failure == "controller_creation":
        client.failures["controller"] = {1}
    else:
        population.world.activation_failures["start"] = 1

    original_destroy = FakeActor.destroy
    rejected_id: int | None = None
    destroy_attempts: Counter[int] = Counter()

    def flaky_destroy(actor: FakeActor) -> None | bool:
        nonlocal rejected_id
        if rejected_id is None and population_kind(actor.type_id) == rejected_kind:
            rejected_id = actor.id
        if actor.id == rejected_id:
            destroy_attempts[actor.id] += 1
            if destroy_attempts[actor.id] == 1:
                if destroy_result == "false":
                    return False
                raise RuntimeError("injected rollback destroy failure")
        original_destroy(actor)
        return None

    monkeypatch.setattr(FakeActor, "destroy", flaky_destroy)
    with pytest.raises(WorkerError, match="rollback was not confirmed") as raised:
        population.worker.prepare({"traffic_count": traffic, "walker_count": walkers})

    assert raised.value.code == "scene_prepare_failed"
    assert rejected_id is not None
    assert destroy_attempts[rejected_id] >= 2  # Final cleanup retries the retained ownership.
    assert population.worker.current_scene()["scene"] is None
    assert population.world.actors == {}
    assert client.attempts["traffic"] == traffic
    assert client.attempts["walker"] == walkers
    assert client.attempts["controller"] == walkers
    assert population.world.serial_spawn_calls == 1


def test_delayed_snapshot_is_visible_after_natural_tick_before_controller_start(
    population: PopulationHarness,
) -> None:
    population.world.delay_batch_visibility = True
    prepared = population.worker.prepare({"traffic_count": 4, "walker_count": 25, "seed": 29})
    assert prepared["scene"]["traffic_count"] == 4
    assert prepared["scene"]["walker_count"] == 25
    assert population.world.early_start_attempts == 0
    assert population.world.pending_snapshot == set()
    assert population.world.wait_count > 0
    client = population.client
    assert isinstance(client, BatchClient)
    assert client.attempts == {"traffic": 4, "walker": 25, "controller": 25}
    for actor in population.world.actors.values():
        if actor.type_id == "controller.ai.walker":
            spawned = population.world.events.index(("spawn", actor.id))
            started = population.world.events.index(("start", actor.id))
            assert any(event[0] == "wait" for event in population.world.events[spawned:started])
    assert_exact_population(population, 4, 25)


@pytest.mark.parametrize("kind", ["traffic", "walker", "controller"])
def test_persistent_known_spawn_errors_fail_exact_population_without_silent_clamping(
    population: PopulationHarness, kind: str
) -> None:
    client = population.client
    assert isinstance(client, BatchClient)
    client.always_fail.add(kind)
    with pytest.raises(WorkerError, match="exact requested population") as raised:
        population.worker.prepare({"traffic_count": 3, "walker_count": 3, "seed": 31})
    assert raised.value.code == "scene_population_shortfall"
    assert population.worker.current_scene()["scene"] is None
    assert population.world.actors == {}
    assert population.world.serial_spawn_calls == 1  # Known failures do not select serial fallback.
    assert 0 < client.attempts[kind] <= (129 * 3 if kind == "traffic" else 3 * 27)
    assert len(client.batches) < 150
    client.always_fail.clear()
    retried = population.worker.prepare({"traffic_count": 3, "walker_count": 3, "seed": 31})
    assert retried["scene"]["traffic_count"] == retried["scene"]["walker_count"] == 3
    assert_exact_population(population, 3, 3)


def test_late_walker_loss_is_replaced_without_rebuilding_scene(
    population: PopulationHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = population.worker._registered_actor_ids
    removed = []

    def lose_one(world: Any, actors: Any) -> set[int]:
        if not removed:
            walker = next((a for a in actors if a.type_id.startswith("walker.")), None)
            if walker is not None:
                removed.append(walker.id)
                walker.destroy()
        return original(world, actors)

    monkeypatch.setattr(population.worker, "_registered_actor_ids", lose_one)
    prepared = population.worker.prepare({"traffic_count": 2, "walker_count": 25})
    assert removed
    assert prepared["scene"]["walker_count"] == 25
    assert_exact_population(population, 2, 25)


def _candidate_signature(client: BatchClient) -> list[tuple[Any, ...]]:
    return [
        (
            command.blueprint.id,
            tuple(
                sorted(
                    (name, str(attribute))
                    for name, attribute in command.blueprint.attributes.items()
                    if name != "role_name"
                )
            ),
            command.transform.location.x,
            command.transform.location.y,
            command.transform.location.z,
            command.transform.rotation.yaw,
            command.parent_id is not None,
        )
        for batch in client.batches
        for command in batch
    ]


def test_same_seed_repeats_population_candidate_choices_with_blueprint_snapshots() -> None:
    signatures: list[list[tuple[Any, ...]]] = []
    for seed in (4096, 4096, 8192):
        harness = make_population_harness()
        try:
            harness.worker.prepare({"traffic_count": 35, "walker_count": 29, "seed": seed})
            assert isinstance(harness.client, BatchClient)
            signatures.append(_candidate_signature(harness.client))
            colors = {
                str(command.blueprint.get_attribute("color"))
                for batch in harness.client.batches_for("traffic")
                for command in batch
            }
            assert colors == {"255,0,0", "0,0,255"}
        finally:
            harness.worker.close()
    assert signatures[0] == signatures[1]
    assert signatures[0] != signatures[2]
