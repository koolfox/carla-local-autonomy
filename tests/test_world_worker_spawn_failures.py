from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from test_world_worker_population import BatchClient, PopulationHarness, make_population_harness

from carla_vision.native.world_worker import WorkerError


class DestroyCommand:
    def __init__(self, actor_id: int) -> None:
        self.actor_id = actor_id


def install_command_cleanup(harness: PopulationHarness) -> list[int]:
    """Destroy by authoritative server ID even while the client snapshot lags."""

    client = harness.client
    assert isinstance(client, BatchClient)
    spawn_batch = client.apply_batch_sync
    destroyed: list[int] = []

    def apply_batch(commands: list[Any], do_tick: bool = False) -> list[Any]:
        assert do_tick is False
        if isinstance(commands[0], DestroyCommand):
            responses = []
            for command in commands:
                actor = harness.world.actors.get(command.actor_id)
                if actor is not None:
                    actor.destroy()
                destroyed.append(command.actor_id)
                responses.append(SimpleNamespace(actor_id=command.actor_id, error=""))
            return responses
        return spawn_batch(commands, do_tick)

    harness.carla.command.DestroyActor = DestroyCommand
    client.apply_batch_sync = apply_batch  # type: ignore[method-assign]
    return destroyed


@pytest.mark.parametrize("barrier_raises", [True, False])
def test_unobserved_success_ids_abort_without_replacement_and_are_destroyed(
    barrier_raises: bool,
) -> None:
    harness = make_population_harness()
    harness.world.delay_batch_visibility = True
    destroyed = install_command_cleanup(harness)

    def unavailable_snapshot(seconds: float) -> None:
        if barrier_raises:
            raise TimeoutError("snapshot unavailable")
        # Simulate a natural tick that still did not deliver the new actors.

    harness.world.wait_for_tick = unavailable_snapshot  # type: ignore[method-assign]
    try:
        with pytest.raises(WorkerError) as caught:
            harness.worker.prepare({"traffic_count": 4})
        assert caught.value.code == "scene_prepare_failed"
        client = harness.client
        assert isinstance(client, BatchClient)
        assert client.attempts["traffic"] == 4
        assert len(client.batches) == 1
        assert len(destroyed) == 5  # Four acknowledged spawn IDs and the ego.
        assert harness.world.actors == {}
        assert harness.world.serial_spawn_calls == 1
    finally:
        harness.worker.close()


def test_unconfirmed_cleanup_is_exposed_in_prepare_error() -> None:
    harness = make_population_harness()
    world = harness.world
    world.delay_batch_visibility = True
    # No DestroyActor capability in this compatibility fixture, and no
    # snapshot proxy: the Worker cannot honestly claim it removed these IDs.
    world.wait_for_tick = lambda seconds: None  # type: ignore[method-assign]
    try:
        with pytest.raises(WorkerError, match="Cleanup could not be fully confirmed") as caught:
            harness.worker.prepare({"traffic_count": 2})
        assert "command cleanup is required" in caught.value.message
        assert len(world.actors) == 2
        client = harness.client
        assert isinstance(client, BatchClient)
        assert client.attempts["traffic"] == 2
    finally:
        for actor in list(world.actors.values()):
            actor.destroy()
        harness.worker.close()


@pytest.mark.parametrize("kind", ["traffic", "walker", "controller"])
@pytest.mark.parametrize("failure", ["timeout", "missing", "invalid", "duplicate"])
def test_uncertain_response_is_not_replayed_and_only_owned_actors_are_recovered(
    kind: str,
    failure: str,
) -> None:
    harness = make_population_harness()
    world = harness.world
    client = harness.client
    assert isinstance(client, BatchClient)
    destroyed = install_command_cleanup(harness)
    original_spawn = world.try_spawn_actor
    original_lookup = world.get_actors
    apply_batch = client.apply_batch_sync

    def spawn_with_parent(blueprint: Any, transform: Any, attach_to: Any = None) -> Any:
        actor = original_spawn(blueprint, transform, attach_to)
        actor.parent = attach_to
        return actor

    def lookup_all(ids: list[int] | None = None) -> list[Any]:
        return original_lookup(list(world.actors) if ids is None else ids)

    world.try_spawn_actor = spawn_with_parent  # type: ignore[method-assign]
    world.get_actors = lookup_all  # type: ignore[method-assign]
    foreign_blueprint = world.library.find("vehicle.audi.tt")
    foreign_blueprint.set_attribute("role_name", "another_operator")
    foreign = world.try_spawn_actor(foreign_blueprint, world.map.spawn_points[0])

    def uncertain_batch(commands: list[Any], do_tick: bool = False) -> list[Any]:
        responses = apply_batch(commands, do_tick)
        if isinstance(commands[0], DestroyCommand) or commands[0].kind != kind:
            return responses
        if failure == "timeout":
            raise TimeoutError("response lost after the server executed the batch")
        if failure == "missing":
            return responses[:-1]
        if failure == "invalid":
            responses[-1].actor_id = 0
        else:
            responses[-1].actor_id = responses[0].actor_id
        return responses

    client.apply_batch_sync = uncertain_batch  # type: ignore[method-assign]
    try:
        with pytest.raises(WorkerError) as caught:
            harness.worker.prepare({"traffic_count": 3, "walker_count": 3})
        assert caught.value.code == "scene_prepare_failed"
        assert client.attempts[kind] == 3
        assert len(client.batches_for(kind)) == 1
        assert world.actors == {foreign.id: foreign}
        assert foreign.id not in destroyed
        assert world.serial_spawn_calls == 2  # Foreign setup and the ego, no replay.
    finally:
        harness.worker.close()
