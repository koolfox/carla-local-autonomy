"""CLI adapter for the read-only closed-loop evaluator.

CARLA's ``World.spawn_actor`` expects ``(blueprint, transform, ...)``. Keeping
this tiny adapter separate also makes that native call directly unit-testable
without importing CARLA in the pure metrics module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import closed_loop_evaluation as implementation


def _spawn_event_sensor(world: Any, carla: Any, vehicle: Any, blueprint_id: str) -> Any:
    blueprint = world.get_blueprint_library().find(blueprint_id)
    if blueprint is None:
        raise RuntimeError(f"sensor blueprint is unavailable: {blueprint_id}")
    return world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)


def run(args: Any) -> dict[str, Any]:
    implementation._spawn_event_sensor = _spawn_event_sensor
    return implementation.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = implementation.build_parser().parse_args(argv)
    run(args)
    return 0


__all__ = ["main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
