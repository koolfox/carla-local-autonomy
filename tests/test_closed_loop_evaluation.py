from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np

from carla_vision import closed_loop_cli, closed_loop_evaluation
from carla_vision.closed_loop_evaluation import (
    EvaluationEvents,
    RedLightMonitor,
    RouteProgressTracker,
)


def test_route_progress_is_monotonic() -> None:
    tracker = RouteProgressTracker(
        np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=np.float64)
    )
    assert tracker.update(11.0, 0.2) == 0.5
    assert tracker.update(1.0, 0.0) == 0.5
    assert tracker.update(20.0, 0.0) == 1.0
    assert tracker.route_length_m == 20.0


def test_red_light_monitor_distinguishes_stop_from_passage() -> None:
    monitor = RedLightMonitor(stop_speed_mps=0.3)
    assert not monitor.update(
        at_traffic_light=True,
        light_id=5,
        light_state="Red",
        speed_mps=3.0,
    )
    assert monitor.update(
        at_traffic_light=False,
        light_id=None,
        light_state=None,
        speed_mps=3.0,
    )
    assert monitor.violations == 1
    monitor.update(
        at_traffic_light=True,
        light_id=6,
        light_state="Red",
        speed_mps=0.1,
    )
    assert not monitor.update(
        at_traffic_light=False,
        light_id=None,
        light_state=None,
        speed_mps=1.0,
    )
    assert monitor.violations == 1


def test_event_accumulator_debounces_collisions_and_brakes() -> None:
    events = EvaluationEvents(collision_debounce_frames=5)
    impulse = SimpleNamespace(x=3.0, y=4.0, z=0.0)
    actor = SimpleNamespace(id=9)
    events.collision_callback(
        SimpleNamespace(frame=10, normal_impulse=impulse, other_actor=actor)
    )
    events.collision_callback(
        SimpleNamespace(frame=12, normal_impulse=impulse, other_actor=actor)
    )
    events.collision_callback(
        SimpleNamespace(frame=20, normal_impulse=impulse, other_actor=actor)
    )
    events.lane_invasion_callback(SimpleNamespace())
    assert events.observe_control(
        throttle=0.0,
        brake=1.0,
        speed_mps=4.0,
        threshold=0.95,
        minimum_speed_mps=1.0,
    )
    assert not events.observe_control(
        throttle=0.0,
        brake=1.0,
        speed_mps=3.0,
        threshold=0.95,
        minimum_speed_mps=1.0,
    )
    events.observe_control(
        throttle=0.2,
        brake=0.0,
        speed_mps=2.0,
        threshold=0.95,
        minimum_speed_mps=1.0,
    )
    assert events.observe_control(
        throttle=0.0,
        brake=1.0,
        speed_mps=2.0,
        threshold=0.95,
        minimum_speed_mps=1.0,
    )
    snapshot = events.snapshot()
    assert snapshot["collision_count"] == 2
    assert snapshot["collision_intensity_sum"] == 15.0
    assert snapshot["lane_invasion_count"] == 1
    assert snapshot["full_brake_intervention_count"] == 2


def test_sensor_adapter_uses_blueprint_then_transform() -> None:
    blueprint = object()
    transform = object()
    vehicle = object()

    class World:
        def get_blueprint_library(self):
            return SimpleNamespace(find=lambda identifier: blueprint)

        def spawn_actor(self, *args, **kwargs):
            assert args == (blueprint, transform)
            assert kwargs == {"attach_to": vehicle}
            return "sensor"

    carla = SimpleNamespace(Transform=lambda: transform)
    assert (
        closed_loop_cli._spawn_event_sensor(World(), carla, vehicle, "sensor.other.collision")
        == "sensor"
    )


def test_closed_loop_observer_has_no_vehicle_control_call() -> None:
    for module in (closed_loop_evaluation, closed_loop_cli):
        tree = ast.parse(inspect.getsource(module))
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "apply_control" not in attributes
