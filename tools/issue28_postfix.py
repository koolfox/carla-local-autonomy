from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "carla_vision/native/world_worker.py",
    '                "route_modes": [\n'
    '                    {"id": "free", "label": "Free Drive"},\n'
    '                    {"id": "random_destination", "label": "Random Destination"},\n'
    '                ],',
    '                "route_modes": [\n'
    '                    {"id": "free", "label": "Free Drive"},\n'
    '                    {"id": "random_destination", "label": "Random Destination"},\n'
    '                    {"id": "selected_destination", "label": "Selected Destination"},\n'
    '                ],',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                    raise RuntimeError("prepared random route cannot be enforced")',
    '                    raise RuntimeError("prepared planned route cannot be enforced")',
)

replace_once(
    "carla_vision/operator/garage_drive.py",
    '        self.destination_index: int | None = destination_index\n'
    '        self.route_generation = 0\n'
    '        self.route_complete = False\n'
    '        self._set_destination()',
    '        self.destination_index: int | None = destination_index\n'
    '        self.fixed_destination = destination_index is not None\n'
    '        self.route_generation = 0\n'
    '        self.route_complete = False\n'
    '        self._set_destination()',
)
replace_once(
    "carla_vision/operator/garage_drive.py",
    '        if self.agent.done():\n'
    '            self.route_complete = True\n'
    '            return (\n'
    '                ControlCommand.service_brake(),\n'
    '                "behavior_route_complete",\n'
    '                False,\n'
    '                {"destination_index": self.destination_index, "route_complete": True},\n'
    '            )\n'
    '        command = _command_from_carla(self.agent.run_step(debug=False))',
    '        if self.agent.done():\n'
    '            if self.fixed_destination:\n'
    '                self.route_complete = True\n'
    '                return (\n'
    '                    ControlCommand.service_brake(),\n'
    '                    "behavior_route_complete",\n'
    '                    False,\n'
    '                    {"destination_index": self.destination_index, "route_complete": True},\n'
    '                )\n'
    '            self.destination_index = None\n'
    '            self._set_destination()\n'
    '        command = _command_from_carla(self.agent.run_step(debug=False))',
)

Path("tests/test_garage_route_policy.py").write_text(
    '''from __future__ import annotations

import random
from types import SimpleNamespace

from carla_vision.operator import garage_drive


class _Location:
    def __init__(self, x: float) -> None:
        self.x = x

    def distance(self, other: _Location) -> float:
        return abs(self.x - other.x)


class _Ego:
    def __init__(self) -> None:
        self.location = _Location(0.0)

    def get_location(self) -> _Location:
        return self.location

    def get_transform(self) -> object:
        raise RuntimeError("not needed by this regression")


class _Agent:
    def __init__(self, *_: object, **__: object) -> None:
        self.destinations: list[object] = []
        self.finished = False

    def set_destination(self, location: object) -> None:
        self.destinations.append(location)

    def done(self) -> bool:
        return self.finished

    def run_step(self, *, debug: bool) -> object:
        assert debug is False
        return SimpleNamespace(
            throttle=0.25,
            steer=0.1,
            brake=0.0,
            hand_brake=False,
            reverse=False,
        )

    def get_local_planner(self) -> object:
        raise RuntimeError("navigation detail is outside this regression")


class _Context:
    def __init__(self) -> None:
        self.ego = _Ego()
        self.spawn_points = [
            SimpleNamespace(location=_Location(100.0)),
            SimpleNamespace(location=_Location(200.0)),
            SimpleNamespace(location=_Location(300.0)),
        ]
        self.rng = random.Random(7)
        self.world = object()


def _config() -> object:
    return SimpleNamespace(behavior="normal", target_speed_kmh=30.0)


def _install_agent(monkeypatch) -> None:
    original = garage_drive.importlib.import_module

    def load(name: str):
        if name == "agents.navigation.behavior_agent":
            return SimpleNamespace(BehaviorAgent=_Agent)
        return original(name)

    monkeypatch.setattr(garage_drive.importlib, "import_module", load)


def test_selected_behavior_destination_stops_at_the_declared_route_end(monkeypatch) -> None:
    _install_agent(monkeypatch)
    policy = garage_drive._BehaviorPolicy(_Context(), _config(), destination_index=1)
    assert policy.agent.destinations == [policy.context.spawn_points[1].location]
    policy.agent.finished = True

    command, source, failsafe, detail = policy.step()

    assert source == "behavior_route_complete"
    assert failsafe is False
    assert command.brake == 1.0
    assert detail == {"destination_index": 1, "route_complete": True}
    assert len(policy.agent.destinations) == 1


def test_free_behavior_mode_keeps_looping_after_an_internal_destination(monkeypatch) -> None:
    _install_agent(monkeypatch)
    policy = garage_drive._BehaviorPolicy(_Context(), _config())
    policy.agent.finished = True

    command, source, failsafe, detail = policy.step()

    assert source == "behavior_agent"
    assert failsafe is False
    assert command.throttle == 0.25
    assert len(policy.agent.destinations) == 2
    assert detail["destination_index"] is not None
''',
    encoding="utf-8",
)

print("issue 28 final fixups applied")
