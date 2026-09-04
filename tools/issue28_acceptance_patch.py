from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:140]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if addition.strip() in text:
        raise RuntimeError(f"{path}: addition already present")
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(f"{path}: expected one append marker, found {count}: {marker[:140]!r}")
    target.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")


acceptance = "carla_vision/operator/garage_acceptance.py"

replace_once(
    acceptance,
    "    mode: str,\n    camera_fps: float,\n) -> dict[str, Any]:",
    "    mode: str,\n    camera_fps: float,\n    start_spawn_index: int | None,\n    destination_spawn_index: int | None,\n) -> dict[str, Any]:",
)
replace_once(
    acceptance,
    '        "route_mode": "free",\n        "initial_control_mode": "manual",',
    '        "route_mode": (\n            "selected_destination"\n            if mode == "behavior" and destination_spawn_index is not None\n            else "free"\n        ),\n        "start_spawn_index": start_spawn_index,\n        "destination_spawn_index": (\n            destination_spawn_index if mode == "behavior" else None\n        ),\n        "initial_control_mode": "manual",',
)
replace_once(
    acceptance,
    "        camera_fps: float,\n        repeat: int,\n        expected_version: str = EXPECTED_CARLA_VERSION,",
    "        camera_fps: float,\n        repeat: int,\n        start_spawn_index: int | None = None,\n        destination_spawn_index: int | None = None,\n        expected_version: str = EXPECTED_CARLA_VERSION,",
)
replace_once(
    acceptance,
    "        self.camera_fps = camera_fps\n        self.repeat = repeat\n        self.expected_version = expected_version",
    "        self.camera_fps = camera_fps\n        self.repeat = repeat\n        self.start_spawn_index = start_spawn_index\n        self.destination_spawn_index = destination_spawn_index\n        self.expected_version = expected_version",
)
replace_once(
    acceptance,
    '                "camera": {"resolution": [1280, 720], "fps": self.camera_fps},\n                "repeat": self.repeat,',
    '                "camera": {"resolution": [1280, 720], "fps": self.camera_fps},\n                "route_selection": {\n                    "start_spawn_index": self.start_spawn_index,\n                    "destination_spawn_index": self.destination_spawn_index,\n                    "behavior_route_mode": (\n                        "selected_destination"\n                        if self.destination_spawn_index is not None\n                        else "free"\n                    ),\n                },\n                "repeat": self.repeat,',
)
replace_once(
    acceptance,
    '        catalog = dict(self.manager.catalog())\n        health = dict(self.worker.health())\n        report["preflight"] = {"operator_catalog": catalog, "worker_health": health}',
    '        catalog = dict(self.manager.catalog())\n        health = dict(self.worker.health())\n        worker_catalog = dict(self.worker.catalog())\n        report["preflight"] = {\n            "operator_catalog": catalog,\n            "worker_health": health,\n            "worker_catalog": worker_catalog,\n        }',
)
insert_after_vehicle = '''        _check(
            checks,
            "vehicle_available",
            vehicle_available,
            observed=self.vehicle,
            expected="vehicle advertised by the live catalog",
        )
'''
route_preflight = insert_after_vehicle + '''        selected_map = (
            _nested(health, "carla", "current_map")
            if self.map_name == "current"
            else self.map_name
        )
        spawn_map = worker_catalog.get("spawn_point_map")
        spawn_points = worker_catalog.get("spawn_points", [])
        spawn_indices = {
            item.get("index")
            for item in spawn_points
            if isinstance(item, Mapping)
            and isinstance(item.get("index"), int)
            and not isinstance(item.get("index"), bool)
        }
        selection_requested = (
            self.start_spawn_index is not None or self.destination_spawn_index is not None
        )
        if selection_requested:
            _check(
                checks,
                "spawn_point_selection_capability",
                _nested(worker_catalog, "capabilities", "spawn_point_selection") is True,
                observed=_nested(worker_catalog, "capabilities", "spawn_point_selection"),
                expected=True,
            )
            _check(
                checks,
                "spawn_catalog_matches_target_map",
                spawn_map == selected_map,
                observed={"spawn_point_map": spawn_map, "target_map": selected_map},
                expected="spawn-point catalog for the selected map",
                note=(
                    "Load the target map in Garage before choosing exact spawn indices; "
                    "the smoke gate refuses stale indices from another map."
                ),
            )
        if self.start_spawn_index is not None:
            _check(
                checks,
                "start_spawn_index_available",
                self.start_spawn_index in spawn_indices,
                observed=self.start_spawn_index,
                expected="index advertised by the selected map's live CARLA spawn catalog",
            )
        if self.destination_spawn_index is not None:
            _check(
                checks,
                "selected_route_capability",
                _nested(worker_catalog, "capabilities", "selected_route") is True,
                observed=_nested(worker_catalog, "capabilities", "selected_route"),
                expected=True,
            )
            _check(
                checks,
                "destination_spawn_index_available",
                self.destination_spawn_index in spawn_indices,
                observed=self.destination_spawn_index,
                expected="index advertised by the selected map's live CARLA spawn catalog",
            )
'''
replace_once(acceptance, insert_after_vehicle, route_preflight)
replace_once(
    acceptance,
    "                mode=mode,\n                camera_fps=self.camera_fps,\n            )",
    "                mode=mode,\n                camera_fps=self.camera_fps,\n                start_spawn_index=self.start_spawn_index,\n                destination_spawn_index=self.destination_spawn_index,\n            )",
)
population_block = '''            _check(
                checks,
                "walker_population_exact",
                running.get("walker_count_actual") == self.walker_count,
                observed=running.get("walker_count_actual"),
                expected=self.walker_count,
            )
'''
route_evidence = population_block + '''            worker_running = dict(self.worker.current_scene())
            scene_payload = worker_running.get("scene")
            if not isinstance(scene_payload, Mapping):
                raise RuntimeError("World Worker did not expose the running scene evidence")
            session["worker_running"] = worker_running
            expected_map = (
                _nested(self.worker.health(), "carla", "current_map")
                if self.map_name == "current"
                else self.map_name
            )
            _check(
                checks,
                "selected_map_applied",
                scene_payload.get("map_name") == expected_map,
                observed=scene_payload.get("map_name"),
                expected=expected_map,
            )
            if self.start_spawn_index is not None:
                _check(
                    checks,
                    "selected_start_spawn_applied",
                    scene_payload.get("spawn_index") == self.start_spawn_index,
                    observed=scene_payload.get("spawn_index"),
                    expected=self.start_spawn_index,
                    note="The authoritative World Worker scene must retain the exact UI-selected start.",
                )
            if mode == "behavior" and self.destination_spawn_index is not None:
                route = scene_payload.get("route")
                destination = scene_payload.get("destination")
                route_ok = (
                    scene_payload.get("route_mode") == "selected_destination"
                    and isinstance(route, Mapping)
                    and route.get("planned") is True
                    and isinstance(route.get("waypoint_count"), int)
                    and int(route["waypoint_count"]) > 1
                    and isinstance(destination, Mapping)
                    and destination.get("spawn_index") == self.destination_spawn_index
                )
                _check(
                    checks,
                    "selected_destination_planned",
                    route_ok,
                    observed={
                        "route_mode": scene_payload.get("route_mode"),
                        "route": route,
                        "destination": destination,
                    },
                    expected={
                        "route_mode": "selected_destination",
                        "destination_spawn_index": self.destination_spawn_index,
                        "planned": True,
                        "waypoint_count": ">1",
                    },
                    note=(
                        "Planning evidence comes from GlobalRoutePlanner on the World Worker; "
                        "Garage BehaviorAgent remains the control owner."
                    ),
                )
'''
replace_once(acceptance, population_block, route_evidence)
behavior_check = '''        _check(
            checks,
            "behavior_agent_actuated",
            True,
            observed={
                "control_source": state.get("control_source"),
                "commands": _nested(state, "autonomy", "commands"),
            },
            expected="official BehaviorAgent produces applied commands",
        )
'''
behavior_route_check = behavior_check + '''        if self.destination_spawn_index is not None:
            actual_destination = _nested(state, "autonomy", "detail", "destination_index")
            _check(
                checks,
                "behavior_destination_matches_selected",
                actual_destination == self.destination_spawn_index,
                observed=actual_destination,
                expected=self.destination_spawn_index,
                note="BehaviorAgent must consume the same destination retained by the Worker scene.",
            )
'''
replace_once(acceptance, behavior_check, behavior_route_check)
replace_once(
    acceptance,
    '    parser.add_argument("--map", dest="map_name", default="Town10HD_Opt")\n    parser.add_argument("--vehicle", default="vehicle.tesla.model3")',
    '    parser.add_argument("--map", dest="map_name", default="Town10HD_Opt")\n    parser.add_argument("--start-spawn-index", type=int, default=None)\n    parser.add_argument("--destination-spawn-index", type=int, default=None)\n    parser.add_argument("--vehicle", default="vehicle.tesla.model3")',
)
validate_anchor = '''    if not 1 <= args.repeat <= 10:
        raise ValueError("--repeat must be in [1, 10]")
'''
validate_insert = '''    if args.start_spawn_index is not None and args.start_spawn_index < 0:
        raise ValueError("--start-spawn-index must be >= 0")
    if args.destination_spawn_index is not None and args.destination_spawn_index < 0:
        raise ValueError("--destination-spawn-index must be >= 0")
    if (
        args.start_spawn_index is not None
        and args.destination_spawn_index is not None
        and args.start_spawn_index == args.destination_spawn_index
    ):
        raise ValueError("start and destination spawn indices must differ")
''' + validate_anchor
replace_once(acceptance, validate_anchor, validate_insert)
replace_once(
    acceptance,
    "            camera_fps=args.camera_fps,\n            repeat=args.repeat,\n            expected_version=args.expected_version,",
    "            camera_fps=args.camera_fps,\n            repeat=args.repeat,\n            start_spawn_index=args.start_spawn_index,\n            destination_spawn_index=args.destination_spawn_index,\n            expected_version=args.expected_version,",
)

# Fake acceptance harness now exposes the same authoritative route evidence.
test_path = "tests/test_garage_acceptance.py"
replace_once(
    test_path,
    "        self.walkers = 0\n\n    def health(self) -> dict[str, Any]:",
    "        self.walkers = 0\n        self.start_spawn_index: int | None = None\n        self.destination_spawn_index: int | None = None\n        self.route_mode = \"free\"\n\n    def health(self) -> dict[str, Any]:",
)
replace_once(
    test_path,
    '            "capabilities": {"camera_pause_resume": True},',
    '            "capabilities": {\n                "camera_pause_resume": True,\n                "spawn_point_selection": True,\n                "selected_route": True,\n            },',
)
replace_once(
    test_path,
    "    def current_scene(self) -> dict[str, Any]:\n",
    '''    def catalog(self) -> dict[str, Any]:
        return {
            "spawn_point_map": "Town10HD_Opt",
            "spawn_count": 4,
            "spawn_points": [
                {"index": index, "label": f"Spawn {index}", "transform": {}}
                for index in range(4)
            ],
            "capabilities": {
                "spawn_point_selection": True,
                "selected_route": True,
            },
        }

    def current_scene(self) -> dict[str, Any]:
''',
)
replace_once(
    test_path,
    '                "spawn_index": 1,\n                "route_mode": "free",\n                "route": {},\n                "destination": None,',
    '                "spawn_index": (\n                    self.start_spawn_index if self.start_spawn_index is not None else 1\n                ),\n                "route_mode": self.route_mode,\n                "route": (\n                    {"planned": True, "waypoint_count": 12, "enforced": False}\n                    if self.route_mode == "selected_destination"\n                    else {"planned": False, "waypoint_count": 0, "enforced": False}\n                ),\n                "destination": (\n                    {"spawn_index": self.destination_spawn_index, "transform": {}}\n                    if self.destination_spawn_index is not None\n                    else None\n                ),',
)
replace_once(
    test_path,
    '        self.worker.walkers = int(payload["walker_count"])\n        self.worker.paused = False',
    '        self.worker.walkers = int(payload["walker_count"])\n        self.worker.start_spawn_index = payload.get("start_spawn_index")\n        self.worker.destination_spawn_index = payload.get("destination_spawn_index")\n        self.worker.route_mode = str(payload.get("route_mode", "free"))\n        self.worker.paused = False',
)
replace_once(
    test_path,
    '                "detail": {\n                    "navigation_intent": {',
    '                "detail": {\n                    "destination_index": self.worker.destination_spawn_index,\n                    "navigation_intent": {',
)
replace_once(
    test_path,
    "    behavior_fails: bool = False,\n) -> tuple[GarageAcceptanceRunner, FakeManager, FakeWorker]:",
    "    behavior_fails: bool = False,\n    start_spawn_index: int | None = None,\n    destination_spawn_index: int | None = None,\n) -> tuple[GarageAcceptanceRunner, FakeManager, FakeWorker]:",
)
replace_once(
    test_path,
    "            repeat=repeat,\n            state_timeout=10.0,",
    "            repeat=repeat,\n            start_spawn_index=start_spawn_index,\n            destination_spawn_index=destination_spawn_index,\n            state_timeout=10.0,",
)
append_once(
    test_path,
    "\ndef test_wrong_carla_version_fails_preflight_without_spawning() -> None:\n",
    '''

def test_selected_map_start_and_destination_are_machine_readable_live_evidence() -> None:
    acceptance, _, _ = runner(
        repeat=1,
        start_spawn_index=1,
        destination_spawn_index=2,
    )
    report = acceptance.run()

    assert report["status"] == "pass"
    assert report["target"]["route_selection"] == {
        "start_spawn_index": 1,
        "destination_spawn_index": 2,
        "behavior_route_mode": "selected_destination",
    }
    preflight = {check["check_id"]: check for check in report["checks"]}
    assert preflight["spawn_point_selection_capability"]["passed"]
    assert preflight["spawn_catalog_matches_target_map"]["passed"]
    assert preflight["start_spawn_index_available"]["passed"]
    assert preflight["selected_route_capability"]["passed"]
    assert preflight["destination_spawn_index_available"]["passed"]

    manual, behavior = report["sessions"]
    assert manual["requested"]["route_mode"] == "free"
    assert manual["requested"]["start_spawn_index"] == 1
    assert manual["requested"]["destination_spawn_index"] is None
    assert behavior["requested"]["route_mode"] == "selected_destination"
    assert behavior["requested"]["start_spawn_index"] == 1
    assert behavior["requested"]["destination_spawn_index"] == 2
    behavior_checks = {check["check_id"]: check for check in behavior["checks"]}
    assert behavior_checks["selected_map_applied"]["passed"]
    assert behavior_checks["selected_start_spawn_applied"]["passed"]
    assert behavior_checks["selected_destination_planned"]["passed"]
    assert behavior_checks["behavior_destination_matches_selected"]["passed"]

''',
)
append_once(
    test_path,
    "\n\n@pytest.mark.parametrize(\n    (\"flag\", \"value\"),",
    '''

def test_cli_accepts_exact_spawn_selection_and_rejects_same_endpoint() -> None:
    args = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--start-spawn-index",
            "1",
            "--destination-spawn-index",
            "2",
        ]
    )
    assert args.start_spawn_index == 1
    assert args.destination_spawn_index == 2
    _validate_args(args)

    same = _parse_args(
        [
            "--world-worker-url",
            "http://127.0.0.1:8766",
            "--start-spawn-index",
            "2",
            "--destination-spawn-index",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="must differ"):
        _validate_args(same)
''',
)

# Keep the focused route/UI regressions in the permanent lean Operator CI lane.
ci = ".github/workflows/ci.yml"
replace_once(
    ci,
    "          tests/test_garage_research_bridge.py\n          tests/test_garage_scene_handoff.py",
    "          tests/test_garage_research_bridge.py\n          tests/test_garage_route_policy.py\n          tests/test_garage_scene_handoff.py",
)
replace_once(
    ci,
    "          tests/test_garage_visual_contract.py\n          tests/test_model_registry.py",
    "          tests/test_garage_visual_contract.py\n          tests/test_garage_world_route_ui.py\n          tests/test_model_registry.py",
)

print("issue 28 live acceptance wiring applied")
