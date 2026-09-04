from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {text.count(old)} for {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_before(path: str, marker: str, addition: str) -> None:
    replace_once(path, marker, addition + marker)


# --- World Worker: exact official CARLA spawn-point selection and selected routes. ---
replace_once(
    "carla_vision/native/world_worker.py",
    'WORKER_API_REVISION = 5',
    'WORKER_API_REVISION = 6',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '_ROUTE_MODES = frozenset({"free", "random_destination"})',
    '_ROUTE_MODES = frozenset({"free", "random_destination", "selected_destination"})',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '    route_mode: str = "free"\n    initial_control_mode: str = "manual"',
    '    route_mode: str = "free"\n    start_spawn_index: int | None = None\n    destination_spawn_index: int | None = None\n    initial_control_mode: str = "manual"',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '            "route_mode",\n            "initial_control_mode",',
    '            "route_mode",\n            "start_spawn_index",\n            "destination_spawn_index",\n            "initial_control_mode",',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                "route_mode must be free or random_destination",\n            )\n        control_mode = str(raw.get("initial_control_mode", "manual")).strip()',
    '                "route_mode must be free, random_destination, or selected_destination",\n            )\n        start_raw = raw.get("start_spawn_index")\n        destination_raw = raw.get("destination_spawn_index")\n        start_spawn_index = (\n            None\n            if start_raw is None\n            else _integer(start_raw, "start_spawn_index", 0, 1_000_000)\n        )\n        destination_spawn_index = (\n            None\n            if destination_raw is None\n            else _integer(destination_raw, "destination_spawn_index", 0, 1_000_000)\n        )\n        if route_mode == "selected_destination" and destination_spawn_index is None:\n            raise WorkerError(\n                HTTPStatus.BAD_REQUEST,\n                "invalid_field",\n                "selected_destination requires destination_spawn_index",\n            )\n        if route_mode != "selected_destination" and destination_spawn_index is not None:\n            raise WorkerError(\n                HTTPStatus.BAD_REQUEST,\n                "invalid_field",\n                "destination_spawn_index requires route_mode=selected_destination",\n            )\n        if (\n            start_spawn_index is not None\n            and destination_spawn_index is not None\n            and start_spawn_index == destination_spawn_index\n        ):\n            raise WorkerError(\n                HTTPStatus.BAD_REQUEST,\n                "invalid_field",\n                "start_spawn_index and destination_spawn_index must differ",\n            )\n        control_mode = str(raw.get("initial_control_mode", "manual")).strip()',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '            route_mode=route_mode,\n            initial_control_mode=control_mode,',
    '            route_mode=route_mode,\n            start_spawn_index=start_spawn_index,\n            destination_spawn_index=destination_spawn_index,\n            initial_control_mode=control_mode,',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '            "route_mode": self.route_mode,\n            "initial_control_mode": self.initial_control_mode,',
    '            "route_mode": self.route_mode,\n            "start_spawn_index": self.start_spawn_index,\n            "destination_spawn_index": self.destination_spawn_index,\n            "initial_control_mode": self.initial_control_mode,',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '            "random_route": random_route,\n            "lease": True,',
    '            "random_route": random_route,\n            "spawn_point_selection": True,\n            "selected_route": random_route,\n            "lease": True,',
)

# Catalog exact official spawn-point transforms for the active map.
replace_once(
    "carla_vision/native/world_worker.py",
    '            vehicles = []\n            for blueprint in self._safe_vehicle_blueprints(world.get_blueprint_library()):',
    '            spawn_points = list(world.get_map().get_spawn_points())\n            vehicles = []\n            for blueprint in self._safe_vehicle_blueprints(world.get_blueprint_library()):',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                "vehicles": vehicles,\n                "weather_presets": [',
    '                "vehicles": vehicles,\n                "spawn_point_map": _map_short_name(str(world.get_map().name)),\n                "spawn_count": len(spawn_points),\n                "spawn_points": [\n                    {\n                        "index": index,\n                        "label": f"Spawn {index}",\n                        "transform": _json_transform(transform),\n                    }\n                    for index, transform in enumerate(spawn_points)\n                ],\n                "weather_presets": [',
)

# Exact start means exact start: never silently fall through to another spawn.
replace_once(
    "carla_vision/native/world_worker.py",
    '        indices = list(range(len(spawn_points)))\n        rng.shuffle(indices)\n        for index in indices[: min(80, len(indices))]:\n            try:\n                actor = world.try_spawn_actor(blueprint, spawn_points[index])\n            except Exception:\n                actor = None\n            if actor is None:\n                continue\n            self._record_actor(owned, actor, kind="ego", role_name=role_name)\n            self._apply_full_brake(actor)\n            return actor, index\n        raise WorkerError(\n            HTTPStatus.SERVICE_UNAVAILABLE,\n            "ego_spawn_failed",\n            "could not spawn the ego vehicle at any official map spawn point",\n        )',
    '        if config.start_spawn_index is not None:\n            index = config.start_spawn_index\n            if index >= len(spawn_points):\n                raise WorkerError(\n                    HTTPStatus.UNPROCESSABLE_ENTITY,\n                    "spawn_index_out_of_range",\n                    f"start_spawn_index {index} is outside this map\'s {len(spawn_points)} spawn points",\n                )\n            try:\n                actor = world.try_spawn_actor(blueprint, spawn_points[index])\n            except Exception as error:\n                raise WorkerError(\n                    HTTPStatus.SERVICE_UNAVAILABLE,\n                    "ego_spawn_failed",\n                    f"CARLA failed while spawning the ego at selected spawn {index}: {error}",\n                ) from error\n            if actor is None:\n                raise WorkerError(\n                    HTTPStatus.UNPROCESSABLE_ENTITY,\n                    "ego_spawn_unavailable",\n                    f"selected start spawn {index} is occupied or unavailable; no fallback was used",\n                )\n            self._record_actor(owned, actor, kind="ego", role_name=role_name)\n            self._apply_full_brake(actor)\n            return actor, index\n\n        indices = list(range(len(spawn_points)))\n        rng.shuffle(indices)\n        for index in indices[: min(80, len(indices))]:\n            try:\n                actor = world.try_spawn_actor(blueprint, spawn_points[index])\n            except Exception:\n                actor = None\n            if actor is None:\n                continue\n            self._record_actor(owned, actor, kind="ego", role_name=role_name)\n            self._apply_full_brake(actor)\n            return actor, index\n        raise WorkerError(\n            HTTPStatus.SERVICE_UNAVAILABLE,\n            "ego_spawn_failed",\n            "could not spawn the ego vehicle at any official map spawn point",\n        )',
)

# Selected destination uses the same official GlobalRoutePlanner primitive as random routes.
append_before(
    "carla_vision/native/world_worker.py",
    '    def prepare(self, raw: Mapping[str, Any]) -> dict[str, Any]:\n',
    '''    def _plan_selected_route(\n        self,\n        world: Any,\n        ego: Any,\n        spawn_points: list[Any],\n        spawn_index: int,\n        destination_index: int,\n    ) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:\n        if destination_index >= len(spawn_points):\n            raise WorkerError(\n                HTTPStatus.UNPROCESSABLE_ENTITY,\n                "spawn_index_out_of_range",\n                f"destination_spawn_index {destination_index} is outside this map's "\n                f"{len(spawn_points)} spawn points",\n            )\n        if destination_index == spawn_index:\n            raise WorkerError(\n                HTTPStatus.UNPROCESSABLE_ENTITY,\n                "route_destination_matches_start",\n                "selected destination must differ from the actual ego start",\n            )\n        factory = self._planner_factory()\n        if factory is None:\n            raise WorkerError(\n                HTTPStatus.UNPROCESSABLE_ENTITY,\n                "selected_route_unavailable",\n                "GlobalRoutePlanner is unavailable on the World Worker host",\n            )\n        destination_transform = spawn_points[destination_index]\n        try:\n            planner = factory(world.get_map())\n            traced = list(planner.trace_route(ego.get_location(), destination_transform.location))\n            locations = [item[0].transform.location for item in traced]\n        except Exception as error:\n            raise WorkerError(\n                HTTPStatus.UNPROCESSABLE_ENTITY,\n                "selected_route_failed",\n                f"could not trace route to selected destination {destination_index}: {error}",\n            ) from error\n        if len(locations) < 2:\n            raise WorkerError(\n                HTTPStatus.UNPROCESSABLE_ENTITY,\n                "selected_route_failed",\n                f"route to selected destination {destination_index} is trivial",\n            )\n        return (\n            {\n                "mode": "selected_destination",\n                "provider": "GlobalRoutePlanner.trace_route+TrafficManager.set_path",\n                "planned": True,\n                "enforced": False,\n                "waypoint_count": len(locations),\n            },\n            {\n                "spawn_index": destination_index,\n                "transform": _json_transform(destination_transform),\n            },\n            locations,\n        )\n\n''',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                if config.route_mode == "random_destination" and (\n                    self._planner_factory() is None or not hasattr(traffic_manager, "set_path")\n                ):',
    '                if config.route_mode in {"random_destination", "selected_destination"} and (\n                    self._planner_factory() is None or not hasattr(traffic_manager, "set_path")\n                ):',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                        "random_destination requires GlobalRoutePlanner and "\n                        "TrafficManager.set_path",',
    '                        "planned routes require GlobalRoutePlanner and TrafficManager.set_path",',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                if config.route_mode == "random_destination":\n                    partial.route, partial.destination, partial.route_locations = (\n                        self._plan_random_route(\n                            world,\n                            ego,\n                            spawn_points,\n                            spawn_index,\n                            role_rng,\n                        )\n                    )\n                else:',
    '                if config.route_mode == "random_destination":\n                    partial.route, partial.destination, partial.route_locations = (\n                        self._plan_random_route(\n                            world,\n                            ego,\n                            spawn_points,\n                            spawn_index,\n                            role_rng,\n                        )\n                    )\n                elif config.route_mode == "selected_destination":\n                    assert config.destination_spawn_index is not None\n                    partial.route, partial.destination, partial.route_locations = (\n                        self._plan_selected_route(\n                            world,\n                            ego,\n                            spawn_points,\n                            spawn_index,\n                            config.destination_spawn_index,\n                        )\n                    )\n                else:',
)
# Reconfigure start must restart; destination can replan in-place.
replace_once(
    "carla_vision/native/world_worker.py",
    '            if config.map_name != original.map_name or config.seed != original.seed:',
    '            if (\n                config.map_name != original.map_name\n                or config.seed != original.seed\n                or config.start_spawn_index != original.start_spawn_index\n            ):',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                    "map or seed changes require a new prepared scene",',
    '                    "map, seed, or start spawn changes require a new prepared scene",',
)
# There is a second route capability check in configure.
replace_once(
    "carla_vision/native/world_worker.py",
    '            if config.route_mode == "random_destination" and (\n                self._planner_factory() is None or not hasattr(tm, "set_path")\n            ):',
    '            if config.route_mode in {"random_destination", "selected_destination"} and (\n                self._planner_factory() is None or not hasattr(tm, "set_path")\n            ):',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                    "random_destination requires a route planner and set_path",',
    '                    "planned routes require a route planner and set_path",',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                if config.route_mode != scene.config.route_mode:\n                    if config.route_mode == "random_destination":\n                        route, destination, locations = self._plan_random_route(\n                            world, scene.ego, spawn_points, scene.spawn_index, rng\n                        )\n                    else:\n                        route = {\n                            "mode": "free",\n                            "provider": None,\n                            "planned": False,\n                            "enforced": False,\n                            "waypoint_count": 0,\n                        }\n                        destination, locations = None, []\n                    scene.route, scene.destination, scene.route_locations = (\n                        route,\n                        destination,\n                        locations,\n                    )\n                    scene.config = replace(scene.config, route_mode=config.route_mode)',
    '                if (\n                    config.route_mode != scene.config.route_mode\n                    or config.destination_spawn_index != scene.config.destination_spawn_index\n                ):\n                    if config.route_mode == "random_destination":\n                        route, destination, locations = self._plan_random_route(\n                            world, scene.ego, spawn_points, scene.spawn_index, rng\n                        )\n                    elif config.route_mode == "selected_destination":\n                        assert config.destination_spawn_index is not None\n                        route, destination, locations = self._plan_selected_route(\n                            world,\n                            scene.ego,\n                            spawn_points,\n                            scene.spawn_index,\n                            config.destination_spawn_index,\n                        )\n                    else:\n                        route = {\n                            "mode": "free",\n                            "provider": None,\n                            "planned": False,\n                            "enforced": False,\n                            "waypoint_count": 0,\n                        }\n                        destination, locations = None, []\n                    scene.route, scene.destination, scene.route_locations = (\n                        route,\n                        destination,\n                        locations,\n                    )\n                    scene.config = replace(\n                        scene.config,\n                        route_mode=config.route_mode,\n                        destination_spawn_index=config.destination_spawn_index,\n                    )',
)
# Start enforces either planned route through TrafficManager.set_path.
replace_once(
    "carla_vision/native/world_worker.py",
    '            if scene.config.route_mode == "random_destination":',
    '            if scene.config.route_mode in {"random_destination", "selected_destination"}:',
)

# Observable worker reports/guards the new selected route with the same progress lane.
replace_once(
    "carla_vision/native/observable_world_worker.py",
    '            "route_mode": config.route_mode,\n            "pedestrian_crossing_factor":',
    '            "route_mode": config.route_mode,\n            "start_spawn_index": config.start_spawn_index,\n            "destination_spawn_index": config.destination_spawn_index,\n            "pedestrian_crossing_factor":',
)
append_before(
    "carla_vision/native/observable_world_worker.py",
    '\n\ndef main(argv: Sequence[str] | None = None) -> int:\n',
    '''\n    def _plan_selected_route(\n        self,\n        world: Any,\n        ego: Any,\n        spawn_points: list[Any],\n        spawn_index: int,\n        destination_index: int,\n    ) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:\n        self._check_prepare_cancelled()\n        self._set_preparation_stage("route")\n        return super()._plan_selected_route(\n            world, ego, spawn_points, spawn_index, destination_index\n        )\n''',
)

# World Worker client allow-list carries optional exact spawn selections.
replace_once(
    "carla_vision/operator/world_worker_client.py",
    '        "following_distance_metres",\n    }\n)',
    '        "following_distance_metres",\n        "start_spawn_index",\n        "destination_spawn_index",\n    }\n)',
)

# Drive contract and retained manifest.
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '_ROUTE_MODES = frozenset({"free", "random_destination"})',
    '_ROUTE_MODES = frozenset({"free", "random_destination", "selected_destination"})',
)
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '        "route_mode",\n        "initial_control_mode",',
    '        "route_mode",\n        "start_spawn_index",\n        "destination_spawn_index",\n        "initial_control_mode",',
)
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '    route_mode: str = "free"\n    initial_control_mode: str = "manual"',
    '    route_mode: str = "free"\n    start_spawn_index: int | None = None\n    destination_spawn_index: int | None = None\n    initial_control_mode: str = "manual"',
)
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '        if route_mode not in _ROUTE_MODES:\n            raise ValueError("route_mode must be free or random_destination")\n        initial_control_mode =',
    '        if route_mode not in _ROUTE_MODES:\n            raise ValueError("route_mode must be free, random_destination, or selected_destination")\n        start_raw = raw.get("start_spawn_index")\n        destination_raw = raw.get("destination_spawn_index")\n        start_spawn_index = (\n            None\n            if start_raw is None\n            else _integer(start_raw, "start_spawn_index", 0, 1_000_000)\n        )\n        destination_spawn_index = (\n            None\n            if destination_raw is None\n            else _integer(destination_raw, "destination_spawn_index", 0, 1_000_000)\n        )\n        if route_mode == "selected_destination" and destination_spawn_index is None:\n            raise ValueError("selected_destination requires destination_spawn_index")\n        if route_mode != "selected_destination" and destination_spawn_index is not None:\n            raise ValueError("destination_spawn_index requires route_mode=selected_destination")\n        if start_spawn_index is not None and start_spawn_index == destination_spawn_index:\n            raise ValueError("start_spawn_index and destination_spawn_index must differ")\n        initial_control_mode =',
)
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '            if route_mode != "free":\n                unsupported.append("route_mode")',
    '            if route_mode != "free":\n                unsupported.append("route_mode")\n            if start_spawn_index is not None:\n                unsupported.append("start_spawn_index")\n            if destination_spawn_index is not None:\n                unsupported.append("destination_spawn_index")',
)
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '            route_mode=route_mode,\n            initial_control_mode=initial_control_mode,',
    '            route_mode=route_mode,\n            start_spawn_index=start_spawn_index,\n            destination_spawn_index=destination_spawn_index,\n            initial_control_mode=initial_control_mode,',
)
replace_once(
    "carla_vision/operator/drive_contracts.py",
    '            "route_mode": self.route_mode,\n            "initial_control_mode": self.initial_control_mode,',
    '            "route_mode": self.route_mode,\n            "start_spawn_index": self.start_spawn_index,\n            "destination_spawn_index": self.destination_spawn_index,\n            "initial_control_mode": self.initial_control_mode,',
)

# Canonical product configuration exposes exact spawn-point intent.
replace_once(
    "carla_vision/operator/configuration.py",
    '    "route": frozenset({"mode"}),',
    '    "route": frozenset({"mode", "startSpawnIndex", "destinationSpawnIndex"}),',
)
replace_once(
    "carla_vision/operator/configuration.py",
    '        "route_mode": str(session["route"]["mode"]).strip(),\n        "pedestrian_crossing_factor":',
    '        "route_mode": str(session["route"]["mode"]).strip(),\n        "start_spawn_index": session["route"]["startSpawnIndex"],\n        "destination_spawn_index": session["route"]["destinationSpawnIndex"],\n        "pedestrian_crossing_factor":',
)
replace_once(
    "carla_vision/operator/configuration.py",
    '        "route": {"mode": "free"},',
    '        "route": {\n            "mode": "free",\n            "startSpawnIndex": None,\n            "destinationSpawnIndex": None,\n        },',
)
replace_once(
    "carla_vision/operator/configuration.py",
    '        "route_mode": str(route["mode"]).strip(),\n        "initial_control_mode": initial_control_mode,',
    '        "route_mode": str(route["mode"]).strip(),\n        "start_spawn_index": route["startSpawnIndex"],\n        "destination_spawn_index": route["destinationSpawnIndex"],\n        "initial_control_mode": initial_control_mode,',
)

# Garage preview request and restart boundary.
replace_once(
    "carla_vision/operator/garage_preview.py",
    '    route_mode: str = "free"\n',
    '    route_mode: str = "free"\n    start_spawn_index: int | None = None\n    destination_spawn_index: int | None = None\n',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '                "route_mode",\n            )',
    '                "route_mode",\n                "start_spawn_index",\n                "destination_spawn_index",\n            )',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '            "route_mode",\n        }',
    '            "route_mode",\n            "start_spawn_index",\n            "destination_spawn_index",\n        }',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '        if route_mode not in {"free", "random_destination"}:\n            raise ValueError("route_mode must be free or random_destination")',
    '        if route_mode not in {"free", "random_destination", "selected_destination"}:\n            raise ValueError(\n                "route_mode must be free, random_destination, or selected_destination"\n            )\n        start_raw = raw.get("start_spawn_index")\n        destination_raw = raw.get("destination_spawn_index")\n        start_spawn_index = (\n            None\n            if start_raw is None\n            else _integer(start_raw, name="start_spawn_index", minimum=0, maximum=1_000_000)\n        )\n        destination_spawn_index = (\n            None\n            if destination_raw is None\n            else _integer(\n                destination_raw,\n                name="destination_spawn_index",\n                minimum=0,\n                maximum=1_000_000,\n            )\n        )\n        if route_mode == "selected_destination" and destination_spawn_index is None:\n            raise ValueError("selected_destination requires destination_spawn_index")\n        if route_mode != "selected_destination" and destination_spawn_index is not None:\n            raise ValueError("destination_spawn_index requires selected_destination")\n        if start_spawn_index is not None and start_spawn_index == destination_spawn_index:\n            raise ValueError("start and destination spawn points must differ")',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '            route_mode=route_mode,\n            fov=_number(',
    '            route_mode=route_mode,\n            start_spawn_index=start_spawn_index,\n            destination_spawn_index=destination_spawn_index,\n            fov=_number(',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '            "route_mode": self.route_mode,\n            "pedestrian_crossing_factor":',
    '            "route_mode": self.route_mode,\n            "start_spawn_index": self.start_spawn_index,\n            "destination_spawn_index": self.destination_spawn_index,\n            "pedestrian_crossing_factor":',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '            if config.map_name != self.config.map_name or config.seed != self.config.seed:\n                return False',
    '            if (\n                config.map_name != self.config.map_name\n                or config.seed != self.config.seed\n                or config.start_spawn_index != self.config.start_spawn_index\n            ):\n                return False',
)
replace_once(
    "carla_vision/operator/garage_preview.py",
    '                "prop_preset": self.config.prop_preset,\n                "spectator_mirror":',
    '                "prop_preset": self.config.prop_preset,\n                "spawn_index": None if scene is None else scene.spawn_index,\n                "route_mode": None if scene is None else scene.route_mode,\n                "route": {} if scene is None else dict(scene.route),\n                "destination": None if scene is None else scene.destination,\n                "spectator_mirror":',
)

# Operator catalog forwards current-map spawn points from the Worker.
replace_once(
    "carla_vision/operator/drive.py",
    '            "spawn_count": 0,\n            "weather_presets":',
    '            "spawn_count": 0,\n            "spawn_point_map": None,\n            "spawn_points": [],\n            "weather_presets":',
)
replace_once(
    "carla_vision/operator/drive.py",
    '                spawn_count = worker_catalog.get("spawn_count")',
    '                spawn_point_map = worker_catalog.get("spawn_point_map")\n                if isinstance(spawn_point_map, str) and spawn_point_map.strip():\n                    base["spawn_point_map"] = _map_short_name(spawn_point_map.strip())\n                spawn_points = worker_catalog.get("spawn_points")\n                if isinstance(spawn_points, list):\n                    base["spawn_points"] = spawn_points\n                spawn_count = worker_catalog.get("spawn_count")',
)

# Behavior/Voxel consume the Worker's selected/random route destination instead of inventing another.
replace_once(
    "carla_vision/operator/garage_drive.py",
    'class _BehaviorPolicy:\n    def __init__(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:',
    'class _BehaviorPolicy:\n    def __init__(\n        self,\n        context: _CarlaContext,\n        config: GarageDriveStartConfig,\n        *,\n        destination_index: int | None = None,\n    ) -> None:',
)
replace_once(
    "carla_vision/operator/garage_drive.py",
    '        self.destination_index: int | None = None\n        self.route_generation = 0\n        self._set_destination()',
    '        self.destination_index: int | None = destination_index\n        self.route_generation = 0\n        self.route_complete = False\n        self._set_destination()',
)
replace_once(
    "carla_vision/operator/garage_drive.py",
    '    def _set_destination(self) -> None:\n        current = self.context.ego.get_location()\n        candidates = [\n            (index, transform)\n            for index, transform in enumerate(self.context.spawn_points)\n            if current.distance(transform.location) >= 80.0\n        ] or list(enumerate(self.context.spawn_points))\n        self.destination_index, destination = self.context.rng.choice(candidates)\n        self.agent.set_destination(destination.location)\n        self.route_generation += 1\n\n    def step(self, *_: Any, **__: Any) -> tuple[ControlCommand, str, bool, dict[str, Any]]:\n        if self.agent.done():\n            self._set_destination()',
    '    def _set_destination(self) -> None:\n        current = self.context.ego.get_location()\n        if self.destination_index is None:\n            candidates = [\n                (index, transform)\n                for index, transform in enumerate(self.context.spawn_points)\n                if current.distance(transform.location) >= 80.0\n            ] or list(enumerate(self.context.spawn_points))\n            self.destination_index, destination = self.context.rng.choice(candidates)\n        else:\n            if self.destination_index >= len(self.context.spawn_points):\n                raise RuntimeError(\n                    f"route destination {self.destination_index} is outside the active map"\n                )\n            destination = self.context.spawn_points[self.destination_index]\n        self.agent.set_destination(destination.location)\n        self.route_generation += 1\n\n    def step(self, *_: Any, **__: Any) -> tuple[ControlCommand, str, bool, dict[str, Any]]:\n        if self.agent.done():\n            self.route_complete = True\n            return (\n                ControlCommand.service_brake(),\n                "behavior_route_complete",\n                False,\n                {"destination_index": self.destination_index, "route_complete": True},\n            )',
)
replace_once(
    "carla_vision/operator/garage_drive.py",
    'class _VoxelPolicy:\n    def __init__(self, context: _CarlaContext, config: GarageDriveStartConfig) -> None:',
    'class _VoxelPolicy:\n    def __init__(\n        self,\n        context: _CarlaContext,\n        config: GarageDriveStartConfig,\n        *,\n        destination_index: int | None = None,\n    ) -> None:',
)
replace_once(
    "carla_vision/operator/garage_drive.py",
    '        self.behavior = _BehaviorPolicy(context, config)',
    '        self.behavior = _BehaviorPolicy(\n            context, config, destination_index=destination_index\n        )',
)
# Add helper on session and use Worker's authoritative route destination.
replace_once(
    "carla_vision/operator/garage_drive.py",
    '    def _ensure_extensions(self) -> None:\n',
    '    def _route_destination_index(self) -> int | None:\n        scene = self._worker_scene\n        if scene is not None and isinstance(scene.destination, Mapping):\n            raw = scene.destination.get("spawn_index")\n            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:\n                return raw\n        return self.config.destination_spawn_index\n\n    def _ensure_extensions(self) -> None:\n',
)
replace_once(
    "carla_vision/operator/garage_drive.py",
    '        if self.config.control_mode == "behavior":\n            self._policy = _BehaviorPolicy(context, self.config)\n        elif self.config.control_mode == "imitation":\n            self._policy = _ImitationPolicy(context, self.config)\n        elif self.config.control_mode == "voxel":\n            self._policy = _VoxelPolicy(context, self.config)',
    '        destination_index = self._route_destination_index()\n        if self.config.control_mode == "behavior":\n            self._policy = _BehaviorPolicy(\n                context, self.config, destination_index=destination_index\n            )\n        elif self.config.control_mode == "imitation":\n            self._policy = _ImitationPolicy(context, self.config)\n        elif self.config.control_mode == "voxel":\n            self._policy = _VoxelPolicy(\n                context, self.config, destination_index=destination_index\n            )',
)

# --- Frontend canonical contract/catalog. ---
replace_once(
    "web/src/lib/domain/config.ts",
    "export type RouteMode = 'free' | 'random_destination';",
    "export type RouteMode = 'free' | 'random_destination' | 'selected_destination';",
)
replace_once(
    "web/src/lib/domain/config.ts",
    '  route: {\n    mode: RouteMode;\n  };',
    '  route: {\n    mode: RouteMode;\n    startSpawnIndex: number | null;\n    destinationSpawnIndex: number | null;\n  };',
)
replace_once(
    "web/src/lib/domain/config.ts",
    'export interface WorkspaceOptions {\n  maps: CatalogOption[];',
    'export interface SpawnPointOption {\n  index: number;\n  label: string;\n  transform?: Record<string, unknown>;\n}\n\nexport interface WorkspaceOptions {\n  maps: CatalogOption[];\n  spawnPointMap: string | null;\n  spawnPoints: SpawnPointOption[];',
)
replace_once(
    "web/src/lib/domain/config.ts",
    "    route: {\n      mode: 'free'\n    },",
    "    route: {\n      mode: 'free',\n      startSpawnIndex: null,\n      destinationSpawnIndex: null\n    },",
)
replace_once(
    "web/src/lib/api/operator.ts",
    '  maps: Array<string | { id: string; label?: string }>;\n  vehicles:',
    '  maps: Array<string | { id: string; label?: string }>;\n  spawn_point_map?: string | null;\n  spawn_points?: Array<{ index: number; label?: string; transform?: Record<string, unknown> }>;\n  vehicles:',
)
replace_once(
    "web/src/lib/api/operator.ts",
    '    maps: normalizeMapOptions(driveCatalog.maps),\n    vehicles:',
    '    maps: normalizeMapOptions(driveCatalog.maps),\n    spawnPointMap: driveCatalog.spawn_point_map ? shortMapName(driveCatalog.spawn_point_map) : null,\n    spawnPoints: Array.isArray(driveCatalog.spawn_points)\n      ? driveCatalog.spawn_points\n          .filter((point) => Number.isInteger(point.index) && point.index >= 0)\n          .map((point) => ({\n            index: point.index,\n            label: String(point.label ?? `Spawn ${point.index}`),\n            transform: point.transform\n          }))\n      : [],\n    vehicles:',
)
replace_once(
    "web/src/lib/api/operator.ts",
    '  getDriveState(): Promise<DriveState> {\n    return readJson<DriveState>(\'/api/drive/state\');\n  }',
    '  getDriveState(): Promise<DriveState> {\n    return readJson<DriveState>(\'/api/drive/state\');\n  }\n\n  getDriveCatalog(): Promise<DriveCatalogPayload> {\n    return readJson<DriveCatalogPayload>(\'/api/drive/catalog\');\n  }',
)
replace_once(
    "web/src/lib/stores/configuration.ts",
    'import type { WorkspaceSnapshot } from \'$lib/api/operator\';',
    'import type { DriveCatalogPayload, WorkspaceSnapshot } from \'$lib/api/operator\';',
)
replace_once(
    "web/src/lib/stores/configuration.ts",
    '  maps: [],\n  vehicles:',
    '  maps: [],\n  spawnPointMap: null,\n  spawnPoints: [],\n  vehicles:',
)
replace_once(
    "web/src/lib/stores/configuration.ts",
    "  if (\n    next.route.mode === 'random_destination' &&\n    !snapshot.system.capabilities.random_route\n  ) {\n    next.route.mode = 'free';\n  }",
    "  if (next.route.mode === 'random_destination' && !snapshot.system.capabilities.random_route) {\n    next.route.mode = 'free';\n  }\n  if (next.route.mode === 'selected_destination' && !snapshot.system.capabilities.selected_route) {\n    next.route.mode = 'free';\n    next.route.destinationSpawnIndex = null;\n  }\n  if (!snapshot.system.capabilities.spawn_point_selection) {\n    next.route.startSpawnIndex = null;\n    next.route.destinationSpawnIndex = null;\n  }",
)
append_before(
    "web/src/lib/stores/configuration.ts",
    '\nexport function patchSessionSection<K extends keyof SessionConfig>(',
    '''\nexport function refreshWorldCatalog(catalog: DriveCatalogPayload): void {\n  systemSettings.update((current) => current ? {\n    ...current,\n    connected: Boolean(catalog.connected),\n    serverVersion: catalog.server_version ?? current.serverVersion,\n    currentMap: catalog.map ?? current.currentMap,\n    workerConnected: Boolean(catalog.world_worker?.connected),\n    capabilities: { ...current.capabilities, ...catalog.capabilities }\n  } : current);\n  workspaceOptions.update((current) => ({\n    ...current,\n    maps: Array.isArray(catalog.maps) ? catalog.maps.map((value) => {\n      const raw = typeof value === 'string' ? value : value.id;\n      const id = String(raw).replace(/\\/+$/, '').split('/').at(-1) ?? String(raw);\n      const label = typeof value === 'string' ? id : String(value.label ?? id);\n      return { id, label };\n    }) : current.maps,\n    spawnPointMap: catalog.spawn_point_map\n      ? String(catalog.spawn_point_map).replace(/\\/+$/, '').split('/').at(-1) ?? null\n      : null,\n    spawnPoints: Array.isArray(catalog.spawn_points)\n      ? catalog.spawn_points\n          .filter((point) => Number.isInteger(point.index) && point.index >= 0)\n          .map((point) => ({\n            index: point.index,\n            label: String(point.label ?? `Spawn ${point.index}`),\n            transform: point.transform\n          }))\n      : []\n  }));\n}\n''',
)

# Preview signature/validation includes the real world-selection contract.
replace_once(
    "web/src/lib/domain/garagePreview.ts",
    '    route_mode: session.route.mode,\n    pedestrian_crossing_factor:',
    '    route_mode: session.route.mode,\n    start_spawn_index: session.route.startSpawnIndex,\n    destination_spawn_index: session.route.destinationSpawnIndex,\n    pedestrian_crossing_factor:',
)
replace_once(
    "web/src/lib/domain/garagePreview.ts",
    "  if (!integer(session.scene.trafficCount, 0, 250) || !integer(session.scene.walkerCount, 0, 250)) {",
    "  for (const [label, value] of [\n    ['Start spawn', session.route.startSpawnIndex],\n    ['Destination spawn', session.route.destinationSpawnIndex]\n  ] as const) {\n    if (value !== null && !integer(value, 0, 1_000_000)) return `${label} must be a valid spawn index.`;\n  }\n  if (session.route.mode === 'selected_destination' && session.route.destinationSpawnIndex === null) {\n    return 'Choose a destination spawn point for the selected route.';\n  }\n  if (session.route.mode !== 'selected_destination' && session.route.destinationSpawnIndex !== null) {\n    return 'Destination spawn point requires Selected destination route mode.';\n  }\n  if (session.route.startSpawnIndex !== null\n    && session.route.startSpawnIndex === session.route.destinationSpawnIndex) {\n    return 'Start and destination spawn points must differ.';\n  }\n  if (!integer(session.scene.trafficCount, 0, 250) || !integer(session.scene.walkerCount, 0, 250)) {",
)

# Scene/route controls: map was already real; add exact start/destination selection only when catalog matches.
replace_once(
    "web/src/lib/components/SceneWorldFields.svelte",
    "  $: workerAvailable = Boolean($systemSettings?.workerConfigured);",
    "  $: workerAvailable = Boolean($systemSettings?.workerConfigured);\n  $: selectedMap = $sessionConfig.scene.mapName === 'current'\n    ? $systemSettings?.currentMap ?? ''\n    : $sessionConfig.scene.mapName;\n  $: spawnCatalogReady = Boolean(\n    workerAvailable\n      && $systemSettings?.capabilities.spawn_point_selection\n      && $workspaceOptions.spawnPointMap\n      && selectedMap === $workspaceOptions.spawnPointMap\n  );\n  $: selectedRouteAvailable = Boolean($systemSettings?.capabilities.selected_route);\n\n  function spawnValue(value: string): number | null {\n    return value === '' ? null : Number(value);\n  }",
)
# Replace the existing route field block with richer world-selection fields. Exact markup sourced from current file.
replace_once(
    "web/src/lib/components/SceneWorldFields.svelte",
    '''    <label class="field">\n      <span>Route</span>\n      <select\n        value={$sessionConfig.route.mode}\n        disabled={!workerAvailable || !$systemSettings?.capabilities.random_route}\n        onchange={(event) => patchSessionSection('route', { mode: event.currentTarget.value as RouteMode })}\n      >\n        <option value="free">Free drive</option>\n        <option value="random_destination">Random destination</option>\n      </select>\n      <small>Random destination uses the World Worker route planner.</small>\n    </label>''',
    '''    <label class="field">\n      <span>Start point</span>\n      <select\n        value={$sessionConfig.route.startSpawnIndex ?? ''}\n        disabled={!spawnCatalogReady}\n        onchange={(event) => patchSessionSection('route', {\n          startSpawnIndex: spawnValue(event.currentTarget.value)\n        })}\n      >\n        <option value="">Automatic from seed</option>\n        {#each $workspaceOptions.spawnPoints as point}\n          <option value={point.index}>#{point.index} · {point.label}</option>\n        {/each}\n      </select>\n      <small>{spawnCatalogReady ? 'Exact CARLA map spawn point; no fallback if occupied.' : 'Load the selected map in Garage to enumerate its CARLA spawn points.'}</small>\n    </label>\n\n    <label class="field">\n      <span>Route</span>\n      <select\n        value={$sessionConfig.route.mode}\n        disabled={!workerAvailable}\n        onchange={(event) => {\n          const mode = event.currentTarget.value as RouteMode;\n          patchSessionSection('route', {\n            mode,\n            destinationSpawnIndex: mode === 'selected_destination'\n              ? $sessionConfig.route.destinationSpawnIndex\n              : null\n          });\n        }}\n      >\n        <option value="free">Free drive</option>\n        <option value="random_destination" disabled={!$systemSettings?.capabilities.random_route}>Random destination</option>\n        <option value="selected_destination" disabled={!selectedRouteAvailable || !spawnCatalogReady}>Selected destination</option>\n      </select>\n      <small>Planned routes use CARLA GlobalRoutePlanner; Traffic Manager or BehaviorAgent consumes the route according to the selected control owner.</small>\n    </label>\n\n    {#if $sessionConfig.route.mode === 'selected_destination'}\n      <label class="field">\n        <span>Destination</span>\n        <select\n          value={$sessionConfig.route.destinationSpawnIndex ?? ''}\n          disabled={!spawnCatalogReady}\n          onchange={(event) => patchSessionSection('route', {\n            destinationSpawnIndex: spawnValue(event.currentTarget.value)\n          })}\n        >\n          <option value="">Choose destination…</option>\n          {#each $workspaceOptions.spawnPoints as point}\n            <option\n              value={point.index}\n              disabled={point.index === $sessionConfig.route.startSpawnIndex}\n            >#{point.index} · {point.label}</option>\n          {/each}\n        </select>\n        <small>The exact selected spawn point is passed to GlobalRoutePlanner and retained in session evidence.</small>\n      </label>\n    {/if}''',
)

# Refresh spawn-point catalog after an automatic map/restart apply, without resetting user config.
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    "  import { sessionConfig, systemSettings, workspaceOptions } from '$lib/stores/configuration';",
    "  import {\n    refreshWorldCatalog,\n    sessionConfig,\n    systemSettings,\n    workspaceOptions\n  } from '$lib/stores/configuration';",
)
replace_once(
    "web/src/lib/components/GaragePreview.svelte",
    '      systemSettings.update((current) =>\n        current ? { ...current, workerConnected: true } : current\n      );',
    '      systemSettings.update((current) =>\n        current ? { ...current, workerConnected: true } : current\n      );\n      void runtimeOperatorApi().getDriveCatalog().then(refreshWorldCatalog).catch(() => {\n        // Preview evidence remains authoritative; catalog refresh is convenience-only.\n      });',
)

# Drive cockpit shows the actual runtime control owner and hides manual controls for policy-owned modes.
replace_once(
    "web/src/lib/components/DriveCockpit.svelte",
    "  $: modelOwnsControl = String(drive.garage_mode ?? '').toLowerCase() === 'model';",
    "  $: garageMode = String(drive.garage_mode ?? '').toLowerCase();\n  $: policyOwnsControl = ['behavior', 'imitation', 'voxel', 'model'].includes(garageMode);\n  $: controlOwner = drive.control_mode === 'autopilot'\n    ? 'CARLA Traffic Manager'\n    : garageMode === 'behavior'\n      ? 'CARLA BehaviorAgent'\n      : garageMode === 'imitation'\n        ? 'Imitation policy'\n        : garageMode === 'voxel'\n          ? 'Voxel supervisor'\n          : garageMode === 'model'\n            ? 'Registered model'\n            : 'Browser manual';",
)
replace_once(
    "web/src/lib/components/DriveCockpit.svelte",
    "        <span class:ok={running} class=\"status-pill\"><i></i>{drive.control_mode ?? drive.garage_mode ?? 'manual'}</span>",
    "        <span class:ok={running} class=\"status-pill\"><i></i>{controlOwner}</span>",
)
replace_once(
    "web/src/lib/components/DriveCockpit.svelte",
    '          <div class="telemetry-row"><span>Control</span><strong>{drive.control_source ?? \'—\'}</strong></div>',
    '          <div class="telemetry-row"><span>Control owner</span><strong>{controlOwner}</strong></div>\n          <div class="telemetry-row"><span>Control source</span><strong>{drive.control_source ?? \'—\'}</strong></div>',
)
replace_once(
    "web/src/lib/components/DriveCockpit.svelte",
    '      {#if running && !modelOwnsControl}\n        <ManualControlPad armRequest={manualArmRequest} />\n      {:else if running && modelOwnsControl}\n        <p class="model-control-note">Model control active · Emergency Brake remains available</p>\n      {/if}',
    '      {#if running && !policyOwnsControl}\n        <ManualControlPad armRequest={manualArmRequest} />\n      {:else if running && policyOwnsControl}\n        <p class="model-control-note">{controlOwner} owns actuation · Emergency Brake remains available</p>\n      {/if}',
)

# --- Regression tests. ---
replace_once(
    "tests/test_world_worker.py",
    '        self.assertEqual(catalog["worker_api_revision"], 5)',
    '        self.assertEqual(catalog["worker_api_revision"], 6)\n        self.assertTrue(catalog["capabilities"]["spawn_point_selection"])\n        self.assertTrue(catalog["capabilities"]["selected_route"])\n        self.assertEqual(catalog["spawn_point_map"], "Town10HD_Opt")\n        self.assertEqual(catalog["spawn_count"], len(self.world.map.spawn_points))\n        self.assertEqual(catalog["spawn_points"][0]["index"], 0)',
)
append_before(
    "tests/test_world_worker.py",
    '    def test_health_reports_busy_without_waiting_for_world_lock(self) -> None:\n',
    '''    def test_selected_start_and_destination_use_exact_official_spawn_points(self) -> None:\n        prepared = self.worker.prepare(\n            {\n                "start_spawn_index": 2,\n                "route_mode": "selected_destination",\n                "destination_spawn_index": 6,\n                "initial_control_mode": "autopilot",\n            }\n        )\n        scene = prepared["scene"]\n        self.assertEqual(scene["spawn_index"], 2)\n        self.assertEqual(scene["destination"]["spawn_index"], 6)\n        self.assertEqual(scene["route"]["mode"], "selected_destination")\n        self.assertTrue(scene["route"]["planned"])\n        scene_id, lease_token = self.lease(prepared)\n        started = self.worker.start(scene_id, {"lease_token": lease_token})\n        self.assertTrue(started["scene"]["route"]["enforced"])\n        self.assertEqual(self.traffic_manager.paths[-1][0], scene["ego_actor_id"])\n        self.assertEqual(\n            self.traffic_manager.paths[-1][1][-1],\n            self.world.map.spawn_points[6].location,\n        )\n\n    def test_selected_start_never_silently_falls_back_when_occupied(self) -> None:\n        original_try_spawn = self.world.try_spawn_actor\n\n        def occupied(blueprint: Any, transform: Any, *args: Any, **kwargs: Any) -> Any:\n            if transform is self.world.map.spawn_points[3] and str(blueprint.id).startswith("vehicle."):\n                return None\n            return original_try_spawn(blueprint, transform, *args, **kwargs)\n\n        self.world.try_spawn_actor = occupied  # type: ignore[method-assign]\n        with self.assertRaisesRegex(WorkerError, "no fallback was used") as raised:\n            self.worker.prepare({"start_spawn_index": 3})\n        self.assertEqual(raised.exception.code, "ego_spawn_unavailable")\n        self.assertEqual(self.world.actors, {})\n\n    def test_selected_route_contract_rejects_missing_or_same_destination(self) -> None:\n        with self.assertRaisesRegex(WorkerError, "requires destination_spawn_index"):\n            SceneConfig.from_mapping({"route_mode": "selected_destination"})\n        with self.assertRaisesRegex(WorkerError, "must differ"):\n            SceneConfig.from_mapping(\n                {\n                    "route_mode": "selected_destination",\n                    "start_spawn_index": 4,\n                    "destination_spawn_index": 4,\n                }\n            )\n\n''',
)

# Frontend queue test ensures the new fields are part of the effective Garage contract.
replace_once(
    "web/tests/garage-preview.test.mjs",
    "    route: { mode: 'random_destination' },",
    "    route: { mode: 'selected_destination', startSpawnIndex: 2, destinationSpawnIndex: 7 },",
)
replace_once(
    "web/tests/garage-preview.test.mjs",
    "    camera: { resolution: '1920x1080', fps: 60, fov: 100, spectatorFollow: false }",
    "    camera: { resolution: '1920x1080', fps: 60, fov: 100, spectatorFollow: false }",
)

# Source-level UI truthfulness contract; complements Svelte type-check and Node behavior tests.
quality_test = ROOT / "tests/test_garage_world_route_ui.py"
quality_test.write_text('''from __future__ import annotations\n\nfrom pathlib import Path\n\nROOT = Path(__file__).parents[1]\n\n\ndef test_scene_world_fields_expose_capability_driven_spawn_and_destination_controls() -> None:\n    source = (ROOT / "web/src/lib/components/SceneWorldFields.svelte").read_text(encoding="utf-8")\n    assert "Start point" in source\n    assert "Selected destination" in source\n    assert "spawn_point_selection" in source\n    assert "selected_route" in source\n    assert "no fallback if occupied" in source\n\n\ndef test_drive_cockpit_names_runtime_control_owner_and_hides_manual_pad_for_policies() -> None:\n    source = (ROOT / "web/src/lib/components/DriveCockpit.svelte").read_text(encoding="utf-8")\n    for label in (\n        "CARLA Traffic Manager",\n        "CARLA BehaviorAgent",\n        "Imitation policy",\n        "Voxel supervisor",\n        "Registered model",\n        "Browser manual",\n    ):\n        assert label in source\n    assert "policyOwnsControl" in source\n    assert "Control owner" in source\n''', encoding="utf-8")

print("issue 28 patch applied")
