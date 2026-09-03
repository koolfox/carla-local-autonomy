from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# World Worker: use CARLA 0.9.16 Sensor.stop()/listen() on the existing sensor.
replace_once(
    "carla_vision/native/world_worker.py",
    'r"(?P<action>start|heartbeat|control|mode|weather|configure|camera|camera_orbit|waypoints|stop)$"',
    'r"(?P<action>start|heartbeat|control|mode|weather|configure|camera|camera_orbit|camera_pause|camera_resume|waypoints|stop)$"',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '            "waypoint_teacher": True,\n',
    '            "waypoint_teacher": True,\n            "camera_pause_resume": True,\n',
)
replace_once(
    "carla_vision/native/world_worker.py",
    "        self._closed = False\n        self._close_complete = False\n",
    "        self._closed = False\n        self._paused = False\n        self._close_complete = False\n",
)
replace_once(
    "carla_vision/native/world_worker.py",
    "    def _on_image(self, image: Any) -> None:\n        with self._condition:\n            if self._closed:\n                return\n",
    "    def _on_image(self, image: Any) -> None:\n        with self._condition:\n            if self._closed or self._paused:\n                return\n",
)
replace_once(
    "carla_vision/native/world_worker.py",
    "    def _on_image(self, image: Any) -> None:\n",
    '''    def pause(self) -> None:\n        """Stop CARLA sensor delivery without destroying the camera actor."""\n\n        with self._lifecycle_lock:\n            with self._condition:\n                if self._closed:\n                    raise RuntimeError("cannot pause a closed camera relay")\n                if self._paused:\n                    return\n                self._paused = True\n                self._pending_image = None\n                self._condition.notify_all()\n            try:\n                self.sensor.stop()\n            except BaseException:\n                with self._condition:\n                    self._paused = False\n                    self._condition.notify_all()\n                raise\n\n    def resume(self) -> None:\n        """Resume the same CARLA sensor subscription after :meth:`pause`."""\n\n        with self._lifecycle_lock:\n            with self._condition:\n                if self._closed:\n                    raise RuntimeError("cannot resume a closed camera relay")\n                if not self._paused:\n                    return\n            try:\n                self.sensor.listen(self._on_image)\n            except BaseException:\n                # Keep the relay visibly paused when CARLA could not subscribe.\n                raise\n            with self._condition:\n                self._paused = False\n                self._condition.notify_all()\n\n    def _on_image(self, image: Any) -> None:\n''',
)
replace_once(
    "carla_vision/native/world_worker.py",
    '                "actor_id": int(self.sensor.id),\n                "sequence": self._sequence,\n',
    '                "actor_id": int(self.sensor.id),\n                "paused": self._paused,\n                "sequence": self._sequence,\n',
)
replace_once(
    "carla_vision/native/world_worker.py",
    "                if not confirmed_dead():\n                    try:\n                        self.sensor.stop()\n",
    "                if not confirmed_dead() and not self._paused:\n                    try:\n                        self.sensor.stop()\n",
)
replace_once(
    "carla_vision/native/world_worker.py",
    "    def camera_frame(\n",
    '''    def _set_camera_subscription(\n        self,\n        scene_id: str,\n        raw: Mapping[str, Any],\n        *,\n        listening: bool,\n    ) -> dict[str, Any]:\n        lease_token = self._lease_token(raw, allowed={"lease_token"})\n        with self._lock:\n            scene = self._require_scene(scene_id, lease_token)\n            if scene.status not in {"prepared", "running"}:\n                raise WorkerError(\n                    HTTPStatus.CONFLICT, "scene_inactive", "scene is stopping"\n                )\n            relay = scene.camera_relay\n            if relay is None:\n                raise WorkerError(\n                    HTTPStatus.NOT_FOUND, "camera_inactive", "compressed camera is not active"\n                )\n\n        # Sensor subscription calls can touch CARLA's streaming client. Keep\n        # them outside the scene mutation lock so heartbeat/control remain live.\n        try:\n            if listening:\n                relay.resume()\n            else:\n                relay.pause()\n        except Exception as error:\n            raise WorkerError(\n                HTTPStatus.SERVICE_UNAVAILABLE,\n                "camera_subscription_failed",\n                f"CARLA camera subscription change failed: {type(error).__name__}: {error}",\n            ) from error\n\n        with self._lock:\n            current = self._require_scene(scene_id, lease_token)\n            if current is not scene or current.camera_relay is not relay:\n                raise WorkerError(\n                    HTTPStatus.CONFLICT,\n                    "camera_changed",\n                    "camera changed while its subscription was updated",\n                )\n            self._refresh_lease(current)\n            return self._scene_response(current, current.status)\n\n    def camera_pause(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:\n        """Pause the live RGB subscription for acceptance/diagnostics."""\n\n        return self._set_camera_subscription(scene_id, raw, listening=False)\n\n    def camera_resume(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:\n        """Resume the same live RGB sensor after a diagnostic pause."""\n\n        return self._set_camera_subscription(scene_id, raw, listening=True)\n\n    def camera_frame(\n''',
)

# Authenticated client methods; no Worker URL/token is exposed to the browser.
replace_once(
    "carla_vision/operator/world_worker_client.py",
    "    def waypoints(\n",
    '''    def pause_camera(self, scene: WorldWorkerScene) -> WorldWorkerScene:\n        if not scene.capabilities.get("camera_pause_resume"):\n            raise WorldWorkerError(\n                "the Windows World Worker does not support diagnostic camera pause/resume; "\n                "pull main and restart the Worker",\n                code="camera_pause_resume_unavailable",\n            )\n        return self._scene_request(\n            scene, "camera_pause", {"lease_token": scene.lease_token}\n        )\n\n    def resume_camera(self, scene: WorldWorkerScene) -> WorldWorkerScene:\n        if not scene.capabilities.get("camera_pause_resume"):\n            raise WorldWorkerError(\n                "the Windows World Worker does not support diagnostic camera pause/resume; "\n                "pull main and restart the Worker",\n                code="camera_pause_resume_unavailable",\n            )\n        return self._scene_request(\n            scene, "camera_resume", {"lease_token": scene.lease_token}\n        )\n\n    def waypoints(\n''',
)

# Acceptance runner corrections and real camera-fault orchestration.
replace_once(
    "carla_vision/operator/garage_acceptance.py",
    "from .world_worker_client import WorldWorkerClient\n\nSCHEMA_VERSION = \"1.0\"\nEXPECTED_CARLA_VERSION = \"0.9.16\"\n_TERMINAL = frozenset({\"success\", \"failed\"})\n",
    "from .world_worker_client import WorldWorkerClient, WorldWorkerScene\n\nSCHEMA_VERSION = \"1.0\"\nEXPECTED_CARLA_VERSION = \"0.9.16\"\n",
)
replace_once(
    "carla_vision/operator/garage_acceptance.py",
    "\ndef _wait_for(\n",
    '''\ndef _catalog_identifier(item: Any) -> str:\n    if isinstance(item, Mapping):\n        value = item.get("id", item.get("name", ""))\n        return str(value)\n    return str(item)\n\n\ndef _wait_for(\n''',
)
old_catalog = '''        maps = catalog.get("maps", [])\n        map_available = self.map_name == "current" or any(\n            str(item.get("id", item)) == self.map_name\n            or str(item.get("name", "")) == self.map_name\n            for item in maps\n            if isinstance(item, (str, Mapping))\n        )\n        _check(\n            checks,\n            "target_map_available",\n            map_available,\n            observed=self.map_name,\n            expected="map advertised by the live Worker catalog",\n        )\n        vehicles = catalog.get("vehicles", [])\n        vehicle_available = any(\n            str(item.get("id", item)) == self.vehicle\n            for item in vehicles\n            if isinstance(item, (str, Mapping))\n        )\n        _check(\n            checks,\n            "vehicle_available",\n            vehicle_available,\n            observed=self.vehicle,\n            expected="vehicle advertised by the live catalog",\n        )\n'''
new_catalog = '''        camera_fault_capability = _nested(health, "capabilities", "camera_pause_resume") is True\n        _check(\n            checks,\n            "camera_pause_resume_capability",\n            camera_fault_capability,\n            observed=_nested(health, "capabilities", "camera_pause_resume"),\n            expected=True,\n            note=(\n                "The gate pauses the existing CARLA sensor with Sensor.stop() and resumes it "\n                "with Sensor.listen(); no timestamp or fake-frame mutation is accepted."\n            ),\n        )\n        current = dict(self.worker.current_scene())\n        clean_start = current.get("status") == "idle" and current.get("scene") is None\n        report["preflight"]["worker_scene"] = {\n            "status": current.get("status"),\n            "scene_id": _nested(current, "scene", "scene_id"),\n        }\n        _check(\n            checks,\n            "worker_clean_start",\n            clean_start,\n            observed=report["preflight"]["worker_scene"],\n            expected={"status": "idle", "scene_id": None},\n            note="A previous run must not require manual actor cleanup.",\n        )\n        maps = catalog.get("maps", [])\n        map_available = self.map_name == "current" or any(\n            _catalog_identifier(item) == self.map_name for item in maps\n        )\n        _check(\n            checks,\n            "target_map_available",\n            map_available,\n            observed=self.map_name,\n            expected="map advertised by the live Worker catalog",\n        )\n        vehicles = catalog.get("vehicles", [])\n        vehicle_available = any(\n            _catalog_identifier(item) == self.vehicle for item in vehicles\n        )\n        _check(\n            checks,\n            "vehicle_available",\n            vehicle_available,\n            observed=self.vehicle,\n            expected="vehicle advertised by the live catalog",\n        )\n'''
replace_once("carla_vision/operator/garage_acceptance.py", old_catalog, new_catalog)
replace_once(
    "carla_vision/operator/garage_acceptance.py",
    "                isinstance(running.get(\"vehicle_id\"), int),\n",
    "                isinstance(running.get(\"vehicle_id\"), int) and int(running[\"vehicle_id\"]) > 0,\n",
)
replace_once(
    "carla_vision/operator/garage_acceptance.py",
    "                isinstance(running.get(\"camera_id\"), int),\n",
    "                isinstance(running.get(\"camera_id\"), int) and int(running[\"camera_id\"]) > 0,\n",
)
replace_once(
    "carla_vision/operator/garage_acceptance.py",
    "                self._exercise_manual(session_id, running, checks)\n",
    "                self._exercise_manual(session_id, running, checks)\n",
)
start = Path("carla_vision/operator/garage_acceptance.py")
text = start.read_text(encoding="utf-8")
begin = text.index("    def _exercise_manual(\n")
end = text.index("    def _exercise_behavior(", begin)
manual = '''    def _exercise_manual(\n        self,\n        session_id: str,\n        running: Mapping[str, Any],\n        checks: list[dict[str, Any]],\n    ) -> None:\n        baseline = abs(float(_nested(running, "telemetry", "speed") or 0.0))\n        sequence = 0\n        for _ in range(12):\n            sequence += 1\n            self.manager.control(\n                {\n                    "session_id": session_id,\n                    "sequence": sequence,\n                    "throttle": 0.35,\n                    "steer": 0.0,\n                    "brake": 0.0,\n                    "hand_brake": False,\n                    "reverse": False,\n                }\n            )\n            self.sleep(0.10)\n        moving = _wait_for(\n            self.manager,\n            lambda state: (\n                abs(float(_nested(state, "telemetry", "speed") or 0.0)) > baseline + 0.15\n                and int(state.get("controls_written") or 0) > 0\n            ),\n            timeout=5.0,\n            clock=self.clock,\n            sleep=self.sleep,\n        )\n        _check(\n            checks,\n            "manual_control_effect",\n            True,\n            observed={\n                "baseline_speed_mps": baseline,\n                "speed_mps": _nested(moving, "telemetry", "speed"),\n                "controls_written": moving.get("controls_written"),\n            },\n            expected="vehicle speed increases after repeated browser throttle commands",\n        )\n\n        stale = _wait_for(\n            self.manager,\n            lambda state: state.get("control_source") == "browser_deadman"\n            and state.get("deadman_active") is True,\n            timeout=3.0,\n            clock=self.clock,\n            sleep=self.sleep,\n        )\n        _check(\n            checks,\n            "control_stale_fail_safe",\n            True,\n            observed={\n                "control_source": stale.get("control_source"),\n                "deadman_active": stale.get("deadman_active"),\n            },\n            expected="browser_deadman applies service brake after input lease expires",\n        )\n\n        before_pause_sequence, _ = self.manager.frame("raw")\n        active_scene = WorldWorkerScene.from_response(self.worker.current_scene())\n        self.worker.pause_camera(active_scene)\n        paused = True\n        try:\n            camera_deadman = _wait_for(\n                self.manager,\n                lambda state: state.get("control_source") == "camera_deadman"\n                and state.get("deadman_active") is True,\n                timeout=4.0,\n                clock=self.clock,\n                sleep=self.sleep,\n            )\n            _check(\n                checks,\n                "camera_stale_fail_safe",\n                _nested(camera_deadman, "stream", "stale") is True,\n                observed={\n                    "stream_stale": _nested(camera_deadman, "stream", "stale"),\n                    "control_source": camera_deadman.get("control_source"),\n                    "deadman_active": camera_deadman.get("deadman_active"),\n                },\n                expected="real sensor pause produces camera_deadman before the session fails",\n            )\n        finally:\n            if paused:\n                self.worker.resume_camera(active_scene)\n\n        recovered_sequence, recovered_jpeg = self.manager.wait_for_frame(\n            "raw", before_pause_sequence, timeout=10.0\n        )\n        recovered = _wait_for(\n            self.manager,\n            lambda state: _nested(state, "stream", "stale") is False,\n            timeout=3.0,\n            clock=self.clock,\n            sleep=self.sleep,\n        )\n        _check(\n            checks,\n            "camera_recovered",\n            recovered_sequence > before_pause_sequence\n            and len(recovered_jpeg) > 100\n            and _nested(recovered, "stream", "stale") is False,\n            observed={\n                "before_sequence": before_pause_sequence,\n                "after_sequence": recovered_sequence,\n                "jpeg_bytes": len(recovered_jpeg),\n                "stream_stale": _nested(recovered, "stream", "stale"),\n            },\n            expected="same CARLA sensor resumes and publishes a newer frame",\n        )\n\n        self.manager.emergency_stop({"session_id": session_id})\n        emergency = _wait_for(\n            self.manager,\n            lambda state: state.get("control_source") == "emergency_stop",\n            timeout=2.0,\n            clock=self.clock,\n            sleep=self.sleep,\n        )\n        _check(\n            checks,\n            "emergency_brake",\n            emergency.get("emergency_stop") is True\n            and emergency.get("deadman_active") is True,\n            observed={\n                "emergency_stop": emergency.get("emergency_stop"),\n                "deadman_active": emergency.get("deadman_active"),\n                "control_source": emergency.get("control_source"),\n            },\n            expected="latched emergency_stop with service braking",\n        )\n\n'''
start.write_text(text[:begin] + manual + text[end:], encoding="utf-8")

# Public command and lean CI regression coverage.
replace_once(
    "pyproject.toml",
    'carla-world-worker = "carla_vision.native.observable_world_worker:main"\n',
    'carla-world-worker = "carla_vision.native.observable_world_worker:main"\ncarla-garage-acceptance = "carla_vision.operator.garage_acceptance:main"\n',
)
replace_once(
    ".github/workflows/ci.yml",
    "          tests/test_garage_async.py\n",
    "          tests/test_garage_async.py\n          tests/test_garage_acceptance.py\n          tests/test_camera_pause_acceptance.py\n",
)
