from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, got {count} for {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Ruff B905: these are deliberately adjacent slices, so unequal-length handling is desired.
rendering = Path("carla_vision/voxel/rendering.py")
text = rendering.read_text(encoding="utf-8")
needle = "zip(route[:-1], route[1:])"
if text.count(needle) != 2:
    raise SystemExit(
        f"rendering.py: expected two adjacent-route zip calls, got {text.count(needle)}"
    )
rendering.write_text(
    text.replace(needle, "zip(route[:-1], route[1:], strict=False)"),
    encoding="utf-8",
)


# Keep the World Worker ownership/lease validation serialized, but release the mutation
# lock before any optional CARLA/TM teacher read. A visualization must never become a
# control-plane blocker.
worker_path = Path("carla_vision/native/world_worker.py")
text = worker_path.read_text(encoding="utf-8")
start = text.index(
    "    def waypoints(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:"
)
end = text.index("    @classmethod\n    def _bounded_waypoint_locations", start)
method = '''    def waypoints(self, scene_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Read bounded map geometry for a teacher overlay, never vehicle control.

        Lease and scene identity are snapshotted under the mutation lock. CARLA/TM
        reads run after releasing it so optional teacher rendering cannot delay
        control, heartbeat, weather, mode changes, or scene teardown.
        """

        lease_token = self._lease_token(
            raw, allowed={"lease_token", "camera_location", "source_frame"}
        )
        camera_location = raw.get("camera_location")
        if camera_location is not None:
            if not isinstance(camera_location, Mapping):
                raise WorkerError(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_field",
                    "camera_location must be an object",
                )
            _strict_keys(
                camera_location,
                allowed={"x", "y", "z"},
                required={"x", "y", "z"},
                name="camera location",
            )
            camera_location = {
                axis: _number(
                    camera_location[axis], f"camera_location.{axis}", -1e7, 1e7
                )
                for axis in ("x", "y", "z")
            }
        source_frame = raw.get("source_frame")
        if source_frame is not None:
            source_frame = _integer(source_frame, "source_frame", 0, 2**63 - 1)

        if not self._lock.acquire(blocking=False):
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "worker_busy",
                "World Worker is updating the scene",
            )
        try:
            scene = self._require_scene(scene_id, lease_token)
            if scene.status not in {"prepared", "running"}:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "scene_inactive", "scene is stopping"
                )
            if self._clock() >= scene.lease_deadline:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "lease_expired", "scene lease expired"
                )
            if self._episode_id(scene.world) != scene.episode_id:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "episode_changed", "CARLA episode changed"
                )
            try:
                owned = next(item for item in scene.owned_actors if item.kind == "ego")
                ego = scene.ego
                if (
                    ego is None
                    or not bool(getattr(ego, "is_alive", True))
                    or int(ego.id) != owned.actor_id
                ):
                    raise RuntimeError("owned ego is no longer available")
                self._validate_owned_actor_identity(ego, owned)
            except Exception as error:
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "scene_identity_changed",
                    "scene ego identity changed",
                ) from error

            world = scene.world
            route_locations = list(scene.route_locations)
            traffic_manager = scene.traffic_manager
            scene_status = str(scene.status)
            control_mode = str(scene.control_mode)
            episode_id = int(scene.episode_id)
            response_scene_id = str(scene.scene_id)
            waypoint_map = scene.waypoint_map
        finally:
            self._lock.release()

        try:
            origin = (
                ego.get_location()
                if camera_location is None
                else self._carla.Location(**camera_location)
            )
        except Exception as error:
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "scene_identity_changed",
                "scene ego is no longer available",
            ) from error

        sampled_frame: int | None = None
        sampled_timestamp: float | None = None
        try:
            snapshot = world.get_snapshot()
            sampled_frame = int(snapshot.frame)
            sampled_timestamp = float(snapshot.timestamp.elapsed_seconds)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass

        source = "planned_route"
        locations = route_locations
        note = "Fixed map route; not a road prediction or collision-free path."
        if len(locations) < 2:
            locations = []
            get_actions = getattr(traffic_manager, "get_all_actions", None)
            if (
                scene_status == "running"
                and control_mode == "autopilot"
                and callable(get_actions)
            ):
                try:
                    actions = get_actions(ego)
                    locations = [action[1].transform.location for action in actions[:512]]
                except Exception:
                    # Teacher loss is optional and must never fail RGB or actuation.
                    locations = []
            if len(locations) >= 2:
                source = "traffic_manager"
                note = "Upcoming TM actions sampled now, not frame-matched route intent."
            else:
                source = "lane_centerline"
                note = "Lane centerline only; stops at a branch and is not the TM route."
                try:
                    if waypoint_map is None:
                        waypoint_map = world.get_map()
                        # Cache opportunistically; never wait for the mutation lock.
                        if self._lock.acquire(blocking=False):
                            try:
                                current = self._scene
                                if current is scene and current.waypoint_map is None:
                                    current.waypoint_map = waypoint_map
                            finally:
                                self._lock.release()
                    waypoint = waypoint_map.get_waypoint(origin)
                    locations = []
                    for _ in range(64):
                        if waypoint is None:
                            break
                        locations.append(waypoint.transform.location)
                        candidates = waypoint.next(2.0)
                        if len(candidates) != 1:
                            break
                        waypoint = candidates[0]
                except Exception as error:
                    raise WorkerError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "waypoint_teacher_unavailable",
                        "CARLA lane geometry is unavailable for the teacher overlay",
                    ) from error

        points = self._bounded_waypoint_locations(locations, origin)

        # Refuse a stale sample if lifecycle ownership changed while the optional
        # read was in flight. This final check is non-blocking by design.
        if not self._lock.acquire(blocking=False):
            raise WorkerError(
                HTTPStatus.CONFLICT,
                "worker_busy",
                "World Worker changed while waypoint geometry was sampled",
            )
        try:
            current = self._require_scene(scene_id, lease_token)
            if (
                current is not scene
                or current.status not in {"prepared", "running"}
                or int(current.episode_id) != episode_id
            ):
                raise WorkerError(
                    HTTPStatus.CONFLICT,
                    "episode_changed",
                    "CARLA scene changed while waypoint geometry was sampled",
                )
            if self._clock() >= current.lease_deadline:
                raise WorkerError(
                    HTTPStatus.CONFLICT, "lease_expired", "scene lease expired"
                )
        finally:
            self._lock.release()

        return {
            "schema_version": SCHEMA_VERSION,
            "worker_api_revision": WORKER_API_REVISION,
            "scene_id": response_scene_id,
            "episode_id": episode_id,
            "source": source,
            "coordinate_frame": "carla_world_metres",
            "teacher_only": True,
            "model_input": False,
            "controls_vehicle": False,
            "route_frame_matched": False,
            "source_frame": source_frame,
            "sampled_frame": sampled_frame,
            "sampled_timestamp": sampled_timestamp,
            "points": points,
            "note": note,
        }

'''
worker_path.write_text(text[:start] + method + text[end:], encoding="utf-8")


# Wire the second Voxel rendering product through the Drive cache/HTTP path and attach
# the read-only teacher only after the Voxel predictor has completed its RGB inference.
drive_path = Path("carla_vision/operator/drive.py")
text = drive_path.read_text(encoding="utf-8")
replace_fields_old = '''        self._voxel_jpeg: bytes | None = None
        self._voxel_frame_sequence = -1
        self._voxel: VoxelViewWorker | None = None
        self._raw_frame_sequence = -1
        self._overlay_frame_sequence = -1
        self._frame_received_monotonic: dict[str, float | None] = {
            "raw": None,
            "overlay": None,
            "voxel": None,
        }
        self._frame_arrivals: dict[str, deque[float]] = {
            "raw": deque(maxlen=180),
            "overlay": deque(maxlen=180),
            "voxel": deque(maxlen=180),
        }
'''
replace_fields_new = '''        self._voxel_jpeg: bytes | None = None
        self._voxel_overlay_jpeg: bytes | None = None
        self._voxel_frame_sequence = -1
        self._voxel_overlay_frame_sequence = -1
        self._voxel: VoxelViewWorker | None = None
        self._raw_frame_sequence = -1
        self._overlay_frame_sequence = -1
        self._frame_received_monotonic: dict[str, float | None] = {
            "raw": None,
            "overlay": None,
            "voxel": None,
            "voxel_overlay": None,
        }
        self._frame_arrivals: dict[str, deque[float]] = {
            "raw": deque(maxlen=180),
            "overlay": deque(maxlen=180),
            "voxel": deque(maxlen=180),
            "voxel_overlay": deque(maxlen=180),
        }
'''
if text.count(replace_fields_old) != 1:
    raise SystemExit("drive.py: expected original Voxel cache field block")
text = text.replace(replace_fields_old, replace_fields_new, 1)

snapshot_old = '''                "voxel_frame_sequence": self._voxel_frame_sequence,
                "stream": stream,
'''
snapshot_new = '''                "voxel_frame_sequence": self._voxel_frame_sequence,
                "voxel_overlay_frame_sequence": self._voxel_overlay_frame_sequence,
                "stream": stream,
'''
if text.count(snapshot_old) != 1:
    raise SystemExit("drive.py: snapshot Voxel sequence block not found")
text = text.replace(snapshot_old, snapshot_new, 1)

frame_start = text.index("    def frame(self, view: str) -> tuple[int, bytes]:")
frame_end = text.index("    def _stream_metrics", frame_start)
frame_methods = '''    def frame(self, view: str) -> tuple[int, bytes]:
        with self._lock:
            if view == "raw":
                sequence, payload = self._raw_frame_sequence, self._raw_jpeg
            elif view == "overlay":
                sequence, payload = self._overlay_frame_sequence, self._overlay_jpeg
            elif view == "voxel":
                sequence, payload = self._voxel_frame_sequence, self._voxel_jpeg
            elif view == "voxel_overlay":
                sequence, payload = (
                    self._voxel_overlay_frame_sequence,
                    self._voxel_overlay_jpeg,
                )
            else:
                raise ValueError(
                    "drive frame view must be raw, overlay, voxel or voxel_overlay"
                )
            if payload is None:
                raise FileNotFoundError(f"{view} drive frame is not ready")
            return sequence, payload

    def wait_for_frame(
        self,
        view: str,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ) -> tuple[int, bytes]:
        if view not in {"raw", "overlay", "voxel", "voxel_overlay"}:
            raise ValueError(
                "drive frame view must be raw, overlay, voxel or voxel_overlay"
            )
        deadline = time.monotonic() + float(timeout)
        with self._frame_condition:
            while True:
                if view == "raw":
                    sequence, payload = self._raw_frame_sequence, self._raw_jpeg
                elif view == "overlay":
                    sequence, payload = self._overlay_frame_sequence, self._overlay_jpeg
                elif view == "voxel":
                    sequence, payload = self._voxel_frame_sequence, self._voxel_jpeg
                else:
                    sequence, payload = (
                        self._voxel_overlay_frame_sequence,
                        self._voxel_overlay_jpeg,
                    )
                if payload is not None and sequence > after_sequence:
                    return sequence, payload
                if self._status in _TERMINAL:
                    raise EOFError("drive camera stream ended")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"timed out waiting for a {view} drive frame")
                self._frame_condition.wait(remaining)

'''
text = text[:frame_start] + frame_methods + text[frame_end:]

cache_start = text.index(
    "    def _cache_frame(self, view: str, sequence: int, payload: bytes) -> None:"
)
cache_end = text.index("    def _record_mode_change", cache_start)
cache_and_provider = '''    def _cache_frame(self, view: str, sequence: int, payload: bytes) -> None:
        with self._frame_condition:
            if view == "raw":
                self._raw_frame_sequence = sequence
                self._raw_jpeg = payload
            elif view == "overlay":
                self._overlay_frame_sequence = sequence
                self._overlay_jpeg = payload
            elif view == "voxel":
                self._voxel_frame_sequence = sequence
                self._voxel_jpeg = payload
            elif view == "voxel_overlay":
                self._voxel_overlay_frame_sequence = sequence
                self._voxel_overlay_jpeg = payload
            else:
                raise ValueError(
                    "drive frame view must be raw, overlay, voxel or voxel_overlay"
                )
            received = time.monotonic()
            self._frame_received_monotonic[view] = received
            self._frame_arrivals[view].append(received)
            self._frame_condition.notify_all()

    def _voxel_waypoint_provider(self) -> Any | None:
        """Return a read-only teacher callback outside the actuation request lane."""

        worker = self._world_worker
        if worker is None:
            return None
        with self._lock:
            scene = self._worker_scene
        if scene is None or not bool(scene.capabilities.get("waypoint_teacher")):
            return None
        scene_id = scene.scene_id

        def provide(frame: Any) -> dict[str, Any]:
            transform = getattr(frame, "transform", None)
            if (
                not isinstance(transform, (tuple, list))
                or len(transform) != 6
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in transform
                )
            ):
                raise RuntimeError("camera frame has no valid CARLA transform")
            with self._lock:
                current = self._worker_scene
                stopped = self._worker_scene_stopped
            if current is None or stopped or current.scene_id != scene_id:
                raise RuntimeError("World Worker scene changed before waypoint sampling")
            return worker.waypoints(
                current,
                camera_location={
                    "x": float(transform[0]),
                    "y": float(transform[1]),
                    "z": float(transform[2]),
                },
                source_frame=int(frame.frame),
            )

        return provide

'''
text = text[:cache_start] + cache_and_provider + text[cache_end:]

voxel_create_old = "                self._voxel = VoxelViewWorker(device=self.config.device)\n"
voxel_create_new = '''                self._voxel = VoxelViewWorker(
                    device=self.config.device,
                    waypoint_provider=self._voxel_waypoint_provider(),
                )
'''
if text.count(voxel_create_old) != 1:
    raise SystemExit("drive.py: VoxelViewWorker construction not found")
text = text.replace(voxel_create_old, voxel_create_new, 1)

voxel_cache_old = '''                        self._cache_frame("voxel", voxel_result.sequence, voxel_result.jpeg)
                        _json_line(voxel_log, {"event": "prediction", **voxel_result.record()})
'''
voxel_cache_new = '''                        self._cache_frame("voxel", voxel_result.sequence, voxel_result.jpeg)
                        self._cache_frame(
                            "voxel_overlay",
                            voxel_result.sequence,
                            voxel_result.overlay_jpeg,
                        )
                        _json_line(voxel_log, {"event": "prediction", **voxel_result.record()})
'''
if text.count(voxel_cache_old) != 1:
    raise SystemExit("drive.py: Voxel result cache block not found")
text = text.replace(voxel_cache_old, voxel_cache_new, 1)
drive_path.write_text(text, encoding="utf-8")


replace_once(
    "carla_vision/operator/server.py",
    '''        if view not in {"raw", "overlay", "voxel"}:
            raise ValueError("drive frame view must be raw, overlay or voxel")
''',
    '''        if view not in {"raw", "overlay", "voxel", "voxel_overlay"}:
            raise ValueError(
                "drive frame view must be raw, overlay, voxel or voxel_overlay"
            )
''',
)


docs = Path("docs/operator_ui.md")
docs_text = docs.read_text(encoding="utf-8")
old_docs = "/api/drive/stream.mjpg?view=raw|overlay|voxel"
if old_docs not in docs_text:
    raise SystemExit("docs/operator_ui.md: Drive stream contract not found")
docs.write_text(
    docs_text.replace(
        old_docs,
        "/api/drive/stream.mjpg?view=raw|overlay|voxel|voxel_overlay",
        1,
    ),
    encoding="utf-8",
)


# Put the new worker contract in the lean Operator regression lane too, not only the
# expensive all-extras suite.
ci = Path(".github/workflows/ci.yml")
ci_text = ci.read_text(encoding="utf-8")
ci_anchor = "          tests/test_voxel_ui_contract.py\n"
if ci_text.count(ci_anchor) != 1:
    raise SystemExit("ci.yml: Voxel UI test anchor not found")
ci.write_text(
    ci_text.replace(
        ci_anchor,
        ci_anchor + "          tests/test_waypoint_teacher_worker.py\n",
        1,
    ),
    encoding="utf-8",
)


# Extend existing runtime tests so the second Voxel stream cannot silently disconnect again.
voxel_tests = Path("tests/test_voxel_live_view.py")
text = voxel_tests.read_text(encoding="utf-8")
cache_test_old = '''    session._cache_frame("voxel", 60, b"voxel")
    assert session.frame("raw") == (80, b"raw")
    assert session.frame("overlay") == (70, b"overlay")
    assert session.wait_for_frame("voxel", 59, timeout=0.1) == (60, b"voxel")
    assert session.snapshot()["voxel"]["actuated"] is False
'''
cache_test_new = '''    session._cache_frame("voxel", 60, b"voxel")
    session._cache_frame("voxel_overlay", 61, b"voxel-overlay")
    assert session.frame("raw") == (80, b"raw")
    assert session.frame("overlay") == (70, b"overlay")
    assert session.wait_for_frame("voxel", 59, timeout=0.1) == (60, b"voxel")
    assert session.wait_for_frame("voxel_overlay", 60, timeout=0.1) == (
        61,
        b"voxel-overlay",
    )
    assert session.snapshot()["voxel_overlay_frame_sequence"] == 61
    assert session.snapshot()["voxel"]["actuated"] is False
'''
if text.count(cache_test_old) != 1:
    raise SystemExit("test_voxel_live_view.py: cache test block not found")
text = text.replace(cache_test_old, cache_test_new, 1)
text += '''


def test_waypoint_teacher_runs_after_rgb_inference_and_stays_separate():
    events = []

    class Predictor:
        def load(self):
            pass

        def predict(self, *_args, **_kwargs):
            events.append("predict")
            return result()

    def teacher(source):
        assert events == ["predict"]
        events.append("teacher")
        return {
            "source": "planned_route",
            "coordinate_frame": "carla_world_metres",
            "teacher_only": True,
            "model_input": False,
            "controls_vehicle": False,
            "route_frame_matched": False,
            "source_frame": source.frame,
            "points": [
                {"x": 2.0, "y": 0.0, "z": 1.7},
                {"x": 8.0, "y": 0.0, "z": 1.7},
            ],
        }

    worker = VoxelViewWorker(
        device="cpu",
        predictor_factory=Predictor,
        max_fps=10,
        waypoint_provider=teacher,
    )
    source = frame(5)
    source.transform = (0.0, 0.0, 1.7, 0.0, 0.0, 0.0)
    try:
        worker.submit(source)
        wait_until(lambda: worker.latest() is not None)
        latest = worker.latest()
        assert events == ["predict", "teacher"]
        assert latest.waypoint_status == "available"
        assert latest.waypoint_teacher["teacher_only"] is True
        assert latest.waypoint_teacher["input_to_model"] is False
        assert latest.record()["input"] == "rgb_only"
        decoded = cv2.imdecode(np.frombuffer(latest.overlay_jpeg, np.uint8), 1)
        assert decoded.shape == (120, 160, 3)
    finally:
        worker.close()


def test_drive_waypoint_provider_uses_current_scene_without_control_request_lock(tmp_path):
    calls = []

    class Worker:
        def waypoints(self, scene, **kwargs):
            calls.append((scene, kwargs))
            return {"points": []}

    config = DriveStartConfig(
        run_id="voxel-provider", host="127.0.0.1", port=2000,
        vehicle_blueprint="vehicle.audi.tt", color=None, seed=7,
        weather_preset="keep", prop_preset="none", detector_enabled=False,
        detector="rtdetr", weights=None, device="cpu", image_size=640, confidence=0.2,
        width=1280, height=720, camera_fps=30, camera_fov=90,
        record_video=False, spectator_follow=False,
    )
    session = DriveSession(
        replace(config, voxel_enabled=True, world_worker_enabled=True),
        workspace=tmp_path,
        world_worker=Worker(),
    )
    scene = SimpleNamespace(
        scene_id="scene-123456789012",
        capabilities={"waypoint_teacher": True},
    )
    session._worker_scene = scene
    provider = session._voxel_waypoint_provider()
    assert provider is not None
    source = frame(9)
    source.transform = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    provider(source)
    assert calls == [
        (
            scene,
            {
                "camera_location": {"x": 1.0, "y": 2.0, "z": 3.0},
                "source_frame": 109,
            },
        )
    ]
'''
voxel_tests.write_text(text, encoding="utf-8")


worker_tests = Path("tests/test_waypoint_teacher_worker.py")
text = worker_tests.read_text(encoding="utf-8")
text += '''


def test_slow_teacher_sampling_does_not_hold_world_worker_lock(native: Any) -> None:
    native.scene.route_locations = []
    native.scene.status = "running"
    native.scene.control_mode = "autopilot"
    entered = threading.Event()
    release = threading.Event()
    actions = [
        ["Straight", FakeWaypoint(FakeLocation(x, 1, 0))]
        for x in (2, 4, 6)
    ]

    def slow_actions(_ego: Any) -> list[Any]:
        entered.set()
        assert release.wait(2)
        return actions

    result: dict[str, Any] = {}
    with mock.patch.object(
        native.tm,
        "get_all_actions",
        create=True,
        side_effect=slow_actions,
    ):
        thread = threading.Thread(
            target=lambda: result.update(
                read(
                    native,
                    camera_location={"x": 0, "y": 1, "z": 2},
                    source_frame=44,
                )
            )
        )
        thread.start()
        assert entered.wait(1)
        try:
            assert native.worker._lock.acquire(blocking=False)
            native.worker._lock.release()
        finally:
            release.set()
            thread.join(2)
    assert not thread.is_alive()
    assert result["source"] == "traffic_manager"
    assert result["source_frame"] == 44
'''
worker_tests.write_text(text, encoding="utf-8")


ui_tests = Path("tests/test_voxel_ui_contract.py")
text = ui_tests.read_text(encoding="utf-8")
text += '''


def test_voxel_overlay_is_wired_through_drive_and_http_stream() -> None:
    drive = (ROOT / "carla_vision/operator/drive.py").read_text(encoding="utf-8")
    server = (ROOT / "carla_vision/operator/server.py").read_text(encoding="utf-8")
    assert '"voxel_overlay"' in drive
    assert 'waypoint_provider=self._voxel_waypoint_provider()' in drive
    assert '"voxel_overlay",' in drive
    assert '{"raw", "overlay", "voxel", "voxel_overlay"}' in server
'''
ui_tests.write_text(text, encoding="utf-8")
